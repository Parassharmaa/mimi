#!/usr/bin/env python3
"""Build the protected-screened licensed replay set for Claude-5 v8."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path

from audit_translation_structures import critical_tokens, tokens
from filter_training_dataset_against_protected import (
    ProtectedIndex,
    normalized,
    sha256,
)
from typed_critical_token_policy import typed_preserves


PREFERENCE_EXPERIMENT = "canonical-pairwise-v7-ja-en-claude5"
EXPERIMENT = "canonical-pairwise-v8-ja-en-licensed-replay"
LEGAL_ORIGIN = "finalized-japanese-law-translation"
OTHER_ORIGINS = (
    "human-alt-parallel",
    "human-kftt-replay",
    "human-tatoeba-bidirectional-agreement-filtered",
    "mimi-shipped-ui-pair",
)
LEGAL_CATEGORIES = ("negation", "critical", "long", "general")
TRAIN_QUOTA = {
    **{(LEGAL_ORIGIN, category): 17 for category in LEGAL_CATEGORIES},
    **{(origin, "general"): 17 for origin in OTHER_ORIGINS},
}
VALID_QUOTA = {
    **{(LEGAL_ORIGIN, category): 16 for category in LEGAL_CATEGORIES},
    **{(origin, "general"): 16 for origin in OTHER_ORIGINS},
}


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def record(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def stable_rank(seed: int, row: dict) -> int:
    key = f"{seed}\0{row.get('id')}\0{row.get('source')}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest(), "big")


def legal_category(row: dict) -> str:
    source = str(row["source"])
    target = str(row["target"])
    source_tokens = tokens(source)
    target_tokens = tokens(target)
    if source_tokens["negative"] and target_tokens["negative"]:
        return "negation"
    if critical_tokens(source) and typed_preserves(
        source, target, "ja-JP", "en-US"
    ):
        return "critical"
    if len(source) >= 80 or len(target) >= 120:
        return "long"
    return "general"


def bucket(row: dict) -> tuple[str, str] | None:
    origin = str(row.get("origin", ""))
    if origin == LEGAL_ORIGIN:
        return origin, legal_category(row)
    if origin in OTHER_ORIGINS:
        return origin, "general"
    return None


def push_candidate(
    heap: list[tuple[int, str, dict]],
    row: dict,
    *,
    seed: int,
    maximum: int,
) -> None:
    rank = stable_rank(seed, row)
    item = (-rank, str(row.get("id", "")), row)
    if len(heap) < maximum:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def protected_match(
    index: ProtectedIndex,
    row: dict,
    *,
    maximum_jaccard: float,
    ngram_size: int,
) -> list[dict]:
    matches = []
    for field in ("source", "target"):
        match = index.match(
            str(row[field]), maximum_jaccard, ngram_size
        )
        if match is not None:
            matches.append({"datasetField": field, **match})
    return matches


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preference_directory", type=Path)
    parser.add_argument("parent_dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--protected-suite", type=Path, action="append", required=True)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--character-ngram-size", type=int, default=5)
    parser.add_argument("--maximum-jaccard", type=float, default=0.8)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    if args.character_ngram_size < 1 or not 0 <= args.maximum_jaccard < 1:
        raise SystemExit("invalid protected-overlap settings")

    preference_manifest_path = args.preference_directory / "manifest.json"
    preference_train_path = args.preference_directory / "train.jsonl"
    preference_valid_path = args.preference_directory / "valid.jsonl"
    preference_manifest = load_json(preference_manifest_path)
    if (
        preference_manifest.get("experiment") != PREFERENCE_EXPERIMENT
        or preference_manifest.get("direction") != "ja-en"
        or preference_manifest.get("promotion_eligible") is not True
    ):
        raise SystemExit("v7 Claude preference dataset is invalid")
    preference_rows = [
        *load_jsonl(preference_train_path),
        *load_jsonl(preference_valid_path),
    ]
    preference_sources = {
        normalized(str(row["source"])) for row in preference_rows
    }

    parent_manifest_path = args.parent_dataset / "manifest.json"
    parent_train_path = args.parent_dataset / "train.jsonl"
    parent_manifest = load_json(parent_manifest_path)
    if (
        parent_manifest.get("direction") != "ja-en"
        or parent_manifest.get("outputs", {}).get("train", {}).get("sha256")
        != sha256(parent_train_path)
    ):
        raise SystemExit("parent replay dataset is not authenticated")

    protected = ProtectedIndex(args.protected_suite, args.character_ngram_size)
    heaps: dict[tuple[str, str], list[tuple[int, str, dict]]] = defaultdict(list)
    input_counts: Counter[tuple[str, str]] = Counter()
    excluded_preference_overlap = 0
    pool_multiplier = 40
    for line in parent_train_path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        current_bucket = bucket(row)
        if current_bucket is None:
            continue
        input_counts[current_bucket] += 1
        if normalized(str(row["source"])) in preference_sources:
            excluded_preference_overlap += 1
            continue
        maximum = (
            TRAIN_QUOTA[current_bucket] + VALID_QUOTA[current_bucket]
        ) * pool_multiplier
        push_candidate(
            heaps[current_bucket],
            row,
            seed=args.seed,
            maximum=maximum,
        )

    selected: dict[str, list[dict]] = {"train": [], "valid": []}
    rejected_protected = []
    seen_sources: set[str] = set(preference_sources)
    for current_bucket in sorted(TRAIN_QUOTA):
        candidates = [
            item[2]
            for item in sorted(
                heaps[current_bucket],
                key=lambda value: (-value[0], value[1]),
            )
        ]
        retained = []
        for row in candidates:
            source_key = normalized(str(row["source"]))
            if source_key in seen_sources:
                continue
            matches = protected_match(
                protected,
                row,
                maximum_jaccard=args.maximum_jaccard,
                ngram_size=args.character_ngram_size,
            )
            if matches:
                rejected_protected.append(
                    {"datasetID": row["id"], "bucket": current_bucket, "matches": matches}
                )
                continue
            seen_sources.add(source_key)
            retained.append(row)
            if len(retained) >= (
                TRAIN_QUOTA[current_bucket] + VALID_QUOTA[current_bucket]
            ):
                break
        train_count = TRAIN_QUOTA[current_bucket]
        valid_count = VALID_QUOTA[current_bucket]
        if len(retained) != train_count + valid_count:
            raise SystemExit(
                f"insufficient protected-clean rows for {current_bucket}: "
                f"{len(retained)} < {train_count + valid_count}"
            )
        origin, category = current_bucket
        for split, rows in (
            ("train", retained[:train_count]),
            ("valid", retained[train_count:]),
        ):
            selected[split].extend(
                {
                    **row,
                    "replay_category": category,
                    "replay_role": "licensed-parent-preservation",
                    "replay_split": split,
                }
                for row in rows
            )

    for split in selected:
        selected[split].sort(
            key=lambda row: (
                str(row["origin"]),
                str(row["replay_category"]),
                stable_rank(args.seed, row),
            )
        )
    train_sources = {normalized(str(row["source"])) for row in selected["train"]}
    valid_sources = {normalized(str(row["source"])) for row in selected["valid"]}
    if train_sources & valid_sources or (train_sources | valid_sources) & preference_sources:
        raise SystemExit("replay source isolation failed")

    args.output.mkdir(parents=True, exist_ok=True)
    output_paths = {
        split: args.output / f"replay-{split}.jsonl"
        for split in ("train", "valid")
    }
    for split, path in output_paths.items():
        write_jsonl(path, selected[split])

    manifest = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": "frozen-protected-screened-replay",
        "direction": "ja-en",
        "purpose": (
            "one-to-one licensed human replay for preference adaptation retention; "
            "not held-out promotion evidence"
        ),
        "target_source": "licensed-human-reference",
        "private_reasoning_traces_used": False,
        "promotion_eligible": False,
        "selection": {
            "seed": args.seed,
            "method": (
                "deterministic SHA-256 rank within origin/category after exact "
                "preference-source exclusion and protected source/target screening"
            ),
            "train_quotas": {
                f"{origin}:{category}": count
                for (origin, category), count in sorted(TRAIN_QUOTA.items())
            },
            "valid_quotas": {
                f"{origin}:{category}": count
                for (origin, category), count in sorted(VALID_QUOTA.items())
            },
            "legal_categories": {
                "negation": "source and licensed target both contain explicit negation",
                "critical": "source contains critical tokens and target passes typed preservation",
                "long": "source >=80 characters or target >=120 characters",
                "general": "remaining ministry-published legal rows",
            },
        },
        "counts": {
            "train": len(selected["train"]),
            "valid": len(selected["valid"]),
            "input_by_bucket": {
                f"{origin}:{category}": count
                for (origin, category), count in sorted(input_counts.items())
            },
            "excluded_exact_preference_source": excluded_preference_overlap,
            "rejected_protected_candidates": len(rejected_protected),
        },
        "origins": {
            split: dict(
                sorted(Counter(str(row["origin"]) for row in rows).items())
            )
            for split, rows in selected.items()
        },
        "categories": {
            split: dict(
                sorted(
                    Counter(
                        f"{row['origin']}:{row['replay_category']}" for row in rows
                    ).items()
                )
            )
            for split, rows in selected.items()
        },
        "effective_licenses": {
            split: dict(
                sorted(Counter(str(row["source_license"]) for row in rows).items())
            )
            for split, rows in selected.items()
        },
        "decontamination": {
            "normalization": "Unicode NFKC, casefold, remove whitespace",
            "character_ngram_size": args.character_ngram_size,
            "maximum_jaccard_exclusive": args.maximum_jaccard,
            "screened_fields": ["source", "target"],
            "protected_suites": [record(path) for path in args.protected_suite],
            "train_valid_source_overlap": False,
            "preference_replay_source_overlap": False,
        },
        "inputs": {
            "preference_manifest": record(preference_manifest_path),
            "preference_train": record(preference_train_path),
            "preference_valid": record(preference_valid_path),
            "parent_manifest": record(parent_manifest_path),
            "parent_train": record(parent_train_path),
        },
        "outputs": {
            split: {
                **record(path),
                "rows": len(selected[split]),
            }
            for split, path in output_paths.items()
        },
        "rejected_protected_candidates": rejected_protected,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "manifest_sha256": sha256(manifest_path),
                "counts": manifest["counts"],
                "origins": manifest["origins"],
                "categories": manifest["categories"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
