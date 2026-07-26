#!/usr/bin/env python3
"""Build the protected-screened, error-stratified JA-to-EN v10 dataset."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from audit_translation_structures import critical_tokens, tokens
from filter_training_dataset_against_protected import (
    ProtectedIndex,
    normalized,
    sha256,
)
from typed_critical_token_policy import typed_preserves


EXPERIMENT = "canonical-sequence-v10-ja-en-error-stratified"
PREFERENCE_EXPERIMENT = "canonical-pairwise-v7-ja-en-claude5"
LEGAL_ORIGIN = "finalized-japanese-law-translation"
TEACHER_ORIGIN = "gpt56-claude5-approved-canonical-sequence"
ANCHOR_ORIGIN = "licensed-human-reference-anchor"
GENERAL_ORIGINS = (
    "human-alt-parallel",
    "human-kftt-replay",
    "human-tatoeba-bidirectional-agreement-filtered",
    "mimi-shipped-ui-pair",
)
LEGAL_CATEGORIES = (
    "negation",
    "critical",
    "repetition-risk",
    "terminology-risk",
    "omission-risk",
    "long",
    "general",
)
TEACHER_REPEAT = 8
TRAIN_LEGAL_QUOTAS = {
    "negation": 900,
    "critical": 900,
    "repetition-risk": 100,
    "terminology-risk": 700,
    "omission-risk": 450,
    "long": 1_000,
    "general": 950,
}
VALID_LEGAL_QUOTAS = {
    "negation": 96,
    "critical": 96,
    "repetition-risk": 6,
    "terminology-risk": 96,
    "omission-risk": 20,
    "long": 163,
    "general": 163,
}
TRAIN_GENERAL_QUOTAS = {
    "human-alt-parallel": 650,
    "human-kftt-replay": 650,
    "human-tatoeba-bidirectional-agreement-filtered": 621,
    "mimi-shipped-ui-pair": 47,
}
VALID_GENERAL_QUOTAS = {
    "human-alt-parallel": 112,
    "human-kftt-replay": 112,
    "human-tatoeba-bidirectional-agreement-filtered": 117,
    "mimi-shipped-ui-pair": 10,
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def authenticate_output(
    manifest_path: Path,
    split: str,
    path: Path,
    *,
    expected_direction: str | None = None,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    if (
        expected_direction is not None
        and manifest.get("direction") not in (None, expected_direction)
    ):
        raise SystemExit(f"manifest direction differs: {manifest_path}")
    output = manifest.get("outputs", {}).get(split, {})
    if output.get("sha256") != sha256(path):
        raise SystemExit(f"manifest does not authenticate {split}: {manifest_path}")
    return manifest


def stable_rank(seed: int, bucket: str, row: dict[str, Any]) -> int:
    key = (
        f"{seed}\0{bucket}\0{row.get('id')}\0{row.get('source')}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest(), "big")


def push_candidate(
    heap: list[tuple[int, str, dict[str, Any]]],
    row: dict[str, Any],
    *,
    seed: int,
    bucket: str,
    maximum: int,
) -> None:
    rank = stable_rank(seed, bucket, row)
    item = (-rank, str(row.get("id", "")), row)
    if len(heap) < maximum:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def repeated_structure(value: str) -> bool:
    clauses = [
        re.sub(r"\s+", "", clause)
        for clause in re.split(r"[、。；;]", value)
        if len(re.sub(r"\s+", "", clause)) >= 8
    ]
    return len(clauses) != len(set(clauses)) or bool(
        re.search(r"(.{5,18})\1", value)
    )


def legal_category(row: dict[str, Any]) -> str | None:
    if (row.get("source_language"), row.get("target_language")) != (
        "ja-JP",
        "en-US",
    ):
        return None
    source = str(row["source"])
    target = str(row["target"])
    if tokens(source)["negative"] and tokens(target)["negative"]:
        return "negation"
    if critical_tokens(source) and typed_preserves(
        source, target, "ja-JP", "en-US"
    ):
        return "critical"
    if repeated_structure(source):
        return "repetition-risk"
    if (
        "（以下" in source
        or "「" in source
        or re.search(r"\b(?:means|referred to as|the term)\b", target, re.I)
    ):
        return "terminology-risk"
    if len(source) >= 80 and len(re.findall(r"[、。；;]", source)) >= 3:
        return "omission-risk"
    if len(source) >= 100 or len(target) >= 150:
        return "long"
    return "general"


def protected_matches(
    index: ProtectedIndex,
    row: dict[str, Any],
    *,
    maximum_jaccard: float,
    ngram_size: int,
) -> list[dict[str, Any]]:
    matches = []
    for field in ("source", "target"):
        match = index.match(str(row[field]), maximum_jaccard, ngram_size)
        if match is not None:
            matches.append({"datasetField": field, **match})
    return matches


def ensure_translation_row(row: dict[str, Any], label: str) -> None:
    if (row.get("source_language"), row.get("target_language")) != (
        "ja-JP",
        "en-US",
    ):
        raise SystemExit(f"{label} contains the wrong direction: {row.get('id')}")
    for field in ("id", "source", "target", "origin", "source_license"):
        if not str(row.get(field, "")).strip():
            raise SystemExit(f"{label} row lacks {field}: {row.get('id')}")


def teacher_row(
    preference: dict[str, Any],
    *,
    repeat_index: int | None,
) -> dict[str, Any]:
    suffix = "valid" if repeat_index is None else f"train-{repeat_index:02d}"
    return {
        "id": f"v10-teacher:{preference['id']}:{suffix}",
        "source_id": preference["source_id"],
        "source": preference["source"],
        "target": preference["chosen"],
        "source_language": "ja-JP",
        "target_language": "en-US",
        "domain": f"v10-teacher:{preference['domain']}",
        "origin": TEACHER_ORIGIN,
        "source_license": preference["source_license"],
        "target_license": preference["target_license"],
        "source_provenance": preference["source_provenance"],
        "target_provenance": preference["target_provenance"],
        "attribution": preference["source_provenance"],
        "teacher_model": preference["teacher_model"],
        "teacher_response_id": preference["teacher_response_id"],
        "judge_model_ids": preference["judge_model_ids"],
        "review_status": preference["review_status"],
        "promotion_eligible": True,
        "private_reasoning_traces_used": False,
        "sequence_role": "independently-approved-canonical-teacher-target",
        "teacher_repeat_index": repeat_index,
    }


def anchor_row(
    preference: dict[str, Any],
    seed: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": f"v10-anchor:{preference['id']}",
        "source_id": preference["source_id"],
        "source": preference["source"],
        "target": seed["reference_translation"],
        "source_language": "ja-JP",
        "target_language": "en-US",
        "domain": f"v10-anchor:{preference['domain']}",
        "origin": ANCHOR_ORIGIN,
        "source_license": seed["license"],
        "target_license": seed["license"],
        "source_provenance": seed["provenance"],
        "target_provenance": seed["reference_provenance"],
        "attribution": seed["reference_provenance"],
        "promotion_eligible": True,
        "private_reasoning_traces_used": False,
        "sequence_role": "same-source-licensed-human-anchor",
    }


def selected_human_row(
    row: dict[str, Any],
    *,
    split: str,
    stratum: str,
) -> dict[str, Any]:
    return {
        **row,
        "id": f"v10-human:{split}:{stratum}:{row['id']}",
        "domain": f"v10-{stratum}",
        "v10_split": split,
        "v10_stratum": stratum,
        "sequence_role": "licensed-human-error-stratified-reference",
    }


def select_bucket(
    heap: list[tuple[int, str, dict[str, Any]]],
    *,
    quota: int,
    split: str,
    stratum: str,
    seed: int,
    protected: ProtectedIndex,
    maximum_jaccard: float,
    ngram_size: int,
    blocked_sources: set[str],
    rejected: Counter[str],
) -> list[dict[str, Any]]:
    candidates = [
        item[2]
        for item in sorted(heap, key=lambda value: (-value[0], value[1]))
    ]
    output = []
    for row in candidates:
        source_key = normalized(str(row["source"]))
        if source_key in blocked_sources:
            rejected[f"{split}:{stratum}:source-overlap"] += 1
            continue
        matches = protected_matches(
            protected,
            row,
            maximum_jaccard=maximum_jaccard,
            ngram_size=ngram_size,
        )
        if matches:
            rejected[f"{split}:{stratum}:protected-overlap"] += 1
            continue
        blocked_sources.add(source_key)
        output.append(selected_human_row(row, split=split, stratum=stratum))
        if len(output) == quota:
            break
    if len(output) != quota:
        raise SystemExit(
            f"insufficient clean rows for {split}:{stratum}: "
            f"{len(output)} < {quota}"
        )
    output.sort(key=lambda row: stable_rank(seed, stratum, row))
    return output


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
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
    parser.add_argument("seed_corpus", type=Path)
    parser.add_argument("legal_directory", type=Path)
    parser.add_argument("general_directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--protected-suite", type=Path, action="append", required=True)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--character-ngram-size", type=int, default=5)
    parser.add_argument("--maximum-jaccard", type=float, default=0.8)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    if args.character_ngram_size < 1 or not 0 <= args.maximum_jaccard < 1:
        raise SystemExit("invalid protected-overlap settings")

    preference_manifest_path = args.preference_directory / "manifest.json"
    preference_manifest = load_json(preference_manifest_path)
    if (
        preference_manifest.get("experiment") != PREFERENCE_EXPERIMENT
        or preference_manifest.get("direction") != "ja-en"
        or preference_manifest.get("promotion_eligible") is not True
    ):
        raise SystemExit("Claude-5 preference dataset is invalid")
    preference_splits = {}
    for split in ("train", "valid"):
        path = args.preference_directory / f"{split}.jsonl"
        if (
            preference_manifest.get("outputs", {}).get(split, {}).get("sha256")
            != sha256(path)
        ):
            raise SystemExit(f"preference manifest does not authenticate {split}")
        preference_splits[split] = load_jsonl(path)
    if len(preference_splits["train"]) != 136 or len(
        preference_splits["valid"]
    ) != 33:
        raise SystemExit("unexpected frozen Claude preference split sizes")

    seeds = {
        str(row["id"]): row
        for row in load_jsonl(args.seed_corpus)
        if row.get("source_language") == "ja-JP"
    }
    preferences = [
        *preference_splits["train"],
        *preference_splits["valid"],
    ]
    missing_seeds = sorted(
        str(row["source_id"])
        for row in preferences
        if str(row["source_id"]) not in seeds
    )
    if missing_seeds:
        raise SystemExit(f"preference rows lack licensed anchors: {missing_seeds[:3]}")

    legal_manifest_path = args.legal_directory / "manifest.json"
    legal_paths = {
        split: args.legal_directory / f"{split}.jsonl"
        for split in ("train", "valid")
    }
    legal_manifest = load_json(legal_manifest_path)
    if any(
        legal_manifest.get("outputs", {}).get(split, {}).get("sha256")
        != sha256(path)
        for split, path in legal_paths.items()
    ):
        raise SystemExit("legal manifest does not authenticate train and valid")
    if legal_manifest.get("promotion_eligible") is not False:
        raise SystemExit("legal corpus must remain training-only")

    general_manifest_path = args.general_directory / "manifest.json"
    general_paths = {
        split: args.general_directory / f"{split}.jsonl"
        for split in ("train", "valid")
    }
    general_manifest = authenticate_output(
        general_manifest_path,
        "train",
        general_paths["train"],
        expected_direction="ja-en",
    )
    authenticate_output(
        general_manifest_path,
        "valid",
        general_paths["valid"],
        expected_direction="ja-en",
    )

    protected = ProtectedIndex(
        args.protected_suite, args.character_ngram_size
    )
    blocked_sources = {
        normalized(str(row["source"])) for row in preferences
    }
    teacher_train = [
        teacher_row(row, repeat_index=repeat_index)
        for row in preference_splits["train"]
        for repeat_index in range(TEACHER_REPEAT)
    ]
    anchor_train = [
        anchor_row(row, seeds[str(row["source_id"])])
        for row in preference_splits["train"]
    ]
    teacher_valid = [
        teacher_row(row, repeat_index=None)
        for row in preference_splits["valid"]
    ]
    for row in [*teacher_train, *anchor_train, *teacher_valid]:
        ensure_translation_row(row, "teacher/anchor")
        matches = protected_matches(
            protected,
            row,
            maximum_jaccard=args.maximum_jaccard,
            ngram_size=args.character_ngram_size,
        )
        if matches:
            raise SystemExit(
                f"teacher/anchor row overlaps protected data: {row['id']}"
            )

    pool_multiplier = 30
    legal_heaps: dict[
        tuple[str, str], list[tuple[int, str, dict[str, Any]]]
    ] = defaultdict(list)
    legal_input_counts: Counter[str] = Counter()
    for split, path in legal_paths.items():
        quotas = (
            TRAIN_LEGAL_QUOTAS if split == "train" else VALID_LEGAL_QUOTAS
        )
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                category = legal_category(row)
                if category is None:
                    continue
                legal_input_counts[f"{split}:{category}"] += 1
                push_candidate(
                    legal_heaps[(split, category)],
                    row,
                    seed=args.seed,
                    bucket=f"{split}:legal:{category}",
                    maximum=quotas[category] * pool_multiplier,
                )

    general_heaps: dict[
        tuple[str, str], list[tuple[int, str, dict[str, Any]]]
    ] = defaultdict(list)
    general_input_counts: Counter[str] = Counter()
    for split, path in general_paths.items():
        quotas = (
            TRAIN_GENERAL_QUOTAS
            if split == "train"
            else VALID_GENERAL_QUOTAS
        )
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                origin = str(row.get("origin", ""))
                if origin not in quotas:
                    continue
                ensure_translation_row(row, f"general {split}")
                general_input_counts[f"{split}:{origin}"] += 1
                push_candidate(
                    general_heaps[(split, origin)],
                    row,
                    seed=args.seed,
                    bucket=f"{split}:general:{origin}",
                    maximum=quotas[origin] * pool_multiplier,
                )

    rejected: Counter[str] = Counter()
    human_splits: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "valid": [],
    }
    for split, quotas in (
        ("train", TRAIN_LEGAL_QUOTAS),
        ("valid", VALID_LEGAL_QUOTAS),
    ):
        for category, quota in quotas.items():
            human_splits[split].extend(
                select_bucket(
                    legal_heaps[(split, category)],
                    quota=quota,
                    split=split,
                    stratum=f"legal:{category}",
                    seed=args.seed,
                    protected=protected,
                    maximum_jaccard=args.maximum_jaccard,
                    ngram_size=args.character_ngram_size,
                    blocked_sources=blocked_sources,
                    rejected=rejected,
                )
            )
    for split, quotas in (
        ("train", TRAIN_GENERAL_QUOTAS),
        ("valid", VALID_GENERAL_QUOTAS),
    ):
        for origin, quota in quotas.items():
            human_splits[split].extend(
                select_bucket(
                    general_heaps[(split, origin)],
                    quota=quota,
                    split=split,
                    stratum=f"general:{origin}",
                    seed=args.seed,
                    protected=protected,
                    maximum_jaccard=args.maximum_jaccard,
                    ngram_size=args.character_ngram_size,
                    blocked_sources=blocked_sources,
                    rejected=rejected,
                )
            )

    output_splits = {
        "train": [*teacher_train, *anchor_train, *human_splits["train"]],
        "valid": [*teacher_valid, *human_splits["valid"]],
    }
    expected_counts = {"train": 8_192, "valid": 1_024}
    for split, rows in output_splits.items():
        if len(rows) != expected_counts[split]:
            raise SystemExit(
                f"{split} row count differs: {len(rows)} != {expected_counts[split]}"
            )
        identifiers = [str(row["id"]) for row in rows]
        if len(identifiers) != len(set(identifiers)):
            raise SystemExit(f"{split} contains duplicate row IDs")
        rows.sort(key=lambda row: stable_rank(args.seed, split, row))
    train_sources = {
        normalized(str(row["source"])) for row in output_splits["train"]
    }
    valid_sources = {
        normalized(str(row["source"])) for row in output_splits["valid"]
    }
    if train_sources & valid_sources:
        raise SystemExit("v10 train and validation sources overlap")

    args.output.mkdir(parents=True, exist_ok=True)
    output_paths = {
        split: args.output / f"{split}.jsonl"
        for split in ("train", "valid")
    }
    for split, path in output_paths.items():
        write_jsonl(path, output_splits[split])

    attribution_counts: Counter[tuple[str, str, str, str]] = Counter()
    for row in [*output_splits["train"], *output_splits["valid"]]:
        attribution_counts[
            (
                str(row["origin"]),
                str(row["source_license"]),
                str(row.get("source_provenance", "")),
                str(row.get("attribution", "")),
            )
        ] += 1
    attribution_path = args.output / "attribution.jsonl"
    attribution_rows = [
        {
            "origin": origin,
            "license": license_name,
            "provenance": provenance,
            "attribution": attribution,
            "rows": count,
        }
        for (origin, license_name, provenance, attribution), count in sorted(
            attribution_counts.items()
        )
    ]
    write_jsonl(attribution_path, attribution_rows)

    effective_licenses = {
        split: dict(
            sorted(
                Counter(str(row["source_license"]) for row in rows).items()
            )
        )
        for split, rows in output_splits.items()
    }
    origins = {
        split: dict(
            sorted(Counter(str(row["origin"]) for row in rows).items())
        )
        for split, rows in output_splits.items()
    }
    strata = {
        split: dict(
            sorted(
                Counter(
                    str(row.get("v10_stratum", row.get("sequence_role")))
                    for row in rows
                ).items()
            )
        )
        for split, rows in output_splits.items()
    }
    synthetic_train = origins["train"][TEACHER_ORIGIN]
    manifest = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": "frozen-ready-for-preregistered-training",
        "direction": "ja-en",
        "purpose": (
            "generation-level canonical-sequence adaptation with explicit "
            "long-legal and safety-risk coverage after v9 protected rejection"
        ),
        "target_source": (
            "mixed independently-approved GPT-5.6 final sequences and "
            "licensed human references"
        ),
        "promotion_eligible": False,
        "training_only": True,
        "private_reasoning_traces_used": False,
        "human_reviewer_required": False,
        "synthetic_policy": {
            "teacher_model": "gpt-5.6-sol-via-codex-cli",
            "admission_judges": ["claude-opus-5", "claude-sonnet-5"],
            "unanimous_pareto_preference_required": True,
            "absolute_quality_required": True,
            "unique_teacher_sources": len(preference_splits["train"]),
            "teacher_repeat_factor": TEACHER_REPEAT,
            "synthetic_train_rows": synthetic_train,
            "synthetic_train_fraction": synthetic_train
            / len(output_splits["train"]),
            "same_source_licensed_anchor_rows": len(anchor_train),
            "reasoning_traces_requested_or_retained": False,
        },
        "selection": {
            "seed": args.seed,
            "method": (
                "deterministic SHA-256 rank inside frozen split/origin/risk "
                "strata after preference-source exclusion and protected "
                "source/target screening"
            ),
            "train_legal_quotas": TRAIN_LEGAL_QUOTAS,
            "valid_legal_quotas": VALID_LEGAL_QUOTAS,
            "train_general_quotas": TRAIN_GENERAL_QUOTAS,
            "valid_general_quotas": VALID_GENERAL_QUOTAS,
            "legal_category_priority": list(LEGAL_CATEGORIES),
            "risk_definitions": {
                "negation": "bilingual explicit negation markers",
                "critical": (
                    "ASCII number, URL, placeholder, markup, or percent "
                    "surface with typed target preservation"
                ),
                "repetition-risk": (
                    "repeated Japanese clause or contiguous 5-18-character phrase"
                ),
                "terminology-risk": (
                    "Japanese quoted/defined term or English definition marker"
                ),
                "omission-risk": (
                    "at least 80 Japanese characters and three clause boundaries"
                ),
                "long": (
                    "at least 100 Japanese or 150 English characters after "
                    "higher-priority risk assignment"
                ),
                "general": "remaining ministry-published legal rows",
            },
        },
        "counts": {
            "train": len(output_splits["train"]),
            "valid": len(output_splits["valid"]),
            "unique_train_sources": len(train_sources),
            "unique_valid_sources": len(valid_sources),
            "teacher_train_rows": len(teacher_train),
            "teacher_valid_rows": len(teacher_valid),
            "licensed_anchor_train_rows": len(anchor_train),
            "human_error_stratified_train_rows": len(human_splits["train"]),
            "human_error_stratified_valid_rows": len(human_splits["valid"]),
            "legal_input_by_split_and_category": dict(
                sorted(legal_input_counts.items())
            ),
            "general_input_by_split_and_origin": dict(
                sorted(general_input_counts.items())
            ),
            "rejected": dict(sorted(rejected.items())),
        },
        "origins": origins,
        "strata": strata,
        "effective_licenses": effective_licenses,
        "distribution_provenance": {
            "all_rows_have_source_license": all(
                row.get("source_license")
                for rows in output_splits.values()
                for row in rows
            ),
            "all_rows_have_source_provenance": all(
                row.get("source_provenance")
                for rows in output_splits.values()
                for row in rows
            ),
            "attribution_sidecar": record(attribution_path),
            "licenses_are_open_or_project_owned": sorted(
                {
                    str(row["source_license"])
                    for rows in output_splits.values()
                    for row in rows
                }
            ),
        },
        "decontamination": {
            "normalization": "Unicode NFKC, casefold, remove whitespace",
            "character_ngram_size": args.character_ngram_size,
            "maximum_jaccard_exclusive": args.maximum_jaccard,
            "screened_fields": ["source", "target"],
            "protected_suites": [
                record(path) for path in args.protected_suite
            ],
            "preference_sources_excluded_from_human_pools": True,
            "train_valid_source_overlap": False,
        },
        "inputs": {
            "preference_manifest": record(preference_manifest_path),
            "preference_train": record(
                args.preference_directory / "train.jsonl"
            ),
            "preference_valid": record(
                args.preference_directory / "valid.jsonl"
            ),
            "licensed_seed_corpus": record(args.seed_corpus),
            "legal_manifest": record(legal_manifest_path),
            "legal_train": record(legal_paths["train"]),
            "legal_valid": record(legal_paths["valid"]),
            "general_manifest": record(general_manifest_path),
            "general_train": record(general_paths["train"]),
            "general_valid": record(general_paths["valid"]),
        },
        "outputs": {
            split: {**record(path), "rows": len(output_splits[split])}
            for split, path in output_paths.items()
        },
        "does_not_authorize_quantization": True,
        "does_not_authorize_protected_evaluation": True,
        "does_not_authorize_app_integration": True,
        "does_not_authorize_public_upload": True,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "manifest_sha256": sha256(manifest_path),
                "counts": manifest["counts"],
                "origins": origins,
                "strata": strata,
                "effective_licenses": effective_licenses,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
