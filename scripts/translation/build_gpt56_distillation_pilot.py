#!/usr/bin/env python3
"""Build a licensed, decontaminated GPT-5.6 translation-distillation pilot.

The output keeps human references and student diagnostics local.  The separate
``prepare_synthetic_batch.py`` program serializes only source text for the
teacher API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


sys.path.insert(0, str(Path(__file__).resolve().parent))
from filter_training_dataset_against_protected import ProtectedIndex  # noqa: E402


LANGUAGES = {
    "en-ja": ("en-US", "ja-JP"),
    "ja-en": ("ja-JP", "en-US"),
}
OPEN_LICENSES = {
    "CC-BY-2.0-FR",
    "CC-BY-4.0",
    "CC-BY-SA-3.0",
    "PDL-1.0-compatible-CC-BY-4.0",
    "project-owned",
}
DEFAULT_QUOTAS = {
    "en-ja": {
        "kftt": 1_900,
        "alt": 1_700,
        "jlt": 1_500,
        "tatoeba": 2_001,
        "ui": 61,
        "alt-document": 539,
        "btec": 299,
    },
    "ja-en": {
        "kftt": 2_000,
        "alt": 1_800,
        "jlt": 1_600,
        "tatoeba": 2_000,
        "ui": 61,
        "alt-document": 539,
    },
}
ORIGIN_BUCKETS = {
    "human-kftt-replay": "kftt",
    "human-alt-parallel": "alt",
    "finalized-japanese-law-translation": "jlt",
    "human-tatoeba-bidirectional-agreement-filtered": "tatoeba",
    "mimi-shipped-ui-pair": "ui",
    "human-alt-document-window": "alt-document",
}


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing JSON input: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected a JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"missing JSONL input: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def deterministic_rank(seed: str, direction: str, bucket: str, source: str) -> bytes:
    return hashlib.sha256(
        f"{seed}\0{direction}\0{bucket}\0{normalized(source)}".encode("utf-8")
    ).digest()


def authenticated_train(dataset: Path) -> tuple[list[dict], dict]:
    manifest_path = dataset / "manifest.json"
    train_path = dataset / "train.jsonl"
    manifest = load_json(manifest_path)
    expected = manifest.get("outputs", {}).get("train", {}).get("sha256")
    actual = sha256(train_path)
    if not expected or expected != actual:
        raise SystemExit(f"train hash does not match authenticated manifest: {dataset}")
    return load_jsonl(train_path), {
        "manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        "train": {"path": str(train_path), "sha256": actual},
    }


def direction_for(row: dict) -> str:
    languages = (row.get("source_language"), row.get("target_language"))
    for direction, expected in LANGUAGES.items():
        if languages == expected:
            return direction
    raise ValueError(f"unsupported language direction: {languages!r}")


def bucket_for(row: dict) -> str:
    identifier = str(row.get("id", ""))
    if identifier.startswith("teacher-btec:"):
        return "btec"
    origin = str(row.get("origin", ""))
    if origin in ORIGIN_BUCKETS:
        return ORIGIN_BUCKETS[origin]
    domain = str(row.get("domain", ""))
    if domain == "professional-wikipedia-hard":
        return "kftt"
    if domain == "mimi-product-ui":
        return "ui"
    raise ValueError("unsupported corpus origin/domain")


def canonical_candidate(row: dict, *, priority: bool) -> dict:
    direction = direction_for(row)
    bucket = bucket_for(row)
    source = str(row.get("source", "")).strip()
    target = str(row.get("target") or row.get("reference_translation") or "").strip()
    license_name = str(row.get("source_license") or row.get("license") or "").strip()
    provenance = str(
        row.get("source_provenance") or row.get("provenance") or ""
    ).strip()
    if not source:
        raise ValueError("empty source")
    if license_name not in OPEN_LICENSES:
        raise ValueError(f"license is not on the distributable allowlist: {license_name!r}")
    if not provenance:
        raise ValueError("missing provenance")
    source_id = str(row.get("source_id") or row.get("id") or "").strip()
    if not source_id:
        raise ValueError("missing source ID")
    candidate = {
        "direction": direction,
        "bucket": bucket,
        "source": source,
        "target": target,
        "source_id": source_id,
        "license": license_name,
        "provenance": provenance,
        "priority": priority,
        "original_id": str(row.get("id") or source_id),
        "domain": str(row.get("domain") or "unknown"),
        "selection": str(row.get("selection") or ""),
    }
    for field in (
        "student_hypothesis",
        "student_chrf_pp",
        "student_sequence_nll",
        "selection_uncertainty_stratum",
        "selection_diversity_distance",
        "selection_rank",
    ):
        if field in row:
            candidate[field] = row[field]
    return candidate


def add_candidates(
    destination: dict[tuple[str, str], list[dict]],
    rows: Iterable[dict],
    *,
    priority: bool,
    rejected: Counter,
) -> None:
    for row in rows:
        try:
            candidate = canonical_candidate(row, priority=priority)
        except ValueError as error:
            rejected[f"invalid:{error}"] += 1
            continue
        destination[(candidate["direction"], candidate["bucket"])].append(candidate)


def protected_match(
    protected: ProtectedIndex,
    candidate: dict,
    maximum_jaccard: float,
    ngram_size: int,
) -> dict | None:
    for field in ("source", "target"):
        value = candidate[field]
        if not value:
            continue
        match = protected.match(value, maximum_jaccard, ngram_size)
        if match is not None:
            return {"candidateField": field, **match}
    return None


def make_seed(candidate: dict, seed: str) -> dict:
    direction = candidate["direction"]
    source_language, target_language = LANGUAGES[direction]
    digest = hashlib.sha256(
        f"{seed}\0{direction}\0{normalized(candidate['source'])}".encode("utf-8")
    ).hexdigest()[:20]
    row = {
        "id": f"gpt56-pilot:{direction}:{candidate['bucket']}:{digest}",
        "split": "train",
        "source_language": source_language,
        "target_language": target_language,
        "domain": candidate["domain"],
        "source": candidate["source"],
        "license": candidate["license"],
        "provenance": candidate["provenance"],
        "source_id": candidate["source_id"],
        "source_corpus": candidate["bucket"],
        "source_selection": (
            "priority 4-bit-student hard-example seed"
            if candidate["priority"]
            else "deterministic licensed corpus-balanced pilot selection"
        ),
        "original_id": candidate["original_id"],
    }
    if candidate["target"]:
        row.update(
            {
                "reference_translation": candidate["target"],
                "reference_provenance": candidate["provenance"],
            }
        )
    for field in (
        "student_hypothesis",
        "student_chrf_pp",
        "student_sequence_nll",
        "selection_uncertainty_stratum",
        "selection_diversity_distance",
        "selection_rank",
    ):
        if field in candidate:
            row[field] = candidate[field]
    return row


def select_rows(
    candidates: dict[tuple[str, str], list[dict]],
    protected: ProtectedIndex,
    quotas: dict[str, dict[str, int]],
    seed: str,
    maximum_jaccard: float,
    ngram_size: int,
) -> tuple[list[dict], dict]:
    output: list[dict] = []
    seen_sources: dict[str, set[str]] = defaultdict(set)
    exclusion_counts: Counter = Counter()
    collapsed_duplicates: Counter = Counter()
    exclusion_examples: list[dict] = []
    selected_counts: Counter = Counter()
    priority_counts: Counter = Counter()
    for direction, direction_quotas in quotas.items():
        for bucket, quota in direction_quotas.items():
            raw_pool = candidates.get((direction, bucket), [])
            unique_pool: dict[str, dict] = {}
            for candidate in raw_pool:
                key = normalized(candidate["source"])
                existing = unique_pool.get(key)
                if existing is None or (
                    candidate["priority"] and not existing["priority"]
                ):
                    unique_pool[key] = candidate
                if existing is not None:
                    collapsed_duplicates[f"{direction}:{bucket}"] += 1
            pool = list(unique_pool.values())
            pool.sort(
                key=lambda row: (
                    not row["priority"],
                    deterministic_rank(seed, direction, bucket, row["source"]),
                )
            )
            selected = 0
            for candidate in pool:
                if selected >= quota:
                    break
                source_key = normalized(candidate["source"])
                if source_key in seen_sources[direction]:
                    exclusion_counts[f"{direction}:{bucket}:duplicate-source"] += 1
                    continue
                match = protected_match(
                    protected, candidate, maximum_jaccard, ngram_size
                )
                if match is not None:
                    exclusion_counts[
                        f"{direction}:{bucket}:protected-{match['kind']}"
                    ] += 1
                    if len(exclusion_examples) < 50:
                        exclusion_examples.append(
                            {
                                "direction": direction,
                                "bucket": bucket,
                                "originalID": candidate["original_id"],
                                "candidateField": match["candidateField"],
                                "matchKind": match["kind"],
                                "jaccard": match["jaccard"],
                                "protectedID": match["protectedID"],
                                "protectedField": match["protectedField"],
                            }
                        )
                    continue
                seen_sources[direction].add(source_key)
                output.append(make_seed(candidate, seed))
                selected += 1
                selected_counts[f"{direction}:{bucket}"] += 1
                if candidate["priority"]:
                    priority_counts[f"{direction}:{bucket}"] += 1
            if selected != quota:
                raise SystemExit(
                    f"{direction}/{bucket}: selected {selected} of {quota}; "
                    f"only {len(pool)} candidates were available"
                )
    output.sort(key=lambda row: row["id"])
    return output, {
        "selected": dict(sorted(selected_counts.items())),
        "selected_priority": dict(sorted(priority_counts.items())),
        "collapsed_pool_duplicates": dict(sorted(collapsed_duplicates.items())),
        "rejected": dict(sorted(exclusion_counts.items())),
        "protected_match_examples": exclusion_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("full_depth_en_ja", type=Path)
    parser.add_argument("full_depth_ja_en", type=Path)
    parser.add_argument("document_en_ja", type=Path)
    parser.add_argument("document_ja_en", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--priority-seeds", type=Path, action="append", default=[])
    parser.add_argument("--protected-suite", type=Path, action="append", required=True)
    parser.add_argument("--seed", default="mimi-gpt56-final-translation-pilot-v1")
    parser.add_argument("--character-ngram-size", type=int, default=5)
    parser.add_argument("--maximum-jaccard", type=float, default=0.8)
    args = parser.parse_args()
    manifest_path = args.output.with_suffix(".manifest.json")
    for path in (args.output, manifest_path):
        if path.exists() and path.stat().st_size:
            raise SystemExit(f"refusing to overwrite non-empty output: {path}")
    if args.character_ngram_size < 1:
        raise SystemExit("character-ngram-size must be positive")
    if not 0 <= args.maximum_jaccard < 1:
        raise SystemExit("maximum-jaccard must be at least zero and below one")

    candidate_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    rejected: Counter = Counter()
    inputs: dict[str, dict] = {}
    for name, dataset in (
        ("full_depth_en_ja", args.full_depth_en_ja),
        ("full_depth_ja_en", args.full_depth_ja_en),
        ("document_en_ja", args.document_en_ja),
        ("document_ja_en", args.document_ja_en),
    ):
        rows, contract = authenticated_train(dataset)
        if name.startswith("document_"):
            rows = [
                row
                for row in rows
                if row.get("origin") == "human-alt-document-window"
            ]
            contract["row_filter"] = {
                "field": "origin",
                "value": "human-alt-document-window",
                "rows": len(rows),
            }
        inputs[name] = contract
        add_candidates(candidate_groups, rows, priority=False, rejected=rejected)

    for index, path in enumerate(args.priority_seeds):
        rows = load_jsonl(path)
        inputs[f"priority_seeds_{index + 1}"] = {
            "path": str(path),
            "sha256": sha256(path),
            "rows": len(rows),
        }
        add_candidates(candidate_groups, rows, priority=True, rejected=rejected)

    protected = ProtectedIndex(args.protected_suite, args.character_ngram_size)
    selected, selection = select_rows(
        candidate_groups,
        protected,
        DEFAULT_QUOTAS,
        args.seed,
        args.maximum_jaccard,
        args.character_ngram_size,
    )
    if len(selected) != 16_000:
        raise SystemExit(f"internal quota error: selected {len(selected)} rows")
    directions = Counter(
        "en-ja" if row["source_language"] == "en-US" else "ja-en"
        for row in selected
    )
    if directions != {"en-ja": 8_000, "ja-en": 8_000}:
        raise SystemExit(f"internal direction-balance error: {dict(directions)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in selected
        ),
        encoding="utf-8",
    )
    licenses = Counter(row["license"] for row in selected)
    corpora = Counter(
        f"{'en-ja' if row['source_language'] == 'en-US' else 'ja-en'}:"
        f"{row['source_corpus']}"
        for row in selected
    )
    manifest = {
        "schema_version": 1,
        "experiment": "GPT-5.6 final-translation sequence-distillation pilot",
        "promotion_eligible": False,
        "teacher_submission_status": "not-submitted",
        "private_reasoning_traces_used": False,
        "teacher_input_policy": (
            "source-only; local references and student hypotheses must be removed "
            "by prepare_synthetic_batch.py"
        ),
        "seed": args.seed,
        "selection_policy": (
            "priority hard examples first, then deterministic SHA-256 rank, "
            "within fixed direction/corpus quotas"
        ),
        "quotas": DEFAULT_QUOTAS,
        "counts": {
            "rows": len(selected),
            "directions": dict(sorted(directions.items())),
            "corpora": dict(sorted(corpora.items())),
            "licenses": dict(sorted(licenses.items())),
            "rows_with_human_reference": sum(
                "reference_translation" in row for row in selected
            ),
            "rows_without_human_reference": sum(
                "reference_translation" not in row for row in selected
            ),
        },
        "license_policy": {
            "allowlist": sorted(OPEN_LICENSES),
            "dataset_release_policy": (
                "retain per-row license and attribution; do not flatten the "
                "multi-license collection to a more permissive license"
            ),
        },
        "decontamination": {
            "normalization": "Unicode NFKC, casefold, collapse whitespace",
            "character_ngram_size": args.character_ngram_size,
            "maximum_jaccard_exclusive": args.maximum_jaccard,
            "fields_checked": ["source", "local human reference when present"],
            "protected_suites": [
                {"path": str(path), "sha256": sha256(path)}
                for path in args.protected_suite
            ],
        },
        "inputs": inputs,
        "candidate_rejections": dict(sorted(rejected.items())),
        "selection": selection,
        "output": {
            "path": str(args.output),
            "sha256": sha256(args.output),
        },
        "next_gate": (
            "prepare and validate a source-only reasoning-none Batch request; "
            "submission requires a secure rotated credential and explicit spend approval"
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
