#!/usr/bin/env python3
"""Build the licensed corpus and fresh legal suite for v14 rollout repair."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_canonical_sequence_v10_dataset import legal_category
from filter_training_dataset_against_protected import (
    ProtectedIndex,
    normalized,
    sha256,
)

EXPERIMENT = "canonical-rollout-repair-v14-ja-en"
V10_EXPERIMENT = "canonical-sequence-v10-ja-en-error-stratified"
V12_EXPERIMENT = "canonical-safety-repair-v12-ja-en"
V10_MANIFEST_SHA256 = "d0eebc93eb4c9237b931293af91d6a1e999bf626420b37d8def15b8290709d38"
V12_MANIFEST_SHA256 = "6ef8e99fe52ce30613119e81c1b0d788655d73c46ca7852d42a302e039b9c3ac"
VALID_QUOTAS = {
    "negation": 96,
    "critical": 128,
    "repetition-risk": 1,
    "terminology-risk": 96,
    "omission-risk": 16,
    "long": 192,
    "general": 239,
}
OPEN_LICENSES = {
    "CC-BY-2.0-FR",
    "CC-BY-4.0",
    "CC-BY-SA-3.0",
    "PDL-1.0-compatible-CC-BY-4.0",
    "project-owned",
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"missing JSONL input: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def display_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": display_path(path, root),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def stable_rank(seed: int, bucket: str, row: dict[str, Any]) -> str:
    return hashlib.sha256(
        (
            f"{seed}\0{bucket}\0{row.get('id', '')}\0"
            f"{row.get('source', '')}\0{row.get('target', '')}"
        ).encode()
    ).hexdigest()


def ensure_translation_row(row: dict[str, Any], label: str) -> None:
    if (row.get("source_language"), row.get("target_language")) != (
        "ja-JP",
        "en-US",
    ):
        raise SystemExit(f"{label} contains wrong-direction row: {row.get('id')}")
    for field in (
        "id",
        "source",
        "target",
        "origin",
        "source_license",
        "source_provenance",
    ):
        if not str(row.get(field, "")).strip():
            raise SystemExit(f"{label} row lacks {field}: {row.get('id')}")
    if row["source_license"] not in OPEN_LICENSES:
        raise SystemExit(
            f"{label} row has an unregistered license: "
            f"{row.get('id')}: {row.get('source_license')}"
        )


def protected_match(
    index: ProtectedIndex,
    row: dict[str, Any],
    *,
    maximum_jaccard: float,
    ngram_size: int,
) -> bool:
    return any(
        index.match(
            str(row[field]),
            maximum_jaccard,
            ngram_size,
        )
        is not None
        for field in ("source", "target")
    )


def authenticated_split(
    directory: Path,
    manifest: dict[str, Any],
    split: str,
) -> tuple[Path, list[dict[str, Any]]]:
    path = directory / f"{split}.jsonl"
    if manifest.get("outputs", {}).get(split, {}).get("sha256") != sha256(path):
        raise SystemExit(f"manifest does not authenticate {directory.name}/{split}")
    return path, load_jsonl(path)


def select_bucket(
    candidates: list[dict[str, Any]],
    *,
    quota: int,
    seed: int,
    bucket: str,
    blocked_sources: set[str],
    protected: ProtectedIndex,
    maximum_jaccard: float,
    ngram_size: int,
    rejected: Counter[str],
) -> list[dict[str, Any]]:
    selected = []
    for row in sorted(
        candidates,
        key=lambda item: (
            stable_rank(seed, bucket, item),
            str(item["id"]),
        ),
    ):
        source_key = normalized(str(row["source"]))
        if source_key in blocked_sources:
            rejected[f"{bucket}:source-overlap"] += 1
            continue
        if protected_match(
            protected,
            row,
            maximum_jaccard=maximum_jaccard,
            ngram_size=ngram_size,
        ):
            rejected[f"{bucket}:protected-overlap"] += 1
            continue
        blocked_sources.add(source_key)
        selected.append(row)
        if len(selected) == quota:
            break
    if len(selected) != quota:
        raise SystemExit(
            f"insufficient clean rows for {bucket}: {len(selected)} < {quota}"
        )
    return selected


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("v10_directory", type=Path)
    parser.add_argument("v12_directory", type=Path)
    parser.add_argument("legal_directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--protected-suite",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--character-ngram-size", type=int, default=5)
    parser.add_argument("--maximum-jaccard", type=float, default=0.8)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    if args.character_ngram_size < 1 or not 0 <= args.maximum_jaccard < 1:
        raise SystemExit("invalid protected-overlap settings")

    root = Path(__file__).resolve().parents[2]
    v10_manifest_path = args.v10_directory / "manifest.json"
    v12_manifest_path = args.v12_directory / "manifest.json"
    v10_manifest = load_json(v10_manifest_path)
    v12_manifest = load_json(v12_manifest_path)
    if (
        sha256(v10_manifest_path) != V10_MANIFEST_SHA256
        or v10_manifest.get("experiment") != V10_EXPERIMENT
        or sha256(v12_manifest_path) != V12_MANIFEST_SHA256
        or v12_manifest.get("experiment") != V12_EXPERIMENT
        or v12_manifest.get("status") != "frozen-ready-for-negative-generation"
    ):
        raise SystemExit("v10/v12 source dataset identity differs")

    v10_splits = {}
    v12_splits = {}
    source_inputs: dict[str, dict[str, Any]] = {
        "v10_manifest": record(v10_manifest_path, root),
        "v12_manifest": record(v12_manifest_path, root),
    }
    for prefix, directory, manifest, destination in (
        ("v10", args.v10_directory, v10_manifest, v10_splits),
        ("v12", args.v12_directory, v12_manifest, v12_splits),
    ):
        for split in ("train", "valid"):
            path, values = authenticated_split(
                directory,
                manifest,
                split,
            )
            destination[split] = values
            source_inputs[f"{prefix}_{split}"] = record(path, root)

    train_rows = []
    train_sources: set[str] = set()
    for row in v12_splits["train"]:
        copied = {
            **row,
            "id": f"v14-train:{row['id']}",
            "domain": f"v14-rollout:{row.get('domain', 'unknown')}",
            "v14_stratum": str(
                row.get("v12_stratum")
                or row.get("v10_stratum")
                or row.get("sequence_role", "unknown")
            ),
            "sequence_role": ("licensed-human-rollout-repair-reference"),
        }
        ensure_translation_row(copied, "v14 train")
        source_key = normalized(str(copied["source"]))
        if source_key in train_sources:
            raise SystemExit(f"v14 train repeats a source: {copied['id']}")
        train_sources.add(source_key)
        train_rows.append(copied)
    if len(train_rows) != 7_104:
        raise SystemExit(f"unexpected v14 train count: {len(train_rows)}")

    protected = ProtectedIndex(
        args.protected_suite,
        args.character_ngram_size,
    )
    for row in train_rows:
        if protected_match(
            protected,
            row,
            maximum_jaccard=args.maximum_jaccard,
            ngram_size=args.character_ngram_size,
        ):
            raise SystemExit(
                f"v14 inherited train row overlaps protected data: {row['id']}"
            )

    blocked_sources = {
        normalized(str(row["source"]))
        for values in (
            *v10_splits.values(),
            *v12_splits.values(),
        )
        for row in values
    }
    legal_manifest_path = args.legal_directory / "manifest.json"
    legal_manifest = load_json(legal_manifest_path)
    legal_test_path = args.legal_directory / "test.jsonl"
    if (
        legal_manifest.get("outputs", {}).get("test", {}).get("sha256")
        != sha256(legal_test_path)
        or legal_manifest.get("license") != "PDL-1.0-compatible-CC-BY-4.0"
        or legal_manifest.get("private_reasoning_traces_used") is not False
    ):
        raise SystemExit("legal test input is not authenticated or distributable")
    source_inputs["legal_manifest"] = record(
        legal_manifest_path,
        root,
    )
    source_inputs["legal_test"] = record(legal_test_path, root)

    by_category: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(legal_test_path):
        category = legal_category(row)
        if category is None:
            continue
        candidate = {
            **row,
            "id": f"v14-valid:{category}:{row['id']}",
            "domain": "v14-fresh-legal-test",
            "v14_stratum": f"legal:{category}",
            "sequence_role": "licensed-human-fresh-v14-reference",
        }
        ensure_translation_row(candidate, "v14 legal validation")
        by_category[category].append(candidate)

    rejected: Counter[str] = Counter()
    valid_rows = []
    for category, quota in VALID_QUOTAS.items():
        valid_rows.extend(
            select_bucket(
                by_category[category],
                quota=quota,
                seed=args.seed,
                bucket=f"legal:{category}",
                blocked_sources=blocked_sources,
                protected=protected,
                maximum_jaccard=args.maximum_jaccard,
                ngram_size=args.character_ngram_size,
                rejected=rejected,
            )
        )
    if len(valid_rows) != 768:
        raise SystemExit(f"unexpected v14 validation count: {len(valid_rows)}")
    valid_rows.sort(key=lambda row: stable_rank(args.seed, "valid", row))
    train_rows.sort(key=lambda row: stable_rank(args.seed, "train", row))
    valid_sources = {normalized(str(row["source"])) for row in valid_rows}
    if train_sources & valid_sources:
        raise SystemExit("v14 train and validation sources overlap")
    if len(valid_sources) != len(valid_rows):
        raise SystemExit("v14 validation contains duplicate sources")

    args.output.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "train": args.output / "train.jsonl",
        "valid": args.output / "valid.jsonl",
    }
    write_jsonl(output_paths["train"], train_rows)
    write_jsonl(output_paths["valid"], valid_rows)

    attribution_counts: Counter[tuple[str, str, str]] = Counter()
    for row in [*train_rows, *valid_rows]:
        attribution_counts[
            (
                str(row["origin"]),
                str(row["source_license"]),
                str(row["source_provenance"]),
            )
        ] += 1
    attribution_path = args.output / "attribution.jsonl"
    write_jsonl(
        attribution_path,
        [
            {
                "origin": origin,
                "license": license_name,
                "provenance": provenance,
                "rows": count,
            }
            for (origin, license_name, provenance), count in sorted(
                attribution_counts.items()
            )
        ],
    )

    manifest = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": "frozen-ready-for-rollout-mining",
        "direction": "ja-en",
        "purpose": (
            "rollout-conditioned safety repair with a fresh, "
            "source-disjoint legal validation suite"
        ),
        "target_source": "licensed human references only",
        "promotion_eligible": False,
        "training_only": True,
        "private_reasoning_traces_used": False,
        "free_form_synthetic_translations_used": False,
        "human_reviewer_required": False,
        "selection": {
            "seed": args.seed,
            "train_policy": (
                "reuse the authenticated v12 licensed-human train split; "
                "never use v12 validation as training"
            ),
            "validation_policy": (
                "deterministic SHA-256 rank inside frozen legal-risk "
                "quotas after excluding every v10/v12 train and validation "
                "source and screening source plus target against protected "
                "suites"
            ),
            "valid_quotas": VALID_QUOTAS,
        },
        "counts": {
            "train": len(train_rows),
            "valid": len(valid_rows),
            "unique_train_sources": len(train_sources),
            "unique_valid_sources": len(valid_sources),
            "train_origins": dict(
                sorted(Counter(str(row["origin"]) for row in train_rows).items())
            ),
            "valid_origins": dict(
                sorted(Counter(str(row["origin"]) for row in valid_rows).items())
            ),
            "train_strata": dict(
                sorted(Counter(str(row["v14_stratum"]) for row in train_rows).items())
            ),
            "valid_strata": dict(
                sorted(Counter(str(row["v14_stratum"]) for row in valid_rows).items())
            ),
            "rejected": dict(sorted(rejected.items())),
        },
        "effective_licenses": {
            split: dict(
                sorted(Counter(str(row["source_license"]) for row in values).items())
            )
            for split, values in (
                ("train", train_rows),
                ("valid", valid_rows),
            )
        },
        "distribution_provenance": {
            "all_rows_have_source_license": True,
            "all_rows_have_source_provenance": True,
            "licenses_are_open_or_project_owned": sorted(
                {str(row["source_license"]) for row in [*train_rows, *valid_rows]}
            ),
            "attribution_sidecar": record(attribution_path, root),
        },
        "decontamination": {
            "normalization": "Unicode NFKC, casefold, remove whitespace",
            "character_ngram_size": args.character_ngram_size,
            "maximum_jaccard_exclusive": args.maximum_jaccard,
            "screened_fields": ["source", "target"],
            "protected_suites": [record(path, root) for path in args.protected_suite],
            "all_v10_and_v12_sources_excluded_from_fresh_validation": True,
            "train_valid_source_overlap": False,
            "protected_hits_in_outputs": 0,
        },
        "inputs": source_inputs,
        "outputs": {
            split: {
                **record(path, root),
                "rows": len(values),
            }
            for split, path, values in (
                ("train", output_paths["train"], train_rows),
                ("valid", output_paths["valid"], valid_rows),
            )
        },
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": display_path(args.output, root),
                "manifest_sha256": sha256(manifest_path),
                "train_rows": len(train_rows),
                "valid_rows": len(valid_rows),
                "valid_strata": manifest["counts"]["valid_strata"],
                "rejected": manifest["counts"]["rejected"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
