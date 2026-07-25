#!/usr/bin/env python3
"""Build licensed training data and a fresh legal suite for v15."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_canonical_rollout_repair_v14_dataset import (
    ensure_translation_row,
    load_json,
    load_jsonl,
    protected_match,
    record,
    stable_rank,
    write_jsonl,
)
from build_canonical_sequence_v10_dataset import legal_category
from filter_training_dataset_against_protected import (
    ProtectedIndex,
    normalized,
    sha256,
)

EXPERIMENT = "canonical-constrained-recovery-v15-ja-en"
V14_EXPERIMENT = "canonical-rollout-repair-v14-ja-en"
V14_MANIFEST_SHA256 = "1cd2e3629513f4662c6c9ffd6854d463bd638f08c8001bdb73027db0dc03d245"
VALID_QUOTAS = {
    "negation": 96,
    "critical": 128,
    "terminology-risk": 80,
    "omission-risk": 6,
    "long": 192,
    "general": 266,
}


def authenticated_path(
    root: Path,
    item: dict[str, Any],
    *,
    label: str,
) -> Path:
    path = root / str(item["path"])
    if sha256(path) != item.get("sha256"):
        raise SystemExit(f"v15 input bytes differ: {label}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("v14_directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")

    root = Path(__file__).resolve().parents[2]
    manifest_path = args.v14_directory / "manifest.json"
    manifest = load_json(manifest_path)
    if (
        sha256(manifest_path) != V14_MANIFEST_SHA256
        or manifest.get("experiment") != V14_EXPERIMENT
        or manifest.get("status") != "frozen-ready-for-rollout-mining"
        or manifest.get("direction") != "ja-en"
        or manifest.get("promotion_eligible") is not False
        or manifest.get("private_reasoning_traces_used") is not False
        or manifest.get("free_form_synthetic_translations_used") is not False
        or manifest.get("counts", {}).get("train") != 7_104
        or manifest.get("counts", {}).get("valid") != 768
    ):
        raise SystemExit("v14 source dataset identity or safety state differs")

    v14_train_path = args.v14_directory / "train.jsonl"
    v14_valid_path = args.v14_directory / "valid.jsonl"
    for split, path in (("train", v14_train_path), ("valid", v14_valid_path)):
        if manifest.get("outputs", {}).get(split, {}).get("sha256") != sha256(path):
            raise SystemExit(f"v14 manifest does not authenticate {split}")
    v14_train = load_jsonl(v14_train_path)
    v14_valid = load_jsonl(v14_valid_path)

    input_paths: dict[str, Path] = {
        "v14_manifest": manifest_path,
        "v14_train": v14_train_path,
        "v14_valid": v14_valid_path,
    }
    prior_rows = [*v14_train, *v14_valid]
    for key in ("v10_train", "v10_valid", "v12_train", "v12_valid"):
        path = authenticated_path(root, manifest["inputs"][key], label=key)
        input_paths[key] = path
        prior_rows.extend(load_jsonl(path))

    legal_manifest_path = authenticated_path(
        root,
        manifest["inputs"]["legal_manifest"],
        label="legal_manifest",
    )
    legal_test_path = authenticated_path(
        root,
        manifest["inputs"]["legal_test"],
        label="legal_test",
    )
    legal_manifest = load_json(legal_manifest_path)
    if (
        legal_manifest.get("outputs", {}).get("test", {}).get("sha256")
        != sha256(legal_test_path)
        or legal_manifest.get("license") != "PDL-1.0-compatible-CC-BY-4.0"
        or legal_manifest.get("private_reasoning_traces_used") is not False
    ):
        raise SystemExit("legal source is not authenticated and distributable")
    input_paths["legal_manifest"] = legal_manifest_path
    input_paths["legal_test"] = legal_test_path

    protected_paths = []
    for index, item in enumerate(
        manifest.get("decontamination", {}).get("protected_suites", [])
    ):
        protected_paths.append(
            authenticated_path(root, item, label=f"protected_suite_{index}")
        )
    if len(protected_paths) != 10:
        raise SystemExit("v15 requires the exact ten protected suites")
    ngram_size = int(manifest["decontamination"]["character_ngram_size"])
    maximum_jaccard = float(manifest["decontamination"]["maximum_jaccard_exclusive"])
    protected = ProtectedIndex(protected_paths, ngram_size)

    train_rows = []
    train_sources: set[str] = set()
    for row in v14_train:
        copied = {
            **row,
            "id": str(row["id"]).replace("v14-train:", "v15-train:", 1),
            "domain": f"v15-constrained:{row.get('domain', 'unknown')}",
            "v15_stratum": str(row.get("v14_stratum", "unknown")),
            "sequence_role": "licensed-human-constrained-recovery-reference",
        }
        ensure_translation_row(copied, "v15 train")
        source_key = normalized(str(copied["source"]))
        if source_key in train_sources:
            raise SystemExit(f"v15 train repeats a source: {copied['id']}")
        if protected_match(
            protected,
            copied,
            maximum_jaccard=maximum_jaccard,
            ngram_size=ngram_size,
        ):
            raise SystemExit(f"v15 train overlaps protected data: {copied['id']}")
        train_sources.add(source_key)
        train_rows.append(copied)
    if len(train_rows) != 7_104:
        raise SystemExit(f"unexpected v15 train count: {len(train_rows)}")

    blocked_sources = {normalized(str(row["source"])) for row in prior_rows}
    by_category: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(legal_test_path):
        category = legal_category(row)
        if category is None:
            continue
        candidate = {
            **row,
            "id": f"v15-valid:{category}:{row['id']}",
            "domain": "v15-fresh-legal-test",
            "v15_stratum": f"legal:{category}",
            "sequence_role": "licensed-human-fresh-v15-reference",
        }
        ensure_translation_row(candidate, "v15 legal validation")
        by_category[category].append(candidate)

    rejected: Counter[str] = Counter()
    clean_by_category: dict[str, list[dict[str, Any]]] = {}
    for category, candidates in by_category.items():
        clean = []
        clean_sources: set[str] = set()
        for row in sorted(
            candidates,
            key=lambda item: (
                stable_rank(args.seed, f"legal:{category}", item),
                str(item["id"]),
            ),
        ):
            source_key = normalized(str(row["source"]))
            if source_key in blocked_sources:
                rejected[f"legal:{category}:source-overlap"] += 1
                continue
            if source_key in clean_sources:
                rejected[f"legal:{category}:duplicate-source"] += 1
                continue
            if protected_match(
                protected,
                row,
                maximum_jaccard=maximum_jaccard,
                ngram_size=ngram_size,
            ):
                rejected[f"legal:{category}:protected-overlap"] += 1
                continue
            clean_sources.add(source_key)
            clean.append(row)
        clean_by_category[category] = clean
    clean_availability = {
        category: len(values) for category, values in sorted(clean_by_category.items())
    }
    insufficient = {
        category: (clean_availability.get(category, 0), quota)
        for category, quota in VALID_QUOTAS.items()
        if clean_availability.get(category, 0) < quota
    }
    if insufficient:
        raise SystemExit(
            "insufficient fresh clean legal rows: "
            f"{insufficient}; availability={clean_availability}"
        )

    valid_rows = []
    for category, quota in VALID_QUOTAS.items():
        selected = sorted(
            clean_by_category[category],
            key=lambda row: (
                stable_rank(args.seed, f"legal:{category}", row),
                str(row["id"]),
            ),
        )[:quota]
        for row in selected:
            blocked_sources.add(normalized(str(row["source"])))
        valid_rows.extend(selected)
    if len(valid_rows) != 768:
        raise SystemExit(f"unexpected v15 validation count: {len(valid_rows)}")
    train_rows.sort(key=lambda row: stable_rank(args.seed, "train", row))
    valid_rows.sort(key=lambda row: stable_rank(args.seed, "valid", row))
    valid_sources = {normalized(str(row["source"])) for row in valid_rows}
    if train_sources & valid_sources or len(valid_sources) != len(valid_rows):
        raise SystemExit("v15 train/validation sources overlap or repeat")

    args.output.mkdir(parents=True, exist_ok=True)
    train_path = args.output / "train.jsonl"
    valid_path = args.output / "valid.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(valid_path, valid_rows)

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

    output_manifest = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": "frozen-ready-for-contrastive-example-building",
        "direction": "ja-en",
        "purpose": (
            "licensed-reference constrained recovery and omission repair "
            "with a fresh source-disjoint legal validation suite"
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
                "reuse the authenticated licensed-human v14 training rows only"
            ),
            "validation_policy": (
                "deterministic rank inside frozen legal-risk quotas after "
                "excluding every v10/v12/v14 train or validation source and "
                "screening both sides against the exact ten protected suites; "
                "the exhausted one-row repetition-risk reservoir is not reused "
                "and repetition remains covered by v12/v14 regression suites"
            ),
            "valid_quotas": VALID_QUOTAS,
            "post_exclusion_clean_availability": clean_availability,
        },
        "counts": {
            "train": len(train_rows),
            "valid": len(valid_rows),
            "unique_train_sources": len(train_sources),
            "unique_valid_sources": len(valid_sources),
            "train_strata": dict(
                sorted(Counter(str(row["v15_stratum"]) for row in train_rows).items())
            ),
            "valid_strata": dict(
                sorted(Counter(str(row["v15_stratum"]) for row in valid_rows).items())
            ),
        },
        "inputs": {
            key: record(path, root) for key, path in sorted(input_paths.items())
        },
        "outputs": {
            "train": {**record(train_path, root), "rows": len(train_rows)},
            "valid": {**record(valid_path, root), "rows": len(valid_rows)},
            "attribution": record(attribution_path, root),
        },
        "effective_licenses": sorted(
            {str(row["source_license"]) for row in [*train_rows, *valid_rows]}
        ),
        "distribution_provenance": {
            "all_positive_targets_are_licensed_human_references": True,
            "all_rows_have_source_license": all(
                bool(row.get("source_license")) for row in [*train_rows, *valid_rows]
            ),
            "all_rows_have_source_provenance": all(
                bool(row.get("source_provenance")) for row in [*train_rows, *valid_rows]
            ),
            "generated_strings_are_positive_targets": False,
            "private_reasoning_traces_used": False,
        },
        "decontamination": {
            "normalization": "Unicode NFKC, casefold, remove whitespace",
            "character_ngram_size": ngram_size,
            "maximum_jaccard_exclusive": maximum_jaccard,
            "protected_suites": [record(path, root) for path in protected_paths],
            "screened_fields": ["source", "target"],
            "protected_hits_in_outputs": 0,
            "all_v10_v12_v14_sources_excluded_from_fresh_validation": True,
            "train_valid_source_overlap": False,
            "rejected": dict(sorted(rejected.items())),
        },
    }
    manifest_output_path = args.output / "manifest.json"
    manifest_output_path.write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "manifest_sha256": sha256(manifest_output_path),
                "counts": output_manifest["counts"],
                "status": output_manifest["status"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
