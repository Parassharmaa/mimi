#!/usr/bin/env python3
"""Freeze V16 active sequence risks and a fresh legal validation suite."""

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

EXPERIMENT = "active-sequence-risk-v16-ja-en"
V15_EXPERIMENT = "canonical-constrained-recovery-v15-ja-en"
V15_MANIFEST_SHA256 = "9b412e0a7d49234ab374f4e47fc71e0f70e9cb432af6f792d26e7ce56910c523"
DIAGNOSTIC_RESULT_SHA256 = (
    "0272e49d4a6ebd9d87df8b51099beb354510c26f277b4b29b29d9ab98d98978d"
)
VALID_QUOTAS = {
    "critical": 128,
    "general": 345,
    "long": 192,
    "negation": 96,
    "terminology-risk": 7,
}


def authenticated_path(
    root: Path,
    item: dict[str, Any],
    *,
    label: str,
) -> Path:
    path = root / str(item["path"])
    if sha256(path) != item.get("sha256"):
        raise SystemExit(f"v16 input bytes differ: {label}")
    return path


def validate_active_row(
    row: dict[str, Any],
    *,
    train_ids: set[str],
) -> None:
    required = (
        "id",
        "parent_id",
        "source",
        "chosen",
        "rejected",
        "risk_role",
        "source_license",
        "source_provenance",
        "chosen_minus_rejected_margin",
    )
    if any(not str(row.get(field, "")).strip() for field in required):
        raise SystemExit(f"incomplete V16 active row: {row.get('id')}")
    if row["risk_role"] not in {"omission", "repetition"}:
        raise SystemExit(f"invalid V16 active role: {row['risk_role']}")
    if row["parent_id"] not in train_ids:
        raise SystemExit(f"V16 active row is not training-only: {row['id']}")
    if row.get("positive_target_source") != "licensed-human-reference":
        raise SystemExit(f"V16 active positive is not a human reference: {row['id']}")
    if row.get("generated_strings_are_positive_targets") is not False:
        raise SystemExit(f"V16 active row promotes generated text: {row['id']}")
    if float(row["chosen_minus_rejected_margin"]) >= 0.25:
        raise SystemExit(f"V16 active row exceeds the frozen margin: {row['id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("v15_directory", type=Path)
    parser.add_argument("diagnostic_result", type=Path)
    parser.add_argument("diagnostic_directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")

    root = Path(__file__).resolve().parents[2]
    v15_manifest_path = args.v15_directory / "manifest.json"
    v15_manifest = load_json(v15_manifest_path)
    if (
        sha256(v15_manifest_path) != V15_MANIFEST_SHA256
        or v15_manifest.get("experiment") != V15_EXPERIMENT
        or v15_manifest.get("status") != "frozen-ready-for-contrastive-example-building"
        or v15_manifest.get("direction") != "ja-en"
        or v15_manifest.get("counts", {}).get("train") != 7_104
        or v15_manifest.get("counts", {}).get("valid") != 768
        or v15_manifest.get("promotion_eligible") is not False
        or v15_manifest.get("distribution_provenance", {}).get(
            "all_positive_targets_are_licensed_human_references"
        )
        is not True
        or v15_manifest.get("decontamination", {}).get("protected_hits_in_outputs") != 0
    ):
        raise SystemExit("V15 source dataset identity or safety state differs")

    train_path = args.v15_directory / "train.jsonl"
    v15_valid_path = args.v15_directory / "valid.jsonl"
    for split, path in (("train", train_path), ("valid", v15_valid_path)):
        if v15_manifest["outputs"][split]["sha256"] != sha256(path):
            raise SystemExit(f"V15 manifest does not authenticate {split}")
    train_rows = load_jsonl(train_path)
    if len(train_rows) != 7_104:
        raise SystemExit("V16 requires the exact 7,104-row V15 training set")
    train_ids = {str(row["id"]) for row in train_rows}
    train_sources = {normalized(str(row["source"])) for row in train_rows}
    if len(train_ids) != len(train_rows) or len(train_sources) != len(train_rows):
        raise SystemExit("V15 training IDs or sources repeat")

    diagnostic = load_json(args.diagnostic_result)
    active_input = args.diagnostic_directory / "active.jsonl"
    scored_input = args.diagnostic_directory / "scored.jsonl"
    if (
        sha256(args.diagnostic_result) != DIAGNOSTIC_RESULT_SHA256
        or diagnostic.get("experiment") != "active-sequence-risk-v16-ja-en-diagnostic"
        or diagnostic.get("status") != "diagnostic-complete-no-training-authorized"
        or diagnostic.get("training_authorized") is not False
        or diagnostic.get("dataset", {}).get("train", {}).get("sha256")
        != sha256(train_path)
        or diagnostic.get("outputs", {}).get("active", {}).get("sha256")
        != sha256(active_input)
        or diagnostic.get("outputs", {}).get("scored", {}).get("sha256")
        != sha256(scored_input)
    ):
        raise SystemExit("V16 diagnostic identity or safety state differs")
    active_rows = load_jsonl(active_input)
    for row in active_rows:
        validate_active_row(row, train_ids=train_ids)
    if len(active_rows) != 677 or Counter(
        str(row["risk_role"]) for row in active_rows
    ) != {"omission": 228, "repetition": 449}:
        raise SystemExit("V16 active-pair population differs")

    input_paths: dict[str, Path] = {
        "diagnostic_active": active_input,
        "diagnostic_result": args.diagnostic_result,
        "diagnostic_scored": scored_input,
        "v15_manifest": v15_manifest_path,
        "v15_train": train_path,
        "v15_valid": v15_valid_path,
    }
    prior_rows = [*train_rows, *load_jsonl(v15_valid_path)]
    for key in (
        "v10_train",
        "v10_valid",
        "v12_train",
        "v12_valid",
        "v14_train",
        "v14_valid",
    ):
        path = authenticated_path(root, v15_manifest["inputs"][key], label=key)
        input_paths[key] = path
        prior_rows.extend(load_jsonl(path))

    legal_manifest_path = authenticated_path(
        root,
        v15_manifest["inputs"]["legal_manifest"],
        label="legal_manifest",
    )
    legal_test_path = authenticated_path(
        root,
        v15_manifest["inputs"]["legal_test"],
        label="legal_test",
    )
    legal_manifest = load_json(legal_manifest_path)
    if (
        legal_manifest.get("outputs", {}).get("test", {}).get("sha256")
        != sha256(legal_test_path)
        or legal_manifest.get("license") != "PDL-1.0-compatible-CC-BY-4.0"
        or legal_manifest.get("private_reasoning_traces_used") is not False
    ):
        raise SystemExit("V16 legal source is not authenticated and distributable")
    input_paths["legal_manifest"] = legal_manifest_path
    input_paths["legal_test"] = legal_test_path

    protected_paths = [
        authenticated_path(root, item, label=f"protected_suite_{index}")
        for index, item in enumerate(
            v15_manifest.get("decontamination", {}).get("protected_suites", [])
        )
    ]
    if len(protected_paths) != 10:
        raise SystemExit("V16 requires the exact ten protected suites")
    ngram_size = int(v15_manifest["decontamination"]["character_ngram_size"])
    maximum_jaccard = float(
        v15_manifest["decontamination"]["maximum_jaccard_exclusive"]
    )
    protected = ProtectedIndex(protected_paths, ngram_size)

    blocked_sources = {normalized(str(row["source"])) for row in prior_rows}
    by_category: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(legal_test_path):
        category = legal_category(row)
        if category is None:
            continue
        candidate = {
            **row,
            "id": f"v16-valid:{category}:{row['id']}",
            "domain": "v16-fresh-legal-test",
            "v16_stratum": f"legal:{category}",
            "sequence_role": "licensed-human-fresh-v16-reference",
        }
        ensure_translation_row(candidate, "V16 legal validation")
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
            "insufficient fresh clean V16 legal rows: "
            f"{insufficient}; availability={clean_availability}"
        )

    valid_rows = []
    for category, quota in VALID_QUOTAS.items():
        valid_rows.extend(
            sorted(
                clean_by_category[category],
                key=lambda row: (
                    stable_rank(args.seed, f"legal:{category}", row),
                    str(row["id"]),
                ),
            )[:quota]
        )
    valid_rows.sort(key=lambda row: stable_rank(args.seed, "valid", row))
    valid_sources = {normalized(str(row["source"])) for row in valid_rows}
    if (
        len(valid_rows) != 768
        or len(valid_sources) != len(valid_rows)
        or train_sources & valid_sources
    ):
        raise SystemExit("V16 validation count, uniqueness, or separation differs")

    args.output.mkdir(parents=True, exist_ok=True)
    active_path = args.output / "active.jsonl"
    valid_path = args.output / "valid.jsonl"
    write_jsonl(active_path, active_rows)
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

    all_rows = [*train_rows, *valid_rows]
    manifest = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": "frozen-ready-for-pcgrad-contract",
        "direction": "ja-en",
        "purpose": (
            "active full-sequence omission and repetition risk training with "
            "fresh source-disjoint legal validation"
        ),
        "promotion_eligible": False,
        "training_only": True,
        "target_source": "licensed human references only",
        "private_reasoning_traces_used": False,
        "free_form_synthetic_translations_used": False,
        "selection": {
            "seed": args.seed,
            "active_margin_exclusive": 0.25,
            "active_policy": (
                "preserve all diagnostic full-sequence comparisons below the "
                "frozen safe-parent margin"
            ),
            "validation_policy": (
                "deterministic rank inside frozen remaining legal-risk quotas "
                "after excluding every V10/V12/V14/V15 train or validation "
                "source and screening both sides against protected suites"
            ),
            "valid_quotas": VALID_QUOTAS,
            "post_exclusion_clean_availability": clean_availability,
            "exhausted_fresh_strata": ["omission-risk", "repetition-risk"],
        },
        "counts": {
            "train": len(train_rows),
            "active": len(active_rows),
            "active_roles": dict(
                sorted(Counter(str(row["risk_role"]) for row in active_rows).items())
            ),
            "negative_preferred": sum(
                float(row["chosen_minus_rejected_margin"]) < 0 for row in active_rows
            ),
            "valid": len(valid_rows),
            "unique_train_sources": len(train_sources),
            "unique_valid_sources": len(valid_sources),
            "valid_strata": dict(
                sorted(Counter(str(row["v16_stratum"]) for row in valid_rows).items())
            ),
        },
        "inputs": {
            key: record(path, root) for key, path in sorted(input_paths.items())
        },
        "outputs": {
            "active": {**record(active_path, root), "rows": len(active_rows)},
            "valid": {**record(valid_path, root), "rows": len(valid_rows)},
            "attribution": record(attribution_path, root),
        },
        "effective_licenses": sorted({str(row["source_license"]) for row in all_rows}),
        "distribution_provenance": {
            "all_positive_targets_are_licensed_human_references": True,
            "all_rows_have_source_license": all(
                bool(row.get("source_license")) for row in all_rows
            ),
            "all_rows_have_source_provenance": all(
                bool(row.get("source_provenance")) for row in all_rows
            ),
            "generated_strings_are_positive_targets": False,
            "negative_strings_are_positive_targets": False,
            "private_reasoning_traces_used": False,
        },
        "decontamination": {
            "normalization": "Unicode NFKC, casefold, remove whitespace",
            "character_ngram_size": ngram_size,
            "maximum_jaccard_exclusive": maximum_jaccard,
            "protected_suites": [record(path, root) for path in protected_paths],
            "screened_fields": ["source", "target"],
            "protected_hits_in_outputs": 0,
            "all_v10_v12_v14_v15_sources_excluded_from_fresh_validation": True,
            "train_valid_source_overlap": False,
            "rejected": dict(sorted(rejected.items())),
        },
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
                "availability": clean_availability,
                "status": manifest["status"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
