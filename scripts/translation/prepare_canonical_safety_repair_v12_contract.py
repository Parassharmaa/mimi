#!/usr/bin/env python3
"""Freeze the one-arm v12 constraint-aware safety-repair contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

EXPERIMENT = "canonical-safety-repair-v12-ja-en"
REPOSITORY = "Mitsua/elan-mt-bt-ja-en"
REVISION = "539f80eb05306e27a166b45e4264c7fa2eb4de97"
INITIAL_MODEL_SHA256 = (
    "33a045164ff00712aa5dbf8c6b0a2f4736aeb976d37053c061f1bd0241c81249"
)
INITIAL_MANIFEST_SHA256 = (
    "409cdc56dc3d49e08c80873dcacd3f86d6d75f68c09a43834d1adf2101347ea9"
)
INITIAL_INTERPOLATION_SHA256 = (
    "84db1af7288e9744eb6182c4a6c74671a32ab83c9b221f3a7cab51ca7e1d5356"
)
PARENT_MODEL_SHA256 = "8e7f7eff76d74b343884fe9a170b6dbad55d42f20ac5f526b6e8ec71e6c94f71"
PARENT_MANIFEST_SHA256 = (
    "0d195dc163250a9fa9312fb7ad8ba3341ab65167b90926d46f0a65d76047bd38"
)
CHECKPOINT_STEPS = [50, 100]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


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


def validate_checkpoint(
    checkpoint: Path,
    *,
    expected_model_sha256: str,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    model = checkpoint / "model.safetensors"
    manifest_path = checkpoint / "mimi_training_manifest.json"
    if (
        sha256(model) != expected_model_sha256
        or sha256(manifest_path) != expected_manifest_sha256
    ):
        raise SystemExit(f"bound checkpoint bytes differ: {checkpoint}")
    manifest = load_json(manifest_path)
    if (
        manifest.get("direction") != "ja-en"
        or manifest.get("student_repository") != REPOSITORY
        or manifest.get("student_revision") != REVISION
        or manifest.get("license") != "CC-BY-SA-4.0"
    ):
        raise SystemExit(f"bound checkpoint identity differs: {checkpoint}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("negative_directory", type=Path)
    parser.add_argument("initial_checkpoint", type=Path)
    parser.add_argument("preservation_checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--protected-suite",
        type=Path,
        action="append",
        required=True,
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")

    root = Path(__file__).resolve().parents[2]
    dataset_manifest_path = args.dataset_directory / "manifest.json"
    dataset_manifest = load_json(dataset_manifest_path)
    dataset_paths = {
        split: args.dataset_directory / f"{split}.jsonl" for split in ("train", "valid")
    }
    if (
        dataset_manifest.get("experiment") != EXPERIMENT
        or dataset_manifest.get("status") != "frozen-ready-for-negative-generation"
        or dataset_manifest.get("direction") != "ja-en"
        or dataset_manifest.get("promotion_eligible") is not False
        or dataset_manifest.get("private_reasoning_traces_used") is not False
        or dataset_manifest.get("free_form_synthetic_translations_used") is not False
        or dataset_manifest.get("counts", {}).get("train") != 7_104
        or dataset_manifest.get("counts", {}).get("valid") != 1_536
    ):
        raise SystemExit("v12 dataset manifest safety state differs")
    for split, path in dataset_paths.items():
        if dataset_manifest.get("outputs", {}).get(split, {}).get("sha256") != sha256(
            path
        ):
            raise SystemExit(f"v12 dataset does not authenticate {split}")
    distribution = dataset_manifest.get("distribution_provenance", {})
    if (
        distribution.get("all_rows_have_source_license") is not True
        or distribution.get("all_rows_have_source_provenance") is not True
    ):
        raise SystemExit("v12 distribution provenance is incomplete")

    negative_manifest_path = args.negative_directory / "manifest.json"
    negative_manifest = load_json(negative_manifest_path)
    negative_paths = {
        split: args.negative_directory / f"{split}.jsonl"
        for split in ("train", "valid")
    }
    if (
        negative_manifest.get("experiment")
        != "deterministic token-local negative-space Marian adaptation"
        or negative_manifest.get("direction") != "ja-en"
        or negative_manifest.get("promotion_eligible") is not False
        or negative_manifest.get("private_reasoning_traces_used") is not False
        or negative_manifest.get("free_form_synthetic_translations_used") is not False
        or negative_manifest.get("selection", {}).get("train_positive_rows") != 6_000
        or negative_manifest.get("selection", {}).get("valid_positive_rows") != 512
        or negative_manifest.get("selection", {}).get("maximum_violations_per_positive")
        != 4
        or negative_manifest.get("parent", {}).get("manifest_sha256")
        != sha256(dataset_manifest_path)
    ):
        raise SystemExit("v12 negative dataset safety state differs")
    for split, path in negative_paths.items():
        if negative_manifest.get("outputs", {}).get(split, {}).get("sha256") != sha256(
            path
        ):
            raise SystemExit(f"v12 negative dataset does not authenticate {split}")

    validate_checkpoint(
        args.initial_checkpoint,
        expected_model_sha256=INITIAL_MODEL_SHA256,
        expected_manifest_sha256=INITIAL_MANIFEST_SHA256,
    )
    validate_checkpoint(
        args.preservation_checkpoint,
        expected_model_sha256=PARENT_MODEL_SHA256,
        expected_manifest_sha256=PARENT_MANIFEST_SHA256,
    )
    interpolation_path = (
        args.initial_checkpoint / "mimi_checkpoint_interpolation_manifest.json"
    )
    interpolation = load_json(interpolation_path)
    if (
        sha256(interpolation_path) != INITIAL_INTERPOLATION_SHA256
        or interpolation.get("operation") != "linear-checkpoint-interpolation"
        or interpolation.get("adapted_weight") != 0.375
        or interpolation.get("output", {}).get("model_sha256") != INITIAL_MODEL_SHA256
        or interpolation.get("parent", {}).get("model_sha256") != PARENT_MODEL_SHA256
    ):
        raise SystemExit("v12 initial interpolation lineage differs")

    protected_records = [record(path, root) for path in args.protected_suite]
    manifest_protected = dataset_manifest.get("decontamination", {}).get(
        "protected_suites"
    )
    if not isinstance(manifest_protected, list) or {
        item["sha256"] for item in manifest_protected
    } != {item["sha256"] for item in protected_records}:
        raise SystemExit("v12 protected-suite inventory differs")

    implementation_paths = [
        Path(__file__).resolve(),
        root / "scripts/translation/build_canonical_safety_repair_v12_dataset.py",
        root / "scripts/translation/build_marian_negative_space_dataset.py",
        root / "scripts/translation/train_canonical_safety_repair_v12.py",
        root / "scripts/translation/evaluate_canonical_safety_repair_v12.py",
        root / "scripts/translation/train_marian_negative_space.py",
        root / "scripts/translation/train_marian_distillation.py",
        root / "scripts/translation/evaluate_canonical_sequence_v10_internal.py",
        root / "scripts/translation/audit_translation_structures.py",
        root / "scripts/translation/typed_critical_token_policy.py",
    ]
    training = {
        "seed": 20260804,
        "batch_size": 8,
        "gradient_accumulation": 4,
        "effective_batch_size": 32,
        "evaluation_batch_size": 16,
        "max_steps": 100,
        "learning_rate": 0.0000005,
        "weight_decay": 0.01,
        "warmup_steps": 10,
        "evaluation_steps": 50,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "max_source_tokens": 192,
        "max_target_tokens": 192,
        "gradient_checkpointing": False,
        "negative_weight": 0.15,
        "ranking_weight": 0.25,
        "ranking_target_margin": 1.0,
        "frozen_parent_kl_weight": 0.10,
        "l2_to_parent_weight": 0.00001,
        "one_arm_only": True,
        "post_result_hyperparameter_changes_forbidden": True,
    }
    result = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": "preregistered-ready-for-one-arm-training",
        "hypothesis": (
            "starting at the v11 quality/safety boundary, low-rate "
            "licensed-human cross-entropy plus token-local unlikelihood and "
            "chosen-over-rejected ranking can remove deterministic safety "
            "failure modes while frozen-parent KL/L2 preserves the safe "
            "parent and greedy-decoding latency"
        ),
        "direction": "ja-en",
        "dataset": {
            "directory": display_path(args.dataset_directory, root),
            "manifest": record(dataset_manifest_path, root),
            "train": record(dataset_paths["train"], root),
            "valid": record(dataset_paths["valid"], root),
            "attribution": record(
                args.dataset_directory / "attribution.jsonl",
                root,
            ),
            "counts": dataset_manifest["counts"],
            "effective_licenses": dataset_manifest["effective_licenses"],
            "distribution_provenance": distribution,
            "private_reasoning_traces_used": False,
            "free_form_synthetic_translations_used": False,
        },
        "negative_dataset": {
            "directory": display_path(args.negative_directory, root),
            "manifest": record(negative_manifest_path, root),
            "train": record(negative_paths["train"], root),
            "valid": record(negative_paths["valid"], root),
            "counts": negative_manifest["counts"],
            "selection": negative_manifest["selection"],
            "negative_strings_are_positive_targets": False,
        },
        "initial_checkpoint": {
            "path": display_path(args.initial_checkpoint, root),
            "repository": REPOSITORY,
            "revision": REVISION,
            "license": "CC-BY-SA-4.0",
            "model": record(
                args.initial_checkpoint / "model.safetensors",
                root,
            ),
            "training_manifest": record(
                args.initial_checkpoint / "mimi_training_manifest.json",
                root,
            ),
            "interpolation_manifest": record(interpolation_path, root),
            "adapted_weight": 0.375,
        },
        "preservation_checkpoint": {
            "path": display_path(args.preservation_checkpoint, root),
            "repository": REPOSITORY,
            "revision": REVISION,
            "license": "CC-BY-SA-4.0",
            "model": record(
                args.preservation_checkpoint / "model.safetensors",
                root,
            ),
            "training_manifest": record(
                args.preservation_checkpoint / "mimi_training_manifest.json",
                root,
            ),
        },
        "training": training,
        "internal_selection": {
            "suite": (
                "v12 frozen 1,536-row test-split suite, source-disjoint from "
                "all v10 rows and source/target-screened against ten protected "
                "suites"
            ),
            "selection_uses_protected_outputs": False,
            "candidate_steps": CHECKPOINT_STEPS,
            "step_zero_is_diagnostic_only": True,
            "baseline": "frozen safe parent, not the v11 initial checkpoint",
            "requirements": {
                "mean_sentence_chrf_pp_delta_minimum": 0.25,
                "corpus_chrf_pp_delta_minimum": 0.25,
                "fresh_general_chrf_pp_delta_minimum": 0.0,
                "long_legal_chrf_pp_delta_minimum": 0.25,
                "worst_stratum_chrf_pp_delta_minimum": -0.50,
                "paired_chrf_pp_90pct_lower_minimum": -0.25,
                "negative_validation_rejected_probability_delta_maximum": 0.0,
                "new_exact_critical_failures_maximum": 0,
                "new_typed_critical_failures_maximum": 0,
                "new_negation_failures_maximum": 0,
                "new_repetition_or_generation_limit_failures_maximum": 0,
            },
            "ordering": (
                "eligible checkpoints only; highest mean sentence chrF++, "
                "then long-legal chrF++, then fresh-general chrF++"
            ),
            "stop_if_no_checkpoint_is_eligible": True,
        },
        "next_step_if_internal_gate_passes": {
            "exact_q4_conversion_required": True,
            "bits": 4,
            "group_size": 64,
            "protected_evaluation_required": True,
            "protected_suites": protected_records,
            "metrics_required": [
                "chrF++",
                "BLEU",
                "COMET-22",
                "two-family blinded judge",
                "critical-structure audits",
                "latency",
                "memory",
                "bundle size",
            ],
        },
        "protected_promotion_gates": {
            "quality_each_direction": {
                "mean_sentence_chrf_pp_delta_minimum": 0.25,
                "chrf_pp_paired_90pct_lower_minimum": -0.25,
                "sacrebleu_corpus_regression_maximum": 0.1,
                "comet22_mean_delta_minimum_for_signal": 0.002,
                "comet22_paired_90pct_lower_minimum": -0.005,
                "independent_judge_mean_delta_minimum_for_signal": 0.1,
                "independent_judge_paired_90pct_lower_minimum": -0.25,
                "minimum_improvement_signals": 2,
                "maximum_domain_chrf_pp_regression": 0.5,
                "maximum_direct_chrf_pp_regression": 0.5,
            },
            "safety": {
                "new_union_critical_errors_maximum": 0,
                "new_negation_errors_maximum": 0,
                "new_number_date_unit_placeholder_errors_maximum": 0,
                "new_judge_critical_errors_maximum": 0,
                "new_repetition_or_nontermination_failures_maximum": 0,
            },
            "runtime": {
                "warm_segment_p95_seconds_maximum_each_direction": 0.175,
                "peak_resident_bytes_maximum": 250_000_000,
            },
            "packaging": {
                "preferred_two_direction_bundle_maximum_bytes": 150_000_000,
                "hard_two_direction_bundle_maximum_bytes": 500_000_000,
                "complete_distribution_provenance_required": True,
                "public_hugging_face_only_after_all_gates": True,
            },
        },
        "implementation": {
            path.name: record(path, root) for path in implementation_paths
        },
        "stop_rule": (
            "Stop before q4 conversion and protected evaluation unless one "
            "checkpoint passes every frozen internal quality, uncertainty, "
            "negative-space, and zero-new-safety-failure gate. Stop before app "
            "integration, fallback changes, bundle replacement, release, or "
            "public upload on any protected quality, safety, runtime, size, or "
            "distribution-provenance failure."
        ),
        "quantization_authorized": False,
        "protected_evaluation_authorized": False,
        "bundle_creation_authorized": False,
        "app_change_authorized": False,
        "promotion_authorized": False,
        "public_upload_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": display_path(args.output, root),
                "sha256": sha256(args.output),
                "status": result["status"],
                "train_rows": result["dataset"]["counts"]["train"],
                "valid_rows": result["dataset"]["counts"]["valid"],
                "negative_train_pairs": result["negative_dataset"]["counts"][
                    "train_pairs"
                ],
                "candidate_steps": CHECKPOINT_STEPS,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
