#!/usr/bin/env python3
"""Freeze the one-arm canonical sequence v10 training and selection contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPERIMENT = "canonical-sequence-v10-ja-en-error-stratified"
PARENT_REPOSITORY = "Mitsua/elan-mt-bt-ja-en"
PARENT_REVISION = "539f80eb05306e27a166b45e4264c7fa2eb4de97"
PARENT_MODEL_SHA256 = (
    "8e7f7eff76d74b343884fe9a170b6dbad55d42f20ac5f526b6e8ec71e6c94f71"
)
PARENT_MANIFEST_SHA256 = (
    "0d195dc163250a9fa9312fb7ad8ba3341ab65167b90926d46f0a65d76047bd38"
)
CHECKPOINT_STEPS = [125, 250]
PRESERVATION_ORIGINS = [
    "finalized-japanese-law-translation",
    "human-alt-parallel",
    "human-kftt-replay",
    "human-tatoeba-bidirectional-agreement-filtered",
    "licensed-human-reference-anchor",
    "mimi-shipped-ui-pair",
]


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


def record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("parent_checkpoint", type=Path)
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
    train_path = args.dataset_directory / "train.jsonl"
    valid_path = args.dataset_directory / "valid.jsonl"
    if (
        dataset_manifest.get("experiment") != EXPERIMENT
        or dataset_manifest.get("status")
        != "frozen-ready-for-preregistered-training"
        or dataset_manifest.get("direction") != "ja-en"
        or dataset_manifest.get("private_reasoning_traces_used") is not False
        or dataset_manifest.get("promotion_eligible") is not False
        or dataset_manifest.get("counts", {}).get("train") != 8_192
        or dataset_manifest.get("counts", {}).get("valid") != 1_024
    ):
        raise SystemExit("v10 dataset manifest does not satisfy the frozen contract")
    for split, path in (("train", train_path), ("valid", valid_path)):
        if (
            dataset_manifest.get("outputs", {}).get(split, {}).get("sha256")
            != sha256(path)
        ):
            raise SystemExit(f"v10 dataset manifest does not authenticate {split}")
    synthetic = dataset_manifest.get("synthetic_policy", {})
    if (
        synthetic.get("teacher_model") != "gpt-5.6-sol-via-codex-cli"
        or synthetic.get("admission_judges")
        != ["claude-opus-5", "claude-sonnet-5"]
        or synthetic.get("unanimous_pareto_preference_required") is not True
        or synthetic.get("absolute_quality_required") is not True
        or synthetic.get("reasoning_traces_requested_or_retained") is not False
        or float(synthetic.get("synthetic_train_fraction", 1)) > 0.15
    ):
        raise SystemExit("v10 synthetic-data policy differs")
    distribution = dataset_manifest.get("distribution_provenance", {})
    if (
        distribution.get("all_rows_have_source_license") is not True
        or distribution.get("all_rows_have_source_provenance") is not True
    ):
        raise SystemExit("v10 distribution provenance is incomplete")

    parent_model = args.parent_checkpoint / "model.safetensors"
    parent_manifest = args.parent_checkpoint / "mimi_training_manifest.json"
    if sha256(parent_model) != PARENT_MODEL_SHA256:
        raise SystemExit("v10 parent model hash differs")
    if sha256(parent_manifest) != PARENT_MANIFEST_SHA256:
        raise SystemExit("v10 parent training manifest hash differs")
    parent_metadata = load_json(parent_manifest)
    if (
        parent_metadata.get("direction") != "ja-en"
        or parent_metadata.get("student_repository") != PARENT_REPOSITORY
        or parent_metadata.get("student_revision") != PARENT_REVISION
        or parent_metadata.get("license") != "CC-BY-SA-4.0"
    ):
        raise SystemExit("v10 parent identity or license differs")

    implementation_paths = [
        root / "scripts/translation/build_canonical_sequence_v10_dataset.py",
        root / "scripts/translation/train_marian_distillation.py",
        root / "scripts/translation/prepare_elanmt_mlx.py",
        root / "scripts/translation/run_mlx_marian_benchmark.py",
        root / "scripts/translation/marian_mlx.py",
        root / "scripts/translation/audit_translation_structures.py",
        root / "scripts/translation/typed_critical_token_policy.py",
    ]
    protected_records = [record(path) for path in args.protected_suite]
    manifest_protected = dataset_manifest.get("decontamination", {}).get(
        "protected_suites"
    )
    if not isinstance(manifest_protected, list) or {
        item["sha256"] for item in manifest_protected
    } != {item["sha256"] for item in protected_records}:
        raise SystemExit("v10 protected-suite inventory differs from the dataset")

    training = {
        "seed": 20260802,
        "batch_size": 8,
        "gradient_accumulation": 4,
        "effective_batch_size": 32,
        "max_steps": 250,
        "learning_rate": 0.000002,
        "weight_decay": 0.01,
        "warmup_steps": 25,
        "evaluation_steps": 125,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "max_source_tokens": 192,
        "max_target_tokens": 192,
        "gradient_checkpointing": False,
        "frozen_parent_kl_weight": 0.10,
        "l2_to_parent_weight": 0.00001,
        "canonical_sequence_loss_weight_start": 4.0,
        "canonical_sequence_loss_weight_end": 2.0,
        "curriculum_ramp_steps": 250,
        "preservation_origins": PRESERVATION_ORIGINS,
        "initial_checkpoint": str(args.parent_checkpoint),
        "preservation_checkpoint": str(args.parent_checkpoint),
        "one_arm_only": True,
        "post_result_hyperparameter_changes_forbidden": True,
    }
    result = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": "preregistered-ready-for-one-arm-training",
        "hypothesis": (
            "generation-level supervised updates from Claude-5-approved GPT-5.6 "
            "final sequences will become usable when combined with a much larger "
            "protected-independent, error-stratified licensed replay set"
        ),
        "direction": "ja-en",
        "dataset": {
            "directory": str(args.dataset_directory),
            "manifest": record(dataset_manifest_path),
            "train": record(train_path),
            "valid": record(valid_path),
            "attribution": record(
                args.dataset_directory / "attribution.jsonl"
            ),
            "counts": dataset_manifest["counts"],
            "origins": dataset_manifest["origins"],
            "strata": dataset_manifest["strata"],
            "effective_licenses": dataset_manifest["effective_licenses"],
            "distribution_provenance": distribution,
            "synthetic_policy": synthetic,
        },
        "parent": {
            "path": str(args.parent_checkpoint),
            "repository": PARENT_REPOSITORY,
            "revision": PARENT_REVISION,
            "license": "CC-BY-SA-4.0",
            "model": record(parent_model),
            "training_manifest": record(parent_manifest),
        },
        "training": training,
        "internal_selection": {
            "suite": "v10 frozen 1,024-row source-disjoint development split",
            "selection_uses_protected_outputs": False,
            "candidate_steps": CHECKPOINT_STEPS,
            "step_zero_is_baseline_only": True,
            "requirements": {
                "mean_sentence_chrf_pp_delta_minimum": 0.25,
                "corpus_chrf_pp_delta_minimum": 0.25,
                "teacher_slice_chrf_pp_delta_minimum": 0.50,
                "long_legal_chrf_pp_delta_minimum": 0.25,
                "worst_stratum_chrf_pp_delta_minimum": -0.50,
                "new_exact_critical_failures_maximum": 0,
                "new_typed_critical_failures_maximum": 0,
                "new_negation_failures_maximum": 0,
                "new_repetition_or_generation_limit_failures_maximum": 0,
            },
            "ordering": (
                "eligible checkpoints only; highest mean sentence chrF++, "
                "then long-legal chrF++, teacher-slice chrF++, and lower loss"
            ),
            "stop_if_no_checkpoint_is_eligible": True,
        },
        "next_step_if_internal_gate_passes": {
            "exact_q4_conversion_required": True,
            "bits": 4,
            "group_size": 64,
            "protected_evaluation_required": True,
            "protected_suites": protected_records,
            "comet22_and_two_family_judge_required_if_decision_relevant": True,
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
            path.name: record(path) for path in implementation_paths
        },
        "stop_rule": (
            "Stop before q4 conversion and protected evaluation unless one "
            "checkpoint passes every frozen internal generation, slice, and "
            "safety gate. Stop before app integration, fallback changes, bundle "
            "replacement, release, or public upload on any protected quality, "
            "safety, runtime, size, or distribution-provenance failure."
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
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": sha256(args.output),
                "status": result["status"],
                "train_rows": result["dataset"]["counts"]["train"],
                "valid_rows": result["dataset"]["counts"]["valid"],
                "candidate_steps": CHECKPOINT_STEPS,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
