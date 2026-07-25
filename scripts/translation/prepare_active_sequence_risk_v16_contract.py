#!/usr/bin/env python3
"""Freeze the one-arm V16 active sequence-risk training contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from diagnose_active_sequence_risk_v16 import display_path, load_json, record
from train_marian_distillation import checkpoint_identity, sha256

EXPERIMENT = "active-sequence-risk-v16-ja-en"
DATASET_MANIFEST_SHA256 = (
    "21a3c7e23c190b28bb6d2ded323f3bd1bbde93e8fda44216d44c5be466b25902"
)
DIAGNOSTIC_RESULT_SHA256 = (
    "0272e49d4a6ebd9d87df8b51099beb354510c26f277b4b29b29d9ab98d98978d"
)
PARENT_MODEL_SHA256 = "8e7f7eff76d74b343884fe9a170b6dbad55d42f20ac5f526b6e8ec71e6c94f71"
PARENT_MANIFEST_SHA256 = (
    "0d195dc163250a9fa9312fb7ad8ba3341ab65167b90926d46f0a65d76047bd38"
)
V12_VALID_SHA256 = "8594d49aad5ab4696d5c20a9ffe142729d6381a00c9532268e42e9b70aca4ba6"
V14_VALID_SHA256 = "3228c256ab1b4b8f878edb52f67d977c2f454bc67a6c3e04e07926a5aba4d040"
V15_VALID_SHA256 = "b074c6ca26697e94c42f75ba783b86690b7d6facc81cad94ae35de5561ad798b"
REPOSITORY = "Mitsua/elan-mt-bt-ja-en"
REVISION = "539f80eb05306e27a166b45e4264c7fa2eb4de97"
CHECKPOINT_STEPS = [25, 50]


def authenticated_output(
    directory: Path,
    manifest: dict[str, Any],
    name: str,
) -> Path:
    path = directory / f"{name}.jsonl"
    if manifest.get("outputs", {}).get(name, {}).get("sha256") != sha256(path):
        raise SystemExit(f"V16 dataset does not authenticate {name}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("diagnostic_result", type=Path)
    parser.add_argument("safe_parent", type=Path)
    parser.add_argument("v12_validation", type=Path)
    parser.add_argument("v14_validation", type=Path)
    parser.add_argument("v15_validation", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")

    root = Path(__file__).resolve().parents[2]
    dataset_manifest_path = args.dataset_directory / "manifest.json"
    dataset_manifest = load_json(dataset_manifest_path)
    if (
        sha256(dataset_manifest_path) != DATASET_MANIFEST_SHA256
        or dataset_manifest.get("experiment") != EXPERIMENT
        or dataset_manifest.get("status") != "frozen-ready-for-pcgrad-contract"
        or dataset_manifest.get("direction") != "ja-en"
        or dataset_manifest.get("promotion_eligible") is not False
        or dataset_manifest.get("counts", {}).get("train") != 7_104
        or dataset_manifest.get("counts", {}).get("active") != 677
        or dataset_manifest.get("counts", {}).get("negative_preferred") != 169
        or dataset_manifest.get("counts", {}).get("valid") != 768
        or dataset_manifest.get("distribution_provenance", {}).get(
            "all_positive_targets_are_licensed_human_references"
        )
        is not True
        or dataset_manifest.get("distribution_provenance", {}).get(
            "negative_strings_are_positive_targets"
        )
        is not False
        or dataset_manifest.get("decontamination", {}).get(
            "all_v10_v12_v14_v15_sources_excluded_from_fresh_validation"
        )
        is not True
        or dataset_manifest.get("decontamination", {}).get("protected_hits_in_outputs")
        != 0
    ):
        raise SystemExit("V16 dataset identity or safety state differs")
    active_path = authenticated_output(
        args.dataset_directory,
        dataset_manifest,
        "active",
    )
    fresh_valid_path = authenticated_output(
        args.dataset_directory,
        dataset_manifest,
        "valid",
    )
    train_item = dataset_manifest["inputs"]["v15_train"]
    train_path = root / train_item["path"]
    if sha256(train_path) != train_item["sha256"]:
        raise SystemExit("V16 training data differs")

    diagnostic = load_json(args.diagnostic_result)
    if (
        sha256(args.diagnostic_result) != DIAGNOSTIC_RESULT_SHA256
        or diagnostic.get("status") != "diagnostic-complete-no-training-authorized"
        or diagnostic.get("gradient_audit", {})
        .get("cosine_summary", {})
        .get(
            "omission_vs_repetition",
            {},
        )
        .get("negative_fraction")
        != 1.0
        or diagnostic.get("gradient_audit", {})
        .get("cosine_summary", {})
        .get(
            "mle_vs_omission",
            {},
        )
        .get("negative_fraction")
        != 0.0
        or diagnostic.get("gradient_audit", {})
        .get("cosine_summary", {})
        .get(
            "mle_vs_repetition",
            {},
        )
        .get("negative_fraction")
        != 0.0
        or diagnostic.get("optimizer_step_executed") is not False
    ):
        raise SystemExit("V16 diagnostic does not justify the frozen gradient rule")

    for path, expected, label in (
        (args.v12_validation, V12_VALID_SHA256, "V12"),
        (args.v14_validation, V14_VALID_SHA256, "V14"),
        (args.v15_validation, V15_VALID_SHA256, "V15"),
    ):
        if sha256(path) != expected:
            raise SystemExit(f"{label} regression suite differs")
    if (
        sha256(args.safe_parent / "model.safetensors") != PARENT_MODEL_SHA256
        or sha256(args.safe_parent / "mimi_training_manifest.json")
        != PARENT_MANIFEST_SHA256
        or checkpoint_identity(args.safe_parent) != ("ja-en", REPOSITORY, REVISION)
    ):
        raise SystemExit("V16 safe-parent identity differs")

    implementation_paths = [
        Path(__file__).resolve(),
        root / "scripts/translation/build_active_sequence_risk_v16_dataset.py",
        root / "scripts/translation/diagnose_active_sequence_risk_v16.py",
        root / "scripts/translation/train_active_sequence_risk_v16.py",
        root / "scripts/translation/evaluate_active_sequence_risk_v16.py",
        root / "scripts/translation/evaluate_canonical_constrained_recovery_v15.py",
        root / "scripts/translation/evaluate_canonical_sequence_v10_internal.py",
        root / "scripts/translation/train_marian_distillation.py",
        root / "scripts/translation/audit_translation_structures.py",
        root / "scripts/translation/typed_critical_token_policy.py",
    ]
    training = {
        "seed": 20260823,
        "batch_size": 4,
        "risk_batch_size": 4,
        "evaluation_batch_size": 16,
        "max_steps": 50,
        "learning_rate": 0.0000003,
        "weight_decay": 0.01,
        "warmup_steps": 5,
        "evaluation_steps": 25,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "max_source_tokens": 192,
        "max_target_tokens": 192,
        "sequence_target_margin": 0.25,
        "omission_weight": 0.35,
        "repetition_weight": 0.35,
        "frozen_parent_kl_weight": 0.10,
        "l2_to_parent_weight": 0.00001,
        "maximum_gradient_norm": 1.0,
        "pcgrad_epsilon": 0.000000000001,
        "gradient_rule": (
            "unprojected-mle-plus-symmetric-pcgrad-between-omission-and-repetition"
        ),
        "gradient_accumulation": 1,
        "one_arm_only": True,
        "post_result_hyperparameter_changes_forbidden": True,
    }
    checkpoint = {
        "path": display_path(args.safe_parent, root),
        "repository": REPOSITORY,
        "revision": REVISION,
        "license": "CC-BY-SA-4.0",
        "model": record(args.safe_parent / "model.safetensors", root),
        "training_manifest": record(
            args.safe_parent / "mimi_training_manifest.json",
            root,
        ),
    }
    contract = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": "preregistered-ready-for-one-arm-training",
        "hypothesis": (
            "starting from the distributable safe parent, licensed-reference "
            "MLE plus active full-sequence omission and repetition ranking can "
            "improve held-out legal translation while symmetric PCGrad removes "
            "the measured conflict between the two safety objectives"
        ),
        "direction": "ja-en",
        "capacity_change": False,
        "moe_added": False,
        "scheduled_sampling_used": False,
        "decoder_or_bundle_size_change_during_training": False,
        "diagnostic": {
            "result": record(args.diagnostic_result, root),
            "active_pairs": 677,
            "negative_preferred_pairs": 169,
            "omission_repetition_gradient_conflict_fraction": 1.0,
            "mle_is_projected": False,
            "optimizer_step_executed_during_diagnostic": False,
        },
        "dataset": {
            "directory": display_path(args.dataset_directory, root),
            "manifest": record(dataset_manifest_path, root),
            "train": record(train_path, root),
            "fresh_valid": record(fresh_valid_path, root),
            "active": record(active_path, root),
            "v12_regression": record(args.v12_validation, root),
            "v14_regression": record(args.v14_validation, root),
            "v15_regression": record(args.v15_validation, root),
            "attribution": record(
                args.dataset_directory / "attribution.jsonl",
                root,
            ),
            "counts": {
                "train": 7_104,
                "fresh_valid": 768,
                "active": 677,
                "active_omission": 228,
                "active_repetition": 449,
                "v12_regression": 1_536,
                "v14_regression": 768,
                "v15_regression": 768,
            },
            "fresh_valid_strata": dataset_manifest["counts"]["valid_strata"],
            "exhausted_fresh_strata": ["omission-risk", "repetition-risk"],
            "effective_licenses": dataset_manifest["effective_licenses"],
            "distribution_provenance": dataset_manifest["distribution_provenance"],
            "private_reasoning_traces_used": False,
            "free_form_synthetic_translations_used_as_targets": False,
        },
        "initial_checkpoint": {
            **checkpoint,
            "source_experiment": "distributable safe parent",
        },
        "preservation_checkpoint": checkpoint,
        "training": training,
        "pcgrad_formula": {
            "scope": "omission and repetition gradients only",
            "condition": "project if original dot product is negative",
            "omission": "g_o - dot(g_o,g_r)/norm2(g_r) * g_r",
            "repetition": "g_r - dot(g_r,g_o)/norm2(g_o) * g_o",
            "projection_inputs": "the two original unprojected safety gradients",
            "combined": "g_mle + 0.35*g_o_projected + 0.35*g_r_projected",
            "mle_projection_forbidden": True,
            "deterministic": True,
        },
        "internal_selection": {
            "selection_uses_protected_outputs": False,
            "candidate_steps": CHECKPOINT_STEPS,
            "step_zero_is_diagnostic_only": True,
            "baseline": "frozen safe parent",
            "suites": {
                "fresh_v16": (
                    "768 source-unique legal examples disjoint from all "
                    "V10/V12/V14/V15 development sources and protected suites"
                ),
                "v12_regression": "frozen 1,536-row V12 validation suite",
                "v14_regression": "frozen 768-row V14 validation suite",
                "v15_regression": "frozen 768-row V15 validation suite",
            },
            "requirements": {
                "fresh_mean_sentence_chrf_pp_delta_minimum": 0.20,
                "fresh_corpus_chrf_pp_delta_minimum": 0.20,
                "fresh_long_legal_chrf_pp_delta_minimum": 0.15,
                "fresh_worst_stratum_chrf_pp_delta_minimum": -0.50,
                "fresh_paired_chrf_pp_90pct_lower_minimum": -0.20,
                "v12_mean_sentence_chrf_pp_delta_minimum": 0.15,
                "v12_worst_stratum_chrf_pp_delta_minimum": -0.50,
                "v14_mean_sentence_chrf_pp_delta_minimum": 0.15,
                "v14_worst_stratum_chrf_pp_delta_minimum": -0.50,
                "v14_omission_risk_chrf_pp_delta_minimum": 0.0,
                "v15_mean_sentence_chrf_pp_delta_minimum": 0.15,
                "v15_worst_stratum_chrf_pp_delta_minimum": -0.50,
                "v15_omission_risk_chrf_pp_delta_minimum": 0.0,
                "active_all_preference_accuracy_delta_minimum": 0.025,
                "active_omission_preference_accuracy_delta_minimum": 0.0,
                "active_repetition_preference_accuracy_delta_minimum": 0.0,
                "active_omission_mean_margin_delta_minimum": 0.02,
                "active_repetition_mean_margin_delta_minimum": 0.02,
                "active_fraction_delta_maximum": -0.02,
                "new_repetition_or_generation_limit_failures_maximum_per_suite": 0,
            },
            "detector_policy": (
                "new exact, typed, or negation-detector cases from any suite "
                "form a semantic-audit queue; detectors do not adjudicate "
                "semantic correctness"
            ),
            "dual_semantic_audit_required_before_internal_pass": True,
            "required_exact_semantic_judges": [
                "claude-sonnet-5",
                "claude-opus-5",
            ],
            "ordering": (
                "pre-semantic eligible checkpoints only; highest fresh mean "
                "sentence chrF++, then V14 omission-risk chrF++, then active "
                "preference improvement"
            ),
            "stop_if_no_checkpoint_is_pre_semantic_eligible": True,
            "no_q4_comet_or_protected_evaluation_before_semantic_audit": True,
        },
        "implementation": {
            path.name: record(path, root) for path in implementation_paths
        },
        "v12_v14_v15_rejections_final": True,
        "exact_q4_conversion_authorized": False,
        "protected_evaluation_authorized": False,
        "comet_authorized": False,
        "app_change_authorized": False,
        "bundle_replacement_authorized": False,
        "public_upload_authorized": False,
        "human_reviewer_required": False,
        "private_reasoning_traces_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "contract": display_path(args.output, root),
                "contract_sha256": sha256(args.output),
                "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
                "training": training,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
