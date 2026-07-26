#!/usr/bin/env python3
"""Freeze the one-arm v15 constrained-recovery experiment contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

EXPERIMENT = "canonical-constrained-recovery-v15-ja-en"
REPOSITORY = "Mitsua/elan-mt-bt-ja-en"
REVISION = "539f80eb05306e27a166b45e4264c7fa2eb4de97"
DATASET_MANIFEST_SHA256 = (
    "9b412e0a7d49234ab374f4e47fc71e0f70e9cb432af6f792d26e7ce56910c523"
)
EXAMPLES_MANIFEST_SHA256 = (
    "60556bba503a41b2d2b4daf95d2fe7434657fa9be0df590ec311b5a2b6e5efdb"
)
V12_VALID_SHA256 = "8594d49aad5ab4696d5c20a9ffe142729d6381a00c9532268e42e9b70aca4ba6"
V14_VALID_SHA256 = "3228c256ab1b4b8f878edb52f67d977c2f454bc67a6c3e04e07926a5aba4d040"
INITIAL_MODEL_SHA256 = (
    "cf67fb44a4e9a0991c95b5e87578a4427610cc8e4110fbd3bb03909356600f2b"
)
INITIAL_MANIFEST_SHA256 = (
    "68de542ff5476800aab88b075cac05b5e3dd3da21330d15d1d7910a587a9c533"
)
PARENT_MODEL_SHA256 = "8e7f7eff76d74b343884fe9a170b6dbad55d42f20ac5f526b6e8ec71e6c94f71"
PARENT_MANIFEST_SHA256 = (
    "0d195dc163250a9fa9312fb7ad8ba3341ab65167b90926d46f0a65d76047bd38"
)
CHECKPOINT_STEPS = [25, 50]


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
    expected_experiment: str | None,
    expected_step: int | None,
) -> None:
    model_path = checkpoint / "model.safetensors"
    manifest_path = checkpoint / "mimi_training_manifest.json"
    if (
        sha256(model_path) != expected_model_sha256
        or sha256(manifest_path) != expected_manifest_sha256
    ):
        raise SystemExit(f"bound checkpoint bytes differ: {checkpoint}")
    manifest = load_json(manifest_path)
    if (
        manifest.get("direction") != "ja-en"
        or manifest.get("student_repository") != REPOSITORY
        or manifest.get("student_revision") != REVISION
        or manifest.get("license") != "CC-BY-SA-4.0"
        or (
            expected_experiment is not None
            and manifest.get("experiment") != expected_experiment
        )
        or (
            expected_step is not None
            and manifest.get("checkpoint_step") != expected_step
        )
    ):
        raise SystemExit(f"bound checkpoint identity differs: {checkpoint}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("examples_directory", type=Path)
    parser.add_argument("v12_validation", type=Path)
    parser.add_argument("v14_validation", type=Path)
    parser.add_argument("initial_checkpoint", type=Path)
    parser.add_argument("preservation_checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite contract: {args.output}")

    root = Path(__file__).resolve().parents[2]
    dataset_manifest_path = args.dataset_directory / "manifest.json"
    dataset_manifest = load_json(dataset_manifest_path)
    dataset_paths = {
        split: args.dataset_directory / f"{split}.jsonl" for split in ("train", "valid")
    }
    if (
        sha256(dataset_manifest_path) != DATASET_MANIFEST_SHA256
        or dataset_manifest.get("experiment") != EXPERIMENT
        or dataset_manifest.get("status")
        != "frozen-ready-for-contrastive-example-building"
        or dataset_manifest.get("direction") != "ja-en"
        or dataset_manifest.get("promotion_eligible") is not False
        or dataset_manifest.get("counts", {}).get("train") != 7_104
        or dataset_manifest.get("counts", {}).get("valid") != 768
        or dataset_manifest.get("distribution_provenance", {}).get(
            "all_positive_targets_are_licensed_human_references"
        )
        is not True
        or dataset_manifest.get("distribution_provenance", {}).get(
            "generated_strings_are_positive_targets"
        )
        is not False
        or dataset_manifest.get("decontamination", {}).get(
            "all_v10_v12_v14_sources_excluded_from_fresh_validation"
        )
        is not True
        or dataset_manifest.get("decontamination", {}).get("protected_hits_in_outputs")
        != 0
    ):
        raise SystemExit("v15 dataset safety state differs")
    for split, path in dataset_paths.items():
        if dataset_manifest.get("outputs", {}).get(split, {}).get("sha256") != sha256(
            path
        ):
            raise SystemExit(f"v15 dataset does not authenticate {split}")

    examples_manifest_path = args.examples_directory / "manifest.json"
    examples_manifest = load_json(examples_manifest_path)
    example_paths = {
        name: args.examples_directory / f"{name}.jsonl"
        for name in ("recovery", "omission")
    }
    if (
        sha256(examples_manifest_path) != EXAMPLES_MANIFEST_SHA256
        or examples_manifest.get("experiment") != EXPERIMENT
        or examples_manifest.get("status") != "contrastive-examples-frozen"
        or examples_manifest.get("direction") != "ja-en"
        or examples_manifest.get("promotion_eligible") is not False
        or examples_manifest.get("counts", {}).get("recovery") != 2_048
        or examples_manifest.get("counts", {}).get("omission") != 2_048
        or examples_manifest.get("generated_strings_are_positive_targets") is not False
        or examples_manifest.get("private_reasoning_traces_used") is not False
        or examples_manifest.get("distribution_provenance", {}).get(
            "all_positive_targets_are_licensed_human_references"
        )
        is not True
    ):
        raise SystemExit("v15 contrastive-example safety state differs")
    for name, path in example_paths.items():
        if examples_manifest.get("outputs", {}).get(name, {}).get("sha256") != sha256(
            path
        ):
            raise SystemExit(f"v15 example manifest does not authenticate {name}")

    if sha256(args.v12_validation) != V12_VALID_SHA256:
        raise SystemExit("v12 regression suite differs")
    if sha256(args.v14_validation) != V14_VALID_SHA256:
        raise SystemExit("v14 regression suite differs")
    validate_checkpoint(
        args.initial_checkpoint,
        expected_model_sha256=INITIAL_MODEL_SHA256,
        expected_manifest_sha256=INITIAL_MANIFEST_SHA256,
        expected_experiment="canonical-safety-repair-v12-ja-en",
        expected_step=50,
    )
    validate_checkpoint(
        args.preservation_checkpoint,
        expected_model_sha256=PARENT_MODEL_SHA256,
        expected_manifest_sha256=PARENT_MANIFEST_SHA256,
        expected_experiment=None,
        expected_step=None,
    )

    implementation_paths = [
        Path(__file__).resolve(),
        root
        / "scripts/translation/build_canonical_constrained_recovery_v15_dataset.py",
        root
        / "scripts/translation/build_canonical_constrained_recovery_v15_examples.py",
        root / "scripts/translation/train_canonical_constrained_recovery_v15.py",
        root / "scripts/translation/evaluate_canonical_constrained_recovery_v15.py",
        root / "scripts/translation/evaluate_canonical_sequence_v10_internal.py",
        root / "scripts/translation/train_marian_distillation.py",
        root / "scripts/translation/audit_translation_structures.py",
        root / "scripts/translation/typed_critical_token_policy.py",
    ]
    training = {
        "seed": 20260818,
        "batch_size": 4,
        "contrast_batch_size": 4,
        "gradient_accumulation": 8,
        "effective_batch_size": 32,
        "evaluation_batch_size": 16,
        "max_steps": 50,
        "learning_rate": 0.0000003,
        "weight_decay": 0.01,
        "warmup_steps": 5,
        "evaluation_steps": 25,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "max_source_tokens": 192,
        "max_target_tokens": 192,
        "gradient_checkpointing": False,
        "recovery_weight": 0.25,
        "constrained_recovery_weight": 0.10,
        "omission_weight": 0.50,
        "recovery_target_margin": 0.10,
        "clean_over_recovery_target_margin": 0.01,
        "omission_target_margin": 0.50,
        "frozen_parent_kl_weight": 0.10,
        "l2_to_parent_weight": 0.00001,
        "one_arm_only": True,
        "post_result_hyperparameter_changes_forbidden": True,
    }
    contract = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": "preregistered-ready-for-one-arm-training",
        "hypothesis": (
            "starting from rejected v12 step 50, licensed-reference MLE plus "
            "clean-over-perturbed recovery ordering and explicit token-span "
            "omission contrasts can retain fresh legal gains without making "
            "a recovery path outrank the clean reference path"
        ),
        "direction": "ja-en",
        "capacity_change": False,
        "moe_added": False,
        "scheduled_sampling_used": False,
        "unconditional_eos_recovery_used": False,
        "decoder_or_bundle_size_change_during_training": False,
        "dataset": {
            "directory": display_path(args.dataset_directory, root),
            "manifest": record(dataset_manifest_path, root),
            "train": record(dataset_paths["train"], root),
            "fresh_valid": record(dataset_paths["valid"], root),
            "v12_regression": record(args.v12_validation, root),
            "v14_regression": record(args.v14_validation, root),
            "attribution": record(
                args.dataset_directory / "attribution.jsonl",
                root,
            ),
            "counts": {
                "train": 7_104,
                "fresh_valid": 768,
                "v12_regression": 1_536,
                "v14_regression": 768,
            },
            "effective_licenses": dataset_manifest["effective_licenses"],
            "distribution_provenance": dataset_manifest["distribution_provenance"],
            "private_reasoning_traces_used": False,
            "free_form_synthetic_translations_used_as_targets": False,
        },
        "contrastive_examples": {
            "directory": display_path(args.examples_directory, root),
            "manifest": record(examples_manifest_path, root),
            "recovery": record(example_paths["recovery"], root),
            "omission": record(example_paths["omission"], root),
            "counts": {
                "recovery": 2_048,
                "omission": 2_048,
            },
            "recovery_roles": examples_manifest["counts"]["recovery_roles"],
            "generated_strings_are_positive_targets": False,
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
            "source_experiment": "rejected v12 step 50; research initialization only",
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
            "selection_uses_protected_outputs": False,
            "candidate_steps": CHECKPOINT_STEPS,
            "step_zero_is_diagnostic_only": True,
            "baseline": "frozen safe parent",
            "suites": {
                "fresh_v15": (
                    "768 source-unique held-out legal examples, disjoint from "
                    "v10, v12, and v14 development rows and screened on both "
                    "sides against ten protected suites"
                ),
                "v12_regression": "frozen 1,536-row v12 validation suite",
                "v14_regression": "frozen 768-row v14 validation suite",
            },
            "requirements": {
                "fresh_mean_sentence_chrf_pp_delta_minimum": 0.25,
                "fresh_corpus_chrf_pp_delta_minimum": 0.25,
                "fresh_long_legal_chrf_pp_delta_minimum": 0.20,
                "fresh_worst_stratum_chrf_pp_delta_minimum": -0.50,
                "fresh_paired_chrf_pp_90pct_lower_minimum": -0.25,
                "v12_mean_sentence_chrf_pp_delta_minimum": 0.20,
                "v12_worst_stratum_chrf_pp_delta_minimum": -0.50,
                "v14_mean_sentence_chrf_pp_delta_minimum": 0.20,
                "v14_worst_stratum_chrf_pp_delta_minimum": -0.50,
                "v14_omission_risk_chrf_pp_delta_minimum": -0.50,
                "recovery_preference_accuracy_delta_minimum": 0.05,
                "omission_preference_accuracy_delta_minimum": 0.05,
                "clean_over_recovery_accuracy_delta_minimum": -0.02,
                "recovery_rejected_probability_delta_maximum": 0.0,
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
                "sentence chrF++, then v14 omission-risk chrF++, then recovery "
                "preference improvement"
            ),
            "stop_if_no_checkpoint_is_pre_semantic_eligible": True,
            "no_q4_comet_or_protected_evaluation_before_semantic_audit": True,
        },
        "implementation": {
            path.name: record(path, root) for path in implementation_paths
        },
        "v12_and_v14_rejections_final": True,
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
                "examples_manifest_sha256": EXAMPLES_MANIFEST_SHA256,
                "training": training,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
