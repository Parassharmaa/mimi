#!/usr/bin/env python3
"""Freeze the one-arm v14 rollout-conditioned safety-repair contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EXPERIMENT = "canonical-rollout-repair-v14-ja-en"
REPOSITORY = "Mitsua/elan-mt-bt-ja-en"
REVISION = "539f80eb05306e27a166b45e4264c7fa2eb4de97"
DATASET_MANIFEST_SHA256 = (
    "1cd2e3629513f4662c6c9ffd6854d463bd638f08c8001bdb73027db0dc03d245"
)
ROLLOUT_MANIFEST_SHA256 = (
    "f93ecd7d724e37f468321cca8fbf3e9ac472ee290bb2f979daa380cb5dddd4e4"
)
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
    import hashlib

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
) -> dict[str, Any]:
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
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("rollout_directory", type=Path)
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
        or dataset_manifest.get("status") != "frozen-ready-for-rollout-mining"
        or dataset_manifest.get("direction") != "ja-en"
        or dataset_manifest.get("promotion_eligible") is not False
        or dataset_manifest.get("private_reasoning_traces_used") is not False
        or dataset_manifest.get("free_form_synthetic_translations_used") is not False
        or dataset_manifest.get("counts", {}).get("train") != 7_104
        or dataset_manifest.get("counts", {}).get("valid") != 768
    ):
        raise SystemExit("v14 dataset safety state differs")
    for split, path in dataset_paths.items():
        if dataset_manifest.get("outputs", {}).get(split, {}).get("sha256") != sha256(
            path
        ):
            raise SystemExit(f"v14 dataset does not authenticate {split}")
    provenance = dataset_manifest.get(
        "distribution_provenance",
        {},
    )
    if (
        provenance.get("all_rows_have_source_license") is not True
        or provenance.get("all_rows_have_source_provenance") is not True
        or dataset_manifest.get("decontamination", {}).get(
            "all_v10_and_v12_sources_excluded_from_fresh_validation"
        )
        is not True
        or dataset_manifest.get("decontamination", {}).get("protected_hits_in_outputs")
        != 0
    ):
        raise SystemExit("v14 dataset provenance or contamination state differs")

    rollout_manifest_path = args.rollout_directory / "manifest.json"
    rollout_manifest = load_json(rollout_manifest_path)
    rollout_paths = {
        key: args.rollout_directory / f"{key}.jsonl"
        for key in ("rollouts", "hard", "recovery")
    }
    if (
        sha256(rollout_manifest_path) != ROLLOUT_MANIFEST_SHA256
        or rollout_manifest.get("experiment") != EXPERIMENT
        or rollout_manifest.get("status") != "rollout-mining-complete"
        or rollout_manifest.get("direction") != "ja-en"
        or rollout_manifest.get("rollout_strings_are_positive_targets") is not False
        or rollout_manifest.get("recovery_target_is_only_eos") is not True
        or rollout_manifest.get("free_form_synthetic_translations_used_as_targets")
        is not False
        or rollout_manifest.get("private_reasoning_traces_used") is not False
        or rollout_manifest.get("counts", {}).get("rollouts") != 7_104
        or rollout_manifest.get("counts", {}).get("hard") != 2_048
        or rollout_manifest.get("counts", {}).get("recovery") != 74
        or rollout_manifest.get("counts", {})
        .get("recovery_reasons", {})
        .get("third-contiguous-phrase-repetition")
        != 74
    ):
        raise SystemExit("v14 rollout evidence safety state differs")
    for key, path in rollout_paths.items():
        if rollout_manifest.get("outputs", {}).get(key, {}).get("sha256") != sha256(
            path
        ):
            raise SystemExit(f"v14 rollout manifest does not authenticate {key}")
    if (
        rollout_manifest.get("dataset", {}).get("manifest", {}).get("sha256")
        != DATASET_MANIFEST_SHA256
        or rollout_manifest.get("checkpoint", {}).get("model", {}).get("sha256")
        != INITIAL_MODEL_SHA256
    ):
        raise SystemExit("v14 rollout source dataset or checkpoint differs")

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
        root / "scripts/translation/build_canonical_rollout_repair_v14_dataset.py",
        root / "scripts/translation/mine_canonical_rollout_repair_v14.py",
        root / "scripts/translation/train_canonical_rollout_repair_v14.py",
        root / "scripts/translation/evaluate_canonical_rollout_repair_v14.py",
        root / "scripts/translation/evaluate_canonical_sequence_v10_internal.py",
        root / "scripts/translation/train_marian_distillation.py",
        root / "scripts/translation/audit_translation_structures.py",
        root / "scripts/translation/typed_critical_token_policy.py",
    ]
    training = {
        "seed": 20260808,
        "batch_size": 4,
        "recovery_batch_size": 4,
        "gradient_accumulation": 8,
        "effective_batch_size": 32,
        "evaluation_batch_size": 16,
        "max_steps": 50,
        "learning_rate": 0.0000002,
        "weight_decay": 0.01,
        "warmup_steps": 5,
        "evaluation_steps": 25,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "max_source_tokens": 192,
        "max_target_tokens": 192,
        "gradient_checkpointing": False,
        "scheduled_sampling_probability": 0.20,
        "scheduled_sampling_weight": 0.25,
        "recovery_unlikelihood_weight": 0.25,
        "recovery_ranking_weight": 0.50,
        "recovery_ranking_target_margin": 1.0,
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
            "starting from rejected v12 step 50, mild scheduled sampling "
            "on free-running hard rows plus explicit EOS recovery under "
            "74 actual repeated prefixes can retain the fresh legal "
            "quality gain while removing exposure-driven loops"
        ),
        "direction": "ja-en",
        "capacity_change": False,
        "moe_added": False,
        "decoder_or_bundle_size_change_during_training": False,
        "dataset": {
            "directory": display_path(args.dataset_directory, root),
            "manifest": record(dataset_manifest_path, root),
            "train": record(dataset_paths["train"], root),
            "hard_train": record(rollout_paths["hard"], root),
            "valid": record(dataset_paths["valid"], root),
            "attribution": record(
                args.dataset_directory / "attribution.jsonl",
                root,
            ),
            "counts": dataset_manifest["counts"],
            "effective_licenses": dataset_manifest["effective_licenses"],
            "distribution_provenance": provenance,
            "private_reasoning_traces_used": False,
            "free_form_synthetic_translations_used_as_targets": False,
        },
        "rollout_dataset": {
            "directory": display_path(args.rollout_directory, root),
            "manifest": record(rollout_manifest_path, root),
            "rollouts": record(rollout_paths["rollouts"], root),
            "hard": record(rollout_paths["hard"], root),
            "recovery": record(rollout_paths["recovery"], root),
            "counts": rollout_manifest["counts"],
            "rollout_strings_are_positive_targets": False,
            "recovery_target_is_only_eos": True,
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
            "source_experiment": ("rejected v12 step 50; research initialization only"),
        },
        "preservation_checkpoint": {
            "path": display_path(
                args.preservation_checkpoint,
                root,
            ),
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
                "fresh 768-row Japanese-law test-split suite, excluding "
                "every v10/v12 train and validation source and screening "
                "both source and target against ten protected suites"
            ),
            "selection_uses_protected_outputs": False,
            "candidate_steps": CHECKPOINT_STEPS,
            "step_zero_is_diagnostic_only": True,
            "baseline": "frozen safe parent",
            "requirements": {
                "mean_sentence_chrf_pp_delta_minimum": 0.25,
                "corpus_chrf_pp_delta_minimum": 0.25,
                "long_legal_chrf_pp_delta_minimum": 0.20,
                "worst_stratum_chrf_pp_delta_minimum": -0.50,
                "paired_chrf_pp_90pct_lower_minimum": -0.25,
                "recovery_rejected_probability_delta_maximum": 0.0,
                "recovery_preference_accuracy_minimum": 0.90,
                "new_repetition_or_generation_limit_failures_maximum": 0,
            },
            "detector_policy": (
                "new exact, typed, or negation-detector cases form a "
                "semantic-audit queue and do not by themselves prove a "
                "semantic failure"
            ),
            "dual_semantic_audit_required_before_internal_pass": True,
            "required_exact_semantic_judges": [
                "claude-sonnet-5",
                "claude-opus-5",
            ],
            "semantic_audit_policy": (
                "compare selected candidate with the safe parent and "
                "licensed reference on every new detector-disagreement "
                "case; fail closed on any critical-error disagreement; "
                "zero new fail-closed semantic failures required"
            ),
            "ordering": (
                "pre-semantic eligible checkpoints only; highest mean "
                "sentence chrF++, then long-legal chrF++, then recovery "
                "preference accuracy"
            ),
            "stop_if_no_checkpoint_is_pre_semantic_eligible": True,
            "no_q4_or_protected_evaluation_before_semantic_audit": True,
        },
        "implementation": {
            path.name: record(path, root) for path in implementation_paths
        },
        "v12_rejection_final": True,
        "v13_is_future_evaluator_calibration_only": True,
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
        json.dumps(
            contract,
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
                "contract": display_path(args.output, root),
                "contract_sha256": sha256(args.output),
                "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
                "rollout_manifest_sha256": ROLLOUT_MANIFEST_SHA256,
                "training": training,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
