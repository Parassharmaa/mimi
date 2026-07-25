#!/usr/bin/env python3
"""Freeze the preservation-aware Claude-5 JA-to-EN v8 experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


EXPERIMENT = "canonical-pairwise-preference-replay-v8-ja-en"
PREFERENCE_EXPERIMENT = "canonical-pairwise-v7-ja-en-claude5"
REPLAY_EXPERIMENT = "canonical-pairwise-v8-ja-en-licensed-replay"
REQUIRED_JUDGES = {"claude-sonnet-5", "claude-opus-5"}
IMPLEMENTATION = (
    Path("scripts/translation/prepare_canonical_pairwise_v8_contract.py"),
    Path("scripts/translation/build_canonical_pairwise_v8_result.py"),
    Path("scripts/translation/build_canonical_pairwise_v8_replay_dataset.py"),
    Path("scripts/translation/train_marian_claude5_preference_replay.py"),
    Path("scripts/translation/train_marian_claude5_preference.py"),
    Path("scripts/translation/train_marian_automated_preference_full.py"),
    Path("scripts/translation/train_marian_dqo.py"),
    Path("scripts/translation/train_marian_distillation.py"),
    Path("scripts/translation/evaluate_gpt56_student_continuation.py"),
    Path("scripts/translation/audit_translation_structures.py"),
    Path("scripts/translation/typed_critical_token_policy.py"),
)
TRAINING = {
    "seed": 20260731,
    "batch_size": 4,
    "replay_batch_size": 4,
    "replay_evaluation_batch_size": 8,
    "gradient_accumulation": 4,
    "max_steps": 40,
    "learning_rate": 0.0000005,
    "weight_decay": 0.01,
    "warmup_steps": 4,
    "evaluation_steps": 10,
    "beta": 0.10,
    "chosen_sft_weight": 0.02,
    "replay_sft_weight": 0.25,
    "replay_kl_weight": 0.10,
    "l2_to_parent_weight": 0.10,
    "max_source_tokens": 192,
    "max_target_tokens": 192,
}
INTERNAL_GATE = {
    "selection": (
        "eligible checkpoints only; highest validation relative-pair accuracy, "
        "then relative margin, legal replay chrF++, and lower preference loss"
    ),
    "relative_pair_accuracy_minimum": 0.8,
    "relative_margin_minimum_exclusive": 0.0,
    "replay_token_nll_delta_maximum": 0.01,
    "replay_chrf_pp_delta_minimum": -0.10,
    "legal_replay_chrf_pp_delta_minimum": -0.10,
    "new_exact_critical_maximum": 0,
    "new_typed_critical_maximum": 0,
    "new_negation_maximum": 0,
    "new_repetition_or_generation_limit_maximum": 0,
    "checkpoint_steps": [10, 20, 30, 40],
    "step_zero_is_not_a_trained_candidate": True,
}


def sha256(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"missing contract input: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def record(path: Path) -> dict:
    return {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def dataset_record(directory: Path, filenames: tuple[str, ...]) -> dict:
    return {
        "directory": str(directory),
        "files": {
            filename: record(directory / filename) for filename in filenames
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("v7_contract", type=Path)
    parser.add_argument("v7_protected_result", type=Path)
    parser.add_argument("preference_directory", type=Path)
    parser.add_argument("replay_directory", type=Path)
    parser.add_argument("parent_checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen contract: {args.output}")

    v7_contract = load_json(args.v7_contract)
    protected = load_json(args.v7_protected_result)
    if (
        v7_contract.get("experiment")
        != "canonical-pairwise-preference-claude5-v7-ja-en"
        or v7_contract.get("status") != "preregistered-ready-for-training"
        or protected.get("status") != "protected-promotion-gate-rejected"
        or protected.get("promotion_authorized") is not False
        or protected.get("app_change_authorized") is not False
        or protected.get("public_upload_authorized") is not False
        or protected.get("contract", {}).get("sha256") != sha256(args.v7_contract)
    ):
        raise SystemExit("v7 stop evidence is invalid or has drifted")

    preference_manifest = load_json(args.preference_directory / "manifest.json")
    if (
        preference_manifest.get("experiment") != PREFERENCE_EXPERIMENT
        or preference_manifest.get("direction") != "ja-en"
        or preference_manifest.get("promotion_eligible") is not True
        or preference_manifest.get("private_reasoning_traces_used") is not False
        or set(preference_manifest.get("required_judge_model_ids", []))
        != REQUIRED_JUDGES
        or preference_manifest.get("counts", {}).get("train") != 136
        or preference_manifest.get("counts", {}).get("valid") != 33
    ):
        raise SystemExit("v8 preference input is invalid or has drifted")

    replay_manifest = load_json(args.replay_directory / "manifest.json")
    if (
        replay_manifest.get("experiment") != REPLAY_EXPERIMENT
        or replay_manifest.get("status") != "frozen-protected-screened-replay"
        or replay_manifest.get("direction") != "ja-en"
        or replay_manifest.get("promotion_eligible") is not False
        or replay_manifest.get("private_reasoning_traces_used") is not False
        or replay_manifest.get("counts", {}).get("train") != 136
        or replay_manifest.get("counts", {}).get("valid") != 128
        or replay_manifest.get("decontamination", {}).get(
            "preference_replay_source_overlap"
        )
        is not False
        or replay_manifest.get("decontamination", {}).get(
            "train_valid_source_overlap"
        )
        is not False
    ):
        raise SystemExit("v8 replay input is invalid or has drifted")

    parent_model = args.parent_checkpoint / "model.safetensors"
    parent_training_manifest = (
        args.parent_checkpoint / "mimi_training_manifest.json"
    )
    parent = copy.deepcopy(v7_contract["parent"])
    if (
        parent.get("path") != str(args.parent_checkpoint)
        or parent.get("model_sha256") != sha256(parent_model)
        or parent.get("training_manifest", {}).get("sha256")
        != sha256(parent_training_manifest)
    ):
        raise SystemExit("v8 parent differs from the authenticated v7 parent")
    parent_dataset_manifest = load_json(parent_training_manifest).get(
        "dataset_manifest"
    )
    if (
        not isinstance(parent_dataset_manifest, dict)
        or parent_dataset_manifest.get("outputs_authenticated") is not True
        or parent_dataset_manifest.get("direction") != "ja-en"
        or parent_dataset_manifest.get("sha256")
        != replay_manifest.get("inputs", {})
        .get("parent_manifest", {})
        .get("sha256")
    ):
        raise SystemExit("v8 parent dataset provenance differs from replay input")

    preference = dataset_record(
        args.preference_directory, ("manifest.json", "train.jsonl", "valid.jsonl")
    )
    replay = dataset_record(
        args.replay_directory,
        ("manifest.json", "replay-train.jsonl", "replay-valid.jsonl"),
    )
    for split in ("train", "valid"):
        if (
            preference["files"][f"{split}.jsonl"]["sha256"]
            != preference_manifest["outputs"][split]["sha256"]
        ):
            raise SystemExit(f"preference {split} hash differs")
        if (
            replay["files"][f"replay-{split}.jsonl"]["sha256"]
            != replay_manifest["outputs"][split]["sha256"]
        ):
            raise SystemExit(f"replay {split} hash differs")

    contract = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": "preregistered-ready-for-training",
        "direction": "ja-en",
        "hypothesis": (
            "The v7 preference-only update passed its internal preference gate but "
            "regressed protected long/legal behavior and introduced negation and "
            "generation failures. One-to-one licensed human replay, replay SFT, "
            "frozen-parent KL, and parent L2 regularization may preserve the parent "
            "while retaining the Claude-5-consensus preference improvement."
        ),
        "design_basis": {
            "primary_intervention": (
                "data mixing/replay plus conservative regularization; no architecture "
                "change and no MoE in this single causal arm"
            ),
            "reasoning_traces_used": False,
            "human_reviewers_used": False,
            "post_judgment_hyperparameter_selection_allowed": False,
        },
        "v7_stop_evidence": record(args.v7_protected_result),
        "v7_contract": record(args.v7_contract),
        "parent": parent,
        "dataset_manifest": parent_dataset_manifest,
        "datasets": {
            "preferences": {
                **preference,
                "pairs": {
                    "train": preference_manifest["counts"]["train"],
                    "valid": preference_manifest["counts"]["valid"],
                },
                "target_source": preference_manifest["target_source"],
                "teacher_models": preference_manifest["teacher_models"],
                "review_policy": preference_manifest["review_policy"],
                "effective_licenses": preference_manifest["effective_licenses"],
                "decontamination": preference_manifest["decontamination"],
            },
            "replay": {
                **replay,
                "rows": {
                    "train": replay_manifest["counts"]["train"],
                    "valid": replay_manifest["counts"]["valid"],
                },
                "purpose": replay_manifest["purpose"],
                "effective_licenses": replay_manifest["effective_licenses"],
                "decontamination": replay_manifest["decontamination"],
            },
        },
        "training": TRAINING,
        "internal_selection_gate": INTERNAL_GATE,
        "conversion": copy.deepcopy(v7_contract["conversion"]),
        "held_out_evaluation": copy.deepcopy(v7_contract["held_out_evaluation"]),
        "implementation": {
            (
                "trainer"
                if path.name == "train_marian_claude5_preference_replay.py"
                else path.stem
            ): record(path)
            for path in IMPLEMENTATION
        },
        "training_authorized": True,
        "quantization_authorized": False,
        "protected_evaluation_authorized": False,
        "bundle_creation_authorized": False,
        "promotion_authorized": False,
        "app_change_authorized": False,
        "public_upload_authorized": False,
        "next_step_if_internal_gate_passes": (
            "assemble an internal result, convert only its selected checkpoint to "
            "exact MLX q4/group-64, then run the unchanged protected suite once"
        ),
        "stop_rule": (
            "Stop before exact-q4 conversion or protected evaluation unless one "
            "trained checkpoint passes every preference and replay preservation "
            "gate. Stop before bundle creation, app integration, or public upload "
            "on any protected quality, safety, runtime, size, or provenance failure."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"output": str(args.output), "sha256": sha256(args.output)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
