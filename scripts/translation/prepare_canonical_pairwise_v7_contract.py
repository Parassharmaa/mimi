#!/usr/bin/env python3
"""Freeze the one-arm Claude-5-consensus JA-to-EN preference experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "canonical-pairwise-preference-claude5-v7-ja-en"
DATASET_EXPERIMENT = "canonical-pairwise-v7-ja-en-claude5"
REQUIRED_JUDGES = {"claude-sonnet-5", "claude-opus-5"}
HISTORICAL_RESULTS = (
    Path("Research/translation/canonical-pairwise-v4-result-2026-07-25.json"),
    Path("Research/translation/canonical-pairwise-v5-result-2026-07-25.json"),
    Path("Research/translation/canonical-pairwise-v6-result-2026-07-25.json"),
)
V6_CONTRACT = Path(
    "Research/translation/canonical-pairwise-v6-contract-2026-07-25.json"
)
IMPLEMENTATION = (
    Path("scripts/translation/train_marian_claude5_preference.py"),
    Path("scripts/translation/train_marian_automated_preference_adapter.py"),
    Path("scripts/translation/train_marian_automated_preference_full.py"),
    Path("scripts/translation/train_marian_dqo.py"),
    Path("scripts/translation/train_marian_distillation.py"),
)
TRAINING = {
    "seed": 20260729,
    "batch_size": 4,
    "gradient_accumulation": 4,
    "max_steps": 40,
    "learning_rate": 0.0000005,
    "weight_decay": 0.01,
    "warmup_steps": 4,
    "evaluation_steps": 10,
    "beta": 0.10,
    "chosen_sft_weight": 0.02,
    "l2_to_parent_weight": 0.10,
    "max_source_tokens": 192,
    "max_target_tokens": 192,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_plan", type=Path)
    parser.add_argument("consensus_result", type=Path)
    parser.add_argument("preference_directory", type=Path)
    parser.add_argument("parent_checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen contract: {args.output}")

    plan = load_json(args.training_plan)
    if (
        plan.get("experiment") != EXPERIMENT
        or plan.get("status")
        not in {
            "frozen-before-judge-collection-complete-or-content-inspection",
            "frozen-compatibility-amendment-before-any-training-step",
        }
        or plan.get("training") != TRAINING
        or set(plan.get("required_judge_model_ids", [])) != REQUIRED_JUDGES
        or plan.get("data_gates", {}).get("absolute_canonical_approvals_minimum")
        != 120
        or plan.get("data_gates", {}).get("unanimous_pareto_pairs_minimum") != 60
        or plan.get("data_gates", {}).get("validation_pairs_minimum") != 12
        or plan.get("single_training_arm") is not True
        or plan.get("hyperparameter_selection_after_judgments_allowed") is not False
    ):
        raise SystemExit("v7 training plan is invalid or has drifted")
    for implementation in plan.get("implementation", {}).values():
        if implementation.get("sha256") != sha256(Path(implementation["path"])):
            raise SystemExit("v7 training-plan implementation has drifted")

    consensus = load_json(args.consensus_result)
    if (
        consensus.get("status") != "absolute-consensus-scale-gate-passed"
        or consensus.get("pairwiseDatasetConstructionAuthorized") is not True
        or consensus.get("excludedJudgeEvidenceUsed") is not False
        or set(consensus.get("judges", {})) != REQUIRED_JUDGES
        or consensus.get("gates", {})
        .get("minimumApprovedSources", {})
        .get("passed")
        is not True
    ):
        raise SystemExit("Claude 5 canonical consensus did not authorize pairwise data")
    if (
        consensus.get("contract", {}).get("sha256")
        != plan.get("judge_contract", {}).get("sha256")
    ):
        raise SystemExit("consensus result does not use the frozen plan's judge contract")

    manifest_path = args.preference_directory / "manifest.json"
    dataset = load_json(manifest_path)
    if (
        dataset.get("schema_version") != 1
        or dataset.get("experiment") != DATASET_EXPERIMENT
        or dataset.get("direction") != "ja-en"
        or set(dataset.get("required_judge_model_ids", [])) != REQUIRED_JUDGES
        or dataset.get("private_reasoning_traces_used") is not False
        or dataset.get("promotion_eligible") is not True
        or dataset.get("counts", {}).get("selected", 0) < 60
        or dataset.get("counts", {}).get("valid", 0) < 12
        or dataset.get("inputs", {}).get("approved", {}).get("sha256")
        != consensus.get("outputs", {}).get("approved", {}).get("sha256")
        or dataset.get("inputs", {}).get("review_queue", {}).get("sha256")
        != consensus.get("reviewQueue", {}).get("sha256")
    ):
        raise SystemExit("Claude 5 pairwise dataset does not satisfy the frozen gate")
    dataset_files = {
        name: record(args.preference_directory / name)
        for name in ("manifest.json", "train.jsonl", "valid.jsonl")
    }
    for name in ("train", "valid"):
        if dataset_files[f"{name}.jsonl"]["sha256"] != dataset.get(
            "outputs", {}
        ).get(name, {}).get("sha256"):
            raise SystemExit(f"Claude 5 pairwise {name} hash has drifted")

    v6_contract = load_json(V6_CONTRACT)
    historical = {}
    for path in HISTORICAL_RESULTS:
        result = load_json(path)
        if (
            result.get("status") != "internal-preference-gate-rejected"
            or result.get("internal_gate", {}).get("passed") is not False
            or result.get("quantization_run") is not None
            or result.get("protected_evaluation_run") is not None
        ):
            raise SystemExit(f"historical stop evidence has drifted: {path}")
        historical[path.stem] = {
            **record(path),
            "bestRelativePairAccuracy": result["training"]["best"][
                "relative_pair_accuracy"
            ],
            "bestRelativeMargin": result["training"]["best"]["relative_margin"],
        }

    parent_model = args.parent_checkpoint / "model.safetensors"
    parent_manifest = args.parent_checkpoint / "mimi_training_manifest.json"
    expected_parent = v6_contract.get("parent", {})
    if (
        expected_parent.get("path") != str(args.parent_checkpoint)
        or expected_parent.get("model_sha256") != sha256(parent_model)
    ):
        raise SystemExit("v7 parent differs from the authenticated current JA-to-EN parent")
    parent = copy.deepcopy(expected_parent)
    parent["training_manifest"] = record(parent_manifest)

    contract = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": "preregistered-ready-for-training",
        "direction": "ja-en",
        "hypothesis": (
            "The prior 51-train/10-validation Qwen/Fable-selected preference set "
            "produced positive mean margins but only 0.60 pair accuracy in three "
            "recipes. A fresh set with at least 60 Sonnet-5/Opus-5 unanimous "
            "teacher-over-current pairs and at least 12 untouched validation pairs "
            "can test the same conservative full-parameter objective without "
            "confounding judge quality, data volume, and optimization changes."
        ),
        "human_reviewers_used": False,
        "private_reasoning_traces_used": False,
        "teacher_authentication": (
            "Codex cached ChatGPT authentication; no OpenAI API key used"
        ),
        "judge_models": sorted(REQUIRED_JUDGES),
        "judge_provider_independence_claimed": False,
        "training_plan": record(args.training_plan),
        "consensus_result": record(args.consensus_result),
        "parent": parent,
        "dataset": {
            "directory": str(args.preference_directory),
            "files": dataset_files,
            "pairs": {
                "selected": dataset["counts"]["selected"],
                "train": dataset["counts"]["train"],
                "valid": dataset["counts"]["valid"],
            },
            "required_judge_model_ids": sorted(REQUIRED_JUDGES),
            "effective_licenses": dataset["effective_licenses"],
            "decontamination": dataset["decontamination"],
            "review_policy": dataset["review_policy"],
        },
        "training": TRAINING,
        "internal_selection_gate": {
            "selection": (
                "highest validation relative-pair accuracy, then relative margin, "
                "then lower loss"
            ),
            "relative_pair_accuracy_minimum": 0.8,
            "relative_margin_minimum_exclusive": 0.0,
            "checkpoint_steps": [10, 20, 30, 40],
            "step_zero_is_not_a_trained_candidate": True,
            "validation_pairs_minimum": 12,
        },
        "conversion": copy.deepcopy(v6_contract["conversion"]),
        "held_out_evaluation": copy.deepcopy(v6_contract["held_out_evaluation"]),
        "implementation": {
            (
                "trainer"
                if path.name == "train_marian_claude5_preference.py"
                else path.stem
            ): record(path)
            for path in IMPLEMENTATION
        },
        "historical_stop_evidence": historical,
        "promotion_authorized": False,
        "app_change_authorized": False,
        "public_upload_authorized": False,
        "next_step_if_internal_gate_passes": (
            "convert the single validation-selected checkpoint to exact MLX "
            "q4/group-64 and run the unchanged held-out promotion suite once"
        ),
        "stop_rule": (
            "Stop before training unless at least 120 absolute canonical approvals "
            "and 60 unanimous Pareto pairs exist. Stop before MLX conversion and "
            "protected evaluation unless a trained checkpoint reaches 0.80 internal "
            "relative-pair accuracy with positive mean margin. Any held-out quality, "
            "safety, runtime, size, lineage, or licensing failure stops integration "
            "and upload."
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
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
