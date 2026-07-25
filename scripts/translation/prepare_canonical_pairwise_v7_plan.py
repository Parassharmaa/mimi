#!/usr/bin/env python3
"""Freeze the Claude-5 JA-to-EN training recipe before judge completion."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from prepare_canonical_pairwise_v7_contract import (
    EXPERIMENT,
    HISTORICAL_RESULTS,
    REQUIRED_JUDGES,
    TRAINING,
    V6_CONTRACT,
)


ACTIVE_JUDGE_CONTRACT = Path(
    "Research/translation/"
    "canonical-target-ja-en-scale-v7-claude5-judge-contract-v3-2026-07-25.json"
)
PARENT = Path(
    "Research/translation/models/"
    "elanmt-release-clean-legal-specialist-ja-en-v1"
)
PLAN_SCRIPTS = (
    Path("scripts/translation/build_canonical_pairwise_preference_dataset.py"),
    Path("scripts/translation/train_marian_claude5_preference.py"),
    Path("scripts/translation/prepare_canonical_pairwise_v7_contract.py"),
)


def sha256(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"missing plan input: {path}")
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
    parser.add_argument("output", type=Path)
    parser.add_argument("--supersedes-plan", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen plan: {args.output}")

    judge = load_json(ACTIVE_JUDGE_CONTRACT)
    if (
        judge.get("status")
        != "frozen-before-complete-collection-and-content-inspection"
        or set(judge.get("judgePolicy", {}).get("requiredJudgeModelIds", []))
        != REQUIRED_JUDGES
        or judge.get("goGateBeforeScaling", {}).get("minimumApprovedSources")
        != 120
    ):
        raise SystemExit("active Claude 5 judge contract has drifted")
    v6 = load_json(V6_CONTRACT)
    parent_model = PARENT / "model.safetensors"
    if v6.get("parent", {}).get("model_sha256") != sha256(parent_model):
        raise SystemExit("authenticated current JA-to-EN parent has drifted")
    historical = {}
    for path in HISTORICAL_RESULTS:
        result = load_json(path)
        if (
            result.get("status") != "internal-preference-gate-rejected"
            or result.get("internal_gate", {}).get("passed") is not False
        ):
            raise SystemExit(f"historical preference result has drifted: {path}")
        historical[path.stem] = record(path)

    supersedes = None
    status = "frozen-before-judge-collection-complete-or-content-inspection"
    if args.supersedes_plan:
        prior = load_json(args.supersedes_plan)
        if (
            prior.get("experiment") != EXPERIMENT
            or prior.get("training") != TRAINING
            or prior.get("data_gates", {}).get(
                "absolute_canonical_approvals_minimum"
            )
            != 120
            or prior.get("data_gates", {}).get("unanimous_pareto_pairs_minimum")
            != 60
            or prior.get("internal_selection_gate", {}).get(
                "relative_pair_accuracy_minimum"
            )
            != 0.8
        ):
            raise SystemExit("superseded v7 plan changed a frozen recipe or gate")
        supersedes = {
            **record(args.supersedes_plan),
            "change_scope": (
                "dataset-loader compatibility only after a pre-training rejection; "
                "no optimization step, recipe, data, threshold, or gate changed"
            ),
            "training_steps_before_amendment": 0,
        }
        status = "frozen-compatibility-amendment-before-any-training-step"

    plan = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "judge_contract": record(ACTIVE_JUDGE_CONTRACT),
        "required_judge_model_ids": sorted(REQUIRED_JUDGES),
        "judge_provider_independence_claimed": False,
        "data_gates": {
            "absolute_canonical_approvals_minimum": 120,
            "unanimous_pareto_pairs_minimum": 60,
            "validation_pairs_minimum": 12,
            "validation_fraction": 0.20,
            "required_direction": "ja-en",
            "required_dataset_experiment": (
                "canonical-pairwise-v7-ja-en-claude5"
            ),
            "protected_jaccard_maximum_exclusive": 0.8,
        },
        "parent": {
            **v6["parent"],
            "model_sha256": sha256(parent_model),
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
        },
        "implementation": {
            path.stem: record(path) for path in PLAN_SCRIPTS
        },
        "historical_stop_evidence": historical,
        "single_training_arm": True,
        "hyperparameter_selection_after_judgments_allowed": False,
        "reasoning_traces_used": False,
        "human_reviewers_required": False,
        "training_authorized": False,
        "quantization_authorized": False,
        "protected_evaluation_authorized": False,
        "promotion_authorized": False,
        "app_change_authorized": False,
        "public_upload_authorized": False,
    }
    if supersedes:
        plan["supersedes"] = supersedes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
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
