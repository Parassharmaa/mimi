#!/usr/bin/env python3
"""Freeze the full-parameter canonical JA-to-EN preference v5 experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


V4_CONTRACT = Path(
    "Research/translation/canonical-pairwise-v4-contract-2026-07-25.json"
)
V4_RESULT = Path(
    "Research/translation/canonical-pairwise-v4-result-2026-07-25.json"
)
IMPLEMENTATION = (
    Path("scripts/translation/train_marian_automated_preference_full.py"),
    Path("scripts/translation/train_marian_automated_preference_adapter.py"),
    Path("scripts/translation/train_marian_dqo.py"),
    Path("scripts/translation/train_marian_distillation.py"),
)
TRAINING = {
    "seed": 20260727,
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


def record(path: Path) -> dict:
    return {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid JSON input {path}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path(
            "Research/translation/canonical-pairwise-v5-contract-2026-07-25.json"
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen contract: {args.output}")

    v4_contract = load_json(V4_CONTRACT)
    v4_result = load_json(V4_RESULT)
    if (
        v4_contract.get("experiment")
        != "canonical-pairwise-preference-adapter-v4-ja-en"
        or v4_result.get("status") != "internal-preference-gate-rejected"
        or v4_result.get("internal_gate", {}).get("passed") is not False
        or v4_result.get("protected_held_out_evaluation_performed") is not False
    ):
        raise SystemExit("v4 evidence does not authorize the v5 hypothesis")
    if v4_result.get("contract", {}).get("sha256") != sha256(V4_CONTRACT):
        raise SystemExit("v4 result is not bound to the frozen v4 contract")
    for name, value in v4_contract["dataset"]["files"].items():
        if value["sha256"] != sha256(Path(value["path"])):
            raise SystemExit(f"v4 dataset binding has drifted: {name}")
    if v4_contract["parent"]["model_sha256"] != sha256(
        Path(v4_contract["parent"]["path"]) / "model.safetensors"
    ):
        raise SystemExit("v4 parent binding has drifted")
    for value in v4_contract["held_out_evaluation"]["benchmarks"].values():
        if value["sha256"] != sha256(Path(value["path"])):
            raise SystemExit(f"held-out benchmark has drifted: {value['path']}")

    contract = {
        "schema_version": 1,
        "experiment": "canonical-pairwise-preference-full-v5-ja-en",
        "status": "preregistered-ready-for-training",
        "direction": "ja-en",
        "hypothesis": (
            "V4's positive mean validation margin but 0.60 pair accuracy reflects "
            "under-capacity or misplaced rank-8 updates. A 40-step full-parameter "
            "update at 5e-7 with explicit parent-displacement regularization should "
            "reach at least 0.80 internal preference accuracy without the broad drift "
            "of v3's 250-step 2e-6 supervised update."
        ),
        "teacher_authentication": v4_contract["teacher_authentication"],
        "human_reviewers_used": False,
        "private_reasoning_traces_used": False,
        "parent": copy.deepcopy(v4_contract["parent"]),
        "dataset": copy.deepcopy(v4_contract["dataset"]),
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
        "conversion": copy.deepcopy(v4_contract["conversion"]),
        "held_out_evaluation": copy.deepcopy(
            v4_contract["held_out_evaluation"]
        ),
        "implementation": {
            (
                "trainer"
                if path.name == "train_marian_automated_preference_full.py"
                else path.stem
            ): record(path)
            for path in IMPLEMENTATION
        },
        "evidence": {
            "v4_contract": record(V4_CONTRACT),
            "v4_result": record(V4_RESULT),
            "v4_best_relative_pair_accuracy": v4_result["training"]["best"][
                "relative_pair_accuracy"
            ],
            "v4_best_relative_margin": v4_result["training"]["best"][
                "relative_margin"
            ],
            "v4_adapter_trainable_parameters": v4_result["training"][
                "trainable_parameters"
            ],
        },
        "promotion_authorized": False,
        "app_change_authorized": False,
        "public_upload_authorized": False,
        "next_step_if_internal_gate_passes": (
            "convert the one validation-selected checkpoint to exact MLX q4/group-64 "
            "and run the unchanged held-out promotion suite once"
        ),
        "stop_rule": (
            "Stop before MLX conversion and protected evaluation if no trained "
            "checkpoint reaches 0.80 internal relative-pair accuracy with positive "
            "mean margin. Otherwise stop without integration or upload if any held-out "
            "quality, safety, runtime, size, or licensing gate fails."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output)}, indent=2))


if __name__ == "__main__":
    main()
