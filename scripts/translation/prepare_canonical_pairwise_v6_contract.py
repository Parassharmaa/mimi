#!/usr/bin/env python3
"""Freeze the domain-balanced JA-to-EN preference v6 experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


V5_CONTRACT = Path(
    "Research/translation/canonical-pairwise-v5-contract-2026-07-25.json"
)
V5_RESULT = Path(
    "Research/translation/canonical-pairwise-v5-result-2026-07-25.json"
)
IMPLEMENTATION = (
    Path("scripts/translation/train_marian_automated_preference_balanced.py"),
    Path("scripts/translation/train_marian_automated_preference_full.py"),
    Path("scripts/translation/train_marian_automated_preference_adapter.py"),
    Path("scripts/translation/train_marian_dqo.py"),
    Path("scripts/translation/train_marian_distillation.py"),
)
TRAINING = {
    "seed": 20260728,
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
    "sampling": "exact-domain-balanced-cyclic-v1",
}


def sha256(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"missing contract input: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
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
            "Research/translation/canonical-pairwise-v6-contract-2026-07-25.json"
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen contract: {args.output}")
    v5_contract = load_json(V5_CONTRACT)
    v5_result = load_json(V5_RESULT)
    if (
        v5_contract.get("experiment")
        != "canonical-pairwise-preference-full-v5-ja-en"
        or v5_result.get("status") != "internal-preference-gate-rejected"
        or v5_result.get("internal_gate", {}).get("passed") is not False
        or v5_result.get("protected_held_out_evaluation_performed") is not False
        or v5_result.get("contract", {}).get("sha256") != sha256(V5_CONTRACT)
    ):
        raise SystemExit("v5 evidence does not authorize the v6 hypothesis")
    failures = v5_result.get("per_pair_diagnostic", {}).get("non_improved", [])
    failure_domains = sorted(str(value.get("domain")) for value in failures)
    if failure_domains != [
        "conversational",
        "human-translated-news",
        "human-translated-news",
        "mimi-product-ui",
    ]:
        raise SystemExit("v5 failure pattern differs from the domain-balance hypothesis")
    for name, value in v5_contract["dataset"]["files"].items():
        if value["sha256"] != sha256(Path(value["path"])):
            raise SystemExit(f"dataset binding has drifted: {name}")
    if v5_contract["parent"]["model_sha256"] != sha256(
        Path(v5_contract["parent"]["path"]) / "model.safetensors"
    ):
        raise SystemExit("parent binding has drifted")
    for value in v5_contract["held_out_evaluation"]["benchmarks"].values():
        if value["sha256"] != sha256(Path(value["path"])):
            raise SystemExit(f"held-out benchmark has drifted: {value['path']}")
    contract = {
        "schema_version": 1,
        "experiment": "canonical-pairwise-preference-balanced-v6-ja-en",
        "status": "preregistered-ready-for-training",
        "direction": "ja-en",
        "hypothesis": (
            "V5 improved every legal, Wikipedia, and long-document validation pair "
            "but worsened the scarce conversation/UI pairs and two news pairs. Exact "
            "inverse-frequency domain balancing should preserve the preference signal "
            "while preventing high-count domains from dominating the update."
        ),
        "teacher_authentication": v5_contract["teacher_authentication"],
        "human_reviewers_used": False,
        "private_reasoning_traces_used": False,
        "parent": copy.deepcopy(v5_contract["parent"]),
        "dataset": copy.deepcopy(v5_contract["dataset"]),
        "training": TRAINING,
        "internal_selection_gate": copy.deepcopy(
            v5_contract["internal_selection_gate"]
        ),
        "conversion": copy.deepcopy(v5_contract["conversion"]),
        "held_out_evaluation": copy.deepcopy(
            v5_contract["held_out_evaluation"]
        ),
        "implementation": {
            (
                "trainer"
                if path.name == "train_marian_automated_preference_balanced.py"
                else path.stem
            ): record(path)
            for path in IMPLEMENTATION
        },
        "evidence": {
            "v5_contract": record(V5_CONTRACT),
            "v5_result": record(V5_RESULT),
            "v5_failure_domains": failure_domains,
            "v5_best_relative_pair_accuracy": v5_result["training"]["best"][
                "relative_pair_accuracy"
            ],
            "v5_best_relative_margin": v5_result["training"]["best"][
                "relative_margin"
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
    args.output.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output)}, indent=2))


if __name__ == "__main__":
    main()
