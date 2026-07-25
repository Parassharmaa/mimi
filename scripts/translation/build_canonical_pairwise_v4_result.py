#!/usr/bin/env python3
"""Assemble the machine-readable result for the stopped v4 adapter run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"missing result input: {path}")
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
        "--contract",
        type=Path,
        default=Path(
            "Research/translation/canonical-pairwise-v4-contract-2026-07-25.json"
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "Research/translation/models/elanmt-canonical-pairwise-v4-ja-en"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "Research/translation/canonical-pairwise-v4-result-2026-07-25.json"
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen result: {args.output}")

    contract = load_json(args.contract)
    manifest_path = args.model / "mimi_automated_preference_manifest.json"
    manifest = load_json(manifest_path)
    if (
        contract.get("experiment")
        != "canonical-pairwise-preference-adapter-v4-ja-en"
        or contract.get("status") != "preregistered-ready-for-training"
        or manifest.get("operation")
        != "automated-consensus teacher-over-current preference adapter"
        or manifest.get("status") != "trained-not-promotion-evaluated"
        or manifest.get("direction") != "ja-en"
        or manifest.get("private_reasoning_traces_used") is not False
        or manifest.get("human_reviewers_used") is not False
    ):
        raise SystemExit("v4 contract/training manifest identity differs")
    if manifest.get("experiment_contract", {}).get("sha256") != sha256(
        args.contract
    ):
        raise SystemExit("v4 training manifest does not bind the contract")
    if manifest.get("parent", {}).get("model_sha256") != contract.get(
        "parent", {}
    ).get("model_sha256"):
        raise SystemExit("v4 parent binding differs")
    if manifest.get("hyperparameters") != contract.get("training"):
        raise SystemExit("v4 trained hyperparameters differ from contract")
    best = manifest.get("best", {})
    gate = contract.get("internal_selection_gate", {})
    accuracy_passed = (
        isinstance(best.get("relative_pair_accuracy"), (int, float))
        and best["relative_pair_accuracy"]
        >= gate["relative_pair_accuracy_minimum"]
    )
    margin_passed = (
        isinstance(best.get("relative_margin"), (int, float))
        and best["relative_margin"] > gate["relative_margin_minimum_exclusive"]
    )
    internal_passed = accuracy_passed and margin_passed and best.get("step", 0) > 0
    if internal_passed:
        raise SystemExit("this result assembler is only for the stopped v4 run")
    checkpoints = manifest.get("checkpoints")
    if (
        not isinstance(checkpoints, list)
        or [value.get("step") for value in checkpoints] != [10, 20, 30, 40]
        or any(
            value.get("sha256") != sha256(Path(value["path"]))
            for value in checkpoints
        )
    ):
        raise SystemExit("v4 checkpoints are incomplete or unauthenticated")

    result = {
        "schema_version": 1,
        "experiment": contract["experiment"],
        "status": "internal-preference-gate-rejected",
        "decision": (
            "Stop the rank-8 v4 recipe before MLX conversion or protected held-out "
            "evaluation. Do not integrate, publish, or claim improvement."
        ),
        "teacher_authentication": contract["teacher_authentication"],
        "private_reasoning_traces_used": False,
        "human_reviewers_used": False,
        "contract": record(args.contract),
        "dataset": contract["dataset"],
        "parent": contract["parent"],
        "training": {
            "manifest": record(manifest_path),
            "model": record(args.model / "model.safetensors"),
            "adapter": record(args.model / "adapters.safetensors"),
            "trainable_parameters": manifest["adapter"]["trainable_parameters"],
            "selected_modules": len(manifest["adapter"]["selected_modules"]),
            "history": manifest["history"],
            "best": best,
            "checkpoints": checkpoints,
        },
        "internal_gate": {
            "passed": internal_passed,
            "checks": [
                {
                    "name": "relative-pair-accuracy",
                    "required": gate["relative_pair_accuracy_minimum"],
                    "actual": best["relative_pair_accuracy"],
                    "passed": accuracy_passed,
                },
                {
                    "name": "relative-margin",
                    "required": f"> {gate['relative_margin_minimum_exclusive']}",
                    "actual": best["relative_margin"],
                    "passed": margin_passed,
                },
                {
                    "name": "trained-checkpoint",
                    "required": "step > 0",
                    "actual": best["step"],
                    "passed": best["step"] > 0,
                },
            ],
        },
        "mlx_conversion_performed": False,
        "protected_held_out_evaluation_performed": False,
        "continue_this_recipe": False,
        "promotion_authorized": False,
        "app_change_authorized": False,
        "public_upload_authorized": False,
        "next_hypothesis": (
            "Test whether rank-8 module placement caused underfit using a separately "
            "preregistered full-parameter preference update at a very low learning "
            "rate with explicit displacement regularization."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output)}, indent=2))


if __name__ == "__main__":
    main()
