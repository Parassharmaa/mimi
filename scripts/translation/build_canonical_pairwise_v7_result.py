#!/usr/bin/env python3
"""Assemble the fail-closed internal result for the Claude-5 v7 training arm."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


EXPERIMENT = "canonical-pairwise-preference-claude5-v7-ja-en"


def sha256(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"missing result input: {path}")
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
    parser.add_argument("contract", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite result: {args.output}")

    contract = load_json(args.contract)
    manifest_path = args.candidate / "mimi_automated_preference_manifest.json"
    model_path = args.candidate / "model.safetensors"
    manifest = load_json(manifest_path)
    if (
        contract.get("experiment") != EXPERIMENT
        or contract.get("status") != "preregistered-ready-for-training"
        or manifest.get("experiment") != EXPERIMENT
        or manifest.get("status") != "trained-not-promotion-evaluated"
        or manifest.get("experiment_contract", {}).get("sha256")
        != sha256(args.contract)
        or manifest.get("direction") != "ja-en"
        or manifest.get("promotion_authorized") is not False
        or manifest.get("app_change_authorized") is not False
        or manifest.get("public_upload_authorized") is not False
    ):
        raise SystemExit("v7 candidate is not bound to the frozen experiment")
    if (
        manifest.get("hyperparameters") != contract.get("training")
        or manifest.get("preferences", {}).get("manifest_sha256")
        != contract.get("dataset", {})
        .get("files", {})
        .get("manifest.json", {})
        .get("sha256")
        or set(manifest.get("preferences", {}).get("required_judge_model_ids", []))
        != {"claude-sonnet-5", "claude-opus-5"}
    ):
        raise SystemExit("v7 candidate recipe or dataset binding differs")

    best = manifest.get("best", {})
    gate = contract["internal_selection_gate"]
    checks = [
        {
            "name": "trained-checkpoint",
            "required": "step > 0",
            "actual": best.get("step"),
            "passed": isinstance(best.get("step"), int) and best["step"] > 0,
        },
        {
            "name": "relative-pair-accuracy",
            "required": gate["relative_pair_accuracy_minimum"],
            "actual": best.get("relative_pair_accuracy"),
            "passed": isinstance(best.get("relative_pair_accuracy"), (int, float))
            and best["relative_pair_accuracy"]
            >= gate["relative_pair_accuracy_minimum"],
        },
        {
            "name": "relative-margin",
            "required": "> 0.0",
            "actual": best.get("relative_margin"),
            "passed": isinstance(best.get("relative_margin"), (int, float))
            and best["relative_margin"]
            > gate["relative_margin_minimum_exclusive"],
        },
    ]
    passed = all(check["passed"] for check in checks)
    result = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": (
            "internal-preference-gate-passed"
            if passed
            else "internal-preference-gate-rejected"
        ),
        "contract": record(args.contract),
        "candidate": {
            "directory": str(args.candidate),
            "model": record(model_path),
            "training_manifest": record(manifest_path),
        },
        "training": {
            "best": best,
            "trainable_parameters": manifest.get("trainable_parameters"),
            "history": manifest.get("history"),
        },
        "internal_gate": {"passed": passed, "checks": checks},
        "quantization_authorized": passed,
        "quantization_run": None,
        "protected_evaluation_authorized": False,
        "protected_evaluation_run": None,
        "promotion_authorized": False,
        "app_change_authorized": False,
        "public_upload_authorized": False,
        "decision": (
            "Convert only this validation-selected checkpoint to exact MLX "
            "q4/group-64; protected evaluation remains separately gated."
            if passed
            else "Stop before MLX conversion or protected held-out evaluation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output),
                "sha256": sha256(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
