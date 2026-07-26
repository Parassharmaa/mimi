#!/usr/bin/env python3
"""Assemble the fail-closed internal result for the preservation-aware v8 arm."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


EXPERIMENT = "canonical-pairwise-preference-replay-v8-ja-en"


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
        or contract.get("implementation", {})
        .get("build_canonical_pairwise_v8_result", {})
        .get("sha256")
        != sha256(Path(__file__).resolve())
        or manifest.get("experiment") != EXPERIMENT
        or manifest.get("direction") != "ja-en"
        or manifest.get("experiment_contract", {}).get("sha256")
        != sha256(args.contract)
        or manifest.get("hyperparameters") != contract.get("training")
        or manifest.get("promotion_authorized") is not False
        or manifest.get("app_change_authorized") is not False
        or manifest.get("public_upload_authorized") is not False
    ):
        raise SystemExit("v8 candidate is not bound to the frozen experiment")
    if (
        manifest.get("datasets", {})
        .get("preferences", {})
        .get("manifest_sha256")
        != contract.get("datasets", {})
        .get("preferences", {})
        .get("files", {})
        .get("manifest.json", {})
        .get("sha256")
        or manifest.get("datasets", {})
        .get("replay", {})
        .get("manifest_sha256")
        != contract.get("datasets", {})
        .get("replay", {})
        .get("files", {})
        .get("manifest.json", {})
        .get("sha256")
    ):
        raise SystemExit("v8 candidate dataset bindings differ")

    best = manifest.get("best")
    best_eligible = manifest.get("best_eligible")
    passed = (
        manifest.get("status") == "internal-preservation-gate-passed"
        and isinstance(best_eligible, dict)
        and best == best_eligible
        and best_eligible.get("internal_gate", {}).get("passed") is True
    )
    if not passed and manifest.get("status") != "internal-preservation-gate-rejected":
        raise SystemExit("v8 training status is inconsistent")
    internal_gate = (
        best_eligible["internal_gate"]
        if passed
        else best.get("internal_gate", {"passed": False, "checks": []})
    )
    result = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": (
            "internal-preservation-gate-passed"
            if passed
            else "internal-preservation-gate-rejected"
        ),
        "contract": record(args.contract),
        "candidate": {
            "directory": str(args.candidate),
            "model": record(model_path),
            "training_manifest": record(manifest_path),
        },
        "training": {
            "best": best,
            "best_eligible": best_eligible,
            "trainable_parameters": manifest.get("trainable_parameters"),
            "history": manifest.get("history"),
        },
        "internal_gate": internal_gate,
        "quantization_authorized": passed,
        "quantization_run": None,
        "protected_evaluation_authorized": False,
        "protected_evaluation_run": None,
        "bundle_creation_authorized": False,
        "promotion_authorized": False,
        "app_change_authorized": False,
        "public_upload_authorized": False,
        "decision": (
            "Convert only this validation-selected checkpoint to exact MLX "
            "q4/group-64; protected evaluation remains separately gated."
            if passed
            else "Stop before MLX conversion, protected evaluation, or bundling."
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
