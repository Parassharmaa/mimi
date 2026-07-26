#!/usr/bin/env python3
"""Assemble the fail-closed internal result for v9 interpolation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


EXPERIMENT = "parent-specialist-interpolation-v9-ja-en"


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
        or contract.get("status") != "preregistered-ready-for-interpolation"
        or contract.get("implementation", {})
        .get("build_canonical_pairwise_v9_result", {})
        .get("sha256")
        != sha256(Path(__file__).resolve())
        or manifest.get("experiment") != EXPERIMENT
        or manifest.get("direction") != "ja-en"
        or manifest.get("experiment_contract", {}).get("sha256")
        != sha256(args.contract)
        or manifest.get("promotion_authorized") is not False
        or manifest.get("app_change_authorized") is not False
        or manifest.get("public_upload_authorized") is not False
    ):
        raise SystemExit("v9 candidate is not bound to the frozen experiment")
    best = manifest.get("best")
    best_eligible = manifest.get("best_eligible")
    passed = (
        manifest.get("status") == "internal-interpolation-gate-passed"
        and isinstance(best_eligible, dict)
        and best == best_eligible
        and best_eligible.get("internal_gate", {}).get("passed") is True
    )
    if not passed and manifest.get("status") != "internal-interpolation-gate-rejected":
        raise SystemExit("v9 interpolation status is inconsistent")
    result = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": (
            "internal-interpolation-gate-passed"
            if passed
            else "internal-interpolation-gate-rejected"
        ),
        "contract": record(args.contract),
        "candidate": {
            "directory": str(args.candidate),
            "model": record(model_path),
            "manifest": record(manifest_path),
        },
        "interpolation": {
            "best": best,
            "best_eligible": best_eligible,
            "history": manifest.get("history"),
            "parameters": manifest.get("parameters"),
            "training_steps": 0,
        },
        "internal_gate": (
            best_eligible["internal_gate"]
            if passed
            else best.get("internal_gate", {"passed": False, "checks": []})
        ),
        "quantization_authorized": passed,
        "quantization_run": None,
        "protected_evaluation_authorized": False,
        "protected_evaluation_run": None,
        "bundle_creation_authorized": False,
        "promotion_authorized": False,
        "app_change_authorized": False,
        "public_upload_authorized": False,
        "decision": (
            "Convert only this selected interpolation to exact MLX q4/group-64; "
            "protected evaluation remains separately gated."
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
