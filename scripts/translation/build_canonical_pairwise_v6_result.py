#!/usr/bin/env python3
"""Assemble the machine-readable internal-gate rejection for v6."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


CONTRACT = Path(
    "Research/translation/canonical-pairwise-v6-contract-2026-07-25.json"
)
MODEL = Path("Research/translation/models/elanmt-canonical-pairwise-v6-ja-en")
MARGINS = Path(
    "Research/translation/results/canonical-pairwise-v6-ja-en-valid-margins.json"
)
OUTPUT = Path(
    "Research/translation/canonical-pairwise-v6-result-2026-07-25.json"
)


def sha256(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"missing result input: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite frozen result: {OUTPUT}")
    contract = load_json(CONTRACT)
    manifest_path = MODEL / "mimi_automated_preference_manifest.json"
    manifest = load_json(manifest_path)
    margins = load_json(MARGINS)
    if (
        contract.get("experiment")
        != "canonical-pairwise-preference-balanced-v6-ja-en"
        or manifest.get("operation")
        != "domain-balanced automated-consensus full-parameter preference update"
        or manifest.get("experiment_contract", {}).get("sha256") != sha256(CONTRACT)
        or manifest.get("hyperparameters") != contract.get("training")
        or margins.get("candidate", {}).get("model_sha256")
        != sha256(MODEL / "model.safetensors")
    ):
        raise SystemExit("v6 training or diagnostic binding differs")
    best = manifest["best"]
    gate = contract["internal_selection_gate"]
    accuracy_passed = (
        best["relative_pair_accuracy"] >= gate["relative_pair_accuracy_minimum"]
    )
    margin_passed = (
        best["relative_margin"] > gate["relative_margin_minimum_exclusive"]
    )
    passed = accuracy_passed and margin_passed and best["step"] > 0
    if passed:
        raise SystemExit("this result assembler is only for the rejected v6 run")
    failures = [
        {
            "source_id": row["source_id"],
            "domain": row["domain"],
            "relative_margin": row["relative_margin"],
        }
        for row in margins["pairs"]
        if row["improved"] is not True
    ]
    result = {
        "schema_version": 1,
        "experiment": contract["experiment"],
        "status": "internal-preference-gate-rejected",
        "decision": (
            "Stop v6 before MLX conversion or protected held-out evaluation. "
            "Do not tune further against the same ten validation pairs."
        ),
        "teacher_authentication": contract["teacher_authentication"],
        "private_reasoning_traces_used": False,
        "human_reviewers_used": False,
        "contract": record(CONTRACT),
        "training": {
            "manifest": record(manifest_path),
            "model": record(MODEL / "model.safetensors"),
            "trainable_parameters": manifest["trainable_parameters"],
            "domain_balance": manifest["preferences"]["domain_balance"],
            "history": manifest["history"],
            "best": best,
        },
        "internal_gate": {
            "passed": passed,
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
            ],
        },
        "per_pair_diagnostic": {
            "artifact": record(MARGINS),
            "summary": margins["summary"],
            "non_improved": failures,
        },
        "mlx_conversion_performed": False,
        "protected_held_out_evaluation_performed": False,
        "continue_this_recipe": False,
        "promotion_authorized": False,
        "app_change_authorized": False,
        "public_upload_authorized": False,
        "next_hypothesis": (
            "Collect a substantially larger, fresh, balanced JA-to-EN teacher corpus "
            "and reserve a new internal split instead of tuning against these ten pairs."
        ),
    }
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(OUTPUT), "sha256": sha256(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
