#!/usr/bin/env python3
"""Assemble the machine-readable internal-gate rejection for v5."""

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
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(
            "Research/translation/canonical-pairwise-v5-contract-2026-07-25.json"
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "Research/translation/models/elanmt-canonical-pairwise-v5-ja-en"
        ),
    )
    parser.add_argument(
        "--margins",
        type=Path,
        default=Path(
            "Research/translation/results/canonical-pairwise-v5-ja-en-valid-margins.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "Research/translation/canonical-pairwise-v5-result-2026-07-25.json"
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen result: {args.output}")
    contract = load_json(args.contract)
    manifest_path = args.model / "mimi_automated_preference_manifest.json"
    manifest = load_json(manifest_path)
    margins = load_json(args.margins)
    if (
        contract.get("experiment")
        != "canonical-pairwise-preference-full-v5-ja-en"
        or manifest.get("operation")
        != "automated-consensus full-parameter preference update"
        or manifest.get("experiment_contract", {}).get("sha256")
        != sha256(args.contract)
        or manifest.get("hyperparameters") != contract.get("training")
        or margins.get("candidate", {}).get("model_sha256")
        != sha256(args.model / "model.safetensors")
        or margins.get("parent", {}).get("model_sha256")
        != contract.get("parent", {}).get("model_sha256")
    ):
        raise SystemExit("v5 training or diagnostic binding differs")
    best = manifest["best"]
    gate = contract["internal_selection_gate"]
    accuracy_passed = (
        best["relative_pair_accuracy"] >= gate["relative_pair_accuracy_minimum"]
    )
    margin_passed = (
        best["relative_margin"] > gate["relative_margin_minimum_exclusive"]
    )
    internal_passed = accuracy_passed and margin_passed and best["step"] > 0
    if internal_passed:
        raise SystemExit("this result assembler is only for the rejected v5 run")
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
            "Stop v5 before MLX conversion or protected held-out evaluation. "
            "Do not integrate, publish, or claim improvement."
        ),
        "teacher_authentication": contract["teacher_authentication"],
        "private_reasoning_traces_used": False,
        "human_reviewers_used": False,
        "contract": record(args.contract),
        "training": {
            "manifest": record(manifest_path),
            "model": record(args.model / "model.safetensors"),
            "trainable_parameters": manifest["trainable_parameters"],
            "history": manifest["history"],
            "best": best,
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
            ],
        },
        "per_pair_diagnostic": {
            "artifact": record(args.margins),
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
            "Use deterministic inverse-frequency domain sampling because all "
            "legal/Wikipedia/document pairs improved while scarce conversation/UI "
            "and two news pairs account for all four failures."
        ),
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output)}, indent=2))


if __name__ == "__main__":
    main()
