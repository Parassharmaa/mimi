#!/usr/bin/env python3
"""Freeze the parent-specialist interpolation v9 experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


EXPERIMENT = "parent-specialist-interpolation-v9-ja-en"
IMPLEMENTATION = (
    Path("scripts/translation/prepare_canonical_pairwise_v9_contract.py"),
    Path("scripts/translation/run_marian_parent_specialist_interpolation.py"),
    Path("scripts/translation/build_canonical_pairwise_v9_result.py"),
    Path("scripts/translation/train_marian_claude5_preference.py"),
    Path("scripts/translation/train_marian_claude5_preference_replay.py"),
    Path("scripts/translation/train_marian_dqo.py"),
    Path("scripts/translation/evaluate_gpt56_student_continuation.py"),
    Path("scripts/translation/audit_translation_structures.py"),
    Path("scripts/translation/typed_critical_token_policy.py"),
)


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


def dataset_record(directory: Path, filenames: tuple[str, ...]) -> dict:
    return {
        "directory": str(directory),
        "files": {
            filename: record(directory / filename) for filename in filenames
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("v7_contract", type=Path)
    parser.add_argument("v7_internal_result", type=Path)
    parser.add_argument("v7_protected_result", type=Path)
    parser.add_argument("v8_result", type=Path)
    parser.add_argument("preference_directory", type=Path)
    parser.add_argument("replay_directory", type=Path)
    parser.add_argument("parent_checkpoint", type=Path)
    parser.add_argument("specialist_checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen contract: {args.output}")

    v7_contract = load_json(args.v7_contract)
    v7_internal = load_json(args.v7_internal_result)
    v7_protected = load_json(args.v7_protected_result)
    v8_result = load_json(args.v8_result)
    if (
        v7_contract.get("status") != "preregistered-ready-for-training"
        or v7_internal.get("status") != "internal-preference-gate-passed"
        or v7_internal.get("internal_gate", {}).get("passed") is not True
        or v7_protected.get("status") != "protected-promotion-gate-rejected"
        or v7_protected.get("promotion_authorized") is not False
        or v8_result.get("status") != "internal-preservation-gate-rejected"
        or v8_result.get("quantization_authorized") is not False
    ):
        raise SystemExit("v7/v8 evidence does not authorize the v9 diagnostic arm")

    parent = copy.deepcopy(v7_contract["parent"])
    if (
        parent.get("path") != str(args.parent_checkpoint)
        or parent.get("model_sha256")
        != sha256(args.parent_checkpoint / "model.safetensors")
    ):
        raise SystemExit("v9 parent differs from the authenticated parent")
    specialist_model = args.specialist_checkpoint / "model.safetensors"
    if (
        v7_internal.get("candidate", {}).get("directory")
        != str(args.specialist_checkpoint)
        or v7_internal.get("candidate", {}).get("model", {}).get("sha256")
        != sha256(specialist_model)
    ):
        raise SystemExit("v9 specialist differs from the selected v7 checkpoint")
    specialist = {
        "path": str(args.specialist_checkpoint),
        "model_sha256": sha256(specialist_model),
        "training_manifest": record(
            args.specialist_checkpoint / "mimi_automated_preference_manifest.json"
        ),
        "source_internal_result": record(args.v7_internal_result),
        "known_protected_status": v7_protected["status"],
        "selectable_at_alpha_one": False,
    }

    preferences = dataset_record(
        args.preference_directory, ("manifest.json", "train.jsonl", "valid.jsonl")
    )
    replay = dataset_record(
        args.replay_directory,
        ("manifest.json", "replay-train.jsonl", "replay-valid.jsonl"),
    )
    preference_manifest = load_json(args.preference_directory / "manifest.json")
    replay_manifest = load_json(args.replay_directory / "manifest.json")
    if (
        preference_manifest.get("experiment")
        != "canonical-pairwise-v7-ja-en-claude5"
        or preference_manifest.get("counts", {}).get("valid") != 33
        or replay_manifest.get("experiment")
        != "canonical-pairwise-v8-ja-en-licensed-replay"
        or replay_manifest.get("counts", {}).get("valid") != 128
        or replay_manifest.get("decontamination", {}).get(
            "preference_replay_source_overlap"
        )
        is not False
    ):
        raise SystemExit("v9 validation inputs are invalid")

    contract = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": "preregistered-ready-for-interpolation",
        "direction": "ja-en",
        "hypothesis": (
            "The v7 specialist has adequate preference accuracy but failed protected "
            "retention, while v8 replay training preserved lexical metrics but lost "
            "preference accuracy. Linear interpolation may remain in the shared low-"
            "loss basin and recover a validation-qualified retention/adaptation tradeoff "
            "without increasing parameters, runtime, or bundle size."
        ),
        "literature_basis": [
            {
                "title": "Revisiting Checkpoint Averaging for Neural Machine Translation",
                "url": "https://aclanthology.org/2022.findings-aacl.18/",
                "use": "NMT checkpoint and weighted parameter averaging",
            },
            {
                "title": "Model soups",
                "url": "https://proceedings.mlr.press/v162/wortsman22a.html",
                "use": "single-model weight averaging without inference overhead",
            },
            {
                "title": "Robust fine-tuning of zero-shot models",
                "url": "https://arxiv.org/abs/2109.01903",
                "use": "parent/fine-tuned interpolation for robustness retention",
            },
        ],
        "v7_contract": record(args.v7_contract),
        "v7_protected_result": record(args.v7_protected_result),
        "v8_result": record(args.v8_result),
        "parent": parent,
        "specialist": specialist,
        "datasets": {
            "preferences": {
                **preferences,
                "valid_pairs": preference_manifest["counts"]["valid"],
                "effective_licenses": preference_manifest["effective_licenses"],
                "decontamination": preference_manifest["decontamination"],
            },
            "replay": {
                **replay,
                "valid_rows": replay_manifest["counts"]["valid"],
                "effective_licenses": replay_manifest["effective_licenses"],
                "decontamination": replay_manifest["decontamination"],
            },
        },
        "interpolation": {
            "formula": "theta(alpha) = theta(parent) + alpha * (theta(v7) - theta(parent))",
            "selectable_alphas": [0.25, 0.5, 0.75],
            "diagnostic_anchor_alphas": [0.0, 1.0],
            "selection": (
                "among eligible selectable alphas: highest relative-pair accuracy, "
                "then relative margin, legal replay chrF++, then lower alpha"
            ),
            "training_steps": 0,
            "additional_inference_parameters": 0,
        },
        "internal_selection_gate": {
            "relative_pair_accuracy_minimum": 0.8,
            "relative_margin_minimum_exclusive": 0.0,
            "replay_token_nll_delta_maximum": 0.01,
            "replay_chrf_pp_delta_minimum": -0.10,
            "legal_replay_chrf_pp_delta_minimum": -0.10,
            "new_exact_critical_maximum": 0,
            "new_typed_critical_maximum": 0,
            "new_negation_maximum": 0,
            "new_repetition_or_nontermination_maximum": 0,
            "padding_after_eos_ignored": True,
        },
        "conversion": copy.deepcopy(v7_contract["conversion"]),
        "held_out_evaluation": copy.deepcopy(v7_contract["held_out_evaluation"]),
        "implementation": {
            (
                "runner"
                if path.name == "run_marian_parent_specialist_interpolation.py"
                else path.stem
            ): record(path)
            for path in IMPLEMENTATION
        },
        "interpolation_authorized": True,
        "quantization_authorized": False,
        "protected_evaluation_authorized": False,
        "bundle_creation_authorized": False,
        "promotion_authorized": False,
        "app_change_authorized": False,
        "public_upload_authorized": False,
        "stop_rule": (
            "Stop before exact-q4 conversion unless a selectable interpolation passes "
            "every frozen internal gate. Stop before bundle creation, app integration, "
            "or upload on any protected quality, safety, runtime, size, or provenance "
            "failure."
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
