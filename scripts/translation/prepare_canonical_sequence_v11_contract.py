#!/usr/bin/env python3
"""Freeze the protected-independent v11 parent/specialist interpolation study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPERIMENT = "canonical-sequence-v11-ja-en-parent-interpolation"
V10_EXPERIMENT = "canonical-sequence-v10-ja-en-error-stratified"
ADAPTED_WEIGHTS = [0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.625, 0.75]
SPECIALIST_STEP = 250


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"missing required file: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("v10_contract", type=Path)
    parser.add_argument("v10_result", type=Path)
    parser.add_argument("parent_checkpoint", type=Path)
    parser.add_argument("specialist_checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")

    root = Path(__file__).resolve().parents[2]
    v10_contract = load_json(args.v10_contract)
    v10_result = load_json(args.v10_result)
    if (
        v10_contract.get("experiment") != V10_EXPERIMENT
        or v10_contract.get("status")
        != "preregistered-ready-for-one-arm-training"
        or v10_contract.get("app_change_authorized") is not False
        or v10_contract.get("public_upload_authorized") is not False
    ):
        raise SystemExit("v10 contract is invalid")
    if (
        v10_result.get("experiment") != V10_EXPERIMENT
        or v10_result.get("status") != "internal-gate-rejected"
        or v10_result.get("selected_step") is not None
        or v10_result.get("exact_q4_conversion_authorized") is not False
        or v10_result.get("protected_evaluation_authorized_after_exact_q4")
        is not False
        or v10_result.get("app_change_authorized") is not False
        or v10_result.get("public_upload_authorized") is not False
        or v10_result.get("contract", {}).get("sha256")
        != sha256(args.v10_contract)
    ):
        raise SystemExit("v10 result is not a bound internal rejection")

    parent_model = args.parent_checkpoint / "model.safetensors"
    parent_manifest = args.parent_checkpoint / "mimi_training_manifest.json"
    if (
        sha256(parent_model)
        != v10_contract["parent"]["model"]["sha256"]
        or sha256(parent_manifest)
        != v10_contract["parent"]["training_manifest"]["sha256"]
    ):
        raise SystemExit("v11 parent differs from the v10 parent")

    candidates = {
        int(candidate["step"]): candidate
        for candidate in v10_result.get("candidates", [])
    }
    specialist_result = candidates.get(SPECIALIST_STEP)
    if specialist_result is None:
        raise SystemExit("v10 result lacks the selected v11 specialist step")
    specialist_model = args.specialist_checkpoint / "model.safetensors"
    specialist_manifest = (
        args.specialist_checkpoint / "mimi_training_manifest.json"
    )
    if (
        sha256(specialist_model)
        != specialist_result["checkpoint"]["model"]["sha256"]
        or sha256(specialist_manifest)
        != specialist_result["checkpoint"]["training_manifest"]["sha256"]
        or specialist_result.get("eligible") is not False
        or specialist_result.get("deltas", {}).get(
            "mean_sentence_chrf_pp", 0.0
        )
        < 0.25
        or not any(
            specialist_result.get("new_failures", {}).get(name)
            for name in ("exact", "typed", "negation", "generation")
        )
    ):
        raise SystemExit("v11 specialist identity or v10 evidence differs")
    specialist_metadata = load_json(specialist_manifest)
    if (
        specialist_metadata.get("checkpoint_step") != SPECIALIST_STEP
        or specialist_metadata.get("direction") != "ja-en"
        or specialist_metadata.get("initial_checkpoint", {}).get(
            "model_sha256"
        )
        != v10_contract["parent"]["model"]["sha256"]
        or specialist_metadata.get("license") != "CC-BY-SA-4.0"
    ):
        raise SystemExit("v11 specialist lineage or license differs")

    valid_path = Path(v10_contract["dataset"]["valid"]["path"])
    if sha256(valid_path) != v10_contract["dataset"]["valid"]["sha256"]:
        raise SystemExit("v10 internal validation data differs")
    implementation_paths = [
        root / "scripts/translation/interpolate_marian_checkpoints.py",
        root
        / "scripts/translation/evaluate_canonical_sequence_v10_internal.py",
        root
        / "scripts/translation/evaluate_canonical_sequence_v11_interpolation.py",
        root / "scripts/translation/prepare_elanmt_mlx.py",
        root / "scripts/translation/run_mlx_marian_benchmark.py",
        root / "scripts/translation/marian_mlx.py",
    ]

    result = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": "preregistered-ready-for-interpolation",
        "direction": "ja-en",
        "hypothesis": (
            "Fine-grained weight interpolation can retain part of v10's "
            "generation-quality and long-legal gains while discrete unsafe "
            "outputs snap back to the frozen parent."
        ),
        "interpretation_scope": {
            "internal_split_reused_after_v10": True,
            "exploratory_model_selection_only": True,
            "final_quality_claims_forbidden": True,
            "protected_evaluation_required_for_any_promotion_claim": True,
        },
        "v10_evidence": {
            "contract": record(args.v10_contract),
            "result": record(args.v10_result),
            "specialist_step": SPECIALIST_STEP,
            "specialist_deltas": specialist_result["deltas"],
            "specialist_new_failure_counts": {
                name: len(specialist_result["new_failures"][name])
                for name in ("exact", "typed", "negation", "generation")
            },
            "rationale": (
                "V10 produced a real aggregate and long-document quality signal "
                "but failed strict safety gates. V11 is a frozen, training-free "
                "retention study and does not reinterpret v10 as eligible."
            ),
        },
        "dataset": {
            "suite": (
                "v10 frozen 1,024-row source-disjoint development split; "
                "protected-independent but reused for successor selection"
            ),
            "valid": record(valid_path),
            "rows": v10_contract["dataset"]["counts"]["valid"],
            "manifest": v10_contract["dataset"]["manifest"],
            "uses_protected_outputs": False,
        },
        "parent": {
            "path": str(args.parent_checkpoint),
            "repository": v10_contract["parent"]["repository"],
            "revision": v10_contract["parent"]["revision"],
            "license": v10_contract["parent"]["license"],
            "model": record(parent_model),
            "training_manifest": record(parent_manifest),
        },
        "specialist": {
            "path": str(args.specialist_checkpoint),
            "step": SPECIALIST_STEP,
            "license": specialist_metadata["license"],
            "model": record(specialist_model),
            "training_manifest": record(specialist_manifest),
        },
        "interpolation": {
            "operation": "linear-checkpoint-interpolation",
            "formula": (
                "output = (1 - adapted_weight) * parent "
                "+ adapted_weight * specialist"
            ),
            "adapted_weights": ADAPTED_WEIGHTS,
            "include_prefixes": ["*"],
            "one_grid_only": True,
            "post_result_grid_changes_forbidden": True,
            "no_additional_training": True,
        },
        "generation": {
            "decoding": "greedy",
            "maximum_source_tokens": 192,
            "maximum_target_tokens": 192,
        },
        "selection": {
            "uses_protected_outputs": False,
            "requirements": {
                "mean_sentence_chrf_pp_delta_minimum": 0.25,
                "corpus_chrf_pp_delta_minimum": 0.25,
                "teacher_slice_chrf_pp_delta_minimum": 0.0,
                "long_legal_chrf_pp_delta_minimum": 0.25,
                "worst_stratum_chrf_pp_delta_minimum": -0.50,
                "new_exact_critical_failures_maximum": 0,
                "new_typed_critical_failures_maximum": 0,
                "new_negation_failures_maximum": 0,
                "new_repetition_or_generation_limit_failures_maximum": 0,
            },
            "teacher_gate_change_from_v10": (
                "V10 tested whether supervised updates improved the approved "
                "teacher slice by at least 0.50. V11 instead tests retention "
                "under interpolation, so teacher non-regression is required. "
                "All protected promotion gates remain unchanged."
            ),
            "ordering": (
                "eligible checkpoints only; highest mean sentence chrF++, "
                "then long-legal chrF++, teacher-slice chrF++, and smaller alpha"
            ),
            "stop_if_no_checkpoint_is_eligible": True,
        },
        "next_step_if_internal_gate_passes": {
            "exact_q4_conversion_required": True,
            "bits": 4,
            "group_size": 64,
            "protected_evaluation_required": True,
            "protected_suites": v10_contract[
                "next_step_if_internal_gate_passes"
            ]["protected_suites"],
            "comet22_and_two_family_judge_required_if_decision_relevant": True,
        },
        "protected_promotion_gates": v10_contract[
            "protected_promotion_gates"
        ],
        "implementation": {
            path.name: record(path) for path in implementation_paths
        },
        "stop_rule": (
            "Stop before q4 conversion and protected evaluation unless one "
            "frozen interpolation checkpoint passes every internal generation, "
            "slice, and safety gate. Stop before app integration, fallback "
            "changes, bundle replacement, release, or public upload on any "
            "protected quality, safety, runtime, size, or provenance failure."
        ),
        "exact_q4_conversion_authorized": False,
        "protected_evaluation_authorized": False,
        "bundle_creation_authorized": False,
        "app_change_authorized": False,
        "promotion_authorized": False,
        "public_upload_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": sha256(args.output),
                "status": result["status"],
                "adapted_weights": ADAPTED_WEIGHTS,
                "uses_protected_outputs": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
