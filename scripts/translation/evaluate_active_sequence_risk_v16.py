#!/usr/bin/env python3
"""Evaluate V16 checkpoints before the required dual-Claude semantic audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from evaluate_canonical_constrained_recovery_v15 import (
    aggregate,
    gate,
    public_aggregate,
    sample_extremes,
    suite_decision,
)
from evaluate_canonical_sequence_v10_internal import bootstrap, generation_rows
from train_active_sequence_risk_v16 import (
    evaluate_active_risks,
    load_json,
    load_jsonl,
)
from train_marian_distillation import sha256
from transformers import MarianMTModel, MarianTokenizer

EXPERIMENT = "active-sequence-risk-v16-ja-en"


def display_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": display_path(path, root),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def validate_candidate(
    checkpoint: Path,
    *,
    step: int,
    contract: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = checkpoint / "mimi_training_manifest.json"
    manifest = load_json(manifest_path)
    if (
        manifest.get("experiment") != EXPERIMENT
        or manifest.get("direction") != "ja-en"
        or manifest.get("checkpoint_step") != step
        or manifest.get("contract", {}).get("sha256") != contract["_contract_sha256"]
        or manifest.get("initial_checkpoint", {}).get("model_sha256")
        != contract["initial_checkpoint"]["model"]["sha256"]
        or manifest.get("preservation_checkpoint", {}).get("model_sha256")
        != contract["preservation_checkpoint"]["model"]["sha256"]
        or manifest.get("dataset_manifest", {}).get("sha256")
        != contract["dataset"]["manifest"]["sha256"]
        or manifest.get("hyperparameters") != contract["training"]
        or manifest.get("capacity_change") is not False
        or manifest.get("promotion_eligible") is not False
    ):
        raise SystemExit(f"V16 candidate lineage differs at step {step}")
    return manifest


def risk_deltas(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    fields = ("mean_margin", "preference_accuracy", "active_fraction")
    return {
        "all": {
            field: candidate["all"][field] - baseline["all"][field] for field in fields
        },
        "by_role": {
            role: {
                field: candidate["by_role"][role][field]
                - baseline["by_role"][role][field]
                for field in fields
            }
            for role in ("omission", "repetition")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=2,
        metavar=("STEP", "CHECKPOINT"),
        required=True,
    )
    parser.add_argument(
        "--device",
        choices=("mps", "cpu", "cuda"),
        default="mps",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")
    if args.batch_size < 1 or args.bootstrap_samples < 100:
        raise SystemExit("invalid V16 evaluation size")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    root = Path(__file__).resolve().parents[2]
    contract = load_json(args.contract)
    contract["_contract_sha256"] = sha256(args.contract)
    if (
        contract.get("experiment") != EXPERIMENT
        or contract.get("status") != "preregistered-ready-for-one-arm-training"
        or contract.get("internal_selection", {}).get(
            "selection_uses_protected_outputs"
        )
        is not False
        or contract.get("internal_selection", {}).get(
            "dual_semantic_audit_required_before_internal_pass"
        )
        is not True
        or contract.get("app_change_authorized") is not False
    ):
        raise SystemExit("V16 contract is invalid")
    candidate_paths = {int(step): Path(path) for step, path in args.candidate}
    if sorted(candidate_paths) != contract["internal_selection"]["candidate_steps"]:
        raise SystemExit("V16 candidate steps differ from the contract")

    suite_specs = {
        "fresh_v16": ("fresh_valid", "v16_stratum"),
        "v12_regression": ("v12_regression", "v12_stratum"),
        "v14_regression": ("v14_regression", "v14_stratum"),
        "v15_regression": ("v15_regression", "v15_stratum"),
    }
    suites: dict[str, dict[str, Any]] = {}
    for suite_name, (contract_key, stratum_field) in suite_specs.items():
        item = contract["dataset"][contract_key]
        path = root / item["path"]
        if sha256(path) != item["sha256"]:
            raise SystemExit(f"{suite_name} suite differs")
        rows = load_jsonl(path)
        if len(rows) != contract["dataset"]["counts"][contract_key]:
            raise SystemExit(f"{suite_name} row count differs")
        suites[suite_name] = {
            "path": path,
            "rows": rows,
            "stratum_field": stratum_field,
        }

    active_item = contract["dataset"]["active"]
    active_path = root / active_item["path"]
    if sha256(active_path) != active_item["sha256"]:
        raise SystemExit("V16 active-risk data differs")
    active_rows = load_jsonl(active_path)

    parent = root / contract["preservation_checkpoint"]["path"]
    if (
        sha256(parent / "model.safetensors")
        != contract["preservation_checkpoint"]["model"]["sha256"]
    ):
        raise SystemExit("V16 safe parent hash differs")
    tokenizer = MarianTokenizer.from_pretrained(parent)
    device = torch.device(args.device)
    training = contract["training"]
    baselines = {}
    for suite_name, suite in suites.items():
        generated = generation_rows(
            parent,
            tokenizer,
            suite["rows"],
            device=device,
            batch_size=args.batch_size,
            maximum_source_tokens=training["max_source_tokens"],
            maximum_target_tokens=training["max_target_tokens"],
        )
        baselines[suite_name] = aggregate(
            suite["rows"],
            generated,
            stratum_field=suite["stratum_field"],
        )
    baseline_risks = evaluate_active_risks(
        parent_model := MarianMTModel.from_pretrained(parent).to(device),
        tokenizer,
        active_rows,
        batch_size=args.batch_size,
        target_margin=training["sequence_target_margin"],
        maximum_source_tokens=training["max_source_tokens"],
        maximum_target_tokens=training["max_target_tokens"],
        device=device,
    )
    del parent_model

    requirements = contract["internal_selection"]["requirements"]
    candidates = []
    for step in sorted(candidate_paths):
        checkpoint = candidate_paths[step]
        manifest = validate_candidate(
            checkpoint,
            step=step,
            contract=contract,
        )
        suite_results = {}
        for suite_name, suite in suites.items():
            generated = generation_rows(
                checkpoint,
                tokenizer,
                suite["rows"],
                device=device,
                batch_size=args.batch_size,
                maximum_source_tokens=training["max_source_tokens"],
                maximum_target_tokens=training["max_target_tokens"],
            )
            metrics = aggregate(
                suite["rows"],
                generated,
                stratum_field=suite["stratum_field"],
            )
            decision = suite_decision(baselines[suite_name], metrics)
            suite_results[suite_name] = {
                "metrics": public_aggregate(metrics),
                **decision,
                "samples": sample_extremes(
                    suite["rows"],
                    baselines[suite_name],
                    metrics,
                ),
                "_full_metrics": metrics,
            }

        model = MarianMTModel.from_pretrained(checkpoint).to(device)
        candidate_risks = evaluate_active_risks(
            model,
            tokenizer,
            active_rows,
            batch_size=args.batch_size,
            target_margin=training["sequence_target_margin"],
            maximum_source_tokens=training["max_source_tokens"],
            maximum_target_tokens=training["max_target_tokens"],
            device=device,
        )
        del model
        active_deltas = risk_deltas(baseline_risks, candidate_risks)

        fresh = suite_results["fresh_v16"]
        fresh_rows = suites["fresh_v16"]["rows"]
        fresh_baseline = baselines["fresh_v16"]
        fresh_metrics = fresh["_full_metrics"]
        paired = bootstrap(
            [
                fresh_metrics["sentence_chrf_pp"][str(row["id"])]
                - fresh_baseline["sentence_chrf_pp"][str(row["id"])]
                for row in fresh_rows
            ],
            samples=args.bootstrap_samples,
            seed=20260822 + step,
        )
        v12 = suite_results["v12_regression"]
        v14 = suite_results["v14_regression"]
        v15 = suite_results["v15_regression"]
        gates = [
            gate(
                "fresh-corpus-chrf++",
                fresh["deltas"]["corpus_chrf_pp"],
                minimum=requirements["fresh_corpus_chrf_pp_delta_minimum"],
            ),
            gate(
                "fresh-mean-sentence-chrf++",
                fresh["deltas"]["mean_sentence_chrf_pp"],
                minimum=requirements["fresh_mean_sentence_chrf_pp_delta_minimum"],
            ),
            gate(
                "fresh-long-legal-chrf++",
                fresh["deltas"]["by_stratum"]["legal:long"],
                minimum=requirements["fresh_long_legal_chrf_pp_delta_minimum"],
            ),
            gate(
                "fresh-worst-stratum-chrf++",
                fresh["deltas"]["worst_stratum_chrf_pp"],
                minimum=requirements["fresh_worst_stratum_chrf_pp_delta_minimum"],
            ),
            gate(
                "fresh-paired-bootstrap-lower",
                paired["lower"],
                minimum=requirements["fresh_paired_chrf_pp_90pct_lower_minimum"],
            ),
            gate(
                "v12-mean-sentence-chrf++",
                v12["deltas"]["mean_sentence_chrf_pp"],
                minimum=requirements["v12_mean_sentence_chrf_pp_delta_minimum"],
            ),
            gate(
                "v12-worst-stratum-chrf++",
                v12["deltas"]["worst_stratum_chrf_pp"],
                minimum=requirements["v12_worst_stratum_chrf_pp_delta_minimum"],
            ),
            gate(
                "v14-mean-sentence-chrf++",
                v14["deltas"]["mean_sentence_chrf_pp"],
                minimum=requirements["v14_mean_sentence_chrf_pp_delta_minimum"],
            ),
            gate(
                "v14-worst-stratum-chrf++",
                v14["deltas"]["worst_stratum_chrf_pp"],
                minimum=requirements["v14_worst_stratum_chrf_pp_delta_minimum"],
            ),
            gate(
                "v14-omission-risk-chrf++",
                v14["deltas"]["by_stratum"]["legal:omission-risk"],
                minimum=requirements["v14_omission_risk_chrf_pp_delta_minimum"],
            ),
            gate(
                "v15-mean-sentence-chrf++",
                v15["deltas"]["mean_sentence_chrf_pp"],
                minimum=requirements["v15_mean_sentence_chrf_pp_delta_minimum"],
            ),
            gate(
                "v15-worst-stratum-chrf++",
                v15["deltas"]["worst_stratum_chrf_pp"],
                minimum=requirements["v15_worst_stratum_chrf_pp_delta_minimum"],
            ),
            gate(
                "v15-omission-risk-chrf++",
                v15["deltas"]["by_stratum"]["legal:omission-risk"],
                minimum=requirements["v15_omission_risk_chrf_pp_delta_minimum"],
            ),
            gate(
                "active-all-preference",
                active_deltas["all"]["preference_accuracy"],
                minimum=requirements["active_all_preference_accuracy_delta_minimum"],
            ),
            gate(
                "active-omission-preference",
                active_deltas["by_role"]["omission"]["preference_accuracy"],
                minimum=requirements[
                    "active_omission_preference_accuracy_delta_minimum"
                ],
            ),
            gate(
                "active-repetition-preference",
                active_deltas["by_role"]["repetition"]["preference_accuracy"],
                minimum=requirements[
                    "active_repetition_preference_accuracy_delta_minimum"
                ],
            ),
            gate(
                "active-omission-mean-margin",
                active_deltas["by_role"]["omission"]["mean_margin"],
                minimum=requirements["active_omission_mean_margin_delta_minimum"],
            ),
            gate(
                "active-repetition-mean-margin",
                active_deltas["by_role"]["repetition"]["mean_margin"],
                minimum=requirements["active_repetition_mean_margin_delta_minimum"],
            ),
            gate(
                "active-fraction-reduction",
                active_deltas["all"]["active_fraction"],
                maximum=requirements["active_fraction_delta_maximum"],
            ),
        ]
        for suite_name, suite_result in suite_results.items():
            gates.append(
                gate(
                    f"{suite_name}-new-generation-failures",
                    len(suite_result["new_failures"]["generation"]),
                    maximum=requirements[
                        "new_repetition_or_generation_limit_failures_maximum_per_suite"
                    ],
                )
            )

        semantic_queue = []
        for suite_name, suite_result in suite_results.items():
            for failure_type in ("exact", "typed", "negation"):
                semantic_queue.extend(
                    {
                        "suite": suite_name,
                        "failure_type": failure_type,
                        "case_id": case_id,
                    }
                    for case_id in suite_result["new_failures"][failure_type]
                )
            suite_result.pop("_full_metrics")
        candidates.append(
            {
                "step": step,
                "checkpoint": {
                    "path": display_path(checkpoint, root),
                    "model": record(checkpoint / "model.safetensors", root),
                    "training_manifest": record(
                        checkpoint / "mimi_training_manifest.json",
                        root,
                    ),
                },
                "training_checkpoint_metrics": next(
                    item
                    for item in manifest["training_history"]
                    if item["step"] == step
                ),
                "suites": suite_results,
                "fresh_paired_sentence_chrf_pp_bootstrap": paired,
                "active_risks": candidate_risks,
                "active_risk_deltas_from_safe_parent": active_deltas,
                "pre_semantic_gates": gates,
                "semantic_audit_queue": semantic_queue,
                "semantic_audit_required": bool(semantic_queue),
                "detector_disagreement_is_not_semantic_approval": True,
                "eligible_for_dual_semantic_audit": all(
                    item["passed"] for item in gates
                ),
            }
        )

    eligible = [item for item in candidates if item["eligible_for_dual_semantic_audit"]]
    eligible.sort(
        key=lambda item: (
            item["suites"]["fresh_v16"]["deltas"]["mean_sentence_chrf_pp"],
            item["suites"]["v14_regression"]["deltas"]["by_stratum"][
                "legal:omission-risk"
            ],
            item["active_risk_deltas_from_safe_parent"]["all"]["preference_accuracy"],
        ),
        reverse=True,
    )
    selected = eligible[0] if eligible else None
    result = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": (
            "semantic-audit-required"
            if selected is not None
            else "pre-semantic-gate-rejected"
        ),
        "decision": (
            "Freeze the selected checkpoint and run exact Sonnet 5 plus Opus "
            "5 on its complete detector-disagreement queue; no later stage "
            "is authorized yet."
            if selected is not None
            else (
                "Stop V16 before semantic judging, quantization, COMET, "
                "protected evaluation, app work, release, or upload."
            )
        ),
        "contract": record(args.contract, root),
        "evaluation_implementation": record(Path(__file__).resolve(), root),
        "device": str(device),
        "batch_size": args.batch_size,
        "baselines": {
            suite_name: public_aggregate(value)
            for suite_name, value in baselines.items()
        },
        "baseline_active_risks": baseline_risks,
        "candidates": candidates,
        "selected_for_semantic_audit_step": (
            selected["step"] if selected is not None else None
        ),
        "selected_checkpoint": (
            selected["checkpoint"] if selected is not None else None
        ),
        "semantic_audit_complete": False,
        "internal_gate_passed": False,
        "exact_q4_conversion_authorized": False,
        "comet_authorized": False,
        "protected_evaluation_authorized_after_exact_q4": False,
        "app_change_authorized": False,
        "bundle_replacement_authorized": False,
        "public_upload_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": display_path(args.output, root),
                "sha256": sha256(args.output),
                "status": result["status"],
                "selected_for_semantic_audit_step": result[
                    "selected_for_semantic_audit_step"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
