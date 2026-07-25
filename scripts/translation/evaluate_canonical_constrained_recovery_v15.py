#!/usr/bin/env python3
"""Evaluate v15 checkpoints before the required dual-Claude semantic audit."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import sacrebleu
import torch
from evaluate_canonical_sequence_v10_internal import (
    bootstrap,
    generation_rows,
    sentence_score,
    structure_failures,
)
from train_marian_distillation import sha256
from transformers import MarianTokenizer

EXPERIMENT = "canonical-constrained-recovery-v15-ja-en"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    values = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not values:
        raise SystemExit(f"expected non-empty JSONL: {path}")
    return values


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


def aggregate(
    rows: list[dict[str, Any]],
    generated: list[dict[str, Any]],
    *,
    stratum_field: str,
) -> dict[str, Any]:
    indexed = {str(row["id"]): row for row in generated}
    expected = {str(row["id"]) for row in rows}
    if set(indexed) != expected:
        raise SystemExit("generation does not cover the exact bound validation set")
    hypotheses = [str(indexed[str(row["id"])]["hypothesis"]) for row in rows]
    references = [str(row["target"]) for row in rows]
    scores = {
        str(row["id"]): sentence_score(hypothesis, reference)
        for row, hypothesis, reference in zip(rows, hypotheses, references)
    }
    strata: defaultdict[str, list[str]] = defaultdict(list)
    row_by_id = {str(row["id"]): row for row in rows}
    for row in rows:
        strata[str(row[stratum_field])].append(str(row["id"]))
    failures = structure_failures(rows, indexed)

    def slice_metrics(case_ids: list[str]) -> dict[str, Any]:
        return {
            "cases": len(case_ids),
            "mean_sentence_chrf_pp": sum(scores[case_id] for case_id in case_ids)
            / len(case_ids),
            "corpus_chrf_pp": float(
                sacrebleu.corpus_chrf(
                    [indexed[case_id]["hypothesis"] for case_id in case_ids],
                    [[str(row_by_id[case_id]["target"]) for case_id in case_ids]],
                    word_order=2,
                ).score
            ),
        }

    return {
        "corpus_chrf_pp": float(
            sacrebleu.corpus_chrf(hypotheses, [references], word_order=2).score
        ),
        "corpus_bleu": float(
            sacrebleu.corpus_bleu(hypotheses, [references], tokenize="13a").score
        ),
        "mean_sentence_chrf_pp": sum(scores.values()) / len(scores),
        "sentence_chrf_pp": scores,
        "strata": {
            stratum: slice_metrics(case_ids)
            for stratum, case_ids in sorted(strata.items())
        },
        "failures": {name: sorted(values) for name, values in failures.items()},
        "outputs": indexed,
    }


def public_aggregate(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key not in {"sentence_chrf_pp", "outputs"}
    }


def suite_decision(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    by_stratum = {
        stratum: candidate["strata"][stratum]["mean_sentence_chrf_pp"]
        - baseline["strata"][stratum]["mean_sentence_chrf_pp"]
        for stratum in baseline["strata"]
    }
    new_failures = {
        name: sorted(set(candidate["failures"][name]) - set(baseline["failures"][name]))
        for name in ("exact", "typed", "negation", "generation")
    }
    return {
        "deltas": {
            "corpus_chrf_pp": candidate["corpus_chrf_pp"] - baseline["corpus_chrf_pp"],
            "corpus_bleu": candidate["corpus_bleu"] - baseline["corpus_bleu"],
            "mean_sentence_chrf_pp": candidate["mean_sentence_chrf_pp"]
            - baseline["mean_sentence_chrf_pp"],
            "worst_stratum_chrf_pp": min(by_stratum.values()),
            "by_stratum": by_stratum,
        },
        "new_failures": new_failures,
    }


def gate(
    name: str,
    actual: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> dict[str, Any]:
    if (minimum is None) == (maximum is None):
        raise ValueError("gate requires exactly one bound")
    passed = actual >= minimum if minimum is not None else actual <= maximum
    result: dict[str, Any] = {
        "name": name,
        "passed": passed,
        "actual": actual,
    }
    result["minimum" if minimum is not None else "maximum"] = (
        minimum if minimum is not None else maximum
    )
    return result


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
        or manifest.get("contrastive_examples_manifest", {}).get("sha256")
        != contract["contrastive_examples"]["manifest"]["sha256"]
        or manifest.get("hyperparameters") != contract["training"]
        or manifest.get("promotion_eligible") is not False
    ):
        raise SystemExit(f"v15 candidate lineage differs at step {step}")
    return manifest


def sample_extremes(
    rows: list[dict[str, Any]],
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    row_by_id = {str(row["id"]): row for row in rows}
    ranked = sorted(
        (
            candidate["sentence_chrf_pp"][case_id]
            - baseline["sentence_chrf_pp"][case_id],
            case_id,
        )
        for case_id in baseline["sentence_chrf_pp"]
    )

    def make(delta: float, case_id: str) -> dict[str, Any]:
        row = row_by_id[case_id]
        return {
            "delta": delta,
            "id": case_id,
            "source": row["source"],
            "reference": row["target"],
            "baseline": baseline["outputs"][case_id]["hypothesis"],
            "candidate": candidate["outputs"][case_id]["hypothesis"],
        }

    return {
        "largest_regressions": [make(delta, case_id) for delta, case_id in ranked[:5]],
        "largest_improvements": [
            make(delta, case_id) for delta, case_id in reversed(ranked[-5:])
        ],
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
        raise SystemExit("invalid evaluation size")
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
        raise SystemExit("v15 contract is invalid")
    candidate_paths = {int(step): Path(path) for step, path in args.candidate}
    if sorted(candidate_paths) != contract["internal_selection"]["candidate_steps"]:
        raise SystemExit("v15 candidate steps differ from the contract")

    suite_specs = {
        "fresh_v15": ("fresh_valid", "v15_stratum"),
        "v12_regression": ("v12_regression", "v12_stratum"),
        "v14_regression": ("v14_regression", "v14_stratum"),
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

    parent = root / contract["preservation_checkpoint"]["path"]
    if (
        sha256(parent / "model.safetensors")
        != contract["preservation_checkpoint"]["model"]["sha256"]
    ):
        raise SystemExit("v15 safe parent hash differs")
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

    requirements = contract["internal_selection"]["requirements"]
    candidates = []
    for step in sorted(candidate_paths):
        checkpoint = candidate_paths[step]
        manifest = validate_candidate(checkpoint, step=step, contract=contract)
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

        fresh = suite_results["fresh_v15"]
        fresh_rows = suites["fresh_v15"]["rows"]
        fresh_baseline = baselines["fresh_v15"]
        fresh_metrics = fresh["_full_metrics"]
        sentence_deltas = [
            fresh_metrics["sentence_chrf_pp"][str(row["id"])]
            - fresh_baseline["sentence_chrf_pp"][str(row["id"])]
            for row in fresh_rows
        ]
        paired = bootstrap(
            sentence_deltas,
            samples=args.bootstrap_samples,
            seed=20260818 + step,
        )
        initial_contrasts = manifest["history"][0]["contrasts"]
        contrasts = manifest["checkpoint_metrics"]["contrasts"]
        contrast_deltas = {
            key: contrasts[key] - initial_contrasts[key]
            for key in (
                "recovery_preference_accuracy",
                "omission_preference_accuracy",
                "clean_over_recovery_accuracy",
                "mean_rejected_token_probability",
            )
        }
        v12 = suite_results["v12_regression"]
        v14 = suite_results["v14_regression"]
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
                "recovery-preference-improvement",
                contrast_deltas["recovery_preference_accuracy"],
                minimum=requirements["recovery_preference_accuracy_delta_minimum"],
            ),
            gate(
                "omission-preference-improvement",
                contrast_deltas["omission_preference_accuracy"],
                minimum=requirements["omission_preference_accuracy_delta_minimum"],
            ),
            gate(
                "clean-over-recovery-retention",
                contrast_deltas["clean_over_recovery_accuracy"],
                minimum=requirements["clean_over_recovery_accuracy_delta_minimum"],
            ),
            gate(
                "recovery-rejected-probability",
                contrast_deltas["mean_rejected_token_probability"],
                maximum=requirements["recovery_rejected_probability_delta_maximum"],
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
                "training_checkpoint_metrics": manifest["checkpoint_metrics"],
                "suites": suite_results,
                "fresh_paired_sentence_chrf_pp_bootstrap": paired,
                "contrast_deltas_from_initial": contrast_deltas,
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
            item["suites"]["fresh_v15"]["deltas"]["mean_sentence_chrf_pp"],
            item["suites"]["v14_regression"]["deltas"]["by_stratum"][
                "legal:omission-risk"
            ],
            item["contrast_deltas_from_initial"]["recovery_preference_accuracy"],
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
                "Stop v15 before semantic judging, quantization, COMET, "
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
