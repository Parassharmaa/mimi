#!/usr/bin/env python3
"""Evaluate preregistered v12 checkpoints on the fresh internal suite."""

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

EXPERIMENT = "canonical-safety-repair-v12-ja-en"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
) -> dict[str, Any]:
    indexed = {str(row["id"]): row for row in generated}
    expected = {str(row["id"]) for row in rows}
    if set(indexed) != expected:
        raise SystemExit("generation does not cover the exact v12 validation set")
    hypotheses = [str(indexed[str(row["id"])]["hypothesis"]) for row in rows]
    references = [str(row["target"]) for row in rows]
    sentence_scores = {
        str(row["id"]): sentence_score(hypothesis, reference)
        for row, hypothesis, reference in zip(
            rows,
            hypotheses,
            references,
        )
    }
    strata: defaultdict[str, list[str]] = defaultdict(list)
    for row in rows:
        strata[str(row["v12_stratum"])].append(str(row["id"]))
    failures = structure_failures(rows, indexed)

    def slice_metrics(case_ids: list[str]) -> dict[str, Any]:
        return {
            "cases": len(case_ids),
            "mean_sentence_chrf_pp": sum(
                sentence_scores[case_id] for case_id in case_ids
            )
            / len(case_ids),
            "corpus_chrf_pp": float(
                sacrebleu.corpus_chrf(
                    [indexed[case_id]["hypothesis"] for case_id in case_ids],
                    [
                        [
                            str(
                                next(
                                    row["target"]
                                    for row in rows
                                    if str(row["id"]) == case_id
                                )
                            )
                            for case_id in case_ids
                        ]
                    ],
                    word_order=2,
                ).score
            ),
        }

    general_ids = [
        case_id
        for stratum, case_ids in strata.items()
        if stratum.startswith("general:")
        for case_id in case_ids
    ]
    legal_ids = [
        case_id
        for stratum, case_ids in strata.items()
        if stratum.startswith("legal:")
        for case_id in case_ids
    ]
    return {
        "corpus_chrf_pp": float(
            sacrebleu.corpus_chrf(
                hypotheses,
                [references],
                word_order=2,
            ).score
        ),
        "corpus_bleu": float(
            sacrebleu.corpus_bleu(
                hypotheses,
                [references],
                tokenize="13a",
            ).score
        ),
        "mean_sentence_chrf_pp": sum(sentence_scores.values()) / len(sentence_scores),
        "sentence_chrf_pp": sentence_scores,
        "strata": {
            stratum: slice_metrics(case_ids)
            for stratum, case_ids in sorted(strata.items())
        },
        "groups": {
            "fresh-general": slice_metrics(general_ids),
            "fresh-legal": slice_metrics(legal_ids),
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
        or manifest.get("negative_dataset", {}).get("manifest_sha256")
        != contract["negative_dataset"]["manifest"]["sha256"]
        or manifest.get("hyperparameters") != contract["training"]
        or manifest.get("promotion_eligible") is not False
    ):
        raise SystemExit(f"v12 candidate lineage differs at step {step}")
    return manifest


def candidate_decision(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    paired_bootstrap: dict[str, Any],
    negative_probability_delta: float,
    requirements: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_stratum = {
        stratum: candidate["strata"][stratum]["mean_sentence_chrf_pp"]
        - baseline["strata"][stratum]["mean_sentence_chrf_pp"]
        for stratum in baseline["strata"]
    }
    deltas = {
        "corpus_chrf_pp": candidate["corpus_chrf_pp"] - baseline["corpus_chrf_pp"],
        "corpus_bleu": candidate["corpus_bleu"] - baseline["corpus_bleu"],
        "mean_sentence_chrf_pp": candidate["mean_sentence_chrf_pp"]
        - baseline["mean_sentence_chrf_pp"],
        "fresh_general_chrf_pp": candidate["groups"]["fresh-general"][
            "mean_sentence_chrf_pp"
        ]
        - baseline["groups"]["fresh-general"]["mean_sentence_chrf_pp"],
        "long_legal_chrf_pp": candidate["strata"]["legal:long"]["mean_sentence_chrf_pp"]
        - baseline["strata"]["legal:long"]["mean_sentence_chrf_pp"],
        "worst_stratum_chrf_pp": min(by_stratum.values()),
        "by_stratum": by_stratum,
        "negative_validation_rejected_probability": negative_probability_delta,
    }
    new_failures = {
        name: sorted(set(candidate["failures"][name]) - set(baseline["failures"][name]))
        for name in ("exact", "typed", "negation", "generation")
    }
    gates = [
        {
            "name": "corpus-chrf++",
            "passed": deltas["corpus_chrf_pp"]
            >= requirements["corpus_chrf_pp_delta_minimum"],
            "actual": deltas["corpus_chrf_pp"],
            "minimum": requirements["corpus_chrf_pp_delta_minimum"],
        },
        {
            "name": "mean-sentence-chrf++",
            "passed": deltas["mean_sentence_chrf_pp"]
            >= requirements["mean_sentence_chrf_pp_delta_minimum"],
            "actual": deltas["mean_sentence_chrf_pp"],
            "minimum": requirements["mean_sentence_chrf_pp_delta_minimum"],
        },
        {
            "name": "fresh-general-chrf++",
            "passed": deltas["fresh_general_chrf_pp"]
            >= requirements["fresh_general_chrf_pp_delta_minimum"],
            "actual": deltas["fresh_general_chrf_pp"],
            "minimum": requirements["fresh_general_chrf_pp_delta_minimum"],
        },
        {
            "name": "long-legal-chrf++",
            "passed": deltas["long_legal_chrf_pp"]
            >= requirements["long_legal_chrf_pp_delta_minimum"],
            "actual": deltas["long_legal_chrf_pp"],
            "minimum": requirements["long_legal_chrf_pp_delta_minimum"],
        },
        {
            "name": "worst-stratum-chrf++",
            "passed": deltas["worst_stratum_chrf_pp"]
            >= requirements["worst_stratum_chrf_pp_delta_minimum"],
            "actual": deltas["worst_stratum_chrf_pp"],
            "minimum": requirements["worst_stratum_chrf_pp_delta_minimum"],
        },
        {
            "name": "paired-bootstrap-lower",
            "passed": paired_bootstrap["lower"]
            >= requirements["paired_chrf_pp_90pct_lower_minimum"],
            "actual": paired_bootstrap["lower"],
            "minimum": requirements["paired_chrf_pp_90pct_lower_minimum"],
        },
        {
            "name": "negative-validation-rejected-probability",
            "passed": negative_probability_delta
            <= requirements["negative_validation_rejected_probability_delta_maximum"],
            "actual": negative_probability_delta,
            "maximum": requirements[
                "negative_validation_rejected_probability_delta_maximum"
            ],
        },
        *[
            {
                "name": f"new-{name}-failures",
                "passed": len(new_failures[name]) <= requirements[requirement],
                "actual": len(new_failures[name]),
                "maximum": requirements[requirement],
                "case_ids": new_failures[name],
            }
            for name, requirement in (
                ("exact", "new_exact_critical_failures_maximum"),
                ("typed", "new_typed_critical_failures_maximum"),
                ("negation", "new_negation_failures_maximum"),
                (
                    "generation",
                    "new_repetition_or_generation_limit_failures_maximum",
                ),
            )
        ],
    ]
    return gates, {
        "deltas": deltas,
        "new_failures": new_failures,
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
        or contract.get("app_change_authorized") is not False
    ):
        raise SystemExit("v12 contract is invalid")
    candidate_paths = {int(step): Path(path) for step, path in args.candidate}
    if sorted(candidate_paths) != contract["internal_selection"]["candidate_steps"]:
        raise SystemExit("v12 candidate steps differ from the contract")

    valid_path = root / contract["dataset"]["valid"]["path"]
    if sha256(valid_path) != contract["dataset"]["valid"]["sha256"]:
        raise SystemExit("v12 validation data hash differs")
    rows = load_jsonl(valid_path)
    if len(rows) != contract["dataset"]["counts"]["valid"]:
        raise SystemExit("v12 validation row count differs")
    parent = root / contract["preservation_checkpoint"]["path"]
    if (
        sha256(parent / "model.safetensors")
        != contract["preservation_checkpoint"]["model"]["sha256"]
    ):
        raise SystemExit("v12 safe parent hash differs")

    tokenizer = MarianTokenizer.from_pretrained(parent)
    device = torch.device(args.device)
    training = contract["training"]
    baseline_generated = generation_rows(
        parent,
        tokenizer,
        rows,
        device=device,
        batch_size=args.batch_size,
        maximum_source_tokens=training["max_source_tokens"],
        maximum_target_tokens=training["max_target_tokens"],
    )
    baseline = aggregate(rows, baseline_generated)

    candidates = []
    for step in sorted(candidate_paths):
        checkpoint = candidate_paths[step]
        manifest = validate_candidate(
            checkpoint,
            step=step,
            contract=contract,
        )
        generated = generation_rows(
            checkpoint,
            tokenizer,
            rows,
            device=device,
            batch_size=args.batch_size,
            maximum_source_tokens=training["max_source_tokens"],
            maximum_target_tokens=training["max_target_tokens"],
        )
        metrics = aggregate(rows, generated)
        sentence_deltas = [
            metrics["sentence_chrf_pp"][str(row["id"])]
            - baseline["sentence_chrf_pp"][str(row["id"])]
            for row in rows
        ]
        paired = bootstrap(
            sentence_deltas,
            samples=args.bootstrap_samples,
            seed=20260804 + step,
        )
        checkpoint_negative = manifest["checkpoint_metrics"]["negative_space"][
            "mean_rejected_token_probability"
        ]
        initial_negative = manifest["history"][0]["negative_space"][
            "mean_rejected_token_probability"
        ]
        gates, decision = candidate_decision(
            baseline,
            metrics,
            paired_bootstrap=paired,
            negative_probability_delta=checkpoint_negative - initial_negative,
            requirements=contract["internal_selection"]["requirements"],
        )
        ranked = sorted(
            (
                metrics["sentence_chrf_pp"][str(row["id"])]
                - baseline["sentence_chrf_pp"][str(row["id"])],
                str(row["id"]),
            )
            for row in rows
        )

        def sample(
            delta: float,
            case_id: str,
            candidate_metrics: dict[str, Any],
        ) -> dict[str, Any]:
            row = next(row for row in rows if str(row["id"]) == case_id)
            return {
                "delta": delta,
                "id": case_id,
                "stratum": row["v12_stratum"],
                "source": row["source"],
                "reference": row["target"],
                "baseline": baseline["outputs"][case_id]["hypothesis"],
                "candidate": candidate_metrics["outputs"][case_id]["hypothesis"],
            }

        candidates.append(
            {
                "step": step,
                "checkpoint": {
                    "path": display_path(checkpoint, root),
                    "model": record(
                        checkpoint / "model.safetensors",
                        root,
                    ),
                    "training_manifest": record(
                        checkpoint / "mimi_training_manifest.json",
                        root,
                    ),
                },
                "training_checkpoint_metrics": manifest["checkpoint_metrics"],
                "metrics": public_aggregate(metrics),
                **decision,
                "paired_sentence_chrf_pp_bootstrap": paired,
                "gates": gates,
                "eligible": all(gate["passed"] for gate in gates),
                "samples": {
                    "largest_regressions": [
                        sample(delta, case_id, metrics)
                        for delta, case_id in ranked[:10]
                    ],
                    "largest_improvements": [
                        sample(delta, case_id, metrics)
                        for delta, case_id in reversed(ranked[-10:])
                    ],
                },
            }
        )

    eligible = [item for item in candidates if item["eligible"]]
    eligible.sort(
        key=lambda item: (
            item["deltas"]["mean_sentence_chrf_pp"],
            item["deltas"]["long_legal_chrf_pp"],
            item["deltas"]["fresh_general_chrf_pp"],
        ),
        reverse=True,
    )
    selected = eligible[0] if eligible else None
    result = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": (
            "internal-gate-passed" if selected is not None else "internal-gate-rejected"
        ),
        "decision": (
            "Authorize exact-q4 conversion and protected evaluation only."
            if selected is not None
            else (
                "Stop v12 before quantization, protected evaluation, app "
                "work, bundle replacement, release, or public upload."
            )
        ),
        "contract": record(args.contract, root),
        "evaluation_implementation": record(
            Path(__file__).resolve(),
            root,
        ),
        "device": str(device),
        "batch_size": args.batch_size,
        "baseline": {
            "checkpoint": contract["preservation_checkpoint"],
            "metrics": public_aggregate(baseline),
        },
        "candidates": candidates,
        "selected_step": selected["step"] if selected is not None else None,
        "selected_checkpoint": (
            selected["checkpoint"] if selected is not None else None
        ),
        "exact_q4_conversion_authorized": selected is not None,
        "protected_evaluation_authorized_after_exact_q4": (selected is not None),
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
                "selected_step": result["selected_step"],
                "candidates": [
                    {
                        "step": item["step"],
                        "eligible": item["eligible"],
                        "deltas": item["deltas"],
                        "failed_gates": [
                            gate["name"] for gate in item["gates"] if not gate["passed"]
                        ],
                    }
                    for item in candidates
                ],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
