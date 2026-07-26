#!/usr/bin/env python3
"""Select a protected-independent parent/v10 interpolation checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import MarianTokenizer

from evaluate_canonical_sequence_v10_internal import (
    aggregate,
    bootstrap,
    generation_rows,
    load_json,
    load_jsonl,
    public_aggregate,
    record,
    sha256,
)


EXPERIMENT = "canonical-sequence-v11-ja-en-parent-interpolation"


def validate_interpolation(
    path: Path,
    *,
    alpha: float,
    contract: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = path / "mimi_checkpoint_interpolation_manifest.json"
    manifest = load_json(manifest_path)
    if (
        manifest.get("operation") != "linear-checkpoint-interpolation"
        or float(manifest.get("adapted_weight", -1)) != alpha
        or manifest.get("include_prefixes") != ["*"]
        or manifest.get("parent", {}).get("model_sha256")
        != contract["parent"]["model"]["sha256"]
        or manifest.get("adapted", {}).get("model_sha256")
        != contract["specialist"]["model"]["sha256"]
        or manifest.get("output", {}).get("model_sha256")
        != sha256(path / "model.safetensors")
    ):
        raise SystemExit(f"interpolation manifest differs for alpha {alpha}")
    training_manifest = path / "mimi_training_manifest.json"
    if sha256(training_manifest) != contract["specialist"][
        "training_manifest"
    ]["sha256"]:
        raise SystemExit(f"copied v10 training manifest differs for alpha {alpha}")
    return {
        "path": str(path),
        "alpha": alpha,
        "model": record(path / "model.safetensors"),
        "interpolation_manifest": record(manifest_path),
        "training_manifest": record(training_manifest),
    }


def decision(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    requirements: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    deltas = {
        "corpus_chrf_pp": candidate["corpus_chrf_pp"]
        - baseline["corpus_chrf_pp"],
        "mean_sentence_chrf_pp": candidate["mean_sentence_chrf_pp"]
        - baseline["mean_sentence_chrf_pp"],
        "teacher_slice_chrf_pp": candidate["strata"]["teacher"][
            "mean_sentence_chrf_pp"
        ]
        - baseline["strata"]["teacher"]["mean_sentence_chrf_pp"],
        "long_legal_chrf_pp": candidate["strata"]["legal:long"][
            "mean_sentence_chrf_pp"
        ]
        - baseline["strata"]["legal:long"]["mean_sentence_chrf_pp"],
        "by_stratum": {
            stratum: candidate["strata"][stratum]["mean_sentence_chrf_pp"]
            - baseline["strata"][stratum]["mean_sentence_chrf_pp"]
            for stratum in baseline["strata"]
        },
    }
    deltas["worst_stratum_chrf_pp"] = min(
        deltas["by_stratum"].values()
    )
    new_failures = {
        name: sorted(
            set(candidate["failures"][name])
            - set(baseline["failures"][name])
        )
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
            "minimum": requirements[
                "mean_sentence_chrf_pp_delta_minimum"
            ],
        },
        {
            "name": "teacher-slice-non-regression",
            "passed": deltas["teacher_slice_chrf_pp"]
            >= requirements["teacher_slice_chrf_pp_delta_minimum"],
            "actual": deltas["teacher_slice_chrf_pp"],
            "minimum": requirements[
                "teacher_slice_chrf_pp_delta_minimum"
            ],
        },
        {
            "name": "long-legal-chrf++",
            "passed": deltas["long_legal_chrf_pp"]
            >= requirements["long_legal_chrf_pp_delta_minimum"],
            "actual": deltas["long_legal_chrf_pp"],
            "minimum": requirements[
                "long_legal_chrf_pp_delta_minimum"
            ],
        },
        {
            "name": "worst-stratum-chrf++",
            "passed": deltas["worst_stratum_chrf_pp"]
            >= requirements["worst_stratum_chrf_pp_delta_minimum"],
            "actual": deltas["worst_stratum_chrf_pp"],
            "minimum": requirements[
                "worst_stratum_chrf_pp_delta_minimum"
            ],
        },
        *[
            {
                "name": f"new-{name}-failures",
                "passed": len(new_failures[name])
                <= requirements[requirement],
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
    return gates, {"deltas": deltas, "new_failures": new_failures}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=2,
        metavar=("ALPHA", "CHECKPOINT"),
        required=True,
    )
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    contract = load_json(args.contract)
    if (
        contract.get("experiment") != EXPERIMENT
        or contract.get("status")
        != "preregistered-ready-for-interpolation"
        or contract.get("selection", {}).get("uses_protected_outputs")
        is not False
        or contract.get("app_change_authorized") is not False
    ):
        raise SystemExit("v11 contract is invalid")
    candidates = {float(alpha): Path(path) for alpha, path in args.candidate}
    if sorted(candidates) != contract["interpolation"]["adapted_weights"]:
        raise SystemExit("v11 interpolation alpha grid differs")

    valid_path = Path(contract["dataset"]["valid"]["path"])
    if sha256(valid_path) != contract["dataset"]["valid"]["sha256"]:
        raise SystemExit("v11 validation data hash differs")
    rows = load_jsonl(valid_path)
    parent = Path(contract["parent"]["path"])
    specialist = Path(contract["specialist"]["path"])
    for path, expected in (
        (parent / "model.safetensors", contract["parent"]["model"]["sha256"]),
        (
            specialist / "model.safetensors",
            contract["specialist"]["model"]["sha256"],
        ),
    ):
        if sha256(path) != expected:
            raise SystemExit(f"v11 bound model differs: {path}")
    tokenizer = MarianTokenizer.from_pretrained(parent)
    device = torch.device(args.device)
    generation = contract["generation"]
    baseline = aggregate(
        rows,
        generation_rows(
            parent,
            tokenizer,
            rows,
            device=device,
            batch_size=args.batch_size,
            maximum_source_tokens=generation["maximum_source_tokens"],
            maximum_target_tokens=generation["maximum_target_tokens"],
        ),
    )

    evaluated = []
    for alpha, path in sorted(candidates.items()):
        identity = validate_interpolation(
            path,
            alpha=alpha,
            contract=contract,
        )
        metrics = aggregate(
            rows,
            generation_rows(
                path,
                tokenizer,
                rows,
                device=device,
                batch_size=args.batch_size,
                maximum_source_tokens=generation["maximum_source_tokens"],
                maximum_target_tokens=generation["maximum_target_tokens"],
            ),
        )
        gates, details = decision(
            baseline,
            metrics,
            contract["selection"]["requirements"],
        )
        paired = [
            metrics["sentence_chrf_pp"][str(row["id"])]
            - baseline["sentence_chrf_pp"][str(row["id"])]
            for row in rows
        ]
        evaluated.append(
            {
                "alpha": alpha,
                "checkpoint": identity,
                "metrics": public_aggregate(metrics),
                **details,
                "paired_sentence_chrf_pp_bootstrap": bootstrap(
                    paired,
                    samples=args.bootstrap_samples,
                    seed=20261100 + round(alpha * 10_000),
                ),
                "gates": gates,
                "eligible": all(gate["passed"] for gate in gates),
            }
        )

    eligible = [item for item in evaluated if item["eligible"]]
    eligible.sort(
        key=lambda item: (
            item["deltas"]["mean_sentence_chrf_pp"],
            item["deltas"]["long_legal_chrf_pp"],
            item["deltas"]["teacher_slice_chrf_pp"],
            -item["alpha"],
        ),
        reverse=True,
    )
    selected = eligible[0] if eligible else None
    result = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": (
            "internal-gate-passed"
            if selected is not None
            else "internal-gate-rejected"
        ),
        "decision": (
            "Authorize exact-q4 conversion and protected evaluation only."
            if selected is not None
            else (
                "Stop v11 before q4, protected evaluation, app changes, "
                "bundle replacement, or public upload."
            )
        ),
        "contract": record(args.contract),
        "evaluation_implementation": record(Path(__file__).resolve()),
        "baseline": {
            "checkpoint": contract["parent"],
            "metrics": public_aggregate(baseline),
        },
        "candidates": evaluated,
        "selected_alpha": selected["alpha"] if selected is not None else None,
        "selected_checkpoint": (
            selected["checkpoint"] if selected is not None else None
        ),
        "exact_q4_conversion_authorized": selected is not None,
        "protected_evaluation_authorized_after_exact_q4": selected
        is not None,
        "app_change_authorized": False,
        "bundle_creation_authorized": False,
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
                "selected_alpha": result["selected_alpha"],
                "candidates": [
                    {
                        "alpha": item["alpha"],
                        "eligible": item["eligible"],
                        "deltas": item["deltas"],
                        "new_failure_counts": {
                            name: len(values)
                            for name, values in item["new_failures"].items()
                        },
                        "failed_gates": [
                            gate["name"]
                            for gate in item["gates"]
                            if not gate["passed"]
                        ],
                    }
                    for item in evaluated
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
