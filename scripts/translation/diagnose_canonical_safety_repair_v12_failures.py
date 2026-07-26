#!/usr/bin/env python3
"""Extract post-result v12 safety counterexamples without changing selection."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from evaluate_canonical_sequence_v10_internal import (
    generation_rows,
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


def flags_by_case(
    rows: list[dict[str, Any]],
    generated: list[dict[str, Any]],
) -> dict[str, list[str]]:
    failures = structure_failures(
        rows,
        {str(row["id"]): row for row in generated},
    )
    output = {str(row["id"]): [] for row in rows}
    for failure_type, case_ids in failures.items():
        for case_id in case_ids:
            output[case_id].append(failure_type)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--device",
        choices=("mps", "cpu", "cuda"),
        default="mps",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    root = Path(__file__).resolve().parents[2]
    result = load_json(args.result)
    if (
        result.get("experiment") != EXPERIMENT
        or result.get("status") != "internal-gate-rejected"
        or result.get("selected_step") is not None
        or result.get("exact_q4_conversion_authorized") is not False
    ):
        raise SystemExit("v12 result is not a stopped internal rejection")
    contract_path = root / result["contract"]["path"]
    if (
        sha256(contract_path) != result["contract"]["sha256"]
        or sha256(args.result)
        != "0db46aee124cb3d9e61898630349e61cfa2a1fcecc5d04f591d3f5c5a2a9dcd1"
    ):
        raise SystemExit("v12 result or contract bytes differ")
    contract = load_json(contract_path)
    valid_path = root / contract["dataset"]["valid"]["path"]
    if sha256(valid_path) != contract["dataset"]["valid"]["sha256"]:
        raise SystemExit("v12 diagnostic validation bytes differ")
    all_rows = {str(row["id"]): row for row in load_jsonl(valid_path)}

    requested_types: dict[int, dict[str, set[str]]] = {}
    union_ids: set[str] = set()
    for candidate in result["candidates"]:
        step = int(candidate["step"])
        requested_types[step] = {}
        for failure_type, case_ids in candidate["new_failures"].items():
            values = set(case_ids)
            requested_types[step][failure_type] = values
            union_ids.update(values)
    rows = [all_rows[case_id] for case_id in sorted(union_ids)]

    baseline_path = root / contract["preservation_checkpoint"]["path"]
    tokenizer = MarianTokenizer.from_pretrained(baseline_path)
    device = torch.device(args.device)
    maximum_source_tokens = contract["training"]["max_source_tokens"]
    maximum_target_tokens = contract["training"]["max_target_tokens"]
    model_paths = {
        0: baseline_path,
        **{
            int(candidate["step"]): root / candidate["checkpoint"]["path"]
            for candidate in result["candidates"]
        },
    }
    generated_by_model: dict[int, dict[str, dict[str, Any]]] = {}
    flags_by_model: dict[int, dict[str, list[str]]] = {}
    model_records = {}
    for step, path in model_paths.items():
        expected_hash = (
            contract["preservation_checkpoint"]["model"]["sha256"]
            if step == 0
            else next(
                candidate["checkpoint"]["model"]["sha256"]
                for candidate in result["candidates"]
                if int(candidate["step"]) == step
            )
        )
        if sha256(path / "model.safetensors") != expected_hash:
            raise SystemExit(f"v12 diagnostic checkpoint differs: {step}")
        generated = generation_rows(
            path,
            tokenizer,
            rows,
            device=device,
            batch_size=args.batch_size,
            maximum_source_tokens=maximum_source_tokens,
            maximum_target_tokens=maximum_target_tokens,
        )
        generated_by_model[step] = {str(row["id"]): row for row in generated}
        flags_by_model[step] = flags_by_case(rows, generated)
        model_records[str(step)] = record(path / "model.safetensors", root)

    cases = []
    for row in rows:
        case_id = str(row["id"])
        cases.append(
            {
                "id": case_id,
                "stratum": row["v12_stratum"],
                "source": row["source"],
                "reference": row["target"],
                "baseline": {
                    "hypothesis": generated_by_model[0][case_id]["hypothesis"],
                    "failure_types": flags_by_model[0][case_id],
                },
                "candidates": {
                    str(step): {
                        "hypothesis": generated_by_model[step][case_id]["hypothesis"],
                        "failure_types": flags_by_model[step][case_id],
                        "registered_new_failure_types": sorted(
                            failure_type
                            for failure_type, case_ids in failures.items()
                            if case_id in case_ids
                        ),
                    }
                    for step, failures in requested_types.items()
                },
            }
        )
    result_counts = {
        str(candidate["step"]): {
            "absolute": {
                name: len(case_ids)
                for name, case_ids in candidate["metrics"]["failures"].items()
            },
            "new": {
                name: len(case_ids)
                for name, case_ids in candidate["new_failures"].items()
            },
            "resolved": {
                name: len(
                    set(result["baseline"]["metrics"]["failures"][name])
                    - set(candidate["metrics"]["failures"][name])
                )
                for name in ("exact", "typed", "negation", "generation")
            },
        }
        for candidate in result["candidates"]
    }
    payload = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": "post-result-diagnostic-only",
        "selection_changed": False,
        "promotion_authorized": False,
        "result": record(args.result, root),
        "contract": record(contract_path, root),
        "models": model_records,
        "case_count": len(cases),
        "registered_new_failure_case_count_by_step": {
            str(step): len(set().union(*failures.values()))
            for step, failures in requested_types.items()
        },
        "failure_counts": result_counts,
        "new_failure_strata": {
            str(step): dict(
                sorted(
                    Counter(
                        all_rows[case_id]["v12_stratum"]
                        for case_ids in failures.values()
                        for case_id in case_ids
                    ).items()
                )
            )
            for step, failures in requested_types.items()
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": display_path(args.output, root),
                "sha256": sha256(args.output),
                "case_count": len(cases),
                "failure_counts": result_counts,
                "new_failure_strata": payload["new_failure_strata"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
