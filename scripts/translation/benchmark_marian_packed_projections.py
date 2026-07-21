#!/usr/bin/env python3
"""Microbenchmark exact concatenated q4 Marian attention projections."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx

from marian_mlx import load_model, pack_linear_projections


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantized_projection(value: mx.array, projection) -> mx.array:
    return mx.quantized_matmul(
        value,
        projection.weight,
        scales=projection.scales,
        biases=projection.biases,
        transpose=True,
        group_size=projection.group_size,
        bits=projection.bits,
        mode=projection.mode,
    ) + projection.bias


def elapsed(operation, iterations: int) -> float:
    for _ in range(20):
        mx.eval(*operation())
    mx.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        mx.eval(*operation())
    mx.synchronize()
    return (time.perf_counter() - started) / iterations


def percent_improvement(control: float, candidate: float) -> float:
    return (1.0 - candidate / control) * 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--blocks", type=int, default=7)
    parser.add_argument("--length", type=int, action="append")
    parser.add_argument("--minimum-improvement-percent", type=float, default=10.0)
    args = parser.parse_args()
    lengths = args.length or [1, 8, 16, 32, 64]
    if (
        args.iterations < 1
        or args.blocks < 3
        or any(length < 1 for length in lengths)
        or 1 not in lengths
    ):
        raise SystemExit("invalid packed-projection microbenchmark configuration")

    model = load_model(
        args.model,
        quantization_bits=4,
        quantization_group_size=64,
    )
    attention = model.decoder.layers[0].self_attn
    qkv_modules = (attention.q_proj, attention.k_proj, attention.v_proj)
    kv_modules = (attention.k_proj, attention.v_proj)
    packed_qkv = pack_linear_projections(qkv_modules)
    packed_kv = pack_linear_projections(kv_modules)
    results = []
    all_exact = True

    for length in lengths:
        value = mx.random.normal((1, length, 512)).astype(mx.float16)
        separate_qkv = tuple(
            quantized_projection(value, projection) for projection in qkv_modules
        )
        candidate_qkv = packed_qkv.split(packed_qkv(value))
        separate_kv = tuple(
            quantized_projection(value, projection) for projection in kv_modules
        )
        candidate_kv = packed_kv.split(packed_kv(value))
        mx.eval(*separate_qkv, *candidate_qkv, *separate_kv, *candidate_kv)
        qkv_differences = [
            float(mx.max(mx.abs(control - candidate)).item())
            for control, candidate in zip(separate_qkv, candidate_qkv, strict=True)
        ]
        kv_differences = [
            float(mx.max(mx.abs(control - candidate)).item())
            for control, candidate in zip(separate_kv, candidate_kv, strict=True)
        ]
        exact = all(value == 0.0 for value in qkv_differences + kv_differences)
        all_exact = all_exact and exact

        qkv_control_blocks = []
        qkv_candidate_blocks = []
        kv_control_blocks = []
        kv_candidate_blocks = []
        for block in range(args.blocks):
            operations = (
                (
                    lambda: [
                        quantized_projection(value, projection)
                        for projection in qkv_modules
                    ],
                    lambda: [packed_qkv(value)],
                    lambda: [
                        quantized_projection(value, projection)
                        for projection in kv_modules
                    ],
                    lambda: [packed_kv(value)],
                )
                if block % 2 == 0
                else (
                    lambda: [packed_qkv(value)],
                    lambda: [
                        quantized_projection(value, projection)
                        for projection in qkv_modules
                    ],
                    lambda: [packed_kv(value)],
                    lambda: [
                        quantized_projection(value, projection)
                        for projection in kv_modules
                    ],
                )
            )
            measured = [elapsed(operation, args.iterations) for operation in operations]
            if block % 2 == 0:
                qkv_control, qkv_candidate, kv_control, kv_candidate = measured
            else:
                qkv_candidate, qkv_control, kv_candidate, kv_control = measured
            qkv_control_blocks.append(qkv_control)
            qkv_candidate_blocks.append(qkv_candidate)
            kv_control_blocks.append(kv_control)
            kv_candidate_blocks.append(kv_candidate)

        qkv_control = statistics.median(qkv_control_blocks)
        qkv_candidate = statistics.median(qkv_candidate_blocks)
        kv_control = statistics.median(kv_control_blocks)
        kv_candidate = statistics.median(kv_candidate_blocks)
        results.append(
            {
                "sequenceLength": length,
                "exactQ4Outputs": exact,
                "maximumAbsoluteDifference": max(qkv_differences + kv_differences),
                "qkv": {
                    "controlSeconds": qkv_control,
                    "candidateSeconds": qkv_candidate,
                    "improvementPercent": percent_improvement(
                        qkv_control, qkv_candidate
                    ),
                },
                "kv": {
                    "controlSeconds": kv_control,
                    "candidateSeconds": kv_candidate,
                    "improvementPercent": percent_improvement(
                        kv_control, kv_candidate
                    ),
                },
            }
        )

    qkv_m1 = next(row["qkv"]["improvementPercent"] for row in results if row["sequenceLength"] == 1)
    qkv_median = statistics.median(
        row["qkv"]["improvementPercent"] for row in results
    )
    passes = (
        all_exact
        and qkv_m1 >= args.minimum_improvement_percent
        and qkv_median >= args.minimum_improvement_percent
    )
    payload = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "purpose": "isolated exact q4 packed-attention projection stop gate",
        "claimEligible": False,
        "model": {
            "path": str(args.model),
            "sha256": sha256(args.model),
        },
        "hardware": platform.machine(),
        "operatingSystem": platform.platform(),
        "configuration": {
            "iterationsPerBlock": args.iterations,
            "blocks": args.blocks,
            "sequenceLengths": lengths,
            "minimumImprovementPercent": args.minimum_improvement_percent,
        },
        "results": results,
        "summary": {
            "exactQ4Outputs": all_exact,
            "qkvM1ImprovementPercent": qkv_m1,
            "medianQKVImprovementPercent": qkv_median,
        },
        "stopGate": {
            "passes": passes,
            "decision": "continue to canary" if passes else "stop before canary",
        },
        "doesNotAuthorizeAppIntegration": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": payload["summary"], **payload["stopGate"]}, indent=2))


if __name__ == "__main__":
    main()
