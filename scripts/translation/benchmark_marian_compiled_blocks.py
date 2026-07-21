#!/usr/bin/env python3
"""Gate MLX compilation on stable Marian decoder residual and FFN blocks."""

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
import mlx.nn as nn

from marian_mlx import load_model


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def elapsed(operation, inputs: tuple[mx.array, ...], iterations: int) -> float:
    for _ in range(100):
        mx.eval(operation(*inputs))
    mx.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        mx.eval(operation(*inputs))
    mx.synchronize()
    return (time.perf_counter() - started) / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--iterations", type=int, default=2_000)
    parser.add_argument("--blocks", type=int, default=7)
    parser.add_argument("--minimum-improvement-percent", type=float, default=10.0)
    args = parser.parse_args()
    if args.iterations < 1 or args.blocks < 3:
        raise SystemExit("invalid compiled-block benchmark configuration")

    model = load_model(
        args.model,
        quantization_bits=4,
        quantization_group_size=64,
    )
    layer = model.decoder.layers[0]
    hidden = mx.random.normal((1, 1, 512)).astype(mx.float16)
    attended = mx.random.normal((1, 1, 512)).astype(mx.float16)
    mx.eval(hidden, attended)

    def residual(value: mx.array, update: mx.array) -> mx.array:
        return layer.self_attn_layer_norm(value + update)

    def feed_forward(value: mx.array) -> mx.array:
        projected = layer.fc2(nn.silu(layer.fc1(value)))
        return layer.final_layer_norm(value + projected)

    operations = {
        "residualLayerNorm": (residual, (hidden, attended)),
        "feedForwardResidualLayerNorm": (feed_forward, (hidden,)),
    }
    results = {}
    all_exact = True
    all_fast = True
    for name, (control, inputs) in operations.items():
        candidate = mx.compile(control)
        control_output = control(*inputs)
        candidate_output = candidate(*inputs)
        mx.eval(control_output, candidate_output)
        maximum_difference = float(
            mx.max(mx.abs(control_output - candidate_output)).item()
        )
        exact = maximum_difference == 0.0
        all_exact = all_exact and exact
        improvements = []
        control_blocks = []
        candidate_blocks = []
        for block in range(args.blocks):
            if block % 2 == 0:
                control_time = elapsed(control, inputs, args.iterations)
                candidate_time = elapsed(candidate, inputs, args.iterations)
            else:
                candidate_time = elapsed(candidate, inputs, args.iterations)
                control_time = elapsed(control, inputs, args.iterations)
            control_blocks.append(control_time)
            candidate_blocks.append(candidate_time)
            improvements.append((1.0 - candidate_time / control_time) * 100.0)
        median_improvement = statistics.median(improvements)
        passes = exact and median_improvement >= args.minimum_improvement_percent
        all_fast = all_fast and passes
        results[name] = {
            "exact": exact,
            "maximumAbsoluteDifference": maximum_difference,
            "medianControlSeconds": statistics.median(control_blocks),
            "medianCompiledSeconds": statistics.median(candidate_blocks),
            "blockImprovementPercent": improvements,
            "medianImprovementPercent": median_improvement,
            "passes": passes,
        }

    passes = all_exact and all_fast
    payload = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "purpose": "isolated exact MLX compiled-block continuation gate",
        "claimEligible": False,
        "model": {"path": str(args.model), "sha256": sha256(args.model)},
        "hardware": platform.machine(),
        "operatingSystem": platform.platform(),
        "configuration": {
            "iterationsPerBlock": args.iterations,
            "blocks": args.blocks,
            "minimumImprovementPercent": args.minimum_improvement_percent,
            "shape": [1, 1, 512],
        },
        "results": results,
        "stopGate": {
            "passes": passes,
            "decision": "continue to runtime canary" if passes else "stop before runtime canary",
        },
        "doesNotAuthorizeAppIntegration": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "results": {
                    name: {
                        "exact": record["exact"],
                        "medianImprovementPercent": record[
                            "medianImprovementPercent"
                        ],
                    }
                    for name, record in results.items()
                },
                **payload["stopGate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
