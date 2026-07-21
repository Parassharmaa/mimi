#!/usr/bin/env python3
"""Estimate same-depth SSRU decoder-layer speed before any student training."""

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

from marian_mlx import load_model, pack_linear_projections


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def elapsed(operation, iterations: int) -> float:
    for _ in range(50):
        mx.eval(*operation())
    mx.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        mx.eval(*operation())
    mx.synchronize()
    return (time.perf_counter() - started) / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--iterations", type=int, default=1_000)
    parser.add_argument("--blocks", type=int, default=7)
    parser.add_argument("--source-length", type=int, default=16)
    parser.add_argument("--prefix-length", type=int, default=16)
    parser.add_argument("--minimum-improvement-percent", type=float, default=10.0)
    args = parser.parse_args()
    if min(args.iterations, args.blocks, args.source_length, args.prefix_length) < 1:
        raise SystemExit("invalid SSRU proxy benchmark configuration")

    model = load_model(
        args.model,
        quantization_bits=4,
        quantization_group_size=64,
    )
    layer = model.decoder.layers[0]
    hidden = mx.random.normal((1, 1, 512)).astype(mx.float16)
    encoder_states = mx.random.normal((1, args.source_length, 512)).astype(mx.float16)
    self_cache = (
        mx.random.normal((1, 8, args.prefix_length, 64)).astype(mx.float16),
        mx.random.normal((1, 8, args.prefix_length, 64)).astype(mx.float16),
    )
    projected_key, projected_value = layer.encoder_attn.project_kv(encoder_states)
    cross_cache = (
        layer.encoder_attn.split_heads(projected_key),
        layer.encoder_attn.split_heads(projected_value),
    )
    cell = mx.zeros((1, 1, 512), dtype=mx.float16)

    # The paper combines W_t x_t and W x_t. Reuse two authenticated q4
    # projection matrices only as a shape/runtime proxy; their values do not
    # define or initialize a trainable SSRU candidate.
    ssru_projection = pack_linear_projections(
        (layer.self_attn.q_proj, layer.self_attn.k_proj)
    )
    mx.eval(
        hidden,
        encoder_states,
        *self_cache,
        *cross_cache,
        cell,
        ssru_projection.weight,
        ssru_projection.scales,
        ssru_projection.quantization_biases,
        ssru_projection.output_bias,
    )

    def transformer_layer() -> list[mx.array]:
        output, next_cache = layer.step(
            hidden,
            encoder_states,
            (self_cache, cross_cache),
        )
        return [
            output,
            next_cache[0][0],
            next_cache[0][1],
            next_cache[1][0],
            next_cache[1][1],
        ]

    def ssru_layer() -> list[mx.array]:
        gate_logits, proposal = ssru_projection.split(ssru_projection(hidden))
        forget = mx.sigmoid(gate_logits)
        next_cell = forget * cell + (1.0 - forget) * proposal
        recurrent = mx.maximum(next_cell, mx.array(0, dtype=next_cell.dtype))
        output = layer.self_attn_layer_norm(hidden + recurrent)
        attended, next_cross_cache = layer.encoder_attn.step(
            output,
            key_value_states=encoder_states,
            cache=cross_cache,
        )
        output = layer.encoder_attn_layer_norm(output + attended)
        feed_forward = layer.fc2(nn.silu(layer.fc1(output)))
        output = layer.final_layer_norm(output + feed_forward)
        return [output, next_cell, next_cross_cache[0], next_cross_cache[1]]

    controls = []
    candidates = []
    improvements = []
    for block in range(args.blocks):
        if block % 2 == 0:
            control = elapsed(transformer_layer, args.iterations)
            candidate = elapsed(ssru_layer, args.iterations)
        else:
            candidate = elapsed(ssru_layer, args.iterations)
            control = elapsed(transformer_layer, args.iterations)
        controls.append(control)
        candidates.append(candidate)
        improvements.append((1.0 - candidate / control) * 100.0)

    median_improvement = statistics.median(improvements)
    passes = median_improvement >= args.minimum_improvement_percent
    payload = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "purpose": "same-depth q4 SSRU decoder-layer compute proxy before training",
        "claimEligible": False,
        "qualityEvaluated": False,
        "model": {"path": str(args.model), "sha256": sha256(args.model)},
        "hardware": platform.machine(),
        "operatingSystem": platform.platform(),
        "configuration": {
            "iterationsPerBlock": args.iterations,
            "blocks": args.blocks,
            "sourceLength": args.source_length,
            "prefixLength": args.prefix_length,
            "minimumImprovementPercent": args.minimum_improvement_percent,
            "ssru": "f=sigmoid(W_t x+b_f); c=f*c_prev+(1-f)*W*x; o=ReLU(c); W_t/W packed",
            "retainsDecoderFFN": True,
            "retainsDecoderDepth": True,
        },
        "result": {
            "medianTransformerLayerSeconds": statistics.median(controls),
            "medianSSRUProxyLayerSeconds": statistics.median(candidates),
            "blockImprovementPercent": improvements,
            "medianImprovementPercent": median_improvement,
        },
        "stopGate": {
            "passes": passes,
            "decision": "continue to student training" if passes else "stop before student training",
        },
        "doesNotAuthorizeAppIntegration": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"result": payload["result"], **payload["stopGate"]}, indent=2))


if __name__ == "__main__":
    main()
