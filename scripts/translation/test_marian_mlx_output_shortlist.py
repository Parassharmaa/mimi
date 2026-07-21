#!/usr/bin/env python3
"""Exact quantized-output contracts for Marian projection shortlists."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from marian_mlx import CompositeOutputProjection, Marian


TOKEN_IDS = [0, 1, 2, 3, 100, 5_000, 32_000]
HIDDEN = mx.random.normal((1, 2, 512))


dense = Marian(encoder_layers=1, decoder_layers=1)
dense_shortlist = dense.prepare_output_shortlist(TOKEN_IDS)
dense_full = dense.project_output(HIDDEN)[:, :, TOKEN_IDS]
dense_subset = dense.project_output(HIDDEN, dense_shortlist)
mx.eval(dense_full, dense_subset)
assert bool(mx.allclose(dense_full, dense_subset, atol=1e-6, rtol=1e-6).item())

quantized = Marian(encoder_layers=1, decoder_layers=1)
quantized.shared = nn.QuantizedEmbedding.from_embedding(
    quantized.shared,
    group_size=64,
    bits=4,
)
quantized_shortlist = quantized.prepare_output_shortlist(TOKEN_IDS)
quantized_full = quantized.project_output(HIDDEN)[:, :, TOKEN_IDS]
quantized_subset = quantized.project_output(HIDDEN, quantized_shortlist)
mx.eval(quantized_full, quantized_subset)
assert bool(mx.array_equal(quantized_full, quantized_subset).item())
assert quantized_shortlist.token_ids == tuple(TOKEN_IDS)
assert quantized_shortlist.pad_index == len(TOKEN_IDS) - 1

static_ids = [0, 1, 2, 3, 100, 32_000]
extension_ids = [5_000]
static = quantized.prepare_output_shortlist(static_ids)
extension = quantized.prepare_output_extension(extension_ids)
composite = CompositeOutputProjection(
    parts=(static, extension),
    token_ids=static.token_ids + extension.token_ids,
    pad_index=static.pad_index,
)
composite_full = quantized.project_output(HIDDEN)[:, :, list(composite.token_ids)]
composite_subset = quantized.project_output(HIDDEN, composite)
mx.eval(composite_full, composite_subset)
assert bool(mx.array_equal(composite_full, composite_subset).item())

for invalid in ([0, 1, 2], [0, 1, 32_000, 32_000], [1, 0, 32_000]):
    try:
        quantized.prepare_output_shortlist(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError(f"invalid shortlist was accepted: {invalid}")

print("Marian MLX output-shortlist contracts passed.")
