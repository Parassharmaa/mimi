#!/usr/bin/env python3
"""Exactness contracts for concatenated Marian attention projections."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from marian_mlx import Attention, pack_linear_projections


def assert_equivalent(
    parts: tuple[mx.array, ...],
    packed: tuple[mx.array, ...],
    *,
    exact: bool,
) -> None:
    assert len(parts) == len(packed)
    mx.eval(*parts, *packed)
    for expected, actual in zip(parts, packed, strict=True):
        assert expected.shape == actual.shape
        maximum_difference = float(mx.max(mx.abs(expected - actual)).item())
        assert maximum_difference == 0.0 if exact else maximum_difference <= 2e-5


for quantized in (False, True):
    attention = Attention()
    if quantized:
        nn.quantize(attention, group_size=64, bits=4)
    value = mx.random.normal((1, 7, 512)).astype(mx.float16)
    qkv_parts = (
        attention.q_proj(value),
        attention.k_proj(value),
        attention.v_proj(value),
    )
    packed_qkv = pack_linear_projections(
        (attention.q_proj, attention.k_proj, attention.v_proj)
    )
    assert_equivalent(
        qkv_parts,
        packed_qkv.split(packed_qkv(value)),
        exact=quantized,
    )

    kv_parts = (attention.k_proj(value), attention.v_proj(value))
    packed_kv = pack_linear_projections((attention.k_proj, attention.v_proj))
    assert_equivalent(
        kv_parts,
        packed_kv.split(packed_kv(value)),
        exact=quantized,
    )

    self_attention = Attention()
    cross_attention = Attention()
    if quantized:
        nn.quantize(self_attention, group_size=64, bits=4)
        nn.quantize(cross_attention, group_size=64, bits=4)
    self_attention.enable_qkv_packing()
    cross_attention.enable_kv_packing()
    assert self_attention.q_proj is None
    assert self_attention.k_proj is None
    assert self_attention.v_proj is None
    assert cross_attention.q_proj is not None
    assert cross_attention.k_proj is None
    assert cross_attention.v_proj is None
    self_output = self_attention(value)
    cross_output = cross_attention(value, value)
    mx.eval(self_output, cross_output)
    assert self_output.shape == value.shape
    assert cross_output.shape == value.shape

print("Marian MLX packed-projection contracts passed.")
