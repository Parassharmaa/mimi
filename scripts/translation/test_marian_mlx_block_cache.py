#!/usr/bin/env python3
"""Exact contract test for the opt-in block-growing Marian self-K/V cache."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx


sys.path.insert(0, str(Path(__file__).resolve().parent))
from marian_mlx import (  # noqa: E402
    DIMENSIONS,
    Attention,
    BlockGrowingKVCache,
)


def exact(left: mx.array, right: mx.array) -> bool:
    return bool(mx.array_equal(left, right).item())


def main() -> None:
    mx.random.seed(20260718)
    attention = Attention()
    mx.eval(attention.parameters())
    hidden_steps = [
        mx.random.normal((1, 1, DIMENSIONS)) for _ in range(5)
    ]

    legacy_cache = None
    block_cache = None
    expected_capacities = [2, 2, 4, 4, 6]
    for index, hidden_states in enumerate(hidden_steps):
        legacy_output, legacy_cache = attention.step(
            hidden_states,
            cache=legacy_cache,
        )
        block_output, block_cache = attention.step(
            hidden_states,
            cache=block_cache,
            self_cache_block_size=2,
        )
        assert isinstance(block_cache, BlockGrowingKVCache)
        active_key, active_value = block_cache.active()
        mx.eval(
            legacy_output,
            block_output,
            legacy_cache[0],
            legacy_cache[1],
            active_key,
            active_value,
        )
        assert exact(legacy_output, block_output)
        assert exact(legacy_cache[0], active_key)
        assert exact(legacy_cache[1], active_value)
        assert block_cache.length == index + 1
        assert block_cache.capacity == expected_capacities[index]
        assert block_cache.key.shape == (1, 8, expected_capacities[index], 64)
        assert active_key.shape == (1, 8, index + 1, 64)

    encoder_states = mx.random.normal((1, 3, DIMENSIONS))
    _, cross_cache = attention.step(
        hidden_steps[0],
        key_value_states=encoder_states,
    )
    _, reused_cross_cache = attention.step(
        hidden_steps[1],
        key_value_states=encoder_states,
        cache=cross_cache,
    )
    assert reused_cross_cache is cross_cache
    assert reused_cross_cache[0] is cross_cache[0]
    assert reused_cross_cache[1] is cross_cache[1]

    print("Marian MLX block-growing cache contract passed.")


if __name__ == "__main__":
    main()
