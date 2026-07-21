#!/usr/bin/env python3
"""Exact contract for Mimi's opt-in precomputed Marian position table."""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx


sys.path.insert(0, str(Path(__file__).resolve().parent))
from marian_mlx import (  # noqa: E402
    DIMENSIONS,
    POSITION_TABLE_LENGTH,
    position_rows,
    precomputed_position_table,
)


def exact(left: mx.array, right: mx.array) -> bool:
    return bool(mx.array_equal(left, right).item())


def main() -> None:
    for dtype in (mx.float16, mx.float32):
        first = precomputed_position_table(dtype)
        second = precomputed_position_table(dtype)
        assert first is second
        assert first.shape == (POSITION_TABLE_LENGTH, DIMENSIONS)

        for offset, length in ((0, 1), (0, 17), (17, 23), (191, 1)):
            dynamic = position_rows(
                length,
                offset,
                dtype,
                use_precomputed_table=False,
            )
            cached = position_rows(
                length,
                offset,
                dtype,
                use_precomputed_table=True,
            )
            mx.eval(dynamic, cached)
            assert dynamic.shape == cached.shape == (length, DIMENSIONS)
            assert exact(dynamic, cached)

    try:
        position_rows(
            1,
            POSITION_TABLE_LENGTH,
            mx.float16,
            use_precomputed_table=True,
        )
    except ValueError as error:
        assert "precomputed positional table" in str(error)
    else:
        raise AssertionError("position lookup beyond the fixed table must fail")

    print("Marian MLX precomputed position-table contract passed.")


if __name__ == "__main__":
    main()
