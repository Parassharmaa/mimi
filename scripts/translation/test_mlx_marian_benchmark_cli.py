#!/usr/bin/env python3
"""Fail-fast CLI contracts for the Python MLX Marian benchmark."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "scripts/translation/run_mlx_marian_benchmark.py"
SUITE = ROOT / "Research/translation/benchmark/canary.jsonl"


def rejected(arguments: list[str], expected: str) -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-mlx-benchmark-cli-") as temporary:
        root = Path(temporary)
        result = subprocess.run(
            [
                sys.executable,
                str(BENCHMARK),
                str(SUITE),
                str(root / "report.json"),
                "--en-ja-model",
                str(root / "missing-en-ja"),
                "--ja-en-model",
                str(root / "missing-ja-en"),
                *arguments,
            ],
            text=True,
            capture_output=True,
            check=False,
        )
    assert result.returncode != 0
    message = result.stdout + result.stderr
    assert expected in message, message


def main() -> None:
    rejected(["--warm-runs", "-1"], "warm runs must be non-negative")
    rejected(["--max-tokens", "0"], "max tokens must be positive")
    rejected(
        ["--precomputed-position-table"],
        "precomputed position table requires --cached-decoding",
    )
    rejected(
        [
            "--cached-decoding",
            "--precomputed-position-table",
            "--max-tokens",
            "193",
        ],
        "precomputed position table requires max tokens at or below 192",
    )
    print("MLX Marian benchmark CLI validation contract passed.")


if __name__ == "__main__":
    main()
