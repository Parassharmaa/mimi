#!/usr/bin/env python3
"""Contract tests for post-hoc Marian expert-delta compression."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


SCRIPT = Path(__file__).with_name("compress_marian_expert_delta.py")
SPEC = importlib.util.spec_from_file_location("compress_marian_expert_delta", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


torch.manual_seed(7)
left = torch.randn(12, 2)
right = torch.randn(2, 10)
rank_two = left @ right
approximation, factors, captured = MODULE.approximate_delta(
    rank_two,
    rank=2,
    niter=4,
)
assert set(factors) == {"factor_a", "factor_b"}
assert torch.allclose(approximation, rank_two, atol=1e-4)
assert abs(captured - float(torch.sum(rank_two**2))) < 1e-3

bias = torch.tensor([1.0, -2.0])
exact, factors, captured = MODULE.approximate_delta(bias, rank=2, niter=0)
assert set(factors) == {"delta"}
assert torch.equal(exact, bias)
assert captured == 5.0

print("Marian expert-delta compression contracts passed")
