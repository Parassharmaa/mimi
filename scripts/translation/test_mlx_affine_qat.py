#!/usr/bin/env python3
"""Contract tests for the pinned MLX affine fake-quantization training path."""

from __future__ import annotations

import numpy as np
import torch
from mlx import core as mx
from torch import nn

from train_marian_distillation import (
    MLXAffineFakeQuantization,
    capture_canonical_state_dict,
    disable_mlx_affine_fake_quantization,
    enable_mlx_affine_fake_quantization,
    mlx_affine_dequantize,
)


class PositionalEmbedding(nn.Embedding):
    """Fixture for Marian's computed, non-quantized positional embedding."""


class TinyTiedMarian(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Embedding(32, 64)
        self.position = PositionalEmbedding(8, 64)
        self.projection = nn.Linear(64, 64)
        self.lm_head = nn.Linear(64, 32, bias=False)
        self.lm_head.weight = self.shared.weight


def assert_exact_mlx_values() -> None:
    rng = np.random.default_rng(20260718)
    source = rng.normal(size=(5, 128)).astype(np.float16)
    torch_weight = torch.from_numpy(source).float()
    simulated = mlx_affine_dequantize(torch_weight, group_size=64, bits=4)
    packed, scales, biases = mx.quantize(
        mx.array(source),
        group_size=64,
        bits=4,
        mode="affine",
    )
    expected = mx.dequantize(
        packed,
        scales=scales,
        biases=biases,
        group_size=64,
        bits=4,
        mode="affine",
    )
    assert np.array_equal(simulated.numpy().astype(np.float16), np.array(expected))


def assert_straight_through_gradient() -> None:
    weight = torch.linspace(-2, 2, 128, dtype=torch.float32).reshape(2, 64)
    weight.requires_grad_(True)
    output = MLXAffineFakeQuantization(group_size=64, bits=4)(weight)
    output.sum().backward()
    assert torch.equal(weight.grad, torch.ones_like(weight))

    quantizer = MLXAffineFakeQuantization(group_size=64, bits=4).eval()
    first = quantizer(weight.detach()).clone()
    with torch.no_grad():
        weight.add_(1)
    assert torch.equal(quantizer(weight.detach()), first)
    quantizer.train().eval()
    assert not torch.equal(quantizer(weight.detach()), first)


def assert_tied_state_round_trip() -> None:
    model = TinyTiedMarian()
    original_position = model.position.weight.detach().clone()
    names = enable_mlx_affine_fake_quantization(model, group_size=64, bits=4)
    assert names == ["shared", "projection", "lm_head"]
    assert model.shared.parametrizations.weight.original is (
        model.lm_head.parametrizations.weight.original
    )
    state = capture_canonical_state_dict(model)
    assert "shared.weight" in state
    assert "lm_head.weight" in state
    assert not any("parametrizations" in name for name in state)
    assert torch.equal(model.position.weight, original_position)
    disable_mlx_affine_fake_quantization(model)
    model.load_state_dict(state, strict=True)
    assert model.shared.weight is model.lm_head.weight


def main() -> None:
    assert_exact_mlx_values()
    assert_straight_through_gradient()
    assert_tied_state_round_trip()
    print("Mimi exact MLX affine fake-quantization contract passed.")


if __name__ == "__main__":
    main()
