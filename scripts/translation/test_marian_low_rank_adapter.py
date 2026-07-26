#!/usr/bin/env python3
"""Contract tests for trainable and mergeable Marian low-rank adapters."""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import torch
from torch import nn


SCRIPT = Path(__file__).with_name("marian_low_rank_adapter.py")
SPEC = importlib.util.spec_from_file_location("marian_low_rank_adapter", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.k_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)
        self.out_proj = nn.Linear(8, 8)


class Layer(nn.Module):
    def __init__(self, *, decoder: bool) -> None:
        super().__init__()
        self.self_attn = Attention()
        if decoder:
            self.encoder_attn = Attention()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)
        self.final_layer_norm = nn.LayerNorm(8)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        value = self.self_attn.out_proj(self.self_attn.v_proj(inputs))
        if hasattr(self, "encoder_attn"):
            value = value + self.encoder_attn.out_proj(
                self.encoder_attn.v_proj(inputs)
            )
        return self.fc2(torch.relu(self.fc1(value)))


class Stack(nn.Module):
    def __init__(self, *, decoder: bool) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Layer(decoder=decoder) for _ in range(2)])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


class Inner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Stack(decoder=False)
        self.decoder = Stack(decoder=True)


class Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = Inner()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model.decoder(self.model.encoder(inputs))


torch.manual_seed(7)
model = Toy().eval()
inputs = torch.randn(3, 4, 8)
baseline = model(inputs)
selected = MODULE.apply_low_rank_adapters(
    model,
    rank=2,
    alpha=4,
    dropout=0.05,
    preset="consultation-v1",
    encoder_layers=2,
    encoder_top_layers=1,
)
assert len(selected) == 12
assert torch.equal(model(inputs), baseline)
assert all(not parameter.requires_grad for module in MODULE.adapter_modules(model).values() for parameter in module.base.parameters())

first_name, first = next(
    (name, module)
    for name, module in MODULE.adapter_modules(model).items()
    if name.endswith("v_proj")
)
assert first_name in selected
with torch.no_grad():
    first.lora_b.normal_()
adapted = model(inputs)
assert not torch.equal(adapted, baseline)

with tempfile.TemporaryDirectory(prefix="mimi-lora-contract-") as temporary:
    path = Path(temporary) / "adapter.safetensors"
    MODULE.save_adapter(model, path)
    expected = MODULE.capture_trainable_state(model)
    with torch.no_grad():
        first.lora_b.zero_()
    MODULE.load_adapter(model, path)
    actual = MODULE.capture_trainable_state(model)
    assert all(torch.equal(actual[name], value) for name, value in expected.items())

model.eval()
adapted = model(inputs)
merged = MODULE.merge_low_rank_adapters(model)
assert merged == selected
assert not MODULE.adapter_modules(model)
assert torch.allclose(model(inputs), adapted, atol=1e-6, rtol=1e-6)

delta = torch.randn(8, 8)
linear = MODULE.LowRankLinear(nn.Linear(8, 8), rank=4, alpha=8, dropout=0)
captured = linear.initialize_from_delta(delta)
assert 0 < captured <= 1.00001

print("Marian low-rank adapter contracts passed.")
