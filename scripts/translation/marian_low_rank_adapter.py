#!/usr/bin/env python3
"""Low-rank adapters that can be merged into ordinary Marian Linear weights."""

from __future__ import annotations

import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn


LAYER = re.compile(r"^model\.(encoder|decoder)\.layers\.(\d+)\.(.+)$")


class LowRankLinear(nn.Module):
    """Frozen Linear plus a trainable, mergeable low-rank update."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank < 1 or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("invalid low-rank adapter configuration")
        self.base = base
        self.rank = rank
        self.alpha = float(alpha)
        self.scale = float(alpha) / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(
            torch.empty(
                rank,
                base.in_features,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        self.lora_b = nn.Parameter(
            torch.zeros(
                base.out_features,
                rank,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        self.base.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        adapted = torch.nn.functional.linear(self.dropout(inputs), self.lora_a)
        adapted = torch.nn.functional.linear(adapted, self.lora_b)
        return self.base(inputs) + adapted * self.scale

    @torch.no_grad()
    def initialize_from_delta(self, delta: torch.Tensor) -> float:
        """Initialize from the best rank-r SVD approximation of a weight delta."""
        if delta.shape != self.base.weight.shape:
            raise ValueError(
                f"delta shape {tuple(delta.shape)} differs from "
                f"{tuple(self.base.weight.shape)}"
            )
        value = delta.float()
        u, singular, vh = torch.linalg.svd(value, full_matrices=False)
        count = min(self.rank, singular.numel())
        root = singular[:count].clamp_min(0).sqrt()
        factor_b = u[:, :count] * root[None, :]
        factor_a = root[:, None] * vh[:count, :]
        factor_b /= self.scale
        self.lora_a.zero_()
        self.lora_b.zero_()
        self.lora_a[:count].copy_(factor_a.to(self.lora_a))
        self.lora_b[:, :count].copy_(factor_b.to(self.lora_b))
        approximation = self.lora_b.float() @ self.lora_a.float() * self.scale
        denominator = value.square().sum().clamp_min(torch.finfo(torch.float32).eps)
        return float((approximation.square().sum() / denominator).cpu())

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        merged = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )
        merged.weight.copy_(
            self.base.weight
            + (self.lora_b @ self.lora_a).to(self.base.weight) * self.scale
        )
        if self.base.bias is not None:
            assert merged.bias is not None
            merged.bias.copy_(self.base.bias)
        merged.requires_grad_(False)
        merged.train(self.training)
        return merged


def _parent_and_attribute(root: nn.Module, name: str) -> tuple[nn.Module, str]:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def consultation_v1_target(
    name: str,
    *,
    encoder_layers: int,
    encoder_top_layers: int,
) -> bool:
    """Fable-5-inspired narrow placement, corrected to Marian module names."""
    match = LAYER.match(name)
    if match is None:
        return False
    stack, layer_text, suffix = match.groups()
    layer = int(layer_text)
    if stack == "decoder":
        return (
            suffix
            in {
                "encoder_attn.q_proj",
                "encoder_attn.v_proj",
                "encoder_attn.out_proj",
                "fc1",
                "fc2",
            }
        )
    first_encoder_layer = max(0, encoder_layers - encoder_top_layers)
    return layer >= first_encoder_layer and suffix in {
        "self_attn.q_proj",
        "self_attn.v_proj",
    }


def all_projection_target(
    name: str,
    *,
    encoder_layers: int,
    encoder_top_layers: int,
) -> bool:
    del encoder_layers, encoder_top_layers
    match = LAYER.match(name)
    if match is None:
        return False
    suffix = match.group(3)
    return suffix.endswith(
        ("q_proj", "k_proj", "v_proj", "out_proj")
    ) or suffix in {"fc1", "fc2"}


PRESETS = {
    "consultation-v1": consultation_v1_target,
    "all-projections": all_projection_target,
}


def apply_low_rank_adapters(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    preset: str,
    encoder_layers: int,
    encoder_top_layers: int,
) -> list[str]:
    if preset not in PRESETS:
        raise ValueError(f"unknown adapter preset: {preset}")
    if not 0 <= encoder_top_layers <= encoder_layers:
        raise ValueError("encoder-top-layers is outside the encoder depth")
    model.requires_grad_(False)
    selector = PRESETS[preset]
    selected = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and selector(
            name,
            encoder_layers=encoder_layers,
            encoder_top_layers=encoder_top_layers,
        )
    ]
    for name in selected:
        parent, attribute = _parent_and_attribute(model, name)
        base = getattr(parent, attribute)
        setattr(
            parent,
            attribute,
            LowRankLinear(base, rank=rank, alpha=alpha, dropout=dropout),
        )
    if not selected:
        raise ValueError("adapter preset selected no Linear modules")
    return selected


def adapter_modules(model: nn.Module) -> dict[str, LowRankLinear]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, LowRankLinear)
    }


def capture_trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def load_trainable_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    current = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(current) != set(state):
        raise ValueError(
            f"trainable state keys differ: missing={sorted(set(current) - set(state))}, "
            f"unexpected={sorted(set(state) - set(current))}"
        )
    with torch.no_grad():
        for name, parameter in current.items():
            parameter.copy_(state[name].to(parameter))


def save_adapter(model: nn.Module, output: Path) -> None:
    state = capture_trainable_state(model)
    if not state:
        raise ValueError("model has no trainable adapter state")
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(output))


def load_adapter(model: nn.Module, path: Path) -> None:
    load_trainable_state(model, load_file(str(path)))


def merge_low_rank_adapters(model: nn.Module) -> list[str]:
    names = list(adapter_modules(model))
    for name in names:
        parent, attribute = _parent_and_attribute(model, name)
        wrapper = getattr(parent, attribute)
        assert isinstance(wrapper, LowRankLinear)
        setattr(parent, attribute, wrapper.merged())
    return names


def unfreeze_layer_norm_scales(model: nn.Module) -> list[str]:
    names: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            module.weight.requires_grad_(True)
            names.append(f"{name}.weight")
    return names
