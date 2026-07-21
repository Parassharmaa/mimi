#!/usr/bin/env python3
"""Fast contracts for Mimi's token-local negative-space objective."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "negative_space_training", ROOT / "scripts/translation/train_marian_negative_space.py"
)
assert spec and spec.loader
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


assert training.first_divergence([4, 8, 2], [4, 9, 2]) == (1, 8, 9)
try:
    training.first_divergence([4, 8, 2], [4, 8, 2])
except ValueError:
    pass
else:
    raise AssertionError("identical token sequences must be rejected")

logits = torch.zeros((2, 3, 5), dtype=torch.float32, requires_grad=True)
positions = torch.tensor([1, 2])
rejected = torch.tensor([3, 4])
severity = torch.tensor([1.0, 0.5])
loss, probability = training.token_local_unlikelihood(
    logits, positions, rejected, severity
)
assert torch.allclose(probability, torch.tensor([0.2, 0.2]))
assert loss.item() > 0
loss.backward()
assert logits.grad is not None
assert logits.grad[0, 1, 3].item() > 0
assert logits.grad[1, 2, 4].item() > 0

chosen = torch.tensor([1, 2])
margin, preferred = training.divergence_metrics(
    logits.detach(), positions, chosen, rejected
)
assert margin.shape == preferred.shape == torch.Size([2])


class FixtureModel:
    def save_pretrained(self, output: Path, safe_serialization: bool) -> None:
        assert safe_serialization is True
        (output / "model.safetensors").write_bytes(b"fixture")


class FixtureTokenizer:
    def save_pretrained(self, output: Path) -> None:
        (output / "tokenizer_config.json").write_text("{}", encoding="utf-8")


with tempfile.TemporaryDirectory(prefix="mimi-negative-space-save-") as temporary:
    output = Path(temporary) / "candidate"
    manifest = {"direction": "en-ja", "promotion_eligible": False}
    training.save_candidate(FixtureModel(), FixtureTokenizer(), output, manifest)
    specialized = output / "mimi_negative_space_training_manifest.json"
    canonical = output / "mimi_training_manifest.json"
    assert specialized.read_bytes() == canonical.read_bytes()
    assert json.loads(canonical.read_text()) == manifest

print("Marian token-local negative-space objective contracts passed.")
