#!/usr/bin/env python3
"""Fast tests for balanced two-teacher Marian distillation primitives."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAIN = load("train_bidirectional_marian", "scripts/translation/train_bidirectional_marian.py")
BUILD = load("build_bidirectional_dataset", "scripts/translation/build_bidirectional_dataset.py")


teacher = torch.tensor([[[3.0, 1.0], [1.0, 3.0]]])
identical_sum, identical_tokens = TRAIN.teacher_student_kl(
    teacher, teacher, torch.tensor([[0, 1]]), 1.0
)
assert identical_tokens.item() == 2
assert abs(identical_sum.item()) < 1e-6

different_sum, different_tokens = TRAIN.teacher_student_kl(
    -teacher, teacher, torch.tensor([[0, -100]]), 2.0
)
assert different_tokens.item() == 1
assert different_sum.item() > 0

states = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
identical_encoder_sum, identical_encoder_values = TRAIN.masked_encoder_mse(
    states,
    states,
    torch.tensor([[1, 0]]),
)
assert identical_encoder_sum.item() == 0
assert identical_encoder_values.item() == 2
different_encoder_sum, different_encoder_values = TRAIN.masked_encoder_mse(
    states,
    torch.zeros_like(states),
    torch.tensor([[1, 0]]),
)
assert different_encoder_values.item() == 2
assert different_encoder_sum.item() == 5

with tempfile.TemporaryDirectory(prefix="mimi-bidirectional-compatibility-") as temporary:
    root = Path(temporary)
    paths = [root / name for name in ("en-ja", "ja-en", "student")]
    base_configuration = {
        "activation_function": "swish",
        "d_model": 4,
        "decoder_attention_heads": 2,
        "decoder_ffn_dim": 8,
        "decoder_layers": 2,
        "encoder_attention_heads": 2,
        "encoder_ffn_dim": 8,
        "encoder_layers": 2,
        "max_position_embeddings": 16,
        "model_type": "marian",
        "normalize_before": False,
        "normalize_embedding": False,
        "scale_embedding": True,
        "share_encoder_decoder_embeddings": True,
        "static_position_embeddings": True,
        "vocab_size": 12,
    }
    for path in paths:
        path.mkdir()
        (path / "model.safetensors").write_bytes(b"fixture")
        (path / "config.json").write_text(
            json.dumps(base_configuration) + "\n",
            encoding="utf-8",
        )
        for name in TRAIN.TOKENIZER_ASSETS:
            (path / name).write_text(f"shared-{name}\n", encoding="utf-8")
    student_configuration = {
        **base_configuration,
        "encoder_ffn_dim": 18,
        "decoder_ffn_dim": 18,
    }
    (paths[2] / "config.json").write_text(
        json.dumps(student_configuration) + "\n",
        encoding="utf-8",
    )
    compatibility = TRAIN.validate_model_compatibility(*paths)
    assert compatibility["ffn_dimensions"]["encoder_ffn_dim"] == {
        "teacher": 8,
        "student": 18,
    }
    incompatible = {**student_configuration, "d_model": 5}
    (paths[2] / "config.json").write_text(
        json.dumps(incompatible) + "\n",
        encoding="utf-8",
    )
    try:
        TRAIN.validate_model_compatibility(*paths)
    except SystemExit as error:
        assert "d_model" in str(error)
    else:
        raise AssertionError("incompatible student dimensions must be rejected")

rows = [
    {"id": "a", "direction": "ja-en"},
    {"id": "b", "direction": "ja-en"},
]
repeated = BUILD.repeat_to_count(rows, 5, 7)
assert len(repeated) == 5
assert len({row["id"] for row in repeated}) == 5
assert sum(row["balance_repeat_index"] > 0 for row in repeated) == 3

left = [{"id": "en-1"}, {"id": "en-2"}]
right = [{"id": "ja-1"}, {"id": "ja-2"}]
assert [row["id"] for row in BUILD.interleave(left, right)] == [
    "en-1",
    "ja-1",
    "en-2",
    "ja-2",
]

print("bidirectional distillation objective contracts passed")
