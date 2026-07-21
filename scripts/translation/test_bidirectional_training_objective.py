#!/usr/bin/env python3
"""Fast tests for balanced two-teacher Marian distillation primitives."""

from __future__ import annotations

import importlib.util
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
