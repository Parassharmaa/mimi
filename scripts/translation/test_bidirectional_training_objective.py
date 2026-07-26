#!/usr/bin/env python3
"""Fast tests for balanced two-teacher Marian distillation primitives."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAIN = load("train_bidirectional_marian", "scripts/translation/train_bidirectional_marian.py")
BUILD = load("build_bidirectional_dataset", "scripts/translation/build_bidirectional_dataset.py")

assert TRAIN.VALIDATION_CACHE_POLICY == {
    "loss_forward": False,
    "greedy_generation": True,
}

exact_loader = DataLoader(
    list(range(12_066)),
    batch_size=4,
    shuffle=False,
    drop_last=True,
)
assert len(exact_loader) == 3_016
assert len(exact_loader) % 8 == 0
assert 12_066 - len(exact_loader) * 4 == 2

objective_mean = TRAIN.mean_objectives(
    [
        {
            "cross_entropy": 2.0,
            "teacher_kl": 4.0,
            "encoder_alignment": 6.0,
            "combined": 8.0,
        },
        {
            "cross_entropy": 4.0,
            "teacher_kl": 6.0,
            "encoder_alignment": 8.0,
            "combined": 10.0,
        },
    ]
)
assert objective_mean == {
    "cross_entropy": 3.0,
    "teacher_kl": 5.0,
    "encoder_alignment": 7.0,
    "combined": 9.0,
}

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


class FakeConfiguration:
    def __init__(self) -> None:
        self.use_cache = False


class FakeModel:
    def __init__(self) -> None:
        self.config = FakeConfiguration()

    def save_pretrained(self, path: Path, *, safe_serialization: bool) -> None:
        assert safe_serialization is True
        assert self.config.use_cache is True
        (path / "model.safetensors").write_bytes(b"immutable-weights")
        (path / "config.json").write_text(
            json.dumps({"use_cache": self.config.use_cache}) + "\n",
            encoding="utf-8",
        )


class FakeTokenizer:
    def save_pretrained(self, path: Path) -> None:
        (path / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")


with tempfile.TemporaryDirectory(prefix="mimi-bidirectional-checkpoint-") as temporary:
    output = Path(temporary) / "output"
    model = FakeModel()
    tokenizer = FakeTokenizer()
    checkpoint, checkpoint_manifest = TRAIN.save_immutable_checkpoint(
        model,
        tokenizer,
        output,
        step=250,
        metrics={"macro_direction_chrf_pp": 40.0},
        objective={"updates_in_interval": 250},
        contract_sha256="contract-sha",
    )
    assert model.config.use_cache is False
    assert checkpoint.name == "step-0000250"
    assert checkpoint_manifest["immutable_scheduled_checkpoint"] is True
    assert TRAIN.authenticate_checkpoint(checkpoint, "contract-sha")["step"] == 250
    try:
        TRAIN.save_immutable_checkpoint(
            model,
            tokenizer,
            output,
            step=250,
            metrics={},
            objective={},
            contract_sha256="contract-sha",
        )
    except SystemExit as error:
        assert "overwrite immutable checkpoint" in str(error)
    else:
        raise AssertionError("an immutable scheduled checkpoint was overwritten")

    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    (parameter.square().sum()).backward()
    optimizer.step()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    generator = torch.Generator().manual_seed(7)
    epoch_start = generator.get_state().clone()
    torch.randperm(8, generator=generator)
    post_iterator = generator.get_state().clone()
    latest = TRAIN.save_rolling_resume_state(
        output,
        checkpoint=checkpoint,
        checkpoint_manifest=checkpoint_manifest,
        optimizer=optimizer,
        scheduler=scheduler,
        train_generator=generator,
        epoch_start_generator_state=epoch_start,
        post_iterator_generator_state=post_iterator,
        progress={
            "update_step": 250,
            "micro_step": 2_000,
            "epoch": 0,
            "next_batch_index": 2_000,
        },
        history=[{"step": 0}, {"step": 250}],
        best={
            "metrics": {"step": 250},
            "artifact": {"path": str(checkpoint)},
            "checkpoints": [{"step": 250}],
        },
        contract_sha256="contract-sha",
    )
    assert latest["step"] == 250
    restored, authenticated_latest = TRAIN.load_authenticated_resume_state(
        output,
        checkpoint,
        "contract-sha",
    )
    assert restored["progress"]["update_step"] == 250
    assert authenticated_latest["state"]["sha256"] == latest["state"]["sha256"]

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
