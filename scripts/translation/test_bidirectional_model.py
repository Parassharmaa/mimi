#!/usr/bin/env python3
"""Contract tests for a one-physical-model bidirectional Marian candidate."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts/translation/merge_directional_marian.py"
SPEC = importlib.util.spec_from_file_location("merge_directional_marian", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_checkpoint(root: Path, direction: str, value: float) -> None:
    root.mkdir()
    save_file(
        {
            "linear.weight": torch.full((2, 2), value),
            "linear.bias": torch.tensor([value, value + 1.0]),
        },
        str(root / "model.safetensors"),
    )
    (root / "config.json").write_text(
        json.dumps({"_name_or_path": str(root), "d_model": 2}) + "\n"
    )
    for name in MODULE.COPY_FILES:
        value = (
            json.dumps(
                {
                    "tokenizer_class": "MarianTokenizer",
                    "source_lang": direction[:2],
                    "target_lang": direction[-2:],
                }
            )
            if name == "tokenizer_config.json"
            else f"shared-{name}"
        )
        (root / name).write_text(value + "\n")
    (root / "mimi_training_manifest.json").write_text(
        json.dumps(
            {
                "direction": direction,
                "student_repository": f"example/{direction}",
                "student_revision": "pinned",
                "license": "CC-BY-SA-4.0",
                "dataset": {"train_sha256": f"{direction}-train"},
            }
        )
        + "\n"
    )


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    en_ja = root / "en-ja"
    ja_en = root / "ja-en"
    write_checkpoint(en_ja, "en-ja", 1.0)
    write_checkpoint(ja_en, "ja-en", 3.0)

    MODULE.validate_compatible(en_ja, ja_en)
    merged = MODULE.merge_weights(en_ja, ja_en, 0.25)
    assert torch.equal(merged["linear.weight"], torch.full((2, 2), 2.5))
    assert torch.equal(merged["linear.bias"], torch.tensor([2.5, 3.5]))

    (ja_en / "source.spm").write_text("different\n")
    try:
        MODULE.validate_compatible(en_ja, ja_en)
    except SystemExit as error:
        assert "tokenizer asset differs" in str(error)
    else:
        raise AssertionError("different tokenizers must be rejected")

print("bidirectional Marian merge contract passed")
