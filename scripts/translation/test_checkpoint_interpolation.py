#!/usr/bin/env python3
"""Contract tests for Marian parent/adapted checkpoint interpolation."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/translation/interpolate_marian_checkpoints.py"


def write_checkpoint(path: Path, value: float) -> None:
    path.mkdir(parents=True)
    save_file(
        {
            "linear.weight": torch.tensor([[value, value + 2]], dtype=torch.float16),
            "linear.bias": torch.tensor([value], dtype=torch.float32),
        },
        str(path / "model.safetensors"),
    )
    (path / "config.json").write_text('{"model_type":"marian"}\n', encoding="utf-8")
    (path / "source.spm").write_bytes(b"shared-tokenizer")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-checkpoint-interpolation-") as temporary:
        root = Path(temporary)
        parent = root / "parent"
        adapted = root / "adapted"
        output = root / "output"
        write_checkpoint(parent, 1.0)
        write_checkpoint(adapted, 5.0)

        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                str(parent),
                str(adapted),
                str(output),
                "--adapted-weight",
                "0.25",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        weights = load_file(str(output / "model.safetensors"))
        assert torch.equal(
            weights["linear.weight"],
            torch.tensor([[2.0, 4.0]], dtype=torch.float16),
        )
        assert torch.equal(weights["linear.bias"], torch.tensor([2.0]))
        manifest = json.loads(
            (output / "mimi_checkpoint_interpolation_manifest.json").read_text()
        )
        assert manifest["adapted_weight"] == 0.25
        assert manifest["include_prefixes"] == ["*"]
        assert manifest["operation"] == "linear-checkpoint-interpolation"

        selected_output = root / "selected-output"
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                str(parent),
                str(adapted),
                str(selected_output),
                "--adapted-weight",
                "0.5",
                "--include-prefix",
                "linear.weight",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        selected = load_file(str(selected_output / "model.safetensors"))
        assert torch.equal(
            selected["linear.weight"],
            torch.tensor([[3.0, 5.0]], dtype=torch.float16),
        )
        assert torch.equal(selected["linear.bias"], torch.tensor([1.0]))
        selected_manifest = json.loads(
            (selected_output / "mimi_checkpoint_interpolation_manifest.json").read_text()
        )
        assert selected_manifest["include_prefixes"] == ["linear.weight"]

        failed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                str(parent),
                str(adapted),
                str(root / "bad"),
                "--adapted-weight",
                "1.1",
            ],
            capture_output=True,
            text=True,
        )
        assert failed.returncode != 0
        assert "between zero and one" in failed.stderr

    print("Marian checkpoint interpolation contracts passed.")


if __name__ == "__main__":
    main()
