#!/usr/bin/env python3
"""Contract tests for variable Marian depth and shallow-decoder pruning."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from marian_mlx import Marian, load_model


ROOT = Path(__file__).resolve().parents[2]
PRUNER = ROOT / "scripts/translation/prune_marian_mlx_decoder.py"


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-decoder-pruning-") as temporary:
        root = Path(temporary)
        source = root / "source"
        source.mkdir()
        mx.save_safetensors(
            str(source / "model.safetensors"),
            {
                **{f"encoder.layers.{index}.fixture": mx.array([index]) for index in range(3)},
                **{f"decoder.layers.{index}.fixture": mx.array([index]) for index in range(4)},
                "shared.fixture": mx.array([99]),
            },
        )
        for name in ("tokenizer.json", "tokenizer_config.json"):
            (source / name).write_text("{}\n", encoding="utf-8")
        source_files = {
            name: {
                "sha256": hashlib.sha256((source / name).read_bytes()).hexdigest()
            }
            for name in (
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
            )
        }
        (source / "manifest.json").write_text(
            json.dumps(
                {
                    "direction": "en-ja",
                    "bits": 4,
                    "group_size": 64,
                    "source_revision": "fixture",
                    "files": source_files,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        output = root / "output"
        result = subprocess.run(
            [
                "python3",
                str(PRUNER),
                str(source),
                str(output),
                "--keep-decoder-layers",
                "0,3",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        weights = mx.load(str(output / "model.safetensors"))
        assert "decoder.layers.0.fixture" in weights
        assert "decoder.layers.1.fixture" in weights
        assert int(weights["decoder.layers.1.fixture"].item()) == 3
        assert not any(name.startswith("decoder.layers.2.") for name in weights)
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["encoder_layers"] == 3
        assert manifest["decoder_layers"] == 2
        assert manifest["structural_pruning"]["kept_decoder_layers"] == [0, 3]
        assert manifest["distribution_status"] == (
            "provenance-incomplete-not-approved-for-distribution"
        )

        invalid = subprocess.run(
            [
                "python3",
                str(PRUNER),
                str(source),
                str(root / "invalid"),
                "--keep-decoder-layers",
                "3,0",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert invalid.returncode != 0
        assert "ascending" in invalid.stderr

        load_fixture = root / "load-fixture.safetensors"
        model = Marian(
            encoder_layers=2,
            decoder_layers=1,
            encoder_ffn_dimensions=4_096,
            decoder_ffn_dimensions=1_024,
        )
        nn.quantize(model, group_size=64, bits=4)
        mx.eval(model.parameters())
        model.save_weights(str(load_fixture))
        loaded = load_model(
            load_fixture,
            quantization_bits=4,
            quantization_group_size=64,
        )
        assert len(loaded.encoder.layers) == 2
        assert len(loaded.decoder.layers) == 1
        assert loaded.encoder.layers[0].fc1.weight.shape[0] == 4_096
        assert loaded.encoder.layers[0].fc1.scales.shape[0] == 4_096
        assert loaded.decoder.layers[0].fc1.scales.shape[0] == 1_024
        output = loaded.generate_cached([1, 2], maximum_tokens=1)
        assert len(output) <= 1

    print("Marian MLX shallow-decoder pruning contract passed.")


if __name__ == "__main__":
    main()
