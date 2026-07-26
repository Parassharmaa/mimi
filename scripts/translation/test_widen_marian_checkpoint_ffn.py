#!/usr/bin/env python3
"""Contract test for zero-output-column Marian FFN widening."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as functional
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[2]
WIDENER = ROOT / "scripts/translation/widen_marian_checkpoint_ffn.py"


def ffn(state: dict[str, torch.Tensor], prefix: str, hidden: torch.Tensor) -> torch.Tensor:
    first = functional.linear(
        hidden,
        state[f"{prefix}fc1.weight"],
        state[f"{prefix}fc1.bias"],
    )
    return functional.linear(
        functional.silu(first),
        state[f"{prefix}fc2.weight"],
        state[f"{prefix}fc2.bias"],
    )


def main() -> None:
    torch.manual_seed(20260726)
    with tempfile.TemporaryDirectory(prefix="mimi-ffn-widening-") as temporary:
        root = Path(temporary)
        source = root / "source"
        source.mkdir()
        state: dict[str, torch.Tensor] = {"model.shared.weight": torch.randn(8, 4)}
        for stack, layers in (("encoder", 2), ("decoder", 3)):
            for index in range(layers):
                prefix = f"model.{stack}.layers.{index}."
                state[f"{prefix}fc1.weight"] = torch.randn(3, 4)
                state[f"{prefix}fc1.bias"] = torch.randn(3)
                state[f"{prefix}fc2.weight"] = torch.randn(4, 3)
                state[f"{prefix}fc2.bias"] = torch.randn(4)
                state[f"{prefix}self_attn.q_proj.weight"] = torch.randn(4, 4)
        save_file(state, source / "model.safetensors", metadata={"format": "pt"})
        (source / "config.json").write_text(
            json.dumps(
                {
                    "encoder_layers": 2,
                    "decoder_layers": 3,
                    "encoder_ffn_dim": 3,
                    "decoder_ffn_dim": 3,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        for name in (
            "generation_config.json",
            "source.spm",
            "target.spm",
            "tokenizer_config.json",
            "vocab.json",
        ):
            (source / name).write_text("{}\n", encoding="utf-8")
        source_weights = source / "model.safetensors"
        identity = root / "identity.json"
        identity.write_text(
            json.dumps(
                {
                    "source_repository": "fixture/marian",
                    "source_revision": "fixture-revision",
                    "source_weights_sha256": hashlib.sha256(
                        source_weights.read_bytes()
                    ).hexdigest(),
                }
            )
            + "\n",
            encoding="utf-8",
        )

        output = root / "output"
        result = subprocess.run(
            [
                "python3",
                str(WIDENER),
                str(source),
                str(output),
                "--encoder-ffn-dim",
                "7",
                "--decoder-ffn-dim",
                "8",
                "--identity-manifest",
                str(identity),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        widened = load_file(output / "model.safetensors")
        original = load_file(source_weights)
        hidden = torch.randn(5, 4)
        for stack, layers, width in (("encoder", 2, 7), ("decoder", 3, 8)):
            for index in range(layers):
                prefix = f"model.{stack}.layers.{index}."
                assert widened[f"{prefix}fc1.weight"].shape == (width, 4)
                assert widened[f"{prefix}fc2.weight"].shape == (4, width)
                assert torch.equal(
                    widened[f"{prefix}fc2.weight"][:, 3:],
                    torch.zeros(4, width - 3),
                )
                original_output = ffn(original, prefix, hidden)
                widened_output = ffn(widened, prefix, hidden)
                assert torch.allclose(
                    original_output,
                    widened_output,
                    atol=1e-6,
                    rtol=1e-6,
                ), float((original_output - widened_output).abs().max())
                assert torch.equal(
                    original[f"{prefix}self_attn.q_proj.weight"],
                    widened[f"{prefix}self_attn.q_proj.weight"],
                )

        configuration = json.loads((output / "config.json").read_text())
        assert configuration["encoder_ffn_dim"] == 7
        assert configuration["decoder_ffn_dim"] == 8
        manifest = json.loads(
            (output / "mimi_ffn_widening_manifest.json").read_text()
        )
        assert manifest["method"] == "zero-output-column-full-marian-ffn-widening"
        assert manifest["source_encoder_ffn_dim"] == 3
        assert manifest["encoder_ffn_dim"] == 7
        assert manifest["source_decoder_ffn_dim"] == 3
        assert manifest["decoder_ffn_dim"] == 8
        assert manifest["training_authorized"] is False
        assert manifest["promotion_eligible"] is False
        assert manifest["private_reasoning_traces_used"] is False

    print("Marian full-model FFN widening contract passed")


if __name__ == "__main__":
    main()
