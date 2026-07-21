#!/usr/bin/env python3
"""Contract test for full-precision Marian decoder pruning."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[2]
PRUNER = ROOT / "scripts/translation/prune_marian_checkpoint_decoder.py"


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-checkpoint-pruning-") as temporary:
        root = Path(temporary)
        source = root / "source"
        source.mkdir()
        save_file(
            {
                **{
                    f"model.encoder.layers.{index}.self_attn.q_proj.weight": torch.tensor(
                        [float(index)]
                    )
                    for index in range(3)
                },
                **{
                    f"model.encoder.layers.{index}.self_attn_layer_norm.weight": torch.tensor(
                        [float(index + 1)]
                    )
                    for index in range(3)
                },
                **{
                    f"model.encoder.layers.{index}.fc1.weight": torch.tensor(
                        [[1.0, 2.0], [3.0, 4.0]]
                    )
                    for index in range(3)
                },
                **{
                    f"model.encoder.layers.{index}.fc1.bias": torch.tensor([1.0, 2.0])
                    for index in range(3)
                },
                **{
                    f"model.encoder.layers.{index}.fc2.weight": torch.tensor(
                        [[5.0, 6.0], [7.0, 8.0]]
                    )
                    for index in range(3)
                },
                **{
                    f"model.encoder.layers.{index}.fc2.bias": torch.tensor([9.0, 10.0])
                    for index in range(3)
                },
                **{
                    f"model.decoder.layers.{index}.fixture": torch.tensor([index])
                    for index in range(4)
                },
                "model.shared.fixture": torch.tensor([99]),
            },
            source / "model.safetensors",
            metadata={"format": "pt"},
        )
        (source / "config.json").write_text(
            json.dumps(
                {"encoder_layers": 3, "decoder_layers": 4, "encoder_ffn_dim": 2}
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
        identity_manifest = root / "identity-manifest.json"
        identity_manifest.write_text(
            json.dumps(
                {
                    "source_repository": "fixture/repository",
                    "source_revision": "fixture-revision",
                    "source_weights_sha256": __import__("hashlib").sha256(
                        (source / "model.safetensors").read_bytes()
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
                str(PRUNER),
                str(source),
                str(output),
                "--keep-decoder-layers",
                "0,3",
                "--append-identity-encoder-layers",
                "2",
                "--encoder-ffn-dim",
                "4",
                "--identity-manifest",
                str(identity_manifest),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        weights = load_file(output / "model.safetensors")
        assert int(weights["model.decoder.layers.1.fixture"].item()) == 3
        assert not any(name.startswith("model.decoder.layers.2.") for name in weights)
        for index in (3, 4):
            assert float(
                weights[
                    f"model.encoder.layers.{index}.self_attn.q_proj.weight"
                ].item()
            ) == 0.0
            assert float(
                weights[
                    f"model.encoder.layers.{index}.self_attn_layer_norm.weight"
                ].item()
            ) == 1.0
        configuration = json.loads((output / "config.json").read_text())
        assert configuration["encoder_layers"] == 5
        assert configuration["encoder_ffn_dim"] == 4
        assert configuration["decoder_layers"] == 2
        assert torch.equal(
            weights["model.encoder.layers.0.fc1.weight"],
            torch.tensor([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [3.0, 4.0]]),
        )
        assert torch.equal(
            weights["model.encoder.layers.0.fc2.weight"][:, 2:],
            torch.zeros((2, 2)),
        )
        manifest = json.loads(
            (output / "mimi_structural_pruning_manifest.json").read_text()
        )
        assert manifest["kept_decoder_layers"] == [0, 3]
        assert manifest["source_encoder_layers"] == [0, 1, 2]
        assert manifest["appended_identity_encoder_layers"] == 2
        assert manifest["encoder_layers"] == 5
        assert manifest["source_encoder_ffn_dim"] == 2
        assert manifest["encoder_ffn_dim"] == 4
        assert manifest["method"] == (
            "wide-encoder-shallow-decoder-reallocation-before-distillation"
        )
        assert manifest["private_reasoning_traces_used"] is False
        assert manifest["source"]["repository"] == "fixture/repository"
        assert manifest["source"]["identity_manifest"]["sha256"]

    print("Marian full-precision decoder pruning contract passed.")


if __name__ == "__main__":
    main()
