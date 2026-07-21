#!/usr/bin/env python3
"""One-update smoke test for the full-parameter Marian training entrypoint."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def row(identifier: str, source: str, target: str, origin: str = "smoke-fixture") -> dict:
    return {
        "id": identifier,
        "source_id": identifier,
        "source_language": "en-US",
        "target_language": "ja-JP",
        "source": source,
        "target": target,
        "origin": origin,
        "source_license": "CC0-1.0",
        "source_provenance": "local smoke test",
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-marian-train-smoke-") as temporary:
        work = Path(temporary)
        dataset = work / "dataset"
        output = work / "output"
        qat_output = work / "qat-output"
        checkpoints = work / "checkpoints"
        dataset.mkdir()
        write_jsonl(
            dataset / "train.jsonl",
            [
                row(
                    "train-1",
                    "Please open the settings window.",
                    "設定ウインドウを開いてください。",
                    "human-kftt-replay",
                ),
                row("train-2", "The microphone is muted.", "マイクはミュートされています。"),
            ],
        )
        write_jsonl(
            dataset / "valid.jsonl",
            [
                row("valid-1", "Translation is ready.", "翻訳の準備ができました。"),
                row("valid-2", "Try again after restarting Mimi.", "Mimiを再起動してからもう一度お試しください。"),
            ],
        )
        subprocess.run(
            [
                sys.executable,
                "scripts/translation/train_marian_distillation.py",
                str(dataset),
                str(output),
                "--direction",
                "en-ja",
                "--repository",
                "Mitsua/elan-mt-bt-en-ja",
                "--revision",
                "02c48e7031386cd2d41974b0ff1aaf52f010c5fa",
                "--device",
                "mps",
                "--batch-size",
                "2",
                "--gradient-accumulation",
                "1",
                "--max-steps",
                "1",
                "--evaluation-steps",
                "1",
                "--warmup-steps",
                "0",
                "--max-source-tokens",
                "48",
                "--max-target-tokens",
                "48",
                "--frozen-base-kl-weight",
                "0.5",
                "--l2-to-base-weight",
                "0.01",
                "--domain-loss-weight-start",
                "0.25",
                "--domain-loss-weight-end",
                "1.0",
                "--curriculum-ramp-steps",
                "1",
                "--checkpoint-directory",
                str(checkpoints),
            ],
            cwd=ROOT,
            check=True,
        )
        manifest = json.loads(
            (output / "mimi_training_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["direction"] == "en-ja"
        assert manifest["dataset"]["train_rows"] == 2
        assert manifest["hyperparameters"]["frozen_base_kl_weight"] == 0.5
        assert manifest["history"][-1]["training_objective"]["domain_loss_weight"] == 0.25
        assert len(manifest["history"]) == 2
        assert len(manifest["checkpoints"]) == 1
        assert (checkpoints / "step-0000001" / "model.safetensors").is_file()
        assert (output / "model.safetensors").is_file()
        subprocess.run(
            [
                sys.executable,
                "scripts/translation/train_marian_distillation.py",
                str(dataset),
                str(qat_output),
                "--direction",
                "en-ja",
                "--repository",
                "Mitsua/elan-mt-bt-en-ja",
                "--revision",
                "02c48e7031386cd2d41974b0ff1aaf52f010c5fa",
                "--initial-checkpoint",
                str(output),
                "--preservation-checkpoint",
                str(output),
                "--device",
                "mps",
                "--batch-size",
                "2",
                "--gradient-accumulation",
                "1",
                "--max-steps",
                "1",
                "--evaluation-steps",
                "1",
                "--warmup-steps",
                "0",
                "--max-source-tokens",
                "48",
                "--max-target-tokens",
                "48",
                "--frozen-base-kl-weight",
                "0.5",
                "--l2-to-base-weight",
                "0.01",
                "--domain-loss-weight-start",
                "0.25",
                "--domain-loss-weight-end",
                "1.0",
                "--curriculum-ramp-steps",
                "1",
                "--mlx-fake-quantization-bits",
                "4",
                "--mlx-fake-quantization-group-size",
                "64",
            ],
            cwd=ROOT,
            check=True,
        )
        qat_manifest = json.loads(
            (qat_output / "mimi_training_manifest.json").read_text(encoding="utf-8")
        )
        assert qat_manifest["hyperparameters"]["mlx_fake_quantization_bits"] == 4
        assert qat_manifest["hyperparameters"]["mlx_fake_quantization_group_size"] == 64
        assert "model.shared" in qat_manifest["hyperparameters"]["mlx_fake_quantized_modules"]
        assert qat_manifest["initial_checkpoint"]["path"] == str(output)
        assert qat_manifest["preservation_checkpoint"]["path"] == str(output)
        assert qat_manifest["initial_checkpoint"]["lineage_manifests"] == [
            {
                "path": str(output / "mimi_training_manifest.json"),
                "sha256": qat_manifest["initial_checkpoint"][
                    "training_manifest_sha256"
                ],
            }
        ]
        assert qat_manifest["preservation_checkpoint"]["lineage_manifests"] == [
            {
                "path": str(output / "mimi_training_manifest.json"),
                "sha256": qat_manifest["preservation_checkpoint"][
                    "training_manifest_sha256"
                ],
            }
        ]
        assert len(qat_manifest["history"]) == 2
        assert (qat_output / "model.safetensors").is_file()
    print("Mimi Marian distillation training smoke passed.")


if __name__ == "__main__":
    main()
