#!/usr/bin/env python3
"""Offline artifact contract for adjacent Marian checkpoint averaging."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[2]


def checkpoint(
    root: Path,
    step: int,
    chrf: float,
    loss: float,
    retention_chrf: float = 10.0,
) -> None:
    path = root / f"step-{step:07d}"
    path.mkdir()
    save_file(
        {"weight": torch.tensor([float(step // 100)], dtype=torch.float32)},
        str(path / "model.safetensors"),
    )
    (path / "config.json").write_text("{}\n", encoding="utf-8")
    (path / "mimi_training_manifest.json").write_text(
        json.dumps(
            {
                "direction": "en-ja",
                "student_repository": "fixture/model",
                "student_revision": "fixture-revision",
                "license": "CC-BY-SA-4.0",
                "dataset": {"train_sha256": "train", "valid_sha256": "valid"},
                "checkpoint_step": step,
                "history": [
                    {
                        "step": 0,
                        "chrf_pp": 2.0,
                        "loss": 5.0,
                        "slices": {
                            "origin": {
                                "reviewed-gpt-teacher": {"cases": 1, "chrf_pp": 2.0},
                                "human-kftt-replay": {"cases": 1, "chrf_pp": 10.0},
                            }
                        },
                    }
                ],
                "checkpoint_metrics": {
                    "chrf_pp": chrf,
                    "loss": loss,
                    "slices": {
                        "origin": {
                            "reviewed-gpt-teacher": {"cases": 1, "chrf_pp": chrf},
                            "human-kftt-replay": {"cases": 1, "chrf_pp": retention_chrf},
                        }
                    },
                },
            }
        ) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-checkpoint-average-") as temporary:
        work = Path(temporary)
        checkpoints = work / "checkpoints"
        output = work / "output"
        checkpoints.mkdir()
        checkpoint(checkpoints, 100, 1.0, 4.0)
        checkpoint(checkpoints, 200, 3.0, 3.0)
        checkpoint(checkpoints, 300, 4.0, 2.0)
        checkpoint(checkpoints, 400, 3.0, 1.0)
        subprocess.run(
            [
                sys.executable,
                "scripts/translation/average_marian_checkpoints.py",
                str(checkpoints),
                str(output),
                "--count",
                "3",
                "--selection-origin",
                "reviewed-gpt-teacher",
                "--selection-origin",
                "human-kftt-replay",
                "--retention-origin",
                "human-kftt-replay",
                "--maximum-retention-regression",
                "0.5",
            ],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        averaged = load_file(str(output / "model.safetensors"))
        assert torch.equal(averaged["weight"], torch.tensor([3.0]))
        manifest = json.loads(
            (output / "mimi_checkpoint_averaging_manifest.json").read_text()
        )
        assert [row["step"] for row in manifest["selected_checkpoints"]] == [200, 300, 400]
        assert manifest["count"] == 3
        assert manifest["selection_origin"] is None
        assert manifest["selection_origins"] == [
            "reviewed-gpt-teacher",
            "human-kftt-replay",
        ]
        assert manifest["retention_origin"] == "human-kftt-replay"
    print("Mimi adjacent checkpoint averaging contract passed.")


if __name__ == "__main__":
    main()
