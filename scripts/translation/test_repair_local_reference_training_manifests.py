#!/usr/bin/env python3
"""Focused idempotency contract for reference-run manifest repair."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from repair_local_reference_training_manifests import Run, repair_run
from training_manifest_provenance import (
    QWEN_SEQUENCE_TARGET,
    QWEN_TRAINING_DESCRIPTION,
    REFERENCE_TEACHER_EXPERIMENT,
    sha256,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        run = Run(
            dataset="work/dataset",
            model="models/run",
            checkpoints="models/run-checkpoints",
            direction="en-ja",
            target_source="qwen",
        )
        dataset = root / run.dataset
        dataset.mkdir(parents=True)
        row = {
            "id": "qwen-row",
            "origin": "strict-local-qwen-reference-distillation",
            "target": "Qwen final",
            "qwen_candidate": "Qwen final",
            "training_only": True,
            "promotion_eligible": False,
            "source_license": "CC-BY-SA-3.0",
            "reference_provenance": "licensed fixture",
            "review_status": "hidden-reference-metric-filtered-provisional",
            "quality_control": {
                "reasoning_trace_requested_or_stored": False,
                "reference_exposed_to_teacher": False,
            },
        }
        train_path = dataset / "train.jsonl"
        valid_path = dataset / "valid.jsonl"
        train_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        valid_path.write_text(json.dumps({"id": "valid"}) + "\n", encoding="utf-8")
        write_json(
            dataset / "manifest.json",
            {
                "schema_version": 1,
                "experiment": REFERENCE_TEACHER_EXPERIMENT,
                "direction": "en-ja",
                "target_source": "qwen",
                "promotion_eligible": False,
                "effective_licenses": {
                    "train": {"CC-BY-SA-3.0": 1},
                    "valid": {"CC-BY-SA-3.0": 1},
                },
                "origins": {
                    "train": {"strict-local-qwen-reference-distillation": 1}
                },
                "outputs": {
                    "train": {"path": str(train_path), "sha256": sha256(train_path)},
                    "valid": {"path": str(valid_path), "sha256": sha256(valid_path)},
                },
            },
        )

        checkpoint = root / run.checkpoints / "step-0000001"
        checkpoint.mkdir(parents=True)
        checkpoint_relative = checkpoint.relative_to(root)
        legacy = {
            "direction": "en-ja",
            "training_description": "reviewed legacy targets",
            "dataset": {
                "train_sha256": sha256(train_path),
                "valid_sha256": sha256(valid_path),
                "train_rows": 1,
                "valid_rows": 1,
            },
            "objective": {"sequence_target": "reviewed canonical translation"},
            "checkpoints": [{"path": str(checkpoint_relative), "step": 1}],
        }
        model_manifest_path = root / run.model / "mimi_training_manifest.json"
        checkpoint_manifest_path = checkpoint / "mimi_training_manifest.json"
        write_json(model_manifest_path, legacy)
        write_json(checkpoint_manifest_path, legacy)

        changed = repair_run(root, run)
        assert changed == [model_manifest_path, checkpoint_manifest_path.resolve()]
        repaired = json.loads(model_manifest_path.read_text(encoding="utf-8"))
        assert repaired["training_description"] == QWEN_TRAINING_DESCRIPTION
        assert repaired["objective"]["sequence_target"] == QWEN_SEQUENCE_TARGET
        assert repaired["dataset_manifest"]["target_source"] == "qwen"
        assert repaired["dataset_manifest"]["outputs_authenticated"] is True
        first_bytes = model_manifest_path.read_bytes()

        assert repair_run(root, run) == []
        assert model_manifest_path.read_bytes() == first_bytes

    print("reference training manifest repair contracts passed")


if __name__ == "__main__":
    run()
