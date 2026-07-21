#!/usr/bin/env python3
"""Focused contracts for dataset-derived training provenance."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from training_manifest_provenance import (
    HUMAN_REFERENCE_SEQUENCE_TARGET,
    LEGACY_SEQUENCE_TARGET,
    LICENSED_HUMAN_REFERENCE_SOURCE,
    LICENSED_PARALLEL_SEQUENCE_TARGET,
    MARIAN_SEQUENCE_TARGET,
    MARIAN_SEQUENCE_TARGET_SOURCE,
    QWEN_SEQUENCE_TARGET,
    REFERENCE_TEACHER_EXPERIMENT,
    authenticate_dataset_manifest,
    authenticate_structural_pruning_manifest,
    derive_target_provenance,
    sha256,
)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def reference_row(target_source: str) -> dict:
    qwen = target_source == "qwen"
    return {
        "id": f"row-{target_source}",
        "origin": (
            "strict-local-qwen-reference-distillation"
            if qwen
            else "matched-licensed-human-reference-control"
        ),
        "target": "Qwen final" if qwen else "Licensed reference",
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


def reference_manifest(directory: Path, target_source: str) -> dict:
    origin = (
        "strict-local-qwen-reference-distillation"
        if target_source == "qwen"
        else "matched-licensed-human-reference-control"
    )
    train_path = directory / "train.jsonl"
    valid_path = directory / "valid.jsonl"
    return {
        "schema_version": 1,
        "experiment": REFERENCE_TEACHER_EXPERIMENT,
        "direction": "en-ja",
        "target_source": target_source,
        "promotion_eligible": False,
        "effective_licenses": {
            "train": {"CC-BY-SA-3.0": 1},
            "valid": {"CC-BY-SA-3.0": 1},
        },
        "origins": {"train": {origin: 1}},
        "outputs": {
            "train": {"path": str(train_path), "sha256": sha256(train_path)},
            "valid": {"path": str(valid_path), "sha256": sha256(valid_path)},
        },
    }


def expect_system_exit(action, message: str) -> None:
    try:
        action()
    except SystemExit as error:
        assert message in str(error), error
    else:
        raise AssertionError(f"expected SystemExit containing {message!r}")


def run() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        train_path = root / "train.jsonl"
        valid_path = root / "valid.jsonl"
        train_path.write_text('{"fixture":"train"}\n', encoding="utf-8")
        valid_path.write_text('{"fixture":"valid"}\n', encoding="utf-8")

        for target_source, expected_target in (
            ("qwen", QWEN_SEQUENCE_TARGET),
            ("human-reference", HUMAN_REFERENCE_SEQUENCE_TARGET),
        ):
            manifest = reference_manifest(root, target_source)
            write_json(root / "manifest.json", manifest)
            loaded, metadata = authenticate_dataset_manifest(
                root,
                direction="en-ja",
                train_path=train_path,
                valid_path=valid_path,
            )
            assert metadata is not None
            assert metadata["path"] == str((root / "manifest.json").resolve())
            assert metadata["sha256"] == sha256(root / "manifest.json")
            assert metadata["target_source"] == target_source
            assert metadata["effective_licenses"] == manifest["effective_licenses"]
            assert metadata["outputs_authenticated"] is True
            provenance = derive_target_provenance(
                loaded,
                [reference_row(target_source)],
                fallback_training_description="legacy reviewed targets",
            )
            assert provenance["sequence_target"] == expected_target
            assert "reasoning traces" in provenance["training_description"]

        generic = {"schema_version": 1, "direction": "en-ja"}
        write_json(root / "manifest.json", generic)
        loaded, metadata = authenticate_dataset_manifest(
            root,
            direction="en-ja",
            train_path=train_path,
            valid_path=valid_path,
        )
        assert metadata is not None and metadata["outputs_authenticated"] is False
        fallback = derive_target_provenance(
            loaded,
            [],
            fallback_training_description="legacy reviewed targets",
        )
        assert fallback == {
            "training_description": "legacy reviewed targets",
            "sequence_target": LEGACY_SEQUENCE_TARGET,
        }

        licensed = {
            "schema_version": 1,
            "direction": "en-ja",
            "counts": {"synthetic_train": 0},
            "origins": {
                "train": {
                    "human-kftt-replay": 1,
                    "mimi-shipped-ui-pair": 1,
                }
            },
        }
        licensed_provenance = derive_target_provenance(
            licensed,
            [
                {"origin": "human-kftt-replay"},
                {"origin": "mimi-shipped-ui-pair"},
            ],
            fallback_training_description="legacy reviewed targets",
        )
        assert licensed_provenance["sequence_target"] == (
            LICENSED_PARALLEL_SEQUENCE_TARGET
        )
        assert "licensed human-authored" in licensed_provenance[
            "training_description"
        ]

        full_depth_rows = [
            {
                "origin": origin,
                "source_license": license_name,
                "source_provenance": f"fixture provenance for {origin}",
                "attribution": f"fixture attribution for {origin}",
            }
            for origin, license_name in (
                ("human-kftt-replay", "CC-BY-SA-3.0"),
                ("human-alt-parallel", "CC-BY-4.0"),
                (
                    "human-tatoeba-bidirectional-agreement-filtered",
                    "CC-BY-2.0-FR",
                ),
                (
                    "finalized-japanese-law-translation",
                    "PDL-1.0-compatible-CC-BY-4.0",
                ),
                ("mimi-shipped-ui-pair", "project-owned"),
            )
        ]
        full_depth_manifest = {
            "target_source": LICENSED_HUMAN_REFERENCE_SOURCE,
            "promotion_eligible": False,
            "effective_licenses": {
                "train": {
                    "CC-BY-SA-3.0": 1,
                    "CC-BY-4.0": 1,
                    "CC-BY-2.0-FR": 1,
                    "PDL-1.0-compatible-CC-BY-4.0": 1,
                    "project-owned": 1,
                }
            },
        }
        full_depth_provenance = derive_target_provenance(
            full_depth_manifest,
            full_depth_rows,
            fallback_training_description="must not be used",
        )
        assert full_depth_provenance["sequence_target"] == (
            LICENSED_PARALLEL_SEQUENCE_TARGET
        )
        expect_system_exit(
            lambda: derive_target_provenance(
                full_depth_manifest,
                [
                    {
                        **full_depth_rows[0],
                        "origin": "unreviewed-synthetic-origin",
                    }
                ],
                fallback_training_description="must not be used",
            ),
            "contains unapproved origins",
        )

        marian_manifest = {
            "target_source": MARIAN_SEQUENCE_TARGET_SOURCE,
            "promotion_eligible": False,
            "references_exposed_to_teacher": False,
            "private_reasoning_traces_used": False,
            "effective_licenses": {"train": {"CC-BY-SA-3.0": 1}},
            "teacher": {
                "weights_sha256": "a" * 64,
                "repository": "teacher/repository",
                "revision": "teacher-revision",
                "license": "CC-BY-SA-4.0",
            },
        }
        marian_row = {
            "target_source": MARIAN_SEQUENCE_TARGET_SOURCE,
            "teacher_model_revision": "teacher/repository@teacher-revision",
            "reference_target_sha256": "b" * 64,
        }
        marian_provenance = derive_target_provenance(
            marian_manifest,
            [marian_row],
            fallback_training_description="must not be used",
        )
        assert marian_provenance["sequence_target"] == MARIAN_SEQUENCE_TARGET
        assert "references were not exposed" in marian_provenance[
            "training_description"
        ]
        expect_system_exit(
            lambda: derive_target_provenance(
                marian_manifest,
                [{**marian_row, "teacher_model_revision": "wrong"}],
                fallback_training_description="must not be used",
            ),
            "contradict declared target provenance",
        )

        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "model.safetensors").write_bytes(b"fixture weights")
        pruning_manifest = {
            "method": "fixture decoder pruning",
            "encoder_layers": 6,
            "source_decoder_layers": [0, 1, 2, 3, 4, 5],
            "kept_decoder_layers": [0, 5],
            "decoder_layers": 2,
            "promotion_eligible": False,
            "private_reasoning_traces_used": False,
            "files": {
                "model.safetensors": {
                    "sha256": sha256(checkpoint / "model.safetensors")
                }
            },
        }
        write_json(
            checkpoint / "mimi_structural_pruning_manifest.json",
            pruning_manifest,
        )
        structural = authenticate_structural_pruning_manifest(checkpoint)
        assert structural is not None
        assert structural["kept_decoder_layers"] == [0, 5]
        assert structural["model_sha256"] == sha256(
            checkpoint / "model.safetensors"
        )
        pruning_manifest["files"]["model.safetensors"]["sha256"] = "0" * 64
        write_json(
            checkpoint / "mimi_structural_pruning_manifest.json",
            pruning_manifest,
        )
        expect_system_exit(
            lambda: authenticate_structural_pruning_manifest(checkpoint),
            "does not authenticate model weights",
        )

        (root / "manifest.json").unlink()
        loaded, metadata = authenticate_dataset_manifest(
            root,
            direction="en-ja",
            train_path=train_path,
            valid_path=valid_path,
        )
        assert loaded is None and metadata is None

        tampered = reference_manifest(root, "qwen")
        tampered["outputs"]["train"]["sha256"] = "0" * 64
        write_json(root / "manifest.json", tampered)
        expect_system_exit(
            lambda: authenticate_dataset_manifest(
                root,
                direction="en-ja",
                train_path=train_path,
                valid_path=valid_path,
            ),
            "train hash differs",
        )

        manifest = reference_manifest(root, "qwen")
        expect_system_exit(
            lambda: derive_target_provenance(
                manifest,
                [reference_row("human-reference")],
                fallback_training_description="legacy reviewed targets",
            ),
            "row count differs",
        )

    print("training manifest provenance contracts passed")


if __name__ == "__main__":
    run()
