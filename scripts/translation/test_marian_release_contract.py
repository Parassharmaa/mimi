#!/usr/bin/env python3
"""Focused contracts for hash-bound Marian release provenance."""

from __future__ import annotations

import gzip
import json
import tempfile
from pathlib import Path

from build_marian_release_contract import (
    ReleaseTrace,
    conversion_provenance_status,
    dataset_policy_status,
    direct_engine_lineage_record,
    extract_attributions,
    sha256,
    validate_pack,
    validate_translation_memory,
    visit_generalist_lineage,
)
from package_elanmt_mlx_experts import generalist_lineage_record


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def build_training_model(root: Path, *, excluded: bool) -> Path:
    model = root / "model"
    model.mkdir()
    weights = model / "model.safetensors"
    weights.write_bytes(b"authenticated fixture weights")
    for split in ("train", "valid"):
        row = {
            "id": f"fixture-{split}",
            "origin": "fixture-human-data",
            "source": "Hello",
            "target": "こんにちは",
            "source_license": "project-owned",
            "source_provenance": "Mimi release-contract fixture",
        }
        if excluded:
            row.update({"promotion_eligible": False, "training_only": True})
        (root / f"{split}.jsonl").write_text(
            json.dumps(row, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    write_json(
        root / "manifest.json",
        {
            "promotion_eligible": not excluded,
            "outputs": {
                split: {
                    "path": str(root / f"{split}.jsonl"),
                    "sha256": sha256(root / f"{split}.jsonl"),
                }
                for split in ("train", "valid")
            },
        },
    )
    write_json(
        model / "mimi_training_manifest.json",
        {
            "dataset": {
                "train_path": str(root / "train.jsonl"),
                "train_sha256": sha256(root / "train.jsonl"),
                "train_rows": 1,
                "valid_path": str(root / "valid.jsonl"),
                "valid_sha256": sha256(root / "valid.jsonl"),
                "valid_rows": 1,
            },
            "student_repository": "fixture/elan-mt",
            "student_revision": "fixture-revision",
            "license": "CC-BY-SA-4.0",
        },
    )
    return model


def assert_case(root: Path, *, excluded: bool, use_manifest_path: bool) -> None:
    model = build_training_model(root, excluded=excluded)
    trace = ReleaseTrace(root)
    lineage = model / "mimi_training_manifest.json" if use_manifest_path else model
    visit_generalist_lineage(trace, lineage, sha256(model / "model.safetensors"))
    summary, sidecar = extract_attributions(trace)
    policy = dataset_policy_status(trace)
    expected_rows = 2 if excluded else 0
    assert summary["datasetRows"] == 2
    assert summary["promotionExcludedRows"] == expected_rows
    assert summary["promotionExcludedOrigins"] == (
        {"fixture-human-data": 2} if excluded else {}
    )
    assert summary["promotionExclusionReasons"] == (
        {"promotion_eligible=false": 2, "training_only=true": 2}
        if excluded
        else {}
    )
    assert summary["licenses"] == {"project-owned": 2}
    assert sidecar == b""
    assert policy["promotionEligible"] is (not excluded)
    assert policy["filesWithoutManifest"] == []
    assert len(policy["promotionIneligibleManifests"]) == (1 if excluded else 0)


def run() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        clean = root / "clean"
        clean.mkdir()
        assert_case(clean, excluded=False, use_manifest_path=False)
        blocked = root / "blocked"
        blocked.mkdir()
        assert_case(blocked, excluded=True, use_manifest_path=True)
        direct = blocked / "model"
        direct_record = generalist_lineage_record(
            direct, sha256(direct / "model.safetensors")
        )
        assert direct_record["kind"] == "direct-training-checkpoint"
        direct_trace = ReleaseTrace(blocked)
        expert_record = direct_engine_lineage_record(
            direct_trace,
            direct,
            sha256(direct / "model.safetensors"),
            sha256(direct / "mimi_training_manifest.json"),
        )
        assert expert_record["modelSha256"] == sha256(
            direct / "model.safetensors"
        )
        assert len(direct_trace.training_manifests) == 1

        transform_root = root / "transform"
        transform_root.mkdir()
        transform_weights = transform_root / "model.safetensors"
        transform_weights.write_bytes(b"transformed fixture weights")
        write_json(
            transform_root / "mimi_checkpoint_interpolation_manifest.json",
            {
                "operation": "linear-checkpoint-interpolation",
                "output": {
                    "path": str(transform_root),
                    "model_sha256": sha256(transform_weights),
                },
                "parent": {
                    "path": str(direct),
                    "model_sha256": sha256(direct / "model.safetensors"),
                },
                "adapted": {
                    "path": str(direct),
                    "model_sha256": sha256(direct / "model.safetensors"),
                },
            },
        )
        transform_trace = ReleaseTrace(root)
        transform_trace.visit_model(str(transform_root), sha256(transform_weights))
        assert len(transform_trace.lineage_manifests) == 1
        assert len(transform_trace.training_manifests) == 1
        averaged = root / "averaged.json"
        write_json(
            averaged,
            {
                "operation": (
                    "arithmetic-mean-of-best-adjacent-full-precision-checkpoints"
                ),
                "output": {"model_sha256": "a" * 64},
            },
        )
        averaged_record = generalist_lineage_record(averaged, "a" * 64)
        assert averaged_record["kind"].startswith("arithmetic-mean")

        memory_pack = root / "memory-pack"
        (memory_pack / "memory").mkdir(parents=True)
        memory_training = root / "memory-train.jsonl"
        memory_training.write_text(
            json.dumps(
                {
                    "source": "（立入調査等）",
                    "target": "(On-site Inspections)",
                    "source_id": "law-1:tu-1",
                    "source_license": "PDL-1.0-compatible-CC-BY-4.0",
                    "source_provenance": "https://example.test/law-1",
                    "attribution": "fixture attribution",
                    "training_only": True,
                    "promotion_eligible": False,
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        memory_audit = root / "memory-audit.json.gz"
        with gzip.open(memory_audit, "wt", encoding="utf-8") as archive:
            json.dump(
                {
                    "schemaVersion": 1,
                    "doesNotAuthorizeAppIntegration": True,
                    "sourceLicense": "PDL-1.0-compatible-CC-BY-4.0",
                    "trainingData": {"sha256": sha256(memory_training)},
                    "counts": {"entries": 1, "trainingRows": 1},
                    "policy": {"selection": "fixture"},
                },
                archive,
            )
        runtime = memory_pack / "memory/exact-translation-memory.json"
        write_json(
            runtime,
            {
                "schemaVersion": 1,
                "doesNotAuthorizeAppIntegration": True,
                "normalization": "NFKC then Unicode-whitespace collapse",
                "sourceLicense": "PDL-1.0-compatible-CC-BY-4.0",
                "trainingDataSHA256": sha256(memory_training),
                "auditSHA256": sha256(memory_audit),
                "entries": {
                    "en-ja": {},
                    "ja-en": {"(立入調査等)": "(On-site Inspections)"},
                },
            },
        )
        memory_manifest = {
            "translationMemory": {
                "path": "memory/exact-translation-memory.json",
                "normalization": "NFKC then Unicode-whitespace collapse",
                "sourceLicense": "PDL-1.0-compatible-CC-BY-4.0",
                "trainingDataSHA256": sha256(memory_training),
                "auditSHA256": sha256(memory_audit),
                "entries": 1,
            }
        }
        memory_record = validate_translation_memory(
            memory_pack,
            memory_manifest,
            memory_audit,
            memory_training,
        )
        assert memory_record is not None
        assert memory_record["entries"] == 1
        assert memory_record["promotionEligible"] is False
        assert memory_record["trainingData"]["trainingOnlyRows"] == 1

        shared_pack = root / "shared-pack"
        (shared_pack / "shared").mkdir(parents=True)
        shared_tokenizer = shared_pack / "shared/tokenizer.json"
        shared_tokenizer.write_bytes(b"authenticated shared tokenizer")
        write_json(
            shared_pack / "manifest.json",
            {
                "files": {
                    "shared/tokenizer.json": {
                        "bytes": shared_tokenizer.stat().st_size,
                        "sha256": sha256(shared_tokenizer),
                    }
                },
                "format": "mimi-mlx-marian-moe-v2",
                "sharedTokenizer": "shared/tokenizer.json",
            },
        )
        shared_manifest, shared_bytes = validate_pack(shared_pack)
        assert shared_manifest["sharedTokenizer"] == "shared/tokenizer.json"
        assert shared_bytes > shared_tokenizer.stat().st_size

        conversion_pack = root / "conversion-pack"
        engine_names = (
            "generalist-en-ja",
            "generalist-ja-en",
            "formal-en-ja",
            "legal-ja-en",
        )
        conversion_root_manifest = {"engines": {}}
        for engine_name in engine_names:
            engine = conversion_pack / "engines" / engine_name
            engine.mkdir(parents=True)
            write_json(
                engine / "manifest.json",
                {
                    "files": {
                        "model.safetensors": {
                            "sha256": f"output-{engine_name}",
                        }
                    }
                },
            )
            conversion_root_manifest["engines"][engine_name] = {
                "sourceWeightsSha256": f"source-{engine_name}"
            }
        conversion_records, missing_conversions = conversion_provenance_status(
            conversion_pack,
            conversion_root_manifest,
        )
        assert conversion_records == {}
        assert missing_conversions == list(engine_names)
    print("Marian release-contract checks passed")


if __name__ == "__main__":
    run()
