#!/usr/bin/env python3
"""Repair provenance metadata in the four local reference-teacher runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training_manifest_provenance import (
    authenticate_dataset_manifest,
    derive_target_provenance,
    sha256,
)


@dataclass(frozen=True)
class Run:
    dataset: str
    model: str
    checkpoints: str
    direction: str
    target_source: str


RUNS = (
    Run(
        dataset="Research/translation/work/local-qwen-reference-en-ja-v1",
        model="Research/translation/models/elanmt-local-qwen-reference-en-ja-v1",
        checkpoints=(
            "Research/translation/models/"
            "elanmt-local-qwen-reference-en-ja-v1-checkpoints"
        ),
        direction="en-ja",
        target_source="qwen",
    ),
    Run(
        dataset="Research/translation/work/local-qwen-reference-ja-en-v1",
        model="Research/translation/models/elanmt-local-qwen-reference-ja-en-v1",
        checkpoints=(
            "Research/translation/models/"
            "elanmt-local-qwen-reference-ja-en-v1-checkpoints"
        ),
        direction="ja-en",
        target_source="qwen",
    ),
    Run(
        dataset="Research/translation/work/local-human-reference-control-en-ja-v1",
        model=(
            "Research/translation/models/"
            "elanmt-local-human-reference-control-en-ja-v1"
        ),
        checkpoints=(
            "Research/translation/models/"
            "elanmt-local-human-reference-control-en-ja-v1-checkpoints"
        ),
        direction="en-ja",
        target_source="human-reference",
    ),
    Run(
        dataset="Research/translation/work/local-human-reference-control-ja-en-v1",
        model=(
            "Research/translation/models/"
            "elanmt-local-human-reference-control-ja-en-v1"
        ),
        checkpoints=(
            "Research/translation/models/"
            "elanmt-local-human-reference-control-ja-en-v1-checkpoints"
        ),
        direction="ja-en",
        target_source="human-reference",
    ),
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def serialize(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def declared_manifest_paths(root: Path, model_manifest: dict[str, Any]) -> set[Path]:
    paths: set[Path] = set()
    for checkpoint in model_manifest.get("checkpoints", []):
        raw_path = checkpoint.get("path") if isinstance(checkpoint, dict) else None
        if not raw_path:
            raise SystemExit("training manifest has a checkpoint without a path")
        checkpoint_path = Path(raw_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = root / checkpoint_path
        paths.add(checkpoint_path.resolve() / "mimi_training_manifest.json")
    return paths


def repair_run(root: Path, run: Run, *, write: bool = True) -> list[Path]:
    dataset_directory = root / run.dataset
    train_path = dataset_directory / "train.jsonl"
    valid_path = dataset_directory / "valid.jsonl"
    train_rows = load_rows(train_path)
    valid_rows = load_rows(valid_path)
    dataset_manifest, dataset_manifest_metadata = authenticate_dataset_manifest(
        dataset_directory,
        direction=run.direction,
        train_path=train_path,
        valid_path=valid_path,
    )
    if dataset_manifest is None or dataset_manifest_metadata is None:
        raise SystemExit(f"repair requires a dataset manifest: {dataset_directory}")
    if dataset_manifest.get("target_source") != run.target_source:
        raise SystemExit(
            f"repair target_source differs for {dataset_directory}: "
            f"expected {run.target_source}, found {dataset_manifest.get('target_source')}"
        )
    provenance = derive_target_provenance(
        dataset_manifest,
        train_rows,
        fallback_training_description="reviewed target fallback must not be used",
    )

    model_manifest_path = root / run.model / "mimi_training_manifest.json"
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    checkpoint_root = (root / run.checkpoints).resolve()
    declared_paths = declared_manifest_paths(root, model_manifest)
    discovered_paths = {
        path.resolve()
        for path in checkpoint_root.glob("step-*/mimi_training_manifest.json")
    }
    if declared_paths != discovered_paths:
        missing = sorted(str(path) for path in declared_paths - discovered_paths)
        undeclared = sorted(str(path) for path in discovered_paths - declared_paths)
        raise SystemExit(
            f"checkpoint manifest set differs for {run.model}; "
            f"missing={missing}, undeclared={undeclared}"
        )

    changed: list[Path] = []
    for manifest_path in [model_manifest_path, *sorted(discovered_paths)]:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("direction") != run.direction:
            raise SystemExit(f"training direction differs in {manifest_path}")
        dataset = payload.get("dataset")
        if not isinstance(dataset, dict):
            raise SystemExit(f"training dataset metadata is missing in {manifest_path}")
        expected_dataset = {
            "train_sha256": sha256(train_path),
            "valid_sha256": sha256(valid_path),
            "train_rows": len(train_rows),
            "valid_rows": len(valid_rows),
        }
        for field, expected in expected_dataset.items():
            if dataset.get(field) != expected:
                raise SystemExit(
                    f"training dataset {field} differs in {manifest_path}: "
                    f"expected {expected}, found {dataset.get(field)}"
                )

        payload["training_description"] = provenance["training_description"]
        payload["dataset_manifest"] = dataset_manifest_metadata
        objective = payload.get("objective")
        if not isinstance(objective, dict):
            raise SystemExit(f"training objective is missing in {manifest_path}")
        objective["sequence_target"] = provenance["sequence_target"]

        updated = serialize(payload)
        current = manifest_path.read_text(encoding="utf-8")
        if updated == current:
            continue
        changed.append(manifest_path)
        if write:
            temporary_path = manifest_path.with_suffix(".json.tmp")
            temporary_path.write_text(updated, encoding="utf-8")
            temporary_path.replace(manifest_path)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root containing Research/translation.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that the repair has already been applied without writing.",
    )
    args = parser.parse_args()

    changed = [
        path
        for run in RUNS
        for path in repair_run(args.root.resolve(), run, write=not args.check)
    ]
    if args.check and changed:
        raise SystemExit(
            f"{len(changed)} training manifests still require provenance repair"
        )
    action = "verified" if args.check else "repaired"
    print(f"{action} {len(changed) if not args.check else 18} training manifests")


if __name__ == "__main__":
    main()
