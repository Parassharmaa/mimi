#!/usr/bin/env python3
"""Repair one generated training manifest from its authenticated dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training_manifest_provenance import (
    authenticate_dataset_manifest,
    authenticate_structural_pruning_manifest,
    derive_target_provenance,
    sha256,
)


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise SystemExit(f"{path}:{line_number} is not a JSON object")
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_directory", type=Path)
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("--direction", choices=("en-ja", "ja-en"), required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    manifest_path = args.model_directory / "mimi_training_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing training manifest: {manifest_path}")
    train_path = args.dataset_directory / "train.jsonl"
    valid_path = args.dataset_directory / "valid.jsonl"
    if not train_path.is_file() or not valid_path.is_file():
        raise SystemExit(f"dataset splits are incomplete: {args.dataset_directory}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("direction") != args.direction:
        raise SystemExit("training manifest direction differs")
    dataset_record = payload.get("dataset", {})
    for split, path in (("train", train_path), ("valid", valid_path)):
        declared = dataset_record.get(f"{split}_sha256")
        actual = sha256(path)
        if declared != actual:
            raise SystemExit(
                f"training manifest {split} hash differs: declared {declared}, "
                f"found {actual}"
            )

    dataset_manifest, dataset_metadata = authenticate_dataset_manifest(
        args.dataset_directory,
        direction=args.direction,
        train_path=train_path,
        valid_path=valid_path,
    )
    provenance = derive_target_provenance(
        dataset_manifest,
        load_rows(train_path),
        fallback_training_description=payload.get("training_description", ""),
    )
    repaired = json.loads(json.dumps(payload))
    repaired["training_description"] = provenance["training_description"]
    repaired["dataset_manifest"] = dataset_metadata
    objective = repaired.setdefault("objective", {})
    objective["sequence_target"] = provenance["sequence_target"]
    initial_checkpoint = repaired.get("initial_checkpoint", {})
    initial_path = initial_checkpoint.get("path")
    if initial_path:
        structural_manifest = authenticate_structural_pruning_manifest(
            Path(initial_path)
        )
        initial_checkpoint["structural_pruning_manifest"] = structural_manifest
    if (
        repaired.get("hyperparameters", {}).get("initial_evaluation_skipped")
        and not initial_checkpoint.get("structural_pruning_manifest")
    ):
        raise SystemExit(
            "skipped initial evaluation lacks an authenticated structural-pruning manifest"
        )

    serialized = json.dumps(
        repaired, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    current = manifest_path.read_text(encoding="utf-8")
    if args.check:
        if current != serialized:
            raise SystemExit(f"training manifest needs provenance repair: {manifest_path}")
        print(f"verified {manifest_path}")
        return
    manifest_path.write_text(serialized, encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "sha256": sha256(manifest_path),
                "training_description": provenance["training_description"],
                "sequence_target": provenance["sequence_target"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
