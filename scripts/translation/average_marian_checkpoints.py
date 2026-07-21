#!/usr/bin/env python3
"""Select and average the best adjacent Marian evaluation checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


COPY_FILES = (
    "config.json",
    "generation_config.json",
    "source.spm",
    "target.spm",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "special_tokens_map.json",
)
IDENTITY_FIELDS = ("direction", "student_repository", "student_revision", "license")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(path: Path) -> dict:
    manifest_path = path / "mimi_training_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"checkpoint lacks training manifest: {path}")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value.get("checkpoint_step"), int):
        raise SystemExit(f"checkpoint manifest lacks step: {path}")
    metrics = value.get("checkpoint_metrics", {})
    if not all(isinstance(metrics.get(name), (int, float)) for name in ("chrf_pp", "loss")):
        raise SystemExit(f"checkpoint manifest lacks selection metrics: {path}")
    if not (path / "model.safetensors").is_file():
        raise SystemExit(f"checkpoint lacks model.safetensors: {path}")
    return value


def identity(value: dict) -> dict:
    dataset = value.get("dataset", {})
    return {
        **{name: value.get(name) for name in IDENTITY_FIELDS},
        "train_sha256": dataset.get("train_sha256"),
        "valid_sha256": dataset.get("valid_sha256"),
    }


def slice_chrf(metrics: dict, origin: str | None) -> float:
    if origin is None:
        return float(metrics["chrf_pp"])
    value = metrics.get("slices", {}).get("origin", {}).get(origin)
    if not isinstance(value, dict) or not isinstance(value.get("chrf_pp"), (int, float)):
        raise SystemExit(f"checkpoint metrics lack required origin slice: {origin}")
    return float(value["chrf_pp"])


def selection_chrf(metrics: dict, origins: list[str] | None) -> float:
    if not origins:
        return float(metrics["chrf_pp"])
    return sum(slice_chrf(metrics, origin) for origin in origins) / len(origins)


def choose_window(
    checkpoints: list[tuple[Path, dict]],
    count: int,
    selection_origins: list[str] | None,
    retention_origin: str | None,
    maximum_retention_regression: float,
) -> list[tuple[Path, dict]]:
    if count < 1 or len(checkpoints) < count:
        raise SystemExit(f"need at least {count} checkpoints; found {len(checkpoints)}")
    ordered = sorted(checkpoints, key=lambda item: item[1]["checkpoint_step"])
    steps = [value["checkpoint_step"] for _, value in ordered]
    if len(steps) != len(set(steps)):
        raise SystemExit("checkpoint steps must be unique")
    windows = [ordered[index:index + count] for index in range(len(ordered) - count + 1)]
    if retention_origin is not None:
        retained: list[list[tuple[Path, dict]]] = []
        for window in windows:
            passes = True
            for _, value in window:
                history = value.get("history", [])
                if not history or history[0].get("step") != 0:
                    raise SystemExit("checkpoint manifest lacks step-zero retention baseline")
                baseline = slice_chrf(history[0], retention_origin)
                current = slice_chrf(value["checkpoint_metrics"], retention_origin)
                if current < baseline - maximum_retention_regression:
                    passes = False
                    break
            if passes:
                retained.append(window)
        windows = retained
    if not windows:
        raise SystemExit("no adjacent checkpoint window passes the retention gate")
    return max(
        windows,
        key=lambda window: (
            sum(
                selection_chrf(value["checkpoint_metrics"], selection_origins)
                for _, value in window
            ) / count,
            -sum(value["checkpoint_metrics"]["loss"] for _, value in window) / count,
            -window[-1][1]["checkpoint_step"],
        ),
    )


def average_weights(paths: list[Path]) -> dict[str, torch.Tensor]:
    averaged: dict[str, torch.Tensor] = {}
    dtypes: dict[str, torch.dtype] = {}
    for checkpoint_index, path in enumerate(paths):
        weights = load_file(str(path / "model.safetensors"), device="cpu")
        if checkpoint_index == 0:
            for name, tensor in weights.items():
                dtypes[name] = tensor.dtype
                averaged[name] = tensor.float().clone()
            continue
        if set(weights) != set(averaged):
            raise SystemExit(f"checkpoint tensor names differ: {path}")
        for name, tensor in weights.items():
            if tensor.shape != averaged[name].shape:
                raise SystemExit(f"checkpoint tensor shape differs: {path} / {name}")
            averaged[name].add_(tensor.float())
    divisor = float(len(paths))
    return {
        name: (tensor / divisor).to(dtype=dtypes[name]).contiguous()
        for name, tensor in averaged.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument(
        "--selection-origin",
        action="append",
        help="Select on this reviewed development origin instead of aggregate chrF++.",
    )
    parser.add_argument(
        "--retention-origin",
        help="Reject windows whose origin slice regresses from step zero.",
    )
    parser.add_argument("--maximum-retention-regression", type=float, default=0.5)
    args = parser.parse_args()

    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_directory}")
    checkpoint_paths = sorted(
        path for path in args.checkpoint_directory.iterdir() if path.is_dir()
    )
    checkpoints = [(path, manifest(path)) for path in checkpoint_paths]
    identities = {json.dumps(identity(value), sort_keys=True) for _, value in checkpoints}
    if len(identities) != 1:
        raise SystemExit("checkpoint model or dataset identities differ")
    if args.maximum_retention_regression < 0:
        raise SystemExit("maximum-retention-regression must be non-negative")
    selected = choose_window(
        checkpoints,
        args.count,
        args.selection_origin,
        args.retention_origin,
        args.maximum_retention_regression,
    )
    selected_paths = [path for path, _ in selected]
    weights = average_weights(selected_paths)

    args.output_directory.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(args.output_directory / "model.safetensors"), metadata={"format": "pt"})
    for name in COPY_FILES:
        source = selected_paths[0] / name
        if source.is_file():
            shutil.copy2(source, args.output_directory / name)
    if not (args.output_directory / "config.json").is_file():
        raise SystemExit("selected checkpoint lacks config.json")
    records = [
        {
            "step": value["checkpoint_step"],
            "path": str(path),
            "model_sha256": sha256(path / "model.safetensors"),
            "metrics": value["checkpoint_metrics"],
        }
        for path, value in selected
    ]
    output_manifest = {
        "schema_version": 1,
        "operation": "arithmetic-mean-of-best-adjacent-full-precision-checkpoints",
        "selection": "maximum mean selected reviewed-dev chrF++; tie-break minimum mean loss then earlier window",
        "selection_origin": (
            args.selection_origin[0]
            if args.selection_origin and len(args.selection_origin) == 1
            else None
        ),
        "selection_origins": args.selection_origin or [],
        "retention_origin": args.retention_origin,
        "maximum_retention_regression": args.maximum_retention_regression,
        "count": args.count,
        "identity": identity(selected[0][1]),
        "selected_checkpoints": records,
        "output": {
            "path": str(args.output_directory),
            "model_sha256": sha256(args.output_directory / "model.safetensors"),
        },
    }
    (args.output_directory / "mimi_checkpoint_averaging_manifest.json").write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output_manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
