#!/usr/bin/env python3
"""Interpolate an adapted Marian checkpoint toward its frozen parent."""

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
    "mimi_training_manifest.json",
)
IDENTICAL_ASSETS = ("source.spm", "target.spm", "vocab.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interpolate_weights(
    parent_path: Path,
    adapted_path: Path,
    adapted_weight: float,
    include_prefixes: tuple[str, ...] | None = None,
) -> dict[str, torch.Tensor]:
    parent = load_file(str(parent_path), device="cpu")
    adapted = load_file(str(adapted_path), device="cpu")
    if set(parent) != set(adapted):
        raise SystemExit("parent and adapted checkpoint tensor names differ")

    output: dict[str, torch.Tensor] = {}
    for name, adapted_tensor in adapted.items():
        parent_tensor = parent[name]
        if parent_tensor.shape != adapted_tensor.shape:
            raise SystemExit(f"checkpoint tensor shape differs: {name}")
        selected = include_prefixes is None or name.startswith(include_prefixes)
        if torch.is_floating_point(adapted_tensor) and selected:
            blended = torch.lerp(
                parent_tensor.float(),
                adapted_tensor.float(),
                adapted_weight,
            )
            output[name] = blended.to(dtype=adapted_tensor.dtype).contiguous()
            continue
        if torch.is_floating_point(adapted_tensor):
            output[name] = parent_tensor.contiguous()
            continue
        if not torch.equal(parent_tensor, adapted_tensor):
            raise SystemExit(f"non-floating checkpoint tensor differs: {name}")
        output[name] = adapted_tensor.contiguous()
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parent", type=Path)
    parser.add_argument("adapted", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--adapted-weight",
        type=float,
        required=True,
        help="Weight assigned to the adapted checkpoint in [0, 1].",
    )
    parser.add_argument(
        "--include-prefix",
        action="append",
        default=[],
        help=(
            "Adapt only tensor names beginning with this prefix. Repeat for a "
            "component merge; omit to interpolate every floating tensor."
        ),
    )
    args = parser.parse_args()

    if not 0.0 <= args.adapted_weight <= 1.0:
        raise SystemExit("adapted-weight must be between zero and one")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")

    parent_weights = args.parent / "model.safetensors"
    adapted_weights = args.adapted / "model.safetensors"
    for path in (parent_weights, adapted_weights):
        if not path.is_file():
            raise SystemExit(f"missing checkpoint weights: {path}")
    for name in IDENTICAL_ASSETS:
        parent_asset = args.parent / name
        adapted_asset = args.adapted / name
        if parent_asset.is_file() != adapted_asset.is_file():
            raise SystemExit(f"parent and adapted checkpoint asset presence differs: {name}")
        if parent_asset.is_file() and sha256(parent_asset) != sha256(adapted_asset):
            raise SystemExit(f"parent and adapted checkpoint assets differ: {name}")

    args.output.mkdir(parents=True, exist_ok=True)
    weights = interpolate_weights(
        parent_weights,
        adapted_weights,
        args.adapted_weight,
        tuple(args.include_prefix) or None,
    )
    output_weights = args.output / "model.safetensors"
    save_file(weights, str(output_weights), metadata={"format": "pt"})

    copied: dict[str, dict[str, str | int]] = {}
    for name in COPY_FILES:
        source = args.adapted / name
        if not source.is_file():
            continue
        destination = args.output / name
        shutil.copy2(source, destination)
        copied[name] = {"bytes": destination.stat().st_size, "sha256": sha256(destination)}
    if not (args.output / "config.json").is_file():
        raise SystemExit("adapted checkpoint lacks config.json")

    manifest = {
        "schema_version": 1,
        "operation": "linear-checkpoint-interpolation",
        "formula": "output = (1 - adapted_weight) * parent + adapted_weight * adapted",
        "adapted_weight": args.adapted_weight,
        "include_prefixes": args.include_prefix or ["*"],
        "parent": {
            "path": str(args.parent),
            "model_sha256": sha256(parent_weights),
        },
        "adapted": {
            "path": str(args.adapted),
            "model_sha256": sha256(adapted_weights),
        },
        "output": {
            "path": str(args.output),
            "model_sha256": sha256(output_weights),
        },
        "copied_assets": copied,
    }
    manifest_path = args.output / "mimi_checkpoint_interpolation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
