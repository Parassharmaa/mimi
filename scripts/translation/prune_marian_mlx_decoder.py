#!/usr/bin/env python3
"""Create a research-only shallow-decoder variant of a quantized Marian pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import mlx.core as mx


COPY_FILES = ("tokenizer.json", "tokenizer_config.json")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def layer_indices(weight_names: list[str], stack: str) -> list[int]:
    prefix = f"{stack}.layers."
    indices = sorted(
        {
            int(name.removeprefix(prefix).split(".", 1)[0])
            for name in weight_names
            if name.startswith(prefix)
        }
    )
    if not indices or indices != list(range(indices[-1] + 1)):
        raise SystemExit(f"{stack} layer indices are missing or non-contiguous")
    return indices


def parse_selection(raw: str, available: list[int]) -> list[int]:
    try:
        selected = [int(value) for value in raw.split(",")]
    except ValueError as error:
        raise SystemExit("--keep-decoder-layers must be comma-separated integers") from error
    if not selected or len(selected) != len(set(selected)):
        raise SystemExit("--keep-decoder-layers must contain unique layer indices")
    if selected != sorted(selected):
        raise SystemExit("--keep-decoder-layers must be in ascending source order")
    if any(index not in available for index in selected):
        raise SystemExit(
            f"decoder layer selection {selected} is outside available layers {available}"
        )
    return selected


def remap_decoder_name(name: str, mapping: dict[int, int]) -> str | None:
    prefix = "decoder.layers."
    if not name.startswith(prefix):
        return name
    suffix = name.removeprefix(prefix)
    raw_index, remainder = suffix.split(".", 1)
    old_index = int(raw_index)
    if old_index not in mapping:
        return None
    return f"{prefix}{mapping[old_index]}.{remainder}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--keep-decoder-layers", required=True)
    args = parser.parse_args()

    required = ("manifest.json", "model.safetensors", *COPY_FILES)
    missing = [name for name in required if not (args.source / name).is_file()]
    if missing:
        raise SystemExit(f"source direction is missing: {', '.join(missing)}")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")

    source_manifest_path = args.source / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_files = source_manifest.get("files", {})
    for name in ("model.safetensors", *COPY_FILES):
        record = source_files.get(name)
        if not isinstance(record, dict) or record.get("sha256") != digest(
            args.source / name
        ):
            raise SystemExit(
                f"source manifest does not authenticate required file: {name}"
            )
    weights = mx.load(str(args.source / "model.safetensors"))
    names = list(weights)
    encoder_layers = layer_indices(names, "encoder")
    decoder_layers = layer_indices(names, "decoder")
    selected = parse_selection(args.keep_decoder_layers, decoder_layers)
    if len(selected) >= len(decoder_layers):
        raise SystemExit("decoder pruning must remove at least one layer")

    mapping = {old: new for new, old in enumerate(selected)}
    pruned = {}
    for name, value in weights.items():
        remapped = remap_decoder_name(name, mapping)
        if remapped is not None:
            pruned[remapped] = value

    args.output.mkdir(parents=True, exist_ok=True)
    output_weights = args.output / "model.safetensors"
    mx.save_safetensors(str(output_weights), pruned)
    for name in COPY_FILES:
        shutil.copy2(args.source / name, args.output / name)

    manifest = {
        **source_manifest,
        "architecture": "Marian encoder-heavy shallow-decoder research ablation",
        "encoder_layers": len(encoder_layers),
        "decoder_layers": len(selected),
        "source_revision": (
            f"decoder-pruning-source-manifest-sha256:{digest(source_manifest_path)}"
        ),
        "structural_pruning": {
            "method": "depth-pruning-without-recovery-training",
            "source_encoder_layers": encoder_layers,
            "source_decoder_layers": decoder_layers,
            "kept_decoder_layers": selected,
            "promotion_eligible": False,
        },
        "distribution_status": source_manifest.get(
            "distribution_status",
            "provenance-incomplete-not-approved-for-distribution",
        ),
    }
    manifest["files"] = {
        item.name: {"bytes": item.stat().st_size, "sha256": digest(item)}
        for item in sorted(args.output.iterdir())
        if item.is_file() and item.name != "manifest.json"
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "encoder_layers": len(encoder_layers),
                "decoder_layers": len(selected),
                "kept_decoder_layers": selected,
                "model_bytes": output_weights.stat().st_size,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
