#!/usr/bin/env python3
"""Depth-prune a full-precision Marian checkpoint for recovery distillation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


COPY_FILES = (
    "generation_config.json",
    "source.spm",
    "target.spm",
    "tokenizer_config.json",
    "vocab.json",
    "special_tokens_map.json",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def layer_indices(weight_names: list[str], stack: str) -> list[int]:
    prefix = f"model.{stack}.layers."
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
    prefix = "model.decoder.layers."
    if not name.startswith(prefix):
        return name
    suffix = name.removeprefix(prefix)
    raw_index, remainder = suffix.split(".", 1)
    old_index = int(raw_index)
    if old_index not in mapping:
        return None
    return f"{prefix}{mapping[old_index]}.{remainder}"


def append_identity_encoder_layers(
    state: dict[str, object],
    *,
    source_layer_count: int,
    additional_layer_count: int,
) -> None:
    """Append trainable post-norm layers initialized as near-identity blocks."""

    if additional_layer_count == 0:
        return
    source_prefix = f"model.encoder.layers.{source_layer_count - 1}."
    template = {
        name.removeprefix(source_prefix): value
        for name, value in state.items()
        if name.startswith(source_prefix)
    }
    if not template:
        raise SystemExit("could not locate the final encoder layer template")
    for layer_index in range(
        source_layer_count,
        source_layer_count + additional_layer_count,
    ):
        prefix = f"model.encoder.layers.{layer_index}."
        for suffix, value in template.items():
            if suffix.endswith("layer_norm.weight"):
                initialized = value.new_ones(value.shape)
            else:
                initialized = value.new_zeros(value.shape)
            state[f"{prefix}{suffix}"] = initialized


def widen_encoder_feed_forward(
    state: dict[str, torch.Tensor],
    *,
    encoder_layer_count: int,
    target_dimensions: int | None,
) -> tuple[int, int]:
    """Add active encoder features behind zero output columns for exact startup."""

    first = state["model.encoder.layers.0.fc1.weight"]
    source_dimensions = int(first.shape[0])
    if target_dimensions is None:
        return source_dimensions, source_dimensions
    if target_dimensions <= source_dimensions:
        raise SystemExit(
            f"encoder-ffn-dim must exceed the source width {source_dimensions}"
        )
    added = target_dimensions - source_dimensions
    copied_indices = torch.arange(added) % source_dimensions
    for layer_index in range(encoder_layer_count):
        prefix = f"model.encoder.layers.{layer_index}."
        fc1_weight = state[f"{prefix}fc1.weight"]
        fc1_bias = state[f"{prefix}fc1.bias"]
        fc2_weight = state[f"{prefix}fc2.weight"]
        state[f"{prefix}fc1.weight"] = torch.cat(
            [fc1_weight, fc1_weight[copied_indices].clone()], dim=0
        )
        state[f"{prefix}fc1.bias"] = torch.cat(
            [fc1_bias, fc1_bias[copied_indices].clone()], dim=0
        )
        state[f"{prefix}fc2.weight"] = torch.cat(
            [
                fc2_weight,
                fc2_weight.new_zeros((fc2_weight.shape[0], added)),
            ],
            dim=1,
        )
    return source_dimensions, target_dimensions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--keep-decoder-layers", required=True)
    parser.add_argument(
        "--append-identity-encoder-layers",
        type=int,
        default=0,
        help=(
            "Append this many trainable zero-residual post-norm encoder layers "
            "after decoder pruning."
        ),
    )
    parser.add_argument(
        "--encoder-ffn-dim",
        type=int,
        help=(
            "Widen every encoder FFN to this dimension by copying active fc1 "
            "features behind zero-initialized fc2 output columns."
        ),
    )
    parser.add_argument(
        "--identity-manifest",
        type=Path,
        required=True,
        help="authenticated converted-direction manifest binding repository/revision to weights",
    )
    args = parser.parse_args()

    required = ("config.json", "model.safetensors", *COPY_FILES[:-1])
    missing = [name for name in required if not (args.source / name).is_file()]
    if missing:
        raise SystemExit(f"source checkpoint is missing: {', '.join(missing)}")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    if args.append_identity_encoder_layers < 0:
        raise SystemExit("append-identity-encoder-layers must be non-negative")

    source_weights = args.source / "model.safetensors"
    identity_manifest = json.loads(
        args.identity_manifest.read_text(encoding="utf-8")
    )
    if identity_manifest.get("source_weights_sha256") != digest(source_weights):
        raise SystemExit("identity manifest does not authenticate source weights")
    if not identity_manifest.get("source_repository") or not identity_manifest.get(
        "source_revision"
    ):
        raise SystemExit("identity manifest is missing repository or revision")
    state = load_file(source_weights)
    names = list(state)
    encoder_layers = layer_indices(names, "encoder")
    decoder_layers = layer_indices(names, "decoder")
    selected = parse_selection(args.keep_decoder_layers, decoder_layers)
    if len(selected) >= len(decoder_layers):
        raise SystemExit("decoder pruning must remove at least one layer")

    mapping = {old: new for new, old in enumerate(selected)}
    pruned = {}
    for name, value in state.items():
        remapped = remap_decoder_name(name, mapping)
        if remapped is not None:
            pruned[remapped] = value
    append_identity_encoder_layers(
        pruned,
        source_layer_count=len(encoder_layers),
        additional_layer_count=args.append_identity_encoder_layers,
    )
    output_encoder_layers = len(encoder_layers) + args.append_identity_encoder_layers
    source_encoder_ffn_dim, output_encoder_ffn_dim = widen_encoder_feed_forward(
        pruned,
        encoder_layer_count=output_encoder_layers,
        target_dimensions=args.encoder_ffn_dim,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    with safe_open(source_weights, framework="pt") as source:
        metadata = source.metadata()
    save_file(pruned, args.output / "model.safetensors", metadata=metadata)

    configuration = json.loads((args.source / "config.json").read_text(encoding="utf-8"))
    configuration["encoder_layers"] = output_encoder_layers
    configuration["encoder_ffn_dim"] = output_encoder_ffn_dim
    configuration["decoder_layers"] = len(selected)
    (args.output / "config.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for name in COPY_FILES:
        source = args.source / name
        if source.is_file():
            shutil.copy2(source, args.output / name)

    manifest = {
        "schema_version": 1,
        "method": (
            "wide-encoder-shallow-decoder-reallocation-before-distillation"
            if args.encoder_ffn_dim is not None
            else "deep-encoder-shallow-decoder-reallocation-before-distillation"
            if args.append_identity_encoder_layers
            else "decoder-depth-pruning-before-logit-recovery-distillation"
        ),
        "source": {
            "path": str(args.source),
            "repository": identity_manifest["source_repository"],
            "revision": identity_manifest["source_revision"],
            "weights_sha256": digest(source_weights),
            "identity_manifest": {
                "path": str(args.identity_manifest),
                "sha256": digest(args.identity_manifest),
            },
        },
        "source_encoder_layers": encoder_layers,
        "appended_identity_encoder_layers": args.append_identity_encoder_layers,
        "encoder_initialization": (
            "zero residual projections with unit layer norms"
            if args.append_identity_encoder_layers
            else None
        ),
        "encoder_layers": output_encoder_layers,
        "source_encoder_ffn_dim": source_encoder_ffn_dim,
        "encoder_ffn_dim": output_encoder_ffn_dim,
        "encoder_ffn_initialization": (
            "copied fc1 features with zero new fc2 output columns"
            if args.encoder_ffn_dim is not None
            else None
        ),
        "source_decoder_layers": decoder_layers,
        "kept_decoder_layers": selected,
        "decoder_layers": len(selected),
        "promotion_eligible": False,
        "private_reasoning_traces_used": False,
        "files": {
            item.name: {"bytes": item.stat().st_size, "sha256": digest(item)}
            for item in sorted(args.output.iterdir())
            if item.is_file()
        },
    }
    (args.output / "mimi_structural_pruning_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
