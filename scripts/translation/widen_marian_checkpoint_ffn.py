#!/usr/bin/env python3
"""Widen Marian encoder/decoder FFNs with a mathematically preserving transform."""

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


def widen_stack(
    state: dict[str, torch.Tensor],
    *,
    stack: str,
    layer_count: int,
    target_dimensions: int,
) -> tuple[int, int]:
    first = state[f"model.{stack}.layers.0.fc1.weight"]
    source_dimensions = int(first.shape[0])
    if target_dimensions <= source_dimensions:
        raise SystemExit(
            f"{stack}-ffn-dim must exceed the source width {source_dimensions}"
        )
    added = target_dimensions - source_dimensions
    copied_indices = torch.arange(added) % source_dimensions
    for layer_index in range(layer_count):
        prefix = f"model.{stack}.layers.{layer_index}."
        fc1_weight = state[f"{prefix}fc1.weight"]
        fc1_bias = state[f"{prefix}fc1.bias"]
        fc2_weight = state[f"{prefix}fc2.weight"]
        if (
            fc1_weight.shape[0] != source_dimensions
            or fc1_bias.shape != (source_dimensions,)
            or fc2_weight.shape[1] != source_dimensions
        ):
            raise SystemExit(f"{stack} FFN shapes differ at layer {layer_index}")
        state[f"{prefix}fc1.weight"] = torch.cat(
            [fc1_weight, fc1_weight[copied_indices].clone()],
            dim=0,
        )
        state[f"{prefix}fc1.bias"] = torch.cat(
            [fc1_bias, fc1_bias[copied_indices].clone()],
            dim=0,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--encoder-ffn-dim", type=int, required=True)
    parser.add_argument("--decoder-ffn-dim", type=int, required=True)
    parser.add_argument(
        "--identity-manifest",
        type=Path,
        required=True,
        help="manifest binding the source repository, revision, and weight hash",
    )
    args = parser.parse_args()

    required = ("config.json", "model.safetensors", *COPY_FILES[:-1])
    missing = [name for name in required if not (args.source / name).is_file()]
    if missing:
        raise SystemExit(f"source checkpoint is missing: {', '.join(missing)}")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")

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
    source_encoder_ffn, output_encoder_ffn = widen_stack(
        state,
        stack="encoder",
        layer_count=len(encoder_layers),
        target_dimensions=args.encoder_ffn_dim,
    )
    source_decoder_ffn, output_decoder_ffn = widen_stack(
        state,
        stack="decoder",
        layer_count=len(decoder_layers),
        target_dimensions=args.decoder_ffn_dim,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    with safe_open(source_weights, framework="pt") as source:
        metadata = source.metadata()
    output_weights = args.output / "model.safetensors"
    save_file(state, output_weights, metadata=metadata)

    configuration = json.loads(
        (args.source / "config.json").read_text(encoding="utf-8")
    )
    if (
        int(configuration.get("encoder_layers", -1)) != len(encoder_layers)
        or int(configuration.get("decoder_layers", -1)) != len(decoder_layers)
        or int(configuration.get("encoder_ffn_dim", -1)) != source_encoder_ffn
        or int(configuration.get("decoder_ffn_dim", -1)) != source_decoder_ffn
    ):
        raise SystemExit("source configuration does not match authenticated weights")
    configuration["encoder_ffn_dim"] = output_encoder_ffn
    configuration["decoder_ffn_dim"] = output_decoder_ffn
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
        "method": "zero-output-column-full-marian-ffn-widening",
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
        "encoder_layers": len(encoder_layers),
        "decoder_layers": len(decoder_layers),
        "source_encoder_ffn_dim": source_encoder_ffn,
        "encoder_ffn_dim": output_encoder_ffn,
        "source_decoder_ffn_dim": source_decoder_ffn,
        "decoder_ffn_dim": output_decoder_ffn,
        "initialization": (
            "copy every existing fc1 row/bias cyclically into added features; "
            "initialize every added fc2 column to exact zero"
        ),
        "expected_initial_function": (
            "mathematically identical FFN function before training; floating-kernel "
            "logit tolerance and exact generated-token parity require separate checks"
        ),
        "promotion_eligible": False,
        "training_authorized": False,
        "private_reasoning_traces_used": False,
        "files": {
            item.name: {"bytes": item.stat().st_size, "sha256": digest(item)}
            for item in sorted(args.output.iterdir())
            if item.is_file()
        },
    }
    manifest_path = args.output / "mimi_ffn_widening_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
