#!/usr/bin/env python3
"""Create a reproducible bidirectional Marian initialization from two specialists."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


COPY_FILES = (
    "generation_config.json",
    "source.spm",
    "target.spm",
    "tokenizer_config.json",
    "vocab.json",
    "special_tokens_map.json",
)
TOKENIZER_FILES = (
    "source.spm",
    "target.spm",
    "vocab.json",
    "special_tokens_map.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected a JSON object: {path}")
    return value


def normalized_config(path: Path) -> dict:
    value = read_json(path)
    value.pop("_name_or_path", None)
    return value


def normalized_tokenizer_config(path: Path) -> dict:
    value = read_json(path)
    value.pop("source_lang", None)
    value.pop("target_lang", None)
    return value


def validate_compatible(en_ja: Path, ja_en: Path) -> None:
    required = ("model.safetensors", "config.json", *COPY_FILES)
    for root in (en_ja, ja_en):
        missing = [name for name in required if not (root / name).is_file()]
        if missing:
            raise SystemExit(f"checkpoint is missing {', '.join(missing)}: {root}")
    if normalized_config(en_ja / "config.json") != normalized_config(ja_en / "config.json"):
        raise SystemExit("directional model architectures differ")
    if normalized_tokenizer_config(
        en_ja / "tokenizer_config.json"
    ) != normalized_tokenizer_config(ja_en / "tokenizer_config.json"):
        raise SystemExit("directional tokenizer configurations differ")
    for name in TOKENIZER_FILES:
        if sha256(en_ja / name) != sha256(ja_en / name):
            raise SystemExit(f"directional tokenizer asset differs: {name}")


def merge_weights(
    en_ja: Path,
    ja_en: Path,
    en_ja_weight: float,
) -> dict[str, torch.Tensor]:
    if not 0.0 <= en_ja_weight <= 1.0:
        raise SystemExit("en-ja-weight must be between zero and one")
    left = load_file(str(en_ja / "model.safetensors"), device="cpu")
    right = load_file(str(ja_en / "model.safetensors"), device="cpu")
    if set(left) != set(right):
        raise SystemExit("directional checkpoint tensor names differ")
    merged: dict[str, torch.Tensor] = {}
    for name in sorted(left):
        a = left[name]
        b = right[name]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise SystemExit(f"directional checkpoint tensor differs: {name}")
        if a.dtype.is_floating_point:
            value = a.float().mul(en_ja_weight).add_(b.float(), alpha=1.0 - en_ja_weight)
            merged[name] = value.to(dtype=a.dtype).contiguous()
        else:
            if not torch.equal(a, b):
                raise SystemExit(f"non-floating directional tensor differs: {name}")
            merged[name] = a.contiguous()
    return merged


def source_record(path: Path, expected_direction: str) -> dict:
    training_path = path / "mimi_training_manifest.json"
    training = read_json(training_path) if training_path.is_file() else {}
    direction = training.get("direction")
    if direction is not None and direction != expected_direction:
        raise SystemExit(
            f"expected {expected_direction} checkpoint, found {direction}: {path}"
        )
    return {
        "direction": expected_direction,
        "path": str(path),
        "model_sha256": sha256(path / "model.safetensors"),
        "training_manifest_sha256": sha256(training_path) if training_path.is_file() else None,
        "student_repository": training.get("student_repository"),
        "student_revision": training.get("student_revision"),
        "license": training.get("license"),
        "dataset": training.get("dataset"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("en_ja_checkpoint", type=Path)
    parser.add_argument("ja_en_checkpoint", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--en-ja-weight", type=float, default=0.5)
    args = parser.parse_args()

    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_directory}")
    validate_compatible(args.en_ja_checkpoint, args.ja_en_checkpoint)
    weights = merge_weights(
        args.en_ja_checkpoint,
        args.ja_en_checkpoint,
        args.en_ja_weight,
    )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    save_file(
        weights,
        str(args.output_directory / "model.safetensors"),
        metadata={"format": "pt"},
    )
    configuration = read_json(args.en_ja_checkpoint / "config.json")
    configuration["_name_or_path"] = str(args.output_directory)
    (args.output_directory / "config.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for name in COPY_FILES:
        shutil.copy2(args.en_ja_checkpoint / name, args.output_directory / name)
    tokenizer_configuration = read_json(args.output_directory / "tokenizer_config.json")
    tokenizer_configuration["source_lang"] = "en/ja"
    tokenizer_configuration["target_lang"] = "ja/en"
    (args.output_directory / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_configuration, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 1,
        "operation": "weighted-parameter-mean-bidirectional-initialization",
        "status": "research-initialization-not-shipping-evidence",
        "directions": ["en-ja", "ja-en"],
        "weights": {
            "en-ja": args.en_ja_weight,
            "ja-en": 1.0 - args.en_ja_weight,
        },
        "sources": [
            source_record(args.en_ja_checkpoint, "en-ja"),
            source_record(args.ja_en_checkpoint, "ja-en"),
        ],
        "tokenizer_sha256": {
            name: sha256(args.output_directory / name)
            for name in (*TOKENIZER_FILES, "tokenizer_config.json")
        },
        "output": {
            "path": str(args.output_directory),
            "model_sha256": sha256(args.output_directory / "model.safetensors"),
        },
    }
    manifest_path = args.output_directory / "mimi_bidirectional_initialization_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
