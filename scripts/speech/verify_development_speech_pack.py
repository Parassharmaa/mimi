#!/usr/bin/env python3
"""Verify Mimi's pinned public Whisper MLX development pack."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


REPOSITORY = "mlx-community/whisper-large-v3-turbo-asr-4bit"
REVISION = "321a6ead9f6e0646bc8188a54d2a470e275c6b76"
EXPECTED_FILES = {
    "README.md": (1_057, "5f8e786ebf20a20d2ca1bd02d6d74939c8e106a2846b4c8ceb3242a2abf6beb0"),
    "added_tokens.json": (34_648, "3c51f66c4c21f9e126970078f11ae77a78c74aee8df606ee9daba86e467108e0"),
    "config.json": (1_506, "9135b2ae07e6450a8f4e87ad1124abe970f705d72ea426030f969cb5014b82e9"),
    "generation_config.json": (3_772, "cce11bfe3aaa6ae9e072ea2637caaec8795e68d9b67e655a5af16ee509681a4c"),
    "merges.txt": (493_869, "2df2990a395e35e8dfbc7511e08c12d56018d8d04691e0133e5d63b21e154dc6"),
    "model.safetensors": (463_462_815, "45298f6dc48df8c11e0a8d1dc5e0197c688bfa530646fa21f1a0238d2b0ecda3"),
    "model.safetensors.index.json": (68_118, "d408891e3b45a13abcb1ccf0a4af6eb50f38331bb71275eec627b182120be015"),
    "normalizer.json": (52_666, "bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd"),
    "preprocessor_config.json": (340, "7ccc62c6f2765af1f3b46c00c9b5894426835a05021c8b9c01eecb6dfb542711"),
    "special_tokens_map.json": (2_186, "baea4ea09372eb4fca86b4e4346139fd73cb807d5087e9de0948e971739c3e74"),
    "tokenizer.json": (2_710_337, "297b13372ac43916285644fb9687add3cc62ee2a1adb60da3dc25cc94c1871fd"),
    "tokenizer_config.json": (282_843, "844b642c73a91359722f47b35705f7174686df33d252695d8572cf9ac03a6389"),
    "vocab.json": (1_036_558, "e2aa043ef015641d363d8288e7c241c85e36a5c761fb303598e0710233344387"),
}
EXPECTED_MODEL_BYTES = 468_150_715


def fail(message: str) -> None:
    raise SystemExit(f"development speech verification failed: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path) -> None:
    actual: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            fail(f"symlink is not allowed: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if "/" in relative:
                fail(f"unexpected nested model file: {relative}")
            actual[relative] = path

    if set(actual) != set(EXPECTED_FILES):
        missing = sorted(set(EXPECTED_FILES) - set(actual))
        extra = sorted(set(actual) - set(EXPECTED_FILES))
        fail(f"file inventory mismatch; missing={missing}, extra={extra}")

    total = 0
    for relative, (expected_bytes, expected_hash) in EXPECTED_FILES.items():
        path = actual[relative]
        measured_bytes = path.stat().st_size
        total += measured_bytes
        if measured_bytes != expected_bytes:
            fail(f"{relative} has {measured_bytes} bytes, expected {expected_bytes}")
        if sha256(path) != expected_hash:
            fail(f"{relative} does not match the evaluated artifact")

    if total != EXPECTED_MODEL_BYTES:
        fail(f"payload is {total} bytes, expected {EXPECTED_MODEL_BYTES}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_root", type=Path)
    args = parser.parse_args()
    verify(args.model_root)
    print(
        "Mimi public development speech pack verification passed "
        f"({EXPECTED_MODEL_BYTES} bytes, {REPOSITORY}@{REVISION})"
    )


if __name__ == "__main__":
    main()
