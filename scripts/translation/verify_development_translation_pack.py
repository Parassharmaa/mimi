#!/usr/bin/env python3
"""Verify Mimi's pinned public EN-JA development model without release promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_MANIFEST_SHA256 = (
    "824214ae6434ef0abdf06e7b208077ff3014d372144ab905b64127bbe7c1b8f7"
)
# The public release adds 1,075 bytes of provenance to the benchmarked root
# manifest. Runtime payloads remain byte-identical to the evaluated pair.
EXPECTED_MODEL_BYTES = 73_426_883
EXPECTED_REVISIONS = {
    "en-ja": "02c48e7031386cd2d41974b0ff1aaf52f010c5fa",
    "ja-en": "539f80eb05306e27a166b45e4264c7fa2eb4de97",
}


def fail(message: str) -> None:
    raise SystemExit(f"development translation verification failed: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        fail(f"cannot read {path}: {error}")
    if not isinstance(value, dict):
        fail(f"{path} is not a JSON object")
    return value


def measured_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            fail(f"symlink is not allowed in the model payload: {path}")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = path
    return files


def verify_model(root: Path) -> None:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        fail(f"missing model manifest: {manifest_path}")
    if sha256(manifest_path) != EXPECTED_MANIFEST_SHA256:
        fail("root manifest does not match the pinned public development revision")

    manifest = load_object(manifest_path)
    expected_header = {
        "format": "mimi-mlx-marian-pair-v1",
        "interface": "bidirectional-en-ja",
        "engines": ["en-ja", "ja-en"],
        "quantization": {"bits": 4, "group_size": 64},
        "license": "CC-BY-SA-4.0",
        "source_revisions": EXPECTED_REVISIONS,
        "distribution_status": "public-open-cc-by-sa-4.0",
    }
    for key, expected in expected_header.items():
        if manifest.get(key) != expected:
            fail(f"unexpected {key} in root manifest")

    declared = manifest.get("files")
    if not isinstance(declared, dict) or not declared:
        fail("root manifest has no file inventory")
    actual = measured_files(root)
    actual_without_root = {
        relative: path for relative, path in actual.items() if relative != "manifest.json"
    }
    if set(actual_without_root) != set(declared):
        fail("model payload contains missing or unmeasured files")

    total_bytes = sum(path.stat().st_size for path in actual.values())
    if total_bytes != EXPECTED_MODEL_BYTES:
        fail(f"model payload is {total_bytes} bytes, expected {EXPECTED_MODEL_BYTES}")

    for relative, record in declared.items():
        if not isinstance(record, dict):
            fail(f"invalid file record for {relative}")
        path = actual_without_root[relative]
        if (
            path.stat().st_size != record.get("bytes")
            or sha256(path) != record.get("sha256")
        ):
            fail(f"file failed integrity verification: {relative}")

    for direction in EXPECTED_REVISIONS:
        direction_manifest = load_object(root / direction / "manifest.json")
        if direction_manifest.get("direction") != direction:
            fail(f"direction mismatch for {direction}")
        if direction_manifest.get("source_revision") != EXPECTED_REVISIONS[direction]:
            fail(f"source revision mismatch for {direction}")
        if direction_manifest.get("license") != "CC-BY-SA-4.0":
            fail(f"license mismatch for {direction}")
        training_data = direction_manifest.get("training_data")
        if (
            not isinstance(training_data, dict)
            or training_data.get("distribution_status")
            != "public-open-cc-by-sa-4.0"
        ):
            fail(f"public distribution status mismatch for {direction}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_root", type=Path)
    args = parser.parse_args()
    verify_model(args.model_root)
    print(
        "Mimi public development translation pack verification passed "
        f"({EXPECTED_MODEL_BYTES} bytes)"
    )


if __name__ == "__main__":
    main()
