#!/usr/bin/env python3
"""Fail closed unless Mimi's embedded ElanMT pack and notices are exact."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


EXPECTED_MANIFEST_SHA256 = "8e55e8f24eed07e89bdad6db0ca1d65aa791905123f764130ed021bc2380807a"
EXPECTED_MODEL_BYTES = 73_403_427
EXPECTED_REVISIONS = {
    "en-ja": "02c48e7031386cd2d41974b0ff1aaf52f010c5fa",
    "ja-en": "539f80eb05306e27a166b45e4264c7fa2eb4de97",
}
EXPECTED_SOURCE_WEIGHTS = {
    "en-ja": "d36a6549863d02a42ad0085ed7eb58d3a81c537e455173bb1e6ce434ecb2eeb8",
    "ja-en": "3cf25766912e952e353fd7632273e6ddb400d4627531bfd749ddd9194e699850",
}


def fail(message: str) -> None:
    raise SystemExit(f"translation release verification failed: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
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
            fail(f"symlink is not allowed in the release payload: {path}")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = path
    return files


def verify_model(model_root: Path) -> None:
    manifest_path = model_root / "manifest.json"
    if not manifest_path.is_file():
        fail(f"missing model manifest: {manifest_path}")
    if sha256(manifest_path) != EXPECTED_MANIFEST_SHA256:
        fail("root model manifest hash changed")

    manifest = load_json(manifest_path)
    if manifest.get("format") != "mimi-mlx-marian-pair-v1":
        fail("unsupported model-pack format")
    if manifest.get("interface") != "bidirectional-en-ja":
        fail("model pack is not bidirectional English-Japanese")
    if manifest.get("license") != "CC-BY-SA-4.0":
        fail("model license is not CC-BY-SA-4.0")
    if manifest.get("source_revisions") != EXPECTED_REVISIONS:
        fail("source model revisions changed")
    if manifest.get("quantization") != {"bits": 4, "group_size": 64}:
        fail("model quantization changed")

    declared = manifest.get("files")
    if not isinstance(declared, dict):
        fail("root manifest has no file inventory")
    actual = measured_files(model_root)
    actual_without_root = {key: value for key, value in actual.items() if key != "manifest.json"}
    if set(actual_without_root) != set(declared):
        fail("model payload contains missing or unmeasured files")

    total_bytes = sum(path.stat().st_size for path in actual.values())
    if total_bytes != EXPECTED_MODEL_BYTES:
        fail(f"model payload is {total_bytes} bytes, expected {EXPECTED_MODEL_BYTES}")

    for relative, record in declared.items():
        if not isinstance(record, dict):
            fail(f"invalid file record for {relative}")
        path = actual_without_root[relative]
        if path.stat().st_size != record.get("bytes") or sha256(path) != record.get("sha256"):
            fail(f"model file failed integrity verification: {relative}")

    for direction in ("en-ja", "ja-en"):
        direction_manifest = load_json(model_root / direction / "manifest.json")
        if direction_manifest.get("direction") != direction:
            fail(f"direction manifest mismatch for {direction}")
        if direction_manifest.get("source_revision") != EXPECTED_REVISIONS[direction]:
            fail(f"source revision mismatch for {direction}")
        if direction_manifest.get("source_weights_sha256") != EXPECTED_SOURCE_WEIGHTS[direction]:
            fail(f"source weight hash mismatch for {direction}")
        if direction_manifest.get("license") != "CC-BY-SA-4.0":
            fail(f"license mismatch for {direction}")
        if (direction_manifest.get("bits"), direction_manifest.get("group_size")) != (4, 64):
            fail(f"quantization mismatch for {direction}")


def verify_licenses(license_root: Path) -> None:
    contract_path = license_root / "release-contract.json"
    if not contract_path.is_file():
        fail(f"missing release contract: {contract_path}")
    contract = load_json(contract_path)
    if contract.get("releaseAuthorization") != "direct-github-release":
        fail("release contract does not authorize direct distribution")
    if contract.get("distributionChannel") != "signed-notarized-direct-download":
        fail("release contract does not cover the GitHub channel")
    if contract.get("macAppStoreAuthorized") is not False:
        fail("Mac App Store must remain explicitly unauthorized")
    if contract.get("qualityStatus") != "initial-preview-not-accuracy-gated":
        fail("initial quality status changed")
    model_record = contract.get("modelPack")
    if not isinstance(model_record, dict):
        fail("release contract has no model-pack record")
    if model_record.get("bytes") != EXPECTED_MODEL_BYTES:
        fail("release contract model byte count changed")
    if model_record.get("manifestSha256") != EXPECTED_MANIFEST_SHA256:
        fail("release contract model manifest hash changed")

    notices = contract.get("requiredNotices")
    if not isinstance(notices, dict) or not notices:
        fail("release contract has no notice inventory")
    actual = measured_files(license_root)
    allowed = set(notices) | {"release-contract.json"}
    if set(actual) != allowed:
        fail("license payload contains missing or unmeasured files")
    for relative, expected_hash in notices.items():
        if sha256(actual[relative]) != expected_hash:
            fail(f"license notice hash changed: {relative}")


def verify_app(app: Path) -> None:
    resources = app / "Contents" / "Resources"
    verify_model(resources / "TranslationModels")
    verify_licenses(resources / "TranslationLicenses")
    executable = app / "Contents" / "MacOS" / "Mimi"
    metallib = app / "Contents" / "MacOS" / "mlx.metallib"
    if not executable.is_file() or not metallib.is_file() or metallib.stat().st_size == 0:
        fail("app is missing the Mimi executable or MLX Metal library")
    architectures = subprocess.run(
        ["lipo", "-archs", str(executable)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    if "arm64" not in architectures:
        fail("app has no Apple Silicon executable slice")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--license-root", type=Path)
    parser.add_argument("--app", type=Path)
    args = parser.parse_args()
    if args.app:
        if args.model_root or args.license_root:
            fail("use --app by itself")
        verify_app(args.app)
    else:
        if not args.model_root or not args.license_root:
            fail("provide --model-root and --license-root")
        verify_model(args.model_root)
        verify_licenses(args.license_root)
    print("Mimi shipped translation pack verification passed")


if __name__ == "__main__":
    main()
