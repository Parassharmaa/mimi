#!/usr/bin/env python3
"""Repack an authenticated Marian MoE v1 bundle with one shared tokenizer.

The operation is intentionally lossless: model, router, tokenizer-config, and
provenance bytes are copied unchanged.  It succeeds only when every unique
engine authenticates the same tokenizer.json payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath
from typing import Any


SOURCE_FORMAT = "mimi-mlx-marian-moe-v1"
OUTPUT_FORMAT = "mimi-mlx-marian-moe-v2"
SHARED_TOKENIZER_PATH = "shared/tokenizer.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def is_safe_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and all(
        part not in ("", ".", "..") for part in path.parts
    )


def validate_record(path: Path, record: Any, label: str) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"missing authenticated record for {label}")
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing regular file for {label}: {path}")
    actual = file_record(path)
    if actual != {"bytes": record.get("bytes"), "sha256": record.get("sha256")}:
        raise ValueError(f"integrity failure for {label}: {path}")


def validate_file_table(root: Path, files: Any) -> None:
    if not isinstance(files, dict) or not files:
        raise ValueError("root manifest has no authenticated file table")
    for relative, record in files.items():
        if not isinstance(relative, str) or not is_safe_relative_path(relative):
            raise ValueError(f"unsafe root file path: {relative!r}")
        validate_record(root / relative, record, relative)


def engine_paths(manifest: dict[str, Any]) -> list[str]:
    generalists = manifest.get("generalists")
    experts = manifest.get("experts")
    if not isinstance(generalists, dict) or not isinstance(experts, dict):
        raise ValueError("source manifest is missing Marian MoE engines")
    if set(generalists) != {"en-ja", "ja-en"} or set(experts) != {"en-ja", "ja-en"}:
        raise ValueError("source manifest must contain both generalists and both experts")
    paths: list[str] = []
    for direction in ("en-ja", "ja-en"):
        generalist = generalists.get(direction)
        expert = experts.get(direction)
        expert_path = expert.get("engine") if isinstance(expert, dict) else None
        for value in (generalist, expert_path):
            if not isinstance(value, str) or not is_safe_relative_path(value):
                raise ValueError(f"unsafe {direction} engine path: {value!r}")
            if value not in paths:
                paths.append(value)
    return paths


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def repack(source: Path, output: Path) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    if source == output:
        raise ValueError("source and output directories must differ")
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    root_manifest_path = source / "manifest.json"
    manifest = read_json(root_manifest_path)
    if manifest.get("format") != SOURCE_FORMAT:
        raise ValueError(f"source manifest format must be {SOURCE_FORMAT}")
    validate_file_table(source, manifest.get("files"))
    declared_source_files = set(manifest["files"])
    actual_source_files = {
        item.relative_to(source).as_posix()
        for item in source.rglob("*")
        if item.is_file() and item != root_manifest_path
    }
    if actual_source_files != declared_source_files:
        extra = sorted(actual_source_files - declared_source_files)
        missing = sorted(declared_source_files - actual_source_files)
        raise ValueError(
            f"source file table is not exhaustive; extra={extra}, missing={missing}"
        )
    source_symlinks = [
        item.relative_to(source).as_posix()
        for item in source.rglob("*")
        if item.is_symlink()
    ]
    if source_symlinks:
        raise ValueError(f"source bundle contains symlinks: {source_symlinks}")

    engines = engine_paths(manifest)
    tokenizer_records: list[dict[str, Any]] = []
    tokenizer_paths: list[Path] = []
    engine_manifests: dict[str, dict[str, Any]] = {}
    for relative in engines:
        engine_root = source / relative
        engine_manifest_path = engine_root / "manifest.json"
        engine_manifest = read_json(engine_manifest_path)
        files = engine_manifest.get("files")
        if not isinstance(files, dict):
            raise ValueError(f"engine has no authenticated file table: {relative}")
        tokenizer_path = engine_root / "tokenizer.json"
        validate_record(tokenizer_path, files.get("tokenizer.json"), f"{relative}/tokenizer.json")
        tokenizer_records.append(file_record(tokenizer_path))
        tokenizer_paths.append(tokenizer_path)
        engine_manifests[relative] = engine_manifest

    first_record = tokenizer_records[0]
    if any(record != first_record for record in tokenizer_records[1:]):
        raise ValueError("engine tokenizer payloads are not byte-identical")

    skipped = {Path("manifest.json")}
    skipped.update(Path(relative) / "tokenizer.json" for relative in engines)
    output.mkdir(parents=True)
    for item in sorted(source.rglob("*")):
        relative = item.relative_to(source)
        if relative in skipped:
            continue
        destination = output / relative
        if item.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, destination)

    shared_tokenizer = output / SHARED_TOKENIZER_PATH
    shared_tokenizer.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tokenizer_paths[0], shared_tokenizer)

    for relative, engine_manifest in engine_manifests.items():
        copied_manifest = dict(engine_manifest)
        copied_files = dict(copied_manifest["files"])
        copied_files.pop("tokenizer.json")
        copied_manifest["files"] = copied_files
        copied_manifest["shared_tokenizer"] = {
            "bytes": first_record["bytes"],
            "sha256": first_record["sha256"],
        }
        (output / relative / "manifest.json").write_text(
            json.dumps(copied_manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    output_manifest = dict(manifest)
    output_manifest["format"] = OUTPUT_FORMAT
    output_manifest["sharedTokenizer"] = SHARED_TOKENIZER_PATH
    output_manifest["tokenizerLayout"] = {
        "bytes": first_record["bytes"],
        "engineCount": len(engines),
        "kind": "authenticated-root-shared-tokenizer",
        "sha256": first_record["sha256"],
    }
    output_files: dict[str, dict[str, Any]] = {}
    for item in sorted(output.rglob("*")):
        if item.is_file():
            relative = item.relative_to(output).as_posix()
            output_files[relative] = file_record(item)
    output_manifest["files"] = output_files
    (output / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validate_file_table(output, output_manifest["files"])

    source_bytes = directory_bytes(source)
    output_bytes = directory_bytes(output)
    return {
        "engineCount": len(engines),
        "format": OUTPUT_FORMAT,
        "output": str(output),
        "outputBytes": output_bytes,
        "savedBytes": source_bytes - output_bytes,
        "sharedTokenizer": SHARED_TOKENIZER_PATH,
        "source": str(source),
        "sourceBytes": source_bytes,
        "tokenizer": first_record,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        result = repack(args.source, args.output)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
