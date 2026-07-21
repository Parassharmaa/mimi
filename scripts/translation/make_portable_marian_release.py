#!/usr/bin/env python3
"""Clone a Marian pack/release audit with repository-relative JSON paths.

The source audit remains immutable. This utility rewrites only JSON metadata,
recomputes every affected hash, preserves model/tokenizer/router bytes, and
keeps all release blockers fail-closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


PORTABLE_BLOCKER = "portable-release-inventory-pending"
REQUIRED_RELEASE_FILES = {"ATTRIBUTIONS.md", "tatoeba-attributions.jsonl.gz"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_record(path: Path) -> dict[str, int | str]:
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def repository_relative(path: Path, repository_root: Path) -> str:
    try:
        relative = path.resolve().relative_to(repository_root.resolve())
    except ValueError as error:
        raise ValueError(f"path is outside repository root: {path}") from error
    if not relative.parts or ".." in relative.parts:
        raise ValueError(f"path is not safely repository-relative: {path}")
    return relative.as_posix()


def portable_value(value: Any, repository_root: Path) -> Any:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, child in value.items():
            portable_key = portable_value(key, repository_root)
            if not isinstance(portable_key, str):
                raise ValueError("JSON object key became non-string")
            if portable_key in output:
                raise ValueError(f"portable path key collision: {portable_key}")
            output[portable_key] = portable_value(child, repository_root)
        return output
    if isinstance(value, list):
        return [portable_value(child, repository_root) for child in value]
    if isinstance(value, str) and Path(value).is_absolute():
        return repository_relative(Path(value), repository_root)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_source_pack(pack: Path) -> dict[str, Any]:
    manifest_path = pack / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("source pack manifest is missing or a symlink")
    for item in pack.rglob("*"):
        if item.is_symlink():
            raise ValueError(f"source pack contains a symlink: {item}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("source pack has no authenticated file table")
    actual = {
        item.relative_to(pack).as_posix()
        for item in pack.rglob("*")
        if item.is_file() and item != manifest_path
    }
    if actual != set(files):
        raise ValueError("source pack file table is incomplete")
    for relative, expected in files.items():
        path = pack / relative
        if (
            not isinstance(expected, dict)
            or expected.get("bytes") != path.stat().st_size
            or expected.get("sha256") != sha256(path)
        ):
            raise ValueError(f"source pack integrity failure: {relative}")
    return manifest


def assert_no_absolute_strings(value: Any, label: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            assert_no_absolute_strings(key, label)
            assert_no_absolute_strings(child, label)
    elif isinstance(value, list):
        for child in value:
            assert_no_absolute_strings(child, label)
    elif isinstance(value, str) and Path(value).is_absolute():
        raise ValueError(f"portable output contains an absolute path in {label}: {value}")


def build_portable_pack(
    source_pack: Path, output_pack: Path, repository_root: Path
) -> dict[str, Any]:
    source_manifest = validate_source_pack(source_pack)
    shutil.copytree(source_pack, output_pack)

    for manifest_path in sorted(output_pack.rglob("manifest.json")):
        if manifest_path == output_pack / "manifest.json":
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        portable = portable_value(manifest, repository_root)
        assert_no_absolute_strings(portable, manifest_path.as_posix())
        write_json(manifest_path, portable)

    root_manifest = portable_value(source_manifest, repository_root)
    root_manifest["files"] = {
        item.relative_to(output_pack).as_posix(): file_record(item)
        for item in sorted(output_pack.rglob("*"))
        if item.is_file() and item != output_pack / "manifest.json"
    }
    root_manifest["portableMetadata"] = {
        "repositoryRelativePathsOnly": True,
        "sourceManifestSha256": sha256(source_pack / "manifest.json"),
        "weightPayloadUnchanged": True,
    }
    assert_no_absolute_strings(root_manifest, "portable pack manifest")
    write_json(output_pack / "manifest.json", root_manifest)
    return root_manifest


def copy_release_files(
    source_release: Path,
    output_release: Path,
    source_contract: dict[str, Any],
    source_pack_sha256: str,
    portable_pack_sha256: str,
) -> dict[str, dict[str, int | str]]:
    declared = source_contract.get("releaseFiles")
    if not isinstance(declared, dict) or not REQUIRED_RELEASE_FILES.issubset(declared):
        raise ValueError("source contract lacks required release files")
    records: dict[str, dict[str, int | str]] = {}
    for relative, expected in sorted(declared.items()):
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe source release path: {relative}")
        source = source_release / relative_path
        if (
            not source.is_file()
            or source.is_symlink()
            or not isinstance(expected, dict)
            or expected.get("bytes") != source.stat().st_size
            or expected.get("sha256") != sha256(source)
        ):
            raise ValueError(f"source release integrity failure: {relative}")
        destination = output_release / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative == "ATTRIBUTIONS.md":
            text = source.read_text(encoding="utf-8")
            if text.count(source_pack_sha256) != 1:
                raise ValueError("attribution notice does not uniquely name source pack")
            destination.write_text(
                text.replace(source_pack_sha256, portable_pack_sha256),
                encoding="utf-8",
            )
        else:
            shutil.copyfile(source, destination)
        records[relative] = file_record(destination)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_pack", type=Path)
    parser.add_argument("source_release", type=Path)
    parser.add_argument("output_pack", type=Path)
    parser.add_argument("output_release", type=Path)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()

    repository_root = args.repository_root.resolve()
    for path in (args.source_pack, args.source_release):
        repository_relative(path, repository_root)
    for path in (args.output_pack, args.output_release):
        repository_relative(path, repository_root)
        if path.exists():
            raise SystemExit(f"refusing to overwrite portable output: {path}")

    source_contract_path = args.source_release / "release-contract.json"
    if not source_contract_path.is_file() or source_contract_path.is_symlink():
        raise SystemExit("source release contract is missing or a symlink")
    source_contract = json.loads(source_contract_path.read_text(encoding="utf-8"))
    source_pack_manifest = args.source_pack / "manifest.json"
    pack_record = source_contract.get("pack") or {}
    if (
        pack_record.get("manifestSha256") != sha256(source_pack_manifest)
        or pack_record.get("bytes") != directory_bytes(args.source_pack)
    ):
        raise SystemExit("source release contract authenticates a different pack")

    args.output_release.mkdir(parents=True)
    portable_manifest = build_portable_pack(
        args.source_pack, args.output_pack, repository_root
    )
    portable_pack_sha256 = sha256(args.output_pack / "manifest.json")
    release_files = copy_release_files(
        args.source_release,
        args.output_release,
        source_contract,
        sha256(source_pack_manifest),
        portable_pack_sha256,
    )

    portable_contract = portable_value(source_contract, repository_root)
    portable_contract["pack"] = {
        "bytes": directory_bytes(args.output_pack),
        "manifestSha256": portable_pack_sha256,
        "path": repository_relative(args.output_pack, repository_root),
    }
    portable_contract["releaseFiles"] = release_files
    portable_contract["blockers"] = [
        blocker
        for blocker in portable_contract.get("blockers", [])
        if blocker != PORTABLE_BLOCKER
    ]
    portable_contract["portableRelease"] = {
        "repositoryRelativePathsOnly": True,
        "sourceInternalContractSha256": sha256(source_contract_path),
        "sourceInternalPackManifestSha256": sha256(source_pack_manifest),
        "modelPayloadUnchanged": True,
    }
    assert_no_absolute_strings(portable_contract, "portable release contract")
    portable_contract_path = args.output_release / "release-contract.json"
    write_json(portable_contract_path, portable_contract)

    pack_files = {
        item.relative_to(args.output_pack).as_posix(): file_record(item)
        for item in sorted(args.output_pack.rglob("*"))
        if item.is_file()
    }
    release_payload = {
        item.relative_to(args.output_release).as_posix(): file_record(item)
        for item in sorted(args.output_release.rglob("*"))
        if item.is_file()
    }
    inventory = {
        "schemaVersion": 1,
        "purpose": "portable hash-bound Marian model and release inventory",
        "repositoryRelativePathsOnly": True,
        "pack": {
            "path": repository_relative(args.output_pack, repository_root),
            "bytes": directory_bytes(args.output_pack),
            "manifestSha256": portable_pack_sha256,
            "files": pack_files,
        },
        "release": {
            "path": repository_relative(args.output_release, repository_root),
            "contractSha256": sha256(portable_contract_path),
            "files": release_payload,
        },
        "sourceInternalContractSha256": sha256(source_contract_path),
        "releaseAuthorization": portable_contract.get("releaseAuthorization"),
        "doesNotAuthorizeDistribution": portable_contract.get(
            "doesNotAuthorizeDistribution"
        ),
        "doesNotAuthorizeAppIntegration": portable_contract.get(
            "doesNotAuthorizeAppIntegration"
        ),
        "modelPromotionEligible": portable_contract.get("modelPromotionEligible"),
        "blockers": portable_contract.get("blockers", []),
        "modelPayloadUnchanged": portable_manifest["portableMetadata"][
            "weightPayloadUnchanged"
        ],
    }
    assert_no_absolute_strings(inventory, "portable release inventory")
    inventory_path = args.output_release / "portable-release-inventory.json"
    write_json(inventory_path, inventory)

    for path in [
        *args.output_pack.rglob("manifest.json"),
        portable_contract_path,
        inventory_path,
    ]:
        value = json.loads(path.read_text(encoding="utf-8"))
        assert_no_absolute_strings(value, path.as_posix())

    print(
        json.dumps(
            {
                "pack": inventory["pack"],
                "releaseContractSha256": inventory["release"]["contractSha256"],
                "inventory": repository_relative(inventory_path, repository_root),
                "inventorySha256": sha256(inventory_path),
                "blockers": inventory["blockers"],
                "releaseAuthorization": inventory["releaseAuthorization"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
