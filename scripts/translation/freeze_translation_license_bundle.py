#!/usr/bin/env python3
"""Freeze pinned official license bytes into a successor portable release audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import urllib.request
from pathlib import Path
from typing import Any


LICENSE_BLOCKER = "public-license-text-bundle-pending"
INVENTORY_NAME = "portable-release-inventory.json"
CONTRACT_NAME = "release-contract.json"
MANIFEST_RELATIVE = "licenses/license-bundle-manifest.json"
REQUIRED_SOURCE_IDS = {
    "cc-by-sa-4.0",
    "cc-by-sa-3.0",
    "cc-by-4.0",
    "cc-by-2.5",
    "cc-by-2.0-fr",
    "pdl-1.0-ja",
    "pdl-1.0-en-reference",
    "japanese-law-translation-terms-en",
}
SUPPORTED_TRANSFORMS = {"redact-cakephp-csrf-token-v1"}
CSRF_PATTERN = re.compile(
    rb'(<input type="hidden" name="_csrfToken" autocomplete="off" value=")[^"]+(">)'
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def file_record(path: Path) -> dict[str, int | str]:
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def repository_relative(path: Path, repository_root: Path) -> str:
    try:
        relative = path.resolve().relative_to(repository_root.resolve())
    except ValueError as error:
        raise ValueError(f"path is outside repository root: {path}") from error
    if not relative.parts or ".." in relative.parts:
        raise ValueError(f"path is not safely repository-relative: {path}")
    return relative.as_posix()


def safe_relative(value: str, label: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"unsafe {label}: {value}")
    return relative


def record_matches(path: Path, expected: object) -> bool:
    return isinstance(expected, dict) and file_record(path) == {
        "bytes": expected.get("bytes"),
        "sha256": expected.get("sha256"),
    }


def assert_no_symlinks(root: Path) -> None:
    for item in [root, *root.rglob("*")]:
        if item.is_symlink():
            raise ValueError(f"release input contains a symlink: {item}")


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


def load_sources(lock_path: Path) -> list[dict[str, Any]]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    sources = lock.get("sources")
    if lock.get("schemaVersion") != 1 or not isinstance(sources, list):
        raise ValueError("license source lock schema is unsupported")
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    validated: list[dict[str, Any]] = []
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("license source entry must be an object")
        identifier = source.get("id")
        filename = source.get("filename")
        url = source.get("sourceUrl")
        media_type = source.get("mediaType")
        digest = source.get("sha256")
        size = source.get("bytes")
        raw_size = source.get("rawBytes", size)
        transform = source.get("transform")
        if (
            not isinstance(identifier, str)
            or not identifier
            or not isinstance(filename, str)
            or not filename
            or not isinstance(url, str)
            or not url.startswith("https://")
            or not isinstance(media_type, str)
            or "/" not in media_type
            or not isinstance(digest, str)
            or len(digest) != 64
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(raw_size, int)
            or raw_size <= 0
            or (transform is not None and transform not in SUPPORTED_TRANSFORMS)
        ):
            raise ValueError(f"invalid license source entry: {identifier!r}")
        relative = safe_relative(filename, "license filename")
        if len(relative.parts) != 1:
            raise ValueError(f"license filename must be flat: {filename}")
        if identifier in seen_ids or filename in seen_names:
            raise ValueError(f"duplicate license source: {identifier} / {filename}")
        seen_ids.add(identifier)
        seen_names.add(filename)
        validated.append(source)
    if seen_ids != REQUIRED_SOURCE_IDS:
        raise ValueError(
            "license source lock IDs differ: "
            f"missing={sorted(REQUIRED_SOURCE_IDS - seen_ids)} "
            f"unexpected={sorted(seen_ids - REQUIRED_SOURCE_IDS)}"
        )
    return validated


def fetch_source(
    source: dict[str, Any], source_directory: Path | None
) -> tuple[bytes, str, str]:
    if source_directory is not None:
        path = source_directory / safe_relative(source["filename"], "license filename")
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"local license source is missing or a symlink: {path}")
        return path.read_bytes(), source["mediaType"], source["sourceUrl"]
    request = urllib.request.Request(
        source["sourceUrl"],
        headers={
            "Accept": "*/*",
            "Accept-Encoding": "identity",
            "User-Agent": "Mimi-Translation-License-Freezer/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read(source.get("rawBytes", source["bytes"]) + 1)
        content_type = response.headers.get_content_type()
        final_url = response.geturl()
    if len(payload) > source.get("rawBytes", source["bytes"]):
        raise ValueError(f"license source grew beyond pinned size: {source['id']}")
    if not final_url.startswith("https://"):
        raise ValueError(f"license source redirected outside HTTPS: {source['id']}")
    return payload, content_type, final_url


def transform_source(payload: bytes, source: dict[str, Any]) -> bytes:
    transform = source.get("transform")
    if transform is None:
        return payload
    if transform == "redact-cakephp-csrf-token-v1":
        normalized, count = CSRF_PATTERN.subn(
            rb"\1__REDACTED_VOLATILE_CSRF_TOKEN__\2", payload
        )
        if count != 1:
            raise ValueError(
                f"expected exactly one volatile CSRF token in {source['id']}, found {count}"
            )
        return normalized
    raise ValueError(f"unsupported license transform: {transform}")


def validate_source_release(source_release: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    assert_no_symlinks(source_release)
    contract_path = source_release / CONTRACT_NAME
    inventory_path = source_release / INVENTORY_NAME
    if not contract_path.is_file() or not inventory_path.is_file():
        raise ValueError("source portable contract or inventory is missing")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if contract.get("schemaVersion") != 1:
        raise ValueError("source release contract schema is unsupported")
    portable = contract.get("portableRelease")
    if not isinstance(portable, dict) or portable.get("repositoryRelativePathsOnly") is not True:
        raise ValueError("source release is not a portable release")
    declared = contract.get("releaseFiles")
    if not isinstance(declared, dict) or not declared:
        raise ValueError("source contract declares no release files")
    for relative, expected in declared.items():
        path = source_release / safe_relative(relative, "release artifact path")
        if not path.is_file() or path.is_symlink() or not record_matches(path, expected):
            raise ValueError(f"source release integrity failure: {relative}")
    if (
        inventory.get("schemaVersion") != 1
        or inventory.get("repositoryRelativePathsOnly") is not True
        or (inventory.get("release") or {}).get("contractSha256") != sha256(contract_path)
        or inventory.get("blockers") != contract.get("blockers")
    ):
        raise ValueError("source portable inventory does not authenticate the contract")
    actual = {
        item.relative_to(source_release).as_posix()
        for item in source_release.rglob("*")
        if item.is_file()
    }
    expected_files = set(declared) | {CONTRACT_NAME, INVENTORY_NAME}
    if actual != expected_files:
        raise ValueError("source release contains untracked files")
    return contract, inventory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_release", type=Path)
    parser.add_argument("output_release", type=Path)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument(
        "--source-lock",
        type=Path,
        default=Path(__file__).with_name("translation-license-sources.json"),
    )
    parser.add_argument("--source-directory", type=Path)
    parser.add_argument("--freeze-date", default="2026-07-21")
    args = parser.parse_args()

    repository_root = args.repository_root.resolve()
    for path in (args.source_release, args.output_release, args.source_lock):
        repository_relative(path, repository_root)
    if args.source_directory is not None:
        repository_relative(args.source_directory, repository_root)
    if args.output_release.exists():
        raise SystemExit(f"refusing to overwrite license-frozen output: {args.output_release}")

    try:
        contract, inventory = validate_source_release(args.source_release)
        sources = load_sources(args.source_lock)
        frozen: list[tuple[dict[str, Any], bytes, str, str]] = []
        for source in sources:
            raw_payload, content_type, final_url = fetch_source(
                source, args.source_directory
            )
            if len(raw_payload) != source.get("rawBytes", source["bytes"]):
                raise ValueError(f"pinned raw license size changed: {source['id']}")
            payload = transform_source(raw_payload, source)
            if len(payload) != source["bytes"] or sha256_bytes(payload) != source["sha256"]:
                raise ValueError(f"pinned license bytes changed: {source['id']}")
            if not content_type.startswith(source["mediaType"]):
                raise ValueError(
                    f"license media type changed for {source['id']}: {content_type}"
                )
            frozen.append((source, payload, content_type, final_url))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error

    shutil.copytree(args.source_release, args.output_release)
    license_directory = args.output_release / "licenses"
    license_directory.mkdir()
    license_files: dict[str, dict[str, Any]] = {}
    for source, payload, content_type, final_url in frozen:
        relative = f"licenses/{source['filename']}"
        destination = args.output_release / relative
        destination.write_bytes(payload)
        license_files[relative] = {
            "bytes": len(payload),
            "canonical": source.get("canonical") is True,
            "id": source["id"],
            "language": source.get("language"),
            "mediaType": content_type,
            "purpose": source.get("purpose"),
            "sha256": sha256_bytes(payload),
            "sourceUrl": source["sourceUrl"],
            "resolvedUrl": final_url,
            "transform": source.get("transform"),
        }

    source_contract_path = args.source_release / CONTRACT_NAME
    source_inventory_path = args.source_release / INVENTORY_NAME
    manifest = {
        "schemaVersion": 1,
        "purpose": "offline official license texts for the Mimi EN-JA model candidate",
        "frozenOn": args.freeze_date,
        "files": license_files,
        "sourceLock": {
            "path": repository_relative(args.source_lock, repository_root),
            "sha256": sha256(args.source_lock),
        },
        "sourcePortableContractSha256": sha256(source_contract_path),
        "doesNotAuthorizeDistribution": True,
        "doesNotResolveLicenseCompatibility": True,
    }
    manifest_path = args.output_release / MANIFEST_RELATIVE
    write_json(manifest_path, manifest)

    release_files = dict(contract["releaseFiles"])
    release_files.update(
        {relative: {"bytes": record["bytes"], "sha256": record["sha256"]}
         for relative, record in license_files.items()}
    )
    release_files[MANIFEST_RELATIVE] = file_record(manifest_path)
    contract["releaseFiles"] = dict(sorted(release_files.items()))
    contract["blockers"] = [
        blocker for blocker in contract.get("blockers", []) if blocker != LICENSE_BLOCKER
    ]
    contract["licenseBundle"] = {
        "files": len(license_files),
        "frozenOn": args.freeze_date,
        "manifestPath": MANIFEST_RELATIVE,
        "manifestSha256": sha256(manifest_path),
        "offlinePayloadComplete": True,
        "sourcePortableContractSha256": sha256(source_contract_path),
        "doesNotResolveLicenseCompatibility": True,
    }
    assert_no_absolute_strings(contract, "license-frozen release contract")
    contract_path = args.output_release / CONTRACT_NAME
    write_json(contract_path, contract)

    inventory["release"] = {
        "path": repository_relative(args.output_release, repository_root),
        "contractSha256": sha256(contract_path),
        "files": {
            item.relative_to(args.output_release).as_posix(): file_record(item)
            for item in sorted(args.output_release.rglob("*"))
            if item.is_file() and item.name != INVENTORY_NAME
        },
    }
    inventory["blockers"] = contract["blockers"]
    inventory["licenseBundle"] = contract["licenseBundle"]
    inventory["sourcePortableInventorySha256"] = sha256(source_inventory_path)
    assert_no_absolute_strings(inventory, "license-frozen portable inventory")
    inventory_path = args.output_release / INVENTORY_NAME
    write_json(inventory_path, inventory)

    print(
        json.dumps(
            {
                "release": repository_relative(args.output_release, repository_root),
                "releaseContractSha256": sha256(contract_path),
                "inventorySha256": sha256(inventory_path),
                "licenseBundleManifestSha256": sha256(manifest_path),
                "licenseFiles": len(license_files),
                "blockers": contract["blockers"],
                "releaseAuthorization": contract.get("releaseAuthorization"),
                "doesNotAuthorizeDistribution": contract.get("doesNotAuthorizeDistribution"),
                "doesNotAuthorizeAppIntegration": contract.get("doesNotAuthorizeAppIntegration"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
