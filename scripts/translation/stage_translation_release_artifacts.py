#!/usr/bin/env python3
"""Authenticate and stage translation release notices beside an app bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


SUPPORTED_PACK_FORMATS = {
    "mimi-mlx-marian-pair-v1",
    "mimi-mlx-marian-moe-v1",
    "mimi-mlx-marian-moe-v2",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def file_record(path: Path) -> dict[str, int | str]:
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_bundle", type=Path)
    parser.add_argument("release_bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--allow-blocked-development",
        action="store_true",
        help=(
            "Stage an explicitly blocked candidate for local development only; "
            "never marks it distributable."
        ),
    )
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite release-artifact output: {args.output}")
    model_manifest_path = args.model_bundle / "manifest.json"
    contract_path = args.release_bundle / "release-contract.json"
    if not model_manifest_path.is_file() or not contract_path.is_file():
        raise SystemExit("model manifest or release contract is missing")
    if model_manifest_path.is_symlink() or contract_path.is_symlink():
        raise SystemExit("release inputs must not be symlinks")

    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    if model_manifest.get("format") not in SUPPORTED_PACK_FORMATS:
        raise SystemExit("translation model pack format is unsupported")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schemaVersion") != 1:
        raise SystemExit("release contract schema is unsupported")
    pack = contract.get("pack") or {}
    if pack.get("manifestSha256") != sha256(model_manifest_path):
        raise SystemExit("release contract authenticates a different model manifest")
    if pack.get("bytes") != directory_bytes(args.model_bundle):
        raise SystemExit("release contract model-byte count differs")

    blockers: list[str] = []
    if contract.get("provenanceComplete") is not True:
        blockers.append("release contract provenance is incomplete")
    contract_blockers = contract.get("blockers", [])
    if contract_blockers is not None and not isinstance(contract_blockers, list):
        raise SystemExit("release contract blockers must be a list")
    for blocker in contract_blockers or []:
        if not isinstance(blocker, str) or not blocker:
            raise SystemExit("release contract contains an invalid blocker")
        blockers.append(f"release contract: {blocker}")
    if model_manifest.get("doesNotAuthorizeAppIntegration") is not False:
        blockers.append("model manifest does not authorize app integration")
    if contract.get("doesNotAuthorizeDistribution") is not False:
        blockers.append("release contract does not authorize distribution")
    if contract.get("modelPromotionEligible") is not True:
        blockers.append("release contract is not promotion eligible")
    if contract.get("distributionStatus") != "authorized-for-distribution":
        blockers.append(
            "distribution status is not explicitly authorized-for-distribution"
        )
    if blockers and not args.allow_blocked_development:
        raise SystemExit(
            "refusing to stage blocked translation release: " + "; ".join(blockers)
        )

    declared_files = contract.get("releaseFiles")
    if not isinstance(declared_files, dict) or not declared_files:
        raise SystemExit("release contract declares no attribution files")
    required = {"ATTRIBUTIONS.md", "tatoeba-attributions.jsonl.gz"}
    if not required.issubset(declared_files):
        raise SystemExit("release contract lacks required attribution artifacts")

    authenticated: dict[str, dict[str, int | str]] = {}
    for relative, expected in sorted(declared_files.items()):
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise SystemExit(f"invalid release artifact path: {relative}")
        source = args.release_bundle / relative_path
        if not source.is_file() or source.is_symlink():
            raise SystemExit(f"release artifact is missing or a symlink: {relative}")
        actual = file_record(source)
        if actual["bytes"] != expected.get("bytes") or actual["sha256"] != expected.get("sha256"):
            raise SystemExit(f"release artifact integrity failure: {relative}")
        authenticated[relative] = actual

    license_bundle = contract.get("licenseBundle")
    if license_bundle is not None:
        if not isinstance(license_bundle, dict):
            raise SystemExit("release license-bundle metadata is invalid")
        manifest_relative = license_bundle.get("manifestPath")
        if not isinstance(manifest_relative, str) or manifest_relative not in declared_files:
            raise SystemExit("release license-bundle manifest is undeclared")
        manifest_path = args.release_bundle / manifest_relative
        if (
            license_bundle.get("manifestSha256") != sha256(manifest_path)
            or license_bundle.get("offlinePayloadComplete") is not True
        ):
            raise SystemExit("release license-bundle metadata does not authenticate its manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        license_files = manifest.get("files")
        if (
            manifest.get("schemaVersion") != 1
            or not isinstance(license_files, dict)
            or len(license_files) != license_bundle.get("files")
            or manifest.get("doesNotAuthorizeDistribution") is not True
        ):
            raise SystemExit("release license-bundle manifest schema is invalid")
        for relative, expected in license_files.items():
            if relative not in declared_files or relative not in authenticated:
                raise SystemExit(f"release license file is undeclared: {relative}")
            if (
                not isinstance(expected, dict)
                or authenticated[relative]["bytes"] != expected.get("bytes")
                or authenticated[relative]["sha256"] != expected.get("sha256")
            ):
                raise SystemExit(f"release license manifest integrity failure: {relative}")

    portable_release = contract.get("portableRelease")
    if portable_release is not None:
        if not isinstance(portable_release, dict) or portable_release.get(
            "repositoryRelativePathsOnly"
        ) is not True:
            raise SystemExit("portable release metadata is invalid")
        relative = "portable-release-inventory.json"
        source = args.release_bundle / relative
        if not source.is_file() or source.is_symlink():
            raise SystemExit("portable release inventory is missing or a symlink")
        inventory = json.loads(source.read_text(encoding="utf-8"))
        inventory_pack = inventory.get("pack") or {}
        inventory_release = inventory.get("release") or {}
        if (
            inventory.get("schemaVersion") != 1
            or inventory.get("repositoryRelativePathsOnly") is not True
            or inventory_pack.get("bytes") != directory_bytes(args.model_bundle)
            or inventory_pack.get("manifestSha256") != sha256(model_manifest_path)
            or inventory_release.get("contractSha256") != sha256(contract_path)
            or inventory.get("doesNotAuthorizeDistribution")
            != contract.get("doesNotAuthorizeDistribution")
            or inventory.get("doesNotAuthorizeAppIntegration")
            != contract.get("doesNotAuthorizeAppIntegration")
            or inventory.get("modelPromotionEligible")
            != contract.get("modelPromotionEligible")
            or inventory.get("blockers") != contract.get("blockers")
        ):
            raise SystemExit("portable release inventory does not match release inputs")
        expected_inventory_files = {
            item.relative_to(args.release_bundle).as_posix(): file_record(item)
            for item in sorted(args.release_bundle.rglob("*"))
            if item.is_file() and item != source
        }
        if inventory_release.get("files") != expected_inventory_files:
            raise SystemExit("portable release inventory file table is incomplete")
        authenticated[relative] = file_record(source)

    args.output.mkdir(parents=True)
    for relative in authenticated:
        destination = args.output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.release_bundle / relative, destination)
    shutil.copyfile(contract_path, args.output / "release-contract.json")
    staged = {
        "schemaVersion": 1,
        "purpose": "hash-bound translation licenses and provenance staged inside Mimi",
        "modelManifestSha256": sha256(model_manifest_path),
        "modelBytes": directory_bytes(args.model_bundle),
        "releaseContractSha256": sha256(contract_path),
        "distributionStatus": contract.get("distributionStatus"),
        "doesNotAuthorizeDistribution": bool(contract.get("doesNotAuthorizeDistribution")),
        "doesNotAuthorizeAppIntegration": bool(
            model_manifest.get("doesNotAuthorizeAppIntegration")
        ),
        "modelPromotionEligible": contract.get("modelPromotionEligible") is True,
        "releaseAuthorization": (
            "authorized" if not blockers else "blocked-development-only"
        ),
        "releaseBlockers": blockers,
        "experimentalLocalOnly": bool(blockers),
        "provenanceComplete": contract.get("provenanceComplete") is True,
        "files": {
            **authenticated,
            "release-contract.json": file_record(args.output / "release-contract.json"),
        },
    }
    staged_path = args.output / "staged-release.json"
    staged_path.write_text(
        json.dumps(staged, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), **staged}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
