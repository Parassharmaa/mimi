#!/usr/bin/env python3
"""Verify an exact Mimi app archive contains the measured MLX translation runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def directory_files(path: Path) -> dict[str, Path]:
    return {
        item.relative_to(path).as_posix(): item
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("model_bundle", type=Path)
    parser.add_argument("metallib", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--release-artifacts",
        type=Path,
        help="Staged TranslationLicenses directory; required for routed MoE packs.",
    )
    parser.add_argument(
        "--allow-blocked-development",
        action="store_true",
        help=(
            "Verify an explicitly blocked local-development archive without "
            "calling it distributable."
        ),
    )
    parser.add_argument("--maximum-archive-bytes", type=int, default=150_000_000)
    args = parser.parse_args()

    for path in (args.archive, args.model_bundle / "manifest.json", args.metallib):
        if not path.exists():
            raise SystemExit(f"missing distribution input: {path}")
    if args.maximum_archive_bytes < 1:
        raise SystemExit("maximum-archive-bytes must be positive")
    if not zipfile.is_zipfile(args.archive):
        raise SystemExit("distribution archive is not a valid ZIP")
    root_manifest_path = args.model_bundle / "manifest.json"
    root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
    pack_format = root_manifest.get("format")
    if pack_format not in {
        "mimi-mlx-marian-pair-v1",
        "mimi-mlx-marian-moe-v1",
        "mimi-mlx-marian-moe-v2",
    }:
        raise SystemExit("model bundle has an unsupported root manifest")
    if pack_format.startswith("mimi-mlx-marian-moe-") and args.release_artifacts is None:
        raise SystemExit("routed MoE distribution requires staged release artifacts")
    revision_kind = "moe" if pack_format.startswith("mimi-mlx-marian-moe-") else "pair"
    model_revision = f"{revision_kind}-manifest-sha256:{sha256(root_manifest_path)}"
    model_files = directory_files(args.model_bundle)
    if not model_files:
        raise SystemExit("model bundle is empty")
    release_files: dict[str, Path] = {}
    staged_release = None
    if args.release_artifacts is not None:
        staged_path = args.release_artifacts / "staged-release.json"
        if not staged_path.is_file():
            raise SystemExit("release-artifact directory lacks staged-release.json")
        staged_release = json.loads(staged_path.read_text(encoding="utf-8"))
        if (
            staged_release.get("schemaVersion") != 1
            or staged_release.get("modelManifestSha256") != sha256(root_manifest_path)
        ):
            raise SystemExit("staged release artifacts authenticate a different model")
        if (
            staged_release.get("releaseAuthorization") != "authorized"
            and not args.allow_blocked_development
        ):
            raise SystemExit(
                "refusing to verify blocked translation archive as distributable"
            )
        release_files = directory_files(args.release_artifacts)
        if not release_files:
            raise SystemExit("release-artifact directory is empty")

    with zipfile.ZipFile(args.archive) as archive:
        file_names = {
            name
            for name in archive.namelist()
            if name and not name.endswith("/") and "__MACOSX/" not in name
        }
        app_roots = {
            PurePosixPath(name).parts[0]
            for name in file_names
            if PurePosixPath(name).parts
            and PurePosixPath(name).parts[0].endswith(".app")
        }
        if len(app_roots) != 1:
            raise SystemExit("archive must contain exactly one top-level .app")
        app_root = next(iter(app_roots))
        executable_name = f"{app_root}/Contents/MacOS/Mimi"
        info_name = f"{app_root}/Contents/Info.plist"
        metallib_name = f"{app_root}/Contents/MacOS/mlx.metallib"
        model_prefix = f"{app_root}/Contents/Resources/TranslationModels/"
        release_prefix = f"{app_root}/Contents/Resources/TranslationLicenses/"
        for required in (executable_name, info_name, metallib_name):
            if required not in file_names or not archive.read(required):
                raise SystemExit(f"archive lacks required app payload: {required}")
        archived_metallib = archive.read(metallib_name)
        if (
            len(archived_metallib) != args.metallib.stat().st_size
            or sha256_bytes(archived_metallib) != sha256(args.metallib)
        ):
            raise SystemExit("archive metallib does not match the version-pinned input")
        for relative, source in model_files.items():
            archived_name = model_prefix + relative
            if archived_name not in file_names:
                raise SystemExit(f"archive lacks model file: {relative}")
            archived = archive.read(archived_name)
            if len(archived) != source.stat().st_size or sha256_bytes(archived) != sha256(source):
                raise SystemExit(f"archived model file differs from measured bundle: {relative}")
        archived_model_files = {
            name.removeprefix(model_prefix)
            for name in file_names
            if name.startswith(model_prefix)
        }
        if archived_model_files != set(model_files):
            raise SystemExit("archive contains unmeasured files inside TranslationModels")
        for relative, source in release_files.items():
            archived_name = release_prefix + relative
            if archived_name not in file_names:
                raise SystemExit(f"archive lacks translation release artifact: {relative}")
            archived = archive.read(archived_name)
            if len(archived) != source.stat().st_size or sha256_bytes(archived) != sha256(source):
                raise SystemExit(f"archived release artifact differs: {relative}")
        archived_release_files = {
            name.removeprefix(release_prefix)
            for name in file_names
            if name.startswith(release_prefix)
        }
        if archived_release_files != set(release_files):
            raise SystemExit("archive contains unmeasured files inside TranslationLicenses")

    archive_bytes = args.archive.stat().st_size
    size_passes = archive_bytes <= args.maximum_archive_bytes
    release_authorized = (
        staged_release is None
        or (
            staged_release.get("releaseAuthorization") == "authorized"
            and staged_release.get("provenanceComplete") is True
        )
    )
    passed = size_passes and (release_authorized or args.allow_blocked_development)
    if passed and not release_authorized:
        status = "passed-development-only"
    elif passed:
        status = "passed"
    elif not size_passes:
        status = "failed-size"
    else:
        status = "blocked-release"
    report = {
        "schemaVersion": 1,
        "status": status,
        "purpose": "exact combined Mimi app plus MLX translation distribution gate",
        "modelRevision": model_revision,
        "archive": {
            "path": str(args.archive.resolve()),
            "bytes": archive_bytes,
            "sha256": sha256(args.archive),
            "maximumBytes": args.maximum_archive_bytes,
        },
        "modelBundle": {
            "path": str(args.model_bundle.resolve()),
            "bytes": sum(path.stat().st_size for path in model_files.values()),
            "files": len(model_files),
            "rootManifestSHA256": sha256(root_manifest_path),
        },
        "metalRuntime": {
            "path": str(args.metallib.resolve()),
            "bytes": args.metallib.stat().st_size,
            "sha256": sha256(args.metallib),
        },
        "releaseArtifacts": (
            {
                "path": str(args.release_artifacts.resolve()),
                "bytes": sum(path.stat().st_size for path in release_files.values()),
                "files": len(release_files),
                "stagedManifestSHA256": sha256(
                    args.release_artifacts / "staged-release.json"
                ),
                "distributionStatus": staged_release.get("distributionStatus"),
                "doesNotAuthorizeDistribution": staged_release.get(
                    "doesNotAuthorizeDistribution"
                ),
                "doesNotAuthorizeAppIntegration": staged_release.get(
                    "doesNotAuthorizeAppIntegration"
                ),
                "modelPromotionEligible": staged_release.get(
                    "modelPromotionEligible"
                ),
                "releaseAuthorization": staged_release.get(
                    "releaseAuthorization"
                ),
                "provenanceComplete": staged_release.get("provenanceComplete"),
                "releaseBlockers": staged_release.get("releaseBlockers", []),
                "experimentalLocalOnly": staged_release.get(
                    "experimentalLocalOnly"
                ),
            }
            if args.release_artifacts is not None
            else None
        ),
        "archiveLayout": {
            "appRoot": app_root,
            "modelDirectory": model_prefix,
            "releaseDirectory": release_prefix if release_files else None,
            "metallib": metallib_name,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
