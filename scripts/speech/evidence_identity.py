#!/usr/bin/env python3
"""Portable source identity for Mimi's adaptive speech evidence."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BUILD_IDENTITY_PREFIX = "mimi-adaptive-segmentation-build-identity:"
GENERATED_BUILD_IDENTITY_FILE = (
    "Sources/Mimi/AdaptiveSegmentationBuildIdentity.swift"
)


def adaptive_segmentation_implementation_files(
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[str, ...]:
    """Return all local compiler and scoring inputs for this evidence."""
    relative_paths = {
        "Package.swift",
        "Package.resolved",
        "scripts/speech/evidence_identity.py",
        "scripts/speech/run_mimi_whisper_live_benchmark.py",
    }
    for target in ("Mimi", "MimiCore", "MimiSession"):
        source_root = repository_root / "Sources" / target
        relative_paths.update(
            path.relative_to(repository_root).as_posix()
            for path in source_root.rglob("*.swift")
        )
    relative_paths.remove(GENERATED_BUILD_IDENTITY_FILE)
    return tuple(sorted(relative_paths))


ADAPTIVE_SEGMENTATION_IMPLEMENTATION_FILES = (
    adaptive_segmentation_implementation_files()
)


def adaptive_segmentation_implementation_sha256(
    repository_root: Path = REPOSITORY_ROOT,
) -> str:
    """Hash relevant paths and bytes without checkout-specific metadata."""
    digest = hashlib.sha256()
    for relative_path in adaptive_segmentation_implementation_files(
        repository_root
    ):
        path = repository_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(
                f"adaptive implementation input does not exist: {path}"
            )
        content = path.read_bytes()
        encoded_path = relative_path.encode("utf-8")
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def executable_adaptive_segmentation_identity(
    executable: Path,
) -> str:
    """Read the source identity embedded in a compiled Mimi executable."""
    completed = subprocess.run(
        [
            str(executable.resolve()),
            "--print-adaptive-segmentation-build-identity",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Mimi build identity command failed with "
            f"{completed.returncode}: {completed.stderr[-500:]}"
        )
    matches = re.findall(
        rf"^{re.escape(BUILD_IDENTITY_PREFIX)}([0-9a-f]{{64}})$",
        completed.stdout,
        flags=re.MULTILINE,
    )
    if len(matches) != 1:
        raise RuntimeError(
            "Mimi did not emit exactly one adaptive build identity"
        )
    return matches[0]
