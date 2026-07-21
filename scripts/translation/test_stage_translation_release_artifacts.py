#!/usr/bin/env python3
"""Contracts for staging hash-bound translation notices into Mimi."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/translation/stage_translation_release_artifacts.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-release-stage-") as temporary:
        root = Path(temporary)
        model = root / "model"
        release = root / "release"
        output = root / "staged"
        model.mkdir()
        release.mkdir()
        (model / "model.safetensors").write_bytes(b"weights")
        (model / "manifest.json").write_text(
            json.dumps(
                {
                    "format": "mimi-mlx-marian-moe-v2",
                    "doesNotAuthorizeAppIntegration": True,
                }
            ),
            encoding="utf-8",
        )
        (release / "ATTRIBUTIONS.md").write_text("Attributions\n", encoding="utf-8")
        (release / "tatoeba-attributions.jsonl.gz").write_bytes(b"gzip-fixture")
        release_files = {
            name: {
                "bytes": (release / name).stat().st_size,
                "sha256": sha256(release / name),
            }
            for name in ("ATTRIBUTIONS.md", "tatoeba-attributions.jsonl.gz")
        }
        contract = {
            "schemaVersion": 1,
            "provenanceComplete": True,
            "doesNotAuthorizeDistribution": True,
            "distributionStatus": "blocked-test-fixture",
            "pack": {
                "bytes": sum(path.stat().st_size for path in model.rglob("*") if path.is_file()),
                "manifestSha256": sha256(model / "manifest.json"),
            },
            "releaseFiles": release_files,
        }
        (release / "release-contract.json").write_text(
            json.dumps(contract), encoding="utf-8"
        )
        blocked = subprocess.run(
            [sys.executable, str(SCRIPT), str(model), str(release), str(output)],
            text=True,
            capture_output=True,
            check=False,
        )
        assert blocked.returncode != 0
        assert "refusing to stage blocked translation release" in (
            blocked.stdout + blocked.stderr
        )
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                str(model),
                str(release),
                str(output),
                "--allow-blocked-development",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        staged = json.loads((output / "staged-release.json").read_text(encoding="utf-8"))
        assert staged["modelManifestSha256"] == sha256(model / "manifest.json")
        assert staged["doesNotAuthorizeDistribution"] is True
        assert staged["doesNotAuthorizeAppIntegration"] is True
        assert staged["releaseAuthorization"] == "blocked-development-only"
        assert staged["experimentalLocalOnly"] is True
        assert len(staged["releaseBlockers"]) == 4
        assert set(staged["files"]) == {
            "ATTRIBUTIONS.md",
            "tatoeba-attributions.jsonl.gz",
            "release-contract.json",
        }

        incomplete_output = root / "staged-incomplete"
        contract["provenanceComplete"] = False
        contract["blockers"] = ["missing-conversion-provenance"]
        (release / "release-contract.json").write_text(
            json.dumps(contract), encoding="utf-8"
        )
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                str(model),
                str(release),
                str(incomplete_output),
                "--allow-blocked-development",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        incomplete = json.loads(
            (incomplete_output / "staged-release.json").read_text(encoding="utf-8")
        )
        assert incomplete["provenanceComplete"] is False
        assert incomplete["releaseAuthorization"] == "blocked-development-only"
        assert "release contract provenance is incomplete" in incomplete["releaseBlockers"]
        assert (
            "release contract: missing-conversion-provenance"
            in incomplete["releaseBlockers"]
        )
        contract["provenanceComplete"] = True
        contract.pop("blockers")
        (release / "release-contract.json").write_text(
            json.dumps(contract), encoding="utf-8"
        )

        metallib = root / "mlx.metallib"
        metallib.write_bytes(b"metallib")
        archive = root / "Mimi.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
            zipped.writestr("Mimi.app/Contents/MacOS/Mimi", b"executable")
            zipped.writestr("Mimi.app/Contents/Info.plist", b"plist")
            zipped.write(metallib, "Mimi.app/Contents/MacOS/mlx.metallib")
            for path in sorted(model.rglob("*")):
                if path.is_file():
                    zipped.write(
                        path,
                        "Mimi.app/Contents/Resources/TranslationModels/"
                        + path.relative_to(model).as_posix(),
                    )
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    zipped.write(
                        path,
                        "Mimi.app/Contents/Resources/TranslationLicenses/"
                        + path.relative_to(output).as_posix(),
                    )
        distribution = root / "distribution.json"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/translation/verify_translation_distribution.py"),
                str(archive),
                str(model),
                str(metallib),
                str(distribution),
                "--release-artifacts",
                str(output),
                "--maximum-archive-bytes",
                "1000000",
                "--allow-blocked-development",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        verified = json.loads(distribution.read_text(encoding="utf-8"))
        assert verified["status"] == "passed-development-only"
        assert (
            verified["releaseArtifacts"]["releaseAuthorization"]
            == "blocked-development-only"
        )
        assert verified["releaseArtifacts"]["files"] == 4

        tampered = root / "tampered"
        (release / "ATTRIBUTIONS.md").write_text("changed\n", encoding="utf-8")
        rejected = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                str(model),
                str(release),
                str(tampered),
                "--allow-blocked-development",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert rejected.returncode != 0
        assert "integrity failure" in rejected.stdout + rejected.stderr
    print("Translation release-artifact staging contracts passed.")


if __name__ == "__main__":
    main()
