#!/usr/bin/env python3
"""Contracts for immutable offline translation-license freezing."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/translation/freeze_translation_license_bundle.py"
SOURCE_IDS = [
    "cc-by-sa-4.0",
    "cc-by-sa-3.0",
    "cc-by-4.0",
    "cc-by-2.5",
    "cc-by-2.0-fr",
    "pdl-1.0-ja",
    "pdl-1.0-en-reference",
    "japanese-law-translation-terms-en",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path: Path) -> dict[str, int | str]:
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-license-bundle-") as temporary:
        repository = Path(temporary)
        model = repository / "model"
        source_release = repository / "release/portable-v2"
        output_release = repository / "release/portable-v3"
        payloads = repository / "license-sources"
        lock_path = repository / "license-sources.json"
        model.mkdir()
        source_release.mkdir(parents=True)
        payloads.mkdir()

        (model / "weights.safetensors").write_bytes(b"weights")
        (model / "manifest.json").write_text(
            json.dumps(
                {
                    "format": "mimi-mlx-marian-moe-v2",
                    "doesNotAuthorizeAppIntegration": True,
                }
            ),
            encoding="utf-8",
        )
        (source_release / "ATTRIBUTIONS.md").write_text("attributions\n", encoding="utf-8")
        (source_release / "tatoeba-attributions.jsonl.gz").write_bytes(b"sidecar")
        release_files = {
            name: record(source_release / name)
            for name in ("ATTRIBUTIONS.md", "tatoeba-attributions.jsonl.gz")
        }
        contract = {
            "schemaVersion": 1,
            "provenanceComplete": True,
            "pack": {
                "path": "model",
                "bytes": sum(path.stat().st_size for path in model.rglob("*") if path.is_file()),
                "manifestSha256": sha256(model / "manifest.json"),
            },
            "portableRelease": {"repositoryRelativePathsOnly": True},
            "releaseFiles": release_files,
            "blockers": ["quality-pending", "public-license-text-bundle-pending"],
            "releaseAuthorization": "blocked",
            "distributionStatus": "blocked-test-fixture",
            "doesNotAuthorizeDistribution": True,
            "doesNotAuthorizeAppIntegration": True,
            "modelPromotionEligible": False,
        }
        contract_path = source_release / "release-contract.json"
        write_json(contract_path, contract)
        inventory = {
            "schemaVersion": 1,
            "repositoryRelativePathsOnly": True,
            "pack": contract["pack"],
            "release": {
                "path": "release/portable-v2",
                "contractSha256": sha256(contract_path),
                "files": {
                    item.relative_to(source_release).as_posix(): record(item)
                    for item in source_release.rglob("*")
                    if item.is_file()
                },
            },
            "blockers": contract["blockers"],
            "doesNotAuthorizeDistribution": True,
            "doesNotAuthorizeAppIntegration": True,
            "modelPromotionEligible": False,
        }
        write_json(source_release / "portable-release-inventory.json", inventory)

        sources = []
        for index, identifier in enumerate(SOURCE_IDS):
            filename = f"license-{index}.txt"
            payload = f"official fixture for {identifier}\n".encode()
            path = payloads / filename
            path.write_bytes(payload)
            sources.append(
                {
                    "id": identifier,
                    "filename": filename,
                    "sourceUrl": f"https://example.invalid/{filename}",
                    "mediaType": "text/plain",
                    "bytes": len(payload),
                    "sha256": sha256(path),
                    "canonical": True,
                    "language": "en",
                    "purpose": "test",
                }
            )
        write_json(lock_path, {"schemaVersion": 1, "sources": sources})

        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                str(source_release),
                str(output_release),
                "--repository-root",
                str(repository),
                "--source-lock",
                str(lock_path),
                "--source-directory",
                str(payloads),
                "--freeze-date",
                "2026-07-21",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr

        frozen_contract = json.loads(
            (output_release / "release-contract.json").read_text(encoding="utf-8")
        )
        frozen_inventory = json.loads(
            (output_release / "portable-release-inventory.json").read_text(encoding="utf-8")
        )
        bundle = json.loads(
            (output_release / "licenses/license-bundle-manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert frozen_contract["blockers"] == ["quality-pending"]
        assert frozen_contract["doesNotAuthorizeDistribution"] is True
        assert frozen_contract["doesNotAuthorizeAppIntegration"] is True
        assert frozen_contract["modelPromotionEligible"] is False
        assert frozen_contract["licenseBundle"]["files"] == 8
        assert len(frozen_contract["releaseFiles"]) == 11
        assert len(bundle["files"]) == 8
        assert frozen_inventory["blockers"] == ["quality-pending"]
        assert frozen_inventory["release"]["contractSha256"] == sha256(
            output_release / "release-contract.json"
        )
        expected_inventory_files = {
            item.relative_to(output_release).as_posix(): record(item)
            for item in output_release.rglob("*")
            if item.is_file() and item.name != "portable-release-inventory.json"
        }
        assert frozen_inventory["release"]["files"] == expected_inventory_files

        staged = repository / "release/staged"
        staged_result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/translation/stage_translation_release_artifacts.py"),
                str(model),
                str(output_release),
                str(staged),
                "--allow-blocked-development",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert staged_result.returncode == 0, staged_result.stdout + staged_result.stderr
        staged_manifest = json.loads(
            (staged / "staged-release.json").read_text(encoding="utf-8")
        )
        assert staged_manifest["releaseAuthorization"] == "blocked-development-only"
        assert len(staged_manifest["files"]) == 13
        assert (staged / "licenses/license-0.txt").is_file()

        duplicate = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                str(source_release),
                str(output_release),
                "--repository-root",
                str(repository),
                "--source-lock",
                str(lock_path),
                "--source-directory",
                str(payloads),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert duplicate.returncode != 0
        assert "refusing to overwrite" in duplicate.stdout + duplicate.stderr

        (payloads / "license-0.txt").write_text("tampered\n", encoding="utf-8")
        rejected_output = repository / "release/rejected"
        rejected = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                str(source_release),
                str(rejected_output),
                "--repository-root",
                str(repository),
                "--source-lock",
                str(lock_path),
                "--source-directory",
                str(payloads),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert rejected.returncode != 0
        assert "pinned" in rejected.stdout + rejected.stderr
        assert "license" in rejected.stdout + rejected.stderr
        assert not rejected_output.exists()
    print("Translation offline-license freezing contracts passed.")


if __name__ == "__main__":
    main()
