#!/usr/bin/env python3
"""Contracts for fail-closed Marian release metadata portabilization."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/translation/make_portable_marian_release.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path: Path) -> dict[str, int | str]:
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def absolute_strings(value: object) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            found.extend(absolute_strings(key))
            found.extend(absolute_strings(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(absolute_strings(child))
    elif isinstance(value, str) and Path(value).is_absolute():
        found.append(value)
    return found


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-portable-release-") as temporary:
        repository = Path(temporary)
        source_pack = repository / "models/source-pack"
        source_release = repository / "release/source-audit"
        output_pack = repository / "models/portable-pack"
        output_release = repository / "release/portable-audit"
        engine = source_pack / "engines/formal-en-ja"
        data = repository / "data/train.jsonl"
        engine.mkdir(parents=True)
        source_release.mkdir(parents=True)
        data.parent.mkdir()
        data.write_text('{"source":"hello","target":"こんにちは"}\n', encoding="utf-8")
        (engine / "model.safetensors").write_bytes(b"unchanged-weights")
        (engine / "manifest.json").write_text(
            json.dumps(
                {
                    "direction": "en-ja",
                    "trainingData": {"dataset": {"path": str(data.resolve())}},
                }
            ),
            encoding="utf-8",
        )
        files = {
            item.relative_to(source_pack).as_posix(): record(item)
            for item in sorted(source_pack.rglob("*"))
            if item.is_file()
        }
        (source_pack / "manifest.json").write_text(
            json.dumps(
                {
                    "format": "mimi-mlx-marian-moe-v2",
                    "doesNotAuthorizeAppIntegration": True,
                    "engines": {
                        "formal-en-ja": {
                            "trainingData": {"path": str(data.resolve())}
                        }
                    },
                    "files": files,
                }
            ),
            encoding="utf-8",
        )
        pack_sha = sha256(source_pack / "manifest.json")
        attribution = (
            "# Attributions\n\nExact pack manifest SHA-256: `" + pack_sha + "`.\n"
        )
        (source_release / "ATTRIBUTIONS.md").write_text(attribution, encoding="utf-8")
        (source_release / "tatoeba-attributions.jsonl.gz").write_bytes(b"fixture")
        release_files = {
            name: record(source_release / name)
            for name in ("ATTRIBUTIONS.md", "tatoeba-attributions.jsonl.gz")
        }
        source_contract = {
            "schemaVersion": 1,
            "pack": {
                "path": str(source_pack.resolve()),
                "bytes": sum(
                    item.stat().st_size
                    for item in source_pack.rglob("*")
                    if item.is_file()
                ),
                "manifestSha256": pack_sha,
            },
            "datasetFiles": {
                str(data.resolve()): {
                    "path": str(data.resolve()),
                    "sha256": sha256(data),
                }
            },
            "releaseFiles": release_files,
            "releaseAuthorization": "blocked",
            "doesNotAuthorizeDistribution": True,
            "doesNotAuthorizeAppIntegration": True,
            "modelPromotionEligible": False,
            "blockers": [
                "quality-pending",
                "portable-release-inventory-pending",
            ],
        }
        (source_release / "release-contract.json").write_text(
            json.dumps(source_contract), encoding="utf-8"
        )

        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                str(source_pack),
                str(source_release),
                str(output_pack),
                str(output_release),
                "--repository-root",
                str(repository),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert (output_pack / "engines/formal-en-ja/model.safetensors").read_bytes() == b"unchanged-weights"

        portable_pack = json.loads(
            (output_pack / "manifest.json").read_text(encoding="utf-8")
        )
        portable_engine = json.loads(
            (output_pack / "engines/formal-en-ja/manifest.json").read_text(
                encoding="utf-8"
            )
        )
        portable_contract = json.loads(
            (output_release / "release-contract.json").read_text(encoding="utf-8")
        )
        inventory = json.loads(
            (output_release / "portable-release-inventory.json").read_text(
                encoding="utf-8"
            )
        )
        for value in (portable_pack, portable_engine, portable_contract, inventory):
            assert absolute_strings(value) == []
        assert portable_engine["trainingData"]["dataset"]["path"] == "data/train.jsonl"
        assert portable_contract["pack"]["path"] == "models/portable-pack"
        assert portable_contract["doesNotAuthorizeDistribution"] is True
        assert portable_contract["doesNotAuthorizeAppIntegration"] is True
        assert portable_contract["modelPromotionEligible"] is False
        assert portable_contract["blockers"] == ["quality-pending"]
        assert inventory["releaseAuthorization"] == "blocked"
        assert inventory["repositoryRelativePathsOnly"] is True
        assert inventory["modelPayloadUnchanged"] is True
        for relative, expected in portable_pack["files"].items():
            assert record(output_pack / relative) == expected
        assert (
            portable_contract["pack"]["manifestSha256"]
            == sha256(output_pack / "manifest.json")
        )
        assert pack_sha not in (
            output_release / "ATTRIBUTIONS.md"
        ).read_text(encoding="utf-8")

        staged = repository / "release/staged-portable"
        staged_result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/translation/stage_translation_release_artifacts.py"),
                str(output_pack),
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
        assert "portable-release-inventory.json" in staged_manifest["files"]
        assert staged_manifest["releaseAuthorization"] == "blocked-development-only"
        assert staged_manifest["experimentalLocalOnly"] is True

        outside = repository.parent / "mimi-portable-outside-fixture.jsonl"
        source_contract["datasetFiles"] = {
            str(outside.resolve()): {"path": str(outside.resolve()), "sha256": "0" * 64}
        }
        bad_release = repository / "release/bad-source"
        bad_release.mkdir()
        for name in ("ATTRIBUTIONS.md", "tatoeba-attributions.jsonl.gz"):
            (bad_release / name).write_bytes((source_release / name).read_bytes())
        source_contract["releaseFiles"] = {
            name: record(bad_release / name)
            for name in ("ATTRIBUTIONS.md", "tatoeba-attributions.jsonl.gz")
        }
        (bad_release / "release-contract.json").write_text(
            json.dumps(source_contract), encoding="utf-8"
        )
        rejected = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                str(source_pack),
                str(bad_release),
                str(repository / "models/rejected-pack"),
                str(repository / "release/rejected-audit"),
                "--repository-root",
                str(repository),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert rejected.returncode != 0
        assert "outside repository root" in rejected.stdout + rejected.stderr
    print("Portable Marian release contracts passed.")


if __name__ == "__main__":
    main()
