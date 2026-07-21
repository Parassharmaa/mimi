#!/usr/bin/env python3
"""Tests for the truthful automated-claim exposure manifest builder."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "scripts/translation/build_automated_claim_exposure_manifest.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


def run(paths: dict[str, Path]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "python3",
            str(BUILDER),
            str(paths["sources"]),
            str(paths["claim"]),
            str(paths["release"]),
            str(paths["attestations"]),
            str(paths["output"]),
            str(paths["output_dir"]),
            "--protected-jsonl",
            str(paths["protected"]),
            "--results-directory",
            str(paths["results"]),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def fixture(work: Path) -> dict[str, Path]:
    sources = work / "sources.jsonl"
    write_jsonl(
        sources,
        [
            {
                "id": "case-1",
                "documentID": "private-1",
                "source": "A private sentence created in 2026.",
                "references": [],
                "sourceCreatedAt": "2026-07-20",
            }
        ],
    )
    claim = work / "claim.json"
    write_json(
        claim,
        {
            "frozenSources": {"sha256": sha256(sources)},
            "sourcePolicy": {"minimumCreationDate": "2026-07-20"},
        },
    )
    train = work / "train.jsonl"
    valid = work / "valid.jsonl"
    memory_train = work / "memory-train.jsonl"
    protected = work / "protected.jsonl"
    write_jsonl(train, [{"source": "Old training text.", "target": "古い学習文。"}])
    write_jsonl(valid, [{"source": "Old validation text.", "target": "古い検証文。"}])
    write_jsonl(memory_train, [{"source": "Memory source.", "target": "メモリ対象。"}])
    write_jsonl(protected, [{"source": "Router source.", "references": ["ルーター参照。"]}])
    runtime = work / "runtime-memory.json"
    write_json(
        runtime,
        {
            "entries": {
                "en-ja": {"Memory source.": "メモリ対象。"},
                "ja-en": {"メモリ対象。": "Memory source."},
            }
        },
    )
    revision = "1" * 40
    release = work / "release.json"
    write_json(
        release,
        {
            "schemaVersion": 1,
            "datasetFiles": {
                "train": {"path": str(train), "sha256": sha256(train), "split": "train"},
                "valid": {"path": str(valid), "sha256": sha256(valid), "split": "valid"},
            },
            "translationMemory": {
                "trainingData": {
                    "path": str(memory_train),
                    "sha256": sha256(memory_train),
                },
                "runtime": {"path": str(runtime), "sha256": sha256(runtime)},
            },
            "trainingManifests": {},
            "lineageManifests": {},
            "upstreamModels": {
                f"fixture/base@{revision}": {
                    "repository": "fixture/base",
                    "revision": revision,
                    "license": "CC-BY-SA-4.0",
                }
            },
        },
    )
    attestations = work / "attestations.json"
    write_json(
        attestations,
        {
            "models": [
                {
                    "repository": "fixture/base",
                    "revision": revision,
                    "license": "CC-BY-SA-4.0",
                    "revisionAPIURL": f"https://huggingface.co/api/models/fixture/base/revision/{revision}",
                    "modelCardURL": f"https://huggingface.co/fixture/base/blob/{revision}/README.md",
                    "revisionMetadata": {
                        "createdAt": "2024-05-20T01:51:18Z",
                        "id": "fixture/base",
                        "lastModified": "2024-05-20T01:53:38Z",
                        "sha": revision,
                    },
                }
            ]
        },
    )
    results = work / "results"
    write_json(
        results / "selection.json",
        {"results": [{"source": "Selection input.", "hypothesis": "選択出力。"}]},
    )
    return {
        "sources": sources,
        "claim": claim,
        "release": release,
        "attestations": attestations,
        "protected": protected,
        "results": results,
        "output": work / "exposure.json",
        "output_dir": work / "exposure-assets",
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-exposure-") as temporary:
        paths = fixture(Path(temporary))
        result = run(paths)
        assert result.returncode == 0, result.stderr
        exposure = json.loads(paths["output"].read_text(encoding="utf-8"))
        assert exposure["schemaVersion"] == 2
        assert exposure["projectControlledExposureComplete"] is True
        assert exposure["upstreamExactRowsComplete"] is False
        assert exposure["trainingTeacherModels"] == []
        observed = {
            scope
            for asset in exposure["assets"]
            for scope in asset["scopes"]
        } | {value["scope"] for value in exposure["zeroTextScopeAttestations"]}
        assert observed == {
            "training",
            "development",
            "teacher-input",
            "teacher-output",
            "router",
            "model-selection",
            "exact-memory",
        }
        assert any(asset["textCount"] == 4 for asset in exposure["assets"])

        second = run(paths)
        assert second.returncode != 0 and "refusing to overwrite" in second.stderr

        bad = fixture(Path(temporary) / "bad")
        attestations = json.loads(bad["attestations"].read_text(encoding="utf-8"))
        attestations["models"][0]["revisionMetadata"]["createdAt"] = "2026-07-20T00:00:00Z"
        write_json(bad["attestations"], attestations)
        rejected = run(bad)
        assert rejected.returncode != 0 and "not temporally excluded" in rejected.stderr

    print("Mimi automated claim exposure manifest contract passed.")


if __name__ == "__main__":
    main()
