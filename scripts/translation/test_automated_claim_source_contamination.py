#!/usr/bin/env python3
"""Contract tests for current-release source contamination auditing."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/translation/audit_automated_claim_source_contamination.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


def run(sources: Path, contract: Path, protected: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "python3",
            str(SCRIPT),
            str(sources),
            str(contract),
            str(output),
            "--protected-jsonl",
            str(protected),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-source-contamination-") as temporary:
        work = Path(temporary)
        sources = work / "sources.jsonl"
        write_jsonl(
            sources,
            [
                {
                    "id": "case-1",
                    "documentID": "fresh-document",
                    "source": "A completely new product sentence.",
                    "references": [],
                }
            ],
        )
        train = work / "train.jsonl"
        memory = work / "memory-train.jsonl"
        protected = work / "protected.jsonl"
        write_jsonl(train, [{"source": "Unrelated training sentence.", "target": "無関係です。"}])
        write_jsonl(memory, [{"source": "Another unrelated sentence.", "target": "別の文です。"}])
        write_jsonl(protected, [{"source": "Protected source.", "references": ["Protected reference."]}])
        contract = work / "release-contract.json"
        write_json(
            contract,
            {
                "schemaVersion": 1,
                "datasetFiles": {
                    str(train): {
                        "path": str(train),
                        "sha256": sha256(train),
                        "rows": 1,
                        "split": "train",
                    }
                },
                "translationMemory": {
                    "trainingData": {
                        "path": str(memory),
                        "sha256": sha256(memory),
                        "rows": 1,
                    }
                },
            },
        )
        output = work / "audit.json"
        result = run(sources, contract, protected, output)
        assert result.returncode == 0, result.stderr
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["status"] == "release-lineage-source-contamination-scan-passed"
        assert report["textsScanned"] == 6

        output.unlink()
        write_jsonl(
            protected,
            [{"source": "Protected source.", "references": ["A completely new product sentence."]}],
        )
        result = run(sources, contract, protected, output)
        assert result.returncode != 0 and "exact-match contamination" in result.stderr

    print("Mimi release-lineage claim-source contamination contract passed.")


if __name__ == "__main__":
    main()
