#!/usr/bin/env python3
"""Contracts for portable Marian runtime-equivalence comparison."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/translation/compare_marian_portable_pack_reports.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report(path: Path, revision: str, hypothesis: str = "こんにちは") -> None:
    path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "status": "failed-runtime-safety",
                "modelRevision": revision,
                "summary": {
                    "status": "failed-runtime-safety",
                    "cases": 1,
                    "failures": 1,
                    "selectedEngineCounts": {"generalist-en-ja": 1},
                    "failureCounts": {"critical-token-mismatch": 1},
                    "runtimeAcceptedCases": 0,
                    "directionShortlistTokens": {},
                    "directionLatency": {"en-ja": {"p50Seconds": 1.0}},
                },
                "results": [
                    {
                        "caseID": "case-1",
                        "sourceLanguage": "en-US",
                        "targetLanguage": "ja-JP",
                        "domain": "test",
                        "source": "Hello",
                        "references": ["こんにちは"],
                        "hypothesis": hypothesis,
                        "outputTokenIDs": [1, 2],
                        "selectedEngine": "generalist-en-ja",
                        "selectedNeuralEngine": "generalist-en-ja",
                        "routedToExpert": False,
                        "routerScore": 0.25,
                        "criticalFallbackDirection": None,
                        "criticalFallbackUsed": False,
                        "criticalTokenGuardPasses": False,
                        "plausibilityGuardPasses": True,
                        "runtimeAccepted": False,
                        "failureReason": "critical-token-mismatch",
                        "outputShortlistTokens": None,
                        "claimEligible": False,
                        "latencySeconds": 1.0,
                        "warmLatencySeconds": [0.5],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-portable-compare-") as temporary:
        root = Path(temporary)
        source = root / "source"
        portable = root / "portable"
        (source / "engines/generalist-en-ja").mkdir(parents=True)
        (portable / "engines/generalist-en-ja").mkdir(parents=True)
        for pack in (source, portable):
            (pack / "engines/generalist-en-ja/model.safetensors").write_bytes(b"weights")
        (source / "engines/generalist-en-ja/manifest.json").write_text(
            '{"path":"/private/source"}', encoding="utf-8"
        )
        (portable / "engines/generalist-en-ja/manifest.json").write_text(
            '{"path":"Research/source"}\n', encoding="utf-8"
        )
        (source / "manifest.json").write_text('{"format":"source"}', encoding="utf-8")
        source_sha = sha256(source / "manifest.json")
        (portable / "manifest.json").write_text(
            json.dumps(
                {
                    "format": "portable",
                    "portableMetadata": {
                        "sourceManifestSha256": source_sha,
                        "weightPayloadUnchanged": True,
                    },
                }
            ),
            encoding="utf-8",
        )
        portable_sha = sha256(portable / "manifest.json")
        source_report = root / "source-report.json"
        portable_report = root / "portable-report.json"
        report(source_report, f"moe-manifest-sha256:{source_sha}")
        report(portable_report, f"moe-manifest-sha256:{portable_sha}")
        output = root / "comparison.json"
        command = [
            sys.executable,
            str(SCRIPT),
            str(source),
            str(portable),
            str(source_report),
            str(portable_report),
            str(output),
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        comparison = json.loads(output.read_text(encoding="utf-8"))
        assert comparison["status"] == "passed"
        assert comparison["runtime"]["exactCases"] == 1
        assert comparison["payload"]["nonManifestPayloadExact"] is True

        report(portable_report, f"moe-manifest-sha256:{portable_sha}", "さようなら")
        rejected = subprocess.run(command, text=True, capture_output=True, check=False)
        assert rejected.returncode != 0
        failed = json.loads(output.read_text(encoding="utf-8"))
        assert failed["status"] == "failed"
        assert failed["runtime"]["mismatchCaseIDs"] == ["case-1"]

        (portable / "engines/generalist-en-ja/model.safetensors").write_bytes(b"tampered")
        payload_rejected = subprocess.run(
            command, text=True, capture_output=True, check=False
        )
        assert payload_rejected.returncode != 0
        assert "changes non-manifest payload" in (
            payload_rejected.stdout + payload_rejected.stderr
        )
    print("Portable Marian pack runtime comparison contracts passed.")


if __name__ == "__main__":
    main()
