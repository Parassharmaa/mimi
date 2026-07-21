#!/usr/bin/env python3
"""Contracts for the Marian target-shortlist canary comparison."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/translation/compare_marian_target_shortlist.py"


def report(*, shortlist: bool) -> dict:
    results = []
    for index, source_language in enumerate(("en-US", "ja-JP"), start=1):
        results.append(
            {
                "caseID": f"case-{index}",
                "sourceLanguage": source_language,
                "targetLanguage": "ja-JP" if source_language == "en-US" else "en-US",
                "domain": "test",
                "source": f"source {index}",
                "references": [f"reference {index}"],
                "claimEligible": False,
                "hypothesis": f"output {index}",
                "outputTokenIDs": [index, 0],
                "selectedEngine": "generalist",
                "selectedNeuralEngine": "generalist",
                "routedToExpert": False,
                "routerScore": 0.0,
                "criticalTokenGuardPasses": True,
                "plausibilityGuardPasses": True,
                "runtimeAccepted": True,
                "failureReason": None,
            }
        )
    latency = 0.009 if shortlist else 0.010
    return {
        "schemaVersion": 1,
        "claimEligible": False,
        "modelRevision": "model+shortlist" if shortlist else "model",
        "peakResidentBytes": 101 if shortlist else 100,
        "preparationSeconds": 0.02 if shortlist else 0.01,
        "summary": {
            "directionLatency": {
                direction: {
                    "samples": 30,
                    "p50Seconds": latency,
                    "p95Seconds": latency,
                }
                for direction in ("en-ja", "ja-en")
            }
        },
        "results": results,
    }


with tempfile.TemporaryDirectory(prefix="mimi-shortlist-comparison-") as directory:
    directory = Path(directory)
    baseline = directory / "baseline.json"
    candidate = directory / "candidate.json"
    output = directory / "comparison.json"
    baseline.write_text(json.dumps(report(shortlist=False)), encoding="utf-8")
    candidate.write_text(json.dumps(report(shortlist=True)), encoding="utf-8")
    subprocess.run(
        ["python3", str(SCRIPT), str(baseline), str(candidate), str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    comparison = json.loads(output.read_text(encoding="utf-8"))
    assert comparison["status"] == "rejected"
    assert comparison["parity"]["exact"] is True
    assert comparison["latency"]["en-ja"]["passesMinimumSpeedup"] is True
    assert comparison["memory"]["passesNoIncreaseGate"] is False
    assert comparison["stopGate"]["decision"] == "stop at canary"
    assert comparison["doesNotAuthorizeAppIntegration"] is True

print("Marian target-shortlist comparison contracts passed.")
