#!/usr/bin/env python3
"""Contract tests for diagnostic report intersection alignment."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/translation/align_translation_report_intersection.py"


def row(case_id: str, source_language: str, target_language: str) -> dict:
    return {
        "caseID": case_id,
        "sourceLanguage": source_language,
        "targetLanguage": target_language,
        "domain": "test",
        "source": f"source-{source_language}",
        "references": [f"reference-{target_language}"],
        "claimEligible": False,
        "hypothesis": f"hypothesis-{case_id}",
        "latencySeconds": 0.01,
        "warmLatencySeconds": [],
    }


def report(results: list[dict], engine: str) -> dict:
    return {
        "schemaVersion": 2,
        "engine": engine,
        "preparationSeconds": 0.0,
        "results": results,
    }


with tempfile.TemporaryDirectory(prefix="mimi-aligned-report-test-") as directory:
    root = Path(directory)
    candidate_path = root / "candidate.json"
    baseline_path = root / "baseline.json"
    candidate_output = root / "candidate-aligned.json"
    baseline_output = root / "baseline-aligned.json"
    suite_output = root / "aligned.jsonl"
    candidate_path.write_text(
        json.dumps(
            report(
                [row("candidate-en-ja", "en-US", "ja-JP"), row("candidate-ja-en", "ja-JP", "en-US")],
                "candidate",
            )
        ),
        encoding="utf-8",
    )
    baseline_path.write_text(
        json.dumps(
            report(
                [row("baseline-ja-en", "ja-JP", "en-US"), row("baseline-en-ja", "en-US", "ja-JP")],
                "baseline",
            )
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            "python3",
            str(SCRIPT),
            str(candidate_path),
            str(baseline_path),
            str(candidate_output),
            str(baseline_output),
            "--minimum-per-direction",
            "1",
            "--suite-output",
            str(suite_output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    candidate = json.loads(candidate_output.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_output.read_text(encoding="utf-8"))
    assert [row["caseID"] for row in candidate["results"]] == [
        row["caseID"] for row in baseline["results"]
    ]
    assert {row["originalCaseID"] for row in candidate["results"]} == {
        "candidate-en-ja",
        "candidate-ja-en",
    }
    assert candidate["diagnosticAlignment"]["postHocIntersection"] is True
    assert all(row["claimEligible"] is False for row in candidate["results"])
    suite = [json.loads(line) for line in suite_output.read_text().splitlines()]
    assert [row["id"] for row in suite] == [row["caseID"] for row in candidate["results"]]

print("Translation report intersection contract passed.")
