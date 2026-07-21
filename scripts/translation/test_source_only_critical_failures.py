#!/usr/bin/env python3
"""Contract for the source-only critical-token taxonomy."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/translation/analyze_source_only_critical_failures.py"


def row(case_id: str, source: str, hypothesis: str) -> dict:
    return {
        "caseID": case_id,
        "sourceLanguage": "en-US",
        "targetLanguage": "ja-JP",
        "domain": "fixture",
        "sourceTemplateID": "fixture:t1",
        "selectedEngine": "generalist-en-ja",
        "source": source,
        "hypothesis": hypothesis,
        "references": [],
    }


with tempfile.TemporaryDirectory(prefix="mimi-source-only-critical-") as directory:
    root = Path(directory)
    report = root / "report.json"
    output = root / "output.json"
    report.write_text(
        json.dumps(
            {
                "summary": {"failureCounts": {"critical-token-mismatch": 3}},
                "results": [
                    row("exact", "Meet on 2027-01-12.", "2027-01-12日に会いましょう。"),
                    row("temporal", "Meet on 2027-01-03.", "2027年1月3日に会いましょう。"),
                    row("wrong-date", "Meet on 2027-01-03.", "2027年1月4日に会いましょう。"),
                    row("duplicated-date", "Meet on 2027-01-03.", "2027年1月3日-3日に会いましょう。"),
                ],
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(
        ["python3", str(SCRIPT), str(report), str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "diagnostic-only"
    assert payload["cases"] == 4
    assert payload["strictFailures"] == 3
    assert payload["counts"]["relation"] == {
        "narrow-temporal-candidate": 1,
        "unresolved-strict-mismatch": 2,
    }, payload["counts"]
    assert payload["policy"]["runtimePolicyChanged"] is False
    assert payload["claimEligible"] is False

print("Source-only critical-token taxonomy contract passed.")
