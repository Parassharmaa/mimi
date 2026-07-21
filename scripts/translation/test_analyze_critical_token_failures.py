#!/usr/bin/env python3
"""Focused contract for critical-token failure taxonomy."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/translation/analyze_critical_token_failures.py"


def row(case_id: str, source: str, hypothesis: str, reference: str) -> dict:
    return {
        "caseID": case_id,
        "sourceLanguage": "en-US",
        "targetLanguage": "ja-JP",
        "domain": "fixture",
        "source": source,
        "hypothesis": hypothesis,
        "references": [reference],
        "selectedEngine": "generalist",
    }


with tempfile.TemporaryDirectory(prefix="mimi-critical-taxonomy-") as directory:
    root = Path(directory)
    report = root / "report.json"
    output = root / "taxonomy.json"
    report.write_text(
        json.dumps(
            {
                "results": [
                    row("suite:tatoeba:1:en-ja", "seven people", "7人", "7人"),
                    row("suite:alt:2:en-ja", "626", "30年", "626年"),
                    row("suite:kftt:3:en-ja", "02:00", "2時", "2時"),
                    row("suite:jlt:4:en-ja", "Keep {name}", "{other}を維持", "{name}を維持"),
                    row("suite:jlt:5:en-ja", "No token", "トークンなし", "トークンなし"),
                    row("suite:alt:6:en-ja", "Record 12.", "記録は12。", "記録は12。"),
                    row("suite:alt:7:en-ja", "Record 12.", "記録は13。", "記録は12。"),
                    row("suite:alt:8:en-ja", "Version 1.2.3.", "版1.2.3。", "版1.2.3。"),
                    row("suite:alt:9:en-ja", "Values 1,2", "値12", "値1、2"),
                ]
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
    assert payload["cases"] == 9
    assert payload["failures"] == 6
    assert payload["counts"]["referenceAlignment"] == {
        "hypothesis-matches-reference": 2,
        "source-and-reference-agree": 4,
    }
    assert payload["counts"]["numericRelation"] == {
        "digit-substitution-or-scale-change": 4,
        "non-numeric-structural-change": 1,
        "output-introduces-digits": 1,
    }
    assert {result["corpus"] for result in payload["results"]} == {
        "alt",
        "jlt",
        "kftt",
        "tatoeba",
    }
    assert payload["doesNotAuthorizeAppIntegration"] is True

print("Critical-token taxonomy contract passed.")
