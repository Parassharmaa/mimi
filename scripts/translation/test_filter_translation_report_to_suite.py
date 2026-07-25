#!/usr/bin/env python3
"""Contracts for exact benchmark-suite report subsetting."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/translation/filter_translation_report_to_suite.py"


def case(
    case_id: str,
    source: str,
    source_language: str = "en-US",
    target_language: str = "ja-JP",
) -> dict:
    return {
        "caseID": case_id,
        "sourceLanguage": source_language,
        "targetLanguage": target_language,
        "domain": "fixture",
        "source": source,
        "references": [f"訳:{source}"],
        "claimEligible": False,
        "hypothesis": f"仮説:{source}",
    }


with tempfile.TemporaryDirectory(prefix="mimi-report-subset-") as directory:
    root = Path(directory)
    suite_path = root / "suite.jsonl"
    report_path = root / "report.json"
    output_path = root / "output.json"
    first = case("first", "First")
    second = case("second", "Second")
    suite_row = {"id": second["caseID"], **{key: second[key] for key in (
        "sourceLanguage",
        "targetLanguage",
        "domain",
        "source",
        "references",
        "claimEligible",
    )}}
    suite_path.write_text(json.dumps(suite_row) + "\n", encoding="utf-8")
    report_path.write_text(
        json.dumps({"schemaVersion": 2, "results": [first, second]}),
        encoding="utf-8",
    )
    subprocess.run(
        ["python3", str(SCRIPT), str(suite_path), str(report_path), str(output_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["authenticatedSubset"]["cases"] == 1
    assert [row["caseID"] for row in payload["results"]] == ["second"]

    ja_en = case("ja-en", "三番目", "ja-JP", "en-US")
    suite_path.write_text(
        "".join(
            json.dumps(
                {"id": row["caseID"], **{key: row[key] for key in (
                    "sourceLanguage",
                    "targetLanguage",
                    "domain",
                    "source",
                    "references",
                    "claimEligible",
                )}}
            )
            + "\n"
            for row in (second, ja_en)
        ),
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps({"schemaVersion": 2, "results": [second, ja_en]}),
        encoding="utf-8",
    )
    subprocess.run(
        [
            "python3",
            str(SCRIPT),
            str(suite_path),
            str(report_path),
            str(output_path),
            "--direction",
            "ja-en",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["authenticatedSubset"]["cases"] == 1
    assert [row["caseID"] for row in payload["results"]] == ["ja-en"]

print("Translation-report suite subset contracts passed.")
