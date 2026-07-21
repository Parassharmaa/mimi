#!/usr/bin/env python3
"""Contracts for safely reusing unaffected local-teacher rows during retries."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from run_local_qwen_teacher import reusable_results, sha256


MODEL = "mlx-community/Qwen3-8B-4bit"
REVISION = "545dc4251c05440727734bcd94334791f6ab0192"
LICENSE = "Apache-2.0"


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-qwen-retry-") as temporary:
        root = Path(temporary)
        suite_path = root / "suite.jsonl"
        suite = [{
            "id": "fixture",
            "sourceLanguage": "en-US",
            "targetLanguage": "ja-JP",
            "domain": "professional-wikipedia-hard",
            "source": "A source.",
            "references": ["参照です。"],
        }]
        suite_path.write_text(json.dumps(suite[0], ensure_ascii=False) + "\n")
        report_path = root / "teacher.json"
        report = {
            "claimEligible": False,
            "referenceExposedToTeacher": False,
            "studentHypothesisExposedToTeacher": False,
            "reasoningTraceRequestedOrStored": False,
            "modelRepository": MODEL,
            "modelRevision": REVISION,
            "modelLicense": LICENSE,
            "suite": {"sha256": sha256(suite_path)},
            "results": [{
                "caseID": "fixture",
                "sourceLanguage": "en-US",
                "targetLanguage": "ja-JP",
                "domain": "professional-wikipedia-hard",
                "source": "A source.",
                "references": ["参照です。"],
                "hypothesis": "翻訳です。",
            }],
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False))
        loaded, indexed = reusable_results(
            report_path,
            suite,
            suite_path=suite_path,
            model=MODEL,
            revision=REVISION,
            model_license=LICENSE,
        )
        assert loaded["modelRevision"] == REVISION
        assert indexed["fixture"]["hypothesis"] == "翻訳です。"

        report["results"][0]["source"] = "Changed."
        report_path.write_text(json.dumps(report, ensure_ascii=False))
        try:
            reusable_results(
                report_path,
                suite,
                suite_path=suite_path,
                model=MODEL,
                revision=REVISION,
                model_license=LICENSE,
            )
        except SystemExit as error:
            assert "disagrees with suite source" in str(error)
        else:
            raise AssertionError("stale reusable source was accepted")

    print("Local Qwen targeted-retry contracts passed.")


if __name__ == "__main__":
    main()
