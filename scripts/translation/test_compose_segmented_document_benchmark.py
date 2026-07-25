#!/usr/bin/env python3
"""Contract test for segmented-document benchmark composition."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/translation/compose_segmented_document_benchmark.py"


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-compose-doc-benchmark-") as temporary:
        root = Path(temporary)
        suite_path = root / "suite.jsonl"
        report_path = root / "segments.json"
        output_path = root / "output.json"
        case = {
            "id": "doc-1",
            "sourceLanguage": "en-US",
            "targetLanguage": "ja-JP",
            "domain": "long-document-news",
            "source": "First.\nSecond.",
            "references": ["一。\n二。"],
            "claimEligible": False,
            "sourceUnit": "document",
            "segmentCount": 2,
            "segments": ["First.", "Second."],
            "referenceSegments": ["一。", "二。"],
            "segmentBenchmarkIDs": ["doc-1:s1", "doc-1:s2"],
        }
        suite_path.write_text(
            json.dumps(case, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        report = {
            "schemaVersion": 1,
            "engine": "fixture-engine",
            "benchmarkConfiguration": {"warmRunsPerCase": 2},
            "results": [
                {
                    "caseID": "doc-1:s1",
                    "sourceLanguage": "en-US",
                    "targetLanguage": "ja-JP",
                    "source": "First.",
                    "references": ["一。"],
                    "hypothesis": "最初。",
                    "latencySeconds": 0.1,
                    "warmLatencySeconds": [0.08, 0.07],
                    "outputTokenIDs": [1],
                },
                {
                    "caseID": "doc-1:s2",
                    "sourceLanguage": "en-US",
                    "targetLanguage": "ja-JP",
                    "source": "Second.",
                    "references": ["二。"],
                    "hypothesis": "次。",
                    "latencySeconds": 0.2,
                    "warmLatencySeconds": [0.18, 0.17],
                    "outputTokenIDs": [2],
                },
            ],
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        result = subprocess.run(
            [
                "python3",
                str(SCRIPT),
                str(suite_path),
                str(report_path),
                str(output_path),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        output = json.loads(output_path.read_text())
        assert output["engine"] == "fixture-engine:segment-then-join"
        assert output["documentAggregation"]["crossSegmentContext"] is False
        assert len(output["results"]) == 1
        composed = output["results"][0]
        assert composed["hypothesis"] == "最初。\n次。"
        assert composed["emptySegmentCount"] == 0
        assert composed["emptySegmentIndexes"] == []
        assert composed["latencySeconds"] == 0.30000000000000004
        assert composed["warmLatencySeconds"] == [0.26, 0.24000000000000002]
        assert composed["segmentOutputTokenIDs"] == [[1], [2]]

        partial_output_path = root / "partial-output.json"
        partial_result = subprocess.run(
            [
                "python3",
                str(SCRIPT),
                str(suite_path),
                str(report_path),
                str(partial_output_path),
                "--direction",
                "en-ja",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert partial_result.returncode == 0, partial_result.stderr
        partial_output = json.loads(partial_output_path.read_text())
        assert [row["caseID"] for row in partial_output["results"]] == ["doc-1"]

    print("Segmented document benchmark composition contract passed.")


if __name__ == "__main__":
    main()
