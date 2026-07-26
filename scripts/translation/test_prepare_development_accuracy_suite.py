#!/usr/bin/env python3
"""Contract test for the mixed sentence/document development-suite builder."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/translation/prepare_development_accuracy_suite.py"
SOURCE = ROOT / "research/translation/benchmark/public-stress-v3.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-development-accuracy-") as temporary:
        root = Path(temporary)
        output = root / "suite.jsonl"
        segments = root / "segments.jsonl"
        command = [
            "python3",
            str(SCRIPT),
            str(SOURCE),
            str(output),
            str(segments),
            "--sentence-pairs-per-corpus",
            "1",
            "--document-segments",
            "2",
            "--alt-documents",
            "1",
            "--kftt-documents",
            "1",
            "--jlt-documents",
            "1",
        ]
        result = subprocess.run(
            command, cwd=ROOT, text=True, capture_output=True, check=False
        )
        assert result.returncode == 0, result.stderr

        rows = load_jsonl(output)
        flat = load_jsonl(segments)
        manifest = json.loads(output.with_suffix(".manifest.json").read_text())
        assert len(rows) == 14
        assert len(flat) == 20
        assert manifest["caseCounts"] == {
            "documentsPerDirection": 3,
            "perDirection": 7,
            "sentencesPerDirection": 4,
            "total": 14,
        }
        assert manifest["segmentCounts"]["perDirection"] == 10
        assert manifest["sealedPromotionSuiteTouched"] is False
        assert sum(row["sourceUnit"] == "document" for row in rows) == 6
        assert all(row["claimEligible"] is False for row in rows)
        assert all(
            len(row["segments"]) == row["segmentCount"]
            and len(row["referenceSegments"]) == row["segmentCount"]
            and len(row["segmentBenchmarkIDs"]) == row["segmentCount"]
            for row in rows
        )
        assert len({row["id"] for row in flat}) == len(flat)

    print("Development accuracy suite contract passed.")


if __name__ == "__main__":
    main()
