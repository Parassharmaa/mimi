#!/usr/bin/env python3
"""Contracts for paired learned-metric comparison."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/translation/compare_learned_metric.py"


def report(engine: str, scores: list[float]) -> dict:
    rows = []
    for index, score in enumerate(scores):
        rows.append(
            {
                "caseID": f"case-{index}",
                "sourceLanguage": "en" if index < 2 else "ja",
                "targetLanguage": "ja" if index < 2 else "en",
                "domain": "conversation" if index % 2 == 0 else "news",
                "score": score,
            }
        )
    return {
        "metric": "COMET-22",
        "modelRepository": "example/model",
        "modelRevision": "revision",
        "modelLicense": "Apache-2.0",
        "package": "unbabel-comet",
        "packageVersion": "2.2.7",
        "setuptoolsVersion": "80.9.0",
        "precision": "float32",
        "multipleReferenceAggregation": "mean",
        "signatureSHA256": "signature",
        "suiteSHA256": "suite",
        "engine": engine,
        "results": rows,
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-learned-metric-compare-") as temporary:
        root = Path(temporary)
        candidate = root / "candidate.json"
        baseline = root / "baseline.json"
        output = root / "comparison.json"
        candidate.write_text(json.dumps(report("candidate", [0.9, 0.8, 0.7, 0.6])))
        baseline.write_text(json.dumps(report("baseline", [0.8, 0.7, 0.6, 0.5])))
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                str(candidate),
                str(baseline),
                str(output),
                "--bootstrap-samples",
                "100",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        comparison = json.loads(output.read_text())
        assert math.isclose(
            comparison["directions"]["en>ja"]["meanPairedDelta"], 0.1
        )
        assert math.isclose(
            comparison["directions"]["ja>en"]["meanPairedDelta"], 0.1
        )
        interval = comparison["domains"]["en>ja/news"]["pairedBootstrapInterval"]
        assert math.isclose(interval["lower"], 0.1)
        assert math.isclose(interval["upper"], 0.1)

        mismatch = report("mismatch", [0.8, 0.7, 0.6, 0.5])
        mismatch["suiteSHA256"] = "other-suite"
        baseline.write_text(json.dumps(mismatch))
        failed = subprocess.run(
            [sys.executable, str(SCRIPT), str(candidate), str(baseline), str(root / "bad.json")],
            capture_output=True,
            text=True,
        )
        assert failed.returncode != 0
        assert "suiteSHA256" in failed.stderr

    print("Learned-metric comparison contracts passed.")


if __name__ == "__main__":
    main()
