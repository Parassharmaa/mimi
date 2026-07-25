#!/usr/bin/env python3
"""Fixture test for the post-hoc canonical EN→JA safety router."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/translation/apply_canonical_safety_router.py"


def row(case_id: str, source_language: str, source: str, hypothesis: str) -> dict:
    target_language = "ja-JP" if source_language == "en-US" else "en-US"
    return {
        "caseID": case_id,
        "sourceLanguage": source_language,
        "targetLanguage": target_language,
        "domain": "fixture",
        "source": source,
        "references": ["reference"],
        "claimEligible": False,
        "hypothesis": hypothesis,
        "outputTokenIDs": [1, 2],
        "latencySeconds": 0.01,
        "warmLatencySeconds": [0.01],
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-canonical-router-") as temporary:
        root = Path(temporary)
        models = []
        for index in range(3):
            model = root / f"model-{index}"
            model.mkdir()
            (model / "manifest.json").write_text(
                json.dumps({"direction": "fixture"}) + "\n", encoding="utf-8"
            )
            (model / "weights.bin").write_bytes(bytes([index]))
            models.append(model)

        baseline_rows = [
            row("clean", "en-US", "Hello.", "こんにちは。"),
            row("critical", "en-US", "Use 42 items.", "42個を使用します。"),
            row("fallback", "en-US", "Welcome.", "ようこそ。"),
            row("ja-en", "ja-JP", "こんにちは。", "Hello."),
        ]
        expert_rows = [
            row("clean", "en-US", "Hello.", "どうも。"),
            row("critical", "en-US", "Use 42 items.", "43個を使用します。"),
            row("fallback", "en-US", "Welcome.", "2026年へようこそ。"),
            row("ja-en", "ja-JP", "こんにちは。", "Wrong expert."),
        ]
        common = {
            "schemaVersion": 1,
            "engine": "fixture",
            "benchmarkConfiguration": {"maximumGeneratedTokens": 192},
        }
        baseline = root / "baseline.json"
        expert = root / "expert.json"
        output = root / "output.json"
        baseline.write_text(
            json.dumps({**common, "results": baseline_rows}) + "\n",
            encoding="utf-8",
        )
        expert.write_text(
            json.dumps({**common, "results": expert_rows}) + "\n",
            encoding="utf-8",
        )
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                str(baseline),
                str(expert),
                *(str(model) for model in models),
                str(output),
            ],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        report = json.loads(output.read_text(encoding="utf-8"))
        routes = {
            value["caseID"]: value["selectedEngine"]
            for value in report["results"]
        }
        assert routes == {
            "clean": "expert-en-ja",
            "critical": "baseline-en-ja-preflight",
            "fallback": "baseline-en-ja-output-fallback",
            "ja-en": "baseline-ja-en",
        }
        assert report["routingPolicy"]["usesReferenceAtRuntime"] is False
        assert report["claimEligible"] is False
        assert report["doesNotAuthorizeAppIntegration"] is True

    print("canonical safety router fixture passed")


if __name__ == "__main__":
    main()
