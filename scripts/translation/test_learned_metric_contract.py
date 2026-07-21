#!/usr/bin/env python3
"""Offline report/signature contract for pinned COMET evaluation."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path


SCRIPT = Path(__file__).with_name("score_comet.py")
SPEC = importlib.util.spec_from_file_location("mimi_score_comet", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SCORER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCORER)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-comet-contract-") as temporary:
        work = Path(temporary)
        suite = work / "suite.jsonl"
        engine = work / "engine.json"
        rows = [
            {
                "id": "en-1",
                "sourceLanguage": "en-US",
                "targetLanguage": "ja-JP",
                "domain": "meeting-and-live-speech",
                "source": "Please wait.",
                "references": ["お待ちください。", "少々お待ちください。"],
                "hypothesis": "お待ちください。",
            },
            {
                "id": "ja-1",
                "sourceLanguage": "ja-JP",
                "targetLanguage": "en-US",
                "domain": "meeting-and-live-speech",
                "source": "少々お待ちください。",
                "references": ["Please wait a moment."],
                "hypothesis": "Please wait a moment.",
            },
        ]
        suite.write_text(
            "".join(
                json.dumps({key: value for key, value in row.items() if key != "hypothesis"})
                + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        engine.write_text(json.dumps({"engine": "fixture"}), encoding="utf-8")
        report = SCORER.build_report(
            suite,
            engine,
            rows,
            [0.8, 0.6, 0.9],
            model_repository=SCORER.DEFAULT_MODEL,
            model_revision=SCORER.DEFAULT_REVISION,
            package_version=SCORER.DEFAULT_PACKAGE_VERSION,
            setuptools_version=SCORER.DEFAULT_SETUPTOOLS_VERSION,
            torch_version="fixture",
        )
        assert report["results"][0]["score"] == 0.7
        assert report["results"][1]["score"] == 0.9
        assert report["precision"] == "float32"
        assert report["modelLicense"] == "Apache-2.0"
        assert len(report["signatureSHA256"]) == 64
    print("Mimi pinned learned-metric report contract passed.")


if __name__ == "__main__":
    main()
