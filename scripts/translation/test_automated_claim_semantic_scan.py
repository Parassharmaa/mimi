#!/usr/bin/env python3
"""Fast unit tests for semantic contamination scan helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/translation/scan_automated_claim_semantic_contamination.py"
SPEC = importlib.util.spec_from_file_location("mimi_semantic_scan", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def main() -> None:
    assert MODULE.language_bucket("Please open Settings.") == "en"
    assert MODULE.language_bucket("設定を開いてください。") == "ja"
    assert MODULE.language_bucket("Open 設定 now.") == "ja"
    queries, ready = MODULE.query_records(
        [
            {
                "id": "one",
                "source": "Hello.",
                "references": ["こんにちは。", "よろしくお願いします。"],
                "sourceLanguage": "en-US",
                "targetLanguage": "ja-JP",
                "claimEligible": True,
            }
        ]
    )
    assert ready is True
    assert [value["role"] for value in queries] == ["source", "reference[0]", "reference[1]"]
    assert [value["language"] for value in queries] == ["en", "ja", "ja"]
    queries, ready = MODULE.query_records(
        [
            {
                "id": "source-only",
                "source": "Fresh source.",
                "references": [],
                "sourceLanguage": "en-US",
                "targetLanguage": "ja-JP",
                "claimEligible": False,
            }
        ]
    )
    assert len(queries) == 1 and ready is False
    print("Mimi automated claim semantic scan helpers passed.")


if __name__ == "__main__":
    main()
