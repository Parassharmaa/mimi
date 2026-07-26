#!/usr/bin/env python3
"""Contract tests for the V20 typed-numeric curriculum builder."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "scripts/translation/build_typed_numeric_curriculum_v20.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def parent_row(identifier: str, source: str, target: str) -> dict:
    return {
        "id": identifier,
        "source": source,
        "target": target,
        "source_language": "en-US",
        "target_language": "ja-JP",
        "origin": "human-fixture",
        "domain": "fixture",
        "source_license": "CC-BY-4.0",
        "source_provenance": "fixture provenance",
        "attribution": "fixture attribution",
    }


def invoke(
    parent: Path,
    output: Path,
    suite: Path,
    *,
    expected_success: bool,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            str(parent),
            str(output),
            "--direction",
            "en-ja",
            "--screen-suite",
            str(suite),
            "--minimum-focus-train-rows",
            "1",
            "--minimum-surface-transform-train-rows",
            "1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if expected_success and result.returncode:
        raise AssertionError(result.stderr or result.stdout)
    if not expected_success and not result.returncode:
        raise AssertionError("builder unexpectedly accepted invalid input")
    return result


def main() -> None:
    with tempfile.TemporaryDirectory() as raw:
        work = Path(raw)
        parent = work / "parent"
        parent.mkdir()
        train = parent / "train.jsonl"
        valid = parent / "valid.jsonl"
        write_jsonl(
            train,
            [
                parent_row("focus", "Article seventy-four applies.", "第74条が適用される。"),
                parent_row("strict", "Keep 20%.", "20%を維持する。"),
                parent_row("plain", "Open the report.", "報告書を開いてください。"),
                parent_row("excluded", "The deadline is 3 days.", "期限は3日です。"),
            ],
        )
        write_jsonl(
            valid,
            [
                parent_row("valid-focus", "Section nine applies.", "第9節が適用される。"),
                parent_row("valid-plain", "Close the report.", "報告書を閉じてください。"),
            ],
        )
        manifest = {
            "schema_version": 1,
            "direction": "en-ja",
            "promotion_eligible": False,
            "outputs": {
                "train": {"sha256": sha256(train)},
                "valid": {"sha256": sha256(valid)},
            },
        }
        (parent / "manifest.json").write_text(
            json.dumps(manifest) + "\n",
            encoding="utf-8",
        )
        suite = work / "suite.jsonl"
        write_jsonl(
            suite,
            [
                {
                    "id": "held-out",
                    "source": "The deadline is 3 days.",
                    "segments": [],
                    "references": ["期限は3日です。"],
                    "referenceSegments": [],
                }
            ],
        )

        output = work / "output"
        invoke(parent, output, suite, expected_success=True)
        result = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        assert result["counts"]["output_train"] == 3
        assert result["counts"]["train"]["focus"] == 2
        assert (
            result["counts"]["train"]["class:bilingual-surface-transformation"]
            == 1
        )
        assert result["contamination_screen"]["excluded"]["train"] == 1
        output_rows = [
            json.loads(line)
            for line in (output / "train.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        by_id = {row["id"]: row for row in output_rows}
        assert by_id["focus"]["origin"] == "typed-numeric-target"
        assert "bilingual-surface-transformation" in by_id["focus"][
            "constraint_classes"
        ]
        assert by_id["plain"]["origin"] == "base-preservation-replay"
        assert all(
            row["text_derived_from_parent_without_modification"]
            for row in output_rows
        )

        manifest["outputs"]["train"]["sha256"] = "0" * 64
        (parent / "manifest.json").write_text(
            json.dumps(manifest) + "\n",
            encoding="utf-8",
        )
        invoke(parent, work / "invalid-output", suite, expected_success=False)

    print("Mimi V20 typed-numeric curriculum contract passed.")


if __name__ == "__main__":
    main()
