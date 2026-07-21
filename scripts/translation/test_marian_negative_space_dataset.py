#!/usr/bin/env python3
"""Offline contracts for deterministic Marian negative-space generation."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/translation/build_marian_negative_space_dataset.py"
spec = importlib.util.spec_from_file_location("negative_space", SCRIPT)
assert spec and spec.loader
negative_space = importlib.util.module_from_spec(spec)
spec.loader.exec_module(negative_space)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def fixture_row(identifier: str, source: str, target: str, split: str) -> dict:
    return {
        "id": identifier,
        "source_id": identifier,
        "source": source,
        "target": target,
        "source_language": "en-US",
        "target_language": "ja-JP",
        "source_license": "CC-BY-4.0",
        "source_provenance": f"fixture/{split}/{identifier}",
        "attribution": "fixture",
        "origin": "human-alt-parallel",
        "domain": "test",
    }


numbered = negative_space.violations("価格は120円ではありません。", "ja-JP")
types = {row["violation_type"] for row in numbered}
assert {"number-substitution", "unit-substitution", "negation-reversal"} <= types
assert all(row["rejected"] != "価格は120円ではありません。" for row in numbered)
assert negative_space.violations(
    "Open {file} at https://example.com for 30 ms and do not close it.", "en-US"
)

with tempfile.TemporaryDirectory(prefix="mimi-negative-space-") as temporary:
    root = Path(temporary)
    parent, output = root / "parent", root / "output"
    parent.mkdir()
    train = [
        fixture_row(
            f"train-{index}",
            f"Open file {index} after 30 seconds.",
            f"ファイル{index}を30秒後に開いてください。これは閉じないでください。",
            "train",
        )
        for index in range(5)
    ]
    valid = [
        fixture_row(
            f"valid-{index}",
            f"Keep validation file {index} for 40 seconds.",
            f"検証ファイル{index}を40秒間保持してください。削除しないでください。",
            "valid",
        )
        for index in range(3)
    ]
    write_jsonl(parent / "train.jsonl", train)
    write_jsonl(parent / "valid.jsonl", valid)
    manifest = {
        "schema_version": 1,
        "direction": "en-ja",
        "outputs": {
            "train": {"sha256": negative_space.sha256(parent / "train.jsonl")},
            "valid": {"sha256": negative_space.sha256(parent / "valid.jsonl")},
        },
    }
    (parent / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    command = [
        sys.executable,
        str(SCRIPT),
        str(parent),
        str(output),
        "--direction",
        "en-ja",
        "--train-positives",
        "4",
        "--valid-positives",
        "2",
        "--maximum-violations",
        "4",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    produced = json.loads((output / "manifest.json").read_text())
    assert produced["selection"]["train_positive_rows"] == 4
    assert produced["counts"]["train_pairs"] == 16
    assert produced["private_reasoning_traces_used"] is False
    assert produced["free_form_synthetic_translations_used"] is False
    assert produced["outputs"]["train"]["sha256"] == negative_space.sha256(output / "train.jsonl")
    assert all(
        row["chosen"] != row["rejected"]
        for row in negative_space.load_jsonl(output / "train.jsonl")
    )

print("Marian negative-space dataset contracts passed.")
