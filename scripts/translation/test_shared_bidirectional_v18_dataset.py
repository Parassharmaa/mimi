#!/usr/bin/env python3
"""Contract test for the V18 licensed bidirectional dataset builder."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "scripts/translation/build_shared_bidirectional_v18_dataset.py"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def row(identifier: str, direction: str, source: str, target: str, **extra) -> dict:
    languages = {
        "en-ja": ("en-US", "ja-JP"),
        "ja-en": ("ja-JP", "en-US"),
    }
    source_language, target_language = languages[direction]
    return {
        "id": identifier,
        "source": source,
        "target": target,
        "source_language": source_language,
        "target_language": target_language,
        "source_license": "CC-BY-4.0",
        "source_provenance": "fixture",
        "attribution": "fixture attribution",
        "origin": "human-alt-parallel",
        **extra,
    }


def write_dataset(root: Path, train: list[dict], valid: list[dict]) -> None:
    root.mkdir()
    paths = {"train": root / "train.jsonl", "valid": root / "valid.jsonl"}
    write_jsonl(paths["train"], train)
    write_jsonl(paths["valid"], valid)
    manifest = {
        "outputs": {
            split: {"sha256": digest(path)} for split, path in paths.items()
        }
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest) + "\n",
        encoding="utf-8",
    )


with tempfile.TemporaryDirectory(prefix="mimi-v18-dataset-") as temporary:
    root = Path(temporary)
    en_ja = root / "en-ja"
    ja_en = root / "ja-en"
    write_dataset(
        en_ja,
        [
            row("en-keep", "en-ja", "A clean source.", "クリーンな原文。"),
            row(
                "en-ineligible",
                "en-ja",
                "A training-only source.",
                "訓練専用。",
                training_only=True,
            ),
            row("en-protected", "en-ja", "Do not leak this.", "漏らさない。"),
        ],
        [row("en-valid", "en-ja", "Fresh validation.", "新しい検証。")],
    )
    write_dataset(
        ja_en,
        [
            row("ja-one", "ja-en", "一つ目の原文。", "First clean source."),
            row("ja-two", "ja-en", "二つ目の原文。", "Second clean source."),
        ],
        [row("ja-valid", "ja-en", "新しい評価。", "Fresh evaluation.")],
    )
    protected = root / "protected.jsonl"
    write_jsonl(
        protected,
        [{"id": "protected", "source": "Do not leak this.", "references": []}],
    )
    output = root / "output"
    result = subprocess.run(
        [
            "python3",
            str(BUILDER),
            str(en_ja),
            str(ja_en),
            str(output),
            "--protected-suite",
            str(protected),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["promotion_eligible"] is True
    assert manifest["training_authorized"] is False
    assert manifest["licensed_human_targets_only"] is True
    assert manifest["v17_generated_candidates_used"] is False
    assert manifest["counts"]["retained"]["en-ja"]["train"] == 1
    assert manifest["counts"]["retained"]["ja-en"]["train"] == 2
    assert manifest["counts"]["balanced_train_per_direction"] == 2
    assert manifest["counts"]["train"] == 4
    assert manifest["counts"]["repeated_train_rows"] == 1
    assert manifest["counts"]["exclusions_by_reason"] == {
        "explicitly-training-only": 1,
        "protected-overlap": 1,
    }
    train_rows = [
        json.loads(line)
        for line in (output / "train.jsonl").read_text().splitlines()
    ]
    assert {item["direction"] for item in train_rows} == {"en-ja", "ja-en"}
    assert all(item.get("training_only") is not True for item in train_rows)
    assert all(item["source"] != "Do not leak this." for item in train_rows)

print("V18 shared bidirectional dataset contract passed")
