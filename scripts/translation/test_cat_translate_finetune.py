#!/usr/bin/env python3
"""Smoke-test CAT-Translate data preparation and contamination filtering."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def training_row(identifier: str, source_language: str, target_language: str, source: str, target: str) -> dict:
    return {
        "id": identifier,
        "source_language": source_language,
        "target_language": target_language,
        "source": source,
        "target": target,
        "domain": "fixture",
        "origin": "fixture",
        "source_license": "CC-BY-4.0",
        "source_provenance": "fixture",
        "attribution": "fixture",
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-cat-data-test-") as temporary:
        root = Path(temporary)
        en_ja, ja_en, output = root / "en-ja", root / "ja-en", root / "output"
        write_jsonl(en_ja / "train.jsonl", [
            training_row("en-good", "en-US", "ja-JP", "A clean training sentence.", "きれいな学習文です。"),
            training_row("en-leak", "en-US", "ja-JP", "Protected benchmark sentence.", "保護された文です。"),
        ])
        write_jsonl(en_ja / "valid.jsonl", [
            training_row("en-valid", "en-US", "ja-JP", "A validation sentence.", "検証文です。")
        ])
        write_jsonl(ja_en / "train.jsonl", [
            training_row("ja-good", "ja-JP", "en-US", "別の学習文です。", "This is another training sentence.")
        ])
        write_jsonl(ja_en / "valid.jsonl", [
            training_row("ja-valid", "ja-JP", "en-US", "別の検証文です。", "This is another validation sentence.")
        ])
        protected = root / "protected.jsonl"
        write_jsonl(protected, [{
            "id": "heldout",
            "source": "Protected benchmark sentence.",
            "references": ["保護されたベンチマーク文です。"],
        }])
        subprocess.run([
            "python3",
            "scripts/translation/prepare_cat_translate_finetune.py",
            str(output),
            "--en-ja", str(en_ja),
            "--ja-en", str(ja_en),
            "--protected-suite", str(protected),
        ], check=True, capture_output=True, text=True)
        train = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
        valid = [json.loads(line) for line in (output / "valid.jsonl").read_text().splitlines()]
        manifest = json.loads((output / "manifest.json").read_text())
        assert [row["id"] for row in train] == ["en-good", "ja-good"]
        assert [row["id"] for row in valid] == ["en-valid", "ja-valid"]
        assert train[0]["prompt"].startswith("Translate the following English text into Japanese.")
        assert train[1]["prompt"].startswith("Translate the following Japanese text into English.")
        assert manifest["rejected"] == {"protected-train": 1}
        assert manifest["outputs"]["train"]["rows"] == 2
    print("CAT-Translate fine-tuning data contract passed.")


if __name__ == "__main__":
    main()
