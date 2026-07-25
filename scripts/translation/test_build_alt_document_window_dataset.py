#!/usr/bin/env python3
"""Contract test for coherent ALT document-window dataset construction."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/translation/build_alt_document_window_dataset.py"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in values),
        encoding="utf-8",
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-alt-window-") as temporary:
        root = Path(temporary)
        base, output = root / "base", root / "output"
        base.mkdir()
        common = {
            "source_language": "en-US",
            "target_language": "ja-JP",
            "source_license": "CC-BY-4.0",
            "source_provenance": "test",
            "attribution": "test",
            "origin": "human-alt-parallel",
        }
        write_jsonl(base / "train.jsonl", [
            {"id": "base", "source": "Base.", "target": "基本。", **common},
        ])
        write_jsonl(base / "valid.jsonl", [
            {"id": "valid", "source": "Valid.", "target": "検証。", **common},
        ])
        (base / "manifest.json").write_text(json.dumps({
            "direction": "en-ja",
            "target_source": "licensed-human-reference",
            "outputs": {
                "train": {"sha256": digest(base / "train.jsonl")},
                "valid": {"sha256": digest(base / "valid.jsonl")},
            },
        }), encoding="utf-8")
        alt = root / "alt.jsonl"
        write_jsonl(alt, [
            {
                "id": f"alt:SNT.1.{index}:en-ja",
                "source_id": f"SNT.1.{index}",
                "source": f"Sentence {index}.",
                "target": f"文{index}。",
                **common,
            }
            for index in range(1, 5)
        ])
        result = subprocess.run(
            [
                "python3",
                str(SCRIPT),
                str(base),
                str(alt),
                str(output),
                "--direction",
                "en-ja",
                "--window-size",
                "2",
                "--maximum-windows",
                "2",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        manifest = json.loads((output / "manifest.json").read_text())
        train = [
            json.loads(line)
            for line in (output / "train.jsonl").read_text().splitlines()
            if line.strip()
        ]
        windows = [row for row in train if row["origin"] == "human-alt-document-window"]
        assert len(windows) == 2
        assert all(row["window_size"] == 2 for row in windows)
        assert all(len(row["component_source_ids"]) == 2 for row in windows)
        assert manifest["counts"]["document_windows"] == 2
        assert manifest["target_source"] == "licensed-human-reference"

    print("ALT document-window dataset contract passed.")


if __name__ == "__main__":
    main()
