#!/usr/bin/env python3
"""Contract test for held-out contamination filtering."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/translation/filter_training_dataset_against_protected.py"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-contamination-filter-") as temporary:
        root = Path(temporary)
        dataset, output = root / "dataset", root / "filtered"
        dataset.mkdir()
        write_jsonl(dataset / "train.jsonl", [
            {"id": "clean", "source": "A separate sentence.", "target": "別の文です。"},
            {"id": "exact", "source": "Held out sentence.", "target": "除外。"},
            {"id": "near", "source": "Held out sentence!", "target": "ほぼ一致。"},
        ])
        write_jsonl(dataset / "valid.jsonl", [
            {"id": "valid", "source": "Validation.", "target": "検証。"},
        ])
        (dataset / "manifest.json").write_text(
            json.dumps({
                "direction": "en-ja",
                "outputs": {
                    "train": {"sha256": digest(dataset / "train.jsonl")},
                    "valid": {"sha256": digest(dataset / "valid.jsonl")},
                },
            }),
            encoding="utf-8",
        )
        protected = root / "protected.jsonl"
        write_jsonl(protected, [{
            "id": "heldout",
            "source": "Held out sentence.",
            "references": ["保留文です。"],
        }])

        result = subprocess.run(
            [
                "python3",
                str(SCRIPT),
                str(dataset),
                str(output),
                "--protected-suite",
                str(protected),
                "--character-ngram-size",
                "3",
                "--maximum-jaccard",
                "0.8",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        manifest = json.loads((output / "manifest.json").read_text())
        filtered = [
            json.loads(line)
            for line in (output / "train.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert [row["id"] for row in filtered] == ["clean"]
        assert manifest["counts"]["excludedRows"] == 2
        assert manifest["counts"]["matchesByKind"] == {"exact": 1, "near": 1}
        assert manifest["zero_hits_at_threshold"] is False
        assert manifest["inputs"]["protected_suites"][0]["sha256"] == digest(protected)

    print("Training contamination filter contract passed.")


if __name__ == "__main__":
    main()
