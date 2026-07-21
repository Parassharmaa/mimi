#!/usr/bin/env python3
"""Contract test for the optional legal slice in public-stress suites."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/translation/prepare_public_stress_suite.py"


def write_pair(path: Path, corpus: str) -> None:
    source_id = f"{corpus}-1"
    rows = [
        {
            "id": f"{source_id}:en-ja",
            "source_id": source_id,
            "source_language": "en-US",
            "target_language": "ja-JP",
            "source": f"English source for {corpus}.",
            "target": f"{corpus}の日本語訳です。",
            "source_license": "PDL-1.0-compatible-CC-BY-4.0",
            "attribution": f"fixture attribution {corpus}",
        },
        {
            "id": f"{source_id}:ja-en",
            "source_id": source_id,
            "source_language": "ja-JP",
            "target_language": "en-US",
            "source": f"{corpus}の日本語訳です。",
            "target": f"English source for {corpus}.",
            "source_license": "PDL-1.0-compatible-CC-BY-4.0",
            "attribution": f"fixture attribution {corpus}",
        },
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-public-stress-") as temporary:
        root = Path(temporary)
        inputs = {}
        for corpus in ("kftt", "alt", "tatoeba", "jlt"):
            path = root / f"{corpus}.jsonl"
            write_pair(path, corpus)
            inputs[corpus] = path
        output = root / "suite.jsonl"
        command = [
            "python3",
            str(SCRIPT),
            str(inputs["kftt"]),
            str(inputs["alt"]),
            str(inputs["tatoeba"]),
            str(output),
            "--kftt-pairs",
            "1",
            "--alt-pairs",
            "1",
            "--tatoeba-pairs",
            "1",
            "--jlt-test",
            str(inputs["jlt"]),
            "--jlt-pairs",
            "1",
        ]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        manifest = json.loads(output.with_suffix(".manifest.json").read_text())
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        assert manifest["cases"] == 8
        assert manifest["cases_per_direction"] == 4
        assert manifest["inputs"]["jlt"]["selected_pairs"] == 1
        assert len(rows) == 8
        assert sum(row["sourceCorpus"] == "jlt" for row in rows) == 2
        assert all(row["claimEligible"] is False for row in rows)

    print("Public stress legal-slice contract passed.")


if __name__ == "__main__":
    main()
