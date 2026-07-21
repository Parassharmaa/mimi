#!/usr/bin/env python3
"""Contracts for the frozen M2M-100 architecture screen."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/translation/prepare_m2m100_feasibility_suite.py"
DOMAINS = [
    "human-translated-news",
    "ministry-published-legal",
    "professional-wikipedia",
    "everyday-conversation",
]
DIRECTIONS = [("en-US", "ja-JP"), ("ja-JP", "en-US")]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-m2m100-suite-") as temporary:
        root = Path(temporary)
        source = root / "source.jsonl"
        output = root / "screen.jsonl"
        manifest = root / "screen.manifest.json"
        rows = []
        for source_language, target_language in DIRECTIONS:
            for domain in DOMAINS:
                for index in range(3):
                    identifier = f"{source_language}:{target_language}:{domain}:{index}"
                    rows.append(
                        {
                            "id": identifier,
                            "sourceLanguage": source_language,
                            "targetLanguage": target_language,
                            "domain": domain,
                            "source": f"source {identifier}",
                            "references": [f"reference {identifier}"],
                            "claimEligible": False,
                        }
                    )
        source.write_text(
            "".join(json.dumps(row) + "\n" for row in reversed(rows)),
            encoding="utf-8",
        )
        command = [
            sys.executable,
            str(SCRIPT),
            str(source),
            str(output),
            str(manifest),
            "--per-domain",
            "2",
            "--seed",
            "fixture-seed",
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        selected = [json.loads(line) for line in output.read_text().splitlines()]
        metadata = json.loads(manifest.read_text())
        assert len(selected) == 16
        assert all(row["claimEligible"] is False for row in selected)
        assert all(row["screenRole"] == "model-independent-development-architecture-gate" for row in selected)
        assert metadata["selectedWithoutCandidateOutputs"] is True
        assert metadata["output"]["sha256"] == sha256(output)
        assert set(metadata["directions"].values()) == {8}
        assert all(value["selected"] == 2 for value in metadata["selection"].values())

        duplicate = subprocess.run(command, text=True, capture_output=True, check=False)
        assert duplicate.returncode != 0
        assert "refusing to overwrite" in duplicate.stdout + duplicate.stderr

        first = json.loads(source.read_text().splitlines()[0])
        first["references"] = []
        bad_source = root / "bad-source.jsonl"
        bad_output = root / "bad-output.jsonl"
        bad_manifest = root / "bad-output.manifest.json"
        bad_source.write_text(json.dumps(first) + "\n", encoding="utf-8")
        rejected = subprocess.run(
            [sys.executable, str(SCRIPT), str(bad_source), str(bad_output), str(bad_manifest)],
            text=True,
            capture_output=True,
            check=False,
        )
        assert rejected.returncode != 0
        assert not bad_output.exists()
    print("M2M-100 feasibility-suite contracts passed.")


if __name__ == "__main__":
    main()
