#!/usr/bin/env python3
"""Reproducibility contract for Mimi's source-only claim-suite draft."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/translation/prepare_automated_claim_sources.py"
EXPECTED_DOMAINS = {
    "meeting-and-live-speech": 120,
    "everyday-conversation": 80,
    "macos-and-technical-ui": 60,
    "numbers-dates-and-entities": 60,
    "politeness-ambiguity-and-omission": 60,
    "code-switching": 20,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output: Path, manifest: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python3", str(SCRIPT), str(output), "--manifest-output", str(manifest)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-claim-sources-") as temporary:
        work = Path(temporary)
        output_a, manifest_a = work / "a.jsonl", work / "a.manifest.json"
        output_b, manifest_b = work / "b.jsonl", work / "b.manifest.json"
        first, second = run(output_a, manifest_a), run(output_b, manifest_b)
        assert first.returncode == 0, first.stderr
        assert second.returncode == 0, second.stderr
        assert sha256(output_a) == sha256(output_b)
        rows = [json.loads(line) for line in output_a.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 800
        assert len({row["id"] for row in rows}) == 800
        assert len({row["source"] for row in rows}) == 800
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        for row in rows:
            direction = f"{row['sourceLanguage']}>{row['targetLanguage']}"
            counts[direction][row["domain"]] += 1
            assert row["claimEligible"] is False
            assert row["sourceGeneratedByAI"] is False
            assert row["references"] == []
            assert row["reviewStatus"] == "references-pending"
        assert set(counts) == {"en-US>ja-JP", "ja-JP>en-US"}
        assert all(dict(value) == EXPECTED_DOMAINS for value in counts.values())
        manifest = json.loads(manifest_a.read_text(encoding="utf-8"))
        assert manifest["status"] == "sources-frozen-references-pending"
        assert manifest["output"]["sha256"] == sha256(output_a)
        overwrite = run(output_a, manifest_a)
        assert overwrite.returncode != 0 and "refusing to overwrite" in overwrite.stderr

    print("Mimi automated claim-source reproducibility contract passed.")


if __name__ == "__main__":
    main()
