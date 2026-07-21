#!/usr/bin/env python3
"""Reproducibility test for the sealed automated reference request batch."""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "Research/translation/benchmark/automated-claim-v1.sources.jsonl"
PLAN = ROOT / "Research/translation/benchmark/automated-reference-model-plan-v1.json"
PROMPT = ROOT / "Research/translation/benchmark/automated-claim-v1.reference-generator.prompt.txt"
EXPECTED = ROOT / "Research/translation/work/automated-claim-v1/reference-generator-final-only-v2.requests.jsonl"
BUILDER = ROOT / "scripts/translation/prepare_automated_claim_reference_batch.py"
RUNNER = ROOT / "scripts/translation/run_synthetic_batch.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-reference-batch-") as temporary:
        output = Path(temporary) / "requests.jsonl"
        built = subprocess.run(
            ["python3", str(BUILDER), str(SOURCES), str(PLAN), str(PROMPT), str(output)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert built.returncode == 0, built.stderr
        assert sha256(output) == sha256(EXPECTED)
        validated = subprocess.run(
            ["python3", str(RUNNER), "validate", str(output)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert validated.returncode == 0, validated.stderr
        assert '"pipeline": "mimi-benchmark-reference-generator-v1"' in validated.stdout
        assert '"request_count": 800' in validated.stdout
        rejected = subprocess.run(
            ["python3", str(BUILDER), str(SOURCES), str(PLAN), str(PROMPT), str(output)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert rejected.returncode != 0 and "refusing to overwrite" in rejected.stderr
    print("Mimi automated claim reference Batch request contract passed.")


if __name__ == "__main__":
    main()
