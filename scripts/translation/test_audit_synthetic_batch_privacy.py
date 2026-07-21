#!/usr/bin/env python3
"""Contracts for the content-free Batch privacy audit."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/translation/audit_synthetic_batch_privacy.py"


def write(path: Path, values: list[dict]) -> None:
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


def run(source: Path, output: Path) -> dict:
    completed = subprocess.run(
        ["python3", str(SCRIPT), str(source), str(output), "--expected-count", "2"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(output.read_text(encoding="utf-8"))


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-batch-privacy-") as temporary:
        work = Path(temporary)
        clean = work / "clean.jsonl"
        write(clean, [
            {"custom_id": str(index), "error": None, "response": {"status_code": 200, "body": {
                "status": "completed", "output": [{"type": "message", "content": []}]
            }}}
            for index in range(2)
        ])
        clean_audit = run(clean, work / "clean-audit.json")
        assert clean_audit["admissible"] is True
        assert clean_audit["rowsWithEncryptedReasoning"] == 0

        rejected = work / "rejected.jsonl"
        write(rejected, [
            {"custom_id": "a", "error": None, "response": {"status_code": 200, "body": {
                "status": "completed", "output": [{"type": "message", "content": []}]
            }}},
            {"custom_id": "b", "error": None, "response": {"status_code": 200, "body": {
                "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
                "output": [{"type": "reasoning", "encrypted_content": "opaque"}]
            }}},
        ])
        rejected_audit = run(rejected, work / "rejected-audit.json")
        assert rejected_audit["admissible"] is False
        assert rejected_audit["bodyStatuses"] == {"completed": 1, "incomplete": 1}
        assert rejected_audit["rowsWithEncryptedReasoning"] == 1
        assert rejected_audit["incompleteReasons"] == {"max_output_tokens": 1}
    print("Mimi synthetic Batch privacy audit contracts passed.")


if __name__ == "__main__":
    main()
