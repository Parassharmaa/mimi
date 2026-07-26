#!/usr/bin/env python3
"""Offline end-to-end proof that Claude shard identity reaches admission evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODEL = "claude-sonnet-5"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-claude-evidence-") as temporary:
        work = Path(temporary)
        queue = work / "queue.jsonl"
        queue_rows = [
            {
                "source_id": "s1",
                "candidate_id": candidate_id,
                "candidate_origin": origin,
                "source_language": "ja-JP",
                "target_language": "en-US",
                "domain": "fixture",
                "source": "開始してください。",
                "translation": translation,
                "teacher_model": "gpt-5.6-sol",
                "teacher_response_id": "teacher-response",
            }
            for candidate_id, origin, translation in (
                ("a", "teacher", "Please begin."),
                ("b", "licensed-reference", "Please start."),
                ("c", "current-mimi-baseline", "Start it."),
            )
        ]
        write_jsonl(queue, queue_rows)
        requests = work / "requests.jsonl"
        subprocess.run(
            [
                "python3",
                "scripts/translation/prepare_distillation_judge_batch.py",
                str(queue),
                str(requests),
                "--model",
                MODEL,
                "--reasoning-effort",
                "none",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assessments = [
            {
                "candidate_id": candidate_id,
                "adequacy": 4,
                "fluency": 4,
                "terminology": 4,
                "protected_tokens_preserved": True,
                "critical_error": False,
                "error_tags": [],
            }
            for candidate_id in ("a", "b", "c")
        ]
        batch_row = {
            "custom_id": "s1",
            "response": {
                "status_code": 200,
                "body": {
                    "id": "claude-cli-fixture",
                    "model": MODEL,
                    "system_fingerprint": None,
                    "output_text": json.dumps(
                        {"source_id": "s1", "assessments": assessments},
                        separators=(",", ":"),
                    ),
                },
            },
        }
        batch_output = work / "batch-output.jsonl"
        write_jsonl(batch_output, [batch_row])
        run = work / "run"
        result = run / "shards" / "00000.results.jsonl"
        write_jsonl(result, [batch_row])
        request_hash = sha256(requests)
        write_json(
            run / "manifest.json",
            {
                "schema_version": 2,
                "judge_model": MODEL,
                "request_sha256": request_hash,
                "candidate_origin_exposed": False,
                "reasoning_trace_stored": False,
                "shard_count": 1,
                "shards": [{"index": 0, "custom_ids": ["s1"]}],
            },
        )
        metadata_path = run / "shards" / "00000.metadata.json"
        write_json(
            metadata_path,
            {
                "custom_ids": ["s1"],
                "request_sha256": request_hash,
                "judge_model": MODEL,
                "result_sha256": sha256(result),
                "candidate_origin_exposed": False,
                "reasoning_trace_stored": False,
                "verified_primary_model": {"canonical_model": MODEL},
            },
        )
        write_json(
            batch_output.with_suffix(".jsonl.manifest.json"),
            {
                "status": "collected",
                "judge_model": MODEL,
                "output_sha256": sha256(batch_output),
                "candidate_origin_exposed": False,
                "reasoning_trace_stored": False,
            },
        )
        judgments = work / "judgments.jsonl"
        command = [
            "python3",
            "scripts/translation/prioritize_distillation_judgments.py",
            str(queue),
            str(batch_output),
            str(judgments),
            "--judge-requests",
            str(requests),
            "--claude-run-directory",
            str(run),
        ]
        subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        evidence = json.loads(
            judgments.with_suffix(".jsonl.manifest.json").read_text(encoding="utf-8")
        )
        assert evidence["actual_canonical_model_usage_verified"] is True
        assert evidence["judge_model"] == MODEL
        assert evidence["verified_shards"] == 1
        assert evidence["judgment_output"]["sha256"] == sha256(judgments)

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["verified_primary_model"]["canonical_model"] = "claude-opus-5"
        write_json(metadata_path, metadata)
        failed = subprocess.run(
            [*command[:4], str(work / "tampered-judgments.jsonl"), *command[5:]],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert failed.returncode != 0
        assert "identity proof is invalid" in (failed.stderr + failed.stdout)
    print("Claude judgment evidence propagation contract passed")


if __name__ == "__main__":
    main()
