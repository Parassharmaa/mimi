#!/usr/bin/env python3
"""Offline contract test for preregistered canonical dual certification."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assessment(candidate_id: str, passing: bool = True) -> dict:
    return {
        "candidate_id": candidate_id,
        "adequacy": 4 if passing else 2,
        "fluency": 4,
        "terminology": 4,
        "protected_tokens_preserved": True,
        "critical_error": False,
        "error_tags": [] if passing else ["omission"],
    }


def judgment(source_id: str, model: str, teacher_passes: bool) -> dict:
    return {
        "source_id": source_id,
        "priority_status": "automated-review-order-only-not-approval",
        "judge_model": model,
        "judge_response_id": f"{model}-response",
        "judge_system_fingerprint": None,
        "assessments": [
            assessment("baseline"),
            assessment("reference"),
            assessment("teacher", teacher_passes),
        ],
    }


def write_evidence(
    path: Path,
    *,
    model: str,
    request_sha256: str,
    queue_sha256: str,
) -> None:
    evidence = {
        "schema_version": 1,
        "judge_model": model,
        "actual_canonical_model_usage_verified": True,
        "candidate_origin_exposed": False,
        "reasoning_trace_stored": False,
        "request_file": {"sha256": request_sha256},
        "review_queue": {"sha256": queue_sha256},
        "judgment_output": {"sha256": sha256(path)},
    }
    path.with_suffix(path.suffix + ".manifest.json").write_text(
        json.dumps(evidence), encoding="utf-8"
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        queue_path = root / "queue.jsonl"
        queue = [
            {
                "source_id": "s1",
                "candidate_id": candidate_id,
                "candidate_origin": origin,
                "source_language": "en-US",
                "target_language": "ja-JP",
                "domain": "fixture",
                "source": "Start Mimi.",
                "translation": translation,
                "teacher_model": "teacher-model",
            }
            for candidate_id, origin, translation in (
                ("teacher", "teacher", "Mimiを開始してください。"),
                ("reference", "licensed-reference", "Mimiを起動してください。"),
                ("baseline", "current-mimi-baseline", "Mimiを始めてください。"),
            )
        ]
        write_jsonl(queue_path, queue)
        pilot_seeds = root / "seeds.jsonl"
        write_jsonl(pilot_seeds, [{"id": "s1"}])
        contract_path = root / "contract.json"
        contract_path.write_text(json.dumps({
            "outputs": {"pilotSeeds": {"sha256": sha256(pilot_seeds)}},
            "judgePolicy": {
                "requiredJudgeModelIds": ["judge-a", "judge-b"],
                "sameBlindedBatchRequired": True,
                "actualCanonicalModelUsageRequiredPerShard": True,
                "fallbackModelAllowed": False,
            },
            "judgeRequests": {
                "judge-a": {"sha256": "request-a"},
                "judge-b": {"sha256": "request-b"},
            },
            "goGateBeforeScaling": {
                "acceptancePolicy": "dual-absolute-canonical",
                "requireBothJudgesCertifyCanonicalAbsoluteThresholds": True,
                "requireZeroDeterministicCriticalFailures": True,
                "trainingAuthorizedByPilotAlone": False,
            },
        }), encoding="utf-8")
        queue_manifest = root / "queue-manifest.json"
        queue_manifest.write_text(json.dumps({
            "candidateOriginExposedToJudges": False,
            "inputs": {"pilotSeeds": {"sha256": sha256(pilot_seeds)}},
            "output": {"sha256": sha256(queue_path)},
        }), encoding="utf-8")

        a = root / "a.jsonl"
        b = root / "b.jsonl"
        write_jsonl(a, [judgment("s1", "judge-a", True)])
        write_jsonl(b, [judgment("s1", "judge-b", True)])
        write_evidence(
            a,
            model="judge-a",
            request_sha256="request-a",
            queue_sha256=sha256(queue_path),
        )
        write_evidence(
            b,
            model="judge-b",
            request_sha256="request-b",
            queue_sha256=sha256(queue_path),
        )
        approved = root / "approved.jsonl"
        rejected = root / "rejected.jsonl"
        command = [
            "python3",
            "scripts/translation/approve_canonical_quality_consensus.py",
            str(contract_path),
            str(queue_manifest),
            str(queue_path),
            str(a),
            str(b),
            str(approved),
            str(rejected),
        ]
        subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
        approved_rows = [
            json.loads(line) for line in approved.read_text().splitlines()
        ]
        assert len(approved_rows) == 1
        assert (
            approved_rows[0]["canonical_absolute_consensus_policy"][
                "unique_best_not_required"
            ]
            is True
        )
        assert rejected.read_text() == ""

        b_fail = root / "b-fail.jsonl"
        write_jsonl(b_fail, [judgment("s1", "judge-b", False)])
        write_evidence(
            b_fail,
            model="judge-b",
            request_sha256="request-b",
            queue_sha256=sha256(queue_path),
        )
        approved_fail = root / "approved-fail.jsonl"
        rejected_fail = root / "rejected-fail.jsonl"
        subprocess.run(
            [
                *command[:-3],
                str(b_fail),
                str(approved_fail),
                str(rejected_fail),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert approved_fail.read_text() == ""
        assert len(rejected_fail.read_text().splitlines()) == 1

    print("canonical absolute consensus contract passed")


if __name__ == "__main__":
    main()
