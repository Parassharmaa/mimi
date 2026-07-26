#!/usr/bin/env python3
"""Contract test for provisional and reference-anchored automated selection."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APPROVER = ROOT / "scripts/translation/approve_automated_consensus.py"
BUILDER = ROOT / "scripts/translation/build_distillation_dataset.py"


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


def assessment(candidate_id: str, scores: tuple[int, int, int], *, critical: bool = False) -> dict:
    return {
        "candidate_id": candidate_id,
        "adequacy": scores[0],
        "fluency": scores[1],
        "terminology": scores[2],
        "protected_tokens_preserved": True,
        "critical_error": critical,
        "error_tags": ["meaning-reversal"] if critical else [],
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-automated-consensus-") as temporary:
        work = Path(temporary)
        queue_path = work / "queue.jsonl"
        judgment_a_path = work / "judge-a.jsonl"
        judgment_b_path = work / "judge-b.jsonl"
        approved_path = work / "approved.jsonl"
        rejected_path = work / "rejected.jsonl"

        queue = []
        for source_index in (1, 2, 3, 4):
            candidate_count = 4 if source_index in {3, 4} else 3
            for candidate_index in range(1, candidate_count + 1):
                queue.append(
                    {
                        "source_id": f"source-{source_index}",
                        "candidate_id": f"source-{source_index}:candidate-{candidate_index}",
                        "source_language": "en-US",
                        "target_language": "ja-JP",
                        "source": f"Example source {source_index}",
                        "translation": f"翻訳 {source_index}-{candidate_index}",
                        "domain": "everyday-conversation",
                        "source_license": "CC-BY-4.0",
                        "source_provenance": "fixture",
                        "teacher_model": "gpt-5.6-sol",
                        "teacher_response_id": f"response-{source_index}",
                        "candidate_origin": (
                            "licensed-reference"
                            if source_index in {3, 4} and candidate_index == 4
                            else "teacher"
                        ),
                        "licensed_reference": (
                            f"翻訳 {source_index}-4"
                            if source_index in {3, 4}
                            else None
                        ),
                        "reference_provenance": (
                            "offline licensed reference fixture"
                            if source_index in {3, 4}
                            else None
                        ),
                    }
                )
        write_jsonl(queue_path, queue)

        def judgment(source_index: int, judge_model: str, winner: int) -> dict:
            assessments = []
            candidate_count = 4 if source_index in {3, 4} else 3
            for candidate_index in range(1, candidate_count + 1):
                candidate_id = f"source-{source_index}:candidate-{candidate_index}"
                if candidate_index == winner:
                    assessments.append(assessment(candidate_id, (4, 4, 4)))
                elif candidate_index == 3:
                    assessments.append(assessment(candidate_id, (4, 4, 4), critical=True))
                else:
                    assessments.append(assessment(candidate_id, (4, 3, 3)))
            return {
                "source_id": f"source-{source_index}",
                "priority_status": "automated-review-order-only-not-approval",
                "judge_model": judge_model,
                "assessments": assessments,
            }

        write_jsonl(
            judgment_a_path,
            [
                judgment(1, "judge-a", 1),
                judgment(2, "judge-a", 1),
                judgment(3, "judge-a", 4),
                judgment(4, "judge-a", 1),
            ],
        )
        write_jsonl(
            judgment_b_path,
            [
                judgment(1, "judge-b", 1),
                judgment(2, "judge-b", 2),
                judgment(3, "judge-b", 4),
                judgment(4, "judge-b", 1),
            ],
        )
        result = subprocess.run(
            [
                "python3",
                str(APPROVER),
                str(queue_path),
                str(judgment_a_path),
                str(judgment_b_path),
                str(approved_path),
                str(rejected_path),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        approved_rows = [
            json.loads(line)
            for line in approved_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rejected = [
            json.loads(line)
            for line in rejected_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        approved = next(
            row for row in approved_rows if row["source_id"] == "source-1"
        )
        anchored_approved = next(
            row for row in approved_rows if row["source_id"] == "source-4"
        )
        assert approved["candidate_id"] == "source-1:candidate-1"
        assert approved["review_status"] == "two-judge-consensus-provisional"
        assert approved["judge_model_ids"] == ["judge-a", "judge-b"]
        assert approved["promotion_eligible"] is False
        assert approved["automated_consensus_policy"]["candidate_count"] == 3
        assert anchored_approved["candidate_id"] == "source-4:candidate-1"
        assert anchored_approved["review_status"] == "two-judge-reference-anchored"
        assert anchored_approved["promotion_eligible"] is True
        assert anchored_approved["automated_consensus_policy"]["candidate_count"] == 4
        assert (
            anchored_approved["automated_consensus_policy"][
                "licensed_reference_blinded_when_available"
            ]
            is True
        )
        assert {row["source_id"] for row in rejected} == {"source-2", "source-3"}
        reference_rejection = next(
            row for row in rejected if row["source_id"] == "source-3"
        )
        assert reference_rejection["judge_a_reason"] == "licensed-reference-won"
        assert reference_rejection["judge_b_reason"] == "licensed-reference-won"

        specification = importlib.util.spec_from_file_location("dataset_builder", BUILDER)
        assert specification and specification.loader
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        try:
            module.synthetic_rows(approved_path, "en-ja", "canonical")
        except SystemExit as error:
            assert "unauthorized automated consensus" in str(error)
        else:
            raise AssertionError("automated rows must require an explicit training-only flag")
        provisional_path = work / "approved-provisional.jsonl"
        anchored_path = work / "approved-reference-anchored.jsonl"
        write_jsonl(provisional_path, [approved])
        write_jsonl(anchored_path, [anchored_approved])
        provisional_training_rows = module.synthetic_rows(
            provisional_path,
            "en-ja",
            "canonical",
            allow_automated_consensus=True,
        )
        assert provisional_training_rows[0]["origin"] == (
            "automated-gpt-teacher-provisional"
        )
        assert provisional_training_rows[0]["promotion_eligible"] is False
        try:
            module.synthetic_rows(anchored_path, "en-ja", "canonical")
        except SystemExit as error:
            assert "reference-anchored consensus" in str(error)
        else:
            raise AssertionError("reference-anchored rows require an explicit flag")
        anchored_training_rows = module.synthetic_rows(
            anchored_path,
            "en-ja",
            "canonical",
            allow_reference_anchored_consensus=True,
        )
        assert anchored_training_rows[0]["origin"] == (
            "automated-gpt-teacher-reference-anchored"
        )
        assert anchored_training_rows[0]["promotion_eligible"] is True

    print("Mimi two-judge automated consensus contracts passed.")


if __name__ == "__main__":
    main()
