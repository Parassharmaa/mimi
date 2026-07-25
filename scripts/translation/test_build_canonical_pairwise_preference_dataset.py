#!/usr/bin/env python3
"""Offline fixture for the canonical teacher-over-current preference builder."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def assessment(candidate_id: str, scores: tuple[int, int, int]) -> dict:
    return {
        "candidate_id": candidate_id,
        "adequacy": scores[0],
        "fluency": scores[1],
        "terminology": scores[2],
        "critical_error": False,
        "protected_tokens_preserved": True,
        "error_tags": [],
    }


def approved_row(index: int, domain: str, preferred: bool = True) -> tuple[dict, list[dict]]:
    source_id = f"source-{index}"
    teacher_id = f"teacher-{index}"
    current_id = f"current-{index}"
    source = f"これは固有の試験文 {index} です。"
    chosen = f"This is unique test sentence {index}."
    current = f"This unique sentence is test {index}."
    judgments = []
    for judge_index, judge_model in enumerate(("judge-a", "judge-b")):
        teacher_scores = (4, 4, 4)
        current_scores = (4, 3, 3) if preferred else (4, 4, 4)
        judgments.append(
            {
                "source_id": source_id,
                "judge_model": judge_model,
                "judge_response_id": f"response-{index}-{judge_index}",
                "assessments": {
                    teacher_id: assessment(teacher_id, teacher_scores),
                    current_id: assessment(current_id, current_scores),
                    f"reference-{index}": assessment(f"reference-{index}", (4, 4, 4)),
                },
            }
        )
    approved = {
        "source_id": source_id,
        "candidate_id": teacher_id,
        "candidate_origin": "teacher",
        "promotion_eligible": True,
        "review_status": "two-judge-reference-anchored-canonical-absolute",
        "source": source,
        "translation": chosen,
        "source_language": "ja-JP",
        "target_language": "en-US",
        "domain": domain,
        "teacher_model": "teacher-model",
        "teacher_response_id": f"teacher-response-{index}",
        "judge_model_ids": ["judge-a", "judge-b"],
        "automated_judgments": judgments,
        "source_license": "CC-BY-4.0",
        "source_provenance": f"fixture source {index}",
    }
    queue = [
        {
            **{
                key: approved[key]
                for key in (
                    "source_id",
                    "source",
                    "source_language",
                    "target_language",
                    "domain",
                )
            },
            "candidate_id": teacher_id,
            "candidate_origin": "teacher",
            "translation": chosen,
        },
        {
            **{
                key: approved[key]
                for key in (
                    "source_id",
                    "source",
                    "source_language",
                    "target_language",
                    "domain",
                )
            },
            "candidate_id": current_id,
            "candidate_origin": "current-mimi-baseline",
            "translation": current,
        },
        {
            **{
                key: approved[key]
                for key in (
                    "source_id",
                    "source",
                    "source_language",
                    "target_language",
                    "domain",
                )
            },
            "candidate_id": f"reference-{index}",
            "candidate_origin": "licensed-reference",
            "translation": f"Reference translation {index}.",
        },
    ]
    return approved, queue


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-canonical-pairwise-test-") as temporary:
        work = Path(temporary)
        approved_path = work / "approved.jsonl"
        queue_path = work / "queue.jsonl"
        protected_path = work / "protected.jsonl"
        output = work / "output"
        approved: list[dict] = []
        queue: list[dict] = []
        for index, domain in enumerate(("news", "news", "legal", "legal", "ui"), start=1):
            row, candidates = approved_row(index, domain)
            approved.append(row)
            queue.extend(candidates)
        tied, tied_candidates = approved_row(99, "news", preferred=False)
        approved.append(tied)
        queue.extend(tied_candidates)
        write_jsonl(approved_path, approved)
        write_jsonl(queue_path, queue)
        write_jsonl(
            protected_path,
            [
                {
                    "id": "unrelated",
                    "source": "Completely unrelated protected source.",
                    "references": ["完全に無関係な保護対象。"],
                }
            ],
        )
        subprocess.run(
            [
                "python3",
                "scripts/translation/build_canonical_pairwise_preference_dataset.py",
                str(approved_path),
                str(queue_path),
                str(output),
                "--direction",
                "ja-en",
                "--protected-suite",
                str(protected_path),
                "--minimum-pairs",
                "2",
                "--required-judge-model",
                "judge-a",
                "--required-judge-model",
                "judge-b",
                "--experiment",
                "fixture-claude5-pairwise",
                "--id-prefix",
                "fixture-pair",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        train = [
            json.loads(line)
            for line in (output / "train.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        valid = [
            json.loads(line)
            for line in (output / "valid.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert manifest["counts"]["selected"] == 5
        assert manifest["experiment"] == "fixture-claude5-pairwise"
        assert manifest["required_judge_model_ids"] == ["judge-a", "judge-b"]
        assert all(row["id"].startswith("fixture-pair:") for row in train + valid)
        assert manifest["counts"]["rejected"] == {
            "not-unanimous-pareto-preferred": 1
        }
        assert len(train) == 3 and len(valid) == 2
        assert not ({row["source_id"] for row in train} & {row["source_id"] for row in valid})
        assert all(
            row["review_status"]
            == "two-model-unanimous-pareto-preferred-over-current"
            for row in train + valid
        )
        assert manifest["effective_licenses"]["train"] == {"CC-BY-4.0": 3}
        for split in ("train", "valid"):
            path = output / f"{split}.jsonl"
            assert (
                hashlib.sha256(path.read_bytes()).hexdigest()
                == manifest["outputs"][split]["sha256"]
            )
    print("canonical pairwise preference dataset fixture passed")


if __name__ == "__main__":
    main()
