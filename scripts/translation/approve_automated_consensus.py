#!/usr/bin/env python3
"""Admit conservative two-judge consensus targets for provisional SFT only."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


MINIMUM_ADEQUACY = 4
MINIMUM_FLUENCY = 3
MINIMUM_TERMINOLOGY = 3


def rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def judgment_map(
    path: Path, queue: dict[str, dict[str, dict]]
) -> tuple[str, dict[str, dict]]:
    output: dict[str, dict] = {}
    judge_models: set[str] = set()
    for row in rows(path):
        source_id = str(row.get("source_id", "")).strip()
        if source_id not in queue or source_id in output:
            raise SystemExit(f"invalid or duplicate judgment source: {source_id}")
        if row.get("priority_status") != "automated-review-order-only-not-approval":
            raise SystemExit(f"invalid judgment status: {source_id}")
        judge_model = str(row.get("judge_model", "")).strip()
        if not judge_model:
            raise SystemExit(f"judgment has no judge model: {source_id}")
        judge_models.add(judge_model)
        assessments = row.get("assessments")
        if not isinstance(assessments, list) or len(assessments) != 3:
            raise SystemExit(f"judgment must contain three assessments: {source_id}")
        by_candidate = {
            str(assessment.get("candidate_id", "")): assessment
            for assessment in assessments
            if isinstance(assessment, dict)
        }
        if len(by_candidate) != 3 or set(by_candidate) != set(queue[source_id]):
            raise SystemExit(f"judgment candidate coverage mismatch: {source_id}")
        output[source_id] = {**row, "assessments": by_candidate}
    if len(judge_models) != 1:
        raise SystemExit(f"each judgment file must contain one judge model: {path}")
    if set(output) != set(queue):
        raise SystemExit(f"judgment source coverage mismatch: {path}")
    return next(iter(judge_models)), output


def select_candidate(
    judgment: dict,
    minimum_adequacy: int,
    minimum_fluency: int,
    minimum_terminology: int,
) -> tuple[str | None, str]:
    eligible: list[tuple[int, str]] = []
    for candidate_id, assessment in judgment["assessments"].items():
        if (
            assessment.get("critical_error") is not False
            or assessment.get("protected_tokens_preserved") is not True
            or assessment.get("error_tags") != []
            or assessment.get("adequacy", -1) < minimum_adequacy
            or assessment.get("fluency", -1) < minimum_fluency
            or assessment.get("terminology", -1) < minimum_terminology
        ):
            continue
        score = sum(
            int(assessment[name]) for name in ("adequacy", "fluency", "terminology")
        )
        eligible.append((score, candidate_id))
    if not eligible:
        return None, "no-error-free-candidate-meets-thresholds"
    eligible.sort(reverse=True)
    if len(eligible) > 1 and eligible[0][0] == eligible[1][0]:
        return None, "judge-has-no-unique-best-candidate"
    return eligible[0][1], "selected"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("review_queue", type=Path)
    parser.add_argument("judgments_a", type=Path)
    parser.add_argument("judgments_b", type=Path)
    parser.add_argument("approved", type=Path)
    parser.add_argument("rejected", type=Path)
    parser.add_argument("--minimum-adequacy", type=int, default=4)
    parser.add_argument("--minimum-fluency", type=int, default=3)
    parser.add_argument("--minimum-terminology", type=int, default=3)
    args = parser.parse_args()

    for value in (
        args.minimum_adequacy,
        args.minimum_fluency,
        args.minimum_terminology,
    ):
        if not 0 <= value <= 4:
            raise SystemExit("judge score thresholds must be between zero and four")
    if (
        args.minimum_adequacy < MINIMUM_ADEQUACY
        or args.minimum_fluency < MINIMUM_FLUENCY
        or args.minimum_terminology < MINIMUM_TERMINOLOGY
    ):
        raise SystemExit("automated consensus score thresholds cannot be weakened")
    for output in (args.approved, args.rejected):
        if output.exists() and output.stat().st_size:
            raise SystemExit(f"refusing to overwrite non-empty output: {output}")

    queue: dict[str, dict[str, dict]] = defaultdict(dict)
    for candidate in rows(args.review_queue):
        source_id = str(candidate.get("source_id", "")).strip()
        candidate_id = str(candidate.get("candidate_id", "")).strip()
        if not source_id or not candidate_id or candidate_id in queue[source_id]:
            raise SystemExit(f"invalid or duplicate review candidate: {source_id}/{candidate_id}")
        queue[source_id][candidate_id] = candidate
    if not queue or any(len(candidates) != 3 for candidates in queue.values()):
        raise SystemExit("every source must contain exactly three candidates")

    judge_a, judgments_a = judgment_map(args.judgments_a, queue)
    judge_b, judgments_b = judgment_map(args.judgments_b, queue)
    if judge_a == judge_b:
        raise SystemExit("automated consensus requires two distinct judge models")
    teacher_models = {
        str(candidate.get("teacher_model", "")).strip()
        for candidates in queue.values()
        for candidate in candidates.values()
    }
    if judge_a in teacher_models or judge_b in teacher_models:
        raise SystemExit("automated judge models must differ from the teacher")

    approved: list[dict] = []
    rejected: list[dict] = []
    for source_id, candidates in sorted(queue.items()):
        selected_a, reason_a = select_candidate(
            judgments_a[source_id],
            args.minimum_adequacy,
            args.minimum_fluency,
            args.minimum_terminology,
        )
        selected_b, reason_b = select_candidate(
            judgments_b[source_id],
            args.minimum_adequacy,
            args.minimum_fluency,
            args.minimum_terminology,
        )
        if selected_a is not None and selected_a == selected_b:
            approved.append(
                {
                    **candidates[selected_a],
                    "review_status": "two-judge-consensus-provisional",
                    "reviewer_ids": [],
                    "judge_model_ids": sorted([judge_a, judge_b]),
                    "automated_judgments": [
                        judgments_a[source_id],
                        judgments_b[source_id],
                    ],
                    "approved_alternative": None,
                    "promotion_eligible": False,
                    "automated_consensus_policy": {
                        "minimum_adequacy": args.minimum_adequacy,
                        "minimum_fluency": args.minimum_fluency,
                        "minimum_terminology": args.minimum_terminology,
                        "require_no_error_tags": True,
                        "require_no_critical_error": True,
                        "require_protected_tokens_preserved": True,
                        "require_unique_best_per_judge": True,
                        "require_matching_selection": True,
                    },
                }
            )
            continue
        rejected.append(
            {
                "source_id": source_id,
                "status": "automated-consensus-rejected",
                "judge_models": sorted([judge_a, judge_b]),
                "judge_a_selection": selected_a,
                "judge_b_selection": selected_b,
                "judge_a_reason": reason_a,
                "judge_b_reason": reason_b,
            }
        )

    write_jsonl(args.approved, approved)
    write_jsonl(args.rejected, rejected)
    print(
        json.dumps(
            {
                "approved": len(approved),
                "rejected": len(rejected),
                "sources": len(queue),
                "judge_models": sorted([judge_a, judge_b]),
                "use": "provisional supervised training only; never DQO or promotion evidence",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
