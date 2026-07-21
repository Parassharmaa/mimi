#!/usr/bin/env python3
"""Validate one fast-judge Batch output and create an auditable judgment file.

One judge can only prioritize review. Two files from distinct judge models may
later feed the explicitly provisional automated-consensus SFT gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ASSESSMENT_KEYS = {
    "candidate_id",
    "adequacy",
    "fluency",
    "terminology",
    "protected_tokens_preserved",
    "critical_error",
    "error_tags",
}
ALLOWED_ERROR_TAGS = {
    "meaning-reversal",
    "negation",
    "number-or-date",
    "named-entity",
    "omission",
    "addition",
    "register",
    "terminology",
    "disfluency",
}


def rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def response_payload(batch_row: dict) -> tuple[dict, dict]:
    response = batch_row.get("response", {})
    if response.get("status_code") not in (None, 200):
        raise ValueError(f"judge response status is {response.get('status_code')}")
    body = response.get("body", batch_row.get("body", {}))
    for output in body.get("output", []):
        if output.get("type") != "message":
            continue
        for content in output.get("content", []):
            if content.get("type") == "output_text":
                return json.loads(content["text"]), body
    if "output_text" in body:
        return json.loads(body["output_text"]), body
    raise ValueError("judge response has no Structured Outputs text")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("review_queue", type=Path)
    parser.add_argument("judge_batch_output", type=Path)
    parser.add_argument("priority_output", type=Path)
    args = parser.parse_args()

    if args.priority_output.exists() and args.priority_output.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.priority_output}")
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for candidate in rows(args.review_queue):
        source_id = str(candidate["source_id"])
        candidate_id = str(candidate["candidate_id"])
        if candidate_id in grouped[source_id]:
            raise SystemExit(f"duplicate candidate in review queue: {candidate_id}")
        grouped[source_id][candidate_id] = candidate
    if not grouped or any(len(candidates) != 3 for candidates in grouped.values()):
        raise SystemExit("every review source must contain exactly three candidates")

    judgments: list[dict] = []
    seen_sources: set[str] = set()
    seen_response_ids: set[str] = set()
    for batch_row in rows(args.judge_batch_output):
        source_id = str(batch_row.get("custom_id", ""))
        if source_id not in grouped:
            raise SystemExit(f"judge output references unknown source: {source_id}")
        if source_id in seen_sources:
            raise SystemExit(f"judge output contains duplicate source: {source_id}")
        seen_sources.add(source_id)
        try:
            payload, body = response_payload(batch_row)
        except (ValueError, json.JSONDecodeError) as error:
            raise SystemExit(f"invalid judge response for {source_id}: {error}") from error
        if payload.get("source_id") != source_id:
            raise SystemExit(f"judge Structured Output source_id mismatch: {source_id}")
        response_id = str(body.get("id", "")).strip()
        if not response_id or response_id in seen_response_ids:
            raise SystemExit(f"judge response ID is empty or duplicated: {source_id}")
        seen_response_ids.add(response_id)
        judge_model = str(body.get("model", "")).strip()
        teacher_models = {
            str(candidate.get("teacher_model", "")).strip()
            for candidate in grouped[source_id].values()
        }
        if not judge_model or judge_model in teacher_models:
            raise SystemExit(f"judge model is missing or matches the teacher: {source_id}")

        assessments = payload.get("assessments")
        if not isinstance(assessments, list) or len(assessments) != 3:
            raise SystemExit(f"judge must assess exactly three candidates: {source_id}")
        by_candidate: dict[str, dict] = {}
        for assessment in assessments:
            if not isinstance(assessment, dict) or set(assessment) != ASSESSMENT_KEYS:
                raise SystemExit(f"judge assessment keys are invalid: {source_id}")
            candidate_id = str(assessment.get("candidate_id", ""))
            if candidate_id in by_candidate or candidate_id not in grouped[source_id]:
                raise SystemExit(f"judge assessment candidate mismatch: {source_id} / {candidate_id}")
            for score_name in ("adequacy", "fluency", "terminology"):
                score = assessment.get(score_name)
                if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 4:
                    raise SystemExit(f"invalid {score_name} score: {source_id} / {candidate_id}")
            for boolean_name in ("protected_tokens_preserved", "critical_error"):
                if not isinstance(assessment.get(boolean_name), bool):
                    raise SystemExit(f"invalid {boolean_name}: {source_id} / {candidate_id}")
            error_tags = assessment.get("error_tags")
            if (
                not isinstance(error_tags, list)
                or any(not isinstance(tag, str) or tag not in ALLOWED_ERROR_TAGS for tag in error_tags)
                or len(error_tags) != len(set(error_tags))
            ):
                raise SystemExit(f"invalid judge error tags: {source_id} / {candidate_id}")
            by_candidate[candidate_id] = assessment
        if set(by_candidate) != set(grouped[source_id]):
            raise SystemExit(f"judge did not cover the exact candidate set: {source_id}")

        ordered = [by_candidate[candidate_id] for candidate_id in sorted(by_candidate)]
        critical_count = sum(bool(value["critical_error"]) for value in ordered)
        protected_failure_count = sum(not value["protected_tokens_preserved"] for value in ordered)
        minimum_adequacy = min(value["adequacy"] for value in ordered)
        minimum_fluency = min(value["fluency"] for value in ordered)
        total_quality = sum(
            value["adequacy"] + value["fluency"] + value["terminology"]
            for value in ordered
        )
        judgments.append({
            "source_id": source_id,
            "priority_status": "automated-review-order-only-not-approval",
            "judge_model": judge_model,
            "judge_response_id": response_id,
            "judge_system_fingerprint": body.get("system_fingerprint"),
            "critical_count": critical_count,
            "protected_failure_count": protected_failure_count,
            "minimum_adequacy": minimum_adequacy,
            "minimum_fluency": minimum_fluency,
            "total_quality": total_quality,
            "assessments": ordered,
        })

    missing = set(grouped) - seen_sources
    if missing:
        raise SystemExit(f"judge output is missing {len(missing)} sources; first: {next(iter(missing))}")
    judgments.sort(key=lambda row: (
        -row["critical_count"],
        -row["protected_failure_count"],
        row["minimum_adequacy"],
        row["minimum_fluency"],
        row["total_quality"],
        hashlib.sha256(row["source_id"].encode()).digest(),
    ))
    for rank, judgment in enumerate(judgments, start=1):
        judgment["priority_rank"] = rank

    args.priority_output.parent.mkdir(parents=True, exist_ok=True)
    args.priority_output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in judgments),
        encoding="utf-8",
    )
    print(json.dumps({
        "sources": len(judgments),
        "critical_sources": sum(row["critical_count"] > 0 for row in judgments),
        "output": str(args.priority_output),
        "output_sha256": hashlib.sha256(args.priority_output.read_bytes()).hexdigest(),
            "use": (
                "one-judge review ordering; two distinct judge files may feed "
                "promotion-ineligible provisional SFT consensus"
            ),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
