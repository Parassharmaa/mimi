#!/usr/bin/env python3
"""Admit a preregistered canonical target certified by two judge families."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from approve_automated_consensus import judgment_map, rows, write_jsonl


MINIMUM_ADEQUACY = 4
MINIMUM_FLUENCY = 3
MINIMUM_TERMINOLOGY = 3
REQUIRED_ORIGINS = {
    "teacher",
    "licensed-reference",
    "current-mimi-baseline",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assessment_failures(assessment: dict) -> list[str]:
    failures: list[str] = []
    if assessment.get("critical_error") is not False:
        failures.append("critical-error")
    if assessment.get("protected_tokens_preserved") is not True:
        failures.append("protected-token-failure")
    if assessment.get("error_tags") != []:
        failures.append("error-tags")
    if assessment.get("adequacy", -1) < MINIMUM_ADEQUACY:
        failures.append("adequacy")
    if assessment.get("fluency", -1) < MINIMUM_FLUENCY:
        failures.append("fluency")
    if assessment.get("terminology", -1) < MINIMUM_TERMINOLOGY:
        failures.append("terminology")
    return failures


def verify_judgment_evidence(
    contract: dict,
    judgment_path: Path,
    judge_model: str,
    review_queue: Path,
) -> None:
    evidence_path = judgment_path.with_suffix(
        judgment_path.suffix + ".manifest.json"
    )
    if not evidence_path.is_file():
        raise SystemExit(f"missing canonical-model evidence: {evidence_path}")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    request_contract = contract.get("judgeRequests", {}).get(judge_model, {})
    if (
        evidence.get("schema_version") != 1
        or evidence.get("judge_model") != judge_model
        or evidence.get("actual_canonical_model_usage_verified") is not True
        or evidence.get("candidate_origin_exposed") is not False
        or evidence.get("reasoning_trace_stored") is not False
        or evidence.get("judgment_output", {}).get("sha256")
        != sha256(judgment_path)
        or evidence.get("review_queue", {}).get("sha256") != sha256(review_queue)
        or evidence.get("request_file", {}).get("sha256")
        != request_contract.get("sha256")
    ):
        raise SystemExit(
            f"judgment does not carry the preregistered model proof: {judge_model}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pilot_contract", type=Path)
    parser.add_argument("queue_manifest", type=Path)
    parser.add_argument("review_queue", type=Path)
    parser.add_argument("judgments_a", type=Path)
    parser.add_argument("judgments_b", type=Path)
    parser.add_argument("approved", type=Path)
    parser.add_argument("rejected", type=Path)
    args = parser.parse_args()
    for output in (args.approved, args.rejected):
        if output.exists() and output.stat().st_size:
            raise SystemExit(f"refusing to overwrite non-empty output: {output}")

    contract = json.loads(args.pilot_contract.read_text(encoding="utf-8"))
    gate = contract.get("goGateBeforeScaling", {})
    judge_policy = contract.get("judgePolicy", {})
    required_judges = judge_policy.get("requiredJudgeModelIds")
    if (
        gate.get("acceptancePolicy") != "dual-absolute-canonical"
        or gate.get("requireBothJudgesCertifyCanonicalAbsoluteThresholds") is not True
        or gate.get("requireZeroDeterministicCriticalFailures") is not True
        or gate.get("trainingAuthorizedByPilotAlone") is not False
        or not isinstance(required_judges, list)
        or len(required_judges) != 2
        or len(set(required_judges)) != 2
        or judge_policy.get("sameBlindedBatchRequired") is not True
        or judge_policy.get("actualCanonicalModelUsageRequiredPerShard") is not True
        or judge_policy.get("fallbackModelAllowed") is not False
    ):
        raise SystemExit("pilot did not preregister the canonical absolute gate")
    queue_manifest = json.loads(args.queue_manifest.read_text(encoding="utf-8"))
    if (
        queue_manifest.get("output", {}).get("sha256") != sha256(args.review_queue)
        or queue_manifest.get("inputs", {}).get("pilotSeeds", {}).get("sha256")
        != contract.get("outputs", {}).get("pilotSeeds", {}).get("sha256")
        or queue_manifest.get("candidateOriginExposedToJudges") is not False
    ):
        raise SystemExit("review queue is not bound to the preregistered pilot")

    queue: dict[str, dict[str, dict]] = defaultdict(dict)
    canonical_ids: dict[str, str] = {}
    for candidate in rows(args.review_queue):
        source_id = str(candidate.get("source_id", ""))
        candidate_id = str(candidate.get("candidate_id", ""))
        if not source_id or not candidate_id or candidate_id in queue[source_id]:
            raise SystemExit(f"invalid review candidate: {source_id}/{candidate_id}")
        queue[source_id][candidate_id] = candidate
        if candidate.get("candidate_origin") == "teacher":
            if source_id in canonical_ids:
                raise SystemExit(f"multiple canonical teacher candidates: {source_id}")
            canonical_ids[source_id] = candidate_id
    if not queue:
        raise SystemExit("canonical review queue is empty")
    for source_id, candidates in queue.items():
        origins = {row.get("candidate_origin") for row in candidates.values()}
        if len(candidates) != 3 or origins != REQUIRED_ORIGINS:
            raise SystemExit(f"canonical source has invalid candidate origins: {source_id}")
    if set(canonical_ids) != set(queue):
        raise SystemExit("canonical teacher candidate coverage is incomplete")

    judge_a, judgments_a = judgment_map(args.judgments_a, queue)
    judge_b, judgments_b = judgment_map(args.judgments_b, queue)
    if judge_a == judge_b:
        raise SystemExit("canonical consensus requires two distinct judge models")
    if {judge_a, judge_b} != set(required_judges):
        raise SystemExit("canonical judgments do not match the preregistered judge models")
    verify_judgment_evidence(contract, args.judgments_a, judge_a, args.review_queue)
    verify_judgment_evidence(contract, args.judgments_b, judge_b, args.review_queue)
    teacher_models = {
        str(candidate.get("teacher_model", "")).strip()
        for candidates in queue.values()
        for candidate in candidates.values()
    }
    if judge_a in teacher_models or judge_b in teacher_models:
        raise SystemExit("canonical judge model must differ from the teacher")

    approved: list[dict] = []
    rejected: list[dict] = []
    for source_id in sorted(queue):
        candidate_id = canonical_ids[source_id]
        assessment_a = judgments_a[source_id]["assessments"][candidate_id]
        assessment_b = judgments_b[source_id]["assessments"][candidate_id]
        failures_a = assessment_failures(assessment_a)
        failures_b = assessment_failures(assessment_b)
        if not failures_a and not failures_b:
            approved.append({
                **queue[source_id][candidate_id],
                "review_status": "two-judge-reference-anchored-canonical-absolute",
                "reviewer_ids": [],
                "judge_model_ids": sorted([judge_a, judge_b]),
                "automated_judgments": [
                    judgments_a[source_id],
                    judgments_b[source_id],
                ],
                "approved_alternative": None,
                "promotion_eligible": True,
                "canonical_absolute_consensus_policy": {
                    "preregistered": True,
                    "minimum_adequacy": MINIMUM_ADEQUACY,
                    "minimum_fluency": MINIMUM_FLUENCY,
                    "minimum_terminology": MINIMUM_TERMINOLOGY,
                    "require_no_error_tags": True,
                    "require_no_critical_error": True,
                    "require_protected_tokens_preserved": True,
                    "require_two_distinct_judge_models": True,
                    "canonical_candidate_selected_before_judging": True,
                    "licensed_reference_anchor_present": True,
                    "current_mimi_baseline_present": True,
                    "unique_best_not_required": True,
                    "training_authorized_by_pilot_alone": False,
                },
            })
        else:
            rejected.append({
                "source_id": source_id,
                "status": "canonical-absolute-consensus-rejected",
                "candidate_id": candidate_id,
                "judge_models": sorted([judge_a, judge_b]),
                "judge_a_failures": failures_a,
                "judge_b_failures": failures_b,
            })

    write_jsonl(args.approved, approved)
    write_jsonl(args.rejected, rejected)
    print(json.dumps({
        "sources": len(queue),
        "approved": len(approved),
        "rejected": len(rejected),
        "approved_directions": dict(sorted(
            {
                direction: sum(
                    f"{row['source_language']}->{row['target_language']}" == direction
                    for row in approved
                )
                for direction in {
                    f"{row['source_language']}->{row['target_language']}"
                    for row in approved
                }
            }.items()
        )),
        "judge_models": sorted([judge_a, judge_b]),
        "training_authorized": False,
        "use": "fresh-pilot scale decision only; not held-out promotion evidence",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
