#!/usr/bin/env python3
"""Validate one fast-judge Batch output and create an auditable judgment file.

One judge can only prioritize review. Two files from distinct judge models may
later feed the fail-closed automated-consensus gate.
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def verify_claude_run(
    requests_path: Path,
    run_directory: Path,
    batch_output: Path,
    judge_model: str,
) -> dict:
    request_rows = rows(requests_path)
    request_models = {
        str(row.get("body", {}).get("model", "")).strip()
        for row in request_rows
    }
    request_ids = [str(row.get("custom_id", "")).strip() for row in request_rows]
    if (
        request_models != {judge_model}
        or not request_ids
        or "" in request_ids
        or len(request_ids) != len(set(request_ids))
    ):
        raise SystemExit("Claude request identity does not match collected judgments")

    run_manifest_path = run_directory / "manifest.json"
    output_manifest_path = batch_output.with_suffix(
        batch_output.suffix + ".manifest.json"
    )
    run_manifest = load_json(run_manifest_path)
    output_manifest = load_json(output_manifest_path)
    request_hash = sha256(requests_path)
    output_hash = sha256(batch_output)
    if (
        run_manifest.get("schema_version") != 2
        or run_manifest.get("judge_model") != judge_model
        or run_manifest.get("request_sha256") != request_hash
        or run_manifest.get("candidate_origin_exposed") is not False
        or run_manifest.get("reasoning_trace_stored") is not False
        or output_manifest.get("status") != "collected"
        or output_manifest.get("judge_model") != judge_model
        or output_manifest.get("output_sha256") != output_hash
        or output_manifest.get("candidate_origin_exposed") is not False
        or output_manifest.get("reasoning_trace_stored") is not False
    ):
        raise SystemExit("Claude run/output manifest does not prove the expected run")

    shards = run_manifest.get("shards")
    if not isinstance(shards, list) or len(shards) != run_manifest.get("shard_count"):
        raise SystemExit("Claude run manifest has invalid shard coverage")
    verified_ids: list[str] = []
    for shard in shards:
        index = int(shard["index"])
        stem = f"{index:05d}"
        result_path = run_directory / "shards" / f"{stem}.results.jsonl"
        metadata_path = run_directory / "shards" / f"{stem}.metadata.json"
        if not result_path.is_file() or not metadata_path.is_file():
            raise SystemExit(f"Claude shard evidence is incomplete: {index}")
        metadata = load_json(metadata_path)
        expected_ids = [str(value) for value in shard.get("custom_ids", [])]
        actual_ids = [str(row.get("custom_id", "")) for row in rows(result_path)]
        if (
            actual_ids != expected_ids
            or metadata.get("custom_ids") != expected_ids
            or metadata.get("request_sha256") != request_hash
            or metadata.get("judge_model") != judge_model
            or metadata.get("result_sha256") != sha256(result_path)
            or metadata.get("candidate_origin_exposed") is not False
            or metadata.get("reasoning_trace_stored") is not False
            or metadata.get("verified_primary_model", {}).get("canonical_model")
            != judge_model
        ):
            raise SystemExit(f"Claude shard identity proof is invalid: {index}")
        verified_ids.extend(expected_ids)
    if verified_ids != request_ids:
        raise SystemExit("Claude verified shards do not cover the exact request order")
    return {
        "actual_canonical_model_usage_verified": True,
        "request_file": {
            "path": str(requests_path),
            "sha256": request_hash,
        },
        "run_manifest": {
            "path": str(run_manifest_path),
            "sha256": sha256(run_manifest_path),
        },
        "runner_output_manifest": {
            "path": str(output_manifest_path),
            "sha256": sha256(output_manifest_path),
        },
        "verified_shards": len(shards),
    }


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
    parser.add_argument("--judge-requests", type=Path)
    parser.add_argument("--claude-run-directory", type=Path)
    args = parser.parse_args()

    if args.priority_output.exists() and args.priority_output.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.priority_output}")
    if bool(args.judge_requests) != bool(args.claude_run_directory):
        raise SystemExit(
            "judge-requests and claude-run-directory must be provided together"
        )
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for candidate in rows(args.review_queue):
        source_id = str(candidate["source_id"])
        candidate_id = str(candidate["candidate_id"])
        if candidate_id in grouped[source_id]:
            raise SystemExit(f"duplicate candidate in review queue: {candidate_id}")
        grouped[source_id][candidate_id] = candidate
    if not grouped or any(
        len(candidates) not in {3, 4} for candidates in grouped.values()
    ):
        raise SystemExit("every review source must contain three or four candidates")

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
        expected_count = len(grouped[source_id])
        if not isinstance(assessments, list) or len(assessments) != expected_count:
            raise SystemExit(
                f"judge must assess all {expected_count} candidates: {source_id}"
            )
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

    judge_models = {str(row["judge_model"]) for row in judgments}
    if len(judge_models) != 1:
        raise SystemExit("one judgment file must contain exactly one judge model")
    judge_model = next(iter(judge_models))
    identity_evidence = {
        "actual_canonical_model_usage_verified": False,
    }
    if args.judge_requests and args.claude_run_directory:
        identity_evidence = verify_claude_run(
            args.judge_requests,
            args.claude_run_directory,
            args.judge_batch_output,
            judge_model,
        )

    args.priority_output.parent.mkdir(parents=True, exist_ok=True)
    args.priority_output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in judgments),
        encoding="utf-8",
    )
    evidence_manifest = {
        "schema_version": 1,
        "judge_model": judge_model,
        "review_queue": {
            "path": str(args.review_queue),
            "sha256": sha256(args.review_queue),
        },
        "batch_output": {
            "path": str(args.judge_batch_output),
            "sha256": sha256(args.judge_batch_output),
        },
        "judgment_output": {
            "path": str(args.priority_output),
            "sha256": sha256(args.priority_output),
        },
        "candidate_origin_exposed": False,
        "reasoning_trace_stored": False,
        **identity_evidence,
    }
    evidence_path = args.priority_output.with_suffix(
        args.priority_output.suffix + ".manifest.json"
    )
    evidence_path.write_text(
        json.dumps(evidence_manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "sources": len(judgments),
        "critical_sources": sum(row["critical_count"] > 0 for row in judgments),
        "output": str(args.priority_output),
        "output_sha256": sha256(args.priority_output),
        "evidence_manifest": str(evidence_path),
        "actual_canonical_model_usage_verified": identity_evidence[
            "actual_canonical_model_usage_verified"
        ],
            "use": (
                "one-judge review ordering; two distinct judge files may feed "
                "the fail-closed automated consensus gate"
            ),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
