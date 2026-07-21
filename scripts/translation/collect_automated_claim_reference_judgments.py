#!/usr/bin/env python3
"""Convert one complete blinded judge Batch response into an auditable report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from collect_automated_claim_reference_candidates import (
    canonical_sha256,
    index,
    output_text,
    rows,
    sha256,
    text_sha256,
    visible_reasoning_trace,
)


ERROR_TAGS = {
    "meaning-reversal",
    "negation",
    "number-or-date",
    "named-entity",
    "omission",
    "addition",
    "placeholder-url-or-markup",
    "code-switching",
    "register",
    "terminology",
    "disfluency",
}
ASSESSMENT_KEYS = {
    "candidate_id",
    "adequacy",
    "fluency",
    "terminology",
    "protected_tokens_preserved",
    "critical_error",
    "error_tags",
    "accept_as_reference",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", type=Path)
    parser.add_argument("generator_report", type=Path)
    parser.add_argument("judge_requests", type=Path)
    parser.add_argument("model_plan", type=Path)
    parser.add_argument("judge_role", choices=("reference-judge-a", "reference-judge-b"))
    parser.add_argument("batch_output", type=Path)
    parser.add_argument("judge_report", type=Path)
    args = parser.parse_args()
    if args.judge_report.exists() and args.judge_report.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.judge_report}")

    source_rows = rows(args.sources)
    sources = index(source_rows, "id", "source suite")
    generator = json.loads(args.generator_report.read_text(encoding="utf-8"))
    generated = index(generator.get("results", []), "caseID", "generator report")
    request_rows = rows(args.judge_requests)
    requests = index(request_rows, "custom_id", "judge request file")
    response_rows = rows(args.batch_output)
    responses = index(response_rows, "custom_id", "judge Batch output")
    if set(sources) != set(generated) or set(sources) != set(requests) or set(sources) != set(responses):
        raise SystemExit("judge inputs and outputs do not have exact source-suite coverage")

    plan = json.loads(args.model_plan.read_text(encoding="utf-8"))
    judges = {value["role"]: value for value in plan.get("judges", [])}
    judge = judges.get(args.judge_role)
    if (
        not isinstance(judge, dict)
        or plan.get("suiteSHA256") != sha256(args.sources)
        or judge.get("store") is not False
        or judge.get("reasoningSummaryRequested") is not False
        or judge.get("model") == generator.get("generatorModel")
        or judge.get("family") == generator.get("generatorModelFamily")
    ):
        raise SystemExit("invalid or overlapping judge plan")
    prompt_hashes = {
        str(value.get("body", {}).get("metadata", {}).get("prompt_sha256", ""))
        for value in request_rows
    }
    if len(prompt_hashes) != 1 or any(len(value) != 64 for value in prompt_hashes):
        raise SystemExit("judge requests do not contain one pinned prompt hash")
    prompt_hash = next(iter(prompt_hashes))

    results: list[dict] = []
    response_ids: set[str] = set()
    system_fingerprints: set[str] = set()
    for case_id, source in sources.items():
        candidates = generated[case_id].get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 3:
            raise SystemExit(f"generator candidate coverage mismatch: {case_id}")
        candidate_by_id = {str(value["candidateID"]): value for value in candidates}
        request = requests[case_id]
        try:
            request_input = json.loads(request["body"]["input"][1]["content"])
            developer_prompt = request["body"]["input"][0]["content"]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise SystemExit(f"invalid blinded judge request: {case_id}") from error
        requested_candidates = request_input.get("candidates")
        if (
            request_input.get("source_id") != case_id
            or request_input.get("source") != source.get("source")
            or not isinstance(requested_candidates, list)
            or {
                (str(value.get("candidate_id")), str(value.get("translation")))
                for value in requested_candidates
            }
            != {(identifier, str(value["text"])) for identifier, value in candidate_by_id.items()}
            or "generator" in request["body"]["input"][1]["content"].casefold()
            or request.get("method") != "POST"
            or request.get("url") != "/v1/responses"
            or request["body"].get("model") != judge.get("model")
            or request["body"].get("store") is not False
            or request["body"].get("metadata", {}).get("pipeline")
            != "mimi-benchmark-reference-judge-v1"
            or request["body"].get("metadata", {}).get("judge_role") != args.judge_role
            or hashlib.sha256(str(developer_prompt).encode("utf-8")).hexdigest() != prompt_hash
        ):
            raise SystemExit(f"judge request is unblinded or candidate-misaligned: {case_id}")
        expected_reasoning = (
            {"effort": judge.get("reasoningEffort")}
            if judge.get("reasoningEffort") is not None
            else None
        )
        if request["body"].get("reasoning") != expected_reasoning:
            raise SystemExit(f"judge request reasoning policy mismatch: {case_id}")

        batch_row = responses[case_id]
        if batch_row.get("error") not in (None, {}):
            raise SystemExit(f"judge Batch response has an error: {case_id}")
        response = batch_row.get("response")
        if not isinstance(response, dict) or response.get("status_code") != 200:
            raise SystemExit(f"judge Batch response is not HTTP 200: {case_id}")
        body = response.get("body")
        if not isinstance(body, dict) or body.get("status") not in (None, "completed"):
            raise SystemExit(f"incomplete judge Responses API body: {case_id}")
        if visible_reasoning_trace(body):
            raise SystemExit(f"visible or encrypted judge reasoning trace found: {case_id}")
        response_id = str(body.get("id", "")).strip()
        if not response_id or response_id in response_ids:
            raise SystemExit(f"empty or duplicate judge response ID: {case_id}")
        response_ids.add(response_id)
        if body.get("model") != judge.get("model"):
            raise SystemExit(f"judge model revision mismatch: {case_id}")
        fingerprint = str(body.get("system_fingerprint") or "").strip()
        if fingerprint:
            system_fingerprints.add(fingerprint)
        try:
            payload = json.loads(output_text(body))
        except (ValueError, json.JSONDecodeError) as error:
            raise SystemExit(f"invalid judge Structured Output: {case_id}: {error}") from error
        if not isinstance(payload, dict) or set(payload) != {"source_id", "assessments"}:
            raise SystemExit(f"unexpected judge response schema: {case_id}")
        assessments = payload.get("assessments")
        if payload.get("source_id") != case_id or not isinstance(assessments, list) or len(assessments) != 3:
            raise SystemExit(f"invalid judge assessment coverage: {case_id}")
        by_candidate: dict[str, dict] = {}
        for assessment in assessments:
            if not isinstance(assessment, dict) or set(assessment) != ASSESSMENT_KEYS:
                raise SystemExit(f"invalid judge assessment schema: {case_id}")
            candidate_id = str(assessment.get("candidate_id", ""))
            if candidate_id in by_candidate or candidate_id not in candidate_by_id:
                raise SystemExit(f"unknown or duplicate assessed candidate: {case_id}/{candidate_id}")
            scores = [assessment.get(value) for value in ("adequacy", "fluency", "terminology")]
            if any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 4 for value in scores):
                raise SystemExit(f"invalid judge score: {case_id}/{candidate_id}")
            if any(
                not isinstance(assessment.get(value), bool)
                for value in ("protected_tokens_preserved", "critical_error", "accept_as_reference")
            ):
                raise SystemExit(f"invalid judge boolean: {case_id}/{candidate_id}")
            tags = assessment.get("error_tags")
            if (
                not isinstance(tags, list)
                or len(tags) != len(set(tags))
                or any(value not in ERROR_TAGS for value in tags)
            ):
                raise SystemExit(f"invalid judge error tags: {case_id}/{candidate_id}")
            should_accept = (
                scores == [4, 4, 4]
                and assessment["protected_tokens_preserved"] is True
                and assessment["critical_error"] is False
                and tags == []
            )
            if assessment["accept_as_reference"] is not should_accept:
                raise SystemExit(f"judge acceptance contradicts its scores: {case_id}/{candidate_id}")
            candidate = candidate_by_id[candidate_id]
            by_candidate[candidate_id] = {
                "candidateID": candidate_id,
                "referenceSHA256": text_sha256(str(candidate["text"])),
                "adequacy": assessment["adequacy"],
                "fluency": assessment["fluency"],
                "terminology": assessment["terminology"],
                "criticalError": assessment["critical_error"],
                "protectedTokensPreserved": assessment["protected_tokens_preserved"],
                "errorTags": tags,
                "acceptAsReference": assessment["accept_as_reference"],
            }
        if set(by_candidate) != set(candidate_by_id):
            raise SystemExit(f"judge did not assess the exact candidate set: {case_id}")
        results.append(
            {
                "caseID": case_id,
                "sourceSHA256": text_sha256(str(source["source"])),
                "requestSHA256": canonical_sha256(request),
                "responseSHA256": canonical_sha256(batch_row),
                "responseID": response_id,
                "assessments": [by_candidate[value["candidateID"]] for value in candidates],
            }
        )

    report = {
        "schemaVersion": 1,
        "purpose": "benchmark-reference-review",
        "judgeRole": args.judge_role,
        "judgeModel": judge["model"],
        "judgeModelFamily": judge["family"],
        "judgeRevision": judge["revision"],
        "promptSHA256": prompt_hash,
        "reasoningTracesStored": False,
        "store": False,
        "sourceSuiteSHA256": sha256(args.sources),
        "generatorReportSHA256": sha256(args.generator_report),
        "requestFileSHA256": sha256(args.judge_requests),
        "rawBatchOutputSHA256": sha256(args.batch_output),
        "systemFingerprints": sorted(system_fingerprints),
        "results": results,
    }
    args.judge_report.parent.mkdir(parents=True, exist_ok=True)
    args.judge_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "cases": len(results),
                "acceptedAssessments": sum(
                    value["acceptAsReference"]
                    for result in results
                    for value in result["assessments"]
                ),
                "judgeModel": judge["model"],
                "judgeModelFamily": judge["family"],
                "judgeReport": str(args.judge_report),
                "judgeReportSHA256": sha256(args.judge_report),
                "reasoningTracesStored": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
