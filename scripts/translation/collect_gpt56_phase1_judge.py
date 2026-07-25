#!/usr/bin/env python3
"""Collect one complete phase-1 blinded judge Batch into evaluator evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from collect_automated_claim_reference_candidates import output_text, visible_reasoning_trace


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def indexed(values: list[dict], field: str, label: str) -> dict[str, dict]:
    output = {str(row.get(field, "")): row for row in values}
    if not output or "" in output or len(output) != len(values):
        raise SystemExit(f"{label} has empty or duplicate IDs")
    return output


def normalized_scores(value: object, label: str) -> dict:
    if not isinstance(value, dict) or set(value) != {
        "adequacy",
        "fluency",
        "terminology",
        "critical_error",
        "error_tags",
    }:
        raise SystemExit(f"invalid judge score schema: {label}")
    for name, maximum in (("adequacy", 4), ("fluency", 4), ("terminology", 2)):
        score = value[name]
        if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= maximum:
            raise SystemExit(f"invalid judge score: {label}/{name}")
    if not isinstance(value["critical_error"], bool) or not isinstance(value["error_tags"], list):
        raise SystemExit(f"invalid judge critical fields: {label}")
    if len(value["error_tags"]) != len(set(value["error_tags"])) or not all(
        isinstance(tag, str) and tag.strip() for tag in value["error_tags"]
    ):
        raise SystemExit(f"invalid judge error tags: {label}")
    return {
        "adequacy": value["adequacy"],
        "fluency": value["fluency"],
        "terminology": value["terminology"],
        "criticalError": value["critical_error"],
        "errorTags": value["error_tags"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("candidate_report", type=Path)
    parser.add_argument("baseline_report", type=Path)
    parser.add_argument("requests", type=Path)
    parser.add_argument("batch_output", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists() and args.output.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")

    suite_rows = rows(args.suite)
    suite = indexed(suite_rows, "id", "suite")
    candidate_report, baseline_report = load(args.candidate_report), load(args.baseline_report)
    candidate = indexed(candidate_report.get("results", []), "caseID", "candidate")
    baseline = indexed(baseline_report.get("results", []), "caseID", "baseline")
    request_rows, response_rows = rows(args.requests), rows(args.batch_output)
    requests = indexed(request_rows, "custom_id", "requests")
    responses = indexed(response_rows, "custom_id", "responses")
    if len(requests) != len(suite) or set(requests) != set(responses):
        raise SystemExit("request/response coverage is incomplete")

    metadata_values = {
        json.dumps(request["body"]["metadata"], sort_keys=True)
        for request in request_rows
    }
    model_values = {str(request["body"].get("model", "")) for request in request_rows}
    prompt_values = {
        str(request["body"]["metadata"].get("prompt_sha256", ""))
        for request in request_rows
    }
    if len(metadata_values) != 1 or len(model_values) != 1 or len(prompt_values) != 1:
        raise SystemExit("judge request identity is not uniform")
    metadata = json.loads(next(iter(metadata_values)))
    model = next(iter(model_values))
    prompt_hash = next(iter(prompt_values))
    if not model or not metadata.get("judge_model_family") or not metadata.get("judge_revision"):
        raise SystemExit("judge request identity is incomplete")

    results = []
    seen_cases: set[str] = set()
    response_ids: set[str] = set()
    for custom_id in sorted(requests):
        request, response = requests[custom_id], responses[custom_id]
        try:
            request_input = json.loads(request["body"]["input"][1]["content"])
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise SystemExit(f"invalid blinded request: {custom_id}") from error
        case_id = str(request_input.get("case_id", ""))
        if case_id not in suite or case_id in seen_cases:
            raise SystemExit(f"unknown or duplicate judge case: {case_id}")
        seen_cases.add(case_id)
        candidate_text = str(candidate[case_id]["hypothesis"])
        baseline_text = str(baseline[case_id]["hypothesis"])
        visible = request["body"]["input"][1]["content"].casefold()
        if (
            request.get("method") != "POST"
            or request.get("url") != "/v1/responses"
            or request["body"].get("store") is not False
            or request["body"].get("metadata", {}).get("pipeline")
            != "mimi-gpt56-phase1-judge-v1"
            or "candidate" in visible
            or "baseline" in visible
            or {request_input.get("output_a"), request_input.get("output_b")}
            != {candidate_text, baseline_text}
        ):
            raise SystemExit(f"judge request is unblinded or misaligned: {case_id}")
        response_wrapper = response.get("response")
        if (
            response.get("error") not in (None, {})
            or not isinstance(response_wrapper, dict)
            or response_wrapper.get("status_code") != 200
        ):
            raise SystemExit(f"judge Batch response failed: {case_id}")
        body = response_wrapper.get("body")
        if not isinstance(body, dict) or visible_reasoning_trace(body):
            raise SystemExit(f"judge response exposes reasoning or is malformed: {case_id}")
        response_id = str(body.get("id", "")).strip()
        if not response_id or response_id in response_ids or body.get("model") != model:
            raise SystemExit(f"judge response identity mismatch: {case_id}")
        response_ids.add(response_id)
        try:
            payload = json.loads(output_text(body))
        except (ValueError, json.JSONDecodeError) as error:
            raise SystemExit(f"invalid judge Structured Output: {case_id}") from error
        if not isinstance(payload, dict) or set(payload) != {"case_id", "output_a", "output_b"}:
            raise SystemExit(f"invalid judge response schema: {case_id}")
        if payload.get("case_id") != case_id:
            raise SystemExit(f"judge response case mismatch: {case_id}")
        decoded = {
            str(request_input["output_a"]): normalized_scores(payload["output_a"], f"{case_id}/A"),
            str(request_input["output_b"]): normalized_scores(payload["output_b"], f"{case_id}/B"),
        }
        results.append(
            {
                "caseID": case_id,
                "candidateHypothesisSHA256": text_sha256(candidate_text),
                "baselineHypothesisSHA256": text_sha256(baseline_text),
                "candidate": decoded[candidate_text],
                "baseline": decoded[baseline_text],
                "requestSHA256": hashlib.sha256(
                    json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "responseSHA256": hashlib.sha256(
                    json.dumps(response, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "responseID": response_id,
            }
        )
    if seen_cases != set(suite) or set(candidate) != set(suite) or set(baseline) != set(suite):
        raise SystemExit("judge report does not cover the exact suite/reports")

    report = {
        "schemaVersion": 1,
        "purpose": "blinded-automated-quality-and-critical-comparison",
        "suiteSHA256": sha256(args.suite),
        "candidateReportSHA256": sha256(args.candidate_report),
        "baselineReportSHA256": sha256(args.baseline_report),
        "requestFileSHA256": sha256(args.requests),
        "batchOutputSHA256": sha256(args.batch_output),
        "judgeModel": model,
        "judgeModelFamily": metadata["judge_model_family"],
        "judgeRevision": metadata["judge_revision"],
        "judgeRole": metadata["judge_role"],
        "promptSHA256": prompt_hash,
        "reasoningTracesStored": False,
        "blinded": True,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "cases": len(results),
                "judgeModel": model,
                "judgeModelFamily": report["judgeModelFamily"],
                "outputSHA256": sha256(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
