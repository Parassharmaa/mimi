#!/usr/bin/env python3
"""Prepare one blinded phase-1 candidate/incumbent quality-judge Batch file."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


SCHEMA = {
    "type": "object",
    "properties": {
        "case_id": {"type": "string"},
        "output_a": {
            "type": "object",
            "properties": {
                "adequacy": {"type": "integer", "minimum": 0, "maximum": 4},
                "fluency": {"type": "integer", "minimum": 0, "maximum": 4},
                "terminology": {"type": "integer", "minimum": 0, "maximum": 2},
                "critical_error": {"type": "boolean"},
                "error_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "adequacy",
                "fluency",
                "terminology",
                "critical_error",
                "error_tags",
            ],
            "additionalProperties": False,
        },
        "output_b": {
            "type": "object",
            "properties": {
                "adequacy": {"type": "integer", "minimum": 0, "maximum": 4},
                "fluency": {"type": "integer", "minimum": 0, "maximum": 4},
                "terminology": {"type": "integer", "minimum": 0, "maximum": 2},
                "critical_error": {"type": "boolean"},
                "error_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "adequacy",
                "fluency",
                "terminology",
                "critical_error",
                "error_tags",
            ],
            "additionalProperties": False,
        },
    },
    "required": ["case_id", "output_a", "output_b"],
    "additionalProperties": False,
}
DEVELOPER_PROMPT = """Blindly compare two translations against the source.
Score each output for adequacy 0-4, fluency 0-4, and terminology/register 0-2.
Mark critical_error for meaning reversal, wrong negation, wrong number/date/unit,
wrong named entity, severe omission/addition, or broken placeholder/URL/markup.
Use concise normalized error tags. Do not infer system identity, prefer an output
for verbosity, or provide chain-of-thought. Return only the structured verdict."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("candidate_report", type=Path)
    parser.add_argument("baseline_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-family", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--judge-role", choices=("phase1-judge-a", "phase1-judge-b"), required=True)
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--seed", default="mimi-gpt56-phase1-judge-v1")
    args = parser.parse_args()
    if args.output.exists() and args.output.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    if args.model != args.model_revision:
        raise SystemExit("Batch model must be the exact pinned model revision")

    suite = indexed(rows(args.suite), "id", "suite")
    candidate_report, baseline_report = load(args.candidate_report), load(args.baseline_report)
    candidate = indexed(candidate_report.get("results", []), "caseID", "candidate")
    baseline = indexed(baseline_report.get("results", []), "caseID", "baseline")
    if set(suite) != set(candidate) or set(suite) != set(baseline):
        raise SystemExit("suite and reports must have exact case coverage")
    if candidate_report.get("engine") == baseline_report.get("engine"):
        raise SystemExit("candidate and baseline identify the same engine")

    prompt_hash = hashlib.sha256(DEVELOPER_PROMPT.encode()).hexdigest()
    lines = []
    for case_id in sorted(suite):
        source = suite[case_id]
        for report_name, row in (("candidate", candidate[case_id]), ("baseline", baseline[case_id])):
            for field in ("sourceLanguage", "targetLanguage", "domain", "source"):
                if row.get(field) != source.get(field):
                    raise SystemExit(f"{report_name} disagrees with suite {field}: {case_id}")
        pair = [
            str(candidate[case_id].get("hypothesis", "")).strip(),
            str(baseline[case_id].get("hypothesis", "")).strip(),
        ]
        if not all(pair) or pair[0] == pair[1]:
            raise SystemExit(f"judge requires two distinct non-empty outputs: {case_id}")
        rng = random.Random(f"{args.seed}\0{args.judge_role}\0{case_id}")
        if rng.randrange(2):
            pair.reverse()
        request_input = {
            "case_id": case_id,
            "source_language": source["sourceLanguage"],
            "target_language": source["targetLanguage"],
            "domain": source["domain"],
            "source": source["source"],
            "output_a": pair[0],
            "output_b": pair[1],
        }
        body = {
            "model": args.model,
            "store": False,
            "reasoning": {"effort": args.reasoning_effort},
            "input": [
                {"role": "developer", "content": DEVELOPER_PROMPT},
                {"role": "user", "content": json.dumps(request_input, ensure_ascii=False)},
            ],
            "text": {
                "verbosity": "low",
                "format": {
                    "type": "json_schema",
                    "name": "mimi_phase1_translation_judgment",
                    "strict": True,
                    "schema": SCHEMA,
                },
            },
            "max_output_tokens": 350,
            "metadata": {
                "pipeline": "mimi-gpt56-phase1-judge-v1",
                "judge_role": args.judge_role,
                "judge_model_family": args.model_family,
                "judge_revision": args.model_revision,
                "prompt_sha256": prompt_hash,
            },
        }
        lines.append(
            json.dumps(
                {
                    "custom_id": f"{args.judge_role}:{hashlib.sha256(case_id.encode()).hexdigest()[:24]}",
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": body,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "requests": len(lines),
                "judgeRole": args.judge_role,
                "model": args.model,
                "modelFamily": args.model_family,
                "promptSHA256": prompt_hash,
                "outputSHA256": sha256(args.output),
                "submitted": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
