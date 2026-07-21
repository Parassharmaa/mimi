#!/usr/bin/env python3
"""Build one blinded, position-shuffled reference-judge Batch request file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ERROR_TAGS = [
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
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def index(values: list[dict], key: str, label: str) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for value in values:
        identifier = str(value.get(key, "")).strip()
        if not identifier or identifier in output:
            raise SystemExit(f"{label} has an empty or duplicate ID: {identifier}")
        output[identifier] = value
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", type=Path)
    parser.add_argument("generator_report", type=Path)
    parser.add_argument("model_plan", type=Path)
    parser.add_argument("prompt", type=Path)
    parser.add_argument("judge_role", choices=("reference-judge-a", "reference-judge-b"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists() and args.output.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")

    source_rows = rows(args.sources)
    sources = index(source_rows, "id", "source suite")
    generator = json.loads(args.generator_report.read_text(encoding="utf-8"))
    generated = index(generator.get("results", []), "caseID", "generator report")
    plan = json.loads(args.model_plan.read_text(encoding="utf-8"))
    prompt = args.prompt.read_text(encoding="utf-8").strip()
    judges = {value["role"]: value for value in plan.get("judges", [])}
    judge = judges.get(args.judge_role)
    if (
        not isinstance(judge, dict)
        or set(sources) != set(generated)
        or plan.get("suiteSHA256") != sha256(args.sources)
        or generator.get("sourceSuiteSHA256") != sha256(args.sources)
        or generator.get("generatorModel") == judge.get("model")
        or generator.get("generatorModelFamily") == judge.get("family")
        or judge.get("store") is not False
        or judge.get("reasoningSummaryRequested") is not False
        or not prompt
    ):
        raise SystemExit("invalid, overlapping, or unbound judge plan")
    other_judges = [value for value in judges.values() if value["role"] != args.judge_role]
    if len(other_judges) != 1 or other_judges[0]["family"] == judge["family"]:
        raise SystemExit("reference judges must use distinct model families")

    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    assessment = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "candidate_id",
            "adequacy",
            "fluency",
            "terminology",
            "protected_tokens_preserved",
            "critical_error",
            "error_tags",
            "accept_as_reference",
        ],
        "properties": {
            "candidate_id": {"type": "string"},
            "adequacy": {"type": "integer", "minimum": 0, "maximum": 4},
            "fluency": {"type": "integer", "minimum": 0, "maximum": 4},
            "terminology": {"type": "integer", "minimum": 0, "maximum": 4},
            "protected_tokens_preserved": {"type": "boolean"},
            "critical_error": {"type": "boolean"},
            "error_tags": {
                "type": "array",
                "items": {"type": "string", "enum": ERROR_TAGS},
            },
            "accept_as_reference": {"type": "boolean"},
        },
    }
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["source_id", "assessments"],
        "properties": {
            "source_id": {"type": "string"},
            "assessments": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": assessment,
            },
        },
    }

    requests: list[dict] = []
    for case_id, source in sources.items():
        candidates = generated[case_id].get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 3:
            raise SystemExit(f"generator candidate coverage mismatch: {case_id}")
        shuffled = sorted(
            candidates,
            key=lambda candidate: hashlib.sha256(
                f"{args.judge_role}\0{case_id}\0{candidate['candidateID']}".encode("utf-8")
            ).digest(),
        )
        request_input = {
            "source_id": case_id,
            "source_language": source["sourceLanguage"],
            "target_language": source["targetLanguage"],
            "domain": source["domain"],
            "source": source["source"],
            "candidates": [
                {
                    "candidate_id": candidate["candidateID"],
                    "translation": candidate["text"],
                }
                for candidate in shuffled
            ],
        }
        body = {
            "model": judge["model"],
            "store": False,
            "input": [
                {"role": "developer", "content": prompt},
                {
                    "role": "user",
                    "content": json.dumps(request_input, ensure_ascii=False, sort_keys=True),
                },
            ],
            "text": {
                "verbosity": "low",
                "format": {
                    "type": "json_schema",
                    "name": "mimi_benchmark_reference_judgments_v1",
                    "strict": True,
                    "schema": schema,
                },
            },
            "max_output_tokens": 2048,
            "metadata": {
                "pipeline": "mimi-benchmark-reference-judge-v1",
                "prompt_sha256": prompt_hash,
                "judge_role": args.judge_role,
            },
        }
        if judge.get("reasoningEffort") is not None:
            body["reasoning"] = {"effort": judge["reasoningEffort"]}
        requests.append(
            {
                "custom_id": case_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in requests),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "requests": len(requests),
                "judgeRole": args.judge_role,
                "model": judge["model"],
                "modelFamily": judge["family"],
                "promptSHA256": prompt_hash,
                "output": str(args.output),
                "outputSHA256": sha256(args.output),
                "candidateOrder": "deterministically shuffled independently per judge and case",
                "reasoningSummaryRequested": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
