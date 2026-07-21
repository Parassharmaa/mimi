#!/usr/bin/env python3
"""Prepare a blinded, optional fast-model screen for teacher candidates.

One invocation never approves a training row. It prioritizes human review, or
serves as one side of a two-distinct-model provisional SFT consensus, and must
use a model distinct from the candidate teacher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


SCHEMA = {
    "type": "object",
    "properties": {
        "source_id": {"type": "string"},
        "assessments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string"},
                    "adequacy": {"type": "integer", "minimum": 0, "maximum": 4},
                    "fluency": {"type": "integer", "minimum": 0, "maximum": 4},
                    "terminology": {"type": "integer", "minimum": 0, "maximum": 4},
                    "protected_tokens_preserved": {"type": "boolean"},
                    "critical_error": {"type": "boolean"},
                    "error_tags": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "meaning-reversal",
                                "negation",
                                "number-or-date",
                                "named-entity",
                                "omission",
                                "addition",
                                "register",
                                "terminology",
                                "disfluency",
                            ],
                        },
                    },
                },
                "required": [
                    "candidate_id",
                    "adequacy",
                    "fluency",
                    "terminology",
                    "protected_tokens_preserved",
                    "critical_error",
                    "error_tags",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["source_id", "assessments"],
    "additionalProperties": False,
}

DEVELOPER_PROMPT = """Blindly assess candidate translations against the source.
Score adequacy, fluency, and terminology from 0 to 4 and identify only the enumerated error tags.
Treat meaning reversal, changed negation, wrong numbers/dates, and wrong named entities as critical.
Do not rank by verbosity and do not infer which system produced a candidate.
Return compact structured judgments only; do not provide chain-of-thought.
One automated screen is never training-data approval. Two distinct judge models
may be combined only by the promotion-ineligible provisional SFT consensus gate."""


def rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("review_queue", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--model",
        required=True,
        help="Fast/instant judge model ID available to the caller's OpenAI project",
    )
    parser.add_argument("--reasoning-effort", default="low")
    args = parser.parse_args()

    if args.output.exists() and args.output.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows(args.review_queue):
        grouped[str(row["source_id"])].append(row)

    prompt_hash = hashlib.sha256(DEVELOPER_PROMPT.encode()).hexdigest()
    output_lines = []
    for source_id, candidates in sorted(grouped.items()):
        teacher_models = {str(row.get("teacher_model")) for row in candidates}
        if args.model in teacher_models:
            raise SystemExit(
                f"judge model must differ from the teacher for source {source_id}"
            )
        first = candidates[0]
        request_input = {
            "source_id": source_id,
            "source_language": first["source_language"],
            "target_language": first["target_language"],
            "domain": first["domain"],
            "source": first["source"],
            "candidates": [
                {
                    "candidate_id": row["candidate_id"],
                    "translation": row["translation"],
                }
                for row in sorted(candidates, key=lambda row: row["candidate_id"])
            ],
        }
        body = {
            "model": args.model,
            "store": False,
            "reasoning": {
                "effort": args.reasoning_effort,
            },
            "input": [
                {"role": "developer", "content": DEVELOPER_PROMPT},
                {"role": "user", "content": json.dumps(request_input, ensure_ascii=False)},
            ],
            "text": {
                "verbosity": "low",
                "format": {
                    "type": "json_schema",
                    "name": "mimi_translation_judgments",
                    "strict": True,
                    "schema": SCHEMA,
                },
            },
            "max_output_tokens": 700,
            "metadata": {
                "pipeline": "mimi-translation-judge-v1",
                "prompt_sha256": prompt_hash,
            },
        }
        output_lines.append(
            json.dumps(
                {
                    "custom_id": source_id,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": body,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "requests": len(output_lines),
                "model": args.model,
                "prompt_sha256": prompt_hash,
                "submitted": False,
            }
        )
    )


if __name__ == "__main__":
    main()
