#!/usr/bin/env python3
"""Build the sealed GPT-5.6 reference-candidate Batch request file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", type=Path)
    parser.add_argument("model_plan", type=Path)
    parser.add_argument("prompt", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists() and args.output.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    plan = json.loads(args.model_plan.read_text(encoding="utf-8"))
    sources = rows(args.sources)
    prompt = args.prompt.read_text(encoding="utf-8").strip()
    generator = plan.get("generator", {})
    if (
        plan.get("schemaVersion") != 1
        or plan.get("suiteSHA256") != sha256(args.sources)
        or generator.get("store") is not False
        or generator.get("reasoningSummaryRequested") is not False
        or generator.get("minimumCandidatesPerCase") != 3
    ):
        raise SystemExit("invalid or unbound reference model plan")
    if len(sources) != 800 or not prompt:
        raise SystemExit("expected the exact 800-source suite and a non-empty prompt")
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["source_id", "translations"],
        "properties": {
            "source_id": {"type": "string"},
            "translations": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {"type": "string", "minLength": 1},
            },
        },
    }
    requests: list[dict] = []
    seen: set[str] = set()
    for row in sources:
        identifier = str(row.get("id", "")).strip()
        source = str(row.get("source", "")).strip()
        if (
            not identifier
            or identifier in seen
            or not source
            or row.get("references") != []
            or row.get("claimEligible") is not False
        ):
            raise SystemExit(f"invalid source-only benchmark row: {identifier}")
        seen.add(identifier)
        user = {
            "source_id": identifier,
            "source_language": row["sourceLanguage"],
            "target_language": row["targetLanguage"],
            "domain": row["domain"],
            "source": source,
        }
        requests.append(
            {
                "custom_id": identifier,
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": generator["model"],
                    "store": False,
                    "reasoning": {"effort": generator["reasoningEffort"]},
                    "input": [
                        {"role": "developer", "content": prompt},
                        {
                            "role": "user",
                            "content": json.dumps(user, ensure_ascii=False, sort_keys=True),
                        },
                    ],
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "mimi_benchmark_reference_candidates_v1",
                            "strict": True,
                            "schema": schema,
                        }
                    },
                    "max_output_tokens": 1024,
                    "metadata": {
                        "pipeline": "mimi-benchmark-reference-generator-v1",
                        "prompt_sha256": prompt_hash,
                    },
                },
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
                "model": generator["model"],
                "modelFamily": generator["family"],
                "promptSHA256": prompt_hash,
                "output": str(args.output),
                "outputSHA256": sha256(args.output),
                "store": False,
                "reasoningSummaryRequested": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
