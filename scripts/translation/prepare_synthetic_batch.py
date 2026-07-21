#!/usr/bin/env python3
"""Prepare, but never submit, a GPT-5.6 Batch API candidate-generation file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MODEL = "gpt-5.6-sol"
SCHEMA = {
    "type": "object",
    "properties": {
        "source_id": {"type": "string"},
        "translation_brief": {
            "type": "object",
            "properties": {
                "register": {
                    "type": "string",
                    "enum": ["casual", "neutral", "polite", "technical"],
                },
                "terms": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "target": {"type": "string"},
                        },
                        "required": ["source", "target"],
                        "additionalProperties": False,
                    },
                },
                "preserve": {"type": "array", "items": {"type": "string"}},
                "ambiguities": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["register", "terms", "preserve", "ambiguities"],
            "additionalProperties": False,
        },
        "candidates": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "translation": {"type": "string"},
                    "style": {
                        "type": "string",
                        "enum": ["natural-spoken", "concise-caption", "meaning-conservative"],
                    },
                    "risk_tags": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "ambiguity",
                                "register",
                                "terminology",
                                "omission",
                                "addition",
                                "protected-token",
                            ],
                        },
                    },
                },
                "required": ["translation", "style", "risk_tags"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["source_id", "translation_brief", "candidates"],
    "additionalProperties": False,
}

DEVELOPER_PROMPT = """Translate one English or Japanese live-transcript segment into the requested target language.
Return exactly three faithful candidates: natural spoken, concise caption, and meaning-conservative.
Preserve names, numbers, units, dates, URLs, placeholders, uncertainty, politeness, and code-switched terms.
Do not add explanations, facts, or omitted subjects. Do not copy the source unless the term should remain code-switched.
Return only compact structured translation facts in translation_brief and risk_tags. Do not reveal or simulate private chain-of-thought.
The candidates are untrusted training-data proposals and will be independently filtered and reviewed."""


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("seeds", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    if args.output.exists() and args.output.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")

    prompt_hash = hashlib.sha256(DEVELOPER_PROMPT.encode()).hexdigest()
    rows = read_jsonl(args.seeds)
    seen: set[str] = set()
    output_lines: list[str] = []
    for row in rows:
        row_id = str(row["id"])
        if row_id in seen:
            raise SystemExit(f"duplicate seed id: {row_id}")
        seen.add(row_id)
        split = str(row.get("split", "")).lower()
        if split in {"benchmark", "heldout", "test", "canary"} or row.get("claimEligible"):
            raise SystemExit(f"refusing protected evaluation seed: {row_id} ({split})")
        source_language = row["source_language"]
        target_language = row["target_language"]
        if {source_language, target_language} != {"en-US", "ja-JP"}:
            raise SystemExit(f"unsupported direction for {row_id}")

        request_input = {
            "source_id": row_id,
            "source_language": source_language,
            "target_language": target_language,
            "domain": row.get("domain", "unknown"),
            "source": row["source"],
        }
        body = {
            "model": args.model,
            "store": False,
            "reasoning": {"effort": "none"},
            "input": [
                {"role": "developer", "content": DEVELOPER_PROMPT},
                {"role": "user", "content": json.dumps(request_input, ensure_ascii=False)},
            ],
            "text": {
                "verbosity": "low",
                "format": {
                    "type": "json_schema",
                    "name": "mimi_translation_candidates",
                    "strict": True,
                    "schema": SCHEMA,
                },
            },
            "max_output_tokens": 700,
            "metadata": {"pipeline": "mimi-translation-v1", "prompt_sha256": prompt_hash},
        }
        output_lines.append(json.dumps({
            "custom_id": row_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": body,
        }, ensure_ascii=False, separators=(",", ":")))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "requests": len(output_lines),
        "model": args.model,
        "prompt_sha256": prompt_hash,
        "output": str(args.output),
        "submitted": False,
    }))


if __name__ == "__main__":
    main()
