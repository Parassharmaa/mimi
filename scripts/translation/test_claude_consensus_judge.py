#!/usr/bin/env python3
"""Offline identity checks for the authenticated Claude judge runner."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from run_claude_consensus_judge import (
    JUDGE_PIPELINE,
    read_requests,
    verified_primary_model_usage,
)


def usage(canonical_model: str, output_tokens: int) -> dict:
    return {
        "canonicalModel": canonical_model,
        "outputTokens": output_tokens,
        "provider": "firstParty",
    }


def must_fail(envelope: dict, requested_model: str) -> None:
    try:
        verified_primary_model_usage(envelope, requested_model)
    except ValueError:
        return
    raise AssertionError("model identity validation unexpectedly passed")


def pairwise_request() -> dict:
    prompt = "Judge blinded translations."
    schema = {
        "type": "object",
        "properties": {"source_id": {"type": "string"}},
        "required": ["source_id"],
        "additionalProperties": False,
    }
    return {
        "custom_id": "source-1",
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": "claude-sonnet-5",
            "store": False,
            "reasoning": {"effort": "low"},
            "input": [
                {"role": "developer", "content": prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "source_id": "source-1",
                            "source_language": "en-US",
                            "target_language": "ja-JP",
                            "domain": "test",
                            "source": "Do not enter.",
                            "candidates": [
                                {
                                    "candidate_id": "candidate-a",
                                    "translation": "立入禁止。",
                                },
                                {
                                    "candidate_id": "candidate-b",
                                    "translation": "入ってください。",
                                },
                            ],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "text": {
                "verbosity": "low",
                "format": {
                    "type": "json_schema",
                    "name": "pairwise_test",
                    "strict": True,
                    "schema": schema,
                }
            },
            "max_output_tokens": 300,
            "metadata": {
                "pipeline": JUDGE_PIPELINE,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            },
        },
    }


def main() -> None:
    evidence = verified_primary_model_usage(
        {
            "modelUsage": {
                "claude-sonnet-5": usage("claude-sonnet-5", 120),
                "claude-haiku-4-5-20251001": usage("claude-haiku-4-5", 8),
            }
        },
        "claude-sonnet-5",
    )
    assert evidence["canonical_model"] == "claude-sonnet-5"
    assert evidence["auxiliary_models"] == ["claude-haiku-4-5"]

    must_fail(
        {"modelUsage": {"claude-opus-5": usage("claude-opus-5", 120)}},
        "claude-sonnet-5",
    )
    must_fail(
        {
            "modelUsage": {
                "claude-sonnet-5": usage("claude-sonnet-5", 120),
                "claude-opus-5": usage("claude-opus-5", 40),
            }
        },
        "claude-sonnet-5",
    )
    must_fail(
        {"modelUsage": {"claude-sonnet-5": usage("claude-sonnet-5", 0)}},
        "claude-sonnet-5",
    )

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "pairwise.jsonl"
        path.write_text(
            json.dumps(pairwise_request(), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        requests, contract = read_requests(path)
        assert requests[0]["custom_id"] == "source-1"
        assert len(requests[0]["source"]["candidates"]) == 2
        assert contract["model"] == "claude-sonnet-5"

    print("Claude consensus judge model identity checks passed")


if __name__ == "__main__":
    main()
