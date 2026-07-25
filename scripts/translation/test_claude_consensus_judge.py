#!/usr/bin/env python3
"""Offline identity checks for the authenticated Claude judge runner."""

from __future__ import annotations

from run_claude_consensus_judge import verified_primary_model_usage


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
    print("Claude consensus judge model identity checks passed")


if __name__ == "__main__":
    main()
