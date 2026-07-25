#!/usr/bin/env python3
"""Check the tracked v13 dual-Claude safety-taxonomy evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = (
    ROOT / "Research/translation/canonical-safety-taxonomy-v13-contract-2026-07-26.json"
)
RESULT = (
    ROOT / "Research/translation/canonical-safety-taxonomy-v13-result-2026-07-26.json"
)
MODELS = {"claude-sonnet-5", "claude-opus-5"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def main() -> None:
    contract = load(CONTRACT)
    result = load(RESULT)

    assert contract["experiment"] == "canonical-safety-taxonomy-v13-ja-en"
    assert contract["status"] == "frozen-before-any-taxonomy-judgment"
    assert set(contract["judge_policy"]["required_exact_models"]) == MODELS
    assert contract["judge_policy"]["fallback_model_allowed"] is False
    assert contract["judge_policy"]["reasoning_trace_requested_or_stored"] is False
    payload_hashes = {
        value["model_independent_payload_sha256"]
        for value in contract["judge_requests"].values()
    }
    assert len(payload_hashes) == 1
    assert contract["queue"]["sources"] == 27
    assert contract["queue"]["candidates"] == 108

    assert result["experiment"] == contract["experiment"]
    assert result["status"] == "dual-claude5-taxonomy-complete"
    assert result["contract"]["sha256"] == sha256(CONTRACT)
    assert set(result["judges"]) == MODELS
    for model, evidence in result["judges"].items():
        assert evidence["model"] == model
        assert evidence["actual_canonical_model_usage_verified"] is True
        assert evidence["verified_shards"] == 7

    counts = result["counts"]
    assert counts["sources"] == 27
    assert counts["candidates"] == 108
    assert counts["registered_failure_events"] == 55
    assert counts["critical_agreement"] == 93
    assert counts["representation_agreement"] == 86
    assert counts["by_role"]["licensed-reference"]["critical_fail_closed"] == 2
    assert counts["by_role"]["safe-parent"]["critical_fail_closed"] == 20
    assert counts["by_role"]["v12-step-50"]["critical_fail_closed"] == 19
    assert counts["by_role"]["v12-step-100"]["critical_fail_closed"] == 20

    audit = counts["registered_failure_audit"]
    for role in ("v12-step-50", "v12-step-100"):
        assert audit[role]["negation"]["registered"] == 5
        assert audit[role]["negation"]["supported_fail_closed"] == 0
        assert audit[role]["generation"]["registered"] == 1
        assert audit[role]["generation"]["supported_unanimously"] == 1

    cases = result["cases"]
    assert len(cases) == 27
    assert len({case["source_id"] for case in cases}) == 27
    assert sum(len(case["candidates"]) for case in cases) == 108
    for case in cases:
        assert len(case["candidates"]) == 4
        for candidate in case["candidates"]:
            assert set(candidate["judgments"]) == MODELS

    for flag in (
        "v12_selection_or_decision_changed",
        "training_authorized",
        "protected_evaluation_authorized",
        "app_change_authorized",
        "public_upload_authorized",
        "reasoning_traces_stored",
        "human_reviewers_used",
    ):
        assert result[flag] is False
    assert result["v12_rejection_final"] is True
    print("canonical safety taxonomy v13 evidence passed")


if __name__ == "__main__":
    main()
