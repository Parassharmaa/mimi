#!/usr/bin/env python3
"""Check the sealed V16 pre-semantic rejection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT = (
    ROOT
    / "Research/translation/active-sequence-risk-v16-presemantic-result-2026-07-26.json"
)
RESULT_SHA256 = "fd986b7099b72affd7a9dc3d51c1413911b087c0d7a21a044c95f24f74f646c1"
CONTRACT_SHA256 = "4a3c8ee0fa08a97bf9707501cb6c1b4d36d11b70bec58dfe2c16a64b720ff6c9"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert sha256(RESULT) == RESULT_SHA256
    assert result["experiment"] == "active-sequence-risk-v16-ja-en"
    assert result["status"] == "pre-semantic-gate-rejected"
    assert result["contract"]["sha256"] == CONTRACT_SHA256
    assert result["selected_for_semantic_audit_step"] is None
    assert result["selected_checkpoint"] is None
    assert [candidate["step"] for candidate in result["candidates"]] == [25, 50]

    expected_failures = {
        25: {
            "fresh-corpus-chrf++",
            "fresh-mean-sentence-chrf++",
            "fresh-worst-stratum-chrf++",
            "v12-mean-sentence-chrf++",
            "v14-mean-sentence-chrf++",
            "v15-mean-sentence-chrf++",
            "v15-omission-risk-chrf++",
            "active-all-preference",
            "active-omission-mean-margin",
            "active-repetition-mean-margin",
            "active-fraction-reduction",
        },
        50: {
            "fresh-corpus-chrf++",
            "fresh-mean-sentence-chrf++",
            "fresh-worst-stratum-chrf++",
            "v12-mean-sentence-chrf++",
            "v14-mean-sentence-chrf++",
            "v15-mean-sentence-chrf++",
            "active-all-preference",
            "active-omission-mean-margin",
            "active-repetition-mean-margin",
            "active-fraction-reduction",
        },
    }
    for candidate in result["candidates"]:
        step = candidate["step"]
        failed = {
            gate["name"]
            for gate in candidate["pre_semantic_gates"]
            if not gate["passed"]
        }
        assert failed == expected_failures[step]
        assert candidate["eligible_for_dual_semantic_audit"] is False
        for suite in candidate["suites"].values():
            assert suite["new_failures"]["generation"] == []

    step_50 = result["candidates"][1]
    assert step_50["suites"]["fresh_v16"]["deltas"]["mean_sentence_chrf_pp"] < 0.20
    assert step_50["suites"]["v12_regression"]["deltas"]["mean_sentence_chrf_pp"] < 0
    active = step_50["active_risk_deltas_from_safe_parent"]
    assert active["all"]["preference_accuracy"] < 0.025
    assert active["by_role"]["omission"]["mean_margin"] < 0.02
    assert active["by_role"]["repetition"]["mean_margin"] < 0.02
    assert active["all"]["active_fraction"] > -0.02

    for flag in (
        "semantic_audit_complete",
        "internal_gate_passed",
        "exact_q4_conversion_authorized",
        "comet_authorized",
        "protected_evaluation_authorized_after_exact_q4",
        "app_change_authorized",
        "bundle_replacement_authorized",
        "public_upload_authorized",
    ):
        assert result[flag] is False
    print("active sequence risk v16 result passed")


if __name__ == "__main__":
    main()
