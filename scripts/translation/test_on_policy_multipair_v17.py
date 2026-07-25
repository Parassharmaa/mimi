#!/usr/bin/env python3
"""Focused tests for the V17 on-policy prediagnostic."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))

from mine_on_policy_multipair_v17 import (
    build_comet_inputs,
    count_tags,
    deterministic_risk_tags,
    preselect_scored_rows,
    stable_digest,
)


def fixture_row() -> dict:
    return {
        "id": "fixture-1",
        "direction": "ja-en",
        "domain": "legal",
        "origin": "licensed-fixture",
        "source": "3日以内に提出しなければならない。",
        "target": "It must be submitted within 3 days.",
        "source_language": "ja-JP",
        "target_language": "en-US",
        "source_license": "CC-BY-4.0",
        "source_provenance": "fixture",
        "reference_score": {
            "mean_log_probability": -2.0,
            "total_log_probability": -16.0,
            "token_count": 8,
        },
    }


def main() -> None:
    assert stable_digest(7, "x", "a") == stable_digest(7, "x", "a")
    assert stable_digest(7, "x", "a") != stable_digest(8, "x", "a")

    row = fixture_row()
    omission = {
        "candidate_id": "candidate-omission",
        "hypothesis": "Submit it.",
        "token_ids": [1, 2],
        "terminated": True,
        "reached_generation_limit": False,
        "repeated_token_loop": False,
        "generation_origins": [{"name": "beam-4", "position": 0, "seed": 7}],
        "score": {
            "mean_log_probability": -1.9,
            "total_log_probability": -3.8,
            "token_count": 2,
        },
    }
    tags = deterministic_risk_tags(row, omission)
    assert "omission-risk" in tags
    assert "exact-critical-mismatch" in tags
    assert "typed-critical-mismatch" in tags

    repetition = {
        **omission,
        "candidate_id": "candidate-repeat",
        "hypothesis": "Submit it. Submit it. Submit it.",
        "repeated_token_loop": True,
        "score": {
            "mean_log_probability": -2.1,
            "total_log_probability": -16.8,
            "token_count": 8,
        },
    }
    assert "repetition-risk" in deterministic_risk_tags(row, repetition)

    source_row = dict(row)
    source_row["candidate_list"] = [
        {
            **omission,
            "reference_minus_candidate_margin": -0.1,
            "chrf_plus_plus": 12.0,
            "sacrebleu_intl": 3.0,
            "deterministic_risk_tags": tags,
            "exact_reference_match": False,
        },
        {
            **repetition,
            "reference_minus_candidate_margin": 0.1,
            "chrf_plus_plus": 40.0,
            "sacrebleu_intl": 10.0,
            "deterministic_risk_tags": deterministic_risk_tags(
                row,
                repetition,
            ),
            "exact_reference_match": False,
        },
    ]
    pairs = preselect_scored_rows(
        [source_row],
        near_margin=0.25,
        maximum_pairs_per_source=2,
    )
    assert len(pairs) == 2
    assert all(pair["semantic_status"] == "not-yet-judged" for pair in pairs)
    assert all(
        pair["generated_strings_are_positive_targets"] is False for pair in pairs
    )
    assert count_tags(pairs)["repetition-risk"] == 1

    suite, report = build_comet_inputs(pairs)
    assert len(suite) == len(report["results"]) == 2
    assert {row["id"] for row in suite} == {
        row["caseID"] for row in report["results"]
    }
    for case, result in zip(suite, report["results"], strict=True):
        for field in (
            "sourceLanguage",
            "targetLanguage",
            "domain",
            "source",
            "references",
        ):
            assert case[field] == result[field]
    print("V17 on-policy prediagnostic tests passed")


if __name__ == "__main__":
    main()
