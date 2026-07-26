#!/usr/bin/env python3
"""Check the sealed V16 diagnostic and its no-training boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT = (
    ROOT
    / "Research/translation/active-sequence-risk-v16-diagnostic-result-2026-07-26.json"
)
RESULT_SHA256 = "0272e49d4a6ebd9d87df8b51099beb354510c26f277b4b29b29d9ab98d98978d"
IMPLEMENTATION_SHA256 = (
    "9dc0ef32b6f80ea58a82542c4f288cb33c5f82be19c464812cd9930a9602ba67"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert sha256(RESULT) == RESULT_SHA256
    assert result["experiment"] == "active-sequence-risk-v16-ja-en-diagnostic"
    assert result["status"] == "diagnostic-complete-no-training-authorized"
    assert result["direction"] == "ja-en"

    implementation = result["implementation"]
    assert implementation["sha256"] == IMPLEMENTATION_SHA256
    assert sha256(ROOT / implementation["path"]) == IMPLEMENTATION_SHA256

    assert result["dataset"]["selected_rows"] == 1_024
    assert result["dataset"]["validation_rows_used"] is False
    assert result["dataset"]["protected_rows_used"] is False
    assert result["counts"] == {
        "active_pairs": 677,
        "active_roles": {"omission": 228, "repetition": 449},
        "active_strata": {
            "legal:long": 437,
            "legal:omission-risk": 181,
            "legal:repetition-risk": 59,
        },
        "candidate_pairs": 2_028,
        "candidate_roles": {"omission": 1_004, "repetition": 1_024},
    }
    assert result["outputs"]["scored"]["rows"] == 2_028
    assert (
        result["outputs"]["scored"]["sha256"]
        == "2de01459ab1b978f870b5bd69c665586a4de7531aa5a375af61b3f318f40da97"
    )
    assert result["outputs"]["active"]["rows"] == 677
    assert (
        result["outputs"]["active"]["sha256"]
        == "ce93785835a7a0c37b6b3661479aef4cc7059c0b1e60f2245c7dfe89a5598e50"
    )

    provenance = result["distribution_provenance"]
    assert provenance == {
        "all_positive_targets_are_licensed_human_references": True,
        "all_rows_have_source_license": True,
        "all_rows_have_source_provenance": True,
        "negative_strings_are_positive_targets": False,
        "private_reasoning_traces_used": False,
    }

    cosine = result["gradient_audit"]["cosine_summary"]
    assert cosine["omission_vs_repetition"]["negative_fraction"] == 1.0
    assert cosine["omission_vs_repetition"]["maximum"] < 0
    for pair in ("mle_vs_omission", "mle_vs_repetition"):
        assert cosine[pair]["negative_fraction"] == 0.0
        assert cosine[pair]["minimum"] > 0
    for replicate in result["gradient_audit"]["replicates"]:
        for objective in ("mle", "omission", "repetition"):
            summary = replicate["objectives"][objective]
            assert summary["gradient_norm"] > 0
            assert summary["active_fraction"] == 1.0

    assert result["gradient_audit"]["optimizer_selected"] is False
    assert result["gradient_audit"]["diagnostic_is_not_optimizer_authorization"]
    for flag in (
        "optimizer_step_executed",
        "model_checkpoint_written",
        "training_authorized",
        "app_change_authorized",
        "bundle_replacement_authorized",
        "public_upload_authorized",
    ):
        assert result[flag] is False

    source = (ROOT / implementation["path"]).read_text(encoding="utf-8")
    assert "optimizer.step(" not in source
    assert "training_authorized" in source
    print("active sequence risk v16 diagnostic passed")


if __name__ == "__main__":
    main()
