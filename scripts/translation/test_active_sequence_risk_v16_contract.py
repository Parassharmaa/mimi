#!/usr/bin/env python3
"""Check the frozen V16 PCGrad contract and core gradient rule."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from train_active_sequence_risk_v16 import (
    assign_combined_gradients,
    gradient_dot,
    symmetric_pcgrad,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = (
    ROOT / "Research/translation/active-sequence-risk-v16-contract-2026-07-26.json"
)
CONTRACT_SHA256 = "4a3c8ee0fa08a97bf9707501cb6c1b4d36d11b70bec58dfe2c16a64b720ff6c9"
DATASET_MANIFEST_SHA256 = (
    "21a3c7e23c190b28bb6d2ded323f3bd1bbde93e8fda44216d44c5be466b25902"
)
DIAGNOSTIC_RESULT_SHA256 = (
    "0272e49d4a6ebd9d87df8b51099beb354510c26f277b4b29b29d9ab98d98978d"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert sha256(CONTRACT) == CONTRACT_SHA256
    assert contract["experiment"] == "active-sequence-risk-v16-ja-en"
    assert contract["status"] == "preregistered-ready-for-one-arm-training"
    assert contract["direction"] == "ja-en"
    assert contract["diagnostic"]["result"]["sha256"] == DIAGNOSTIC_RESULT_SHA256
    assert contract["diagnostic"]["active_pairs"] == 677
    assert contract["diagnostic"]["negative_preferred_pairs"] == 169
    assert (
        contract["diagnostic"]["omission_repetition_gradient_conflict_fraction"] == 1.0
    )
    assert contract["diagnostic"]["mle_is_projected"] is False
    assert contract["dataset"]["manifest"]["sha256"] == DATASET_MANIFEST_SHA256
    assert contract["dataset"]["counts"] == {
        "active": 677,
        "active_omission": 228,
        "active_repetition": 449,
        "fresh_valid": 768,
        "train": 7_104,
        "v12_regression": 1_536,
        "v14_regression": 768,
        "v15_regression": 768,
    }
    assert contract["dataset"]["exhausted_fresh_strata"] == [
        "omission-risk",
        "repetition-risk",
    ]
    provenance = contract["dataset"]["distribution_provenance"]
    assert provenance["all_positive_targets_are_licensed_human_references"]
    assert provenance["negative_strings_are_positive_targets"] is False
    assert provenance["private_reasoning_traces_used"] is False

    training = contract["training"]
    assert training["checkpoint_steps"] == [25, 50]
    assert training["max_steps"] == 50
    assert training["sequence_target_margin"] == 0.25
    assert training["omission_weight"] == training["repetition_weight"] == 0.35
    assert training["gradient_rule"] == (
        "unprojected-mle-plus-symmetric-pcgrad-between-omission-and-repetition"
    )
    assert training["one_arm_only"] is True
    assert training["post_result_hyperparameter_changes_forbidden"] is True
    formula = contract["pcgrad_formula"]
    assert formula["scope"] == "omission and repetition gradients only"
    assert formula["projection_inputs"] == (
        "the two original unprojected safety gradients"
    )
    assert formula["mle_projection_forbidden"] is True
    assert formula["deterministic"] is True

    assert set(contract["internal_selection"]["required_exact_semantic_judges"]) == {
        "claude-sonnet-5",
        "claude-opus-5",
    }
    assert (
        contract["internal_selection"][
            "dual_semantic_audit_required_before_internal_pass"
        ]
        is True
    )
    for flag in (
        "capacity_change",
        "moe_added",
        "scheduled_sampling_used",
        "exact_q4_conversion_authorized",
        "protected_evaluation_authorized",
        "comet_authorized",
        "app_change_authorized",
        "bundle_replacement_authorized",
        "public_upload_authorized",
        "human_reviewer_required",
        "private_reasoning_traces_used",
    ):
        assert contract[flag] is False
    for item in contract["implementation"].values():
        assert sha256(ROOT / item["path"]) == item["sha256"]

    omission = [torch.tensor([1.0, 0.0]), torch.tensor([0.5])]
    repetition = [torch.tensor([-1.0, 1.0]), torch.tensor([-0.5])]
    original_dot = float(gradient_dot(omission, repetition))
    projected_omission, projected_repetition, audit = symmetric_pcgrad(
        omission,
        repetition,
        epsilon=1e-12,
    )
    assert original_dot < 0
    assert audit["conflict"] is True
    assert audit["cosine_before"] < 0
    assert float(gradient_dot(projected_omission, projected_repetition)) >= -1e-6
    assert audit["cosine_after"] >= -1e-6

    aligned_left = [torch.tensor([1.0, 0.0])]
    aligned_right = [torch.tensor([1.0, 1.0])]
    kept_left, kept_right, aligned_audit = symmetric_pcgrad(
        aligned_left,
        aligned_right,
        epsilon=1e-12,
    )
    assert aligned_audit["conflict"] is False
    assert torch.equal(kept_left[0], aligned_left[0])
    assert torch.equal(kept_right[0], aligned_right[0])

    parameter = torch.nn.Parameter(torch.zeros(2))
    assign_combined_gradients(
        [parameter],
        [torch.tensor([1.0, 2.0])],
        [torch.tensor([3.0, 4.0])],
        [torch.tensor([5.0, 6.0])],
        omission_weight=0.35,
        repetition_weight=0.35,
    )
    assert parameter.grad is not None
    assert torch.allclose(
        parameter.grad,
        torch.tensor([3.8, 5.5]),
    )
    print("active sequence risk v16 contract passed")


if __name__ == "__main__":
    main()
