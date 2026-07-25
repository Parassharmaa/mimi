#!/usr/bin/env python3
"""Check the frozen v15 contract and constrained-recovery objectives."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from train_canonical_constrained_recovery_v15 import (
    constrained_recovery_losses,
    omission_losses,
    pack_prefixes,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = (
    ROOT
    / "Research/translation/canonical-constrained-recovery-v15-contract-2026-07-26.json"
)
CONTRACT_SHA256 = "f342d8bf027f88143159c1b0ae2d5da3fb5ccad3cabb9aeb73e6d3175699549a"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert sha256(CONTRACT) == CONTRACT_SHA256
    assert contract["experiment"] == "canonical-constrained-recovery-v15-ja-en"
    assert contract["status"] == "preregistered-ready-for-one-arm-training"
    assert contract["capacity_change"] is False
    assert contract["moe_added"] is False
    assert contract["scheduled_sampling_used"] is False
    assert contract["unconditional_eos_recovery_used"] is False
    assert contract["dataset"]["counts"] == {
        "fresh_valid": 768,
        "train": 7_104,
        "v12_regression": 1_536,
        "v14_regression": 768,
    }
    assert contract["contrastive_examples"]["counts"] == {
        "omission": 2_048,
        "recovery": 2_048,
    }
    assert (
        contract["contrastive_examples"]["generated_strings_are_positive_targets"]
        is False
    )
    assert contract["training"]["checkpoint_steps"] == [25, 50]
    assert contract["training"]["one_arm_only"] is True
    assert contract["training"]["post_result_hyperparameter_changes_forbidden"] is True
    assert (
        contract["internal_selection"][
            "dual_semantic_audit_required_before_internal_pass"
        ]
        is True
    )
    assert set(contract["internal_selection"]["required_exact_semantic_judges"]) == {
        "claude-sonnet-5",
        "claude-opus-5",
    }
    for flag in (
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
    assert contract["v12_and_v14_rejections_final"] is True
    for item in contract["implementation"].values():
        path = ROOT / item["path"]
        assert sha256(path) == item["sha256"]

    ids, mask, positions = pack_prefixes(
        [[4, 5], [6]],
        decoder_start_token_id=9,
        pad_token_id=0,
    )
    assert ids.tolist() == [[9, 4, 5], [9, 6, 0]]
    assert mask.tolist() == [[1, 1, 1], [1, 1, 0]]
    assert positions.tolist() == [2, 1]

    clean_logits = torch.tensor(
        [[0.0, 0.5, 0.0, 0.0]],
        requires_grad=True,
    )
    perturbed_logits = torch.tensor(
        [[0.0, -0.5, 1.0, 0.0]],
        requires_grad=True,
    )
    recovery, constrained, recovery_margins, clean_margins, rejected = (
        constrained_recovery_losses(
            clean_logits,
            perturbed_logits,
            torch.tensor([1]),
            torch.tensor([2]),
            recovery_margin=0.1,
            clean_margin=0.01,
        )
    )
    assert recovery_margins.item() < 0
    assert clean_margins.item() > 0
    assert recovery.item() > 0
    assert constrained.item() == 0
    assert 0 < rejected.item() < 1
    (recovery + constrained).backward()
    assert torch.isfinite(clean_logits.grad).all()
    assert torch.isfinite(perturbed_logits.grad).all()

    omission_logits = torch.tensor(
        [[0.0, -0.5, 1.0, 0.0]],
        requires_grad=True,
    )
    omission, omission_margins = omission_losses(
        omission_logits,
        torch.tensor([1]),
        torch.tensor([2]),
        target_margin=0.5,
    )
    assert omission_margins.item() < 0
    assert omission.item() > 0
    omission.backward()
    assert torch.isfinite(omission_logits.grad).all()
    print("canonical constrained recovery v15 contract passed")


if __name__ == "__main__":
    main()
