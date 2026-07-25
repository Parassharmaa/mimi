#!/usr/bin/env python3
"""Check the frozen v14 rollout-repair contract and objective helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from mine_canonical_rollout_repair_v14 import first_repeat_trigger
from train_canonical_rollout_repair_v14 import (
    recovery_losses,
    scheduled_decoder_inputs,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = (
    ROOT / "Research/translation/canonical-rollout-repair-v14-contract-2026-07-26.json"
)
CONTRACT_SHA256 = "63ae84a0661b3b84aba62232b2c0115fe2ee61b23a07578c6e18717e4f9b4618"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DummyShiftModel:
    def prepare_decoder_input_ids_from_labels(
        self,
        *,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        result = torch.full_like(labels, 9)
        result[:, 1:] = torch.where(
            labels[:, :-1].eq(-100),
            torch.tensor(0),
            labels[:, :-1],
        )
        return result


def main() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert sha256(CONTRACT) == CONTRACT_SHA256
    assert contract["experiment"] == "canonical-rollout-repair-v14-ja-en"
    assert contract["status"] == "preregistered-ready-for-one-arm-training"
    assert contract["capacity_change"] is False
    assert contract["moe_added"] is False
    assert contract["dataset"]["counts"]["train"] == 7_104
    assert contract["dataset"]["counts"]["valid"] == 768
    assert contract["rollout_dataset"]["counts"]["rollouts"] == 7_104
    assert contract["rollout_dataset"]["counts"]["hard"] == 2_048
    assert contract["rollout_dataset"]["counts"]["recovery"] == 74
    assert contract["rollout_dataset"]["rollout_strings_are_positive_targets"] is False
    assert contract["rollout_dataset"]["recovery_target_is_only_eos"]
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
    assert contract["v12_rejection_final"] is True
    for item in contract["implementation"].values():
        path = ROOT / item["path"]
        assert sha256(path) == item["sha256"]

    assert first_repeat_trigger([1, 2, 3, 1, 2, 3, 1, 2, 3]) == {
        "phrase_start": 0,
        "phrase_width": 3,
        "trigger_index": 6,
    }
    assert first_repeat_trigger([1, 2, 3, 4, 5]) is None

    logits = torch.tensor(
        [[0.0, 3.0, -1.0]],
        requires_grad=True,
    )
    unlikelihood, ranking, probability, preferred = recovery_losses(
        logits,
        torch.tensor([1]),
        torch.tensor([2]),
        target_margin=1.0,
    )
    assert preferred.tolist() == [True]
    assert probability.item() < 0.02
    (unlikelihood + ranking).backward()
    assert torch.isfinite(logits.grad).all()

    labels = torch.tensor([[4, 5, 6, -100]])
    teacher_forced = torch.zeros((1, 4, 10))
    teacher_forced[0, 0, 7] = 10
    teacher_forced[0, 1, 8] = 10
    teacher_forced[0, 2, 0] = 10
    mixed, replacement_mask = scheduled_decoder_inputs(
        DummyShiftModel(),
        labels,
        teacher_forced,
        probability=1.0,
        generator=torch.Generator().manual_seed(1),
        pad_token_id=0,
    )
    assert mixed.tolist() == [[9, 7, 8, 6]]
    assert replacement_mask.tolist() == [[True, True, False]]
    print("canonical rollout repair v14 contract passed")


if __name__ == "__main__":
    main()
