#!/usr/bin/env python3
"""Tensor-level contract for the regularized DQRD Marian objective."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


SCRIPT = Path(__file__).with_name("train_marian_distillation.py")
SPEC = importlib.util.spec_from_file_location("mimi_train_marian", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
TRAINING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAINING)


def main() -> None:
    variant_rows = TRAINING.TranslationRows(
        [
            {
                "id": "variant-source",
                "target": "canonical",
                "target_variants": [
                    {"candidate_id": "a", "translation": "canonical"},
                    {"candidate_id": "b", "translation": "alternative"},
                ],
            }
        ],
        seed=7,
        sample_target_variants=True,
    )
    observed = set()
    for epoch in range(16):
        variant_rows.set_epoch(epoch)
        observed.add(variant_rows[0]["target"])
    assert observed == {"canonical", "alternative"}

    labels = torch.tensor([[0, 1, -100], [1, 0, 1]])
    preservation = torch.tensor([True, False])
    logits = torch.tensor(
        [
            [[4.0, 0.0], [0.0, 4.0], [1.0, 1.0]],
            [[4.0, 0.0], [0.0, 4.0], [4.0, 0.0]],
        ],
        requires_grad=True,
    )
    full = TRAINING.weighted_sequence_cross_entropy(
        logits, labels, preservation, 1.0
    )
    replay_only = TRAINING.weighted_sequence_cross_entropy(
        logits, labels, preservation, 0.0
    )
    assert full > 0 and replay_only > 0
    assert full > replay_only
    full.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()

    matching_kl = TRAINING.frozen_base_kl(
        logits.detach(), logits.detach(), labels, preservation
    )
    assert abs(float(matching_kl)) < 1e-6
    changed_base = logits.detach().clone()
    changed_base[0, 0] = torch.tensor([0.0, 4.0])
    replay_kl = TRAINING.frozen_base_kl(
        logits.detach(), changed_base, labels, preservation
    )
    assert replay_kl > 0
    no_replay_kl = TRAINING.frozen_base_kl(
        logits.detach(), changed_base, labels, torch.tensor([False, False])
    )
    assert float(no_replay_kl) == 0

    assert TRAINING.curriculum_domain_weight(0, 100, 0.25, 1.0) == 0.25
    assert TRAINING.curriculum_domain_weight(50, 100, 0.25, 1.0) == 0.625
    assert TRAINING.curriculum_domain_weight(200, 100, 0.25, 1.0) == 1.0
    print("Mimi DQRD regularized training objective contract passed.")


if __name__ == "__main__":
    main()
