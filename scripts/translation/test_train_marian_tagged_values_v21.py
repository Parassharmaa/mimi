#!/usr/bin/env python3
"""Focused objective and restoration tests for the V21 tagged-value trainer."""

from __future__ import annotations

import torch

from train_marian_tagged_values_v21 import (
    extra_tag_unlikelihood,
    frozen_parent_kl_on_plain,
    restore_tagged_output,
    tagged_cross_entropy,
)


def main() -> None:
    tag_ids = torch.tensor([4, 5], dtype=torch.long)
    labels = torch.tensor([[1, 4, 2, -100]], dtype=torch.long)
    good = torch.full((1, 4, 6), -4.0)
    good[0, 0, 1] = 6.0
    good[0, 1, 4] = 6.0
    good[0, 2, 2] = 6.0
    bad_tag = good.clone()
    bad_tag[0, 1, 4] = -4.0
    bad_tag[0, 1, 5] = 6.0
    assert tagged_cross_entropy(
        good,
        labels,
        tag_ids,
        correct_tag_weight=8.0,
    ) < tagged_cross_entropy(
        bad_tag,
        labels,
        tag_ids,
        correct_tag_weight=8.0,
    )

    extra = good.clone()
    extra[0, 0, 4] = 6.0
    assert extra_tag_unlikelihood(good, labels, tag_ids) < extra_tag_unlikelihood(
        extra,
        labels,
        tag_ids,
    )

    student = torch.tensor([[[2.0, 1.0, -1.0, -2.0, -8.0, -8.0]]])
    parent = torch.tensor([[[2.0, 1.0, -1.0, -2.0]]])
    kl = frozen_parent_kl_on_plain(
        student,
        parent,
        torch.tensor([[1]]),
        torch.tensor([True]),
        base_vocabulary_size=4,
    )
    assert float(kl) < 1e-6

    sidecar = [
        {
            "tag": "<v00>",
            "kind": "number",
            "source_surface": "74",
            "target_surface": "seventy-four",
            "source_has_ascii_digits": True,
        },
        {
            "tag": "<v01>",
            "kind": "number",
            "source_surface": "百四十",
            "target_surface": "one hundred forty",
            "source_has_ascii_digits": False,
        },
    ]
    assert (
        restore_tagged_output("Article <v00> applies to <v01> people.", sidecar)
        == "Article 74 applies to one hundred forty people."
    )
    assert restore_tagged_output("Article <v00> applies.", sidecar) is None
    assert (
        restore_tagged_output(
            "Article <v00> applies to <v01> and <v01> people.",
            sidecar,
        )
        is None
    )

    print("Mimi V21 tagged-value training objective passed.")


if __name__ == "__main__":
    main()
