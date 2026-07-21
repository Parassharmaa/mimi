#!/usr/bin/env python3
"""Offline deterministic contract for DQRD hybrid source selection."""

from __future__ import annotations

import math
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from dqrd_selection import hybrid_select, selection_summary  # noqa: E402


def fixture() -> list[dict]:
    angles = [0, 2, 4, 45, 47, 49, 90, 135, 180, 225, 270, 315]
    return [
        {
            "id": f"source-{index:02d}",
            "student_sequence_nll": float(index + 1),
            "_selection_embedding": [
                math.cos(math.radians(angle)),
                math.sin(math.radians(angle)),
            ],
        }
        for index, angle in enumerate(angles)
    ]


def main() -> None:
    selected = hybrid_select(fixture(), 6, "dqrd-test")
    repeated = hybrid_select(fixture(), 6, "dqrd-test")
    assert selected == repeated
    assert len({row["id"] for row in selected}) == 6
    summary = selection_summary(selected)
    assert summary["selectedByStratum"] == {"high": 3, "low": 1, "medium": 2}
    assert all("_selection_embedding" not in row for row in selected)
    assert [row["selection_rank"] for row in selected] == list(range(1, 7))
    assert all(0 <= row["selection_diversity_distance"] <= 2 for row in selected)

    duplicate = fixture()
    duplicate[1]["id"] = duplicate[0]["id"]
    try:
        hybrid_select(duplicate, 2, "dqrd-test")
    except ValueError as error:
        assert "unique IDs" in str(error)
    else:
        raise AssertionError("duplicate IDs were accepted")

    invalid = fixture()
    invalid[0]["student_sequence_nll"] = float("nan")
    try:
        hybrid_select(invalid, 2, "dqrd-test")
    except ValueError as error:
        assert "non-finite" in str(error)
    else:
        raise AssertionError("non-finite uncertainty was accepted")

    print("Mimi DQRD hybrid selection contract passed.")


if __name__ == "__main__":
    main()
