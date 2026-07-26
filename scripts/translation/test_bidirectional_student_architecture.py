#!/usr/bin/env python3
"""Contract checks for the shared bidirectional Marian architecture analysis."""

from analyze_bidirectional_student_architecture import analyze


def main() -> None:
    result = analyze()
    rows = {row["name"]: row for row in result["architectures"]}

    incumbent = rows["incumbent-shape-6e6d-ffn2048"]
    assert incumbent["parameters"] == 60_555_009
    # The authenticated preferred-v3 child is 34,297,451 bytes. The analytical
    # projection must remain close enough to catch a changed q4 storage model.
    assert abs(incumbent["projected_q4_model_bytes"] - 34_297_451) < 5_000

    recommended = rows["shared-wide-6e6d-ffn4608"]
    assert 90_000_000 <= recommended["parameters"] <= 130_000_000
    assert recommended["projected_single_model_pack_bytes"] < 150_000_000
    assert (
        recommended["compute_profiles"]["long-segment"][
            "ratio_to_incumbent_shape"
        ]
        < 1.8
    )

    for name in (
        "shared-deep-18e4d-ffn2048",
        "shared-deep-24e4d-ffn2048",
        "shared-deep-30e4d-ffn2048",
    ):
        row = rows[name]
        assert 85_000_000 <= row["parameters"] <= 130_000_000
        assert row["projected_single_model_pack_bytes"] < 150_000_000

    assert result["decision"]["training_authorized"] is False
    assert result["decision"]["app_change_authorized"] is False
    print("Bidirectional student architecture checks passed")


if __name__ == "__main__":
    main()
