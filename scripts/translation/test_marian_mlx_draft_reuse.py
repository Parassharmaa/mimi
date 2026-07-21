#!/usr/bin/env python3
"""Contract tests for exact prior-output verification."""

from benchmark_marian_partial_retranslation import (
    EOS_TOKEN_ID,
    verified_draft_matches,
)


def main() -> None:
    assert verified_draft_matches([41, 42, EOS_TOKEN_ID], [41, 42])
    assert not verified_draft_matches([41, 99, EOS_TOKEN_ID], [41, 42])
    assert not verified_draft_matches([41, 42, 17], [41, 42])
    assert not verified_draft_matches([41, 42], [41, 42])
    assert verified_draft_matches([EOS_TOKEN_ID], [])
    print("Marian MLX verified-draft contract passed.")


if __name__ == "__main__":
    main()
