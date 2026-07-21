#!/usr/bin/env python3
"""Contracts for deterministic Marian target-vocabulary shortlists."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from transformers import PreTrainedTokenizerFast

from marian_target_shortlist import MarianTargetShortlist


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/translation/build_marian_target_shortlist.py"
TOKENIZER = (
    ROOT
    / "Research/translation/models/elanmt-release-clean-human-routed-moe-v2-memory-v2-mlx-4bit-shared-tokenizer-pack/shared/tokenizer.json"
)


with tempfile.TemporaryDirectory(prefix="mimi-target-shortlist-") as directory:
    output = Path(directory) / "shortlist.json"
    subprocess.run(
        ["python3", str(SCRIPT), str(TOKENIZER), str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(TOKENIZER),
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    shortlist = MarianTargetShortlist.load(output, TOKENIZER, tokenizer)
    assert 0 in shortlist.static_ids["en-ja"]
    assert 32_000 in shortlist.static_ids["ja-en"]
    assert len(shortlist.static_ids["en-ja"]) < 24_000
    assert len(shortlist.static_ids["ja-en"]) < 18_000

    vocabulary = tokenizer.get_vocab()
    linux_variants = sorted(
        token_id for token, token_id in vocabulary.items() if token.lstrip("▁") == "Linux"
    )
    assert len(linux_variants) >= 2
    expanded = shortlist.expand("en-ja", [linux_variants[0]])
    assert all(token_id in expanded for token_id in linux_variants)

    accented = [
        token_id
        for token, token_id in vocabulary.items()
        if any(character in token for character in "éöëí")
    ]
    assert accented and all(token_id in shortlist.static_ids["ja-en"] for token_id in accented)

print("Marian target-shortlist contracts passed.")
