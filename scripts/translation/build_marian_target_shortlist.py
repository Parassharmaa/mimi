#!/usr/bin/env python3
"""Build a deterministic tokenizer-only Marian target shortlist artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import PreTrainedTokenizerFast

from marian_target_shortlist import artifact_payload, sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists() and args.output.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(args.tokenizer),
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    payload = artifact_payload(args.tokenizer, tokenizer.get_vocab())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": sha256(args.output),
                "staticTokenCounts": {
                    key: len(value["staticTokenIDs"])
                    for key, value in payload["directions"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
