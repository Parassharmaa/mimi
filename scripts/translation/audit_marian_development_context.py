#!/usr/bin/env python3
"""Audit direct and segmented Marian context lengths for a development suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--en-ja-tokenizer", type=Path, required=True)
    parser.add_argument("--ja-en-tokenizer", type=Path, required=True)
    parser.add_argument("--position-table-length", type=int, default=192)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")
    if args.position_table_length < 1:
        raise SystemExit("position table length must be positive")

    tokenizers = {
        ("en-US", "ja-JP"): Tokenizer.from_file(str(args.en_ja_tokenizer)),
        ("ja-JP", "en-US"): Tokenizer.from_file(str(args.ja_en_tokenizer)),
    }
    rows = load_jsonl(args.suite)
    results = []
    direct_overflows = Counter()
    segment_overflows = Counter()
    for row in rows:
        direction = (row.get("sourceLanguage"), row.get("targetLanguage"))
        tokenizer = tokenizers.get(direction)
        if tokenizer is None:
            raise SystemExit(f"unsupported direction: {row.get('id')}")
        segments = row.get("segments")
        if not isinstance(segments, list) or not segments:
            raise SystemExit(f"case has no segments: {row.get('id')}")
        direct_tokens = len(tokenizer.encode(str(row["source"])).ids)
        segment_tokens = [
            len(tokenizer.encode(str(segment)).ids) for segment in segments
        ]
        direct_supported = direct_tokens <= args.position_table_length
        segments_supported = all(
            length <= args.position_table_length for length in segment_tokens
        )
        label = f"{direction[0]}>{direction[1]}"
        if not direct_supported:
            direct_overflows[label] += 1
        if not segments_supported:
            segment_overflows[label] += 1
        results.append(
            {
                "caseID": row["id"],
                "direction": label,
                "domain": row["domain"],
                "sourceUnit": row["sourceUnit"],
                "segmentCount": len(segments),
                "sourceCharacters": len(str(row["source"])),
                "directInputTokens": direct_tokens,
                "directInputSupported": direct_supported,
                "segmentInputTokens": segment_tokens,
                "maximumSegmentInputTokens": max(segment_tokens),
                "allSegmentInputsSupported": segments_supported,
            }
        )

    output = {
        "schemaVersion": 1,
        "purpose": "Marian direct-input and app-matched segmented-context audit",
        "suiteSHA256": sha256(args.suite),
        "positionTableLength": args.position_table_length,
        "tokenizers": {
            "en-ja": {
                "path": str(args.en_ja_tokenizer),
                "sha256": sha256(args.en_ja_tokenizer),
            },
            "ja-en": {
                "path": str(args.ja_en_tokenizer),
                "sha256": sha256(args.ja_en_tokenizer),
            },
        },
        "cases": len(results),
        "documentCases": sum(result["sourceUnit"] == "document" for result in results),
        "directInputOverflowCases": sum(direct_overflows.values()),
        "directInputOverflowByDirection": dict(sorted(direct_overflows.items())),
        "segmentInputOverflowCases": sum(segment_overflows.values()),
        "segmentInputOverflowByDirection": dict(sorted(segment_overflows.items())),
        "contract": {
            "directInput": (
                "joined source must fit the 192-position Marian table in one call"
            ),
            "segmentedInput": (
                "every ordered source segment must fit independently; outputs are "
                "joined without cross-segment model context"
            ),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "cases": len(results),
                "documents": output["documentCases"],
                "directInputOverflows": output["directInputOverflowCases"],
                "segmentInputOverflows": output["segmentInputOverflowCases"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
