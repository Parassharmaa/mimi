#!/usr/bin/env python3
"""Freeze the non-cherry-picked direct-document slice for Marian evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer


LANGUAGES = {
    "en-ja": ("en-US", "ja-JP"),
    "ja-en": ("ja-JP", "en-US"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def model_contract(path: Path, direction: str) -> tuple[dict, Tokenizer, str]:
    manifest_path = path / "manifest.json"
    tokenizer_path = path / "tokenizer.json"
    if not manifest_path.is_file() or not tokenizer_path.is_file():
        raise SystemExit(f"{direction} model lacks manifest/tokenizer: {path}")
    manifest = load_json(manifest_path)
    if manifest.get("direction") != direction:
        raise SystemExit(f"{direction} model manifest identifies another direction")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    prefix = str((manifest.get("source_prefixes") or {}).get(direction, ""))
    return (
        {
            "path": str(path),
            "manifestSHA256": sha256(manifest_path),
            "tokenizerSHA256": sha256(tokenizer_path),
            "sourcePrefix": prefix,
        },
        tokenizer,
        prefix,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_suite", type=Path)
    parser.add_argument("en_ja_model", type=Path)
    parser.add_argument("ja_en_model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--maximum-source-tokens", type=int, default=192)
    args = parser.parse_args()
    for path in (args.output, args.manifest):
        if path.exists() and path.stat().st_size:
            raise SystemExit(f"refusing to overwrite non-empty output: {path}")
    if args.maximum_source_tokens < 1:
        raise SystemExit("maximum-source-tokens must be positive")

    model_inputs = {}
    runtimes = {}
    for direction, path in (
        ("en-ja", args.en_ja_model),
        ("ja-en", args.ja_en_model),
    ):
        model_inputs[direction], tokenizer, prefix = model_contract(path, direction)
        runtimes[direction] = (tokenizer, prefix)

    rows = load_jsonl(args.case_suite)
    seen: set[str] = set()
    selected: list[dict] = []
    excluded: list[dict] = []
    direction_counts: Counter[str] = Counter()
    selected_direction_counts: Counter[str] = Counter()
    for row in rows:
        case_id = str(row.get("id", "")).strip()
        pair = (row.get("sourceLanguage"), row.get("targetLanguage"))
        direction = next(
            (name for name, languages in LANGUAGES.items() if languages == pair),
            None,
        )
        if not case_id or case_id in seen or direction is None:
            raise SystemExit(f"case suite has invalid or duplicate row: {case_id}")
        seen.add(case_id)
        direction_counts[direction] += 1
        tokenizer, prefix = runtimes[direction]
        token_count = len(tokenizer.encode(prefix + str(row.get("source", ""))).ids)
        if token_count <= args.maximum_source_tokens:
            selected.append({**row, "directSourceTokenCount": token_count})
            selected_direction_counts[direction] += 1
        else:
            excluded.append(
                {
                    "caseID": case_id,
                    "direction": direction,
                    "sourceTokens": token_count,
                }
            )
    if not selected or set(selected_direction_counts) != set(LANGUAGES):
        raise SystemExit("direct slice must retain cases in both directions")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in selected
        ),
        encoding="utf-8",
    )
    payload = {
        "schemaVersion": 1,
        "purpose": "all frozen development documents whose exact runtime tokenizer input fits the registered direct limit",
        "selection": {
            "method": "tokenizers.Tokenizer.encode(sourcePrefix + source), including tokenizer post-processor",
            "maximumSourceTokensInclusive": args.maximum_source_tokens,
            "manualInclusions": [],
            "manualExclusions": [],
        },
        "caseSuite": {
            "path": str(args.case_suite),
            "sha256": sha256(args.case_suite),
            "cases": len(rows),
            "directions": dict(sorted(direction_counts.items())),
        },
        "models": model_inputs,
        "output": {
            "path": str(args.output),
            "sha256": sha256(args.output),
            "cases": len(selected),
            "directions": dict(sorted(selected_direction_counts.items())),
        },
        "excluded": excluded,
        "claimEligible": False,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selected": len(selected),
                "excluded": len(excluded),
                "directions": payload["output"]["directions"],
                "outputSHA256": payload["output"]["sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
