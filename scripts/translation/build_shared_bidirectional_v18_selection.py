#!/usr/bin/env python3
"""Freeze V18's direction/domain/length-stratified checkpoint selector."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SEED = 20260726
LONG_PAIR_CHARACTERS = 200
QUOTAS = {
    "en-ja": {
        "mimi-product-ui": 13,
        "conversational": 128,
        "wikipedia-long-pair": 24,
        "wikipedia-short-pair": 91,
    },
    "ja-en": {
        "mimi-product-ui": 13,
        "conversational": 64,
        "human-translated-news": 64,
        "wikipedia-long-pair": 24,
        "wikipedia-short-pair": 91,
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit(f"selection source is empty: {path}")
    return rows


def stratum(row: dict) -> str:
    domain = str(row.get("domain", ""))
    if domain != "wikipedia":
        return domain
    pair_characters = max(len(str(row["source"])), len(str(row["target"])))
    suffix = "long-pair" if pair_characters >= LONG_PAIR_CHARACTERS else "short-pair"
    return f"wikipedia-{suffix}"


def select(rows: list[dict], seed: int) -> list[dict]:
    selected: list[dict] = []
    identifiers: set[str] = set()
    for direction_index, direction in enumerate(QUOTAS):
        direction_rows = [row for row in rows if row.get("direction") == direction]
        for stratum_index, (name, count) in enumerate(QUOTAS[direction].items()):
            candidates = [row for row in direction_rows if stratum(row) == name]
            if len(candidates) < count:
                raise SystemExit(
                    f"insufficient {direction}/{name} rows: {len(candidates)} < {count}"
                )
            candidates.sort(key=lambda row: str(row["id"]))
            random.Random(seed + direction_index * 100 + stratum_index).shuffle(
                candidates
            )
            for row in candidates[:count]:
                identifier = str(row.get("id", ""))
                if not identifier or identifier in identifiers:
                    raise SystemExit(f"missing or duplicate selected ID: {identifier}")
                identifiers.add(identifier)
                selected.append(
                    {
                        **row,
                        "selection_stratum": name,
                        "selection_seed": seed,
                    }
                )
    return sorted(selected, key=lambda row: (row["direction"], row["id"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_validation", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("output_manifest", type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    for output in (args.output_jsonl, args.output_manifest):
        if output.exists():
            raise SystemExit(f"refusing to overwrite selection artifact: {output}")

    rows = load_rows(args.source_validation)
    selected = select(rows, args.seed)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in selected
        ),
        encoding="utf-8",
    )
    counts = Counter(
        (str(row["direction"]), str(row["selection_stratum"])) for row in selected
    )
    manifest = {
        "schema_version": 1,
        "experiment": "shared-bidirectional-v18-phase1-selection",
        "operation": "deterministic-direction-domain-length-stratified-selection",
        "builder": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": sha256(Path(__file__)),
        },
        "seed": args.seed,
        "source": {
            "path": str(args.source_validation),
            "sha256": sha256(args.source_validation),
            "rows": len(rows),
        },
        "output": {
            "path": str(args.output_jsonl),
            "sha256": sha256(args.output_jsonl),
            "rows": len(selected),
        },
        "quotas": QUOTAS,
        "counts": {
            direction: {
                name: counts[(direction, name)] for name in QUOTAS[direction]
            }
            for direction in QUOTAS
        },
        "cases_per_direction": {
            direction: sum(
                row["direction"] == direction for row in selected
            )
            for direction in QUOTAS
        },
        "long_pair_definition": (
            "max Unicode code-point length of source or target is at least "
            f"{LONG_PAIR_CHARACTERS}"
        ),
        "known_selection_limits": [
            "the licensed source validation set has no EN-JA news rows",
            "the licensed source validation set has no legal rows",
            (
                "the licensed source validation set has no JA-EN source at or "
                "above 100 characters"
            ),
            (
                "legal and true long-input evaluation therefore remain mandatory "
                "external gates for every saved checkpoint"
            ),
        ],
        "selection_evidence_only": True,
        "promotion_evidence": False,
        "training_rows_used": False,
        "private_reasoning_traces_used": False,
    }
    args.output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
