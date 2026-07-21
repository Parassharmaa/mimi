#!/usr/bin/env python3
"""Balance two licensed directional datasets for one shared Marian student."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path


ALLOWED_LICENSES = {"CC-BY-2.0-FR", "CC-BY-SA-3.0", "project-owned"}
DIRECTIONS = {
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


def load_rows(path: Path, direction: str) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit(f"dataset is empty: {path}")
    expected = DIRECTIONS[direction]
    identifiers: set[str] = set()
    output: list[dict] = []
    for row in rows:
        if (row.get("source_language"), row.get("target_language")) != expected:
            raise SystemExit(f"wrong direction in {path}: {row.get('id')}")
        if row.get("source_license") not in ALLOWED_LICENSES:
            raise SystemExit(
                f"unapproved or missing source license in {path}: "
                f"{row.get('source_license')}"
            )
        identifier = str(row.get("id", ""))
        if not identifier or identifier in identifiers:
            raise SystemExit(f"missing or duplicate row ID in {path}: {identifier}")
        identifiers.add(identifier)
        output.append({**row, "direction": direction})
    return output


def normalized(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", "", value)


def ngrams(text: str, size: int = 3) -> set[str]:
    value = normalized(text)
    if len(value) <= size:
        return {value} if value else set()
    return {value[index:index + size] for index in range(len(value) - size + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / max(1, len(left | right))


def protected_texts(path: Path) -> list[set[str]]:
    output: list[set[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for field in ("source", "references"):
            value = row.get(field, [])
            texts = value if isinstance(value, list) else [value]
            output.extend(ngrams(str(text)) for text in texts if text)
    if not output:
        raise SystemExit("protected benchmark has no text")
    return output


def validate_contamination(
    rows: list[dict],
    protected: list[set[str]],
    maximum_jaccard: float,
) -> None:
    for row in rows:
        for field in ("source", "target"):
            candidate = ngrams(str(row[field]))
            if any(jaccard(candidate, heldout) > maximum_jaccard for heldout in protected):
                raise SystemExit(
                    f"protected benchmark near-overlap: {row['id']} / {field}"
                )


def repeat_to_count(rows: list[dict], count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    output: list[dict] = []
    repeats: Counter[str] = Counter()
    for index in range(count):
        row = rows[order[index % len(order)]]
        repeats[row["id"]] += 1
        repeat_index = repeats[row["id"]] - 1
        identifier = f"{row['direction']}:{row['id']}"
        if repeat_index:
            identifier += f":balance-repeat-{repeat_index}"
        output.append(
            {
                **row,
                "id": identifier,
                "original_id": row["id"],
                "balance_repeat_index": repeat_index,
            }
        )
    return output


def interleave(left: list[dict], right: list[dict]) -> list[dict]:
    output: list[dict] = []
    for index in range(max(len(left), len(right))):
        if index < len(left):
            output.append(left[index])
        if index < len(right):
            output.append(right[index])
    return output


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("en_ja_directory", type=Path)
    parser.add_argument("ja_en_directory", type=Path)
    parser.add_argument("protected_benchmark", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--maximum-jaccard", type=float, default=0.8)
    args = parser.parse_args()
    if not 0 <= args.maximum_jaccard <= 1:
        raise SystemExit("maximum-jaccard must be between zero and one")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_directory}")

    roots = {"en-ja": args.en_ja_directory, "ja-en": args.ja_en_directory}
    datasets: dict[str, dict[str, list[dict]]] = {}
    input_records: dict[str, dict] = {}
    expected_protected_hash = sha256(args.protected_benchmark)
    for direction, root in roots.items():
        source_manifest_path = root / "manifest.json"
        source_manifest = load_json(source_manifest_path)
        protected_record = source_manifest.get("inputs", {}).get("protected_benchmark", {})
        if protected_record.get("sha256") != expected_protected_hash:
            raise SystemExit(
                f"source dataset was not screened against this protected benchmark: {root}"
            )
        train_path = root / "train.jsonl"
        valid_path = root / "valid.jsonl"
        train_rows = load_rows(train_path, direction)
        valid_rows = load_rows(valid_path, direction)
        train_pairs = {(normalized(row["source"]), normalized(row["target"])) for row in train_rows}
        valid_pairs = {(normalized(row["source"]), normalized(row["target"])) for row in valid_rows}
        if train_pairs & valid_pairs:
            raise SystemExit(f"source dataset has exact train/valid overlap: {root}")
        datasets[direction] = {"train": train_rows, "valid": valid_rows}
        input_records[direction] = {
            "directory": str(root),
            "manifest_sha256": sha256(source_manifest_path),
            "train_sha256": sha256(train_path),
            "train_rows": len(train_rows),
            "valid_sha256": sha256(valid_path),
            "valid_rows": len(valid_rows),
        }

    all_rows = [
        row
        for direction in DIRECTIONS
        for split in ("train", "valid")
        for row in datasets[direction][split]
    ]
    validate_contamination(
        all_rows,
        protected_texts(args.protected_benchmark),
        args.maximum_jaccard,
    )

    balanced_count = max(len(datasets[direction]["train"]) for direction in DIRECTIONS)
    balanced_train = {
        direction: repeat_to_count(
            datasets[direction]["train"],
            balanced_count,
            args.seed + index,
        )
        for index, direction in enumerate(DIRECTIONS)
    }
    valid = {
        direction: repeat_to_count(
            datasets[direction]["valid"],
            len(datasets[direction]["valid"]),
            args.seed + 100 + index,
        )
        for index, direction in enumerate(DIRECTIONS)
    }
    train_rows = interleave(balanced_train["en-ja"], balanced_train["ja-en"])
    valid_rows = interleave(valid["en-ja"], valid["ja-en"])

    args.output_directory.mkdir(parents=True, exist_ok=True)
    train_path = args.output_directory / "train.jsonl"
    valid_path = args.output_directory / "valid.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(valid_path, valid_rows)
    manifest = {
        "schema_version": 1,
        "operation": "deterministic-balanced-bidirectional-mixture",
        "seed": args.seed,
        "inputs": input_records,
        "protected_benchmark": {
            "path": str(args.protected_benchmark),
            "sha256": expected_protected_hash,
            "maximum_character_trigram_jaccard": args.maximum_jaccard,
            "independently_rescreened": True,
        },
        "license_policy": {
            "allowed": sorted(ALLOWED_LICENSES),
            "counts": dict(sorted(Counter(row["source_license"] for row in train_rows).items())),
            "all_rows_retain_attribution_and_provenance": True,
        },
        "balance_policy": (
            "deterministically shuffle each direction, repeat the smaller train direction "
            "to the larger count, then alternate directions; validation is never repeated"
        ),
        "counts": {
            "train": len(train_rows),
            "valid": len(valid_rows),
            "train_by_direction": {
                direction: len(balanced_train[direction]) for direction in DIRECTIONS
            },
            "valid_by_direction": {
                direction: len(valid[direction]) for direction in DIRECTIONS
            },
            "repeated_train_rows": sum(
                row["balance_repeat_index"] > 0 for row in train_rows
            ),
        },
        "outputs": {
            "train": {"path": str(train_path), "sha256": sha256(train_path)},
            "valid": {"path": str(valid_path), "sha256": sha256(valid_path)},
        },
        "private_chain_of_thought_stored": False,
    }
    (args.output_directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
