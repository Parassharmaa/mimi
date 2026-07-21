#!/usr/bin/env python3
"""Merge strict reviewer-free local-teacher rows into a frozen Marian control."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import unicodedata
from collections import Counter
from pathlib import Path


def rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"missing input: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalized(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def ngrams(text: str, size: int = 5) -> set[str]:
    value = normalized(text)
    if len(value) < size:
        return {value}
    return {value[index:index + size] for index in range(len(value) - size + 1)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def near(text: str, protected: list[set[str]], maximum: float) -> bool:
    candidate = ngrams(text)
    return any(
        len(candidate & value) / max(1, len(candidate | value)) > maximum
        for value in protected
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("accepted_teacher_rows", type=Path)
    parser.add_argument("base_dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--protected-suite", type=Path, action="append", required=True)
    parser.add_argument("--maximum-jaccard", type=float, default=0.8)
    parser.add_argument("--seed", default="mimi-local-teacher-ablation-v1")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")

    protected = [
        ngrams(text)
        for path in args.protected_suite
        for row in rows(path)
        for text in [row.get("source", ""), *row.get("references", [])]
        if str(text).strip()
    ]
    base_train = rows(args.base_dataset / "train.jsonl")
    base_valid = rows(args.base_dataset / "valid.jsonl")
    teacher = rows(args.accepted_teacher_rows)
    valid_texts = {
        normalized(row[field])
        for row in base_valid
        for field in ("source", "target")
    }
    train_sources = {normalized(row["source"]) for row in base_train}
    accepted = []
    rejected: Counter[str] = Counter()
    for row in teacher:
        identifier = str(row.get("id", "")).strip()
        source, target = str(row.get("source", "")).strip(), str(row.get("target", "")).strip()
        judgment = row.get("local_judge", {}).get("judgment")
        if (
            not identifier or not source or not target
            or row.get("promotion_eligible") is not False
            or row.get("review_status") != "local-multimodel-plus-bilingual-judge-provisional"
            or judgment != {
                "adequacy": 5,
                "fluency": 5,
                "meaning_preserved": True,
                "critical_error": False,
                "error_tags": [],
                "verdict": "accept",
            }
        ):
            raise SystemExit(f"teacher row lacks strict provisional evidence: {identifier}")
        if (row.get("source_language"), row.get("target_language")) != ("en-US", "ja-JP"):
            raise SystemExit(f"teacher row has the wrong direction: {identifier}")
        if near(source, protected, args.maximum_jaccard) or near(
            target, protected, args.maximum_jaccard
        ):
            raise SystemExit(f"teacher row is near a protected suite: {identifier}")
        if normalized(source) in train_sources:
            rejected["duplicate-base-source"] += 1
            continue
        if normalized(source) in valid_texts or normalized(target) in valid_texts:
            rejected["validation-overlap"] += 1
            continue
        train_sources.add(normalized(source))
        accepted.append({
            **row,
            "origin": "strict-local-teacher-consensus-provisional",
            "attribution": row["source_provenance"],
            "training_only": True,
        })

    train = [*base_train, *accepted]
    random.Random(args.seed).shuffle(train)
    args.output.mkdir(parents=True, exist_ok=True)
    train_path, valid_path = args.output / "train.jsonl", args.output / "valid.jsonl"
    for path, values in ((train_path, train), (valid_path, base_valid)):
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in values),
            encoding="utf-8",
        )
    manifest = {
        "schema_version": 1,
        "experiment": "strict local-teacher BTEC EN-JA ablation",
        "promotion_eligible": False,
        "seed": args.seed,
        "maximum_protected_five_gram_jaccard": args.maximum_jaccard,
        "counts": {
            "base_train": len(base_train),
            "teacher_input": len(teacher),
            "teacher_accepted": len(accepted),
            "teacher_rejected": dict(sorted(rejected.items())),
            "train": len(train),
            "valid": len(base_valid),
        },
        "origins": {
            "train": dict(sorted(Counter(row.get("origin", "unknown") for row in train).items())),
            "valid": dict(sorted(Counter(row.get("origin", "unknown") for row in base_valid).items())),
        },
        "inputs": {
            "teacher": {"path": str(args.accepted_teacher_rows.resolve()), "sha256": sha256(args.accepted_teacher_rows)},
            "base_train": {"path": str((args.base_dataset / "train.jsonl").resolve()), "sha256": sha256(args.base_dataset / "train.jsonl")},
            "base_valid": {"path": str((args.base_dataset / "valid.jsonl").resolve()), "sha256": sha256(args.base_dataset / "valid.jsonl")},
            "protected_suites": [
                {"path": str(path.resolve()), "sha256": sha256(path)}
                for path in args.protected_suite
            ],
        },
        "outputs": {
            "train": {"path": str(train_path.resolve()), "sha256": sha256(train_path)},
            "valid": {"path": str(valid_path.resolve()), "sha256": sha256(valid_path)},
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
