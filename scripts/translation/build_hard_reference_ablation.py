#!/usr/bin/env python3
"""Build a licensed human-reference baseline from hard student examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import unicodedata
from collections import Counter
from pathlib import Path


LANGUAGES = {
    "en-ja": ("en-US", "ja-JP"),
    "ja-en": ("ja-JP", "en-US"),
}


def rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def ngrams(text: str, size: int = 5) -> set[str]:
    value = normalized(text).replace(" ", "")
    return {value[index:index + size] for index in range(max(1, len(value) - size + 1))}


def near(text: str, protected: list[set[str]], maximum: float) -> bool:
    candidate = ngrams(text)
    return any(
        len(candidate & heldout) / max(1, len(candidate | heldout)) > maximum
        for heldout in protected
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ranked(values: list[dict], count: int, seed: str) -> list[dict]:
    return sorted(
        values,
        key=lambda row: hashlib.sha256(f"{seed}\0{row['id']}".encode()).digest(),
    )[:count]


def kftt(path: Path, direction: str, origin: str) -> list[dict]:
    source_language, target_language = LANGUAGES[direction]
    output: list[dict] = []
    for row in rows(path):
        metadata, messages = row["metadata"], row["messages"]
        if metadata["direction"] != direction:
            continue
        output.append(
            {
                "id": f"{origin}:{metadata['source_id']}:{direction}",
                "source_id": metadata["source_id"],
                "source_language": source_language,
                "target_language": target_language,
                "source": messages[1]["content"].strip(),
                "target": messages[2]["content"].strip(),
                "domain": "wikipedia",
                "origin": origin,
                "source_license": metadata["license"],
                "source_provenance": metadata["source"],
                "attribution": metadata["attribution"],
            }
        )
    return output


def ui(path: Path, direction: str) -> list[dict]:
    expected = LANGUAGES[direction]
    return [
        row
        for row in rows(path)
        if (row.get("source_language"), row.get("target_language")) == expected
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("hard_seeds", type=Path)
    parser.add_argument("kftt_directory", type=Path)
    parser.add_argument("mimi_ui_directory", type=Path)
    parser.add_argument("protected_benchmark", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--direction", choices=tuple(LANGUAGES), required=True)
    parser.add_argument("--kftt-replay", type=int, default=1800)
    parser.add_argument("--kftt-validation", type=int, default=400)
    parser.add_argument("--maximum-jaccard", type=float, default=0.80)
    parser.add_argument("--seed", default="mimi-hard-reference-ablation-v1")
    args = parser.parse_args()

    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_directory}")
    expected = LANGUAGES[args.direction]
    protected = [
        ngrams(text)
        for row in rows(args.protected_benchmark)
        for text in (row["source"], *row.get("references", []))
    ]
    hard: list[dict] = []
    for seed in rows(args.hard_seeds):
        if (seed.get("source_language"), seed.get("target_language")) != expected:
            continue
        source, target = str(seed["source"]).strip(), str(seed["reference_translation"]).strip()
        if near(source, protected, args.maximum_jaccard) or near(target, protected, args.maximum_jaccard):
            raise SystemExit(f"hard seed is near protected benchmark: {seed['id']}")
        hard.append(
            {
                "id": f"hard-reference:{seed['id']}",
                "source_id": seed["id"],
                "source_language": expected[0],
                "target_language": expected[1],
                "source": source,
                "target": target,
                "domain": seed["domain"],
                "origin": "licensed-human-hard-reference",
                "source_license": seed["license"],
                "source_provenance": seed["provenance"],
                "student_chrf_pp": seed.get("student_chrf_pp"),
            }
        )

    hard_values = {normalized(row[field]) for row in hard for field in ("source", "target")}
    replay_pool = [
        row
        for row in kftt(args.kftt_directory / "train.jsonl", args.direction, "human-kftt-replay")
        if normalized(row["source"]) not in hard_values and normalized(row["target"]) not in hard_values
    ]
    replay = ranked(replay_pool, args.kftt_replay, f"{args.seed}:replay:{args.direction}")
    valid_kftt = ranked(
        kftt(args.kftt_directory / "valid.jsonl", args.direction, "human-kftt-validation"),
        args.kftt_validation,
        f"{args.seed}:valid:{args.direction}",
    )
    valid_ui = ui(args.mimi_ui_directory / "valid.jsonl", args.direction)
    train_before_overlap = hard + replay
    valid = valid_kftt + valid_ui
    valid_values = {normalized(row[field]) for row in valid for field in ("source", "target")}
    train = [
        row
        for row in train_before_overlap
        if normalized(row["source"]) not in valid_values and normalized(row["target"]) not in valid_values
    ]
    overlap_removed = len(train_before_overlap) - len(train)
    for split_name, values in (("train", train), ("valid", valid)):
        for row in values:
            if near(row["source"], protected, args.maximum_jaccard) or near(
                row["target"], protected, args.maximum_jaccard
            ):
                raise SystemExit(f"{split_name} row is near protected benchmark: {row['id']}")
    random.Random(f"{args.seed}:train:{args.direction}").shuffle(train)
    random.Random(f"{args.seed}:valid:{args.direction}").shuffle(valid)

    args.output_directory.mkdir(parents=True, exist_ok=True)
    train_path, valid_path = args.output_directory / "train.jsonl", args.output_directory / "valid.jsonl"
    for path, values in ((train_path, train), (valid_path, valid)):
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in values),
            encoding="utf-8",
        )
    manifest = {
        "schema_version": 1,
        "experiment": "licensed human-reference hard-example ablation; no GPT output",
        "direction": args.direction,
        "seed": args.seed,
        "maximum_protected_jaccard": args.maximum_jaccard,
        "counts": {
            "hard_reference_requested": len(hard),
            "kftt_replay_requested": len(replay),
            "cross_split_overlap_removed": overlap_removed,
            "train": len(train),
            "valid": len(valid),
        },
        "origins": {
            "train": dict(Counter(row["origin"] for row in train)),
            "valid": dict(Counter(row["origin"] for row in valid)),
        },
        "inputs": {
            "hard_seeds": {"path": str(args.hard_seeds), "sha256": sha256(args.hard_seeds)},
            "kftt_train": {
                "path": str(args.kftt_directory / "train.jsonl"),
                "sha256": sha256(args.kftt_directory / "train.jsonl"),
            },
            "kftt_valid": {
                "path": str(args.kftt_directory / "valid.jsonl"),
                "sha256": sha256(args.kftt_directory / "valid.jsonl"),
            },
            "mimi_ui_valid": {
                "path": str(args.mimi_ui_directory / "valid.jsonl"),
                "sha256": sha256(args.mimi_ui_directory / "valid.jsonl"),
            },
            "protected_benchmark": {
                "path": str(args.protected_benchmark),
                "sha256": sha256(args.protected_benchmark),
            },
        },
        "outputs": {
            "train": {"path": str(train_path), "sha256": sha256(train_path)},
            "valid": {"path": str(valid_path), "sha256": sha256(valid_path)},
        },
    }
    (args.output_directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
