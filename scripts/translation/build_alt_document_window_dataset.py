#!/usr/bin/env python3
"""Add coherent licensed ALT document windows to a Marian training dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


DIRECTIONS = {
    "en-ja": ("en-US", "ja-JP"),
    "ja-en": ("ja-JP", "en-US"),
}
SOURCE_ID = re.compile(r"^(?P<document>.+)\.(?P<position>\d+(?:-\d+)*)$")


def rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"missing JSONL input: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def effective_license(row: dict) -> str:
    return str(row.get("source_license") or row.get("license") or "unknown")


def authenticate_dataset(root: Path) -> tuple[dict, Path, Path]:
    manifest_path = root / "manifest.json"
    train_path, valid_path = root / "train.jsonl", root / "valid.jsonl"
    if not manifest_path.is_file():
        raise SystemExit(f"dataset lacks a manifest: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for split, path in (("train", train_path), ("valid", valid_path)):
        if manifest.get("outputs", {}).get(split, {}).get("sha256") != sha256(path):
            raise SystemExit(f"base {split} hash does not match its manifest")
    return manifest, train_path, valid_path


def join_sentences(values: list[str], language: str) -> str:
    separator = "" if language == "ja-JP" else " "
    return separator.join(value.strip() for value in values).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dataset", type=Path)
    parser.add_argument("alt_train", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--direction", choices=tuple(DIRECTIONS), required=True)
    parser.add_argument("--window-size", type=int, action="append", default=[])
    parser.add_argument("--maximum-windows", type=int, default=2000)
    parser.add_argument("--maximum-source-characters", type=int, default=800)
    parser.add_argument("--maximum-target-characters", type=int, default=800)
    parser.add_argument("--seed", default="mimi-alt-document-window-v1")
    args = parser.parse_args()
    window_sizes = sorted(set(args.window_size or [2, 3, 4]))
    if (
        not window_sizes
        or min(window_sizes) < 2
        or args.maximum_windows < 1
        or min(
            args.maximum_source_characters,
            args.maximum_target_characters,
        ) < 1
    ):
        raise SystemExit("window sizes, limits, and maximum-windows must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")

    parent, base_train_path, base_valid_path = authenticate_dataset(args.base_dataset)
    expected = DIRECTIONS[args.direction]
    base_train, base_valid = rows(base_train_path), rows(base_valid_path)
    if any(
        (row.get("source_language"), row.get("target_language")) != expected
        for row in [*base_train, *base_valid]
    ):
        raise SystemExit("base dataset contains the wrong direction")

    grouped: dict[str, list[tuple[tuple[int, ...], str, dict]]] = defaultdict(list)
    rejected = Counter()
    for row in rows(args.alt_train):
        if (row.get("source_language"), row.get("target_language")) != expected:
            continue
        if effective_license(row) != "CC-BY-4.0":
            raise SystemExit(f"ALT row has unexpected license: {row.get('id')}")
        match = SOURCE_ID.match(str(row.get("source_id", "")))
        if match is None:
            rejected["malformed-source-id"] += 1
            continue
        position_text = match.group("position")
        grouped[match.group("document")].append((
            tuple(int(part) for part in position_text.split("-")),
            position_text,
            row,
        ))

    candidates: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for document, members in sorted(grouped.items()):
        members.sort(key=lambda item: item[0])
        for size in window_sizes:
            for start in range(0, len(members) - size + 1, size):
                window = members[start:start + size]
                positions = [position for _, position, _ in window]
                source = join_sentences(
                    [str(row["source"]) for _, _, row in window], expected[0]
                )
                target = join_sentences(
                    [str(row["target"]) for _, _, row in window], expected[1]
                )
                if (
                    len(source) > args.maximum_source_characters
                    or len(target) > args.maximum_target_characters
                ):
                    rejected["character-limit"] += 1
                    continue
                key = (normalized(source), normalized(target))
                if key in seen:
                    rejected["duplicate"] += 1
                    continue
                seen.add(key)
                first, last = window[0][2], window[-1][2]
                candidates.append({
                    "id": (
                        f"alt-window:{document}:{positions[0]}-{positions[-1]}:"
                        f"{args.direction}"
                    ),
                    "source_id": f"{document}.{positions[0]}-{positions[-1]}",
                    "source_language": expected[0],
                    "target_language": expected[1],
                    "source": source,
                    "target": target,
                    "domain": "long-document-news",
                    "origin": "human-alt-document-window",
                    "source_license": "CC-BY-4.0",
                    "source_provenance": (
                        f"NICT Asian Language Treebank document {document}, "
                        f"sentences {positions[0]}-{positions[-1]}"
                    ),
                    "attribution": first["attribution"],
                    "component_source_ids": [
                        str(row["source_id"]) for _, _, row in window
                    ],
                    "window_size": size,
                    "review_status": "licensed-human-reference-window",
                    "training_only": True,
                    "promotion_eligible": False,
                    "component_first_id": first["id"],
                    "component_last_id": last["id"],
                })

    candidates.sort(
        key=lambda row: hashlib.sha256(
            f"{args.seed}\0{args.direction}\0{row['id']}".encode()
        ).hexdigest()
    )
    selected = candidates[:args.maximum_windows]
    if not selected:
        raise SystemExit("no coherent ALT document windows survived")
    output_train = [*base_train, *selected]
    random.Random(f"{args.seed}:{args.direction}").shuffle(output_train)
    output_valid = list(base_valid)
    args.output.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "train": args.output / "train.jsonl",
        "valid": args.output / "valid.jsonl",
    }
    for split, values in (("train", output_train), ("valid", output_valid)):
        output_paths[split].write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in values
            ),
            encoding="utf-8",
        )

    manifest = {
        "schema_version": 1,
        "experiment": "licensed coherent ALT document-window adaptation",
        "promotion_eligible": False,
        "direction": args.direction,
        "target_source": "licensed-human-reference",
        "seed": args.seed,
        "window_sizes": window_sizes,
        "stride": "non-overlapping within each window size",
        "selection": "ascending SHA-256(seed, direction, candidate ID)",
        "counts": {
            "base_train": len(base_train),
            "document_windows": len(selected),
            "train": len(output_train),
            "valid": len(output_valid),
            "rejected": dict(sorted(rejected.items())),
        },
        "origins": {
            split: dict(
                sorted(Counter(str(row.get("origin", "unknown")) for row in values).items())
            )
            for split, values in (("train", output_train), ("valid", output_valid))
        },
        "window_size_counts": dict(
            sorted(Counter(row["window_size"] for row in selected).items())
        ),
        "effective_licenses": {
            split: dict(
                sorted(Counter(effective_license(row) for row in values).items())
            )
            for split, values in (("train", output_train), ("valid", output_valid))
        },
        "inputs": {
            "base_manifest": {
                "path": str((args.base_dataset / "manifest.json").resolve()),
                "sha256": sha256(args.base_dataset / "manifest.json"),
            },
            "base_train": {
                "path": str(base_train_path.resolve()),
                "sha256": sha256(base_train_path),
            },
            "base_valid": {
                "path": str(base_valid_path.resolve()),
                "sha256": sha256(base_valid_path),
            },
            "alt_train": {
                "path": str(args.alt_train.resolve()),
                "sha256": sha256(args.alt_train),
            },
            "parent_target_source": parent.get("target_source"),
        },
        "outputs": {
            split: {"path": str(path.resolve()), "sha256": sha256(path)}
            for split, path in output_paths.items()
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
