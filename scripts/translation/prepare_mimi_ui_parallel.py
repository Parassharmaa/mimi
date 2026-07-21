#!/usr/bin/env python3
"""Extract project-owned English/Japanese UI pairs from Mimi's Swift sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import unicodedata
from collections import defaultdict
from pathlib import Path


CALL_RE = re.compile(r"preferences\.text\s*\(|(?<![A-Za-z0-9_.])t\s*\(")
JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]")


def load_rows(path: Path) -> list[dict]:
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


def jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / max(1, len(left | right))


def skip_whitespace(source: str, position: int) -> int:
    while position < len(source) and source[position].isspace():
        position += 1
    return position


def parse_swift_string(source: str, position: int) -> tuple[str, int] | None:
    position = skip_whitespace(source, position)
    if not source.startswith('"', position) or source.startswith('"""', position):
        return None
    start = position
    position += 1
    escaped = False
    while position < len(source):
        character = source[position]
        if character == '"' and not escaped:
            raw = source[start:position + 1]
            if "\\(" in raw:
                return None
            try:
                return json.loads(raw), position + 1
            except json.JSONDecodeError:
                return None
        if character == "\\" and not escaped:
            escaped = True
        else:
            escaped = False
        position += 1
    return None


def pairs(path: Path) -> list[tuple[int, str, str]]:
    source = path.read_text(encoding="utf-8")
    output: list[tuple[int, str, str]] = []
    for match in CALL_RE.finditer(source):
        first = parse_swift_string(source, match.end())
        if first is None:
            continue
        english, position = first
        position = skip_whitespace(source, position)
        if position >= len(source) or source[position] != ",":
            continue
        second = parse_swift_string(source, position + 1)
        if second is None:
            continue
        japanese, _ = second
        output.append((source.count("\n", 0, match.start()) + 1, english.strip(), japanese.strip()))
    return output


def revision(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def split_for(path: Path, seed: str, validation_fraction: float) -> str:
    digest = hashlib.sha256(f"{seed}\0{path.as_posix()}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(2**64)
    return "valid" if bucket < validation_fraction else "train"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_directory", type=Path)
    parser.add_argument("protected_benchmark", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--split-seed", default="mimi-ui-parallel-v1")
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--minimum-english-characters", type=int, default=8)
    parser.add_argument("--maximum-english-characters", type=int, default=240)
    parser.add_argument("--maximum-japanese-characters", type=int, default=160)
    parser.add_argument("--maximum-jaccard", type=float, default=0.80)
    args = parser.parse_args()

    if not 0 < args.validation_fraction < 0.5:
        raise SystemExit("validation-fraction must be between 0 and 0.5")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_directory}")
    source_directory = args.source_directory.resolve()
    root = source_directory.parent.parent
    protected = [
        ngrams(text)
        for row in load_rows(args.protected_benchmark)
        for text in (row["source"], *row.get("references", []))
    ]
    output: dict[str, list[dict]] = {"train": [], "valid": []}
    eligible_pairs: list[tuple[str, Path, int, str, str]] = []
    seen: set[tuple[str, str]] = set()
    rejects = {
        "language-or-length": 0,
        "duplicate": 0,
        "contamination": 0,
        "ambiguous-source": 0,
    }
    repository_revision = revision(root)

    for swift_path in sorted(source_directory.rglob("*.swift")):
        relative = swift_path.relative_to(root)
        split = split_for(relative, args.split_seed, args.validation_fraction)
        for line, english, japanese in pairs(swift_path):
            if not (
                args.minimum_english_characters <= len(english) <= args.maximum_english_characters
                and 2 <= len(japanese) <= args.maximum_japanese_characters
                and LATIN_RE.search(english)
                and JAPANESE_RE.search(japanese)
            ):
                rejects["language-or-length"] += 1
                continue
            key = (normalized(english), normalized(japanese))
            if key in seen:
                rejects["duplicate"] += 1
                continue
            if any(
                jaccard(candidate, heldout) > args.maximum_jaccard
                for candidate in (ngrams(english), ngrams(japanese))
                for heldout in protected
            ):
                rejects["contamination"] += 1
                continue
            seen.add(key)
            eligible_pairs.append((split, relative, line, english, japanese))

    english_targets: dict[str, set[str]] = defaultdict(set)
    japanese_targets: dict[str, set[str]] = defaultdict(set)
    for _, _, _, english, japanese in eligible_pairs:
        english_targets[normalized(english)].add(normalized(japanese))
        japanese_targets[normalized(japanese)].add(normalized(english))

    for split, relative, line, english, japanese in eligible_pairs:
        if (
            len(english_targets[normalized(english)]) != 1
            or len(japanese_targets[normalized(japanese)]) != 1
        ):
            rejects["ambiguous-source"] += 1
            continue
        pair_id = hashlib.sha256(
            f"{relative.as_posix()}:{line}\0{english}\0{japanese}".encode()
        ).hexdigest()[:20]
        common = {
            "source_id": f"mimi-ui:{relative.as_posix()}:{line}",
            "domain": "mimi-product-ui",
            "origin": "mimi-shipped-ui-pair",
            "source_license": "project-owned",
            "source_provenance": f"{relative.as_posix()}:{line}@{repository_revision}",
            "review_status": "shipping-product-copy",
        }
        output[split].extend(
            [
                {
                    **common,
                    "id": f"mimi-ui:{pair_id}:en-ja",
                    "source_language": "en-US",
                    "target_language": "ja-JP",
                    "source": english,
                    "target": japanese,
                },
                {
                    **common,
                    "id": f"mimi-ui:{pair_id}:ja-en",
                    "source_language": "ja-JP",
                    "target_language": "en-US",
                    "source": japanese,
                    "target": english,
                },
            ]
        )

    if not output["train"] or not output["valid"]:
        raise SystemExit("source-file split produced an empty train or validation set")
    args.output_directory.mkdir(parents=True, exist_ok=True)
    for split, rows in output.items():
        write_jsonl(args.output_directory / f"{split}.jsonl", rows)
    manifest = {
        "schema_version": 1,
        "source": str(args.source_directory),
        "repository_revision": repository_revision,
        "license": "project-owned",
        "quality": "paired English/Japanese copy already shipped in Mimi's UI",
        "split_policy": "source-file grouped deterministic hash",
        "split_seed": args.split_seed,
        "validation_fraction": args.validation_fraction,
        "protected_benchmark": str(args.protected_benchmark),
        "maximum_jaccard": args.maximum_jaccard,
        "counts": {split: len(rows) for split, rows in output.items()},
        "pairs": {split: len(rows) // 2 for split, rows in output.items()},
        "rejected": rejects,
    }
    (args.output_directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
