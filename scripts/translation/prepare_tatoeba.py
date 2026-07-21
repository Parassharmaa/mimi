#!/usr/bin/env python3
"""Create attribution-preserving, benchmark-deduplicated MLX LoRA data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import zipfile
from pathlib import Path


ID_RE = re.compile(r"#(\d+)")
JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]")


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def ngrams(text: str, size: int = 5) -> set[str]:
    text = normalized(text).replace(" ", "")
    return {text[index:index + size] for index in range(max(1, len(text) - size + 1))}


def jaccard(left: str, right: str) -> float:
    a, b = ngrams(left), ngrams(right)
    return len(a & b) / max(1, len(a | b))


def benchmark_text(path: Path) -> list[str]:
    result: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        result.extend([row["source"], *row.get("references", [])])
    return result


def split_for(source_id: str) -> str:
    bucket = int(hashlib.sha256(source_id.encode()).hexdigest()[:8], 16) % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "valid"
    return "test"


def chat_row(source: str, target: str, direction: str, metadata: dict) -> dict:
    instruction = {
        "en-ja": "Translate this English live-transcript segment into natural Japanese. Output only the translation.",
        "ja-en": "Translate this Japanese live-transcript segment into natural English. Output only the translation.",
    }[direction]
    return {
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": source},
            {"role": "assistant", "content": target},
        ],
        "metadata": {**metadata, "direction": direction},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path, help="ManyThings jpn-eng.zip")
    parser.add_argument("protected_benchmark", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--maximum-pairs", type=int, default=40000)
    parser.add_argument("--maximum-jaccard", type=float, default=0.80)
    parser.add_argument(
        "--sampling-modulus",
        type=int,
        default=8,
        help="Keep one stable hash bucket out of N before expensive contamination checks.",
    )
    args = parser.parse_args()

    protected = benchmark_text(args.protected_benchmark)
    pairs: list[tuple[str, str, str, str]] = []
    seen_pairs: set[str] = set()
    with zipfile.ZipFile(args.archive) as archive:
        lines = archive.read("jpn.txt").decode("utf-8").splitlines()
    for line in lines:
        columns = line.split("\t")
        if len(columns) != 3:
            continue
        english, japanese, attribution = (column.strip() for column in columns)
        if not (3 <= len(english) <= 240 and 2 <= len(japanese) <= 180):
            continue
        if len(LATIN_RE.findall(english)) < 2 or len(JAPANESE_RE.findall(japanese)) < 2:
            continue
        if "http://" in english or "https://" in english or "http://" in japanese or "https://" in japanese:
            continue
        pair_hash = hashlib.sha256(f"{normalized(english)}\0{normalized(japanese)}".encode()).hexdigest()
        if pair_hash in seen_pairs:
            continue
        if int(pair_hash[:8], 16) % max(1, args.sampling_modulus) != 0:
            continue
        if any(
            jaccard(text, protected_text) > args.maximum_jaccard
            for text in (english, japanese)
            for protected_text in protected
        ):
            continue
        sentence_ids = ID_RE.findall(attribution)
        if len(sentence_ids) < 2:
            continue
        seen_pairs.add(pair_hash)
        pairs.append((english, japanese, attribution, sentence_ids[0]))
        if len(pairs) >= args.maximum_pairs:
            break

    # Stable hash order avoids source-order and random-library drift.
    pairs.sort(key=lambda pair: hashlib.sha256(f"{pair[3]}\0{pair[0]}\0{pair[1]}".encode()).hexdigest())
    output: dict[str, list[dict]] = {"train": [], "valid": [], "test": []}
    for english, japanese, attribution, source_id in pairs:
        split = split_for(source_id)
        metadata = {
            "source": "Tatoeba via ManyThings",
            "source_id": source_id,
            "license": "CC-BY-2.0-FR",
            "attribution": attribution,
        }
        output[split].append(chat_row(english, japanese, "en-ja", metadata))
        output[split].append(chat_row(japanese, english, "ja-en", metadata))

    args.output_directory.mkdir(parents=True, exist_ok=True)
    for split, split_rows in output.items():
        (args.output_directory / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in split_rows),
            encoding="utf-8",
        )
    (args.output_directory / "manifest.json").write_text(json.dumps({
        "source": "https://www.manythings.org/anki/",
        "upstream": "https://tatoeba.org/",
        "license": "CC-BY-2.0-FR per row",
        "unique_pairs": len(pairs),
        "examples": {split: len(split_rows) for split, split_rows in output.items()},
        "split_key": "English Tatoeba sentence ID",
        "protected_benchmark": str(args.protected_benchmark),
        "maximum_jaccard": args.maximum_jaccard,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({split: len(split_rows) for split, split_rows in output.items()}))


if __name__ == "__main__":
    main()
