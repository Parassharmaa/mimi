#!/usr/bin/env python3
"""Prepare the professionally translated KFTT corpus for bidirectional LoRA."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
import unicodedata
from pathlib import Path


ARCHIVE_SHA256 = "fcfcaa670d6d59aa691b0e909c0d7c393852dd2fb1d6310fda9b3282dc6d1638"
SOURCE_URL = "https://www.phontron.com/kftt/download/kftt-data-1.0.tar.gz"
ATTRIBUTION = (
    "English contents translated by NICT from Japanese Wikipedia; "
    "CC-BY-SA-3.0; https://alaginrc.nict.go.jp/WikiCorpus/"
)
JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]")


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def ngrams(text: str, size: int = 5) -> set[str]:
    value = normalized(text).replace(" ", "")
    return {value[index:index + size] for index in range(max(1, len(value) - size + 1))}


def jaccard(left: str, right: str) -> float:
    a, b = ngrams(left), ngrams(right)
    return len(a & b) / max(1, len(a | b))


def protected_text(path: Path) -> list[str]:
    texts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            texts.extend([row["source"], *row.get("references", [])])
    return texts


def chat_row(source: str, target: str, direction: str, source_id: str) -> dict:
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
        "metadata": {
            "source": "Kyoto Free Translation Task 1.0",
            "source_id": source_id,
            "license": "CC-BY-SA-3.0",
            "attribution": ATTRIBUTION,
            "direction": direction,
        },
    }


def read_member(archive: tarfile.TarFile, name: str) -> list[str]:
    member = archive.extractfile(f"kftt-data-1.0/data/orig/{name}")
    if member is None:
        raise SystemExit(f"archive is missing {name}")
    return member.read().decode("utf-8").splitlines()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("protected_benchmark", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--maximum-jaccard", type=float, default=0.80)
    parser.add_argument("--maximum-english-characters", type=int, default=220)
    parser.add_argument("--maximum-japanese-characters", type=int, default=60)
    args = parser.parse_args()

    digest = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise SystemExit(f"unexpected KFTT archive SHA-256: {digest}")

    protected = [ngrams(text) for text in protected_text(args.protected_benchmark)]
    source_splits = {"train": "train", "valid": "tune", "test": "test"}
    output: dict[str, list[dict]] = {split: [] for split in source_splits}
    rejected = {"empty_or_language": 0, "length": 0, "duplicate": 0, "contamination": 0}
    seen: set[str] = set()

    with tarfile.open(args.archive, "r:gz") as archive:
        for output_split, source_split in source_splits.items():
            english = read_member(archive, f"kyoto-{source_split}.en")
            japanese = read_member(archive, f"kyoto-{source_split}.ja")
            if len(english) != len(japanese):
                raise SystemExit(f"unaligned KFTT split: {source_split}")
            for index, (en, ja) in enumerate(zip(english, japanese), start=1):
                en, ja = en.strip(), ja.strip()
                if not en or not ja or not LATIN_RE.search(en) or not JAPANESE_RE.search(ja):
                    rejected["empty_or_language"] += 1
                    continue
                if not (
                    3 <= len(en) <= args.maximum_english_characters
                    and 2 <= len(ja) <= args.maximum_japanese_characters
                ):
                    rejected["length"] += 1
                    continue
                pair_hash = hashlib.sha256(f"{normalized(en)}\0{normalized(ja)}".encode()).hexdigest()
                if pair_hash in seen:
                    rejected["duplicate"] += 1
                    continue
                candidate_ngrams = (ngrams(en), ngrams(ja))
                if any(
                    len(candidate & heldout) / max(1, len(candidate | heldout)) > args.maximum_jaccard
                    for candidate in candidate_ngrams
                    for heldout in protected
                ):
                    rejected["contamination"] += 1
                    continue
                seen.add(pair_hash)
                source_id = f"kftt-{source_split}-{index:06d}"
                output[output_split].append(chat_row(en, ja, "en-ja", source_id))
                output[output_split].append(chat_row(ja, en, "ja-en", source_id))

    args.output_directory.mkdir(parents=True, exist_ok=True)
    for split, rows in output.items():
        (args.output_directory / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    manifest = {
        "source": SOURCE_URL,
        "archive_sha256": ARCHIVE_SHA256,
        "license": "CC-BY-SA-3.0",
        "attribution": ATTRIBUTION,
        "quality": "Japanese-English Wikipedia sentences translated and checked by professional translators",
        "examples": {split: len(rows) for split, rows in output.items()},
        "source_splits": source_splits,
        "protected_benchmark": str(args.protected_benchmark),
        "maximum_jaccard": args.maximum_jaccard,
        "maximum_english_characters": args.maximum_english_characters,
        "maximum_japanese_characters": args.maximum_japanese_characters,
        "length_policy": "Short live-caption pairs chosen to avoid tokenizer truncation at 256 tokens",
        "rejected": rejected,
    }
    (args.output_directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
