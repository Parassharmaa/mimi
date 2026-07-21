#!/usr/bin/env python3
"""Prepare NICT's human-translated ALT English/Japanese parallel corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import zipfile
from pathlib import Path


ARCHIVE_SHA256 = "05f7b31b517d4c4e074bb7fb57277758c0e3e15d1ad9cfc5727e9bce79b07bbd"
SOURCE_URL = "https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/"
ARCHIVE_ROOT = "ALT-Parallel-Corpus-20191206"
ATTRIBUTION = (
    "NICT Asian Language Treebank Parallel Corpus; NICT translations CC BY 4.0; "
    "English Wikinews source text CC BY 2.5; cite Riza et al. (2016), "
    "Introduction of the Asian Language Treebank"
)
SENTENCE_ID_RE = re.compile(r"^SNT\.(\d+)\.(\d+(?:-\d*)?)$")
JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]")


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def ngrams(text: str, size: int = 5) -> set[str]:
    value = normalized(text).replace(" ", "")
    return {value[index:index + size] for index in range(max(1, len(value) - size + 1))}


def near_protected(text: str, protected: list[set[str]], threshold: float) -> bool:
    candidate = ngrams(text)
    return any(
        len(candidate & heldout) / max(1, len(candidate | heldout)) > threshold
        for heldout in protected
    )


def load_protected(path: Path) -> list[set[str]]:
    output: list[set[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        output.extend(ngrams(text) for text in (row["source"], *row.get("references", [])))
    return output


def member_bytes(archive: zipfile.ZipFile, name: str) -> bytes:
    full_name = f"{ARCHIVE_ROOT}/{name}"
    try:
        return archive.read(full_name)
    except KeyError as error:
        raise SystemExit(f"ALT archive is missing {full_name}") from error


def text_map(raw: bytes, label: str) -> tuple[dict[str, str], set[str]]:
    output: dict[str, str] = {}
    duplicates: set[str] = set()
    for line_number, line in enumerate(raw.decode("utf-8-sig").splitlines(), start=1):
        sentence_id, separator, text = line.partition("\t")
        if not SENTENCE_ID_RE.fullmatch(sentence_id):
            raise SystemExit(f"invalid {label} line {line_number}: {line[:80]!r}")
        if not separator:
            text = ""
        if sentence_id in output:
            duplicates.add(sentence_id)
            continue
        output[sentence_id] = text.strip()
    return output, duplicates


def url_map(raw: bytes) -> dict[str, str]:
    output: dict[str, str] = {}
    for line_number, line in enumerate(raw.decode("utf-8-sig").splitlines(), start=1):
        identifier, separator, url = line.partition("\t")
        if not separator or not identifier.startswith("URL.") or not url:
            raise SystemExit(f"invalid ALT URL line {line_number}")
        output[identifier.removeprefix("URL.")] = url.strip()
    return output


def split_for(document_id: str, seed: str) -> str:
    bucket = int(hashlib.sha256(f"{seed}\0{document_id}".encode()).hexdigest()[:8], 16) % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "valid"
    return "test"


def parallel_row(
    source: str,
    target: str,
    direction: str,
    sentence_id: str,
    source_url: str,
) -> dict:
    source_language, target_language = {
        "en-ja": ("en-US", "ja-JP"),
        "ja-en": ("ja-JP", "en-US"),
    }[direction]
    return {
        "id": f"alt:{sentence_id}:{direction}",
        "source_id": sentence_id,
        "source_language": source_language,
        "target_language": target_language,
        "source": source,
        "target": target,
        "domain": "human-translated-news",
        "origin": "human-alt-parallel",
        "source_license": "CC-BY-4.0",
        "source_provenance": f"{SOURCE_URL} / {sentence_id} / {source_url}",
        "attribution": ATTRIBUTION,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("protected_benchmark", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--maximum-jaccard", type=float, default=0.80)
    parser.add_argument("--maximum-english-characters", type=int, default=180)
    parser.add_argument("--maximum-japanese-characters", type=int, default=100)
    parser.add_argument("--split-seed", default="mimi-alt-v1")
    args = parser.parse_args()

    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_directory}")
    archive_raw = args.archive.read_bytes()
    digest = hashlib.sha256(archive_raw).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise SystemExit(f"unexpected ALT archive SHA-256: {digest}")
    protected = load_protected(args.protected_benchmark)

    with zipfile.ZipFile(args.archive) as archive:
        english_raw = member_bytes(archive, "data_en.txt")
        japanese_raw = member_bytes(archive, "data_ja.txt")
        urls_raw = member_bytes(archive, "URL.txt")
    english, english_duplicate_ids = text_map(english_raw, "English")
    japanese, japanese_duplicate_ids = text_map(japanese_raw, "Japanese")
    ambiguous_ids = (
        english_duplicate_ids
        | japanese_duplicate_ids
        | (english.keys() ^ japanese.keys())
    )
    urls = url_map(urls_raw)

    output: dict[str, list[dict]] = {"train": [], "valid": [], "test": []}
    rejected = {
        "ambiguous_alignment": 0,
        "empty_or_language": 0,
        "length": 0,
        "duplicate": 0,
        "contamination": 0,
    }
    seen_pairs: set[str] = set()
    for sentence_id in english:
        match = SENTENCE_ID_RE.fullmatch(sentence_id)
        assert match is not None
        document_id = match.group(1)
        if sentence_id in ambiguous_ids:
            rejected["ambiguous_alignment"] += 1
            continue
        if sentence_id not in japanese:
            rejected["ambiguous_alignment"] += 1
            continue
        en, ja = english[sentence_id], japanese[sentence_id]
        if not en or not ja or len(LATIN_RE.findall(en)) < 2 or len(JAPANESE_RE.findall(ja)) < 2:
            rejected["empty_or_language"] += 1
            continue
        if not (
            3 <= len(en) <= args.maximum_english_characters
            and 2 <= len(ja) <= args.maximum_japanese_characters
        ):
            rejected["length"] += 1
            continue
        pair_hash = hashlib.sha256(f"{normalized(en)}\0{normalized(ja)}".encode()).hexdigest()
        if pair_hash in seen_pairs:
            rejected["duplicate"] += 1
            continue
        if near_protected(en, protected, args.maximum_jaccard) or near_protected(
            ja, protected, args.maximum_jaccard
        ):
            rejected["contamination"] += 1
            continue
        source_url = urls.get(document_id)
        if source_url is None:
            raise SystemExit(f"ALT sentence has no source URL: {sentence_id}")
        seen_pairs.add(pair_hash)
        split = split_for(document_id, args.split_seed)
        output[split].append(parallel_row(en, ja, "en-ja", sentence_id, source_url))
        output[split].append(parallel_row(ja, en, "ja-en", sentence_id, source_url))

    args.output_directory.mkdir(parents=True, exist_ok=True)
    for split, rows in output.items():
        path = args.output_directory / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    manifest = {
        "schema_version": 1,
        "source": SOURCE_URL,
        "archive_sha256": ARCHIVE_SHA256,
        "member_sha256": {
            "data_en.txt": hashlib.sha256(english_raw).hexdigest(),
            "data_ja.txt": hashlib.sha256(japanese_raw).hexdigest(),
            "URL.txt": hashlib.sha256(urls_raw).hexdigest(),
        },
        "license": "CC-BY-4.0; English Wikinews source text CC-BY-2.5",
        "attribution": ATTRIBUTION,
        "quality": "NICT human-translated English Wikinews parallel text",
        "document_grouped_split": True,
        "split_seed": args.split_seed,
        "examples": {split: len(rows) for split, rows in output.items()},
        "unique_pairs": len(seen_pairs),
        "protected_benchmark": str(args.protected_benchmark),
        "maximum_jaccard": args.maximum_jaccard,
        "maximum_english_characters": args.maximum_english_characters,
        "maximum_japanese_characters": args.maximum_japanese_characters,
        "rejected": rejected,
    }
    (args.output_directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
