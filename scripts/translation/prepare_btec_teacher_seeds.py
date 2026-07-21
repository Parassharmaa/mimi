#!/usr/bin/env python3
"""Prepare CC BY 4.0 English BTEC utterances as source-only teacher seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import zipfile
from pathlib import Path


ARCHIVE_SHA256 = "9c0ffaf912cb02eacdff0f3882a2bbcb53a7996af8b2299b6a13b9745c4cb955"
SOURCE_URL = "https://att-astrec.nict.go.jp/en/product/"
ATTRIBUTION = (
    'NICT (2024), "20K English sentences of Basic Travel Expression Corpus (BTEC)", '
    "CC BY 4.0"
)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("protected_benchmark", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--maximum-seeds", type=int, default=300)
    parser.add_argument("--maximum-characters", type=int, default=180)
    parser.add_argument("--maximum-jaccard", type=float, default=0.80)
    parser.add_argument("--selection-seed", default="mimi-btec-teacher-v1")
    args = parser.parse_args()

    if args.output.exists() and args.output.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    if args.maximum_seeds < 1:
        raise SystemExit("maximum-seeds must be positive")
    raw_archive = args.archive.read_bytes()
    digest = hashlib.sha256(raw_archive).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise SystemExit(f"unexpected BTEC archive SHA-256: {digest}")
    protected = load_protected(args.protected_benchmark)
    with zipfile.ZipFile(args.archive) as archive:
        raw_text = archive.read("enBTEC20K.txt")

    candidates: list[dict] = []
    seen: set[str] = set()
    rejected = {"malformed": 0, "language_or_length": 0, "duplicate": 0, "contamination": 0}
    for line_number, line in enumerate(raw_text.decode("utf-8-sig").splitlines(), start=1):
        record_id, separator, utterance = line.partition("@@@@")
        if not separator or not record_id or not utterance:
            rejected["malformed"] += 1
            continue
        record_hash = hashlib.sha256(record_id.encode()).hexdigest()[:16]
        for segment_index, segment in enumerate(utterance.split("|"), start=1):
            source = segment.strip()
            if not (
                3 <= len(source) <= args.maximum_characters
                and len(LATIN_RE.findall(source)) >= 2
                and "http://" not in source
                and "https://" not in source
            ):
                rejected["language_or_length"] += 1
                continue
            source_norm = normalized(source)
            if source_norm in seen:
                rejected["duplicate"] += 1
                continue
            if near_protected(source, protected, args.maximum_jaccard):
                rejected["contamination"] += 1
                continue
            seen.add(source_norm)
            source_id = f"btec:{line_number}:{record_hash}:{segment_index}"
            candidates.append({
                "id": f"teacher-btec:en-ja:{line_number}:{record_hash}:{segment_index}",
                "split": "train",
                "source_language": "en-US",
                "target_language": "ja-JP",
                "domain": "travel-and-service-conversation",
                "source": source,
                "license": "CC-BY-4.0",
                "provenance": f"{SOURCE_URL} / {source_id} / upstream-id={record_id} / {ATTRIBUTION}",
                "selection": "hash-sampled source-only BTEC utterance for reviewed teacher translation",
            })

    candidates.sort(
        key=lambda row: hashlib.sha256(
            f"{args.selection_seed}\0{row['id']}\0{normalized(row['source'])}".encode()
        ).digest()
    )
    selected = candidates[: args.maximum_seeds]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "source": SOURCE_URL,
        "archive_sha256": ARCHIVE_SHA256,
        "member_sha256": hashlib.sha256(raw_text).hexdigest(),
        "license": "CC-BY-4.0",
        "attribution": ATTRIBUTION,
        "quality": "NICT Basic Travel Expression Corpus; English source-only utterances",
        "parallel_gold": False,
        "teacher_target_requires_independent_bilingual_selection": True,
        "eligible_unique_utterances": len(candidates),
        "selected": len(selected),
        "direction": "en-ja",
        "selection_seed": args.selection_seed,
        "protected_benchmark": str(args.protected_benchmark),
        "maximum_jaccard": args.maximum_jaccard,
        "rejected": rejected,
        "output": str(args.output),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
