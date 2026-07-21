#!/usr/bin/env python3
"""Prepare finalized Japanese Law Translation TMX files for Mimi training."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SOURCE_ROOT = "https://www.japaneselawtranslation.go.jp/en/laws"
TERMS_URL = "https://www.japaneselawtranslation.go.jp/en/index/terms"
PDL_URL = "https://www.digital.go.jp/en/resources/open_data/public_data_license_v1.0"
LICENSE = "PDL-1.0-compatible-CC-BY-4.0"
ATTRIBUTION = (
    "Created by Mimi based on finalized Japanese Law Translation Database System "
    "content published by the Ministry of Justice, Japan; PDL 1.0; source content "
    "was filtered, normalized, and converted to parallel training rows by Mimi"
)
JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]")
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def normalized_hash(text: str) -> str:
    return sha256_bytes(normalized(text).encode("utf-8"))


def ngrams(text: str, size: int = 5) -> set[str]:
    value = normalized(text).replace(" ", "")
    return {
        value[index:index + size]
        for index in range(max(1, len(value) - size + 1))
    }


class NgramIndex:
    def __init__(self, texts: Iterable[str]) -> None:
        self.values = [ngrams(text) for text in texts if text.strip()]
        postings: dict[str, list[int]] = defaultdict(list)
        for index, grams in enumerate(self.values):
            for gram in grams:
                postings[gram].append(index)
        self.postings = dict(postings)

    def matches(self, text: str, threshold: float) -> bool:
        candidate = ngrams(text)
        intersections: Counter[int] = Counter()
        for gram in candidate:
            intersections.update(self.postings.get(gram, ()))
        return any(
            overlap / max(1, len(candidate) + len(self.values[index]) - overlap)
            > threshold
            for index, overlap in intersections.items()
        )


def load_protected(paths: Iterable[Path]) -> NgramIndex:
    protected: list[str] = []
    for path in paths:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(
                    f"invalid protected JSONL {path}:{line_number}: {error}"
                ) from error
            texts = (row.get("source"), *row.get("references", []))
            protected.extend(text for text in texts if isinstance(text, str))
    return NgramIndex(protected)


def element_text(element: ET.Element) -> str:
    return " ".join("".join(element.itertext()).split())


def parse_tmx(raw: bytes, label: str) -> tuple[dict[str, str], list[dict[str, str]]]:
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as error:
        raise SystemExit(f"invalid TMX {label}: {error}") from error
    if root.tag != "tmx" or root.attrib.get("version") != "1.4":
        raise SystemExit(f"unexpected TMX root in {label}")
    header = root.find("header")
    body = root.find("body")
    if header is None or body is None:
        raise SystemExit(f"TMX lacks header/body in {label}")
    if header.attrib.get("srclang") != "ja-JP":
        raise SystemExit(f"TMX source language is not ja-JP in {label}")

    pairs: list[dict[str, str]] = []
    for index, unit in enumerate(body.findall("tu"), start=1):
        by_language: dict[str, str] = {}
        for variant in unit.findall("tuv"):
            language = variant.attrib.get(XML_LANG)
            segment = variant.find("seg")
            if language and segment is not None:
                by_language[language] = element_text(segment)
        pairs.append(
            {
                "tuid": unit.attrib.get("tuid", str(index)),
                "ja": by_language.get("ja-JP", ""),
                "en": by_language.get("en-US", ""),
            }
        )
    return dict(header.attrib), pairs


def split_for(law_id: str, seed: str) -> str:
    bucket = int(hashlib.sha256(f"{seed}\0{law_id}".encode()).hexdigest()[:8], 16) % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "valid"
    return "test"


def law_id_for(path: Path) -> str:
    match = re.search(r"(?:^|[-_])(\d{2,})(?:[-_.]|$)", path.stem)
    return match.group(1) if match else path.stem


def valid_language_pair(en: str, ja: str) -> bool:
    return len(LATIN_RE.findall(en)) >= 2 and len(JAPANESE_RE.findall(ja)) >= 2


def parallel_row(
    *,
    source: str,
    target: str,
    direction: str,
    law_id: str,
    tuid: str,
    tmx_sha256: str,
) -> dict[str, Any]:
    source_language, target_language = {
        "en-ja": ("en-US", "ja-JP"),
        "ja-en": ("ja-JP", "en-US"),
    }[direction]
    source_id = f"law-{law_id}:tu-{tuid}"
    return {
        "id": f"jlt:{source_id}:{direction}",
        "source_id": source_id,
        "source_language": source_language,
        "target_language": target_language,
        "source": source,
        "target": target,
        "source_normalized_sha256": normalized_hash(source),
        "target_normalized_sha256": normalized_hash(target),
        "domain": "ministry-published-legal",
        "origin": "finalized-japanese-law-translation",
        "source_license": LICENSE,
        "source_provenance": f"{SOURCE_ROOT}/view/{law_id} / TMX unit {tuid}",
        "source_tmx_sha256": tmx_sha256,
        "translation_status": "finalized",
        "attribution": ATTRIBUTION,
        "training_only": True,
        "promotion_eligible": False,
    }


def prepare(
    tmx_paths: list[Path],
    protected_paths: list[Path],
    *,
    maximum_jaccard: float,
    maximum_english_characters: int,
    maximum_japanese_characters: int,
    split_seed: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    protected_index = load_protected(protected_paths)
    output: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    rejected: Counter[str] = Counter()
    seen_pairs: set[str] = set()
    inputs: list[dict[str, Any]] = []

    for path in sorted(tmx_paths):
        raw = path.read_bytes()
        digest = sha256_bytes(raw)
        law_id = law_id_for(path)
        header, pairs = parse_tmx(raw, str(path))
        inputs.append(
            {
                "path": str(path),
                "law_id": law_id,
                "sha256": digest,
                "bytes": len(raw),
                "creationdate": header.get("creationdate"),
                "translation_units": len(pairs),
            }
        )
        split = split_for(law_id, split_seed)
        for pair in pairs:
            en, ja = pair["en"], pair["ja"]
            if not en or not ja or not valid_language_pair(en, ja):
                rejected["empty_or_language"] += 1
                continue
            if not (
                3 <= len(en) <= maximum_english_characters
                and 2 <= len(ja) <= maximum_japanese_characters
            ):
                rejected["length"] += 1
                continue
            pair_hash = sha256_bytes(
                f"{normalized(en)}\0{normalized(ja)}".encode("utf-8")
            )
            if pair_hash in seen_pairs:
                rejected["duplicate"] += 1
                continue
            if protected_index.matches(en, maximum_jaccard) or protected_index.matches(
                ja, maximum_jaccard
            ):
                rejected["contamination"] += 1
                continue
            seen_pairs.add(pair_hash)
            output[split].append(
                parallel_row(
                    source=en,
                    target=ja,
                    direction="en-ja",
                    law_id=law_id,
                    tuid=pair["tuid"],
                    tmx_sha256=digest,
                )
            )
            output[split].append(
                parallel_row(
                    source=ja,
                    target=en,
                    direction="ja-en",
                    law_id=law_id,
                    tuid=pair["tuid"],
                    tmx_sha256=digest,
                )
            )

    manifest: dict[str, Any] = {
        "schema_version": 2,
        "source": SOURCE_ROOT,
        "terms_url": TERMS_URL,
        "public_data_license_url": PDL_URL,
        "license": LICENSE,
        "attribution": ATTRIBUTION,
        "quality": (
            "finalized sentence-aligned TMX published by the Japanese Ministry of "
            "Justice; legal-domain training only"
        ),
        "tentative_translations_included": False,
        "promotion_eligible": False,
        "private_reasoning_traces_used": False,
        "document_grouped_split": True,
        "split_seed": split_seed,
        "protected_benchmarks": [str(path) for path in protected_paths],
        "maximum_jaccard": maximum_jaccard,
        "maximum_english_characters": maximum_english_characters,
        "maximum_japanese_characters": maximum_japanese_characters,
        "inputs": inputs,
        "input_files": len(inputs),
        "input_bytes": sum(item["bytes"] for item in inputs),
        "input_translation_units": sum(item["translation_units"] for item in inputs),
        "unique_pairs": len(seen_pairs),
        "examples": {split: len(rows) for split, rows in output.items()},
        "rejected": dict(sorted(rejected.items())),
    }
    return output, manifest


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return {"path": str(path), "rows": len(rows), "bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tmx_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--protected-benchmark", type=Path, action="append", default=[], required=True
    )
    parser.add_argument("--maximum-jaccard", type=float, default=0.80)
    parser.add_argument("--maximum-english-characters", type=int, default=240)
    parser.add_argument("--maximum-japanese-characters", type=int, default=160)
    parser.add_argument("--split-seed", default="mimi-jlt-finalized-v1")
    args = parser.parse_args()

    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_directory}")
    tmx_paths = sorted(args.tmx_directory.rglob("*.tmx"))
    if not tmx_paths:
        raise SystemExit(f"no TMX files found under {args.tmx_directory}")
    output, manifest = prepare(
        tmx_paths,
        args.protected_benchmark,
        maximum_jaccard=args.maximum_jaccard,
        maximum_english_characters=args.maximum_english_characters,
        maximum_japanese_characters=args.maximum_japanese_characters,
        split_seed=args.split_seed,
    )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    manifest["outputs"] = {
        split: write_jsonl(args.output_directory / f"{split}.jsonl", rows)
        for split, rows in output.items()
    }
    manifest_path = args.output_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**manifest, "manifest_sha256": sha256(manifest_path)}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
