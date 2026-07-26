#!/usr/bin/env python3
"""Materialize Mimi's pinned 24-clip Japanese ASR screen from FLEURS."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import Audio, load_dataset

DATASET = "google/fleurs"
CONFIG = "ja_jp"
SPLIT = "test"
REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
SELECTED_POSITIONS = (
    0,
    4,
    10,
    13,
    16,
    20,
    21,
    22,
    26,
    29,
    33,
    45,
    47,
    49,
    59,
    68,
    77,
    81,
    83,
    84,
    89,
    95,
    106,
    119,
)
PROTECTED_TERMS = {
    0: ("敵対的環境コース",),
    4: ("ロスビー数", "磁気反転"),
    13: ("m16",),
    20: ("微表情",),
    21: ("80 km", "50マイル"),
    26: ("70 km", "100 km"),
    33: ("xdr-tb", "33万人", "6,000人"),
    45: ("asus", "eee pc", "2007", "taipei it month"),
    47: ("ms", "中枢神経系"),
    49: ("1963",),
    59: ("civilis", "civis", "civitas"),
    68: ("fkp", "gbp", "1ポンド"),
    77: ("2005", "fbi", "10人"),
    81: ("放射性セシウム", "ヨウ素"),
    83: ("プリンストン大学", "ウプサラ大学", "2世代"),
    84: ("原子核",),
    89: ("ph", "水素イオン"),
    95: ("交通流研究",),
    106: ("35 mm",),
    119: ("ai", "sf", "コンピュータサイエンス"),
}
PROTECTED_TERM_ALIASES = {
    "80 km": ("80キロメートル", "八十キロメートル"),
    "50マイル": ("五十マイル",),
    "70 km": ("70キロメートル", "七十キロメートル"),
    "100 km": ("100キロメートル", "百キロメートル"),
    "33万人": ("330,000人", "330000人", "三十三万人"),
    "6,000人": ("6000人", "六千人"),
    "2007": ("二千七", "二〇〇七"),
    "1963": ("千九百六十三", "一九六三"),
    "1ポンド": ("一ポンド",),
    "2005": ("二千五", "二〇〇五"),
    "10人": ("十人",),
    "2世代": ("二世代",),
    "35 mm": ("35ミリ", "35ミリメートル", "三十五ミリ"),
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    audio_directory = args.output / "audio"
    audio_directory.mkdir(parents=True, exist_ok=True)

    wanted = set(SELECTED_POSITIONS)
    dataset = load_dataset(
        DATASET,
        CONFIG,
        split=SPLIT,
        streaming=True,
        revision=REVISION,
    ).cast_column("audio", Audio(decode=False))

    rows: list[dict] = []
    for position, row in enumerate(dataset):
        if position not in wanted:
            if position > SELECTED_POSITIONS[-1]:
                break
            continue
        audio = row["audio"]
        payload = audio["bytes"]
        if not isinstance(payload, bytes) or not payload:
            raise SystemExit(f"FLEURS row {position} has no embedded audio")
        filename = f"fleurs-ja-test-{position:04d}-{row['id']}.wav"
        path = audio_directory / filename
        path.write_bytes(payload)
        rows.append(
            {
                "caseID": f"fleurs-ja-screen-v1:{position:04d}:{row['id']}",
                "dataset": DATASET,
                "datasetConfig": CONFIG,
                "datasetRevision": REVISION,
                "datasetSplit": SPLIT,
                "datasetPosition": position,
                "license": "CC-BY-4.0",
                "audio": f"audio/{filename}",
                "audioSha256": sha256(payload),
                "reference": row["transcription"],
                "rawReference": row["raw_transcription"],
                "protectedTerms": list(PROTECTED_TERMS.get(position, ())),
                "protectedTermAliases": {
                    term: list(PROTECTED_TERM_ALIASES.get(term, ()))
                    for term in PROTECTED_TERMS.get(position, ())
                },
                "gender": row["gender"],
            }
        )

    if [row["datasetPosition"] for row in rows] != list(SELECTED_POSITIONS):
        raise SystemExit("FLEURS stream did not yield the exact registered positions")

    manifest = args.output / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} cases to {manifest}")


if __name__ == "__main__":
    main()
