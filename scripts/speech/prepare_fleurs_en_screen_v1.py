#!/usr/bin/env python3
"""Materialize Mimi's pinned 24-clip English ASR screen from FLEURS."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import Audio, load_dataset

DATASET = "google/fleurs"
CONFIG = "en_us"
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
    0: ("25", "30"),
    10: ("150", "200", "Dunlap broadsides"),
    13: ("Nyiragongo",),
    16: ("35mm",),
    26: ("Bennet School", "North Carolina"),
    33: ("Addenbrooke's Hospital", "Cambridge"),
    47: ("Vatican City", "800"),
    68: ("1966", "Sundarbans", "400", "30,000", "Bengal tigers"),
    77: ("pH",),
    81: ("Giza Plateau", "Giza Necropolis", "Great Pyramid", "Great Sphinx"),
    83: ("USA Gymnastics", "USOC"),
    84: ("MS", "central nervous system", "spinal cord", "optic nerve"),
    95: ("Mashhad", "17"),
    106: ("traffic flow", "100%"),
}
PROTECTED_TERM_ALIASES = {
    "25": ("twenty five", "twenty-five"),
    "30": ("thirty",),
    "150": ("one hundred fifty", "one hundred and fifty"),
    "200": ("two hundred",),
    "35mm": ("35 mm", "thirty five millimeter", "thirty-five millimeter"),
    "800": ("eight hundred",),
    "1966": ("nineteen sixty six", "nineteen sixty-six"),
    "400": ("four hundred",),
    "30,000": ("30000", "thirty thousand"),
    "17": ("seventeen",),
    "100%": ("100 percent", "one hundred percent"),
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
        filename = f"fleurs-en-test-{position:04d}-{row['id']}.wav"
        path = audio_directory / filename
        path.write_bytes(payload)
        terms = PROTECTED_TERMS.get(position, ())
        rows.append(
            {
                "caseID": f"fleurs-en-screen-v1:{position:04d}:{row['id']}",
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
                "protectedTerms": list(terms),
                "protectedTermAliases": {
                    term: list(PROTECTED_TERM_ALIASES.get(term, ())) for term in terms
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
