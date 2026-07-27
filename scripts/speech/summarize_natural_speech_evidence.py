#!/usr/bin/env python3
"""Create deterministic diagnostics for Mimi's natural speech evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import unicodedata
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
JAPANESE_ROOT = REPOSITORY_ROOT / "Research/speech/work/natural-ja-kokoro-v1"
ENGLISH_ROOT = REPOSITORY_ROOT / "Research/speech/work/natural-en-ami-v1"
REPORTS = {
    "japanese": {
        "product": JAPANESE_ROOT / "mimi-product-paced-v1.json",
        "adaptive": JAPANESE_ROOT / "mimi-adaptive-ja30-6-paced-v1.json",
    },
    "english": {
        "headset": {
            "product": (
                ENGLISH_ROOT / "mimi-product-headset-paced-v1.json"
            ),
            "hard24": (
                ENGLISH_ROOT / "mimi-hard24-headset-paced-v1.json"
            ),
            "adaptive": (
                ENGLISH_ROOT
                / "mimi-adaptive-en24-6-headset-paced-v1.json"
            ),
        },
        "array1-01": {
            "product": (
                ENGLISH_ROOT / "mimi-product-array1-01-paced-v1.json"
            ),
            "hard24": (
                ENGLISH_ROOT / "mimi-hard24-array1-01-paced-v1.json"
            ),
            "adaptive": (
                ENGLISH_ROOT
                / "mimi-adaptive-en24-6-array1-01-paced-v1.json"
            ),
        },
    },
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_characters(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return [
        character
        for character in normalized
        if unicodedata.category(character)[0] not in {"P", "S", "Z", "C"}
    ]


def normalized_words(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    surface = "".join(
        (
            " "
            if unicodedata.category(character)[0] in {"P", "S", "Z", "C"}
            else character
        )
        for character in normalized
    )
    return surface.split()


def katakana_to_hiragana(text: str) -> str:
    return "".join(
        chr(ord(character) - 0x60)
        if "\u30a1" <= character <= "\u30f6"
        else character
        for character in text
    )


def normalized_reading_characters(text: str, tagger) -> list[str]:
    readings: list[str] = []
    for word in tagger(text):
        reading = getattr(word.feature, "kana", None)
        if reading is None or reading == "*":
            reading = word.surface
        readings.append(reading)
    normalized = unicodedata.normalize(
        "NFKC",
        katakana_to_hiragana("".join(readings)),
    ).casefold()
    return [
        character
        for character in normalized
        if unicodedata.category(character)[0] not in {"P", "S", "Z", "C"}
    ]


def edit_breakdown(
    reference: list[str],
    hypothesis: list[str],
) -> dict[str, int | float]:
    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    costs = [[0] * columns for _ in range(rows)]
    operations = [[""] * columns for _ in range(rows)]
    for row in range(1, rows):
        costs[row][0] = row
        operations[row][0] = "deletion"
    for column in range(1, columns):
        costs[0][column] = column
        operations[0][column] = "insertion"
    for row in range(1, rows):
        for column in range(1, columns):
            if reference[row - 1] == hypothesis[column - 1]:
                costs[row][column] = costs[row - 1][column - 1]
                operations[row][column] = "match"
                continue
            candidates = (
                (costs[row - 1][column - 1] + 1, 0, "substitution"),
                (costs[row - 1][column] + 1, 1, "deletion"),
                (costs[row][column - 1] + 1, 2, "insertion"),
            )
            cost, _, operation = min(candidates)
            costs[row][column] = cost
            operations[row][column] = operation

    counts = {
        "matches": 0,
        "substitutions": 0,
        "deletions": 0,
        "insertions": 0,
    }
    row = len(reference)
    column = len(hypothesis)
    while row > 0 or column > 0:
        operation = operations[row][column]
        if operation == "match":
            counts["matches"] += 1
            row -= 1
            column -= 1
        elif operation == "substitution":
            counts["substitutions"] += 1
            row -= 1
            column -= 1
        elif operation == "deletion":
            counts["deletions"] += 1
            row -= 1
        elif operation == "insertion":
            counts["insertions"] += 1
            column -= 1
        else:
            raise ValueError(
                f"missing edit operation at ({row}, {column})"
            )
    edits = (
        counts["substitutions"]
        + counts["deletions"]
        + counts["insertions"]
    )
    return {
        **counts,
        "edits": edits,
        "referenceUnits": len(reference),
        "hypothesisUnits": len(hypothesis),
        "errorRate": edits / max(1, len(reference)),
    }


def load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def operational_summary(report: dict) -> dict[str, float | int]:
    result = report["results"][0]
    return {
        "firstTextSeconds": report["firstTextP50Seconds"],
        "inputDeliveryRTF": report["meanInputDeliveryRealTimeFactor"],
        "pacedWallRTF": report["meanPacedWallRealTimeFactor"],
        "maximumQueuedAudioSeconds": report["maximumQueuedAudioSeconds"],
        "postAudioFinalizationSeconds": report[
            "postAudioFinalizationP50Seconds"
        ],
        "peakRSSBytes": report["peakRSSBytes"],
        "droppedAudioSamples": report["totalDroppedAudioSamples"],
        "audioDropEvents": report["totalAudioDropEventCount"],
        "backpressureEvents": report["totalBackpressureEventCount"],
        "segmentCount": len(result["finalizedSegments"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    try:
        import fugashi
    except ImportError as error:
        raise SystemExit(
            "install fugashi and unidic-lite; use the pinned command in "
            "scripts/speech/README.md"
        ) from error
    tagger = fugashi.Tagger()
    output = {
        "format": "mimi-natural-speech-summary-v1",
        "diagnosticDependencies": {
            "fugashi": importlib.metadata.version("fugashi"),
            "unidic-lite": importlib.metadata.version("unidic-lite"),
        },
        "japanese": {},
        "english": {},
        "sourceReports": {},
    }

    for profile, path in REPORTS["japanese"].items():
        report = load_report(path)
        result = report["results"][0]
        raw = edit_breakdown(
            normalized_characters(result["reference"]),
            normalized_characters(result["hypothesis"]),
        )
        reading = edit_breakdown(
            normalized_reading_characters(result["reference"], tagger),
            normalized_reading_characters(result["hypothesis"], tagger),
        )
        output["japanese"][profile] = {
            "rawCharacterError": raw,
            "readingCharacterError": reading,
            "operational": operational_summary(report),
        }
        output["sourceReports"][str(path.relative_to(REPOSITORY_ROOT))] = (
            sha256_file(path)
        )

    for condition, profiles in REPORTS["english"].items():
        output["english"][condition] = {}
        for profile, path in profiles.items():
            report = load_report(path)
            result = report["results"][0]
            words = edit_breakdown(
                normalized_words(result["reference"]),
                normalized_words(result["hypothesis"]),
            )
            if abs(words["errorRate"] - report["corpusErrorRate"]) > 1e-12:
                raise SystemExit(f"{path}: WER reconstruction mismatch")
            output["english"][condition][profile] = {
                "wordError": words,
                "operational": operational_summary(report),
            }
            output["sourceReports"][
                str(path.relative_to(REPOSITORY_ROOT))
            ] = sha256_file(path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote natural speech diagnostics to {args.output}")


if __name__ == "__main__":
    main()
