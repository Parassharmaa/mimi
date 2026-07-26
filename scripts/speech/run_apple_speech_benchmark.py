#!/usr/bin/env python3
"""Run Mimi's Apple progressive speech benchmark over an ASR suite."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import time
import unicodedata
from pathlib import Path


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
        " " if unicodedata.category(character)[0] in {"P", "S", "Z", "C"} else character
        for character in normalized
    )
    return surface.split()


def normalized_units(text: str, metric: str) -> list[str]:
    if metric == "cer":
        return normalized_characters(text)
    return normalized_words(text)


def edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_item in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def parse_report(output: str) -> dict:
    start = output.find("{")
    if start < 0:
        raise ValueError(f"Apple benchmark did not emit JSON: {output[-500:]}")
    return json.loads(output[start:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("app", type=Path)
    parser.add_argument("suite", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("staging", type=Path)
    parser.add_argument("--language", default="ja")
    parser.add_argument("--metric", choices=("cer", "wer"), default="cer")
    args = parser.parse_args()

    if not args.app.is_file():
        raise SystemExit(f"Mimi executable does not exist: {args.app}")
    suite = [
        json.loads(line)
        for line in args.suite.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    suite_root = args.suite.parent
    args.staging.mkdir(parents=True, exist_ok=True)

    total_edits = 0
    total_reference_characters = 0
    results: list[dict] = []
    benchmark_started = time.perf_counter()
    for index, row in enumerate(suite, start=1):
        staged_audio = args.staging / Path(row["audio"]).name
        shutil.copy2(suite_root / row["audio"], staged_audio)
        completed = subprocess.run(
            [
                str(args.app),
                "--benchmark-realtime",
                "apple-progressive",
                "--audio",
                str(staged_audio),
                "--language",
                args.language,
                "--reference",
                row["reference"],
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        raw = parse_report(completed.stdout)
        hypothesis = raw["finalText"]
        reference_units = normalized_units(row["reference"], args.metric)
        hypothesis_units = normalized_units(hypothesis, args.metric)
        edits = edit_distance(reference_units, hypothesis_units)
        total_edits += edits
        total_reference_characters += len(reference_units)
        result = {
            "caseID": row["caseID"],
            "reference": row["reference"],
            "hypothesis": hypothesis,
            "audioDurationSeconds": raw["audioDurationSeconds"],
            "wallSeconds": raw["wallSeconds"],
            "realTimeFactor": raw["realTimeFactor"],
            "firstTextAtSeconds": raw["firstTextAtSeconds"],
            "firstFinalAtSeconds": raw["firstFinalAtSeconds"],
            "hypothesisChurn": raw["hypothesisChurn"],
            "updateCount": raw["updateCount"],
            "editDistance": edits,
            "referenceUnits": len(reference_units),
            "errorRate": edits / max(1, len(reference_units)),
        }
        if args.metric == "cer":
            result.update(
                {
                    "characterEdits": edits,
                    "referenceCharacters": len(reference_units),
                    "characterErrorRate": result["errorRate"],
                }
            )
        else:
            result.update(
                {
                    "wordEdits": edits,
                    "referenceWords": len(reference_units),
                    "wordErrorRate": result["errorRate"],
                }
            )
        results.append(result)
        print(
            f"[{index:02d}/{len(suite):02d}] {row['caseID']} "
            f"{args.metric.upper()}={results[-1]['errorRate']:.3f}",
            flush=True,
        )

    first_text = [row["firstTextAtSeconds"] for row in results]
    report = {
        "format": "mimi-asr-benchmark-v1",
        "engine": "Apple SpeechAnalyzer progressive",
        "language": args.language,
        "metric": args.metric,
        "caseCount": len(results),
        "wallSeconds": time.perf_counter() - benchmark_started,
        "corpusErrorRate": total_edits / max(1, total_reference_characters),
        "meanRealTimeFactor": statistics.fmean(
            row["realTimeFactor"] for row in results
        ),
        "firstTextP50Seconds": percentile(first_text, 0.50),
        "firstTextP95Seconds": percentile(first_text, 0.95),
        "results": results,
    }
    if args.metric == "cer":
        report["corpusCharacterErrorRate"] = report["corpusErrorRate"]
    else:
        report["corpusWordErrorRate"] = report["corpusErrorRate"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(results)} cases to {args.output}")


if __name__ == "__main__":
    main()
