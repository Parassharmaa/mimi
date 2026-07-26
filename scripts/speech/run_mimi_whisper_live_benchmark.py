#!/usr/bin/env python3
"""Run Mimi's native bounded Whisper path over a pinned ASR suite."""

from __future__ import annotations

import argparse
import json
import os
import re
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
        raise ValueError(f"Mimi Speech benchmark did not emit JSON: {output[-500:]}")
    return json.loads(output[start:])


def parse_peak_rss(stderr: str) -> int | None:
    match = re.search(r"(\d+)\s+maximum resident set size", stderr)
    return int(match.group(1)) if match else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("app", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("suite", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--language", choices=("en", "ja"), default="ja")
    parser.add_argument("--metric", choices=("cer", "wer"), default="cer")
    args = parser.parse_args()

    if not args.app.is_file():
        raise SystemExit(f"Mimi executable does not exist: {args.app}")
    if not args.model.is_dir():
        raise SystemExit(f"Mimi Speech model does not exist: {args.model}")
    suite = [
        json.loads(line)
        for line in args.suite.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not suite:
        raise SystemExit("suite is empty")
    suite_root = args.suite.parent
    environment = os.environ.copy()
    environment["MIMI_WHISPER_MLX_MODEL_DIR"] = str(args.model.resolve())

    total_edits = 0
    total_reference_units = 0
    results: list[dict] = []
    peak_rss_values: list[int] = []
    benchmark_started = time.perf_counter()
    for index, row in enumerate(suite, start=1):
        completed = subprocess.run(
            [
                "/usr/bin/time",
                "-l",
                str(args.app.resolve()),
                "--benchmark-realtime",
                "mimi-whisper",
                "--audio",
                str((suite_root / row["audio"]).resolve()),
                "--language",
                args.language,
                "--reference",
                row["reference"],
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        raw = parse_report(completed.stdout)
        hypothesis = raw["finalText"]
        reference_units = (
            normalized_characters(row["reference"])
            if args.metric == "cer"
            else normalized_words(row["reference"])
        )
        hypothesis_units = (
            normalized_characters(hypothesis)
            if args.metric == "cer"
            else normalized_words(hypothesis)
        )
        edits = edit_distance(reference_units, hypothesis_units)
        total_edits += edits
        total_reference_units += len(reference_units)
        if (peak_rss := parse_peak_rss(completed.stderr)) is not None:
            peak_rss_values.append(peak_rss)
        result = {
            "caseID": row["caseID"],
            "reference": row["reference"],
            "hypothesis": hypothesis,
            "audioDurationSeconds": raw["audioDurationSeconds"],
            "computeWallSeconds": raw["wallSeconds"],
            "modelLoadSeconds": raw["modelLoadSeconds"],
            "computeRealTimeFactor": raw.get("realTimeFactor"),
            "firstTextAtSeconds": raw.get("firstTextAtSeconds"),
            "firstFinalAtSeconds": raw.get("firstFinalAtSeconds"),
            "hypothesisChurn": raw["hypothesisChurn"],
            "updateCount": raw["updateCount"],
            "editDistance": edits,
            "referenceUnits": len(reference_units),
            "errorRate": edits / max(1, len(reference_units)),
        }
        results.append(result)
        print(
            f"[{index:02d}/{len(suite):02d}] {row['caseID']} "
            f"{args.metric.upper()}={result['errorRate']:.3f} "
            f"RTF={raw.get('realTimeFactor', float('nan')):.3f}",
            flush=True,
        )

    first_text = [
        row["firstTextAtSeconds"]
        for row in results
        if row["firstTextAtSeconds"] is not None
    ]
    report = {
        "format": "mimi-native-live-asr-benchmark-v1",
        "engine": "Mimi Speech Preview",
        "runtime": "MLX Audio Swift",
        "mode": "bounded-6s-partial-3s-stride-30s-final",
        "language": args.language,
        "metric": args.metric,
        "caseCount": len(results),
        "wallSeconds": time.perf_counter() - benchmark_started,
        "corpusErrorRate": total_edits / max(1, total_reference_units),
        "meanComputeRealTimeFactor": statistics.fmean(
            row["computeRealTimeFactor"] for row in results
        ),
        "firstTextP50Seconds": percentile(first_text, 0.50),
        "firstTextP95Seconds": percentile(first_text, 0.95),
        "meanHypothesisChurn": statistics.fmean(
            row["hypothesisChurn"] for row in results
        ),
        "peakRSSBytes": max(peak_rss_values) if peak_rss_values else None,
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
    print(f"wrote {len(results)} native live cases to {args.output}")


if __name__ == "__main__":
    main()
