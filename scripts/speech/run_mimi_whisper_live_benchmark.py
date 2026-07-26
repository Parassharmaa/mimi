#!/usr/bin/env python3
"""Run Mimi's native bounded Whisper path over a pinned ASR suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import time
import unicodedata
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
    parser.add_argument("--initial-partial-stride", type=float)
    parser.add_argument("--partial-stride", type=float, default=3.0)
    parser.add_argument("--endpoint-silence", type=float, default=0.75)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if not args.app.is_file():
        raise SystemExit(f"Mimi executable does not exist: {args.app}")
    if not args.model.is_dir():
        raise SystemExit(f"Mimi Speech model does not exist: {args.model}")
    suite_sha256 = sha256_file(args.suite)
    suite = [
        json.loads(line)
        for line in args.suite.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not suite:
        raise SystemExit("suite is empty")
    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be positive")
        suite = suite[: args.limit]
    suite_root = args.suite.parent
    audio_paths: list[Path] = []
    for row in suite:
        audio_path = (suite_root / row["audio"]).resolve()
        expected_sha256 = row.get("audioSha256")
        if not audio_path.is_file():
            raise SystemExit(f"audio does not exist: {audio_path}")
        if not expected_sha256:
            raise SystemExit(f"{row['caseID']} does not declare audioSha256")
        actual_sha256 = sha256_file(audio_path)
        if actual_sha256 != expected_sha256:
            raise SystemExit(
                f"{row['caseID']} audio hash mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        audio_paths.append(audio_path)

    model_weights = args.model / "model.safetensors"
    if not model_weights.is_file():
        raise SystemExit(f"Mimi Speech weights do not exist: {model_weights}")
    executable_sha256 = sha256_file(args.app)
    model_weights_sha256 = sha256_file(model_weights)
    environment = os.environ.copy()
    environment["MIMI_WHISPER_MLX_MODEL_DIR"] = str(args.model.resolve())

    total_edits = 0
    total_reference_units = 0
    total_first_update_edits = 0
    total_first_update_units = 0
    first_update_coverages: list[float] = []
    results: list[dict] = []
    peak_rss_values: list[int] = []
    benchmark_started = time.perf_counter()
    for index, (row, audio_path) in enumerate(
        zip(suite, audio_paths, strict=True),
        start=1,
    ):
        command = [
            "/usr/bin/time",
            "-l",
            str(args.app.resolve()),
            "--benchmark-realtime",
            "mimi-whisper",
            "--audio",
            str(audio_path),
            "--language",
            args.language,
            "--partial-stride",
            str(args.partial_stride),
            "--endpoint-silence",
            str(args.endpoint_silence),
        ]
        if args.initial_partial_stride is not None:
            command.extend(
                [
                    "--initial-partial-stride",
                    str(args.initial_partial_stride),
                ]
            )
        command.extend(
            [
                "--reference",
                row["reference"],
            ]
        )
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        if completed.returncode != 0:
            details = completed.stderr.strip() or completed.stdout.strip()
            raise SystemExit(
                f"{row['caseID']} benchmark failed with exit "
                f"{completed.returncode}: {details[-2_000:]}"
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
        first_update = raw["firstUpdates"][0] if raw["firstUpdates"] else ""
        first_update_units = (
            normalized_characters(first_update)
            if args.metric == "cer"
            else normalized_words(first_update)
        )
        reference_prefix = reference_units[: len(first_update_units)]
        reference_prefix_units = len(reference_prefix)
        first_update_edits = edit_distance(
            first_update_units,
            reference_prefix,
        )
        edits = edit_distance(reference_units, hypothesis_units)
        total_edits += edits
        total_reference_units += len(reference_units)
        total_first_update_edits += first_update_edits
        total_first_update_units += reference_prefix_units
        first_update_coverages.append(
            reference_prefix_units / max(1, len(reference_units))
        )
        if (peak_rss := parse_peak_rss(completed.stderr)) is not None:
            peak_rss_values.append(peak_rss)
        result = {
            "caseID": row["caseID"],
            "mode": raw["mode"],
            "feedChunkSeconds": raw["feedChunkSeconds"],
            "initialPartialStrideSeconds": raw[
                "initialPartialStrideSeconds"
            ],
            "partialStrideSeconds": raw["partialStrideSeconds"],
            "endpointSilenceSeconds": raw["endpointSilenceSeconds"],
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
            "firstUpdates": raw["firstUpdates"],
            "firstUpdatePrefixEditDistance": first_update_edits,
            "firstUpdateUnits": len(first_update_units),
            "firstUpdateReferencePrefixUnits": reference_prefix_units,
            "firstUpdateReferenceCoverage": first_update_coverages[-1],
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
        "mode": results[0]["mode"],
        "executableSha256": executable_sha256,
        "feedChunkSeconds": results[0]["feedChunkSeconds"],
        "suiteSha256": suite_sha256,
        "selectedCaseIDs": [row["caseID"] for row in suite],
        "modelWeightsSha256": model_weights_sha256,
        "effectiveProfile": {
            "initialPartialStrideSeconds": results[0][
                "initialPartialStrideSeconds"
            ],
            "partialStrideSeconds": results[0]["partialStrideSeconds"],
            "endpointSilenceSeconds": results[0]["endpointSilenceSeconds"],
        },
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
        "corpusFirstUpdatePrefixErrorRate": (
            total_first_update_edits / max(1, total_first_update_units)
        ),
        "meanFirstUpdateReferenceCoverage": statistics.fmean(
            first_update_coverages
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
