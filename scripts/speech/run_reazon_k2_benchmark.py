#!/usr/bin/env python3
"""Benchmark the public ReazonSpeech K2 INT8 Japanese model."""

from __future__ import annotations

import argparse
import json
import resource
import time
import unicodedata
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile


def normalized_characters(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return [
        character
        for character in normalized
        if unicodedata.category(character)[0] not in {"P", "S", "Z", "C"}
    ]


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


def read_wave(path: Path) -> tuple[int, np.ndarray, float]:
    samples, sample_rate = soundfile.read(path, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    return sample_rate, mono, len(mono) / sample_rate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("suite", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    model_files = {
        "tokens": args.model / "tokens.txt",
        "encoder": args.model / "encoder-epoch-99-avg-1.int8.onnx",
        "decoder": args.model / "decoder-epoch-99-avg-1.int8.onnx",
        "joiner": args.model / "joiner-epoch-99-avg-1.int8.onnx",
    }
    missing = [str(path) for path in model_files.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"missing model files: {missing}")

    suite = [
        json.loads(line)
        for line in args.suite.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    suite_root = args.suite.parent
    load_started = time.perf_counter()
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        tokens=str(model_files["tokens"]),
        encoder=str(model_files["encoder"]),
        decoder=str(model_files["decoder"]),
        joiner=str(model_files["joiner"]),
        num_threads=args.threads,
        sample_rate=16_000,
        feature_dim=80,
        decoding_method="greedy_search",
        provider="cpu",
    )
    load_seconds = time.perf_counter() - load_started

    total_edits = 0
    total_reference_characters = 0
    results: list[dict] = []
    for row in suite:
        sample_rate, samples, duration = read_wave(suite_root / row["audio"])
        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        started = time.perf_counter()
        recognizer.decode_stream(stream)
        decode_seconds = time.perf_counter() - started
        hypothesis = stream.result.text
        reference_characters = normalized_characters(row["reference"])
        hypothesis_characters = normalized_characters(hypothesis)
        edits = edit_distance(reference_characters, hypothesis_characters)
        total_edits += edits
        total_reference_characters += len(reference_characters)
        results.append(
            {
                "caseID": row["caseID"],
                "reference": row["reference"],
                "hypothesis": hypothesis,
                "audioDurationSeconds": duration,
                "decodeSeconds": decode_seconds,
                "realTimeFactor": decode_seconds / duration,
                "characterEdits": edits,
                "referenceCharacters": len(reference_characters),
                "characterErrorRate": edits / max(1, len(reference_characters)),
            }
        )

    report = {
        "format": "mimi-asr-benchmark-v1",
        "engine": "reazon-research/reazonspeech-k2-v2-int8",
        "modelRevision": "291488c8151be24d7da4bf7af26e533fad96e407",
        "threads": args.threads,
        "loadSeconds": load_seconds,
        "peakRSSBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "caseCount": len(results),
        "corpusCharacterErrorRate": total_edits / max(1, total_reference_characters),
        "meanRealTimeFactor": sum(row["realTimeFactor"] for row in results)
        / len(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(results)} cases to {args.output}")


if __name__ == "__main__":
    main()
