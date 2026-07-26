#!/usr/bin/env python3
"""Benchmark a local Nemotron 3.5 ASR MLX conversion on a fixed suite."""

from __future__ import annotations

import argparse
import json
import resource
import time
import unicodedata
from pathlib import Path

import soundfile
from mlx_audio.stt import load

MODEL_REVISION = "7279359e4481b5e9e185a318bd618e429c6d86cd"
MLX_AUDIO_REVISION = "d28d68c6ac4e28f7d2d66007f640b06cf3fd8ceb"


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


def parse_context(value: str) -> list[int]:
    try:
        left, right = value.split(",", maxsplit=1)
        context = [int(left), int(right)]
    except ValueError as error:
        raise argparse.ArgumentTypeError("context must be LEFT,RIGHT") from error
    if context not in ([56, 0], [56, 3], [56, 6], [56, 13]):
        raise argparse.ArgumentTypeError(
            "context must be one of 56,0; 56,3; 56,6; or 56,13"
        )
    return context


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("suite", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--language", default="ja-JP")
    parser.add_argument("--context", type=parse_context, default=[56, 13])
    args = parser.parse_args()

    required_files = [
        args.model / "config.json",
        args.model / "model.safetensors",
        args.model / "tokenizer.model",
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise SystemExit(f"missing model files: {missing}")

    suite = [
        json.loads(line)
        for line in args.suite.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not suite:
        raise SystemExit("suite is empty")
    suite_root = args.suite.parent

    load_started = time.perf_counter()
    model = load(str(args.model))
    load_seconds = time.perf_counter() - load_started

    # MLX lazily compiles kernels on the first generation. Record that cost
    # separately so steady-state RTF is meaningful for a long-running app.
    warmup_audio = suite_root / suite[0]["audio"]
    warmup_started = time.perf_counter()
    model.generate(
        str(warmup_audio),
        language=args.language,
        att_context_size=args.context,
    )
    warmup_seconds = time.perf_counter() - warmup_started

    total_edits = 0
    total_reference_characters = 0
    results: list[dict] = []
    for row in suite:
        audio_path = suite_root / row["audio"]
        audio_duration = soundfile.info(audio_path).duration
        started = time.perf_counter()
        generated = model.generate(
            str(audio_path),
            language=args.language,
            att_context_size=args.context,
        )
        decode_seconds = time.perf_counter() - started
        hypothesis = generated.text
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
                "audioDurationSeconds": audio_duration,
                "decodeSeconds": decode_seconds,
                "realTimeFactor": decode_seconds / audio_duration,
                "characterEdits": edits,
                "referenceCharacters": len(reference_characters),
                "characterErrorRate": edits / max(1, len(reference_characters)),
            }
        )

    report = {
        "format": "mimi-asr-benchmark-v1",
        "engine": "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit",
        "modelRevision": MODEL_REVISION,
        "mlxAudioRevision": MLX_AUDIO_REVISION,
        "language": args.language,
        "attentionContext": args.context,
        "loadSeconds": load_seconds,
        "warmupSeconds": warmup_seconds,
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
    print(
        json.dumps(
            {
                "output": str(args.output),
                "caseCount": report["caseCount"],
                "corpusCharacterErrorRate": report["corpusCharacterErrorRate"],
                "meanRealTimeFactor": report["meanRealTimeFactor"],
                "peakRSSBytes": report["peakRSSBytes"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
