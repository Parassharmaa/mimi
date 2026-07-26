#!/usr/bin/env python3
"""Benchmark the pinned Whisper Large-v3 Turbo Q4 MLX conversion."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import resource
import time
import unicodedata
from pathlib import Path

import soundfile
from mlx_audio.stt import load

MODEL_REVISION = "321a6ead9f6e0646bc8188a54d2a470e275c6b76"
MLX_AUDIO_REVISION = "d28d68c6ac4e28f7d2d66007f640b06cf3fd8ceb"
MODEL_FILE_SHA256 = {
    "config.json": "9135b2ae07e6450a8f4e87ad1124abe970f705d72ea426030f969cb5014b82e9",
    "model.safetensors": "45298f6dc48df8c11e0a8d1dc5e0197c688bfa530646fa21f1a0238d2b0ecda3",
    "tokenizer.json": "297b13372ac43916285644fb9687add3cc62ee2a1adb60da3dc25cc94c1871fd",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model(model_directory: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, expected in MODEL_FILE_SHA256.items():
        path = model_directory / name
        if not path.is_file():
            raise SystemExit(f"missing model file: {path}")
        observed[name] = sha256(path)
        if observed[name] != expected:
            raise SystemExit(
                f"unexpected SHA-256 for {path}: {observed[name]} != {expected}"
            )
    return observed


def mlx_audio_provenance() -> tuple[str, str]:
    distribution = importlib.metadata.distribution("mlx-audio")
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise SystemExit(
            "mlx-audio must be installed from the pinned Git revision; "
            "see scripts/speech/README.md"
        )
    direct_url = json.loads(direct_url_text)
    commit = direct_url.get("vcs_info", {}).get("commit_id")
    if commit != MLX_AUDIO_REVISION:
        raise SystemExit(
            f"unexpected mlx-audio commit: {commit!r} != {MLX_AUDIO_REVISION}"
        )
    return distribution.version, commit


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("suite", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--language", default="ja")
    parser.add_argument("--chunk-duration", type=float, default=1.0)
    parser.add_argument("--metric", choices=("cer", "wer"), default="cer")
    args = parser.parse_args()

    model_file_hashes = validate_model(args.model)
    mlx_audio_version, mlx_audio_commit = mlx_audio_provenance()
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

    warmup_path = suite_root / suite[0]["audio"]
    warmup_started = time.perf_counter()
    model.generate(
        str(warmup_path),
        language=args.language,
        chunk_duration=args.chunk_duration,
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
            chunk_duration=args.chunk_duration,
        )
        decode_seconds = time.perf_counter() - started
        hypothesis = generated.text
        reference_units = normalized_units(row["reference"], args.metric)
        hypothesis_units = normalized_units(hypothesis, args.metric)
        edits = edit_distance(reference_units, hypothesis_units)
        total_edits += edits
        total_reference_characters += len(reference_units)
        result = {
            "caseID": row["caseID"],
            "reference": row["reference"],
            "hypothesis": hypothesis,
            "audioDurationSeconds": audio_duration,
            "decodeSeconds": decode_seconds,
            "realTimeFactor": decode_seconds / audio_duration,
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

    report = {
        "format": "mimi-asr-benchmark-v1",
        "engine": "mlx-community/whisper-large-v3-turbo-asr-4bit",
        "modelRevision": MODEL_REVISION,
        "modelFilesSHA256": model_file_hashes,
        "mlxAudioVersion": mlx_audio_version,
        "mlxAudioRevision": mlx_audio_commit,
        "language": args.language,
        "metric": args.metric,
        "chunkDurationSeconds": args.chunk_duration,
        "loadSeconds": load_seconds,
        "warmupSeconds": warmup_seconds,
        "peakRSSBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "caseCount": len(results),
        "corpusErrorRate": total_edits / max(1, total_reference_characters),
        "meanRealTimeFactor": sum(row["realTimeFactor"] for row in results)
        / len(results),
        "durationWeightedRealTimeFactor": sum(
            row["decodeSeconds"] for row in results
        )
        / sum(row["audioDurationSeconds"] for row in results),
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
    print(
        json.dumps(
            {
                "output": str(args.output),
                "caseCount": report["caseCount"],
                "metric": report["metric"],
                "corpusErrorRate": report["corpusErrorRate"],
                "meanRealTimeFactor": report["meanRealTimeFactor"],
                "peakRSSBytes": report["peakRSSBytes"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
