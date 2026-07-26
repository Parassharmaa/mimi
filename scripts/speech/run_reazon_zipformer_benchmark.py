#!/usr/bin/env python3
"""Benchmark Reazon's 2025 Japanese Zipformer CTC checkpoint."""

from __future__ import annotations

import argparse
import json
import resource
import time
import unicodedata
from pathlib import Path

import numpy as np
import soundfile
import torch
from transformers import AutoModelForCTC, AutoProcessor

MODEL_REVISION = "df19e126d86994fb72a0a3653fcb31ebe49e6081"
TRANSFORMERS_VERSION = "4.57.0"


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


def load_audio(path: Path) -> tuple[np.ndarray, int, float]:
    samples, sample_rate = soundfile.read(
        path,
        dtype="float32",
        always_2d=True,
    )
    mono = samples.mean(axis=1)
    duration = len(mono) / sample_rate
    padded = np.pad(mono, int(0.5 * sample_rate))
    return padded, sample_rate, duration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("suite", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()

    if not (args.model / "model.safetensors").is_file():
        raise SystemExit(f"missing model weights in {args.model}")
    suite = [
        json.loads(line)
        for line in args.suite.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not suite:
        raise SystemExit("suite is empty")
    suite_root = args.suite.parent

    load_started = time.perf_counter()
    model = AutoModelForCTC.from_pretrained(
        args.model,
        trust_remote_code=True,
    ).to(args.device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model)
    load_seconds = time.perf_counter() - load_started

    def transcribe(path: Path) -> str:
        audio, sample_rate, _ = load_audio(path)
        inputs = processor(
            audio,
            return_tensors="pt",
            sampling_rate=sample_rate,
        )
        input_values = inputs.input_values.to(args.device)
        padding_mask = torch.zeros_like(input_values, dtype=torch.bool)
        with torch.inference_mode():
            logits = model(input_values, padding_mask=padding_mask).logits
        predicted_ids = torch.argmax(logits, dim=-1)[0].cpu()
        return processor.decode(
            predicted_ids,
            skip_special_tokens=True,
        ).removeprefix("▁")

    warmup_path = suite_root / suite[0]["audio"]
    warmup_started = time.perf_counter()
    transcribe(warmup_path)
    warmup_seconds = time.perf_counter() - warmup_started

    total_edits = 0
    total_reference_characters = 0
    results: list[dict] = []
    for row in suite:
        audio_path = suite_root / row["audio"]
        _, _, audio_duration = load_audio(audio_path)
        started = time.perf_counter()
        hypothesis = transcribe(audio_path)
        decode_seconds = time.perf_counter() - started
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
        "engine": "reazon-research/japanese-zipformer-base-k2-rs35kh",
        "modelRevision": MODEL_REVISION,
        "transformersVersion": TRANSFORMERS_VERSION,
        "device": args.device,
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
