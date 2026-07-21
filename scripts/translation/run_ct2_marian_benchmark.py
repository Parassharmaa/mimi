#!/usr/bin/env python3
"""Benchmark a local, quantized CTranslate2 Marian pair."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import ctranslate2
from transformers import MarianTokenizer


def load_suite(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit("benchmark suite is empty")
    identifiers = [row["id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise SystemExit("benchmark suite contains duplicate IDs")
    return rows


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def hardware_name() -> str:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.machine()


def translate(
    translator: ctranslate2.Translator,
    tokenizer: MarianTokenizer,
    text: str,
    max_tokens: int,
) -> str:
    source_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
    result = translator.translate_batch(
        [source_tokens],
        beam_size=1,
        max_decoding_length=max_tokens,
        return_scores=False,
    )[0]
    target_ids = tokenizer.convert_tokens_to_ids(result.hypotheses[0])
    return tokenizer.decode(target_ids, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--en-ja-model", type=Path, required=True)
    parser.add_argument("--ja-en-model", type=Path, required=True)
    parser.add_argument("--en-ja-tokenizer", type=Path, required=True)
    parser.add_argument("--ja-en-tokenizer", type=Path, required=True)
    parser.add_argument("--warm-runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    suite = load_suite(args.suite)
    models = {
        ("en-US", "ja-JP"): (args.en_ja_model, args.en_ja_tokenizer),
        ("ja-JP", "en-US"): (args.ja_en_model, args.ja_en_tokenizer),
    }
    results_by_id: dict[str, dict] = {}
    preparation_seconds = 0.0

    for direction, (model_path, tokenizer_path) in models.items():
        direction_rows = [
            row
            for row in suite
            if (row["sourceLanguage"], row["targetLanguage"]) == direction
        ]
        preparation_started = time.perf_counter()
        translator = ctranslate2.Translator(
            str(model_path),
            device="cpu",
            compute_type="int8",
            inter_threads=1,
            intra_threads=args.threads,
        )
        tokenizer = MarianTokenizer.from_pretrained(tokenizer_path)
        preparation_seconds += time.perf_counter() - preparation_started

        for row in direction_rows:
            started = time.perf_counter()
            hypothesis = translate(
                translator,
                tokenizer,
                row["source"],
                args.max_tokens,
            )
            first_latency = time.perf_counter() - started
            warm_latencies = []
            for _ in range(args.warm_runs):
                warm_started = time.perf_counter()
                translate(
                    translator,
                    tokenizer,
                    row["source"],
                    args.max_tokens,
                )
                warm_latencies.append(time.perf_counter() - warm_started)
            results_by_id[row["id"]] = {
                "caseID": row["id"],
                "sourceLanguage": row["sourceLanguage"],
                "targetLanguage": row["targetLanguage"],
                "domain": row["domain"],
                "source": row["source"],
                "references": row["references"],
                "claimEligible": bool(row["claimEligible"]),
                "hypothesis": hypothesis,
                "latencySeconds": first_latency,
                "warmLatencySeconds": warm_latencies,
            }

        del translator, tokenizer
        gc.collect()

    report = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "engine": f"ctranslate2:{ctranslate2.__version__}:ElanMT-BT-pair:int8",
        "hardware": hardware_name(),
        "operatingSystem": platform.platform(),
        "preparationSeconds": preparation_seconds,
        "peakResidentBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "modelBytes": sum(directory_bytes(path) for path, _ in models.values()),
        "results": [results_by_id[row["id"]] for row in suite],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(suite)} cases to {args.output}")


if __name__ == "__main__":
    main()
