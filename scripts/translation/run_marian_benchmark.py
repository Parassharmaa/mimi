#!/usr/bin/env python3
"""Benchmark the pinned Apache-2.0 OPUS-MT English/Japanese pair.

This is a research baseline. The Hugging Face/PyTorch weights are not app
artifacts; a candidate must still be converted, quantized, and exercised
through Mimi's shipping runtime before it can be promoted.
"""

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

import torch
from huggingface_hub import snapshot_download
from transformers import MarianMTModel, MarianTokenizer


DEFAULT_MODELS = {
    ("en-US", "ja-JP"): (
        "Helsinki-NLP/opus-mt-en-jap",
        "a863894cdd2b80f3bc1c5966734aee9ffec207d1",
    ),
    ("ja-JP", "en-US"): (
        "Helsinki-NLP/opus-mt-jap-en",
        "7f7d5b92a9b5a9731b6b509df7527f642a5962e8",
    ),
}

MODEL_FILES = [
    "config.json",
    "generation_config.json",
    "pytorch_model.bin",
    "model.safetensors",
    "source.spm",
    "target.spm",
    "tokenizer_config.json",
    "vocab.json",
]


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


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def translate(
    model: MarianMTModel,
    tokenizer: MarianTokenizer,
    text: str,
    device: torch.device,
    max_tokens: int,
) -> str:
    encoded = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_tokens,
        )
    sync(device)
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--warm-runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--en-ja-model", default=DEFAULT_MODELS[("en-US", "ja-JP")][0])
    parser.add_argument("--en-ja-revision", default=DEFAULT_MODELS[("en-US", "ja-JP")][1])
    parser.add_argument("--ja-en-model", default=DEFAULT_MODELS[("ja-JP", "en-US")][0])
    parser.add_argument("--ja-en-revision", default=DEFAULT_MODELS[("ja-JP", "en-US")][1])
    parser.add_argument(
        "--hf-home", type=Path, default=Path("Research/translation/models/hf-cache")
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable; pass --device cpu for a CPU-only run")

    suite = load_suite(args.suite)
    device = torch.device(args.device)
    models = {
        ("en-US", "ja-JP"): (args.en_ja_model, args.en_ja_revision),
        ("ja-JP", "en-US"): (args.ja_en_model, args.ja_en_revision),
    }
    snapshots: list[Path] = []
    results_by_id: dict[str, dict] = {}
    preparation_seconds = 0.0

    for direction, (repository, revision) in models.items():
        direction_rows = [
            row
            for row in suite
            if (row["sourceLanguage"], row["targetLanguage"]) == direction
        ]
        if not direction_rows:
            continue

        preparation_started = time.perf_counter()
        local_model = Path(repository)
        snapshot = (
            local_model.resolve()
            if local_model.is_dir()
            else Path(
                snapshot_download(
                    repo_id=repository,
                    revision=revision,
                    cache_dir=args.hf_home,
                    allow_patterns=MODEL_FILES,
                )
            )
        )
        tokenizer = MarianTokenizer.from_pretrained(snapshot)
        model = MarianMTModel.from_pretrained(snapshot).to(device).eval()
        sync(device)
        preparation_seconds += time.perf_counter() - preparation_started
        snapshots.append(snapshot)

        for row in direction_rows:
            started = time.perf_counter()
            hypothesis = translate(model, tokenizer, row["source"], device, args.max_tokens)
            first_latency = time.perf_counter() - started
            warm_latencies = []
            for _ in range(args.warm_runs):
                warm_started = time.perf_counter()
                translate(model, tokenizer, row["source"], device, args.max_tokens)
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

        del model, tokenizer
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    missing = [row["id"] for row in suite if row["id"] not in results_by_id]
    if missing:
        raise SystemExit(f"no Marian model configured for cases: {', '.join(missing)}")

    report = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "engine": (
            f"transformers:{args.en_ja_model}@{args.en_ja_revision[:12]}+"
            f"{args.ja_en_model}@{args.ja_en_revision[:12]}:{args.device}"
        ),
        "hardware": hardware_name(),
        "operatingSystem": platform.platform(),
        "preparationSeconds": preparation_seconds,
        "peakResidentBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "modelBytes": sum(directory_bytes(path) for path in snapshots),
        "results": [results_by_id[row["id"]] for row in suite],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(suite)} cases to {args.output}")


if __name__ == "__main__":
    main()
