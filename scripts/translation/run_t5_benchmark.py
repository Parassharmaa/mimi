#!/usr/bin/env python3
"""Benchmark one pinned T5-style checkpoint in both Mimi directions."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DEFAULT_REPOSITORY = "WhirlwindAI/Translate-15L"
DEFAULT_REVISION = "ce860c33668440b031e30f50cc31377c6b6fac59"
LANGUAGE_CODES = {"en-US": "en", "ja-JP": "ja"}
MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


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
    for row in rows:
        for language in (row["sourceLanguage"], row["targetLanguage"]):
            if language not in LANGUAGE_CODES:
                raise SystemExit(f"unsupported benchmark language: {language}")
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
    model,
    tokenizer,
    row: dict,
    device: torch.device,
    max_tokens: int,
    num_beams: int,
) -> str:
    source_code = LANGUAGE_CODES[row["sourceLanguage"]]
    target_code = LANGUAGE_CODES[row["targetLanguage"]]
    prompt = f"translate {source_code} to {target_code}: {row['source']}"
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            do_sample=False,
            num_beams=num_beams,
            use_cache=True,
            max_new_tokens=max_tokens,
        )
    sync(device)
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--warm-runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument(
        "--hf-home", type=Path, default=Path("Research/translation/models/hf-cache")
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()

    if args.warm_runs < 0 or args.num_beams <= 0:
        raise SystemExit("warm-runs must be non-negative and num-beams must be positive")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable; pass --device cpu for a CPU-only run")
    suite = load_suite(args.suite)
    device = torch.device(args.device)

    preparation_started = time.perf_counter()
    local_model = Path(args.repository)
    snapshot = (
        local_model.resolve()
        if local_model.is_dir()
        else Path(
            snapshot_download(
                repo_id=args.repository,
                revision=args.revision,
                cache_dir=args.hf_home,
                allow_patterns=MODEL_FILES,
            )
        )
    )
    # Translate-15L was exported by Transformers 5 with
    # ``extra_special_tokens`` as a list. Transformers 4.57 expects that new
    # field to be a mapping, while the same sentinel tokens are already
    # represented by T5's ``extra_ids`` setting. Override only the incompatible
    # redundant field; do not rewrite the authenticated snapshot.
    tokenizer = AutoTokenizer.from_pretrained(snapshot, extra_special_tokens={})
    dtype = torch.float16 if device.type == "mps" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(snapshot, torch_dtype=dtype)
    model = model.to(device).eval()
    sync(device)
    preparation_seconds = time.perf_counter() - preparation_started

    results = []
    for row in suite:
        started = time.perf_counter()
        hypothesis = translate(
            model, tokenizer, row, device, args.max_tokens, args.num_beams
        )
        first_latency = time.perf_counter() - started
        warm_latencies = []
        for _ in range(args.warm_runs):
            warm_started = time.perf_counter()
            translate(model, tokenizer, row, device, args.max_tokens, args.num_beams)
            warm_latencies.append(time.perf_counter() - warm_started)
        results.append(
            {
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
        )

    report = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "engine": f"transformers-t5:{args.repository}@{args.revision[:12]}:{args.device}",
        "hardware": hardware_name(),
        "operatingSystem": platform.platform(),
        "preparationSeconds": preparation_seconds,
        "peakResidentBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "modelBytes": directory_bytes(snapshot),
        "physicalModelCount": 1,
        "modelRevision": f"huggingface-revision:{args.revision}",
        "benchmarkConfiguration": {
            "decoding": "greedy" if args.num_beams == 1 else f"beam-{args.num_beams}",
            "prompt": "translate {source_code} to {target_code}: {source}",
            "maxNewTokens": args.max_tokens,
            "numBeams": args.num_beams,
            "warmRuns": args.warm_runs,
            "tokenizerCompatibilityOverride": "extra_special_tokens={} (extra_ids retained)",
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(results)} cases to {args.output}")


if __name__ == "__main__":
    main()
