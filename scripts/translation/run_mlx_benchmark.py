#!/usr/bin/env python3
"""Run an MLX-LM translation candidate against a Mimi JSONL suite."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler


DEFAULT_MODEL = "mlx-community/Qwen3-0.6B-4bit"
DEFAULT_REVISION = "73e3e38d981303bc594367cd910ea6eb48349da8"


def directory_bytes(path: Path) -> int:
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file()
        and "__pycache__" not in item.parts
        and item.suffix != ".pyc"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def authenticated_model_files(path: Path) -> dict[str, dict[str, int | str]]:
    names = (
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "sherry_model.py",
    )
    return {
        name: {"bytes": (path / name).stat().st_size, "sha256": sha256(path / name)}
        for name in names
        if (path / name).is_file()
    }


def adapter_bytes(path: Path) -> int:
    return sum(
        item.stat().st_size
        for item in (path / "adapter_config.json", path / "adapters.safetensors")
        if item.is_file()
    )


def load_suite(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit("benchmark suite is empty")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("benchmark suite contains duplicate IDs")
    return rows


def instruction_for(identifier: str) -> str:
    if identifier.startswith("ja"):
        return "Translate this English live-transcript segment into natural Japanese. Output only the translation."
    if identifier.startswith("en"):
        return "Translate this Japanese live-transcript segment into natural English. Output only the translation."
    raise SystemExit(f"unsupported language identifier: {identifier}")


def prompt_for(tokenizer, row: dict, profile: str) -> str:
    if profile in {"cat-translate", "hymt2", "hymt2-mimi", "lmt60"}:
        names = {"en-US": "English", "ja-JP": "Japanese"}
        try:
            source_name = names[row["sourceLanguage"]]
            target_name = names[row["targetLanguage"]]
        except KeyError as error:
            raise SystemExit(
                f"unsupported {profile} language: {error.args[0]}"
            ) from error
        if profile == "lmt60":
            content = (
                f"Translate the following text from {source_name} into "
                f"{target_name}:\n{source_name}: {row['source']}\n{target_name}:"
            )
        elif profile == "hymt2-mimi":
            content = (
                f"Translate the following text into {target_name} accurately and "
                "naturally for a live caption. Preserve the full meaning, including "
                "idioms, negation, hedging, politeness, named entities, numbers, "
                "dates, times, keyboard keys, UI terms, placeholders, and "
                "code-switched terms. Do not add or omit information. Note that you "
                "should only output the translated result without any additional "
                f"explanation:\n\n{row['source']}"
            )
        elif profile == "hymt2":
            content = (
                f"Translate the following text into {target_name}. Note that you "
                "should only output the translated result without any additional "
                f"explanation:\n\n{row['source']}"
            )
        else:
            content = (
                f"Translate the following {source_name} text into {target_name}.\n\n"
                f"{row['source']}"
            )
        messages = [{
            "role": "user",
            "content": content,
        }]
    else:
        messages = [
            {
                "role": "system",
                "content": instruction_for(row["targetLanguage"]),
            },
            {
                "role": "user",
                "content": row["source"],
            },
        ]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def clean_output(text: str) -> str:
    text = re.sub(r"(?s)^.*?</think>\s*", "", text)
    return text.strip().strip('"').strip()


def hardware_name() -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.machine()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--warm-runs", type=int, default=3)
    parser.add_argument("--direction", choices=("en-ja", "ja-en"))
    parser.add_argument(
        "--prompt-profile",
        choices=("mimi", "cat-translate", "hymt2", "hymt2-mimi", "lmt60"),
        default="mimi",
    )
    parser.add_argument(
        "--local-revision",
        help="authenticated repository revision for a local model snapshot",
    )
    parser.add_argument("--hf-home", type=Path, default=Path("Research/translation/models/hf-cache"))
    args = parser.parse_args()

    args.hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(args.hf_home.resolve())
    suite = load_suite(args.suite)
    if args.direction is not None:
        expected = {
            "en-ja": ("en-US", "ja-JP"),
            "ja-en": ("ja-JP", "en-US"),
        }[args.direction]
        suite = [
            row
            for row in suite
            if (row["sourceLanguage"], row["targetLanguage"]) == expected
        ]
        if not suite:
            raise SystemExit(f"suite has no cases for direction: {args.direction}")

    preparation_started = time.perf_counter()
    local_model = Path(args.model)
    if local_model.is_dir():
        snapshot = local_model.resolve()
        engine_revision = args.local_revision or "local"
    else:
        snapshot = Path(
            snapshot_download(
                repo_id=args.model,
                revision=args.revision,
                cache_dir=args.hf_home,
            )
        )
        engine_revision = args.revision[:12]
    model, tokenizer = load(
        str(snapshot),
        adapter_path=str(args.adapter_path) if args.adapter_path else None,
    )
    mx.eval(model.parameters())
    preparation_seconds = time.perf_counter() - preparation_started

    sampler = make_sampler(temp=0.0)
    results = []
    for row in suite:
        prompt = prompt_for(tokenizer, row, args.prompt_profile)
        started = time.perf_counter()
        hypothesis = clean_output(
            generate(
                model,
                tokenizer,
                prompt,
                max_tokens=args.max_tokens,
                sampler=sampler,
                verbose=False,
            )
        )
        latency = time.perf_counter() - started
        warm_latencies = []
        for _ in range(args.warm_runs):
            warm_started = time.perf_counter()
            generate(
                model,
                tokenizer,
                prompt,
                max_tokens=args.max_tokens,
                sampler=sampler,
                verbose=False,
            )
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
                "latencySeconds": latency,
                "warmLatencySeconds": warm_latencies,
            }
        )

    report = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "engine": (
            f"mlx-lm:{args.model}@{engine_revision}:prompt-{args.prompt_profile}"
            + (":lora" if args.adapter_path else ":base")
        ),
        "hardware": hardware_name(),
        "operatingSystem": platform.platform(),
        "preparationSeconds": preparation_seconds,
        "peakResidentBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "modelBytes": directory_bytes(snapshot) + (adapter_bytes(args.adapter_path) if args.adapter_path else 0),
        "physicalModelCount": 1,
        "modelRevision": (
            f"huggingface-revision:{engine_revision}"
            if engine_revision != "local"
            else "local-unpinned"
        ),
        "benchmarkConfiguration": {
            "decoding": "greedy",
            "maxNewTokens": args.max_tokens,
            "warmRuns": args.warm_runs,
            "promptProfile": args.prompt_profile,
        },
        "runtimeImplementation": {
            "runner": str(Path(__file__)),
            "runnerSha256": sha256(Path(__file__)),
            "mlxVersion": importlib.metadata.version("mlx"),
            "mlxLMVersion": importlib.metadata.version("mlx-lm"),
            "modelFiles": authenticated_model_files(snapshot),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(results)} cases to {args.output}")


if __name__ == "__main__":
    main()
