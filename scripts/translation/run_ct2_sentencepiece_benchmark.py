#!/usr/bin/env python3
"""Benchmark an authenticated CTranslate2 pair with separate SentencePiece models."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import ctranslate2
import sentencepiece as spm


REQUIRED_FILES = (
    "config.json",
    "model.bin",
    "source_vocabulary.json",
    "target_vocabulary.json",
    "src.spm.model",
    "tgt.spm.model",
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
    return rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def authenticate_model(path: Path, revision: str) -> dict:
    if not revision or len(revision) != 40:
        raise SystemExit(f"model revision must be a full 40-character commit: {path}")
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        raise SystemExit(f"model is missing {', '.join(missing)}: {path}")
    return {
        "path": str(path),
        "revision": revision,
        "bytes": directory_bytes(path),
        "files": {
            name: {
                "bytes": (path / name).stat().st_size,
                "sha256": sha256(path / name),
            }
            for name in REQUIRED_FILES
        },
    }


def hardware_name() -> str:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.machine()


def translate(
    translator: ctranslate2.Translator,
    source_tokenizer: spm.SentencePieceProcessor,
    target_tokenizer: spm.SentencePieceProcessor,
    text: str,
    *,
    beam_size: int,
    max_tokens: int,
) -> tuple[str, list[str]]:
    # Match quickmt/quickmt's pinned Translator: SentencePiece pieces plus source EOS.
    source_tokens = source_tokenizer.encode(text, out_type=str) + ["</s>"]
    result = translator.translate_batch(
        [source_tokens],
        beam_size=beam_size,
        max_decoding_length=max_tokens,
        disable_unk=True,
        return_scores=False,
    )[0]
    output_tokens = result.hypotheses[0]
    return target_tokenizer.decode(output_tokens).strip(), output_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--en-ja-model", type=Path, required=True)
    parser.add_argument("--ja-en-model", type=Path, required=True)
    parser.add_argument("--en-ja-revision", required=True)
    parser.add_argument("--ja-en-revision", required=True)
    parser.add_argument("--warm-runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--compute-type", default="int8")
    args = parser.parse_args()
    if args.warm_runs < 0:
        raise SystemExit("warm runs must be non-negative")
    if args.max_tokens < 1 or args.beam_size < 1 or args.threads < 1:
        raise SystemExit("max tokens, beam size, and threads must be positive")

    suite = load_suite(args.suite)
    models = {
        ("en-US", "ja-JP"): (args.en_ja_model, args.en_ja_revision),
        ("ja-JP", "en-US"): (args.ja_en_model, args.ja_en_revision),
    }
    identities = {
        "en-ja": authenticate_model(args.en_ja_model, args.en_ja_revision),
        "ja-en": authenticate_model(args.ja_en_model, args.ja_en_revision),
    }
    results_by_id: dict[str, dict] = {}
    preparation_seconds = 0.0

    for direction, (model_path, _) in models.items():
        direction_rows = [
            row
            for row in suite
            if (row["sourceLanguage"], row["targetLanguage"]) == direction
        ]
        preparation_started = time.perf_counter()
        translator = ctranslate2.Translator(
            str(model_path),
            device="cpu",
            compute_type=args.compute_type,
            inter_threads=1,
            intra_threads=args.threads,
        )
        source_tokenizer = spm.SentencePieceProcessor(
            model_file=str(model_path / "src.spm.model")
        )
        target_tokenizer = spm.SentencePieceProcessor(
            model_file=str(model_path / "tgt.spm.model")
        )
        preparation_seconds += time.perf_counter() - preparation_started

        for row in direction_rows:
            started = time.perf_counter()
            hypothesis, output_tokens = translate(
                translator,
                source_tokenizer,
                target_tokenizer,
                row["source"],
                beam_size=args.beam_size,
                max_tokens=args.max_tokens,
            )
            first_latency = time.perf_counter() - started
            warm_latencies = []
            for _ in range(args.warm_runs):
                warm_started = time.perf_counter()
                translate(
                    translator,
                    source_tokenizer,
                    target_tokenizer,
                    row["source"],
                    beam_size=args.beam_size,
                    max_tokens=args.max_tokens,
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
                "outputTokens": output_tokens,
                "latencySeconds": first_latency,
                "warmLatencySeconds": warm_latencies,
            }

        del translator, source_tokenizer, target_tokenizer
        gc.collect()

    runtime_script = Path(__file__).resolve()
    report = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "engine": (
            f"ctranslate2:{ctranslate2.__version__}:sentencepiece-pair:"
            f"{args.compute_type}:beam{args.beam_size}"
        ),
        "modelRevision": f"en-ja:{args.en_ja_revision};ja-en:{args.ja_en_revision}",
        "hardware": hardware_name(),
        "operatingSystem": platform.platform(),
        "preparationSeconds": preparation_seconds,
        "peakResidentBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "modelBytes": sum(identity["bytes"] for identity in identities.values()),
        "physicalModelCount": 2,
        "modelIdentities": identities,
        "runtimeImplementation": {
            "runner": str(runtime_script),
            "runnerSha256": sha256(runtime_script),
            "ctranslate2Version": ctranslate2.__version__,
            "sentencepieceVersion": importlib.metadata.version("sentencepiece"),
        },
        "benchmarkConfiguration": {
            "beamSize": args.beam_size,
            "computeType": args.compute_type,
            "maximumGeneratedTokens": args.max_tokens,
            "threads": args.threads,
            "warmRunsPerCase": args.warm_runs,
        },
        "results": [results_by_id[row["id"]] for row in suite],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(suite)} cases to {args.output}")


if __name__ == "__main__":
    main()
