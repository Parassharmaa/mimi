#!/usr/bin/env python3
"""Benchmark the pinned MIT M2M-100 418M checkpoint in both Mimi directions."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import resource
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from typed_critical_token_policy import single_percentage_preserves  # noqa: E402


DEFAULT_REPOSITORY = "facebook/m2m100_418M"
DEFAULT_REVISION = "55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636"
LANGUAGE_CODES = {"en-US": "en", "ja-JP": "ja"}
MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "pytorch_model.bin",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json",
)
EXPECTED_WEIGHT_BYTES = 1_935_796_948
EXPECTED_WEIGHT_SHA256 = "d907ea45e4e4b9db163382a6674f6218b3c59566fe06d77f4055c208b4e87ed1"
STRICT_CRITICAL_TOKEN_RE = re.compile(
    r"https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+"
    r"|\{[^{}]+\}|%[A-Za-z]|<[A-Za-z][^<>]*>|%"
    r"|(?<![\d.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)*(?!\d|\.\d)"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


def strict_tokens(value: str) -> list[str]:
    return sorted(
        token.replace(",", "")
        for token in STRICT_CRITICAL_TOKEN_RE.findall(
            unicodedata.normalize("NFKC", value)
        )
    )


def preserves_critical_tokens(source: str, output: str) -> bool:
    return strict_tokens(source) == strict_tokens(output) or single_percentage_preserves(
        source, output
    )


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


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def load_suite(path: Path) -> list[dict]:
    if not path.is_file() or path.is_symlink():
        raise SystemExit("benchmark suite is missing or a symlink")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    identifiers = [row.get("id") for row in rows]
    if not rows or any(not isinstance(value, str) or not value for value in identifiers):
        raise SystemExit("benchmark suite has missing IDs")
    if len(identifiers) != len(set(identifiers)):
        raise SystemExit("benchmark suite has duplicate IDs")
    for row in rows:
        if (
            row.get("sourceLanguage") not in LANGUAGE_CODES
            or row.get("targetLanguage") not in LANGUAGE_CODES
            or row.get("sourceLanguage") == row.get("targetLanguage")
            or not isinstance(row.get("source"), str)
            or not row["source"].strip()
            or not isinstance(row.get("references"), list)
        ):
            raise SystemExit(f"invalid benchmark row: {row.get('id')}")
    return rows


def translate(
    model,
    tokenizer,
    row: dict,
    device: torch.device,
    max_tokens: int,
    num_beams: int,
) -> tuple[str, list[int]]:
    tokenizer.src_lang = LANGUAGE_CODES[row["sourceLanguage"]]
    encoded = tokenizer(row["source"], return_tensors="pt", truncation=True).to(device)
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.get_lang_id(
                LANGUAGE_CODES[row["targetLanguage"]]
            ),
            do_sample=False,
            num_beams=num_beams,
            use_cache=True,
            max_new_tokens=max_tokens,
        )
    sync(device)
    output_ids = output[0].tolist()
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip(), output_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--warm-runs", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--num-beams", type=int, choices=range(1, 6), default=1)
    parser.add_argument(
        "--hf-home", type=Path, default=Path("Research/translation/models/hf-cache")
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"refusing to overwrite M2M-100 report: {args.output}")
    if args.warm_runs < 0 or args.max_tokens <= 0:
        raise SystemExit("warm-runs must be non-negative and max-tokens positive")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable; pass --device cpu")
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
                local_files_only=args.local_files_only,
            )
        )
    )
    missing = [name for name in MODEL_FILES if not (snapshot / name).is_file()]
    if missing:
        raise SystemExit(f"pinned M2M-100 snapshot is incomplete: {missing}")
    weight_path = snapshot / "pytorch_model.bin"
    if (
        weight_path.stat().st_size != EXPECTED_WEIGHT_BYTES
        or sha256(weight_path) != EXPECTED_WEIGHT_SHA256
    ):
        raise SystemExit("M2M-100 checkpoint differs from the pinned revision")
    tokenizer = M2M100Tokenizer.from_pretrained(snapshot)
    dtype = torch.float16 if device.type == "mps" else torch.float32
    model = M2M100ForConditionalGeneration.from_pretrained(
        snapshot, torch_dtype=dtype
    ).to(device).eval()
    sync(device)
    preparation_seconds = time.perf_counter() - preparation_started

    results = []
    critical_failures: Counter[str] = Counter()
    direction_latencies: dict[str, list[float]] = {"en-ja": [], "ja-en": []}
    for row in suite:
        started = time.perf_counter()
        hypothesis, output_ids = translate(
            model, tokenizer, row, device, args.max_tokens, args.num_beams
        )
        first_latency = time.perf_counter() - started
        warm_latencies = []
        for _ in range(args.warm_runs):
            warm_started = time.perf_counter()
            translate(model, tokenizer, row, device, args.max_tokens, args.num_beams)
            warm_latencies.append(time.perf_counter() - warm_started)
        direction = "en-ja" if row["sourceLanguage"] == "en-US" else "ja-en"
        direction_latencies[direction].extend(warm_latencies or [first_latency])
        preserved = preserves_critical_tokens(row["source"], hypothesis)
        if not preserved:
            critical_failures[direction] += 1
        results.append(
            {
                "caseID": row["id"],
                "sourceLanguage": row["sourceLanguage"],
                "targetLanguage": row["targetLanguage"],
                "domain": row["domain"],
                "source": row["source"],
                "references": row["references"],
                "claimEligible": bool(row.get("claimEligible")),
                "hypothesis": hypothesis,
                "outputTokenIDs": output_ids,
                "latencySeconds": first_latency,
                "warmLatencySeconds": warm_latencies,
                "criticalTokensPreserved": preserved,
                "sourceCriticalTokens": strict_tokens(row["source"]),
                "hypothesisCriticalTokens": strict_tokens(hypothesis),
            }
        )

    files = {name: {**{"bytes": (snapshot / name).stat().st_size}, "sha256": sha256(snapshot / name)} for name in MODEL_FILES}
    report = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "engine": f"transformers-m2m100:{args.repository}@{args.revision[:12]}:{args.device}:beam{args.num_beams}",
        "license": "MIT",
        "hardware": hardware_name(),
        "operatingSystem": platform.platform(),
        "preparationSeconds": preparation_seconds,
        "peakResidentBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "modelBytes": directory_bytes(snapshot),
        "physicalModelCount": 1,
        "modelRevision": f"huggingface-revision:{args.revision}",
        "suite": {"path": args.suite.as_posix(), "sha256": sha256(args.suite)},
        "files": files,
        "benchmarkConfiguration": {
            "decoding": "greedy" if args.num_beams == 1 else f"beam-{args.num_beams}",
            "maxNewTokens": args.max_tokens,
            "numBeams": args.num_beams,
            "warmRuns": args.warm_runs,
            "dtype": str(dtype),
        },
        "runtimeSafety": {
            "criticalTokenFailures": dict(sorted(critical_failures.items())),
            "directionLatency": {
                direction: {
                    "samples": len(values),
                    "p50Seconds": percentile(values, 0.50),
                    "p95Seconds": percentile(values, 0.95),
                }
                for direction, values in direction_latencies.items()
            },
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "cases": len(results),
                "criticalTokenFailures": dict(critical_failures),
                "directionLatency": report["runtimeSafety"]["directionLatency"],
                "modelBytes": report["modelBytes"],
                "peakResidentBytes": report["peakResidentBytes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
