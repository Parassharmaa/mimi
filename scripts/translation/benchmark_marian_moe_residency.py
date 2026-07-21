#!/usr/bin/env python3
"""Measure worst-case loaded residency and router cost for a Mimi Marian MoE pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parent))
from marian_mlx import load_model  # noqa: E402
from source_expert_router import SourceExpertRouter  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("suite", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--router-repetitions", type=int, default=20)
    args = parser.parse_args()
    if args.router_repetitions < 1:
        raise SystemExit("router repetitions must be positive")

    manifest_path = args.bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pack_format = manifest.get("format")
    if pack_format not in {
        "mimi-mlx-marian-moe-v1",
        "mimi-mlx-marian-moe-v2",
    }:
        raise SystemExit("unsupported Marian MoE manifest")
    for relative, record in manifest["files"].items():
        path = args.bundle / relative
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256(path) != record["sha256"]
        ):
            raise SystemExit(f"bundle integrity failure: {relative}")
    bundle_bytes = sum(
        item.stat().st_size for item in args.bundle.rglob("*") if item.is_file()
    )
    if bundle_bytes >= 150_000_000:
        raise SystemExit(f"bundle exceeds preferred ceiling: {bundle_bytes}")

    rows = [
        json.loads(line)
        for line in args.suite.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    samples = {
        "en-ja": next(row["source"] for row in rows if row["sourceLanguage"] == "en-US"),
        "ja-en": next(row["source"] for row in rows if row["sourceLanguage"] == "ja-JP"),
    }
    engine_paths = {
        "generalist-en-ja": args.bundle / manifest["generalists"]["en-ja"],
        "generalist-ja-en": args.bundle / manifest["generalists"]["ja-en"],
        "formal-en-ja": args.bundle / manifest["experts"]["en-ja"]["engine"],
        "legal-ja-en": args.bundle / manifest["experts"]["ja-en"]["engine"],
    }
    shared_tokenizer = None
    if pack_format == "mimi-mlx-marian-moe-v2":
        relative = manifest.get("sharedTokenizer")
        if not isinstance(relative, str) or relative not in manifest["files"]:
            raise SystemExit("invalid shared-tokenizer Marian MoE manifest")
        shared_tokenizer = args.bundle / relative
    loaded = {}
    load_started = time.perf_counter()
    for name, path in engine_paths.items():
        engine_manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        model = load_model(
            path / "model.safetensors",
            quantization_bits=int(engine_manifest["bits"]),
            quantization_group_size=int(engine_manifest["group_size"]),
        )
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(shared_tokenizer or path / "tokenizer.json"),
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
        )
        loaded[name] = (model, tokenizer, engine_manifest["direction"])
    mx.synchronize()
    load_seconds = time.perf_counter() - load_started

    smoke_latencies = {}
    for name, (model, tokenizer, direction) in loaded.items():
        encoded = tokenizer.encode(samples[direction])
        started = time.perf_counter()
        model.generate_cached(encoded, 192)
        mx.synchronize()
        smoke_latencies[name] = time.perf_counter() - started

    router_latencies = {}
    route_counts = {}
    for direction in ("en-ja", "ja-en"):
        router_path = args.bundle / manifest["experts"][direction]["router"]
        router = SourceExpertRouter.load(router_path)
        directional_sources = [
            row["source"]
            for row in rows
            if (row["sourceLanguage"] == "en-US") == (direction == "en-ja")
        ]
        for source in directional_sources:
            router.routes_to_expert(source)
        latencies = []
        routed = 0
        for _ in range(args.router_repetitions):
            for source in directional_sources:
                started = time.perf_counter()
                selected = router.routes_to_expert(source)
                latencies.append(time.perf_counter() - started)
                routed += int(selected)
        router_latencies[direction] = {
            "calls": len(latencies),
            "p50Seconds": percentile(latencies, 0.50),
            "p95Seconds": percentile(latencies, 0.95),
        }
        route_counts[direction] = routed // args.router_repetitions

    report = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "purpose": "development-only four-engine loaded-residency and router-cost measurement",
        "promotionEligible": False,
        "hardware": platform.machine(),
        "bundle": {
            "path": str(args.bundle.resolve()),
            "manifestSha256": sha256(manifest_path),
            "bytes": bundle_bytes,
            "physicalModelCount": 4,
        },
        "loadSeconds": load_seconds,
        "peakResidentBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "firstSmokeLatencySeconds": smoke_latencies,
        "routerLatency": router_latencies,
        "routeCountsOnPublicSuite": route_counts,
        "doesNotAuthorizeAppIntegration": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
