#!/usr/bin/env python3
"""Benchmark a local quantized MLX Marian pair."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import mlx.core as mx
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parent))
from marian_mlx import DIMENSIONS, POSITION_TABLE_LENGTH, load_model  # noqa: E402


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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_file_table(root: Path, files: dict, label: str) -> None:
    if not isinstance(files, dict) or not files:
        raise SystemExit(f"{label} manifest has no file integrity table")
    for relative, expected in files.items():
        path = root / relative
        if not path.is_file():
            raise SystemExit(f"{label} manifest-listed file is missing: {relative}")
        if not isinstance(expected, dict):
            raise SystemExit(f"{label} manifest has invalid file record: {relative}")
        if path.stat().st_size != expected.get("bytes") or sha256(path) != expected.get("sha256"):
            raise SystemExit(f"{label} manifest integrity failure: {relative}")


def shared_tokenizer_pack_root(engine: Path) -> Path | None:
    """Return the authenticated MoE-v2 root for a deduplicated engine."""

    if engine.parent.name != "engines":
        return None
    root = engine.parent.parent
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "mimi-mlx-marian-moe-v2":
        return None
    relative = engine.relative_to(root).as_posix()
    declared_engines = {
        str(value)
        for value in manifest.get("generalists", {}).values()
        if isinstance(value, str)
    }
    declared_engines.update(
        str(value.get("engine"))
        for value in manifest.get("experts", {}).values()
        if isinstance(value, dict) and isinstance(value.get("engine"), str)
    )
    if relative not in declared_engines:
        raise SystemExit(f"engine is not declared by shared-tokenizer pack: {engine}")
    return root


def tokenizer_path_for_model(model_path: Path, manifest: dict) -> Path:
    direct = model_path / "tokenizer.json"
    if direct.is_file():
        return direct
    root = shared_tokenizer_pack_root(model_path)
    if root is None:
        raise SystemExit(f"model has no tokenizer.json or shared-tokenizer pack: {model_path}")
    root_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    relative = root_manifest.get("sharedTokenizer")
    relative_path = PurePosixPath(relative) if isinstance(relative, str) else None
    if (
        relative_path is None
        or relative_path.is_absolute()
        or any(part in ("", ".", "..") for part in relative_path.parts)
    ):
        raise SystemExit(f"invalid shared tokenizer path: {root}")
    tokenizer = root / relative_path
    expected = root_manifest.get("files", {}).get(relative)
    engine_expected = manifest.get("shared_tokenizer")
    actual = {"bytes": tokenizer.stat().st_size, "sha256": sha256(tokenizer)}
    if expected != actual or engine_expected != actual:
        raise SystemExit(f"shared tokenizer authentication failed: {model_path}")
    return tokenizer


def exact_model_revision(paths: dict[tuple[str, str], Path]) -> str:
    unique_paths = set(paths.values())
    if len(unique_paths) == 1:
        path = next(iter(unique_paths))
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise SystemExit(f"single model lacks manifest: {path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("direction") != "bidirectional":
            raise SystemExit("one physical model must declare direction=bidirectional")
        validate_file_table(path, manifest.get("files"), "single bidirectional model")
        return f"single-manifest-sha256:{sha256(manifest_path)}"

    shared_roots = {shared_tokenizer_pack_root(path) for path in paths.values()}
    if len(shared_roots) == 1 and None not in shared_roots:
        root = next(iter(shared_roots))
        root_manifest_path = root / "manifest.json"
        root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
        validate_file_table(root, root_manifest.get("files"), "shared-tokenizer MoE pack")
        return f"moe-manifest-sha256:{sha256(root_manifest_path)}"

    parents = {path.parent for path in paths.values()}
    if len(parents) == 1:
        root = next(iter(parents))
        root_manifest_path = root / "manifest.json"
        if root_manifest_path.is_file():
            root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
            if root_manifest.get("format") != "mimi-mlx-marian-pair-v1":
                raise SystemExit("model pair root manifest has an unsupported format")
            validate_file_table(root, root_manifest.get("files"), "model pair")
            return f"pair-manifest-sha256:{sha256(root_manifest_path)}"

    digests: list[str] = []
    for direction, path in sorted(paths.items()):
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise SystemExit(f"model direction lacks manifest: {path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_file_table(path, manifest.get("files"), f"model {direction}")
        digests.append(f"{direction[0]}>{direction[1]}:{sha256(manifest_path)}")
    combined = hashlib.sha256("\n".join(digests).encode()).hexdigest()
    return f"direction-manifests-sha256:{combined}"


def model_bundle_bytes(paths: list[Path]) -> int:
    paths = list(dict.fromkeys(paths))
    shared_roots = {shared_tokenizer_pack_root(path) for path in paths}
    if len(shared_roots) == 1 and None not in shared_roots:
        return directory_bytes(next(iter(shared_roots)))
    parents = {path.parent for path in paths}
    if len(parents) == 1:
        parent = next(iter(parents))
        if (parent / "manifest.json").is_file():
            return directory_bytes(parent)
    return sum(directory_bytes(path) for path in paths)


def hardware_name() -> str:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.machine()


def runtime_implementation() -> dict:
    """Bind benchmark outputs to the exact Python MLX inference implementation."""
    script_directory = Path(__file__).resolve().parent
    return {
        "benchmarkScriptSha256": sha256(Path(__file__).resolve()),
        "marianRuntimeSha256": sha256(script_directory / "marian_mlx.py"),
        "pythonVersion": platform.python_version(),
        "packages": {
            package: importlib.metadata.version(package)
            for package in ("mlx", "tokenizers", "transformers")
        },
    }


def declared_model_records(paths: dict[tuple[str, str], Path]) -> dict[str, dict]:
    names = {("en-US", "ja-JP"): "en-ja", ("ja-JP", "en-US"): "ja-en"}
    output = {}
    for direction, path in sorted(paths.items()):
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        output[names[direction]] = {
            "path": str(path),
            "manifestSha256": sha256(manifest_path),
            "sourceWeightsSha256": manifest.get("source_weights_sha256"),
            "quantizedWeightsSha256": manifest.get("files", {})
            .get("model.safetensors", {})
            .get("sha256"),
        }
    return output


def generate(
    model,
    encoded: list[int],
    *,
    beam_size: int,
    max_tokens: int,
    cached_decoding: bool,
    preallocated_kv_cache_block_size: int | None,
) -> list[int]:
    if cached_decoding:
        return model.generate_cached(
            encoded,
            max_tokens,
            self_cache_block_size=preallocated_kv_cache_block_size,
        )
    if beam_size > 1:
        return model.generate_beam(encoded, beam_size, max_tokens)
    return model.generate(encoded, max_tokens)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--en-ja-model", type=Path, required=True)
    parser.add_argument("--ja-en-model", type=Path, required=True)
    parser.add_argument("--warm-runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument(
        "--direction",
        choices=("en-ja", "ja-en"),
        help=(
            "Benchmark only one direction while still binding the report to both "
            "declared direction manifests."
        ),
    )
    parser.add_argument(
        "--cached-decoding",
        action="store_true",
        help="Use incremental greedy K/V caching instead of full-prefix decoding.",
    )
    parser.add_argument(
        "--preallocated-kv-cache-block-size",
        type=int,
        help=(
            "Opt into block-growing decoder self-K/V storage with this capacity "
            "increment; requires --cached-decoding."
        ),
    )
    parser.add_argument(
        "--precomputed-position-table",
        action="store_true",
        help=(
            "Opt into one runtime-cached 192x512 sinusoidal table for encoder and "
            "incremental decoder positions; requires --cached-decoding."
        ),
    )
    args = parser.parse_args()
    if args.warm_runs < 0:
        raise SystemExit("warm runs must be non-negative")
    if args.max_tokens < 1:
        raise SystemExit("max tokens must be positive")
    if args.beam_size < 1:
        raise SystemExit("beam size must be at least one")
    if args.cached_decoding and args.beam_size != 1:
        raise SystemExit("cached decoding currently supports greedy beam-size 1 only")
    if args.preallocated_kv_cache_block_size is not None:
        if not args.cached_decoding:
            raise SystemExit("preallocated K/V cache requires --cached-decoding")
        if args.preallocated_kv_cache_block_size < 1:
            raise SystemExit("preallocated K/V cache block size must be positive")
    if args.precomputed_position_table:
        if not args.cached_decoding:
            raise SystemExit("precomputed position table requires --cached-decoding")
        if args.max_tokens > POSITION_TABLE_LENGTH:
            raise SystemExit(
                "precomputed position table requires max tokens at or below "
                f"{POSITION_TABLE_LENGTH}"
            )

    suite = load_suite(args.suite)
    models = {
        ("en-US", "ja-JP"): args.en_ja_model,
        ("ja-JP", "en-US"): args.ja_en_model,
    }
    model_revision = exact_model_revision(models)
    requested_direction = {
        "en-ja": ("en-US", "ja-JP"),
        "ja-en": ("ja-JP", "en-US"),
    }.get(args.direction)
    active_models = (
        {requested_direction: models[requested_direction]}
        if requested_direction is not None
        else models
    )
    benchmark_suite = [
        row
        for row in suite
        if requested_direction is None
        or (row["sourceLanguage"], row["targetLanguage"]) == requested_direction
    ]
    if not benchmark_suite:
        raise SystemExit(f"benchmark suite has no {args.direction} cases")
    results_by_id: dict[str, dict] = {}
    preparation_seconds = 0.0
    quantizations: set[tuple[int, int]] = set()

    for direction, model_path in active_models.items():
        direction_rows = [
            row
            for row in benchmark_suite
            if (row["sourceLanguage"], row["targetLanguage"]) == direction
        ]
        manifest = json.loads((model_path / "manifest.json").read_text(encoding="utf-8"))
        direction_name = {
            ("en-US", "ja-JP"): "en-ja",
            ("ja-JP", "en-US"): "ja-en",
        }[direction]
        source_prefixes = manifest.get("source_prefixes") or {}
        source_prefix = source_prefixes.get(direction_name, "")
        quantization = (int(manifest["bits"]), int(manifest["group_size"]))
        quantizations.add(quantization)
        preparation_started = time.perf_counter()
        model = load_model(
            model_path / "model.safetensors",
            quantization_bits=quantization[0],
            quantization_group_size=quantization[1],
            precompute_position_table=args.precomputed_position_table,
        )
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_path_for_model(model_path, manifest)),
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
        )
        preparation_seconds += time.perf_counter() - preparation_started

        if args.precomputed_position_table:
            encoded_rows = [
                (row, tokenizer.encode(source_prefix + row["source"]))
                for row in direction_rows
            ]
            oversized = next(
                (
                    (row, encoded)
                    for row, encoded in encoded_rows
                    if len(encoded) > POSITION_TABLE_LENGTH
                ),
                None,
            )
            if oversized is not None:
                row, encoded = oversized
                raise SystemExit(
                    "precomputed position table supports source token length at most "
                    f"{POSITION_TABLE_LENGTH}; case {row['id']} encoded to {len(encoded)}"
                )
        else:
            # Preserve the established default path: tokenize each case directly
            # before its first untimed model invocation.
            encoded_rows = (
                (row, tokenizer.encode(source_prefix + row["source"]))
                for row in direction_rows
            )

        for row, encoded in encoded_rows:
            started = time.perf_counter()
            output_ids = generate(
                model,
                encoded,
                beam_size=args.beam_size,
                max_tokens=args.max_tokens,
                cached_decoding=args.cached_decoding,
                preallocated_kv_cache_block_size=args.preallocated_kv_cache_block_size,
            )
            mx.synchronize()
            hypothesis = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            first_latency = time.perf_counter() - started
            warm_latencies = []
            for _ in range(args.warm_runs):
                warm_started = time.perf_counter()
                generate(
                    model,
                    encoded,
                    beam_size=args.beam_size,
                    max_tokens=args.max_tokens,
                    cached_decoding=args.cached_decoding,
                    preallocated_kv_cache_block_size=args.preallocated_kv_cache_block_size,
                )
                mx.synchronize()
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
                "outputTokenIDs": output_ids,
                "latencySeconds": first_latency,
                "warmLatencySeconds": warm_latencies,
            }

        del model, tokenizer
        gc.collect()
        mx.clear_cache()

    if len(quantizations) != 1:
        raise SystemExit("direction models use different quantization")
    bits, group_size = quantizations.pop()
    physical_model_count = len(set(active_models.values()))
    model_layout = "single-direction" if args.direction is not None else "pair"
    if args.preallocated_kv_cache_block_size is not None:
        decoding = f"kv-cache-preallocated-b{args.preallocated_kv_cache_block_size}"
    else:
        decoding = "kv-cache" if args.cached_decoding else "full-prefix"
    if args.precomputed_position_table:
        decoding += f"-pos-table{POSITION_TABLE_LENGTH}"
    report = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "engine": (
            f"mlx:ElanMT-BT-{model_layout}:{bits}bit-g{group_size}-"
            f"beam{args.beam_size}-{decoding}"
        ),
        "modelRevision": model_revision,
        "declaredModels": declared_model_records(models),
        "hardware": hardware_name(),
        "operatingSystem": platform.platform(),
        "preparationSeconds": preparation_seconds,
        "peakResidentBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "modelBytes": model_bundle_bytes(list(active_models.values())),
        "physicalModelCount": physical_model_count,
        "runtimeImplementation": runtime_implementation(),
        "benchmarkConfiguration": {
            "warmRunsPerCase": args.warm_runs,
            "maximumGeneratedTokens": args.max_tokens,
            "direction": args.direction,
        },
        "decoderSelfKVCache": {
            "strategy": (
                "block-growing-preallocated"
                if args.preallocated_kv_cache_block_size is not None
                else ("concatenate" if args.cached_decoding else "none")
            ),
            "blockSize": args.preallocated_kv_cache_block_size,
            "crossAttentionImmutable": bool(args.cached_decoding),
        },
        "positionEmbeddings": {
            "strategy": (
                "precomputed-runtime-table"
                if args.precomputed_position_table
                else "dynamic-sinusoidal"
            ),
            "tableShape": (
                [POSITION_TABLE_LENGTH, DIMENSIONS]
                if args.precomputed_position_table
                else None
            ),
            "appliedTo": (
                ["encoder", "cached-decoder"]
                if args.precomputed_position_table
                else []
            ),
        },
        "results": [results_by_id[row["id"]] for row in benchmark_suite],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(benchmark_suite)} cases to {args.output}")


if __name__ == "__main__":
    main()
