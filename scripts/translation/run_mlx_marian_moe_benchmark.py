#!/usr/bin/env python3
"""Run an authenticated Mimi Marian MoE pack on a frozen benchmark suite.

This runner deliberately supports source-only suites. Empty references remain
empty and every row remains claim-ineligible; the resulting report is useful for
runtime, latency, routing, and deterministic structural-safety evidence only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import resource
import sys
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlx.core as mx
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parent))
from marian_mlx import CompositeOutputProjection, load_model  # noqa: E402
from marian_target_shortlist import MarianTargetShortlist  # noqa: E402
from source_expert_router import SourceExpertRouter  # noqa: E402
from typed_critical_token_policy import (  # noqa: E402
    single_percentage_preserves,
)


DIRECTION_BY_LANGUAGES = {
    ("en-US", "ja-JP"): "en-ja",
    ("ja-JP", "en-US"): "ja-en",
}
ROLE_BY_DIRECTION = {
    "en-ja": ("generalist-en-ja", "formal-en-ja"),
    "ja-en": ("generalist-ja-en", "legal-ja-en"),
}
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


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def normalize_memory_source(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def clean_output(value: str) -> str:
    output = value
    marker = output.rfind("</think>")
    if marker >= 0:
        output = output[marker + len("</think>") :]
    output = output.strip()
    if len(output) >= 2 and output[0] == output[-1] == '"':
        output = output[1:-1]
    return output.strip()


def strict_tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", value)
    return sorted(
        token.replace(",", "")
        for token in STRICT_CRITICAL_TOKEN_RE.findall(normalized)
    )


def preserves_critical_tokens(source: str, output: str) -> bool:
    return strict_tokens(source) == strict_tokens(output) or single_percentage_preserves(
        source, output
    )


def is_plausible(output: str, source: str, direction: str) -> bool:
    if not output or output == source or len(output) > max(64, len(source) * 5):
        return False
    if direction == "en-ja":
        return any(
            "\u3040" <= value <= "\u30ff" or "\u3400" <= value <= "\u9fff"
            for value in output
        )
    return any("A" <= value <= "Z" or "a" <= value <= "z" for value in output)


def load_suite(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit("benchmark suite is empty")
    identifiers = [str(row.get("id", "")) for row in rows]
    if "" in identifiers or len(identifiers) != len(set(identifiers)):
        raise SystemExit("benchmark suite has a missing or duplicate case ID")
    for row in rows:
        languages = (row.get("sourceLanguage"), row.get("targetLanguage"))
        if languages not in DIRECTION_BY_LANGUAGES:
            raise SystemExit(f"unsupported benchmark direction: {languages}")
        if not isinstance(row.get("source"), str) or not row["source"].strip():
            raise SystemExit(f"benchmark case has no source: {row['id']}")
        if not isinstance(row.get("references"), list):
            raise SystemExit(f"benchmark case has invalid references: {row['id']}")
    return rows


def validate_file_record(path: Path, record: Any, label: str) -> None:
    if not isinstance(record, dict):
        raise SystemExit(f"invalid file record: {label}")
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != record.get("bytes")
        or sha256(path) != record.get("sha256")
    ):
        raise SystemExit(f"bundle integrity failure: {label}")


def validate_pack(bundle: Path) -> dict[str, Any]:
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise SystemExit("Marian MoE pack lacks a regular root manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") not in {
        "mimi-mlx-marian-moe-v1",
        "mimi-mlx-marian-moe-v2",
    }:
        raise SystemExit("unsupported Marian MoE pack format")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise SystemExit("Marian MoE manifest has no file table")
    physical = {
        str(item.relative_to(bundle))
        for item in bundle.rglob("*")
        if item.is_file() and item != manifest_path
    }
    if set(files) != physical:
        missing = sorted(set(files) - physical)
        unlisted = sorted(physical - set(files))
        raise SystemExit(
            "Marian MoE file table is not exhaustive: "
            f"missing={missing}, unlisted={unlisted}"
        )
    for relative, record in files.items():
        validate_file_record(bundle / relative, record, relative)
    return manifest


def load_memory(
    bundle: Path, manifest: dict[str, Any]
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    metadata = manifest.get("translationMemory")
    if metadata is None:
        return {"en-ja": {}, "ja-en": {}}, {"present": False, "entries": 0}
    if not isinstance(metadata, dict) or not isinstance(metadata.get("path"), str):
        raise SystemExit("invalid translation-memory metadata")
    path = bundle / metadata["path"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries")
    if (
        payload.get("schemaVersion") != 1
        or payload.get("normalization") != "NFKC then Unicode-whitespace collapse"
        or set(entries or {}) != {"en-ja", "ja-en"}
        or not all(isinstance(value, dict) for value in entries.values())
    ):
        raise SystemExit("unsupported translation-memory artifact")
    for direction, values in entries.items():
        for source, target in values.items():
            if source != normalize_memory_source(source) or not preserves_critical_tokens(
                source, target
            ):
                raise SystemExit(f"unsafe translation-memory entry: {direction}:{source}")
    return entries, {
        "present": True,
        "path": metadata["path"],
        "sha256": sha256(path),
        "entries": sum(len(value) for value in entries.values()),
        "promotionEligible": False,
    }


def engine_paths(bundle: Path, manifest: dict[str, Any]) -> dict[str, Path]:
    generalists = manifest.get("generalists")
    experts = manifest.get("experts")
    if not isinstance(generalists, dict) or not isinstance(experts, dict):
        raise SystemExit("Marian MoE pack lacks generalists or experts")
    paths = {
        "generalist-en-ja": bundle / generalists["en-ja"],
        "generalist-ja-en": bundle / generalists["ja-en"],
        "formal-en-ja": bundle / experts["en-ja"]["engine"],
        "legal-ja-en": bundle / experts["ja-en"]["engine"],
    }
    if len(set(paths.values())) != 4:
        raise SystemExit("Marian MoE pack must contain four physical engines")
    return paths


def load_runtime(
    bundle: Path,
    manifest: dict[str, Any],
    *,
    pack_attention_projections: bool = False,
):
    paths = engine_paths(bundle, manifest)
    shared_relative = manifest.get("sharedTokenizer")
    tokenizer_path = bundle / shared_relative if isinstance(shared_relative, str) else None
    if tokenizer_path is None:
        tokenizer_path = paths["generalist-en-ja"] / "tokenizer.json"
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    models = {}
    source_prefixes = {}
    quantizations = set()
    for role, path in paths.items():
        engine_manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        bits = int(engine_manifest["bits"])
        group_size = int(engine_manifest["group_size"])
        quantizations.add((bits, group_size))
        models[role] = load_model(
            path / "model.safetensors",
            quantization_bits=bits,
            quantization_group_size=group_size,
            pack_attention_projections=pack_attention_projections,
        )
        direction = str(engine_manifest["direction"])
        source_prefixes[role] = (engine_manifest.get("source_prefixes") or {}).get(
            direction, ""
        )
    if len(quantizations) != 1:
        raise SystemExit("Marian MoE engines use different quantization contracts")
    mx.synchronize()
    return models, tokenizer, source_prefixes, quantizations.pop()


def load_routers(
    bundle: Path, manifest: dict[str, Any]
) -> dict[str, SourceExpertRouter]:
    routers = {}
    for direction in ("en-ja", "ja-en"):
        relative = manifest["experts"][direction]["router"]
        router = SourceExpertRouter.load(bundle / relative)
        if router.direction != direction:
            raise SystemExit(f"router direction mismatch: {relative}")
        routers[direction] = router
    return routers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("suite", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--warm-runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument(
        "--target-shortlist",
        type=Path,
        help="opt-in authenticated tokenizer-derived output projection shortlist",
    )
    parser.add_argument(
        "--packed-attention-projections",
        action="store_true",
        help="opt in to exact concatenated self-QKV and cross-KV projections",
    )
    parser.add_argument(
        "--symmetric-critical-fallback",
        action="store_true",
        help=(
            "If a router-chosen generalist fails the critical-token guard, also "
            "try the bundled expert; expert-to-generalist fallback remains default."
        ),
    )
    args = parser.parse_args()
    if args.warm_runs < 0 or args.max_tokens < 1:
        raise SystemExit("warm runs must be non-negative and max tokens positive")

    rows = load_suite(args.suite)
    manifest = validate_pack(args.bundle)
    memories, memory_metadata = load_memory(args.bundle, manifest)
    routers = load_routers(args.bundle, manifest)
    load_started = time.perf_counter()
    models, tokenizer, source_prefixes, quantization = load_runtime(
        args.bundle,
        manifest,
        pack_attention_projections=args.packed_attention_projections,
    )
    target_shortlist = None
    static_output_projections = {}
    static_output_id_sets = {}
    tokenizer_relative = manifest.get("sharedTokenizer")
    tokenizer_path = (
        args.bundle / tokenizer_relative
        if isinstance(tokenizer_relative, str)
        else engine_paths(args.bundle, manifest)["generalist-en-ja"] / "tokenizer.json"
    )
    if args.target_shortlist is not None:
        target_shortlist = MarianTargetShortlist.load(
            args.target_shortlist,
            tokenizer_path,
            tokenizer,
        )
        for role, model in models.items():
            role_direction = "en-ja" if role.endswith("en-ja") else "ja-en"
            static_output_projections[role] = model.prepare_output_shortlist(
                target_shortlist.static_ids[role_direction]
            )
            static_output_id_sets[role] = frozenset(
                target_shortlist.static_ids[role_direction]
            )
    load_seconds = time.perf_counter() - load_started

    def translate(row: dict[str, Any]) -> dict[str, Any]:
        source = str(row["source"])
        direction = DIRECTION_BY_LANGUAGES[
            (row["sourceLanguage"], row["targetLanguage"])
        ]
        memory_target = memories[direction].get(normalize_memory_source(source))
        if memory_target is not None:
            output = clean_output(memory_target)
            critical_pass = preserves_critical_tokens(source, output)
            plausible = is_plausible(output, source, direction)
            return {
                "hypothesis": output,
                "outputTokenIDs": None,
                "selectedEngine": "exact-translation-memory",
                "selectedNeuralEngine": None,
                "routerScore": None,
                "routedToExpert": False,
                "criticalFallbackUsed": False,
                "criticalFallbackDirection": None,
                "criticalTokenGuardPasses": critical_pass,
                "plausibilityGuardPasses": plausible,
                "runtimeAccepted": critical_pass and plausible,
                "failureReason": (
                    None if critical_pass and plausible else "unsafe-translation-memory-output"
                ),
                "outputShortlistTokens": None,
            }

        generalist_role, expert_role = ROLE_BY_DIRECTION[direction]
        router_score = routers[direction].score(source)
        use_expert = routers[direction].routes_to_expert(source)
        first_role = expert_role if use_expert else generalist_role

        def generate(role: str) -> tuple[list[int], str, int | None]:
            encoded = tokenizer.encode(source_prefixes[role] + source)
            shortlist_ids = (
                target_shortlist.expand(direction, encoded)
                if target_shortlist is not None
                else None
            )
            output_projection = None
            if shortlist_ids is not None:
                static_projection = static_output_projections[role]
                extension_ids = tuple(
                    token_id
                    for token_id in shortlist_ids
                    if token_id not in static_output_id_sets[role]
                )
                if extension_ids:
                    extension = models[role].prepare_output_extension(extension_ids)
                    output_projection = CompositeOutputProjection(
                        parts=(static_projection, extension),
                        token_ids=static_projection.token_ids + extension.token_ids,
                        pad_index=static_projection.pad_index,
                    )
                else:
                    output_projection = static_projection
            token_ids = (
                models[role].generate_cached_prepared_shortlist(
                    encoded,
                    output_projection,
                    args.max_tokens,
                )
                if shortlist_ids is not None
                else models[role].generate_cached(encoded, args.max_tokens)
            )
            mx.synchronize()
            return (
                token_ids,
                clean_output(tokenizer.decode(token_ids, skip_special_tokens=True)),
                len(shortlist_ids) if shortlist_ids is not None else None,
            )

        output_ids, output, shortlist_tokens = generate(first_role)
        critical_pass = preserves_critical_tokens(source, output)
        selected_role = first_role
        fallback_used = False
        if not critical_pass and (use_expert or args.symmetric_critical_fallback):
            alternate_role = generalist_role if use_expert else expert_role
            output_ids, output, shortlist_tokens = generate(alternate_role)
            selected_role = alternate_role
            fallback_used = True
            critical_pass = preserves_critical_tokens(source, output)
        plausible = is_plausible(output, source, direction)
        accepted = critical_pass and plausible
        failure_reason = None
        if not critical_pass:
            failure_reason = "critical-token-mismatch"
        elif not plausible:
            failure_reason = "implausible-output"
        return {
            "hypothesis": output,
            "outputTokenIDs": output_ids,
            "selectedEngine": selected_role,
            "selectedNeuralEngine": selected_role,
            "routerScore": router_score,
            "routedToExpert": use_expert,
            "criticalFallbackUsed": fallback_used,
            "criticalFallbackDirection": (
                f"{first_role}-to-{selected_role}" if fallback_used else None
            ),
            "criticalTokenGuardPasses": critical_pass,
            "plausibilityGuardPasses": plausible,
            "runtimeAccepted": accepted,
            "failureReason": failure_reason,
            "outputShortlistTokens": shortlist_tokens,
        }

    results = []
    selected_counts: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    direction_latencies: dict[str, list[float]] = {"en-ja": [], "ja-en": []}
    shortlist_sizes: dict[str, list[int]] = {"en-ja": [], "ja-en": []}
    for index, row in enumerate(rows, start=1):
        started = time.perf_counter()
        result = translate(row)
        first_latency = time.perf_counter() - started
        warm_latencies = []
        for _ in range(args.warm_runs):
            warm_started = time.perf_counter()
            warm = translate(row)
            warm_latencies.append(time.perf_counter() - warm_started)
            if (
                warm["hypothesis"] != result["hypothesis"]
                or warm["outputTokenIDs"] != result["outputTokenIDs"]
                or warm["selectedEngine"] != result["selectedEngine"]
                or warm["failureReason"] != result["failureReason"]
            ):
                raise SystemExit(f"non-deterministic runtime result: {row['id']}")
        direction = DIRECTION_BY_LANGUAGES[
            (row["sourceLanguage"], row["targetLanguage"])
        ]
        direction_latencies[direction].extend(warm_latencies or [first_latency])
        if result["outputShortlistTokens"] is not None:
            shortlist_sizes[direction].append(int(result["outputShortlistTokens"]))
        selected_counts[result["selectedEngine"]] += 1
        if result["failureReason"] is not None:
            failures[result["failureReason"]] += 1
        results.append(
            {
                "caseID": row["id"],
                "sourceLanguage": row["sourceLanguage"],
                "targetLanguage": row["targetLanguage"],
                "domain": row["domain"],
                "source": row["source"],
                "references": row["references"],
                "claimEligible": False,
                **{
                    key: row[key]
                    for key in (
                        "documentID",
                        "license",
                        "provenance",
                        "sourceTemplateID",
                        "sourceVariables",
                        "split",
                    )
                    if key in row
                },
                **result,
                "latencySeconds": first_latency,
                "warmLatencySeconds": warm_latencies,
            }
        )
        if index % 100 == 0:
            print(f"completed {index}/{len(rows)} cases", file=sys.stderr, flush=True)

    bits, group_size = quantization
    script_path = Path(__file__).resolve()
    report = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "purpose": "source-only exact-pack runtime, routing, safety, and latency audit",
        "status": "passed" if not failures else "failed-runtime-safety",
        "claimEligible": False,
        "claimBlocker": "references-pending",
        "engine": (
            f"mlx:Mimi-Marian-MoE:{bits}bit-g{group_size}-kv-cache"
            + (
                "-packed-attention-projections"
                if args.packed_attention_projections
                else ""
            )
            + ("-target-shortlist" if target_shortlist is not None else "")
            + (
                "-symmetric-critical-fallback"
                if args.symmetric_critical_fallback
                else ""
            )
        ),
        "modelRevision": (
            f"moe-manifest-sha256:{sha256(args.bundle / 'manifest.json')}"
            + (
                "+packed-attention-projections-v1"
                if args.packed_attention_projections
                else ""
            )
            + (
                f"+target-shortlist-sha256:{sha256(args.target_shortlist)}"
                if args.target_shortlist is not None
                else ""
            )
        ),
        "suite": {
            "path": str(args.suite),
            "sha256": sha256(args.suite),
            "cases": len(rows),
            "referencesPresent": sum(bool(row["references"]) for row in rows),
        },
        "hardware": platform.machine(),
        "operatingSystem": platform.platform(),
        "preparationSeconds": load_seconds,
        "peakResidentBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "modelBytes": directory_bytes(args.bundle),
        "physicalModelCount": 4,
        "translationMemory": memory_metadata,
        "benchmarkConfiguration": {
            "warmRunsPerCase": args.warm_runs,
            "maximumGeneratedTokens": args.max_tokens,
            "symmetricCriticalFallback": args.symmetric_critical_fallback,
            "packedAttentionProjections": args.packed_attention_projections,
            "targetShortlist": (
                {
                    "path": str(args.target_shortlist),
                    "sha256": sha256(args.target_shortlist),
                    "staticTokenCounts": {
                        direction: len(target_shortlist.static_ids[direction])
                        for direction in DIRECTION_BY_LANGUAGES.values()
                    },
                    "dynamicSourceSurfaceExpansion": True,
                }
                if target_shortlist is not None
                else None
            ),
            "latencyIncludes": [
                "memory-lookup",
                "source-routing",
                "tokenization",
                "cached-neural-generation",
                "critical-token-fallback",
                "decoding",
                "runtime-guards",
            ],
        },
        "runtimeImplementation": {
            "benchmarkScriptSha256": sha256(script_path),
            "marianRuntimeSha256": sha256(script_path.with_name("marian_mlx.py")),
            "routerRuntimeSha256": sha256(
                script_path.with_name("source_expert_router.py")
            ),
            "criticalPolicySha256": sha256(
                script_path.with_name("typed_critical_token_policy.py")
            ),
            "targetShortlistRuntimeSha256": (
                sha256(script_path.with_name("marian_target_shortlist.py"))
                if target_shortlist is not None
                else None
            ),
            "pythonVersion": platform.python_version(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("mlx", "tokenizers", "transformers")
            },
        },
        "summary": {
            "selectedEngineCounts": dict(sorted(selected_counts.items())),
            "failureCounts": dict(sorted(failures.items())),
            "runtimeAcceptedCases": len(rows) - sum(failures.values()),
            "directionLatency": {
                direction: {
                    "samples": len(values),
                    "p50Seconds": percentile(values, 0.50),
                    "p95Seconds": percentile(values, 0.95),
                }
                for direction, values in direction_latencies.items()
            },
            "directionShortlistTokens": {
                direction: {
                    "samples": len(values),
                    "minimum": min(values) if values else None,
                    "median": percentile(values, 0.50),
                    "maximum": max(values) if values else None,
                }
                for direction, values in shortlist_sizes.items()
            },
        },
        "doesNotAuthorizeAppIntegration": True,
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
                "status": report["status"],
                "cases": len(rows),
                "failures": sum(failures.values()),
                **report["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
