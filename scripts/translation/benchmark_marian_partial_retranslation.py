#!/usr/bin/env python3
"""Benchmark exact prior-output reuse for growing Marian source prefixes."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import resource
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parent))
from marian_mlx import EOS_TOKEN_ID, PAD_TOKEN_ID, load_model  # noqa: E402


DIRECTIONS = {
    ("en-US", "ja-JP"): "en-ja",
    ("ja-JP", "en-US"): "ja-en",
}
MAXIMUM_DRAFT_TOKENS = 64


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def token_sha256(tokens: list[int]) -> str:
    payload = json.dumps(tokens, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def verified_draft_matches(
    greedy_predictions: list[int],
    draft_tokens: list[int],
) -> bool:
    return greedy_predictions[: len(draft_tokens) + 1] == [
        *draft_tokens,
        EOS_TOKEN_ID,
    ]


def truncate_prefill_caches(caches, length: int):
    """Drop speculative self-attention positions after the first divergence."""

    if length < 1:
        raise ValueError("at least the decoder start position must be retained")
    return [
        (
            (
                self_cache[0][:, :, :length, :],
                self_cache[1][:, :, :length, :],
            ),
            cross_cache,
        )
        for self_cache, cross_cache in caches
    ]


def generate_with_parallel_draft_verification(
    model,
    input_ids: list[int],
    draft_tokens: list[int],
    maximum_tokens: int,
) -> tuple[list[int], dict]:
    if any(token in (EOS_TOKEN_ID, PAD_TOKEN_ID) for token in draft_tokens):
        raise ValueError("draft tokens must not contain EOS or PAD")
    if maximum_tokens < 1:
        raise ValueError("maximum token count must be positive")
    if len(draft_tokens) > maximum_tokens:
        raise ValueError("draft cannot exceed the generation token cap")

    encoder_states = model.encode(mx.array([input_ids], dtype=mx.int32))
    if len(draft_tokens) > MAXIMUM_DRAFT_TOKENS:
        return model._generate_cached_from_encoder(encoder_states, maximum_tokens), {
            "attempted": False,
            "bypassReason": "draft-too-long",
            "draftTokenCount": len(draft_tokens),
            "acceptedDraftTokens": 0,
            "firstDivergencePosition": None,
            "fullDraftAndEOSAccepted": False,
        }
    decoder_ids = mx.array([[PAD_TOKEN_ID, *draft_tokens]], dtype=mx.int32)
    logits, caches = model.decode_prefill(decoder_ids, encoder_states)
    expected = list(draft_tokens)
    if len(expected) < maximum_tokens:
        expected.append(EOS_TOKEN_ID)
    predictions = [
        int(token)
        for token in mx.argmax(
            logits[0, : len(expected), :PAD_TOKEN_ID],
            axis=-1,
        ).tolist()
    ]
    divergence = next(
        (
            index
            for index, (prediction, wanted) in enumerate(
                zip(predictions, expected, strict=True)
            )
            if prediction != wanted
        ),
        None,
    )
    diagnostics = {
        "attempted": True,
        "draftTokenCount": len(draft_tokens),
        "acceptedDraftTokens": (
            len(draft_tokens) if divergence is None else min(divergence, len(draft_tokens))
        ),
        "firstDivergencePosition": divergence,
        "fullDraftAndEOSAccepted": divergence is None,
    }
    if divergence is None:
        return list(draft_tokens), diagnostics

    if divergence == 0:
        diagnostics["bypassReason"] = "no-verified-prefix"
        return model._generate_cached_from_encoder(encoder_states, maximum_tokens), diagnostics

    output = list(draft_tokens[:divergence])
    decoder_id = predictions[divergence]
    if decoder_id == EOS_TOKEN_ID:
        return output, diagnostics
    output.append(decoder_id)
    if len(output) >= maximum_tokens:
        return output, diagnostics

    # The parallel pass includes speculative inputs after the divergence. Keep
    # only [PAD] plus the verified draft prefix; the corrected token becomes the
    # next incremental decoder input.
    caches = truncate_prefill_caches(caches, divergence + 1)
    for position_offset in range(divergence + 1, maximum_tokens):
        next_logits, caches = model.decode_step(
            decoder_id,
            encoder_states,
            caches,
            position_offset,
        )
        token = int(mx.argmax(next_logits[0, -1, :PAD_TOKEN_ID]).item())
        if token == EOS_TOKEN_ID:
            break
        output.append(token)
        decoder_id = token
    return output, diagnostics


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def load_suite(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    identifiers = [str(row["id"]) for row in rows]
    if not rows or len(identifiers) != len(set(identifiers)):
        raise SystemExit("suite must be non-empty and contain unique IDs")
    return rows


def select_rows(rows: list[dict], per_domain: int) -> dict[tuple[str, str], list[dict]]:
    selected: dict[tuple[str, str], list[dict]] = {}
    for direction in DIRECTIONS:
        directional = [
            row
            for row in rows
            if (row.get("sourceLanguage"), row.get("targetLanguage")) == direction
        ]
        domains = sorted({str(row["domain"]) for row in directional})
        chosen: list[dict] = []
        for domain in domains:
            candidates = [row for row in directional if row["domain"] == domain]
            candidates.sort(
                key=lambda row: hashlib.sha256(str(row["id"]).encode()).digest()
            )
            if len(candidates) < per_domain:
                raise SystemExit(
                    f"{direction} / {domain} has only {len(candidates)} cases"
                )
            chosen.extend(candidates[:per_domain])
        selected[direction] = chosen
    return selected


def prefix_token_ids(tokens: list[int], fractions: list[float]) -> list[list[int]]:
    lengths = {
        min(len(tokens), max(1, math.ceil(len(tokens) * fraction)))
        for fraction in fractions
    }
    lengths.add(len(tokens))
    return [tokens[:length] for length in sorted(lengths)]


def run_baseline_session(model, prefixes: list[list[int]], maximum_tokens: int):
    outputs: list[list[int]] = []
    latencies: list[float] = []
    for prefix in prefixes:
        started = time.perf_counter()
        output = model.generate_cached(prefix, maximum_tokens)
        mx.synchronize()
        latencies.append(time.perf_counter() - started)
        outputs.append(output)
    return outputs, latencies


def run_reuse_session(model, prefixes: list[list[int]], maximum_tokens: int):
    outputs: list[list[int]] = []
    latencies: list[float] = []
    verifications: list[dict] = []
    previous: list[int] = []
    for index, prefix in enumerate(prefixes):
        started = time.perf_counter()
        if index == 0:
            output = model.generate_cached(prefix, maximum_tokens)
            verification = {
                "attempted": False,
                "bypassReason": "first-update-has-no-draft",
                "draftTokenCount": 0,
                "acceptedDraftTokens": 0,
                "firstDivergencePosition": None,
                "fullDraftAndEOSAccepted": False,
            }
        else:
            output, verification = generate_with_parallel_draft_verification(
                model,
                prefix,
                previous,
                maximum_tokens,
            )
        mx.synchronize()
        latencies.append(time.perf_counter() - started)
        outputs.append(output)
        verifications.append(verification)
        previous = output
    return outputs, latencies, verifications


def load_direction_model(path: Path):
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model = load_model(
        path / "model.safetensors",
        quantization_bits=int(manifest["bits"]),
        quantization_group_size=int(manifest["group_size"]),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(path / "tokenizer.json"),
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    return model, tokenizer, {
        "path": str(path.resolve()),
        "manifestSHA256": sha256(manifest_path),
    }


def summarize(cases: list[dict]) -> dict:
    baseline = [latency for case in cases for latency in case["baselineLatencySeconds"]]
    candidate = [latency for case in cases for latency in case["reuseLatencySeconds"]]
    attempts = [
        verification
        for case in cases
        for verification in case["draftVerification"]
        if verification["attempted"]
    ]
    fully_accepted = sum(
        verification["fullDraftAndEOSAccepted"] for verification in attempts
    )
    draft_tokens = sum(verification["draftTokenCount"] for verification in attempts)
    accepted_tokens = sum(
        verification["acceptedDraftTokens"] for verification in attempts
    )
    return {
        "sessions": len(cases),
        "prefixUpdates": len(baseline),
        "draftAttempts": len(attempts),
        "fullDraftAndEOSAccepted": fully_accepted,
        "fullDraftAcceptanceRate": fully_accepted / len(attempts) if attempts else 0.0,
        "draftTokens": draft_tokens,
        "acceptedDraftTokens": accepted_tokens,
        "draftTokenAcceptanceRate": accepted_tokens / draft_tokens if draft_tokens else 0.0,
        "exactOutputParity": all(case["exactOutputParity"] for case in cases),
        "baselineTotalSeconds": sum(baseline),
        "reuseTotalSeconds": sum(candidate),
        "totalSpeedup": sum(baseline) / sum(candidate),
        "baselineP50Seconds": statistics.median(baseline),
        "baselineP95Seconds": percentile(baseline, 0.95),
        "reuseP50Seconds": statistics.median(candidate),
        "reuseP95Seconds": percentile(candidate, 0.95),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--en-ja-model", type=Path, required=True)
    parser.add_argument("--ja-en-model", type=Path, required=True)
    parser.add_argument("--cases-per-domain", type=int, default=20)
    parser.add_argument("--prefix-fractions", default="0.5,0.75,1.0")
    parser.add_argument("--maximum-tokens", type=int, default=192)
    parser.add_argument("--warm-runs", type=int, default=3)
    args = parser.parse_args()
    if args.cases_per_domain < 1 or args.maximum_tokens < 1 or args.warm_runs < 0:
        raise SystemExit("case count and token cap must be positive; warm runs non-negative")
    fractions = [float(value) for value in args.prefix_fractions.split(",")]
    if not fractions or any(not 0 < value <= 1 for value in fractions):
        raise SystemExit("prefix fractions must be in (0, 1]")

    suite = load_suite(args.suite)
    selected = select_rows(suite, args.cases_per_domain)
    paths = {
        ("en-US", "ja-JP"): args.en_ja_model,
        ("ja-JP", "en-US"): args.ja_en_model,
    }
    all_cases: list[dict] = []
    model_records: dict[str, dict] = {}

    for direction, rows in selected.items():
        model, tokenizer, model_record = load_direction_model(paths[direction])
        model_records[DIRECTIONS[direction]] = model_record
        first_tokens = tokenizer.encode(str(rows[0]["source"]))
        warm_output = model.generate_cached(first_tokens, args.maximum_tokens)
        mx.synchronize()
        for _ in range(args.warm_runs):
            model.generate_cached(first_tokens, args.maximum_tokens)
            mx.synchronize()
            generate_with_parallel_draft_verification(
                model,
                first_tokens,
                warm_output,
                args.maximum_tokens,
            )
            mx.synchronize()

        for index, row in enumerate(rows):
            source_tokens = tokenizer.encode(str(row["source"]))
            prefixes = prefix_token_ids(source_tokens, fractions)
            if index % 2 == 0:
                baseline_outputs, baseline_latencies = run_baseline_session(
                    model, prefixes, args.maximum_tokens
                )
                reuse_outputs, reuse_latencies, verifications = run_reuse_session(
                    model, prefixes, args.maximum_tokens
                )
            else:
                reuse_outputs, reuse_latencies, verifications = run_reuse_session(
                    model, prefixes, args.maximum_tokens
                )
                baseline_outputs, baseline_latencies = run_baseline_session(
                    model, prefixes, args.maximum_tokens
                )
            exact = baseline_outputs == reuse_outputs
            if not exact:
                mismatch_index = next(
                    offset
                    for offset, (baseline, candidate) in enumerate(
                        zip(baseline_outputs, reuse_outputs)
                    )
                    if baseline != candidate
                )
                baseline = baseline_outputs[mismatch_index]
                candidate = reuse_outputs[mismatch_index]
                first_difference = next(
                    (
                        offset
                        for offset, (baseline_token, candidate_token) in enumerate(
                            zip(baseline, candidate)
                        )
                        if baseline_token != candidate_token
                    ),
                    min(len(baseline), len(candidate)),
                )
                failure = {
                    "schemaVersion": 1,
                    "createdAt": datetime.now(timezone.utc).isoformat(),
                    "purpose": "rejected parallel prior-output verification ablation",
                    "promotionEligible": False,
                    "suite": {
                        "path": str(args.suite.resolve()),
                        "sha256": sha256(args.suite),
                    },
                    "model": model_record,
                    "failure": {
                        "reason": "teacher-forced and cached greedy kernels changed tokens",
                        "caseID": row["id"],
                        "direction": f"{direction[0]}>{direction[1]}",
                        "domain": row["domain"],
                        "prefixTokenCount": len(prefixes[mismatch_index]),
                        "draftVerification": verifications[mismatch_index],
                        "firstDifferentTargetToken": first_difference,
                        "baselineTokenCount": len(baseline),
                        "candidateTokenCount": len(candidate),
                        "baselineTokenSHA256": token_sha256(baseline),
                        "candidateTokenSHA256": token_sha256(candidate),
                    },
                }
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True)
                    + "\n",
                    encoding="utf-8",
                )
                raise SystemExit(f"draft reuse changed output tokens for {row['id']}")
            all_cases.append({
                "id": row["id"],
                "direction": f"{direction[0]}>{direction[1]}",
                "domain": row["domain"],
                "sourceTokenCount": len(source_tokens),
                "prefixTokenCounts": [len(prefix) for prefix in prefixes],
                "baselineLatencySeconds": baseline_latencies,
                "reuseLatencySeconds": reuse_latencies,
                "draftVerification": verifications,
                "exactOutputParity": exact,
                "outputTokenCounts": [len(tokens) for tokens in reuse_outputs],
                "outputTokenSHA256": [token_sha256(tokens) for tokens in reuse_outputs],
            })
        del model, tokenizer
        gc.collect()
        mx.clear_cache()

    directions = {}
    for direction in DIRECTIONS:
        label = f"{direction[0]}>{direction[1]}"
        directions[label] = summarize(
            [case for case in all_cases if case["direction"] == label]
        )
    report = {
        "schemaVersion": 2,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "purpose": "development-only monotonic-source partial-retranslation speed ablation",
        "promotionEligible": False,
        "algorithm": {
            "name": "parallel-verify-prior-output-resume-from-first-divergence",
            "draftBias": 0.0,
            "maximumDraftTokens": MAXIMUM_DRAFT_TOKENS,
            "prefixFractions": fractions,
            "maximumTokens": args.maximum_tokens,
            "warmRuns": args.warm_runs,
        },
        "suite": {
            "path": str(args.suite.resolve()),
            "sha256": sha256(args.suite),
        },
        "models": model_records,
        "runtime": {
            "mlx": getattr(mx, "__version__", "unknown"),
            "python": platform.python_version(),
            "machine": platform.machine(),
            "peakResidentBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "directions": directions,
        "cases": all_cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(directions, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
