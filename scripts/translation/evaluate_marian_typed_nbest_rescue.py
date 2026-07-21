#!/usr/bin/env python3
"""Evaluate failure-triggered Marian n-best rescue under a narrow typed gate."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx

from run_mlx_marian_moe_benchmark import (
    clean_output,
    is_plausible,
    load_runtime,
    preserves_critical_tokens,
    sha256,
    validate_pack,
)
from typed_critical_token_policy import (
    narrow_temporal_preserves,
    narrow_temporal_signature,
)


def direction(row: dict) -> str:
    return "en-ja" if row["sourceLanguage"] == "en-US" else "ja-en"


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def eligible_source(row: dict) -> tuple[bool, dict]:
    signature = narrow_temporal_signature(str(row["source"]), str(row["sourceLanguage"]))
    facts = len(signature.dates) + len(signature.times)
    eligible = (
        facts == 1
        and not signature.protected
        and signature.literal_percentages == 0
        and not signature.other_numbers
        and not signature.ambiguous_context
    )
    return eligible, {
        "dates": signature.dates,
        "times": signature.times,
        "protected": signature.protected,
        "literalPercentages": signature.literal_percentages,
        "otherNumbers": signature.other_numbers,
        "ambiguousContext": signature.ambiguous_context,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("baseline_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=192)
    args = parser.parse_args()
    if args.beam_size < 2 or args.max_tokens < 1:
        raise SystemExit("beam size must be at least two and max tokens positive")

    baseline = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    rows = baseline.get("results")
    if not isinstance(rows, list) or not rows:
        raise SystemExit("baseline report has no results")
    expected_revision = f"moe-manifest-sha256:{sha256(args.bundle / 'manifest.json')}"
    baseline_revision = str(baseline.get("modelRevision") or "").split("+", 1)[0]
    if baseline_revision != expected_revision:
        raise SystemExit("baseline report does not authenticate the supplied bundle")

    manifest = validate_pack(args.bundle)
    started = time.perf_counter()
    models, tokenizer, source_prefixes, quantization = load_runtime(args.bundle, manifest)
    load_seconds = time.perf_counter() - started

    counts: Counter[str] = Counter()
    direction_counts: Counter[str] = Counter()
    latencies: dict[str, list[float]] = {"en-ja": [], "ja-en": []}
    results: list[dict] = []
    for row in rows:
        source = str(row["source"])
        hypothesis = str(row["hypothesis"])
        current_direction = direction(row)
        if preserves_critical_tokens(source, hypothesis):
            counts["baseline-strict-safe"] += 1
            continue
        if narrow_temporal_preserves(
            source,
            hypothesis,
            str(row["sourceLanguage"]),
            str(row["targetLanguage"]),
        ):
            counts["greedy-narrow-temporal-candidate"] += 1
            direction_counts[f"{current_direction}:greedy-narrow-temporal-candidate"] += 1
            continue
        eligible, signature = eligible_source(row)
        role = row.get("selectedNeuralEngine")
        if not eligible or role not in models:
            status = "not-eligible-for-nbest-arm"
            counts[status] += 1
            direction_counts[f"{current_direction}:{status}"] += 1
            continue

        encoded = tokenizer.encode(source_prefixes[role] + source)
        beam_started = time.perf_counter()
        beam = models[role].generate_beam_nbest(
            encoded,
            beam_size=args.beam_size,
            maximum_tokens=args.max_tokens,
            num_return_sequences=args.beam_size,
        )
        mx.synchronize()
        elapsed = time.perf_counter() - beam_started
        latencies[current_direction].append(elapsed)
        candidates = []
        selected = None
        for rank, (token_ids, score) in enumerate(beam, start=1):
            output = clean_output(tokenizer.decode(token_ids, skip_special_tokens=True))
            typed_pass = narrow_temporal_preserves(
                source,
                output,
                str(row["sourceLanguage"]),
                str(row["targetLanguage"]),
            )
            plausible = is_plausible(output, source, current_direction)
            candidate = {
                "rank": rank,
                "score": score,
                "outputTokenIDs": token_ids,
                "hypothesis": output,
                "narrowTemporalPasses": typed_pass,
                "plausibilityPasses": plausible,
            }
            candidates.append(candidate)
            if selected is None and typed_pass and plausible:
                selected = candidate
        status = "nbest-rescued-candidate" if selected is not None else "nbest-no-valid-candidate"
        counts[status] += 1
        direction_counts[f"{current_direction}:{status}"] += 1
        references = row.get("references") or []
        reference_compatible = (
            len(references) == 1
            and narrow_temporal_preserves(
                source,
                str(references[0]),
                str(row["sourceLanguage"]),
                str(row["targetLanguage"]),
            )
        )
        results.append(
            {
                "caseID": row["caseID"],
                "direction": current_direction,
                "domain": row.get("domain"),
                "selectedEngine": role,
                "source": source,
                "baselineHypothesis": hypothesis,
                "references": references,
                "sourceTemporalSignature": signature,
                "referenceNarrowTemporalCompatible": reference_compatible,
                "beamLatencySeconds": elapsed,
                "status": status,
                "selected": selected,
                "candidates": candidates,
            }
        )

    bits, group_size = quantization
    script_path = Path(__file__).resolve()
    report = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "diagnostic-only",
        "purpose": "failure-triggered beam n-best typed temporal rescue ablation",
        "claimEligible": False,
        "doesNotAuthorizeAppIntegration": True,
        "bundle": {
            "path": str(args.bundle),
            "manifestSHA256": sha256(args.bundle / "manifest.json"),
            "quantization": {"bits": bits, "groupSize": group_size},
        },
        "baselineReport": {
            "path": str(args.baseline_report),
            "sha256": sha256(args.baseline_report),
        },
        "implementation": {
            "runnerSHA256": sha256(script_path),
            "marianRuntimeSHA256": sha256(script_path.with_name("marian_mlx.py")),
            "typedPolicySHA256": sha256(
                script_path.with_name("typed_critical_token_policy.py")
            ),
        },
        "configuration": {
            "beamSize": args.beam_size,
            "maximumGeneratedTokens": args.max_tokens,
            "trigger": "baseline strict critical-token failure",
            "sourceEligibility": (
                "exactly one unambiguous Gregorian date or 24-hour time; no other "
                "digits, protected token, or literal percent sign"
            ),
            "selection": (
                "highest model-score candidate passing narrow temporal equality and plausibility"
            ),
            "runtimePolicyChanged": False,
        },
        "hardware": platform.machine(),
        "operatingSystem": platform.platform(),
        "preparationSeconds": load_seconds,
        "peakResidentBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "counts": dict(sorted(counts.items())),
        "directionCounts": dict(sorted(direction_counts.items())),
        "triggeredLatency": {
            key: {
                "samples": len(values),
                "p50Seconds": percentile(values, 0.50),
                "p95Seconds": percentile(values, 0.95),
            }
            for key, values in latencies.items()
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
                "status": report["status"],
                "counts": report["counts"],
                "triggeredLatency": report["triggeredLatency"],
                "peakResidentBytes": report["peakResidentBytes"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
