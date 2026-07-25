#!/usr/bin/env python3
"""Seal the fail-closed protected result for v9 interpolation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from finalize_canonical_pairwise_v7_result import (
    aligned,
    comparison,
    directory_bytes,
    gate,
    ja_en_rows,
    load,
    percentile,
    record,
    safety_delta,
    sha256,
)


ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "Research/translation/results"
M = ROOT / "Research/translation/models"
CONTRACT = ROOT / "Research/translation/canonical-pairwise-v9-contract-2026-07-25.json"
INTERNAL_RESULT = ROOT / "Research/translation/canonical-pairwise-v9-result-2026-07-25.json"
CANDIDATE_Q4 = M / "elanmt-parent-specialist-interpolation-v9-ja-en-mlx-4bit"
CURRENT_EN_JA_Q4 = M / "elanmt-release-clean-full-depth-en-ja-v1-avg3-mlx-4bit"
REPORTS = {
    "candidate_documents": R / "development-accuracy-v1-interpolation-v9-ja-en.json",
    "candidate_segments": R / "development-accuracy-v1-interpolation-v9-ja-en-segments.json",
    "candidate_direct": R / "development-accuracy-v1-interpolation-v9-ja-en-direct-under-192.json",
    "baseline_documents": R / "development-accuracy-v1-candidate-clean-pair.json",
    "baseline_segments": R / "development-accuracy-v1-candidate-clean-pair-segments.json",
    "baseline_direct": R / "development-accuracy-v1-candidate-clean-pair-direct-under-192.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite protected result: {output}")

    contract = load(CONTRACT)
    internal = load(INTERNAL_RESULT)
    if (
        contract.get("experiment") != "parent-specialist-interpolation-v9-ja-en"
        or contract.get("status") != "preregistered-ready-for-interpolation"
        or internal.get("status") != "internal-interpolation-gate-passed"
        or internal.get("internal_gate", {}).get("passed") is not True
        or internal.get("quantization_authorized") is not True
    ):
        raise SystemExit("v9 internal gate does not authorize protected finalization")

    q4_manifest_path = CANDIDATE_Q4 / "manifest.json"
    q4_manifest = load(q4_manifest_path)
    expected_source_hash = internal["candidate"]["model"]["sha256"]
    if (
        q4_manifest.get("direction") != "ja-en"
        or q4_manifest.get("bits") != 4
        or q4_manifest.get("group_size") != 64
        or q4_manifest.get("source_weights_sha256") != expected_source_hash
    ):
        raise SystemExit("v9 q4 conversion does not match the selected checkpoint")

    reports = {name: load(path) for name, path in REPORTS.items()}
    candidates = {
        "documents": reports["candidate_documents"],
        "segments": reports["candidate_segments"],
        "direct": reports["candidate_direct"],
    }
    baselines = {
        "documents": reports["baseline_documents"],
        "segments": reports["baseline_segments"],
        "direct": reports["baseline_direct"],
    }
    document = comparison(
        ja_en_rows(candidates["documents"], "candidate documents"),
        ja_en_rows(baselines["documents"], "baseline documents"),
        "documents",
    )
    direct = comparison(
        ja_en_rows(candidates["direct"], "candidate direct"),
        ja_en_rows(baselines["direct"], "baseline direct"),
        "direct",
    )
    # Explicitly authenticate alignment before latency/safety aggregation too.
    aligned(
        ja_en_rows(candidates["segments"], "candidate segments"),
        ja_en_rows(baselines["segments"], "baseline segments"),
        "segments",
    )
    segment_rows = ja_en_rows(candidates["segments"], "candidate segments")
    segment_latencies = [
        float(value)
        for row in segment_rows
        for value in (row.get("warmLatencySeconds") or [row["latencySeconds"]])
    ]
    segment_p95 = percentile(segment_latencies, 0.95)
    peak_rss = max(
        int(candidates[name].get("peakResidentBytes", 0))
        for name in ("documents", "segments", "direct")
    )
    bundle_bytes = directory_bytes(CURRENT_EN_JA_Q4) + directory_bytes(CANDIDATE_Q4)
    safety = safety_delta(candidates, baselines)
    minimum_domain_delta = min(
        value["meanSentenceChrFPlusPlusDelta"]
        for value in document["domains"].values()
    )
    minimum_direct_domain_delta = min(
        value["meanSentenceChrFPlusPlusDelta"]
        for value in direct["domains"].values()
    )
    paired = document["pairedSentenceChrFPlusPlus"]
    chr_signal = paired["mean"] >= 0.25
    quality_gates = [
        gate(
            "minimum-two-improvement-signals",
            False,
            {
                "passedSignals": int(chr_signal),
                "chrF++": chr_signal,
                "COMET-22": "not-run-after-fixed-rejection",
                "independentJudge": "not-run-after-fixed-rejection",
            },
            2,
        ),
        gate("mean-sentence-chrf++-signal", chr_signal, paired["mean"], 0.25),
        gate(
            "chrf++-paired-90pct-lower",
            paired["lower"] >= -0.25,
            paired["lower"],
            -0.25,
        ),
        gate(
            "sacrebleu-corpus-regression",
            document["sacreBLEUIntlDelta"] >= -0.10,
            document["sacreBLEUIntlDelta"],
            ">= -0.10",
        ),
        gate(
            "maximum-domain-chrf++-regression",
            minimum_domain_delta >= -0.50,
            minimum_domain_delta,
            ">= -0.50",
        ),
        gate(
            "maximum-direct-chrf++-regression",
            direct["pairedSentenceChrFPlusPlus"]["mean"] >= -0.50,
            direct["pairedSentenceChrFPlusPlus"]["mean"],
            ">= -0.50",
        ),
        gate(
            "maximum-direct-domain-chrf++-regression",
            minimum_direct_domain_delta >= -0.50,
            minimum_direct_domain_delta,
            ">= -0.50",
        ),
    ]
    safety_gates = [
        gate(
            "new-union-critical-errors",
            not safety["newUnionCritical"],
            safety["newUnionCritical"],
            0,
        ),
        gate(
            "new-negation-errors",
            not safety["newNegation"],
            safety["newNegation"],
            0,
        ),
        gate(
            "new-typed-critical-errors",
            not safety["newTypedCritical"],
            safety["newTypedCritical"],
            0,
        ),
        gate(
            "new-repetition-or-generation-limit-errors",
            not safety["newRepetitionOrGenerationLimit"],
            safety["newRepetitionOrGenerationLimit"],
            0,
        ),
    ]
    operational_gates = [
        gate(
            "warm-segment-p95-seconds",
            segment_p95 <= 0.175,
            segment_p95,
            0.175,
        ),
        gate(
            "peak-resident-bytes",
            peak_rss <= 250_000_000,
            peak_rss,
            250_000_000,
        ),
        gate(
            "preferred-two-direction-bundle-bytes",
            bundle_bytes <= 150_000_000,
            bundle_bytes,
            150_000_000,
        ),
        gate(
            "hard-two-direction-bundle-bytes",
            bundle_bytes <= 500_000_000,
            bundle_bytes,
            500_000_000,
        ),
        gate(
            "complete-distribution-provenance",
            q4_manifest.get("training_data") is not None,
            q4_manifest.get("training_data"),
            "complete per-row provenance and attribution sidecar",
        ),
    ]
    all_gates = [*quality_gates, *safety_gates, *operational_gates]
    if all(value["passed"] for value in all_gates):
        raise SystemExit("v9 requires the preregistered COMET and judge continuation")

    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "experiment": contract["experiment"],
        "status": "protected-promotion-gate-rejected",
        "promotion_authorized": False,
        "app_change_authorized": False,
        "bundle_creation_authorized": False,
        "public_upload_authorized": False,
        "current_app_bundle_changed": False,
        "contract": record(CONTRACT),
        "internal_result": record(INTERNAL_RESULT),
        "candidate_q4": {
            "directory": str(CANDIDATE_Q4.relative_to(ROOT)),
            "bytes": directory_bytes(CANDIDATE_Q4),
            "manifest": record(q4_manifest_path),
            "weights": record(CANDIDATE_Q4 / "model.safetensors"),
        },
        "would_be_two_direction_bundle_bytes": bundle_bytes,
        "evaluation": {
            "documents": document,
            "direct": direct,
            "warmSegmentP95Seconds": segment_p95,
            "peakResidentBytes": peak_rss,
            "safety": safety,
            "skippedAfterFixedRejection": [
                "COMET-22",
                "independent blinded quality judge",
            ],
        },
        "reports": {name: record(path) for name, path in REPORTS.items()},
        "gates": {
            "quality": quality_gates,
            "safety": safety_gates,
            "operationalAndDistribution": operational_gates,
        },
        "decision": (
            "Do not bundle or integrate v9. Internal interpolation and exact-q4 "
            "conversion passed, but at least one mandatory protected quality, "
            "safety, runtime, size, or distribution-provenance gate failed."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256(output),
                "status": result["status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
