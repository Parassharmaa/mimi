#!/usr/bin/env python3
"""Compare an exact Marian output-projection shortlist canary with its control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CASE_FIELDS = (
    "sourceLanguage",
    "targetLanguage",
    "domain",
    "source",
    "references",
    "claimEligible",
)
PARITY_FIELDS = (
    "hypothesis",
    "outputTokenIDs",
    "selectedEngine",
    "selectedNeuralEngine",
    "routedToExpert",
    "routerScore",
    "criticalTokenGuardPasses",
    "plausibilityGuardPasses",
    "runtimeAccepted",
    "failureReason",
)
DIRECTIONS = {"en-ja": "en-US", "ja-en": "ja-JP"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schemaVersion") != 1 or not isinstance(payload.get("results"), list):
        raise SystemExit(f"invalid Marian runtime report: {path}")
    if payload.get("claimEligible") is not False:
        raise SystemExit(f"runtime report is not explicitly claim-ineligible: {path}")
    return payload


def indexed(report: dict, path: Path) -> dict[str, dict]:
    rows = {str(row.get("caseID", "")): row for row in report["results"]}
    if "" in rows or len(rows) != len(report["results"]):
        raise SystemExit(f"report has empty or duplicate case IDs: {path}")
    return rows


def percent_delta(baseline: float, candidate: float) -> float:
    if baseline <= 0:
        raise SystemExit("runtime comparison requires positive baseline values")
    return (candidate / baseline - 1.0) * 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--minimum-speedup-percent", type=float, default=5.0)
    parser.add_argument(
        "--purpose",
        default="claim-ineligible exact output-projection acceleration canary",
    )
    args = parser.parse_args()

    baseline_report = load(args.baseline)
    candidate_report = load(args.candidate)
    baseline = indexed(baseline_report, args.baseline)
    candidate = indexed(candidate_report, args.candidate)
    if set(baseline) != set(candidate):
        raise SystemExit("runtime reports cover different case IDs")
    for case_id in baseline:
        for field in CASE_FIELDS:
            if baseline[case_id].get(field) != candidate[case_id].get(field):
                raise SystemExit(f"runtime reports disagree on {field}: {case_id}")

    parity_mismatches = {
        field: sorted(
            case_id
            for case_id in baseline
            if baseline[case_id].get(field) != candidate[case_id].get(field)
        )
        for field in PARITY_FIELDS
    }
    exact_parity = all(not cases for cases in parity_mismatches.values())

    latency: dict[str, dict] = {}
    speed_gate_passes = True
    for direction in DIRECTIONS:
        baseline_latency = baseline_report["summary"]["directionLatency"][direction]
        candidate_latency = candidate_report["summary"]["directionLatency"][direction]
        deltas = {
            percentile: percent_delta(
                float(baseline_latency[percentile]),
                float(candidate_latency[percentile]),
            )
            for percentile in ("p50Seconds", "p95Seconds")
        }
        improvements = {key: -value for key, value in deltas.items()}
        direction_passes = all(
            value >= args.minimum_speedup_percent for value in improvements.values()
        )
        speed_gate_passes = speed_gate_passes and direction_passes
        latency[direction] = {
            "baseline": baseline_latency,
            "candidate": candidate_latency,
            "candidateDeltaPercent": deltas,
            "candidateImprovementPercent": improvements,
            "passesMinimumSpeedup": direction_passes,
        }

    baseline_rss = int(baseline_report["peakResidentBytes"])
    candidate_rss = int(candidate_report["peakResidentBytes"])
    rss_delta = candidate_rss - baseline_rss
    rss_gate_passes = rss_delta <= 0
    baseline_preparation = float(baseline_report["preparationSeconds"])
    candidate_preparation = float(candidate_report["preparationSeconds"])

    passes = exact_parity and speed_gate_passes and rss_gate_passes
    reasons = []
    if not exact_parity:
        reasons.append("candidate does not preserve exact routed output")
    if not speed_gate_passes:
        reasons.append(
            f"candidate does not improve both p50 and p95 by at least "
            f"{args.minimum_speedup_percent:g}% in both directions"
        )
    if not rss_gate_passes:
        reasons.append("candidate increases peak resident memory")

    payload = {
        "schemaVersion": 1,
        "status": "passed" if passes else "rejected",
        "purpose": args.purpose,
        "claimEligible": False,
        "cases": len(baseline),
        "baseline": {
            "path": str(args.baseline),
            "sha256": sha256(args.baseline),
            "modelRevision": baseline_report.get("modelRevision"),
            "peakResidentBytes": baseline_rss,
            "preparationSeconds": baseline_preparation,
        },
        "candidate": {
            "path": str(args.candidate),
            "sha256": sha256(args.candidate),
            "modelRevision": candidate_report.get("modelRevision"),
            "peakResidentBytes": candidate_rss,
            "preparationSeconds": candidate_preparation,
        },
        "parity": {
            "exact": exact_parity,
            "mismatchCasesByField": parity_mismatches,
        },
        "latency": latency,
        "memory": {
            "candidateDeltaBytes": rss_delta,
            "candidateDeltaPercent": percent_delta(baseline_rss, candidate_rss),
            "passesNoIncreaseGate": rss_gate_passes,
        },
        "preparation": {
            "candidateDeltaSeconds": candidate_preparation - baseline_preparation,
            "candidateDeltaPercent": percent_delta(
                baseline_preparation, candidate_preparation
            ),
        },
        "stopGate": {
            "minimumSpeedupPercentAtP50AndP95PerDirection": (
                args.minimum_speedup_percent
            ),
            "requiresExactOutputParity": True,
            "requiresNoPeakResidentIncrease": True,
            "passes": passes,
            "decision": (
                "continue to full benchmark" if passes else "stop at canary"
            ),
            "reasons": reasons,
        },
        "doesNotAuthorizeAppIntegration": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "cases": payload["cases"],
                "exactParity": exact_parity,
                "speedGatePasses": speed_gate_passes,
                "rssGatePasses": rss_gate_passes,
                "decision": payload["stopGate"]["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
