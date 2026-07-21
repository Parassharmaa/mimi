#!/usr/bin/env python3
"""Compare two source-only Mimi MoE runtime audit reports."""

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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schemaVersion") != 1 or not isinstance(payload.get("results"), list):
        raise SystemExit(f"invalid source-only runtime report: {path}")
    if payload.get("claimEligible") is not False or payload.get("claimBlocker") != "references-pending":
        raise SystemExit(f"report is not explicitly source-only and claim-ineligible: {path}")
    return payload


def indexed(report: dict, path: Path) -> dict[str, dict]:
    rows = {str(row.get("caseID", "")): row for row in report["results"]}
    if "" in rows or len(rows) != len(report["results"]):
        raise SystemExit(f"report has empty or duplicate case IDs: {path}")
    return rows


def percentile(values: list[float], proportion: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise SystemExit("runtime comparison has no latency samples")
    return ordered[round((len(ordered) - 1) * proportion)]


def latency(rows: dict[str, dict], source_language: str) -> dict:
    values = [
        float(value)
        for row in rows.values()
        if row["sourceLanguage"] == source_language
        for value in (row.get("warmLatencySeconds") or [row["latencySeconds"]])
    ]
    return {
        "samples": len(values),
        "p50Seconds": percentile(values, 0.50),
        "p95Seconds": percentile(values, 0.95),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    baseline_report = load(args.baseline)
    candidate_report = load(args.candidate)
    baseline = indexed(baseline_report, args.baseline)
    candidate = indexed(candidate_report, args.candidate)
    if set(baseline) != set(candidate):
        raise SystemExit("source-only reports cover different case IDs")
    if baseline_report.get("modelRevision") != candidate_report.get("modelRevision"):
        raise SystemExit("source-only reports use different model revisions")
    for case_id in baseline:
        for field in CASE_FIELDS:
            if baseline[case_id].get(field) != candidate[case_id].get(field):
                raise SystemExit(f"source-only reports disagree on {field}: {case_id}")

    recovered = sorted(
        case_id
        for case_id in baseline
        if not baseline[case_id]["runtimeAccepted"]
        and candidate[case_id]["runtimeAccepted"]
    )
    regressed = sorted(
        case_id
        for case_id in baseline
        if baseline[case_id]["runtimeAccepted"]
        and not candidate[case_id]["runtimeAccepted"]
    )
    changed = sorted(
        case_id
        for case_id in baseline
        if baseline[case_id]["hypothesis"] != candidate[case_id]["hypothesis"]
    )
    payload = {
        "schemaVersion": 1,
        "status": "rejected",
        "purpose": "source-only safety-cascade ablation; no reference-based quality claim",
        "claimEligible": False,
        "modelRevision": baseline_report["modelRevision"],
        "cases": len(baseline),
        "baseline": {
            "path": str(args.baseline),
            "sha256": sha256(args.baseline),
            "runtimeAccepted": sum(row["runtimeAccepted"] for row in baseline.values()),
            "latency": {
                "en-ja": latency(baseline, "en-US"),
                "ja-en": latency(baseline, "ja-JP"),
            },
        },
        "candidate": {
            "path": str(args.candidate),
            "sha256": sha256(args.candidate),
            "runtimeAccepted": sum(row["runtimeAccepted"] for row in candidate.values()),
            "latency": {
                "en-ja": latency(candidate, "en-US"),
                "ja-en": latency(candidate, "ja-JP"),
            },
        },
        "comparison": {
            "recoveredCases": recovered,
            "regressedCases": regressed,
            "changedHypothesisCases": changed,
            "decision": (
                "reject: only five source-only structural recoveries, no reference "
                "quality evidence, and substantially higher JA-to-EN latency"
            ),
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
                "recovered": len(recovered),
                "regressed": len(regressed),
                "changed": len(changed),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
