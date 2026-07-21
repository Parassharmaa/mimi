#!/usr/bin/env python3
"""Compare different authenticated MoE candidates on a source-only audit.

This deliberately does not score translation quality. It records fail-closed
runtime transitions and the upper-bound acceptance of retaining both outputs.
"""

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


def report_record(path: Path, report: dict, rows: dict[str, dict]) -> dict:
    return {
        "path": str(path),
        "sha256": sha256(path),
        "modelRevision": report.get("modelRevision"),
        "modelBytes": report.get("modelBytes"),
        "runtimeAccepted": sum(bool(row["runtimeAccepted"]) for row in rows.values()),
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
    if baseline_report.get("runtimeImplementation") != candidate_report.get("runtimeImplementation"):
        raise SystemExit("source-only reports use different runtime implementations")
    if baseline_report.get("benchmarkConfiguration") != candidate_report.get("benchmarkConfiguration"):
        raise SystemExit("source-only reports use different benchmark configurations")
    for case_id in baseline:
        for field in CASE_FIELDS:
            if baseline[case_id].get(field) != candidate[case_id].get(field):
                raise SystemExit(f"source-only reports disagree on {field}: {case_id}")

    baseline_only = sorted(
        case_id
        for case_id in baseline
        if baseline[case_id]["runtimeAccepted"]
        and not candidate[case_id]["runtimeAccepted"]
    )
    candidate_only = sorted(
        case_id
        for case_id in baseline
        if candidate[case_id]["runtimeAccepted"]
        and not baseline[case_id]["runtimeAccepted"]
    )
    both_accepted = sorted(
        case_id
        for case_id in baseline
        if baseline[case_id]["runtimeAccepted"]
        and candidate[case_id]["runtimeAccepted"]
    )
    both_failed = sorted(
        case_id
        for case_id in baseline
        if not baseline[case_id]["runtimeAccepted"]
        and not candidate[case_id]["runtimeAccepted"]
    )
    changed = sorted(
        case_id
        for case_id in baseline
        if baseline[case_id]["hypothesis"] != candidate[case_id]["hypothesis"]
    )
    union_accepted = len(both_accepted) + len(baseline_only) + len(candidate_only)
    baseline_accepted = len(both_accepted) + len(baseline_only)
    candidate_accepted = len(both_accepted) + len(candidate_only)
    candidate_delta = candidate_accepted - baseline_accepted
    union_gain = union_accepted - baseline_accepted
    if candidate_delta < 0:
        status = "candidate-rejected-runtime-safety"
        decision = (
            f"reject candidate pack: fail-closed acceptance decreases by "
            f"{-candidate_delta}; retaining both outputs recovers {union_gain} "
            "cases but has no reference-based quality authorization"
        )
    elif candidate_delta > 0:
        status = "candidate-runtime-improvement-unvalidated"
        decision = (
            f"candidate accepts {candidate_delta} additional sources, but "
            "source-only structural evidence cannot authorize translation quality"
        )
    else:
        status = "candidate-runtime-neutral-unvalidated"
        decision = (
            "candidate preserves fail-closed acceptance, but source-only structural "
            "evidence cannot authorize translation quality"
        )
    payload = {
        "schemaVersion": 1,
        "status": status,
        "purpose": "different-model source-only runtime comparison; no translation-quality claim",
        "claimEligible": False,
        "cases": len(baseline),
        "baseline": report_record(args.baseline, baseline_report, baseline),
        "candidate": report_record(args.candidate, candidate_report, candidate),
        "comparison": {
            "bothAcceptedCases": both_accepted,
            "baselineOnlyAcceptedCases": baseline_only,
            "candidateOnlyAcceptedCases": candidate_only,
            "bothFailedCases": both_failed,
            "changedHypothesisCases": changed,
            "hypotheticalFailClosedUnionAccepted": union_accepted,
            "hypotheticalUnionGainOverBaseline": union_gain,
            "candidateAcceptanceDelta": candidate_delta,
            "decision": decision,
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
                "baselineAccepted": baseline_accepted,
                "candidateAccepted": candidate_accepted,
                "candidateDelta": candidate_delta,
                "unionAccepted": union_accepted,
                "unionGain": union_gain,
                "changed": len(changed),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
