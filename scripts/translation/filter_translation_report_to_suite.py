#!/usr/bin/env python3
"""Create a hash-bound report subset matching an exact benchmark suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MATCH_FIELDS = (
    "sourceLanguage",
    "targetLanguage",
    "domain",
    "source",
    "references",
    "claimEligible",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_suite(path: Path) -> list[dict]:
    rows = []
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        case_id = row.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise SystemExit(f"invalid or duplicate suite id at {path}:{line_number}")
        seen.add(case_id)
        rows.append(row)
    if not rows:
        raise SystemExit(f"suite is empty: {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    suite = load_suite(args.suite)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    results = report.get("results")
    if not isinstance(results, list):
        raise SystemExit(f"report lacks results: {args.report}")
    indexed = {}
    for row in results:
        case_id = row.get("caseID")
        if not isinstance(case_id, str) or not case_id or case_id in indexed:
            raise SystemExit(f"report has an invalid or duplicate caseID: {case_id}")
        indexed[case_id] = row

    subset = []
    for suite_row in suite:
        case_id = suite_row["id"]
        report_row = indexed.get(case_id)
        if report_row is None:
            raise SystemExit(f"suite case is absent from report: {case_id}")
        for field in MATCH_FIELDS:
            if suite_row.get(field) != report_row.get(field):
                raise SystemExit(f"suite/report {field} mismatch: {case_id}")
        subset.append(report_row)

    payload = {
        **{key: value for key, value in report.items() if key != "results"},
        "authenticatedSubset": {
            "suite": {"path": str(args.suite), "sha256": sha256(args.suite)},
            "sourceReport": {
                "path": str(args.report),
                "sha256": sha256(args.report),
            },
            "cases": len(subset),
        },
        "results": subset,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"cases": len(subset), "output": str(args.output)}))


if __name__ == "__main__":
    main()
