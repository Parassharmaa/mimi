#!/usr/bin/env python3
"""Build a non-claimable reverse suite from one local teacher's hypotheses."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("forward_suite", type=Path)
    parser.add_argument("teacher_report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists() and args.output.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")

    forward = rows(args.forward_suite)
    report = json.loads(args.teacher_report.read_text(encoding="utf-8"))
    hypotheses: dict[str, str] = {}
    for result in report.get("results", []):
        identifier = str(result.get("caseID", ""))
        hypothesis = str(result.get("hypothesis", "")).strip()
        if not identifier or not hypothesis or identifier in hypotheses:
            raise SystemExit("teacher report has missing, empty, or duplicate hypotheses")
        hypotheses[identifier] = hypothesis
    expected = {str(row["id"]) for row in forward}
    if set(hypotheses) != expected:
        raise SystemExit("teacher report does not cover the exact forward suite")

    output = [{
        "id": row["id"],
        "sourceLanguage": row["targetLanguage"],
        "targetLanguage": row["sourceLanguage"],
        "domain": row.get("domain", "unknown"),
        "source": hypotheses[str(row["id"])],
        "references": [],
        "claimEligible": False,
        "roundTripOriginalSource": row["source"],
    } for row in forward]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "purpose": "local teacher round-trip filtering; never evaluation evidence",
        "claim_eligible": False,
        "rows": len(output),
        "inputs": {
            "forward_suite": {"path": str(args.forward_suite.resolve()), "sha256": sha256(args.forward_suite)},
            "teacher_report": {"path": str(args.teacher_report.resolve()), "sha256": sha256(args.teacher_report)},
        },
        "output": {"path": str(args.output.resolve()), "sha256": sha256(args.output)},
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
