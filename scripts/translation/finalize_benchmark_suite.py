#!/usr/bin/env python3
"""Finalize only fully reviewed and independently adjudicated held-out cases."""

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


def required_attestations(row: dict, case_id: str) -> dict:
    attestations = row.get("attestations")
    names = ("human", "bilingualQualified", "independent", "noAIAssistance")
    if not isinstance(attestations, dict) or not all(
        attestations.get(name) is True for name in names
    ):
        raise SystemExit(f"decision is missing required human attestations: {case_id}")
    return {name: True for name in names}


def mapping(path: Path, key: str, expected: set[str]) -> tuple[str, dict[str, dict]]:
    output: dict[str, dict] = {}
    people: set[str] = set()
    for row in rows(path):
        case_id = str(row.get("caseID", "")).strip()
        person = str(row.get(key, "")).strip()
        if case_id not in expected or case_id in output or not person:
            raise SystemExit(f"unknown, duplicate, or unattributed decision: {case_id}")
        if row.get("decision") not in {"approve", "reject"}:
            raise SystemExit(f"pending or invalid decision: {case_id}")
        required_attestations(row, case_id)
        people.add(person)
        output[case_id] = row
    if len(people) != 1 or set(output) != expected:
        raise SystemExit(f"decision file must cover every case for exactly one person: {path}")
    return next(iter(people)), output


def case_digest(row: dict) -> str:
    protected = {
        key: row[key]
        for key in (
            "id", "documentID", "sourceLanguage", "targetLanguage", "domain",
            "source", "references", "sourceAuthorID", "referenceAuthorIDs",
            "sourceGeneratedByAI", "referenceGeneratedByAI", "split", "license",
            "provenance",
        )
    }
    return hashlib.sha256(
        json.dumps(protected, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("draft_suite", type=Path)
    parser.add_argument("review_a", type=Path)
    parser.add_argument("review_b", type=Path)
    parser.add_argument("adjudications", type=Path)
    parser.add_argument("final_suite", type=Path)
    parser.add_argument("review_records", type=Path)
    parser.add_argument("rejected", type=Path)
    args = parser.parse_args()

    draft = rows(args.draft_suite)
    by_id = {str(row["id"]): row for row in draft}
    if not draft or len(by_id) != len(draft):
        raise SystemExit("draft must have non-empty unique IDs")
    expected = set(by_id)
    reviewer_a, decisions_a = mapping(args.review_a, "reviewerID", expected)
    reviewer_b, decisions_b = mapping(args.review_b, "reviewerID", expected)
    adjudicator, adjudications = mapping(args.adjudications, "adjudicatorID", expected)
    if len({reviewer_a, reviewer_b, adjudicator}) != 3:
        raise SystemExit("two reviewers and adjudicator must be distinct")

    final: list[dict] = []
    records: list[dict] = []
    rejected: list[dict] = []
    for case_id, draft_row in by_id.items():
        left, right, adjudication = decisions_a[case_id], decisions_b[case_id], adjudications[case_id]
        critical = bool(left.get("criticalError") or right.get("criticalError") or adjudication.get("criticalError"))
        if adjudication["decision"] != "approve" or critical:
            rejected.append(
                {
                    "caseID": case_id,
                    "reason": "critical-error" if critical else "adjudicator-rejected",
                    "reviewA": left,
                    "reviewB": right,
                    "adjudication": adjudication,
                }
            )
            continue
        author_ids = {
            str(draft_row.get("sourceAuthorID", "")).strip(),
            *(
                str(value).strip()
                for value in draft_row.get("referenceAuthorIDs", [])
            ),
        }
        if "" in author_ids or author_ids & {reviewer_a, reviewer_b, adjudicator}:
            raise SystemExit(f"authors, reviewers, and adjudicator must be distinct: {case_id}")
        final_row = {
            **draft_row,
            "split": "heldout",
            "reviewStatus": "adjudicated",
            "claimEligible": True,
        }
        final.append(final_row)
        records.append(
            {
                "caseID": case_id,
                "blinded": bool(left.get("blinded") and right.get("blinded")),
                "reviewerIDs": sorted([reviewer_a, reviewer_b]),
                "reviewerAttestations": {
                    reviewer_a: required_attestations(left, case_id),
                    reviewer_b: required_attestations(right, case_id),
                },
                "reviewDecisions": [left, right],
                "adjudicatorID": adjudicator,
                "adjudicatorAttestations": required_attestations(adjudication, case_id),
                "adjudication": adjudication,
                "decision": "approved",
                "approvedReferences": final_row["references"],
                "suiteCaseSHA256": case_digest(final_row),
            }
        )
    final.sort(key=lambda row: str(row["id"]))
    records.sort(key=lambda row: str(row["caseID"]))
    rejected.sort(key=lambda row: str(row["caseID"]))
    write_jsonl(args.final_suite, final)
    write_jsonl(args.review_records, records)
    write_jsonl(args.rejected, rejected)
    print(
        json.dumps(
            {
                "draftCases": len(draft),
                "finalCases": len(final),
                "rejectedCases": len(rejected),
                "reviewers": [reviewer_a, reviewer_b],
                "adjudicator": adjudicator,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
