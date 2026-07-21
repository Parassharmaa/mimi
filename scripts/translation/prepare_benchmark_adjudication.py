#!/usr/bin/env python3
"""Prepare an independent adjudication packet after two reference reviews."""

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


def has_required_attestations(row: dict) -> bool:
    attestations = row.get("attestations")
    return isinstance(attestations, dict) and all(
        attestations.get(name) is True
        for name in ("human", "bilingualQualified", "independent", "noAIAssistance")
    )


def response_map(path: Path, case_ids: set[str]) -> tuple[str, dict[str, dict]]:
    output: dict[str, dict] = {}
    reviewers: set[str] = set()
    for row in rows(path):
        case_id = str(row.get("caseID", "")).strip()
        reviewer = str(row.get("reviewerID", "")).strip()
        if case_id not in case_ids or case_id in output:
            raise SystemExit(f"unknown or duplicate review case: {case_id}")
        if not reviewer or row.get("blinded") is not True:
            raise SystemExit(f"review is not attributed and blind: {case_id}")
        if not has_required_attestations(row):
            raise SystemExit(f"review is missing required human attestations: {case_id}")
        if row.get("decision") not in {"approve", "reject"}:
            raise SystemExit(f"review remains pending or invalid: {case_id}")
        if row.get("decision") == "approve" and row.get("criticalError"):
            raise SystemExit(f"review cannot approve with criticalError: {case_id}")
        reviewers.add(reviewer)
        output[case_id] = row
    if len(reviewers) != 1 or set(output) != case_ids:
        raise SystemExit(f"review file must contain every case for exactly one reviewer: {path}")
    return next(iter(reviewers)), output


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("draft_suite", type=Path)
    parser.add_argument("review_a", type=Path)
    parser.add_argument("review_b", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--adjudicator", required=True)
    args = parser.parse_args()

    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_directory}")
    draft = rows(args.draft_suite)
    by_id = {str(row["id"]): row for row in draft}
    if len(by_id) != len(draft):
        raise SystemExit("draft has duplicate IDs")
    reviewer_a, reviews_a = response_map(args.review_a, set(by_id))
    reviewer_b, reviews_b = response_map(args.review_b, set(by_id))
    adjudicator = args.adjudicator.strip()
    if not adjudicator or len({reviewer_a, reviewer_b, adjudicator}) != 3:
        raise SystemExit("reviewers and adjudicator must be three distinct people")

    packet: list[dict] = []
    template: list[dict] = []
    for case_id, row in by_id.items():
        packet.append(
            {
                "caseID": case_id,
                "adjudicatorID": adjudicator,
                "sourceLanguage": row["sourceLanguage"],
                "targetLanguage": row["targetLanguage"],
                "domain": row["domain"],
                "source": row["source"],
                "references": row["references"],
                "reviewA": reviews_a[case_id],
                "reviewB": reviews_b[case_id],
                "instruction": (
                    "Approve the unchanged source and references only if every material issue is "
                    "resolved. Any critical reviewer flag requires rejection and revision followed "
                    "by a fresh two-reviewer cycle."
                ),
            }
        )
        template.append(
            {
                "caseID": case_id,
                "adjudicatorID": adjudicator,
                "attestations": {
                    "human": False,
                    "bilingualQualified": False,
                    "independent": False,
                    "noAIAssistance": False,
                },
                "decision": "pending",
                "criticalError": False,
                "notes": "",
            }
        )
    packet.sort(key=lambda row: hashlib.sha256(str(row["caseID"]).encode()).digest())
    template.sort(key=lambda row: str(row["caseID"]))
    args.output_directory.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_directory / f"{adjudicator}.packet.jsonl", packet)
    write_jsonl(args.output_directory / f"{adjudicator}.responses.jsonl", template)
    manifest = {
        "schemaVersion": 1,
        "reviewers": [reviewer_a, reviewer_b],
        "adjudicator": adjudicator,
        "cases": len(draft),
        "draftSuiteSHA256": hashlib.sha256(args.draft_suite.read_bytes()).hexdigest(),
        "reviewASHA256": hashlib.sha256(args.review_a.read_bytes()).hexdigest(),
        "reviewBSHA256": hashlib.sha256(args.review_b.read_bytes()).hexdigest(),
        "requiredAttestations": [
            "human",
            "bilingualQualified",
            "independent",
            "noAIAssistance",
        ],
    }
    (args.output_directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
