#!/usr/bin/env python3
"""Create two independent, blinded review packets for a human-authored suite."""

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


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(seed: str, reviewer: str, case_id: str, value: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{reviewer}\0{case_id}\0{value}".encode()).digest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("draft_suite", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--reviewer", action="append", required=True)
    parser.add_argument("--shuffle-seed", default="mimi-heldout-reference-review-v1")
    args = parser.parse_args()

    reviewers = [value.strip() for value in args.reviewer if value.strip()]
    if len(reviewers) != 2 or len(set(reviewers)) != 2:
        raise SystemExit("provide exactly two distinct --reviewer values")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_directory}")

    draft = rows(args.draft_suite)
    identifiers = [str(row.get("id", "")).strip() for row in draft]
    if not draft or not all(identifiers) or len(identifiers) != len(set(identifiers)):
        raise SystemExit("draft suite must have non-empty unique IDs")
    for row in draft:
        case_id = row["id"]
        references = row.get("references", [])
        if row.get("claimEligible") is not False:
            raise SystemExit(f"draft must not be claim eligible: {case_id}")
        if row.get("sourceGeneratedByAI") is not False or row.get("referenceGeneratedByAI") is not False:
            raise SystemExit(f"draft must explicitly declare non-AI source and references: {case_id}")
        source_author = str(row.get("sourceAuthorID", "")).strip()
        reference_authors = [
            str(value).strip() for value in row.get("referenceAuthorIDs", [])
        ]
        if (
            not source_author
            or len(reference_authors) != len(references)
            or not all(reference_authors)
            or len(set(reference_authors)) != len(reference_authors)
            or source_author in set(reference_authors)
        ):
            raise SystemExit(f"draft lacks independent source/reference authors: {case_id}")
        if not str(row.get("documentID", "")).strip() or not str(row.get("source", "")).strip():
            raise SystemExit(f"draft is missing documentID/source: {case_id}")
        if len(references) < 2 or not all(str(value).strip() for value in references):
            raise SystemExit(f"draft needs at least two non-empty references: {case_id}")

    args.output_directory.mkdir(parents=True, exist_ok=True)
    for reviewer in reviewers:
        packet: list[dict] = []
        template: list[dict] = []
        for row in draft:
            case_id = str(row["id"])
            shuffled_references = sorted(
                row["references"],
                key=lambda value: rank(args.shuffle_seed, reviewer, case_id, str(value)),
            )
            packet.append(
                {
                    "reviewItemID": hashlib.sha256(
                        f"{reviewer}\0{case_id}".encode()
                    ).hexdigest()[:24],
                    "caseID": case_id,
                    "reviewerID": reviewer,
                    "sourceLanguage": row["sourceLanguage"],
                    "targetLanguage": row["targetLanguage"],
                    "domain": row["domain"],
                    "source": row["source"],
                    "references": shuffled_references,
                    "instruction": (
                        "Blindly verify that the source is natural and every reference is faithful, "
                        "complete, fluent, and suitable for the declared domain. Reject on any "
                        "meaning, negation, number, entity, safety, or material register error."
                    ),
                }
            )
            template.append(
                {
                    "caseID": case_id,
                    "reviewerID": reviewer,
                    "blinded": True,
                    "attestations": {
                        "human": False,
                        "bilingualQualified": False,
                        "independent": False,
                        "noAIAssistance": False,
                    },
                    "decision": "pending",
                    "criticalError": False,
                    "issueTags": [],
                    "notes": "",
                }
            )
        packet.sort(key=lambda row: rank(args.shuffle_seed, reviewer, str(row["caseID"]), "case"))
        template.sort(key=lambda row: str(row["caseID"]))
        write_jsonl(args.output_directory / f"{reviewer}.packet.jsonl", packet)
        write_jsonl(args.output_directory / f"{reviewer}.responses.jsonl", template)

    manifest = {
        "schemaVersion": 1,
        "purpose": "blind human source/reference review; no model output",
        "reviewers": reviewers,
        "cases": len(draft),
        "shuffleSeed": args.shuffle_seed,
        "draftSuite": {"path": str(args.draft_suite), "sha256": sha256(args.draft_suite)},
        "packetPolicy": "case and reference order independently shuffled; provenance and author identity omitted",
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
