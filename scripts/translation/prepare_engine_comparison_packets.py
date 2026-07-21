#!/usr/bin/env python3
"""Blind and independently randomize Apple/student outputs for human scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


IMMUTABLE_RESULT_FIELDS = (
    "sourceLanguage",
    "targetLanguage",
    "domain",
    "source",
    "references",
    "claimEligible",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def index(report: dict, name: str) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for row in report.get("results", []):
        case_id = str(row.get("caseID", "")).strip()
        if not case_id or case_id in output:
            raise SystemExit(f"{name} report has empty or duplicate caseID: {case_id}")
        output[case_id] = row
    if not output:
        raise SystemExit(f"{name} report has no results")
    return output


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_report", type=Path)
    parser.add_argument("apple_report", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--reviewer", action="append", required=True)
    parser.add_argument("--shuffle-seed", default="mimi-apple-candidate-blind-v1")
    parser.add_argument(
        "--baseline-key",
        choices=("apple", "base"),
        default="apple",
        help="Sealed identity for the second report; use base for supervised-student ablations.",
    )
    args = parser.parse_args()

    reviewers = [value.strip() for value in args.reviewer if value.strip()]
    if len(reviewers) != 2 or len(set(reviewers)) != 2:
        raise SystemExit("provide exactly two distinct --reviewer values")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_directory}")

    candidate_report, apple_report = load(args.candidate_report), load(args.apple_report)
    candidate, apple = index(candidate_report, "candidate"), index(apple_report, "Apple")
    if set(candidate) != set(apple):
        raise SystemExit("candidate and Apple reports must contain exactly the same case IDs")
    for case_id in candidate:
        for field in IMMUTABLE_RESULT_FIELDS:
            if candidate[case_id].get(field) != apple[case_id].get(field):
                raise SystemExit(f"reports disagree on {field}: {case_id}")

    args.output_directory.mkdir(parents=True, exist_ok=True)
    assignments: list[dict] = []
    for reviewer in reviewers:
        packet: list[dict] = []
        template: list[dict] = []
        for case_id in sorted(candidate):
            candidate_first = hashlib.sha256(
                f"{args.shuffle_seed}\0{reviewer}\0{case_id}\0assignment".encode()
            ).digest()[0] % 2 == 0
            engines = (
                ("candidate", args.baseline_key)
                if candidate_first
                else (args.baseline_key, "candidate")
            )
            hypotheses = {
                "candidate": str(candidate[case_id]["hypothesis"]),
                args.baseline_key: str(apple[case_id]["hypothesis"]),
            }
            assignment = {
                "caseID": case_id,
                "reviewerID": reviewer,
                "outputAEngine": engines[0],
                "outputBEngine": engines[1],
                "outputASHA256": text_hash(hypotheses[engines[0]]),
                "outputBSHA256": text_hash(hypotheses[engines[1]]),
            }
            assignments.append(assignment)
            packet.append(
                {
                    "reviewItemID": hashlib.sha256(
                        f"{reviewer}\0{case_id}".encode()
                    ).hexdigest()[:24],
                    "caseID": case_id,
                    "reviewerID": reviewer,
                    "sourceLanguage": candidate[case_id]["sourceLanguage"],
                    "targetLanguage": candidate[case_id]["targetLanguage"],
                    "domain": candidate[case_id]["domain"],
                    "source": candidate[case_id]["source"],
                    "outputA": hypotheses[engines[0]],
                    "outputB": hypotheses[engines[1]],
                    "rubric": {
                        "adequacy": "integer 0-4; meaning completeness and correctness",
                        "fluency": "integer 0-4; natural target-language expression",
                        "terminology": "integer 0-2; entities, UI terms, numbers, and register",
                        "criticalError": (
                            "true for meaning reversal, negation, material omission/addition, "
                            "wrong entity/number, or safety-impacting translation"
                        ),
                    },
                }
            )
            template.append(
                {
                    "caseID": case_id,
                    "reviewerID": reviewer,
                    "blinded": True,
                    "outputA": {
                        "adequacy": None,
                        "fluency": None,
                        "terminology": None,
                        "criticalError": False,
                        "errorTags": [],
                        "notes": "",
                    },
                    "outputB": {
                        "adequacy": None,
                        "fluency": None,
                        "terminology": None,
                        "criticalError": False,
                        "errorTags": [],
                        "notes": "",
                    },
                }
            )
        packet.sort(
            key=lambda row: hashlib.sha256(
                f"{args.shuffle_seed}\0{reviewer}\0{row['caseID']}\0case-order".encode()
            ).digest()
        )
        template.sort(key=lambda row: str(row["caseID"]))
        write_jsonl(args.output_directory / f"{reviewer}.packet.jsonl", packet)
        write_jsonl(args.output_directory / f"{reviewer}.responses.jsonl", template)

    assignments.sort(key=lambda row: (str(row["reviewerID"]), str(row["caseID"])))
    write_jsonl(args.output_directory / "sealed-assignments.jsonl", assignments)
    manifest = {
        "schemaVersion": 1,
        "purpose": f"blind bilingual {args.baseline_key}-vs-candidate human quality comparison",
        "baselineKey": args.baseline_key,
        "reviewers": reviewers,
        "cases": len(candidate),
        "shuffleSeed": args.shuffle_seed,
        "candidateReport": {"path": str(args.candidate_report), "sha256": sha256(args.candidate_report)},
        "appleReport": {"path": str(args.apple_report), "sha256": sha256(args.apple_report)},
        "sealedAssignments": "Do not provide sealed-assignments.jsonl to reviewers before responses are frozen.",
        "scoreRange": {"adequacy": [0, 4], "fluency": [0, 4], "terminology": [0, 2]},
    }
    (args.output_directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
