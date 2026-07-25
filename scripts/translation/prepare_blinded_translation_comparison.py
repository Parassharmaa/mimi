#!/usr/bin/env python3
"""Prepare a reproducible source-based blinded A/B translation judge batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

ALLOWED_TAGS = [
    "meaning-reversal",
    "agency",
    "negation",
    "tense-or-aspect",
    "number-or-date",
    "named-entity",
    "omission",
    "addition",
    "placeholder-or-markup",
    "code-switching",
    "register",
    "terminology",
    "disfluency",
    "wrong-language",
    "empty-output",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def report_index(path: Path, label: str) -> tuple[dict, dict[str, dict]]:
    report = load_json(path)
    results = report.get("results")
    if not isinstance(results, list) or not results:
        raise SystemExit(f"{label} report has no results")
    indexed = {}
    for row in results:
        case_id = str(row.get("caseID", ""))
        if not case_id or case_id in indexed:
            raise SystemExit(f"{label} report has missing or duplicate case ID")
        indexed[case_id] = row
    return report, indexed


def stable_rank(seed: str, *parts: str) -> str:
    return hashlib.sha256("\0".join((seed, *parts)).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("candidate_report", type=Path)
    parser.add_argument("baseline_report", type=Path)
    parser.add_argument("batch_output", type=Path)
    parser.add_argument("private_mapping_output", type=Path)
    parser.add_argument("rubric_output", type=Path)
    parser.add_argument(
        "--seed", default="mimi-development-accuracy-v1-blind-order-20260725"
    )
    parser.add_argument("--candidate-label", default="mimi")
    parser.add_argument("--baseline-label", default="apple")
    args = parser.parse_args()
    outputs = (
        args.batch_output,
        args.private_mapping_output,
        args.rubric_output,
    )
    if any(path.exists() for path in outputs):
        raise SystemExit("refusing to overwrite blinded comparison artifacts")
    if args.candidate_label == args.baseline_label:
        raise SystemExit("candidate and baseline labels must differ")

    suite_rows = load_jsonl(args.suite)
    suite = {str(row.get("id", "")): row for row in suite_rows}
    if not suite or "" in suite or len(suite) != len(suite_rows):
        raise SystemExit("suite has missing or duplicate IDs")
    candidate_report, candidate = report_index(args.candidate_report, "candidate")
    baseline_report, baseline = report_index(args.baseline_report, "baseline")
    if set(suite) != set(candidate) or set(suite) != set(baseline):
        raise SystemExit("suite and reports must contain identical case IDs")

    grouped: dict[str, list[str]] = defaultdict(list)
    for case_id, row in suite.items():
        direction = f"{row['sourceLanguage']}>{row['targetLanguage']}"
        grouped[direction].append(case_id)
        for label, result in (
            ("candidate", candidate[case_id]),
            ("baseline", baseline[case_id]),
        ):
            for field in (
                "sourceLanguage",
                "targetLanguage",
                "domain",
                "source",
                "references",
            ):
                if result.get(field) != row.get(field):
                    raise SystemExit(f"{label} disagrees with suite {field}: {case_id}")

    candidate_is_a: set[str] = set()
    direction_balance = {}
    for direction, case_ids in sorted(grouped.items()):
        ordered = sorted(
            case_ids,
            key=lambda case_id: stable_rank(args.seed, direction, case_id),
        )
        for index, case_id in enumerate(ordered):
            if index % 2 == 0:
                candidate_is_a.add(case_id)
        direction_balance[direction] = {
            "cases": len(ordered),
            "candidateA": sum(case_id in candidate_is_a for case_id in ordered),
            "candidateB": sum(case_id not in candidate_is_a for case_id in ordered),
        }

    batch_rows = []
    mapping_rows = []
    for case_id in sorted(suite):
        row = suite[case_id]
        candidate_a = candidate_is_a.__contains__(case_id)
        a_result = candidate[case_id] if candidate_a else baseline[case_id]
        b_result = baseline[case_id] if candidate_a else candidate[case_id]
        batch_rows.append(
            {
                "case_id": case_id,
                "source_language": row["sourceLanguage"],
                "target_language": row["targetLanguage"],
                "domain": row["domain"],
                "source_unit": row.get("sourceUnit", "sentence"),
                "segment_count": int(row.get("segmentCount", 1)),
                "source": row["source"],
                "candidate_A": str(a_result.get("hypothesis", "")),
                "candidate_B": str(b_result.get("hypothesis", "")),
            }
        )
        mapping_rows.append(
            {
                "case_id": case_id,
                "candidate_A_system": (
                    args.candidate_label if candidate_a else args.baseline_label
                ),
                "candidate_B_system": (
                    args.baseline_label if candidate_a else args.candidate_label
                ),
            }
        )

    args.batch_output.parent.mkdir(parents=True, exist_ok=True)
    args.private_mapping_output.parent.mkdir(parents=True, exist_ok=True)
    args.rubric_output.parent.mkdir(parents=True, exist_ok=True)
    args.batch_output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in batch_rows
        ),
        encoding="utf-8",
    )
    args.private_mapping_output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in mapping_rows
        ),
        encoding="utf-8",
    )
    os.chmod(args.private_mapping_output, 0o600)
    rubric = {
        "schemaVersion": 2,
        "title": "Mimi blinded source-based translation comparison",
        "claimEligible": False,
        "referenceShownToJudge": False,
        "candidateOrderSeed": args.seed,
        "batchSHA256": sha256(args.batch_output),
        "instructions": [
            "Judge every case independently from the source; model identities are hidden.",
            "Score adequacy before fluency. Preserve the complete proposition, agency, polarity, tense/aspect, names, numbers, dates, terminology, placeholders, and code-switching.",
            "For document cases, assess all ordered segments, omissions, terminology/name consistency, and register consistency across the complete document.",
            "Choose A, B, or tie. Use tie only when differences are immaterial or neither output is clearly better.",
            "Do not reward reference-like wording, verbosity, or unsupported detail.",
            "Return only structured fields and a concise audit justification; never retain or emit private reasoning traces.",
        ],
        "criticalErrorDefinition": (
            "true only for an empty or wrong-language output, a materially wrong core "
            "proposition, agency or polarity reversal, omission of a material clause "
            "or document segment, corruption of a consequential name/number/date/"
            "placeholder, or text too broken to use; style-only issues are not critical"
        ),
        "allowedErrorTags": ALLOWED_TAGS,
        "outputSchema": {
            "case_id": "string",
            "adequacy_winner": "A|B|tie",
            "fluency_winner": "A|B|tie",
            "overall_preference": "A|B|tie",
            "critical_error_A": "boolean",
            "critical_error_B": "boolean",
            "candidate_A_error_tags": "array of allowedErrorTags",
            "candidate_B_error_tags": "array of allowedErrorTags",
            "brief_justification": "maximum 50 words; no chain of thought",
        },
    }
    args.rubric_output.write_text(
        json.dumps(rubric, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schemaVersion": 1,
        "purpose": "non-claimable blinded development comparison",
        "suiteSHA256": sha256(args.suite),
        "candidateReport": {
            "path": str(args.candidate_report),
            "sha256": sha256(args.candidate_report),
            "engine": candidate_report.get("engine"),
            "modelRevision": candidate_report.get("modelRevision"),
        },
        "baselineReport": {
            "path": str(args.baseline_report),
            "sha256": sha256(args.baseline_report),
            "engine": baseline_report.get("engine"),
            "modelRevision": baseline_report.get("modelRevision"),
        },
        "batch": {
            "path": str(args.batch_output),
            "sha256": sha256(args.batch_output),
            "cases": len(batch_rows),
        },
        "privateMapping": {
            "path": str(args.private_mapping_output),
            "sha256": sha256(args.private_mapping_output),
            "mode": "0600",
        },
        "rubric": {
            "path": str(args.rubric_output),
            "sha256": sha256(args.rubric_output),
        },
        "directionBalance": direction_balance,
        "reasoningTracesStored": False,
    }
    manifest_path = args.batch_output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
