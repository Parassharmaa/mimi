#!/usr/bin/env python3
"""Create authenticated, content-aligned report intersections for diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ALIGNMENT_FIELDS = (
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
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise SystemExit(f"report lacks a results array: {path}")
    return payload


def key(row: dict) -> str:
    payload = {field: row.get(field) for field in ALIGNMENT_FIELDS}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def index(report: dict, path: Path) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for row in report["results"]:
        content_key = key(row)
        if content_key in output:
            raise SystemExit(f"report contains duplicate aligned content: {path}")
        output[content_key] = row
    return output


def aligned_id(content_key: str) -> str:
    digest = hashlib.sha256(content_key.encode("utf-8")).hexdigest()[:24]
    return f"diagnostic-intersection:{digest}"


def write_aligned(
    report: dict,
    report_path: Path,
    other_path: Path,
    common_keys: list[str],
    indexed: dict[str, dict],
    output_path: Path,
) -> None:
    results = []
    for content_key in common_keys:
        row = dict(indexed[content_key])
        row["originalCaseID"] = row.get("caseID")
        row["caseID"] = aligned_id(content_key)
        row["claimEligible"] = False
        results.append(row)
    output = {
        **{name: value for name, value in report.items() if name != "results"},
        "diagnosticAlignment": {
            "alignmentFields": list(ALIGNMENT_FIELDS),
            "postHocIntersection": True,
            "claimEligible": False,
            "sourceReport": {"path": str(report_path), "sha256": sha256(report_path)},
            "otherReport": {"path": str(other_path), "sha256": sha256(other_path)},
            "cases": len(results),
        },
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_suite(common_keys: list[str], indexed: dict[str, dict], output_path: Path) -> None:
    rows = []
    for content_key in common_keys:
        source = indexed[content_key]
        rows.append(
            {
                "id": aligned_id(content_key),
                "sourceLanguage": source["sourceLanguage"],
                "targetLanguage": source["targetLanguage"],
                "domain": source["domain"],
                "source": source["source"],
                "references": source["references"],
                "claimEligible": False,
                "split": "diagnostic-posthoc-intersection",
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_report", type=Path)
    parser.add_argument("baseline_report", type=Path)
    parser.add_argument("candidate_output", type=Path)
    parser.add_argument("baseline_output", type=Path)
    parser.add_argument("--suite-output", type=Path)
    parser.add_argument("--minimum-per-direction", type=int, default=400)
    args = parser.parse_args()
    if args.minimum_per_direction <= 0:
        raise SystemExit("--minimum-per-direction must be positive")

    candidate = load(args.candidate_report)
    baseline = load(args.baseline_report)
    candidate_index = index(candidate, args.candidate_report)
    baseline_index = index(baseline, args.baseline_report)
    common_keys = sorted(set(candidate_index) & set(baseline_index))
    if not common_keys:
        raise SystemExit("reports have no content-aligned cases")

    direction_counts: dict[str, int] = {}
    for content_key in common_keys:
        row = candidate_index[content_key]
        direction = f"{row['sourceLanguage']}>{row['targetLanguage']}"
        direction_counts[direction] = direction_counts.get(direction, 0) + 1
    if len(direction_counts) != 2 or any(
        count < args.minimum_per_direction for count in direction_counts.values()
    ):
        raise SystemExit(
            "aligned intersection does not meet the two-direction minimum: "
            f"{direction_counts}"
        )

    write_aligned(
        candidate,
        args.candidate_report,
        args.baseline_report,
        common_keys,
        candidate_index,
        args.candidate_output,
    )
    write_aligned(
        baseline,
        args.baseline_report,
        args.candidate_report,
        common_keys,
        baseline_index,
        args.baseline_output,
    )
    if args.suite_output is not None:
        write_suite(common_keys, candidate_index, args.suite_output)
    print(json.dumps({"cases": len(common_keys), "directions": direction_counts}))


if __name__ == "__main__":
    main()
