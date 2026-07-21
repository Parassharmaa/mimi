#!/usr/bin/env python3
"""Build reverse-translation suites for source-routed expert candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("results"), list):
        raise SystemExit(f"invalid translation report: {path}")
    return value


def indexed(report: dict, label: str) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for row in report["results"]:
        identifier = str(row.get("caseID", "")).strip()
        if not identifier or identifier in output:
            raise SystemExit(f"{label} has an empty or duplicate case ID")
        output[identifier] = row
    if not output:
        raise SystemExit(f"{label} has no cases")
    return output


def assert_same_case(left: dict, right: dict, case_id: str) -> None:
    for field in (
        "caseID",
        "sourceLanguage",
        "targetLanguage",
        "domain",
        "source",
        "references",
        "claimEligible",
    ):
        if left.get(field) != right.get(field):
            raise SystemExit(f"candidate reports disagree on {field}: {case_id}")


def reverse_row(original: dict, candidate: dict, kind: str) -> dict:
    hypothesis = str(candidate.get("hypothesis", "")).strip()
    if not hypothesis:
        raise SystemExit(f"empty {kind} hypothesis: {original['caseID']}")
    return {
        "id": f"roundtrip-{kind}:{original['caseID']}",
        "originalCaseID": original["caseID"],
        "forwardCandidate": kind,
        "sourceLanguage": original["targetLanguage"],
        "targetLanguage": original["sourceLanguage"],
        "domain": original["domain"],
        "source": hypothesis,
        "references": [original["source"]],
        "claimEligible": False,
        "split": "public-development-roundtrip-reranking",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("generalist_report", type=Path)
    parser.add_argument("en_ja_expert_report", type=Path)
    parser.add_argument("ja_en_expert_report", type=Path)
    parser.add_argument("routed_report", type=Path)
    parser.add_argument("generalist_suite", type=Path)
    parser.add_argument("expert_suite", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    for output in (args.generalist_suite, args.expert_suite, args.manifest):
        if output.exists() and output.stat().st_size:
            raise SystemExit(f"refusing to overwrite non-empty output: {output}")

    reports = {
        "generalist": load(args.generalist_report),
        "en-ja-expert": load(args.en_ja_expert_report),
        "ja-en-expert": load(args.ja_en_expert_report),
        "routed": load(args.routed_report),
    }
    indexed_reports = {
        name: indexed(report, name) for name, report in reports.items()
    }
    case_ids = set(indexed_reports["generalist"])
    if any(set(values) != case_ids for values in indexed_reports.values()):
        raise SystemExit("candidate reports do not have exact common coverage")

    declared_inputs = reports["routed"].get("inputs", {})
    expected_inputs = {
        "generalistReport": args.generalist_report,
        "enJAExpertReport": args.en_ja_expert_report,
        "jaENExpertReport": args.ja_en_expert_report,
    }
    for key, path in expected_inputs.items():
        record = declared_inputs.get(key, {})
        if record.get("sha256") != sha256(path):
            raise SystemExit(f"routed report is not bound to {key}")

    generalist_rows: list[dict] = []
    expert_rows: list[dict] = []
    direction_counts = {"en-ja": 0, "ja-en": 0}
    for routed in reports["routed"]["results"]:
        if routed.get("selectedEngine") != "expert":
            continue
        case_id = routed["caseID"]
        generalist = indexed_reports["generalist"][case_id]
        direction = "en-ja" if routed["sourceLanguage"] == "en-US" else "ja-en"
        expert = indexed_reports[f"{direction}-expert"][case_id]
        assert_same_case(routed, generalist, case_id)
        assert_same_case(routed, expert, case_id)
        if routed.get("hypothesis") != expert.get("hypothesis"):
            raise SystemExit(f"routed expert output is not the expert candidate: {case_id}")
        generalist_rows.append(reverse_row(routed, generalist, "generalist"))
        expert_rows.append(reverse_row(routed, expert, "expert"))
        direction_counts[direction] += 1

    if not generalist_rows or len(generalist_rows) != len(expert_rows):
        raise SystemExit("routed report has no exact expert-candidate population")
    args.generalist_suite.parent.mkdir(parents=True, exist_ok=True)
    args.expert_suite.parent.mkdir(parents=True, exist_ok=True)
    for path, values in (
        (args.generalist_suite, generalist_rows),
        (args.expert_suite, expert_rows),
    ):
        path.write_text(
            "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values),
            encoding="utf-8",
        )
    manifest = {
        "schemaVersion": 1,
        "purpose": "public-development source-routed expert roundtrip reranking ablation",
        "claimEligible": False,
        "policy": "only source-router expert cases; reverse with frozen generalists",
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in (
                ("generalistReport", args.generalist_report),
                ("enJAExpertReport", args.en_ja_expert_report),
                ("jaENExpertReport", args.ja_en_expert_report),
                ("routedReport", args.routed_report),
            )
        },
        "counts": {"cases": len(generalist_rows), "directions": direction_counts},
        "outputs": {
            "generalistSuite": {
                "path": str(args.generalist_suite),
                "sha256": sha256(args.generalist_suite),
            },
            "expertSuite": {
                "path": str(args.expert_suite),
                "sha256": sha256(args.expert_suite),
            },
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
