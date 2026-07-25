#!/usr/bin/env python3
"""Unblind and summarize a structured pairwise translation judge report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

WIN_FIELDS = ("adequacy_winner", "fluency_winner", "overall_preference")
ALLOWED_ERROR_TAGS = {
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
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def index(rows: list[dict], key: str, label: str) -> dict[str, dict]:
    output = {}
    for row in rows:
        identifier = str(row.get(key, ""))
        if not identifier or identifier in output:
            raise SystemExit(f"{label} has missing or duplicate {key}")
        output[identifier] = row
    return output


def wilson(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return [center - radius, center + radius]


def exact_two_sided_sign_test(left_wins: int, right_wins: int) -> float:
    total = left_wins + right_wins
    if total == 0:
        return 1.0
    observed = min(left_wins, right_wins)
    tail = sum(math.comb(total, value) for value in range(observed + 1)) / (2**total)
    return min(1.0, 2 * tail)


def preference_summary(
    rows: list[dict], field: str, candidate_label: str, baseline_label: str
) -> dict:
    candidate_wins = sum(row[f"{field}_system"] == candidate_label for row in rows)
    baseline_wins = sum(row[f"{field}_system"] == baseline_label for row in rows)
    ties = sum(row[f"{field}_system"] == "tie" for row in rows)
    decisive = candidate_wins + baseline_wins
    return {
        "candidateWins": candidate_wins,
        "baselineWins": baseline_wins,
        "ties": ties,
        "decisiveCases": decisive,
        "candidateDecisiveWinRate": (candidate_wins / decisive if decisive else None),
        "candidateDecisiveWinRateWilson95": wilson(candidate_wins, decisive),
        "twoSidedExactSignTestP": exact_two_sided_sign_test(
            candidate_wins, baseline_wins
        ),
    }


def group_summary(rows: list[dict], candidate_label: str, baseline_label: str) -> dict:
    critical = Counter()
    tags: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        for system in (candidate_label, baseline_label):
            if row[f"{system}_critical_error"]:
                critical[system] += 1
            tags[system].update(row[f"{system}_error_tags"])
    return {
        "cases": len(rows),
        "adequacy": preference_summary(
            rows, "adequacy_winner", candidate_label, baseline_label
        ),
        "fluency": preference_summary(
            rows, "fluency_winner", candidate_label, baseline_label
        ),
        "overallPreference": preference_summary(
            rows, "overall_preference", candidate_label, baseline_label
        ),
        "criticalErrors": {
            candidate_label: critical[candidate_label],
            baseline_label: critical[baseline_label],
        },
        "errorTags": {
            system: dict(sorted(tags[system].items()))
            for system in (candidate_label, baseline_label)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("verdicts", type=Path)
    parser.add_argument("private_mapping", type=Path)
    parser.add_argument("summary_output", type=Path)
    parser.add_argument("unblinded_output", type=Path)
    parser.add_argument("--candidate-label", default="mimi")
    parser.add_argument("--baseline-label", default="apple")
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-family", required=True)
    parser.add_argument("--judge-revision", required=True)
    args = parser.parse_args()
    if args.summary_output.exists() or args.unblinded_output.exists():
        raise SystemExit("refusing to overwrite comparison outputs")

    suite = index(load_jsonl(args.suite), "id", "suite")
    verdicts = index(load_jsonl(args.verdicts), "case_id", "verdicts")
    mapping = index(load_jsonl(args.private_mapping), "case_id", "mapping")
    if set(suite) != set(verdicts) or set(suite) != set(mapping):
        raise SystemExit("suite, verdicts, and mapping must cover identical cases")

    unblinded = []
    for case_id in sorted(suite):
        case, verdict, identities = suite[case_id], verdicts[case_id], mapping[case_id]
        systems = {
            "A": identities.get("candidate_A_system"),
            "B": identities.get("candidate_B_system"),
        }
        if set(systems.values()) != {args.candidate_label, args.baseline_label}:
            raise SystemExit(f"invalid private mapping: {case_id}")
        for field in WIN_FIELDS:
            if verdict.get(field) not in {"A", "B", "tie"}:
                raise SystemExit(f"invalid {field}: {case_id}")
        for side in ("A", "B"):
            if not isinstance(
                verdict.get(f"critical_error_{side}"), bool
            ) or not isinstance(verdict.get(f"candidate_{side}_error_tags"), list):
                raise SystemExit(f"invalid critical evidence: {case_id}/{side}")
            tags = verdict[f"candidate_{side}_error_tags"]
            if (
                len(tags) != len(set(tags))
                or not all(isinstance(tag, str) for tag in tags)
                or not set(tags).issubset(ALLOWED_ERROR_TAGS)
            ):
                raise SystemExit(f"invalid error tags: {case_id}/{side}")
        justification = str(verdict.get("brief_justification", "")).strip()
        if not justification or len(justification.split()) > 50:
            raise SystemExit(f"invalid brief justification: {case_id}")
        row = {
            "caseID": case_id,
            "sourceLanguage": case["sourceLanguage"],
            "targetLanguage": case["targetLanguage"],
            "direction": f"{case['sourceLanguage']}>{case['targetLanguage']}",
            "domain": case["domain"],
            "sourceUnit": case.get("sourceUnit", "sentence"),
            "segmentCount": int(case.get("segmentCount", 1)),
            "briefJustification": justification,
        }
        for field in WIN_FIELDS:
            value = verdict[field]
            row[f"{field}_system"] = "tie" if value == "tie" else systems[value]
        for side, system in systems.items():
            row[f"{system}_critical_error"] = verdict[f"critical_error_{side}"]
            row[f"{system}_error_tags"] = verdict[f"candidate_{side}_error_tags"]
        unblinded.append(row)

    args.unblinded_output.parent.mkdir(parents=True, exist_ok=True)
    args.unblinded_output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in unblinded
        ),
        encoding="utf-8",
    )
    slices = {
        "directions": lambda row: row["direction"],
        "domains": lambda row: f"{row['direction']}/{row['domain']}",
        "sourceUnits": lambda row: f"{row['direction']}/{row['sourceUnit']}",
    }
    summaries = {}
    for name, key in slices.items():
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in unblinded:
            grouped[key(row)].append(row)
        summaries[name] = {
            group: group_summary(rows, args.candidate_label, args.baseline_label)
            for group, rows in sorted(grouped.items())
        }
    output = {
        "schemaVersion": 1,
        "purpose": "non-claimable blinded development translation comparison",
        "status": "complete-diagnostic",
        "candidateLabel": args.candidate_label,
        "baselineLabel": args.baseline_label,
        "judge": {
            "model": args.judge_model,
            "family": args.judge_family,
            "revision": args.judge_revision,
            "candidateIdentitiesVisible": False,
            "referenceVisible": False,
            "privateReasoningRetained": False,
        },
        "artifacts": {
            "suiteSHA256": sha256(args.suite),
            "verdictsSHA256": sha256(args.verdicts),
            "privateMappingSHA256": sha256(args.private_mapping),
            "unblindedResultsSHA256": sha256(args.unblinded_output),
        },
        "overall": group_summary(unblinded, args.candidate_label, args.baseline_label),
        **summaries,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "cases": len(unblinded),
                "overall": output["overall"],
                "output": str(args.summary_output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
