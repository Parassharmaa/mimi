#!/usr/bin/env python3
"""Conservative deterministic structure audit for a translation report."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


STRUCTURAL_RE = re.compile(r"https?://[^\s]+|\{[^{}]+\}|%[A-Za-z]|<[A-Za-z][^<>]*>|%")
NUMBER_RE = re.compile(
    r"(?<![\d.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)*(?!\d|\.\d)"
)
CRITICAL_TOKEN_RE = re.compile(
    r"https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+"
    r"|\{[^{}]+\}|%[A-Za-z]|<[A-Za-z][^<>]*>|%"
    r"|(?<![\d.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)*(?!\d|\.\d)"
)
EN_NEGATION_RE = re.compile(
    r"\b(?:no|not|never|neither|nor|without|cannot|can't|isn't|aren't|won't|don't|doesn't|didn't)\b",
    re.IGNORECASE,
)
JA_NEGATION_RE = re.compile(r"ない|ません|なかった|ませんでした|ぬ|ず|禁止|不可")


def tokens(value: str) -> dict[str, list[str] | bool]:
    normalized = unicodedata.normalize("NFKC", value)
    return {
        "structural": sorted(STRUCTURAL_RE.findall(normalized)),
        "numbers": sorted(match.group(0).replace(",", "") for match in NUMBER_RE.finditer(normalized)),
        "negative": bool(EN_NEGATION_RE.search(normalized) or JA_NEGATION_RE.search(normalized)),
    }


def critical_tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", value)
    return sorted(token.replace(",", "") for token in CRITICAL_TOKEN_RE.findall(normalized))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    failures = []
    counts: Counter[str] = Counter()
    critical_failures = []
    critical_direction_counts: Counter[str] = Counter()
    critical_engine_counts: Counter[str] = Counter()
    for row in report.get("results", []):
        source = tokens(str(row["source"]))
        hypothesis = tokens(str(row["hypothesis"]))
        reasons = []
        if source["structural"] != hypothesis["structural"]:
            reasons.append("url-placeholder-markup")
        if source["numbers"] != hypothesis["numbers"]:
            reasons.append("atomic-number")
        if source["negative"] != hypothesis["negative"]:
            reasons.append("negation-marker")
        if reasons:
            counts.update(reasons)
            failures.append(
                {
                    "caseID": row["caseID"],
                    "sourceLanguage": row["sourceLanguage"],
                    "targetLanguage": row["targetLanguage"],
                    "selectedEngine": row.get("selectedEngine"),
                    "reasons": reasons,
                    "source": row["source"],
                    "hypothesis": row["hypothesis"],
                    "sourceTokens": source,
                    "hypothesisTokens": hypothesis,
                }
            )
        source_critical = critical_tokens(str(row["source"]))
        hypothesis_critical = critical_tokens(str(row["hypothesis"]))
        if source_critical != hypothesis_critical:
            direction = (
                "en-ja" if row["sourceLanguage"] == "en-US" else "ja-en"
            )
            selected_engine = str(row.get("selectedEngine") or "generalist")
            critical_direction_counts[direction] += 1
            critical_engine_counts[f"{direction}:{selected_engine}"] += 1
            critical_failures.append(
                {
                    "caseID": row["caseID"],
                    "direction": direction,
                    "selectedEngine": selected_engine,
                    "sourceTokens": source_critical,
                    "hypothesisTokens": hypothesis_critical,
                }
            )
    payload = {
        "schemaVersion": 1,
        "purpose": "conservative deterministic development audit; not semantic review",
        "report": str(args.report),
        "cases": len(report.get("results", [])),
        "flaggedCases": len(failures),
        "reasonCounts": dict(sorted(counts.items())),
        "promotionEligible": False,
        "exactCriticalTokenAudit": {
            "contract": (
                "exact NFKC multiset of ASCII URLs, placeholders, markup, "
                "percent, and digits"
            ),
            "flaggedCases": len(critical_failures),
            "byDirection": dict(sorted(critical_direction_counts.items())),
            "byDirectionAndSelectedEngine": dict(sorted(critical_engine_counts.items())),
            "failures": critical_failures,
        },
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                **{
                    key: payload[key]
                    for key in ("cases", "flaggedCases", "reasonCounts")
                },
                "exactCriticalTokenAudit": {
                    key: payload["exactCriticalTokenAudit"][key]
                    for key in (
                        "flaggedCases",
                        "byDirection",
                        "byDirectionAndSelectedEngine",
                    )
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
