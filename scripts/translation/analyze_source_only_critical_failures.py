#!/usr/bin/env python3
"""Classify source-only critical-token failures without weakening runtime policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from evaluate_typed_critical_token_policy import strict_tokens
from typed_critical_token_policy import (
    narrow_temporal_preserves,
    single_percentage_preserves,
    typed_preserves,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def direction(row: dict) -> str:
    return "en-ja" if row["sourceLanguage"] == "en-US" else "ja-en"


def keyed(counter: Counter[tuple[str, ...]]) -> dict[str, int]:
    return {"|".join(key): value for key, value in sorted(counter.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = report.get("results")
    if not isinstance(rows, list) or not rows:
        raise SystemExit("translation report has no results")

    reported = int(report.get("summary", {}).get("failureCounts", {}).get("critical-token-mismatch", 0))
    relation_counts: Counter[str] = Counter()
    direction_relation_counts: Counter[tuple[str, ...]] = Counter()
    domain_relation_counts: Counter[tuple[str, ...]] = Counter()
    template_relation_counts: Counter[tuple[str, ...]] = Counter()
    failures: list[dict] = []
    for row in rows:
        source = str(row["source"])
        hypothesis = str(row["hypothesis"])
        source_tokens = strict_tokens(source)
        hypothesis_tokens = strict_tokens(hypothesis)
        if source_tokens == hypothesis_tokens:
            continue
        current_direction = direction(row)
        source_language = str(row["sourceLanguage"])
        target_language = str(row["targetLanguage"])
        if narrow_temporal_preserves(
            source,
            hypothesis,
            source_language,
            target_language,
        ):
            relation = "narrow-temporal-candidate"
        elif single_percentage_preserves(source, hypothesis):
            relation = "single-percentage-candidate"
        elif typed_preserves(
            source,
            hypothesis,
            source_language,
            target_language,
        ):
            relation = "broad-typed-candidate-rejected-policy"
        else:
            relation = "unresolved-strict-mismatch"
        domain = str(row.get("domain") or "unknown")
        template = str(row.get("sourceTemplateID") or "unknown")
        relation_counts[relation] += 1
        direction_relation_counts[(current_direction, relation)] += 1
        domain_relation_counts[(domain, relation)] += 1
        template_relation_counts[(template, relation)] += 1
        failures.append(
            {
                "caseID": row["caseID"],
                "direction": current_direction,
                "domain": domain,
                "sourceTemplateID": template,
                "selectedEngine": row.get("selectedEngine"),
                "relation": relation,
                "source": source,
                "hypothesis": hypothesis,
                "sourceTokens": source_tokens,
                "hypothesisTokens": hypothesis_tokens,
            }
        )

    if reported and reported != len(failures):
        raise SystemExit(
            f"report/runtime mismatch: report says {reported}, strict audit found {len(failures)}"
        )
    failures.sort(key=lambda row: row["caseID"])
    payload = {
        "schemaVersion": 1,
        "status": "diagnostic-only",
        "purpose": "source-only critical-token taxonomy; no reference-based quality claim",
        "report": {"path": str(args.report), "sha256": sha256(args.report)},
        "cases": len(rows),
        "strictFailures": len(failures),
        "counts": {
            "relation": dict(sorted(relation_counts.items())),
            "directionRelation": keyed(direction_relation_counts),
            "domainRelation": keyed(domain_relation_counts),
            "templateRelation": keyed(template_relation_counts),
        },
        "policy": {
            "runtimePolicyChanged": False,
            "narrowTemporal": (
                "valid ISO Gregorian dates and 24-hour times only; exact residual "
                "digits, literal percent signs, URLs, placeholders, printf tokens, and markup"
            ),
            "promotionGate": (
                "candidate relaxations remain disabled until the sealed references and "
                "independent automated judges establish zero unsafe acceptances"
            ),
        },
        "claimEligible": False,
        "doesNotAuthorizeAppIntegration": True,
        "results": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "cases": payload["cases"],
                "strictFailures": payload["strictFailures"],
                "relation": payload["counts"]["relation"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
