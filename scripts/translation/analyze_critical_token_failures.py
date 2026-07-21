#!/usr/bin/env python3
"""Classify strict critical-token failures without changing the runtime policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


CRITICAL_TOKEN_RE = re.compile(
    r"https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+"
    r"|\{[^{}]+\}|%[A-Za-z]|<[A-Za-z][^<>]*>|%"
    r"|(?<![\d.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)*(?!\d|\.\d)"
)
NUMBER_RE = re.compile(r"\d+(?:\.\d+)*")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def critical_tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", value)
    return sorted(
        token.replace(",", "") for token in CRITICAL_TOKEN_RE.findall(normalized)
    )


def numeric_tokens(tokens: list[str]) -> list[str]:
    return [token for token in tokens if NUMBER_RE.fullmatch(token)]


def numeric_relation(source: list[str], hypothesis: list[str]) -> str:
    source_numbers = numeric_tokens(source)
    hypothesis_numbers = numeric_tokens(hypothesis)
    if not source_numbers and hypothesis_numbers:
        return "output-introduces-digits"
    if source_numbers and not hypothesis_numbers:
        return "output-drops-digits"
    if source_numbers != hypothesis_numbers:
        if set(source_numbers) == set(hypothesis_numbers):
            return "digit-multiplicity-change"
        return "digit-substitution-or-scale-change"
    return "non-numeric-structural-change"


def reference_alignment(
    source: list[str], hypothesis: list[str], reference: list[str]
) -> str:
    if hypothesis == reference and source != reference:
        return "hypothesis-matches-reference"
    if source == reference and hypothesis != reference:
        return "source-and-reference-agree"
    return "neither-matches-reference"


def corpus(case_id: str) -> str:
    fields = case_id.split(":")
    return fields[1] if len(fields) >= 3 else "unknown"


def counter_payload(counter: Counter[tuple[str, ...]]) -> dict[str, int]:
    return {
        ":".join(key): value
        for key, value in sorted(counter.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = report.get("results")
    if not isinstance(rows, list) or not rows:
        raise SystemExit("translation report has no results")

    alignment_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    direction_counts: Counter[str] = Counter()
    engine_counts: Counter[tuple[str, ...]] = Counter()
    domain_counts: Counter[tuple[str, ...]] = Counter()
    corpus_counts: Counter[tuple[str, ...]] = Counter()
    failures = []
    for row in rows:
        references = row.get("references")
        if not isinstance(references, list) or len(references) != 1:
            raise SystemExit(f"case must have exactly one reference: {row.get('caseID')}")
        source_tokens = critical_tokens(str(row["source"]))
        hypothesis_tokens = critical_tokens(str(row["hypothesis"]))
        if source_tokens == hypothesis_tokens:
            continue
        reference_tokens = critical_tokens(str(references[0]))
        alignment = reference_alignment(
            source_tokens,
            hypothesis_tokens,
            reference_tokens,
        )
        relation = numeric_relation(source_tokens, hypothesis_tokens)
        direction = "en-ja" if row["sourceLanguage"] == "en-US" else "ja-en"
        selected_engine = str(row.get("selectedEngine") or "generalist")
        current_corpus = corpus(str(row["caseID"]))
        domain = str(row["domain"])
        alignment_counts[alignment] += 1
        relation_counts[relation] += 1
        direction_counts[direction] += 1
        engine_counts[(direction, selected_engine, alignment)] += 1
        domain_counts[(domain, alignment)] += 1
        corpus_counts[(current_corpus, alignment)] += 1
        failures.append(
            {
                "caseID": row["caseID"],
                "direction": direction,
                "corpus": current_corpus,
                "domain": domain,
                "selectedEngine": selected_engine,
                "numericRelation": relation,
                "referenceAlignment": alignment,
                "source": row["source"],
                "hypothesis": row["hypothesis"],
                "reference": references[0],
                "sourceTokens": source_tokens,
                "hypothesisTokens": hypothesis_tokens,
                "referenceTokens": reference_tokens,
            }
        )
    failures.sort(key=lambda row: row["caseID"])
    payload = {
        "schemaVersion": 1,
        "purpose": (
            "reference-assisted taxonomy of strict runtime critical-token failures; "
            "diagnostic only"
        ),
        "report": {"path": str(args.report), "sha256": sha256(args.report)},
        "contract": (
            "exact NFKC multiset of ASCII URLs, placeholders, markup, percent, "
            "and digits"
        ),
        "cases": len(rows),
        "failures": len(failures),
        "counts": {
            "referenceAlignment": dict(sorted(alignment_counts.items())),
            "numericRelation": dict(sorted(relation_counts.items())),
            "direction": dict(sorted(direction_counts.items())),
            "directionEngineAlignment": counter_payload(engine_counts),
            "domainAlignment": counter_payload(domain_counts),
            "corpusAlignment": counter_payload(corpus_counts),
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
                "failures": payload["failures"],
                "referenceAlignment": payload["counts"]["referenceAlignment"],
                "numericRelation": payload["counts"]["numericRelation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
