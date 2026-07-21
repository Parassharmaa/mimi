#!/usr/bin/env python3
"""Evaluate a typed critical-token policy against public human references."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from typed_critical_token_policy import (
    narrow_temporal_preserves,
    single_percentage_preserves,
    typed_preserves,
    typed_signature,
)


STRICT_RE = re.compile(
    r"https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+"
    r"|\{[^{}]+\}|%[A-Za-z]|<[A-Za-z][^<>]*>|%"
    r"|(?<![\d.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)*(?!\d|\.\d)"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", value)
    return sorted(token.replace(",", "") for token in STRICT_RE.findall(normalized))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = report.get("results")
    if not isinstance(rows, list) or not rows:
        raise SystemExit("translation report has no results")

    counts: Counter[str] = Counter(
        {
            "strictFailures": 0,
            "typedAccepted": 0,
            "typedRejected": 0,
            "referenceValidatedAccepted": 0,
            "unsafeAccepted": 0,
        }
    )
    percent_counts: Counter[str] = Counter(
        {"accepted": 0, "referenceValidated": 0, "unsafeAccepted": 0}
    )
    temporal_counts: Counter[str] = Counter(
        {"accepted": 0, "referenceValidated": 0, "unsafeAccepted": 0}
    )
    accepted = []
    for row in rows:
        references = row.get("references")
        if not isinstance(references, list) or len(references) != 1:
            raise SystemExit(f"case must have exactly one reference: {row.get('caseID')}")
        source = str(row["source"])
        hypothesis = str(row["hypothesis"])
        reference = str(references[0])
        if strict_tokens(source) == strict_tokens(hypothesis):
            continue
        counts["strictFailures"] += 1
        if single_percentage_preserves(source, hypothesis):
            percent_counts["accepted"] += 1
            if single_percentage_preserves(source, reference):
                percent_counts["referenceValidated"] += 1
            else:
                percent_counts["unsafeAccepted"] += 1
        if narrow_temporal_preserves(
            source,
            hypothesis,
            str(row["sourceLanguage"]),
            str(row["targetLanguage"]),
        ):
            temporal_counts["accepted"] += 1
            if narrow_temporal_preserves(
                source,
                reference,
                str(row["sourceLanguage"]),
                str(row["targetLanguage"]),
            ):
                temporal_counts["referenceValidated"] += 1
            else:
                temporal_counts["unsafeAccepted"] += 1
        typed_passes = typed_preserves(
            source,
            hypothesis,
            str(row["sourceLanguage"]),
            str(row["targetLanguage"]),
        )
        if not typed_passes:
            counts["typedRejected"] += 1
            continue
        counts["typedAccepted"] += 1
        source_signature = typed_signature(source, str(row["sourceLanguage"]))
        reference_signature = typed_signature(reference, str(row["targetLanguage"]))
        hypothesis_signature = typed_signature(hypothesis, str(row["targetLanguage"]))
        reference_safe = source_signature == reference_signature == hypothesis_signature
        counts["referenceValidatedAccepted" if reference_safe else "unsafeAccepted"] += 1
        accepted.append(
            {
                "caseID": row["caseID"],
                "direction": (
                    "en-ja" if row["sourceLanguage"] == "en-US" else "ja-en"
                ),
                "domain": row["domain"],
                "selectedEngine": row.get("selectedEngine"),
                "referenceValidated": reference_safe,
                "source": source,
                "hypothesis": hypothesis,
                "reference": reference,
                "typedSignature": {
                    "protected": source_signature.protected,
                    "percentages": source_signature.percentages,
                    "numbers": source_signature.numbers,
                    "opaqueNumbers": source_signature.opaque_numbers,
                },
            }
        )
    accepted.sort(key=lambda row: row["caseID"])
    unsafe = counts["unsafeAccepted"]
    payload = {
        "schemaVersion": 1,
        "status": "passed" if unsafe == 0 and counts["typedAccepted"] > 0 else "rejected",
        "purpose": "offline typed critical-token ablation; never runtime authorization",
        "report": {"path": str(args.report), "sha256": sha256(args.report)},
        "policy": {
            "implementation": {
                "path": str(Path(__file__).with_name("typed_critical_token_policy.py")),
                "sha256": sha256(
                    Path(__file__).with_name("typed_critical_token_policy.py")
                ),
            },
            "protected": "exact NFKC URLs, placeholders, printf tokens, and markup",
            "percent": "bilingual percent word/symbol count",
            "numbers": (
                "decimal-equivalent English number words, Japanese numerals, "
                "large units, and Japanese era years"
            ),
            "acceptanceGate": (
                "zero typed-accepted public cases whose human-reference typed "
                "signature differs"
            ),
        },
        "policyArms": {
            "narrowIsoDateAnd24HourTime": {
                "status": (
                    "passed"
                    if temporal_counts["referenceValidated"] >= 600
                    and temporal_counts["unsafeAccepted"] == 0
                    else (
                        "insufficient-evidence"
                        if temporal_counts["accepted"] > 0
                        and temporal_counts["unsafeAccepted"] == 0
                        else "rejected"
                    )
                ),
                "counts": dict(sorted(temporal_counts.items())),
                "scope": (
                    "at most one valid Gregorian date and one 24-hour time; "
                    "ISO/Japanese/full-English-month date surfaces; exact residual digits, "
                    "literal percent signs, URLs, placeholders, printf tokens, and markup"
                ),
                "promotionEvidenceFloor": (
                    "zero unsafe accepts plus at least 300 held-out reference-validated "
                    "cases per direction; per-format and adversarial gates remain separate"
                ),
            },
            "singleExplicitDigitPercentage": {
                "status": (
                    "passed"
                    if percent_counts["referenceValidated"] >= 600
                    and percent_counts["unsafeAccepted"] == 0
                    else (
                        "insufficient-evidence"
                        if percent_counts["accepted"] > 0
                        and percent_counts["unsafeAccepted"] == 0
                        else "rejected"
                    )
                ),
                "counts": dict(sorted(percent_counts.items())),
                "scope": (
                    "exactly one explicit digit percentage; exact percentage value, "
                    "other numbers, URLs, placeholders, printf tokens, and markup"
                ),
            }
        },
        "cases": len(rows),
        "counts": dict(sorted(counts.items())),
        "claimEligible": False,
        "doesNotAuthorizeAppIntegration": True,
        "results": accepted,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], **payload["counts"]}, indent=2))


if __name__ == "__main__":
    main()
