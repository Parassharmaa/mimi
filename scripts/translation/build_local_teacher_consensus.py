#!/usr/bin/env python3
"""Filter a larger local teacher through independent model agreement.

This is a reviewer-free research lane, not a substitute for bilingual gold.
Accepted rows are permanently marked promotion-ineligible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]")
TOKEN_RE = re.compile(r"https?://\S+|\{[^{}]+\}|%\w|\b\d[\d,.:/-]*\b")


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"missing suite: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalized(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def ngram_counts(text: str, size: int) -> Counter[str]:
    value = normalized(text)
    if len(value) < size:
        return Counter({value: 1}) if value else Counter()
    return Counter(value[index:index + size] for index in range(len(value) - size + 1))


def agreement(left: str, right: str) -> float:
    """Symmetric mean character 1-4 gram Dice score on a 0-100 scale."""

    scores = []
    for size in range(1, 5):
        a, b = ngram_counts(left, size), ngram_counts(right, size)
        overlap = sum((a & b).values())
        scores.append(200.0 * overlap / max(1, sum(a.values()) + sum(b.values())))
    return sum(scores) / len(scores)


def ngrams(text: str, size: int = 5) -> set[str]:
    value = normalized(text)
    if len(value) < size:
        return {value}
    return {value[index:index + size] for index in range(len(value) - size + 1)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def indexed_results(report: dict, label: str) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for row in report.get("results", []):
        identifier = str(row.get("caseID", ""))
        if not identifier or identifier in output:
            raise SystemExit(f"{label} report has missing or duplicate case ID")
        output[identifier] = row
    if not output:
        raise SystemExit(f"{label} report has no results")
    return output


def valid_translation(source: str, target: str, target_language: str) -> str | None:
    source_norm, target_norm = normalized(source), normalized(target)
    if not target_norm or target_norm == source_norm:
        return "empty-or-copied"
    if target_language == "ja-JP" and len(JAPANESE_RE.findall(target)) < 2:
        return "target-script"
    if target_language == "en-US" and len(LATIN_RE.findall(target)) < 2:
        return "target-script"
    if len(target_norm) > max(24, len(source_norm) * 4) or len(target_norm) * 5 < len(source_norm):
        return "length-ratio"
    if sorted(TOKEN_RE.findall(source)) != sorted(TOKEN_RE.findall(target)):
        return "protected-token-mismatch"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("preferred_report", type=Path)
    parser.add_argument("teacher_report", type=Path)
    parser.add_argument("independent_report", type=Path)
    parser.add_argument("protected_suite", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--additional-protected-suite", type=Path, action="append", default=[])
    parser.add_argument("--minimum-teacher-preferred", type=float, default=45.0)
    parser.add_argument("--minimum-teacher-independent", type=float, default=35.0)
    parser.add_argument("--minimum-preferred-independent", type=float, default=30.0)
    parser.add_argument("--teacher-backtranslation-report", type=Path)
    parser.add_argument("--preferred-backtranslation-report", type=Path)
    parser.add_argument("--independent-backtranslation-report", type=Path)
    parser.add_argument("--minimum-roundtrip-agreement", type=float, default=45.0)
    parser.add_argument("--roundtrip-nli-report", type=Path)
    parser.add_argument("--minimum-mutual-entailment", type=float, default=0.5)
    parser.add_argument("--maximum-contradiction", type=float, default=0.2)
    parser.add_argument("--maximum-jaccard", type=float, default=0.8)
    parser.add_argument(
        "--allow-forward-report-superset",
        action="store_true",
        help="Allow forward reports from a sealed parent suite while filtering a subset.",
    )
    args = parser.parse_args()
    if args.output.exists() and args.output.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")

    suite_rows = rows(args.suite)
    reports = {
        "preferred": load_json(args.preferred_report),
        "teacher": load_json(args.teacher_report),
        "independent": load_json(args.independent_report),
    }
    backtranslation_paths = {
        "teacher": args.teacher_backtranslation_report,
        "preferred": args.preferred_backtranslation_report,
        "independent": args.independent_backtranslation_report,
    }
    provided_backtranslations = {name: path for name, path in backtranslation_paths.items() if path}
    if provided_backtranslations and len(provided_backtranslations) != 3:
        raise SystemExit("round-trip filtering requires all three backtranslation reports")
    backtranslation_reports = {
        name: load_json(path)
        for name, path in provided_backtranslations.items()
    }
    indexed = {name: indexed_results(report, name) for name, report in reports.items()}
    expected = {str(row["id"]) for row in suite_rows}
    for name, result in indexed.items():
        covered = set(result)
        if covered != expected and not (
            args.allow_forward_report_superset and expected < covered
        ):
            raise SystemExit(f"{name} report does not cover the exact suite")
    engines = {name: str(report.get("engine", "")).strip() for name, report in reports.items()}
    if not all(engines.values()) or len(set(engines.values())) != 3:
        raise SystemExit("local consensus requires three distinct model engines")
    backtranslation_results = {
        name: indexed_results(report, f"{name} backtranslation")
        for name, report in backtranslation_reports.items()
    }
    for name, result in backtranslation_results.items():
        if set(result) != expected:
            raise SystemExit(f"{name} backtranslation report does not cover the exact suite")
    backtranslation_engines = {
        name: str(report.get("engine", "")).strip()
        for name, report in backtranslation_reports.items()
    }
    if backtranslation_engines and (
        not all(backtranslation_engines.values())
        or len(set(backtranslation_engines.values())) != 3
    ):
        raise SystemExit("round-trip filtering requires three distinct model engines")
    nli_report = load_json(args.roundtrip_nli_report) if args.roundtrip_nli_report else None
    nli_results = {}
    if nli_report is not None:
        if nli_report.get("claimEligible") is not False:
            raise SystemExit("round-trip NLI report must be claim-ineligible")
        for row in nli_report.get("results", []):
            identifier = str(row.get("caseID", ""))
            if not identifier or identifier in nli_results:
                raise SystemExit("round-trip NLI report has missing or duplicate case ID")
            nli_results[identifier] = row
        if set(nli_results) != expected:
            raise SystemExit("round-trip NLI report does not cover the exact suite")

    protected_suites = [args.protected_suite, *args.additional_protected_suite]
    protected = [
        ngrams(text)
        for path in protected_suites
        for row in rows(path)
        for text in [row.get("source", ""), *row.get("references", [])]
        if str(text).strip()
    ]
    accepted: list[dict] = []
    audit: list[dict] = []
    reject_counts: Counter[str] = Counter()
    for seed in suite_rows:
        identifier = str(seed["id"])
        source = str(seed["source"])
        hypotheses = {
            name: str(indexed[name][identifier].get("hypothesis", "")).strip()
            for name in indexed
        }
        reason = None
        for name, hypothesis in hypotheses.items():
            reason = valid_translation(source, hypothesis, str(seed["targetLanguage"]))
            if reason is not None:
                reason = f"{name}-{reason}"
                break
        scores = {
            "teacher_preferred": agreement(hypotheses["teacher"], hypotheses["preferred"]),
            "teacher_independent": agreement(hypotheses["teacher"], hypotheses["independent"]),
            "preferred_independent": agreement(hypotheses["preferred"], hypotheses["independent"]),
        }
        backtranslations = {
            name: str(result[identifier].get("hypothesis", "")).strip()
            for name, result in backtranslation_results.items()
        }
        roundtrip_scores = {
            name: agreement(source, backtranslation)
            for name, backtranslation in backtranslations.items()
        }
        nli_scores = (
            nli_results[identifier].get("backtranslations", {})
            if nli_results else {}
        )
        if reason is None and normalized(hypotheses["teacher"]) == normalized(hypotheses["preferred"]):
            reason = "no-new-student-signal"
        if reason is None and scores["teacher_preferred"] < args.minimum_teacher_preferred:
            reason = "low-teacher-preferred-agreement"
        if reason is None and scores["teacher_independent"] < args.minimum_teacher_independent:
            reason = "low-teacher-independent-agreement"
        if reason is None and scores["preferred_independent"] < args.minimum_preferred_independent:
            reason = "low-preferred-independent-agreement"
        if reason is None and any(not value for value in backtranslations.values()):
            reason = "empty-backtranslation"
        if reason is None and any(
            score < args.minimum_roundtrip_agreement
            for score in roundtrip_scores.values()
        ):
            reason = "low-roundtrip-agreement"
        if reason is None and nli_scores:
            if set(nli_scores) != set(backtranslations):
                raise SystemExit(f"round-trip NLI engine mismatch: {identifier}")
            for values in nli_scores.values():
                if min(
                    float(values.get("source_entails_backtranslation", -1)),
                    float(values.get("backtranslation_entails_source", -1)),
                ) < args.minimum_mutual_entailment:
                    reason = "low-mutual-entailment"
                    break
                if max(
                    float(values.get("source_contradicts_backtranslation", 1)),
                    float(values.get("backtranslation_contradicts_source", 1)),
                ) > args.maximum_contradiction:
                    reason = "high-roundtrip-contradiction"
                    break
        candidate_grams = [ngrams(source), ngrams(hypotheses["teacher"])]
        if reason is None and any(
            len(candidate & heldout) / max(1, len(candidate | heldout)) > args.maximum_jaccard
            for candidate in candidate_grams
            for heldout in protected
        ):
            reason = "near-protected"

        accepted_row = None
        if reason is None:
            candidate_id = hashlib.sha256(
                f"{identifier}\0{normalized(hypotheses['teacher'])}".encode()
            ).hexdigest()[:24]
            accepted_row = {
                "id": f"local-teacher:{candidate_id}",
                "source_id": identifier,
                "source_language": seed["sourceLanguage"],
                "target_language": seed["targetLanguage"],
                "source": source,
                "target": hypotheses["teacher"],
                "domain": seed.get("domain", "unknown"),
                "origin": "three-model-local-teacher-consensus-provisional",
                "source_license": seed.get("sourceLicense"),
                "source_provenance": seed.get("sourceProvenance"),
                "teacher_model": engines["teacher"],
                "filter_models": [engines["preferred"], engines["independent"]],
                "agreement_scores": scores,
                "roundtrip_hypotheses": backtranslations,
                "roundtrip_agreement_scores": roundtrip_scores,
                "roundtrip_nli_scores": nli_scores,
                "review_status": "three-model-agreement-provisional",
                "promotion_eligible": False,
            }
            accepted.append(accepted_row)
        else:
            reject_counts[reason] += 1
        audit.append({
            "id": identifier,
            "source": source,
            "hypotheses": hypotheses,
            "agreement_scores": scores,
            "roundtrip_hypotheses": backtranslations,
            "roundtrip_agreement_scores": roundtrip_scores,
            "roundtrip_nli_scores": nli_scores,
            "accepted_id": accepted_row["id"] if accepted_row else None,
            "rejection_reason": reason,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in accepted),
        encoding="utf-8",
    )
    audit_path = args.output.with_suffix(args.output.suffix + ".audit.jsonl")
    audit_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in audit),
        encoding="utf-8",
    )
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest = {
        "schema_version": 1,
        "purpose": "reviewer-free local teacher ablation; never promotion evidence",
        "promotion_eligible": False,
        "policy": {
            "candidate": "CAT teacher output only",
            "distinct_engines_required": 3,
            "minimum_agreement": {
                "teacher_preferred": args.minimum_teacher_preferred,
                "teacher_independent": args.minimum_teacher_independent,
                "preferred_independent": args.minimum_preferred_independent,
            },
            "minimum_roundtrip_agreement": (
                args.minimum_roundtrip_agreement if backtranslation_results else None
            ),
            "minimum_mutual_entailment": (
                args.minimum_mutual_entailment if nli_results else None
            ),
            "maximum_roundtrip_contradiction": (
                args.maximum_contradiction if nli_results else None
            ),
            "maximum_protected_five_gram_jaccard": args.maximum_jaccard,
            "forward_report_superset_allowed": args.allow_forward_report_superset,
            "deterministic_checks": [
                "target-script", "length-ratio", "number-url-placeholder-preservation",
                "not-source-copy", "not-identical-to-student", "protected-suite-overlap",
            ],
        },
        "engines": engines,
        "backtranslation_engines": backtranslation_engines,
        "inputs": {
            "suite": {"path": str(args.suite.resolve()), "sha256": sha256(args.suite)},
            "preferred_report": {"path": str(args.preferred_report.resolve()), "sha256": sha256(args.preferred_report)},
            "teacher_report": {"path": str(args.teacher_report.resolve()), "sha256": sha256(args.teacher_report)},
            "independent_report": {"path": str(args.independent_report.resolve()), "sha256": sha256(args.independent_report)},
            "backtranslation_reports": {
                name: {"path": str(path.resolve()), "sha256": sha256(path)}
                for name, path in provided_backtranslations.items()
            },
            "roundtrip_nli_report": (
                {
                    "path": str(args.roundtrip_nli_report.resolve()),
                    "sha256": sha256(args.roundtrip_nli_report),
                    "model": nli_report.get("model"),
                    "model_revision": nli_report.get("modelRevision"),
                    "model_license": nli_report.get("modelLicense"),
                }
                if nli_report is not None else None
            ),
            "protected_suites": [
                {"path": str(path.resolve()), "sha256": sha256(path)}
                for path in protected_suites
            ],
        },
        "counts": {
            "source_rows": len(suite_rows),
            "accepted": len(accepted),
            "rejected": dict(sorted(reject_counts.items())),
        },
        "outputs": {
            "accepted": {"path": str(args.output.resolve()), "sha256": sha256(args.output)},
            "audit": {"path": str(audit_path.resolve()), "sha256": sha256(audit_path)},
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
