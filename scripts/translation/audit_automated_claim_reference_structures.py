#!/usr/bin/env python3
"""Audit frozen references with exact rules plus hash-bound judge consensus."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

from assemble_automated_claim_reference_suite import accepted_assessments, load, report_results
from collect_automated_claim_reference_candidates import index, rows, sha256, text_sha256
from typed_critical_token_policy import typed_preserves


URL_RE = re.compile(r"https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+")
PLACEHOLDER_RE = re.compile(r"\{[^{}]+\}|%[A-Za-z]")
MARKUP_RE = re.compile(r"<[A-Za-z][^<>]*>")
OPAQUE_ID_RE = re.compile(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9]*-\d+(?![A-Za-z0-9])")
JA_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
EN_RE = re.compile(r"[A-Za-z]")
CODE_SWITCH_TERMS = (
    "日本語",
    "設定",
    "処理中",
    "Mimi-字幕",
    "Language",
    "English",
    "Auto Detect",
    "Microphone",
    "Settings",
    "Processing",
    "Mimi-Captions",
)
CHECKS = (
    "numbers",
    "entities",
    "negation",
    "placeholders",
    "urls",
    "markup",
    "codeSwitching",
    "omission",
)


def exact_tokens(pattern: re.Pattern[str], value: str) -> list[str]:
    return sorted(pattern.findall(unicodedata.normalize("NFKC", value)))


def consensus_by_case(report: dict, label: str, expected: set[str]) -> dict[str, dict[str, dict]]:
    return {
        case_id: accepted_assessments(result, f"{label}/{case_id}")
        for case_id, result in report_results(report, label, expected).items()
    }


def target_script_present(value: str, language: str) -> bool:
    return bool(JA_RE.search(value)) if language == "ja-JP" else bool(EN_RE.search(value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("judge_report_a", type=Path)
    parser.add_argument("judge_report_b", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists() and args.output.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")

    suite_rows = rows(args.suite)
    suite = index(suite_rows, "id", "frozen reference suite")
    expected = set(suite)
    judge_a = load(args.judge_report_a)
    judge_b = load(args.judge_report_b)
    a = consensus_by_case(judge_a, "judge A", expected)
    b = consensus_by_case(judge_b, "judge B", expected)
    if (
        judge_a.get("judgeModel") == judge_b.get("judgeModel")
        or judge_a.get("judgeModelFamily") == judge_b.get("judgeModelFamily")
        or judge_a.get("judgeRole") != "reference-judge-a"
        or judge_b.get("judgeRole") != "reference-judge-b"
        or judge_a.get("reasoningTracesStored") is not False
        or judge_b.get("reasoningTracesStored") is not False
        or judge_a.get("store") is not False
        or judge_b.get("store") is not False
        or judge_a.get("sourceSuiteSHA256") != judge_b.get("sourceSuiteSHA256")
        or judge_a.get("generatorReportSHA256") != judge_b.get("generatorReportSHA256")
    ):
        raise SystemExit("judge reports are not independent, equally bound, or trace-free")

    results: list[dict] = []
    failed_cases: list[str] = []
    for case_id, row in suite.items():
        source = str(row["source"])
        source_language = str(row["sourceLanguage"])
        target_language = str(row["targetLanguage"])
        references = [str(value) for value in row.get("references", [])]
        accepted_ids = [str(value) for value in row.get("acceptedReferenceCandidateIDs", [])]
        if len(references) != 2 or len(accepted_ids) != 2:
            raise SystemExit(f"suite does not have exactly two bound references: {case_id}")
        check_values = {name: True for name in CHECKS}
        error_tags: set[str] = set()
        reference_details: list[dict] = []
        for candidate_id, reference in zip(accepted_ids, references):
            reference_hash = text_sha256(reference)
            if candidate_id not in a[case_id] or candidate_id not in b[case_id]:
                raise SystemExit(f"accepted reference is absent from a judge report: {case_id}/{candidate_id}")
            assessments = (a[case_id][candidate_id], b[case_id][candidate_id])
            unanimous = all(
                assessment.get("referenceSHA256") == reference_hash
                and assessment.get("acceptAsReference") is True
                and assessment.get("adequacy") == 4
                and assessment.get("fluency") == 4
                and assessment.get("terminology") == 4
                and assessment.get("criticalError") is False
                and assessment.get("protectedTokensPreserved") is True
                and assessment.get("errorTags") == []
                for assessment in assessments
            )
            exact = {
                "numbers": typed_preserves(source, reference, source_language, target_language),
                "entities": exact_tokens(OPAQUE_ID_RE, source) == exact_tokens(OPAQUE_ID_RE, reference),
                "placeholders": exact_tokens(PLACEHOLDER_RE, source)
                == exact_tokens(PLACEHOLDER_RE, reference),
                "urls": exact_tokens(URL_RE, source) == exact_tokens(URL_RE, reference),
                "markup": exact_tokens(MARKUP_RE, source) == exact_tokens(MARKUP_RE, reference),
                "codeSwitching": all(term in reference for term in CODE_SWITCH_TERMS if term in source),
            }
            source_length = len("".join(unicodedata.normalize("NFKC", source).split()))
            target_length = len("".join(unicodedata.normalize("NFKC", reference).split()))
            ratio = target_length / max(1, source_length)
            exact["omission"] = bool(reference.strip()) and target_script_present(
                reference, target_language
            ) and 0.12 <= ratio <= 8.0
            # These semantic categories cannot be proved by regex across languages.
            # They are true only when both hash-bound blinded judges unanimously
            # assigned perfect scores and no error tags.
            exact["negation"] = unanimous
            exact["entities"] = exact["entities"] and unanimous
            exact["omission"] = exact["omission"] and unanimous
            for name, passed in exact.items():
                check_values[name] = check_values[name] and passed
                if not passed:
                    error_tags.add(name)
            reference_details.append(
                {
                    "candidateID": candidate_id,
                    "referenceSHA256": reference_hash,
                    "checks": exact,
                    "lengthRatio": ratio,
                    "unanimousPerfectJudgeConsensus": unanimous,
                }
            )
        failed = [name for name in CHECKS if not check_values[name]]
        if failed:
            failed_cases.append(case_id)
        results.append(
            {
                "caseID": case_id,
                "sourceSHA256": text_sha256(source),
                "referenceSHA256s": [text_sha256(value) for value in references],
                "criticalError": bool(failed),
                "errorTags": sorted(error_tags),
                "checks": check_values,
                "referenceDetails": reference_details,
            }
        )

    report = {
        "schemaVersion": 1,
        "status": "passed" if not failed_cases else "failed",
        "purpose": "deterministic exact-token audit with hash-bound blinded semantic consensus",
        "suiteSHA256": sha256(args.suite),
        "judgeReportASHA256": sha256(args.judge_report_a),
        "judgeReportBSHA256": sha256(args.judge_report_b),
        "methods": {
            "numbers": "bilingual typed numeric signature",
            "entities": "exact opaque-ID multiset plus unanimous perfect judge consensus",
            "negation": "unanimous perfect judge consensus; no bilingual regex claim",
            "placeholders": "exact NFKC placeholder multiset",
            "urls": "exact NFKC URL multiset",
            "markup": "exact NFKC markup multiset",
            "codeSwitching": "exact suite-preregistered code-switched term preservation",
            "omission": "target script and conservative length ratio plus unanimous perfect judge consensus",
        },
        "cases": len(results),
        "failedCases": failed_cases,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if failed_cases:
        raise SystemExit(f"deterministic reference audit failed for {len(failed_cases)} cases")
    print(
        json.dumps(
            {
                "cases": len(results),
                "status": report["status"],
                "report": str(args.output),
                "reportSHA256": sha256(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
