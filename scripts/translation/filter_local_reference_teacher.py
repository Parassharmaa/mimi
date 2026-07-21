#!/usr/bin/env python3
"""Admit local Qwen targets only when hidden-reference metrics beat the student."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

import sacrebleu


JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]")
STRUCTURAL_TOKEN_RE = re.compile(
    r"https?://[^\s]+|\{[^{}]+\}|%[A-Za-z]|<[A-Za-z][^<>]*>|%"
)
NUMBER_RE = re.compile(
    r"(?<![\d.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)*(?!\d|\.\d)"
)
ENGLISH_MONTHS = {
    name.casefold(): index
    for index, name in enumerate((
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ), start=1)
}
ENGLISH_MONTH_RE = re.compile(
    r"\b(" + "|".join(ENGLISH_MONTHS) + r")\b",
    re.IGNORECASE,
)
JAPANESE_MONTH_RE = re.compile(r"(?<!\d)(0?[1-9]|1[0-2])月")
REPEATED_SPAN_RE = re.compile(r"(?is)(.{8,80}?)(?:\s*\1){3,}")
SACREBLEU_VERSION = "2.6.0"
ALLOWED_LICENSES = {
    "Apache-2.0",
    "CC-BY-2.0-FR",
    "CC-BY-4.0",
    "CC-BY-SA-3.0",
    "CC0-1.0",
    "MIT",
    "project-owned",
}
COMET_SIGNATURE_VALUE = {
    "metric": "COMET-22",
    "modelRepository": "Unbabel/wmt22-comet-da",
    "modelRevision": "371e9839ca4e213dde891b066cf3080f75ec7e72",
    "modelLicense": "Apache-2.0",
    "package": "unbabel-comet",
    "packageVersion": "2.2.7",
    "setuptoolsVersion": "80.9.0",
    "precision": "float32",
    "multipleReferenceAggregation": "mean",
}
COMET_SIGNATURE_SHA256 = hashlib.sha256(
    json.dumps(COMET_SIGNATURE_VALUE, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
TEACHER_MODEL = "mlx-community/Qwen3-8B-4bit"
TEACHER_REVISION = "545dc4251c05440727734bcd94334791f6ab0192"
TEACHER_LICENSE = "Apache-2.0"
PROTECTED_TOKEN_POLICY_VERSION = 2


def rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"missing JSONL input: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing JSON report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def protected_tokens(text: str) -> list[str]:
    value = unicodedata.normalize("NFKC", text)
    structural = [f"s:{token}" for token in STRUCTURAL_TOKEN_RE.findall(value)]
    months = [
        f"m:{ENGLISH_MONTHS[match.group(1).casefold()]}"
        for match in ENGLISH_MONTH_RE.finditer(value)
    ]
    japanese_months = list(JAPANESE_MONTH_RE.finditer(value))
    months.extend(f"m:{int(match.group(1))}" for match in japanese_months)
    month_number_spans = {match.span(1) for match in japanese_months}
    numbers = [
        f"n:{match.group(0).replace(',', '')}"
        for match in NUMBER_RE.finditer(value)
        if match.span() not in month_number_spans
    ]
    return sorted([*structural, *months, *numbers])


def indexed_results(report: dict, label: str) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for row in report.get("results", []):
        identifier = str(row.get("caseID", ""))
        if not identifier or identifier in indexed:
            raise SystemExit(f"{label} has a missing or duplicate case ID")
        indexed[identifier] = row
    if not indexed:
        raise SystemExit(f"{label} has no results")
    return indexed


def valid_translation(source: str, target: str, target_language: str) -> str | None:
    source_norm, target_norm = normalized(source), normalized(target)
    if not target_norm or source_norm == target_norm:
        return "empty-or-source-copy"
    if target_language == "ja-JP" and len(JAPANESE_RE.findall(target)) < 2:
        return "target-script"
    if target_language == "en-US" and len(LATIN_RE.findall(target)) < 2:
        return "target-script"
    if REPEATED_SPAN_RE.search(target):
        return "degenerate-repetition"
    ratio = len(target_norm) / max(1, len(source_norm))
    if ratio < 0.15 or ratio > 5.0:
        return "length-ratio"
    source_tokens = protected_tokens(source)
    target_tokens = protected_tokens(target)
    if source_tokens != target_tokens:
        return "protected-token-mismatch"
    return None


def metric_signature(report: dict) -> tuple:
    return tuple(report.get(field) for field in (
        "metric",
        "modelRepository",
        "modelRevision",
        "modelLicense",
        "package",
        "packageVersion",
        "setuptoolsVersion",
        "precision",
        "multipleReferenceAggregation",
        "signatureSHA256",
    ))


def validate_suite_manifest(suite_path: Path) -> tuple[dict, Path, dict]:
    manifest_path = suite_path.with_suffix(suite_path.suffix + ".manifest.json")
    manifest = load_json(manifest_path)
    if (
        manifest.get("purpose")
        != "reference-hidden local Qwen teacher training suite; never evaluation evidence"
        or manifest.get("promotion_eligible") is not False
        or manifest.get("reference_exposed_to_teacher") is not False
        or set(manifest.get("allowed_licenses", [])) != ALLOWED_LICENSES
        or manifest.get("outputs", {}).get("suite", {}).get("sha256") != sha256(suite_path)
        or not manifest.get("inputs", {}).get("protected_suites")
    ):
        raise SystemExit("suite lacks an authentic preparation/license manifest")
    for item in manifest["inputs"]["protected_suites"]:
        protected_path = Path(str(item.get("path", "")))
        if not protected_path.is_file() or item.get("sha256") != sha256(protected_path):
            raise SystemExit("suite preparation manifest has stale protected-suite evidence")
    baseline = manifest.get("outputs", {}).get("baseline_report", {})
    baseline_path = Path(str(baseline.get("path", "")))
    if not baseline_path.is_file() or baseline.get("sha256") != sha256(baseline_path):
        raise SystemExit("suite preparation manifest has stale student baseline evidence")
    return manifest, manifest_path, load_json(baseline_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("teacher_report", type=Path)
    parser.add_argument("teacher_comet_report", type=Path)
    parser.add_argument("student_comet_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--minimum-teacher-comet", type=float, default=0.85)
    parser.add_argument("--minimum-comet-delta", type=float, default=0.01)
    parser.add_argument("--minimum-teacher-chrf", type=float, default=25.0)
    parser.add_argument("--minimum-chrf-delta", type=float, default=2.0)
    parser.add_argument("--minimum-accepted-per-domain-direction", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() and args.output.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    thresholds = {
        "minimum-teacher-comet": (args.minimum_teacher_comet, 0.85, 1.5),
        "minimum-comet-delta": (args.minimum_comet_delta, 0.01, 2.0),
        "minimum-teacher-chrf": (args.minimum_teacher_chrf, 25.0, 100.0),
        "minimum-chrf-delta": (args.minimum_chrf_delta, 2.0, 100.0),
    }
    for name, (value, lower, upper) in thresholds.items():
        if not math.isfinite(value) or not lower <= value <= upper:
            raise SystemExit(f"{name} must be finite and within [{lower}, {upper}]")
    if args.minimum_accepted_per_domain_direction < 0:
        raise SystemExit("minimum accepted per domain/direction cannot be negative")

    installed_sacrebleu = importlib.metadata.version("sacrebleu")
    if installed_sacrebleu != SACREBLEU_VERSION:
        raise SystemExit(
            f"sacrebleu version mismatch: installed={installed_sacrebleu} required={SACREBLEU_VERSION}"
        )
    suite_manifest, suite_manifest_path, student_report = validate_suite_manifest(args.suite)
    baseline_path = Path(suite_manifest["outputs"]["baseline_report"]["path"])
    suite_rows = rows(args.suite)
    suite = {str(row.get("id", "")): row for row in suite_rows}
    if not suite or len(suite) != len(suite_rows) or "" in suite:
        raise SystemExit("suite has missing or duplicate IDs")
    teacher_report = load_json(args.teacher_report)
    teacher_comet = load_json(args.teacher_comet_report)
    student_comet = load_json(args.student_comet_report)
    if (
        teacher_report.get("claimEligible") is not False
        or teacher_report.get("referenceExposedToTeacher") is not False
        or teacher_report.get("studentHypothesisExposedToTeacher") is not False
        or teacher_report.get("reasoningTraceRequestedOrStored") is not False
    ):
        raise SystemExit("teacher report violates reference/reasoning isolation contract")
    if (
        teacher_report.get("modelRepository"),
        teacher_report.get("modelRevision"),
        teacher_report.get("modelLicense"),
    ) != (TEACHER_MODEL, TEACHER_REVISION, TEACHER_LICENSE):
        raise SystemExit("teacher report does not use the pinned distributable Qwen teacher")
    if teacher_report.get("suite", {}).get("sha256") != sha256(args.suite):
        raise SystemExit("teacher report was not generated from this exact suite")
    if metric_signature(teacher_comet) != metric_signature(student_comet):
        raise SystemExit("teacher and student COMET reports use different pinned metrics")
    expected_metric_signature = (*COMET_SIGNATURE_VALUE.values(), COMET_SIGNATURE_SHA256)
    if metric_signature(teacher_comet) != expected_metric_signature:
        raise SystemExit("COMET reports do not use the exact pinned COMET-22 signature")
    if teacher_comet.get("engineReportSHA256") != sha256(args.teacher_report):
        raise SystemExit("teacher COMET report is not bound to this teacher report")
    if student_comet.get("engineReportSHA256") != sha256(baseline_path):
        raise SystemExit("student COMET report is not bound to the frozen baseline report")
    if teacher_comet.get("engine") != teacher_report.get("engine"):
        raise SystemExit("teacher COMET engine identity mismatch")
    if student_comet.get("engine") != student_report.get("engine"):
        raise SystemExit("student COMET engine identity mismatch")
    for label, report in (("teacher COMET", teacher_comet), ("student COMET", student_comet)):
        if report.get("suiteSHA256") != sha256(args.suite):
            raise SystemExit(f"{label} does not score this exact suite")

    teacher_results = indexed_results(teacher_report, "teacher report")
    teacher_scores = indexed_results(teacher_comet, "teacher COMET report")
    student_scores = indexed_results(student_comet, "student COMET report")
    expected = set(suite)
    if any(set(values) != expected for values in (teacher_results, teacher_scores, student_scores)):
        raise SystemExit("suite, teacher, and metric reports must cover identical IDs")

    accepted: list[dict] = []
    audit: list[dict] = []
    rejected: Counter[str] = Counter()
    chrf_metric = sacrebleu.metrics.CHRF(word_order=2)
    for identifier in sorted(suite):
        seed = suite[identifier]
        result = teacher_results[identifier]
        for field in ("sourceLanguage", "targetLanguage", "domain", "source", "references"):
            if result.get(field) != seed.get(field):
                raise SystemExit(f"teacher report disagrees with suite {field}: {identifier}")
        source = str(seed["source"])
        candidate = str(result.get("hypothesis", "")).strip()
        references = [str(value).strip() for value in seed.get("references", [])]
        if not references or not all(references):
            raise SystemExit(f"suite case has no valid hidden reference: {identifier}")
        student = str(seed.get("studentHypothesis", "")).strip()
        embedded_student_chrf = seed.get("studentChrFPlusPlus")
        if not student or not isinstance(embedded_student_chrf, (int, float)):
            raise SystemExit(f"suite case has no student baseline: {identifier}")
        teacher_chrf = chrf_metric.sentence_score(candidate, references).score
        student_chrf = chrf_metric.sentence_score(student, references).score
        if abs(student_chrf - float(embedded_student_chrf)) > 1e-6:
            raise SystemExit(f"embedded student chrF++ is not reproducible: {identifier}")
        teacher_comet_score = float(teacher_scores[identifier]["score"])
        student_comet_score = float(student_scores[identifier]["score"])
        if not math.isfinite(teacher_comet_score) or not math.isfinite(student_comet_score):
            raise SystemExit(f"COMET report contains a non-finite score: {identifier}")
        chrf_delta = teacher_chrf - student_chrf
        comet_delta = teacher_comet_score - student_comet_score
        reason = valid_translation(source, candidate, str(seed["targetLanguage"]))
        if reason is None and normalized(candidate) == normalized(student):
            reason = "no-new-student-signal"
        if reason is None and teacher_comet_score < args.minimum_teacher_comet:
            reason = "teacher-comet-below-minimum"
        if reason is None and comet_delta < args.minimum_comet_delta:
            reason = "comet-delta-below-minimum"
        if reason is None and teacher_chrf < args.minimum_teacher_chrf:
            reason = "teacher-chrf-below-minimum"
        if reason is None and chrf_delta < args.minimum_chrf_delta:
            reason = "chrf-delta-below-minimum"

        accepted_id = None
        if reason is None:
            candidate_id = hashlib.sha256(
                f"{identifier}\0{normalized(candidate)}".encode()
            ).hexdigest()[:24]
            accepted_id = f"local-qwen-reference:{candidate_id}"
            accepted.append({
                "id": accepted_id,
                "source_id": identifier,
                "source_language": seed["sourceLanguage"],
                "target_language": seed["targetLanguage"],
                "source": source,
                "target": candidate,
                "domain": seed.get("domain", "unknown"),
                "origin": "strict-local-qwen-reference-filtered",
                "source_license": seed.get("sourceLicense"),
                "source_provenance": seed.get("sourceProvenance"),
                "reference_provenance": seed.get("referenceProvenance"),
                "teacher_model": teacher_report.get("modelRepository"),
                "teacher_revision": teacher_report.get("modelRevision"),
                "teacher_license": teacher_report.get("modelLicense"),
                "quality_control": {
                    "teacher_chrf_pp": teacher_chrf,
                    "student_chrf_pp": student_chrf,
                    "chrf_pp_delta": chrf_delta,
                    "teacher_comet_22": teacher_comet_score,
                    "student_comet_22": student_comet_score,
                    "comet_22_delta": comet_delta,
                    "reference_exposed_to_teacher": False,
                    "reasoning_trace_requested_or_stored": False,
                },
                "review_status": "hidden-reference-metric-filtered-provisional",
                "training_only": True,
                "promotion_eligible": False,
            })
        else:
            rejected[reason] += 1
        audit.append({
            "id": identifier,
            "source": source,
            "candidate": candidate,
            "student": student,
            "teacher_chrf_pp": teacher_chrf,
            "student_chrf_pp": student_chrf,
            "chrf_pp_delta": chrf_delta,
            "teacher_comet_22": teacher_comet_score,
            "student_comet_22": student_comet_score,
            "comet_22_delta": comet_delta,
            "accepted_id": accepted_id,
            "rejection_reason": reason,
        })

    accepted_by_cell = Counter(
        f"{row['domain']}:{row['source_language']}>{row['target_language']}"
        for row in accepted
    )
    expected_cells = sorted({
        f"{row['domain']}:{row['sourceLanguage']}>{row['targetLanguage']}"
        for row in suite.values()
    })
    shortfalls = {
        cell: {
            "accepted": accepted_by_cell[cell],
            "required": args.minimum_accepted_per_domain_direction,
        }
        for cell in expected_cells
        if accepted_by_cell[cell] < args.minimum_accepted_per_domain_direction
    }
    if shortfalls:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        failure_path = args.output.with_suffix(args.output.suffix + ".floor-failure.json")
        if failure_path.exists() and failure_path.stat().st_size:
            raise SystemExit(f"refusing to overwrite non-empty floor failure: {failure_path}")
        failure_report = {
            "schema_version": 1,
            "purpose": "rejected hidden-reference teacher round; domain/direction floor failure",
            "promotion_eligible": False,
            "training_rows_emitted": False,
            "policy": {
                "minimum_teacher_comet_22": args.minimum_teacher_comet,
                "minimum_comet_22_delta": args.minimum_comet_delta,
                "minimum_teacher_chrf_pp": args.minimum_teacher_chrf,
                "minimum_chrf_pp_delta": args.minimum_chrf_delta,
                "minimum_accepted_per_domain_direction": (
                    args.minimum_accepted_per_domain_direction
                ),
                "protected_token_policy_version": PROTECTED_TOKEN_POLICY_VERSION,
            },
            "counts": {
                "input": len(suite_rows),
                "potentially_accepted": len(accepted),
                "accepted_by_domain_direction": dict(sorted(accepted_by_cell.items())),
                "rejected": dict(sorted(rejected.items())),
            },
            "shortfalls": shortfalls,
            "inputs": {
                "suite": {"path": str(args.suite.resolve()), "sha256": sha256(args.suite)},
                "suite_manifest": {"path": str(suite_manifest_path.resolve()), "sha256": sha256(suite_manifest_path)},
                "student_baseline": {"path": str(baseline_path.resolve()), "sha256": sha256(baseline_path)},
                "teacher_report": {"path": str(args.teacher_report.resolve()), "sha256": sha256(args.teacher_report)},
                "teacher_comet": {"path": str(args.teacher_comet_report.resolve()), "sha256": sha256(args.teacher_comet_report)},
                "student_comet": {"path": str(args.student_comet_report.resolve()), "sha256": sha256(args.student_comet_report)},
            },
        }
        failure_path.write_text(
            json.dumps(failure_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise SystemExit(
            "accepted teacher rows miss predeclared domain/direction floors: "
            + json.dumps(shortfalls, sort_keys=True)
        )

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
    manifest = {
        "schema_version": 1,
        "purpose": "reviewer-free hidden-reference local teacher filter; never promotion evidence",
        "promotion_eligible": False,
        "policy": {
            "reference_exposed_to_teacher": False,
            "reasoning_trace_requested_or_stored": False,
            "minimum_teacher_comet_22": args.minimum_teacher_comet,
            "minimum_comet_22_delta": args.minimum_comet_delta,
            "minimum_teacher_chrf_pp": args.minimum_teacher_chrf,
            "minimum_chrf_pp_delta": args.minimum_chrf_delta,
            "minimum_accepted_per_domain_direction": (
                args.minimum_accepted_per_domain_direction
            ),
            "protected_token_policy_version": PROTECTED_TOKEN_POLICY_VERSION,
            "deterministic_checks": [
                "non-empty and not source copy",
                "target script",
                "no fourfold repeated 8-80 character span",
                "length ratio 0.15-5.0",
                "NFKC-equivalent URL, placeholder, markup, percent, and atomic-number preservation",
                "English month names and Japanese N-month forms canonicalized before exact number comparison",
            ],
        },
        "counts": {
            "input": len(suite_rows),
            "accepted": len(accepted),
            "rejected": dict(sorted(rejected.items())),
            "accepted_by_direction": dict(sorted(Counter(
                f"{row['source_language']}>{row['target_language']}" for row in accepted
            ).items())),
            "accepted_by_domain_direction": dict(sorted(accepted_by_cell.items())),
        },
        "metric": {
            "comet_signature": list(metric_signature(teacher_comet)),
            "sacrebleu_version": installed_sacrebleu,
            "chrf_pp_signature": str(chrf_metric.get_signature()),
        },
        "teacher": {
            "repository": TEACHER_MODEL,
            "revision": TEACHER_REVISION,
            "license": TEACHER_LICENSE,
        },
        "inputs": {
            "suite": {"path": str(args.suite.resolve()), "sha256": sha256(args.suite)},
            "suite_manifest": {"path": str(suite_manifest_path.resolve()), "sha256": sha256(suite_manifest_path)},
            "student_baseline": {"path": str(baseline_path.resolve()), "sha256": sha256(baseline_path)},
            "teacher_report": {"path": str(args.teacher_report.resolve()), "sha256": sha256(args.teacher_report)},
            "teacher_comet": {"path": str(args.teacher_comet_report.resolve()), "sha256": sha256(args.teacher_comet_report)},
            "student_comet": {"path": str(args.student_comet_report.resolve()), "sha256": sha256(args.student_comet_report)},
        },
        "outputs": {
            "accepted": {"path": str(args.output.resolve()), "sha256": sha256(args.output)},
            "audit": {"path": str(audit_path.resolve()), "sha256": sha256(audit_path)},
        },
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
