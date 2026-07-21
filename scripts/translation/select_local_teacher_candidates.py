#!/usr/bin/env python3
"""Select strict training-only local-teacher candidates without lowering gates."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import sacrebleu

from filter_local_reference_teacher import (
    TEACHER_LICENSE,
    TEACHER_MODEL,
    TEACHER_REVISION,
    metric_signature,
    normalized,
    valid_translation,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing JSON input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"missing JSONL input: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def index_results(report: dict, label: str) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for row in report.get("results", []):
        identifier = str(row.get("caseID", ""))
        if not identifier or identifier in indexed:
            raise SystemExit(f"{label} has missing or duplicate case IDs")
        indexed[identifier] = row
    if not indexed:
        raise SystemExit(f"{label} has no case results")
    return indexed


def validate_teacher(
    suite_path: Path,
    suite: dict[str, dict],
    teacher_path: Path,
    metric_path: Path,
    student_metric: dict,
) -> tuple[dict, dict, dict[str, dict], dict[str, dict]]:
    teacher = load_json(teacher_path)
    metric = load_json(metric_path)
    if (
        teacher.get("claimEligible") is not False
        or teacher.get("referenceExposedToTeacher") is not False
        or teacher.get("studentHypothesisExposedToTeacher") is not False
        or teacher.get("reasoningTraceRequestedOrStored") is not False
        or teacher.get("modelRepository") != TEACHER_MODEL
        or teacher.get("modelRevision") != TEACHER_REVISION
        or teacher.get("modelLicense") != TEACHER_LICENSE
        or teacher.get("suite", {}).get("sha256") != sha256(suite_path)
    ):
        raise SystemExit(f"candidate violates hidden-reference teacher contract: {teacher_path}")
    if (
        metric.get("engine") != teacher.get("engine")
        or metric.get("engineReportSHA256") != sha256(teacher_path)
        or metric.get("suiteSHA256") != sha256(suite_path)
        or metric_signature(metric) != metric_signature(student_metric)
    ):
        raise SystemExit(f"candidate metric is not bound to the teacher and suite: {metric_path}")
    teacher_rows = index_results(teacher, str(teacher_path))
    metric_rows = index_results(metric, str(metric_path))
    if set(teacher_rows) != set(suite) or set(metric_rows) != set(suite):
        raise SystemExit("every candidate must cover the identical full suite")
    for identifier, seed in suite.items():
        result = teacher_rows[identifier]
        for field in ("sourceLanguage", "targetLanguage", "domain", "source", "references"):
            if result.get(field) != seed.get(field):
                raise SystemExit(f"candidate disagrees with suite {field}: {identifier}")
        if not isinstance(metric_rows[identifier].get("score"), (int, float)):
            raise SystemExit(f"candidate metric lacks a numeric score: {identifier}")
    return teacher, metric, teacher_rows, metric_rows


def candidate_passes(
    seed: dict,
    hypothesis: str,
    teacher_chrf: float,
    student_chrf: float,
    teacher_comet: float,
    student_comet: float,
    *,
    minimum_teacher_comet: float,
    minimum_comet_delta: float,
    minimum_teacher_chrf: float,
    minimum_chrf_delta: float,
) -> bool:
    reason = valid_translation(
        str(seed["source"]),
        hypothesis,
        str(seed["targetLanguage"]),
    )
    if reason is None and normalized(hypothesis) == normalized(str(seed["studentHypothesis"])):
        reason = "no-new-student-signal"
    if reason is None and teacher_comet < minimum_teacher_comet:
        reason = "teacher-comet-below-minimum"
    if reason is None and teacher_comet - student_comet < minimum_comet_delta:
        reason = "comet-delta-below-minimum"
    if reason is None and teacher_chrf < minimum_teacher_chrf:
        reason = "teacher-chrf-below-minimum"
    if reason is None and teacher_chrf - student_chrf < minimum_chrf_delta:
        reason = "chrf-delta-below-minimum"
    return reason is None


def aggregate_metric(metric: dict, suite_order: list[dict], results: list[dict]) -> dict:
    report = copy.deepcopy(metric)
    by_direction: dict[str, list[float]] = defaultdict(list)
    by_domain: dict[str, list[float]] = defaultdict(list)
    for seed, result in zip(suite_order, results, strict=True):
        score = float(result["score"])
        direction = f"{seed['sourceLanguage']}>{seed['targetLanguage']}"
        by_direction[direction].append(score)
        by_domain[f"{direction}/{seed['domain']}"].append(score)
    report["directions"] = {
        key: {"cases": len(values), "meanScore": sum(values) / len(values)}
        for key, values in sorted(by_direction.items())
    }
    report["domains"] = {
        key: {"cases": len(values), "meanScore": sum(values) / len(values)}
        for key, values in sorted(by_domain.items())
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("student_comet", type=Path)
    parser.add_argument("output_teacher", type=Path)
    parser.add_argument("output_comet", type=Path)
    parser.add_argument(
        "--candidate",
        nargs=2,
        action="append",
        metavar=("TEACHER_REPORT", "COMET_REPORT"),
        required=True,
    )
    parser.add_argument("--minimum-teacher-comet", type=float, default=0.85)
    parser.add_argument("--minimum-comet-delta", type=float, default=0.01)
    parser.add_argument("--minimum-teacher-chrf", type=float, default=25.0)
    parser.add_argument("--minimum-chrf-delta", type=float, default=2.0)
    args = parser.parse_args()
    for path in (args.output_teacher, args.output_comet):
        if path.exists() and path.stat().st_size:
            raise SystemExit(f"refusing to overwrite non-empty output: {path}")
    if len(args.candidate) < 2:
        raise SystemExit("candidate selection requires at least two teacher/metric pairs")
    threshold_values = (
        args.minimum_teacher_comet,
        args.minimum_comet_delta,
        args.minimum_teacher_chrf,
        args.minimum_chrf_delta,
    )
    if not all(math.isfinite(value) for value in threshold_values):
        raise SystemExit("selection thresholds must be finite")

    suite_order = load_jsonl(args.suite)
    suite = {str(row.get("id", "")): row for row in suite_order}
    if not suite or len(suite) != len(suite_order) or "" in suite:
        raise SystemExit("suite must contain unique non-empty IDs")
    student_metric = load_json(args.student_comet)
    student_scores = index_results(student_metric, "student metric")
    if (
        student_metric.get("suiteSHA256") != sha256(args.suite)
        or set(student_scores) != set(suite)
    ):
        raise SystemExit("student metric does not cover the exact suite")

    candidate_paths = [(Path(pair[0]), Path(pair[1])) for pair in args.candidate]
    candidates = [
        validate_teacher(
            args.suite,
            suite,
            teacher_path,
            metric_path,
            student_metric,
        )
        for teacher_path, metric_path in candidate_paths
    ]
    chrf = sacrebleu.metrics.CHRF(word_order=2)
    selected_teacher_rows: list[dict] = []
    selected_metric_rows: list[dict] = []
    selected_counts: Counter[int] = Counter()
    newly_eligible = 0
    eligible_counts: Counter[int] = Counter()
    for seed in suite_order:
        identifier = str(seed["id"])
        references = [str(value) for value in seed.get("references", [])]
        student_hypothesis = str(seed.get("studentHypothesis", ""))
        if not references or not student_hypothesis:
            raise SystemExit(f"suite lacks references or student hypothesis: {identifier}")
        student_chrf = chrf.sentence_score(student_hypothesis, references).score
        student_comet = float(student_scores[identifier]["score"])
        eligible: list[tuple[float, float, int]] = []
        for index, (_, _, teacher_rows, metric_rows) in enumerate(candidates):
            hypothesis = str(teacher_rows[identifier].get("hypothesis", "")).strip()
            teacher_chrf = chrf.sentence_score(hypothesis, references).score
            teacher_comet = float(metric_rows[identifier]["score"])
            if candidate_passes(
                seed,
                hypothesis,
                teacher_chrf,
                student_chrf,
                teacher_comet,
                student_comet,
                minimum_teacher_comet=args.minimum_teacher_comet,
                minimum_comet_delta=args.minimum_comet_delta,
                minimum_teacher_chrf=args.minimum_teacher_chrf,
                minimum_chrf_delta=args.minimum_chrf_delta,
            ):
                eligible_counts[index] += 1
                eligible.append((
                    teacher_comet - student_comet,
                    teacher_chrf - student_chrf,
                    -index,
                ))
        selected_index = -max(eligible)[2] if eligible else 0
        if selected_index > 0 and not any(item[2] == 0 for item in eligible):
            newly_eligible += 1
        selected_counts[selected_index] += 1
        selected_teacher_rows.append(copy.deepcopy(candidates[selected_index][2][identifier]))
        selected_metric_rows.append(copy.deepcopy(candidates[selected_index][3][identifier]))

    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    teacher_report = copy.deepcopy(candidates[0][0])
    teacher_report.update({
        "createdAt": created_at,
        "engine": (
            f"mlx-lm:{TEACHER_MODEL}@{TEACHER_REVISION[:12]}:"
            "reference-hidden-candidate-selection"
        ),
        "purpose": "strict multi-candidate local sequence teacher for training only",
        "referenceExposedToCandidateSelection": True,
        "sampling": {"mode": "per-case strict candidate selection"},
        "targetedRetry": None,
        "candidateSelection": {
            "policy": (
                "keep candidate zero unless another candidate passes every frozen filter; "
                "among passing candidates maximize COMET delta then chrF++ delta"
            ),
            "thresholds": {
                "minimumTeacherCOMET": args.minimum_teacher_comet,
                "minimumCOMETDelta": args.minimum_comet_delta,
                "minimumTeacherChrFPlusPlus": args.minimum_teacher_chrf,
                "minimumChrFPlusPlusDelta": args.minimum_chrf_delta,
            },
            "inputs": [
                {
                    "teacherPath": str(teacher_path.resolve()),
                    "teacherSHA256": sha256(teacher_path),
                    "metricPath": str(metric_path.resolve()),
                    "metricSHA256": sha256(metric_path),
                    "eligibleCases": eligible_counts[index],
                    "selectedCases": selected_counts[index],
                }
                for index, (teacher_path, metric_path) in enumerate(candidate_paths)
            ],
            "newlyEligibleCases": newly_eligible,
            "promotionEligible": False,
        },
        "results": selected_teacher_rows,
    })
    args.output_teacher.parent.mkdir(parents=True, exist_ok=True)
    args.output_teacher.write_text(
        json.dumps(teacher_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    metric_report = aggregate_metric(candidates[0][1], suite_order, selected_metric_rows)
    metric_report.update({
        "createdAt": created_at,
        "engine": teacher_report["engine"],
        "engineReportSHA256": sha256(args.output_teacher),
        "results": selected_metric_rows,
        "candidateSelection": teacher_report["candidateSelection"],
    })
    args.output_comet.parent.mkdir(parents=True, exist_ok=True)
    args.output_comet.write_text(
        json.dumps(metric_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "outputTeacher": str(args.output_teacher),
        "outputCOMET": str(args.output_comet),
        "newlyEligibleCases": newly_eligible,
        "selectedCases": dict(sorted(selected_counts.items())),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
