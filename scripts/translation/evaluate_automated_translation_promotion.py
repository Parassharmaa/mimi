#!/usr/bin/env python3
"""Evaluate Mimi's separate no-human-review promotion contract.

Apple may be measured separately, but this evaluator compares a candidate with
the frozen best prior local model. It never treats Apple as a promotion gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from evaluate_translation_promotion import (
    FALLBACK_ASSERTIONS,
    RESULT_FIELDS,
    bootstrap,
    corpus_score,
    gate,
    learned_metric_deltas,
    learned_metric_index,
    load,
    load_jsonl,
    metric_deltas,
    report_index,
    sha256,
)


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def require_hash(value: object, label: str) -> str:
    candidate = str(value or "").strip().lower()
    if len(candidate) != 64 or any(character not in "0123456789abcdef" for character in candidate):
        raise SystemExit(f"invalid SHA-256 for {label}")
    return candidate


def validate_scores(value: object, case_id: str, engine: str) -> tuple[float, bool]:
    if not isinstance(value, dict):
        raise SystemExit(f"invalid automated scores: {case_id}/{engine}")
    maxima = {"adequacy": 4, "fluency": 4, "terminology": 2}
    total = 0
    for name, maximum in maxima.items():
        score = value.get(name)
        if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= maximum:
            raise SystemExit(f"invalid automated score: {case_id}/{engine}/{name}")
        total += score
    critical = value.get("criticalError")
    if not isinstance(critical, bool) or not isinstance(value.get("errorTags"), list):
        raise SystemExit(f"invalid critical evidence: {case_id}/{engine}")
    return float(total), critical or bool(value["errorTags"])


def pairwise_report(
    path: Path,
    label: str,
    suite_path: Path,
    case_ids: set[str],
    candidate_report_path: Path,
    baseline_report_path: Path,
    candidate: dict[str, dict],
    baseline: dict[str, dict],
) -> tuple[str, str, dict[str, tuple[float, float, bool]]]:
    report = load(path)
    if (
        report.get("schemaVersion") != 1
        or report.get("purpose") != "blinded-automated-engine-comparison"
        or report.get("suiteSHA256") != sha256(suite_path)
        or report.get("candidateReportSHA256") != sha256(candidate_report_path)
        or report.get("baselineReportSHA256") != sha256(baseline_report_path)
        or report.get("reasoningTracesStored") is not False
    ):
        raise SystemExit(f"invalid {label} metadata or input hashes")
    model = str(report.get("judgeModel", "")).strip()
    family = str(report.get("judgeModelFamily", "")).strip()
    revision = str(report.get("judgeRevision", "")).strip()
    if not model or not family or not revision:
        raise SystemExit(f"{label} model is not pinned")
    require_hash(report.get("promptSHA256"), f"{label} prompt")
    indexed: dict[str, tuple[float, float, bool]] = {}
    for result in report.get("results", []):
        case_id = str(result.get("caseID", "")).strip()
        if not case_id or case_id in indexed or case_id not in case_ids:
            raise SystemExit(f"{label} has missing, duplicate, or unknown case: {case_id}")
        if result.get("blinded") is not True:
            raise SystemExit(f"{label} result is not blind: {case_id}")
        if (
            result.get("candidateHypothesisSHA256")
            != text_hash(str(candidate[case_id]["hypothesis"]))
            or result.get("baselineHypothesisSHA256")
            != text_hash(str(baseline[case_id]["hypothesis"]))
        ):
            raise SystemExit(f"{label} output hash mismatch: {case_id}")
        require_hash(result.get("requestSHA256"), f"{label} request {case_id}")
        require_hash(result.get("responseSHA256"), f"{label} response {case_id}")
        candidate_score, candidate_critical = validate_scores(
            result.get("candidate"), case_id, "candidate"
        )
        baseline_score, _ = validate_scores(result.get("baseline"), case_id, "baseline")
        indexed[case_id] = (candidate_score, baseline_score, candidate_critical)
    if set(indexed) != case_ids:
        raise SystemExit(f"{label} does not cover the exact frozen suite")
    return model, family, indexed


def deterministic_critical_report(
    path: Path,
    suite_path: Path,
    candidate_report_path: Path,
    case_ids: set[str],
    candidate: dict[str, dict],
) -> set[str]:
    report = load(path)
    if (
        report.get("schemaVersion") != 1
        or report.get("suiteSHA256") != sha256(suite_path)
        or report.get("candidateReportSHA256") != sha256(candidate_report_path)
        or report.get("status") not in {"passed", "failed"}
    ):
        raise SystemExit("invalid deterministic critical-error report")
    indexed: set[str] = set()
    seen: set[str] = set()
    for result in report.get("results", []):
        case_id = str(result.get("caseID", "")).strip()
        if not case_id or case_id in seen or case_id not in case_ids:
            raise SystemExit(f"invalid deterministic critical result: {case_id}")
        seen.add(case_id)
        if result.get("hypothesisSHA256") != text_hash(str(candidate[case_id]["hypothesis"])):
            raise SystemExit(f"deterministic critical output hash mismatch: {case_id}")
        critical = result.get("criticalError")
        tags = result.get("errorTags")
        if not isinstance(critical, bool) or not isinstance(tags, list):
            raise SystemExit(f"invalid deterministic critical fields: {case_id}")
        if critical or tags:
            indexed.add(case_id)
    if seen != case_ids:
        raise SystemExit("deterministic critical report does not cover the exact suite")
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("suite_validation", type=Path)
    parser.add_argument("candidate_report", type=Path)
    parser.add_argument("baseline_report", type=Path)
    parser.add_argument("candidate_learned_metric", type=Path)
    parser.add_argument("baseline_learned_metric", type=Path)
    parser.add_argument("pairwise_judge_a", type=Path)
    parser.add_argument("pairwise_judge_b", type=Path)
    parser.add_argument("deterministic_critical_report", type=Path)
    parser.add_argument("failure_path_verification", type=Path)
    parser.add_argument("parity_verification", type=Path)
    parser.add_argument("distribution_verification", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    manifest, validation = load(args.manifest), load(args.suite_validation)
    suite_rows = load_jsonl(args.suite)
    suite = {str(row.get("id", "")): row for row in suite_rows}
    if not suite or len(suite) != len(suite_rows) or "" in suite:
        raise SystemExit("suite must have non-empty unique IDs")
    case_ids = set(suite)
    candidate_report, baseline_report = load(args.candidate_report), load(args.baseline_report)
    candidate = report_index(candidate_report, "candidate")
    baseline = report_index(baseline_report, "baseline")
    if set(candidate) != case_ids or set(baseline) != case_ids:
        raise SystemExit("suite, candidate, and baseline case IDs must match exactly")
    if candidate_report.get("engine") == baseline_report.get("engine"):
        raise SystemExit("candidate and baseline identify the same engine")
    if not str(candidate_report.get("modelRevision", "")).strip() or not str(
        baseline_report.get("modelRevision", "")
    ).strip():
        raise SystemExit("candidate and baseline model revisions must be pinned")

    learned_configuration = manifest["measurement"]["learnedMetric"]
    candidate_learned = learned_metric_index(
        load(args.candidate_learned_metric),
        "candidate",
        candidate_report["engine"],
        args.candidate_report,
        args.suite,
        learned_configuration,
    )
    baseline_learned = learned_metric_index(
        load(args.baseline_learned_metric),
        "baseline",
        baseline_report["engine"],
        args.baseline_report,
        args.suite,
        learned_configuration,
    )
    if set(candidate_learned) != case_ids or set(baseline_learned) != case_ids:
        raise SystemExit("learned-metric reports must cover the exact suite")

    required_warm_runs = int(manifest["measurement"]["warmRuns"])
    for case_id, suite_row in suite.items():
        for report_name, result in (("candidate", candidate[case_id]), ("baseline", baseline[case_id])):
            for field in RESULT_FIELDS:
                if result.get(field) != suite_row.get(field):
                    raise SystemExit(f"{report_name} result disagrees with suite {field}: {case_id}")
            if len(result.get("warmLatencySeconds", [])) < required_warm_runs:
                raise SystemExit(f"{report_name} result lacks warm runs: {case_id}")

    judge_a_model, judge_a_family, judge_a = pairwise_report(
        args.pairwise_judge_a,
        "pairwise judge A",
        args.suite,
        case_ids,
        args.candidate_report,
        args.baseline_report,
        candidate,
        baseline,
    )
    judge_b_model, judge_b_family, judge_b = pairwise_report(
        args.pairwise_judge_b,
        "pairwise judge B",
        args.suite,
        case_ids,
        args.candidate_report,
        args.baseline_report,
        candidate,
        baseline,
    )
    if judge_a_model == judge_b_model or judge_a_family == judge_b_family:
        raise SystemExit("pairwise judges must use distinct models and model families")
    training_teachers = {
        str(value).strip()
        for value in candidate_report.get("trainingTeacherModels", [])
        if str(value).strip()
    }
    if {judge_a_model, judge_b_model} & training_teachers:
        raise SystemExit("pairwise judges overlap candidate training teachers")

    deterministic_critical = deterministic_critical_report(
        args.deterministic_critical_report,
        args.suite,
        args.candidate_report,
        case_ids,
        candidate,
    )
    automated_deltas: dict[str, list[float]] = defaultdict(list)
    automated_candidate_scores: dict[str, list[float]] = defaultdict(list)
    critical_cases: set[str] = set(deterministic_critical)
    for case_id in case_ids:
        for evidence in (judge_a[case_id], judge_b[case_id]):
            candidate_score, baseline_score, candidate_critical = evidence
            automated_deltas[case_id].append(candidate_score - baseline_score)
            automated_candidate_scores[case_id].append(candidate_score)
            if candidate_critical:
                critical_cases.add(case_id)

    suite_validation_gate = (
        validation.get("status") == "claim-ready-automated-suite-validated"
        and validation.get("suiteID") == manifest.get("suiteID")
        and validation.get("suite", {}).get("sha256") == sha256(args.suite)
        and validation.get("manifest", {}).get("sha256") == sha256(args.manifest)
    )
    failure = load(args.failure_path_verification)
    failure_gate = failure.get("status") == "passed" and all(
        failure.get(name) is True for name in FALLBACK_ASSERTIONS
    )
    parity = load(args.parity_verification)
    parity_results = parity.get("results", [])
    parity_gate = (
        parity.get("schemaVersion") == 1
        and parity.get("status") == "passed"
        and parity.get("engine") == "swift-mlx-marian-exact-output-parity"
        and parity.get("modelRevision") == candidate_report.get("modelRevision")
        and parity.get("suiteSHA256") == sha256(args.suite)
        and parity.get("pythonReportSHA256") == sha256(args.candidate_report)
        and parity.get("cases") == len(suite)
        and parity.get("exactMatches") == len(suite)
        and isinstance(parity_results, list)
        and len(parity_results) == len(suite)
        and {str(row.get("caseID", "")) for row in parity_results} == case_ids
        and all(row.get("exactMatch") is True for row in parity_results)
    )
    distribution = load(args.distribution_verification)
    archive = distribution.get("archive", {})
    archive_path = Path(str(archive.get("path", "")))
    archive_current = archive_path.is_file()
    promotion = manifest["promotionGate"]
    distribution_gate = (
        distribution.get("schemaVersion") == 1
        and distribution.get("status") == "passed"
        and distribution.get("modelRevision") == candidate_report.get("modelRevision")
        and isinstance(archive.get("bytes"), int)
        and 0 < archive["bytes"] <= promotion["maximumDistributionArchiveBytes"]
        and archive_current
        and archive_path.stat().st_size == archive["bytes"]
        and sha256(archive_path) == archive.get("sha256")
        and distribution.get("modelBundle", {}).get("bytes") == candidate_report.get("modelBytes")
    )
    model_bytes = candidate_report.get("modelBytes")
    peak_bytes = candidate_report.get("peakResidentBytes")
    global_gates = [
        gate(
            "automated-claim-suite-validation",
            suite_validation_gate,
            validation.get("status"),
            "validated automated artifact bound to suite and manifest hashes",
        ),
        gate(
            "same-hardware-and-os",
            candidate_report.get("hardware") == baseline_report.get("hardware")
            and candidate_report.get("operatingSystem") == baseline_report.get("operatingSystem"),
            {
                "candidate": [candidate_report.get("hardware"), candidate_report.get("operatingSystem")],
                "baseline": [baseline_report.get("hardware"), baseline_report.get("operatingSystem")],
            },
            "exact match",
        ),
        gate(
            "model-bytes-hard-cap",
            isinstance(model_bytes, int) and 0 < model_bytes <= promotion["maximumModelBytes"],
            model_bytes,
            promotion["maximumModelBytes"],
        ),
        gate(
            "peak-resident-bytes",
            isinstance(peak_bytes, int) and 0 < peak_bytes <= promotion["maximumPeakResidentBytes"],
            peak_bytes,
            promotion["maximumPeakResidentBytes"],
        ),
        gate("non-apple-failure-path", failure_gate, failure, "all non-Apple assertions true"),
        gate(
            "swift-python-mlx-exact-output-parity",
            parity_gate,
            {"status": parity.get("status"), "cases": parity.get("cases")},
            "every frozen-suite output exactly matches",
        ),
        gate(
            "combined-distribution-archive-hard-cap",
            distribution_gate,
            {"status": distribution.get("status"), "bytes": archive.get("bytes")},
            promotion["maximumDistributionArchiveBytes"],
        ),
    ]

    samples = int(promotion["pairedBootstrapSamples"])
    confidence = float(promotion["confidenceLevel"])
    seed = int(manifest["randomSeed"])
    direction_reports: dict[str, dict] = {}
    direction_passes: list[bool] = []
    for direction in manifest["directions"]:
        ids = sorted(
            case_id
            for case_id, row in suite.items()
            if f"{row['sourceLanguage']}>{row['targetLanguage']}" == direction
        )
        candidate_score = corpus_score([candidate[case_id] for case_id in ids])
        baseline_score = corpus_score([baseline[case_id] for case_id in ids])
        chrf_interval = bootstrap(metric_deltas(candidate, baseline, ids), samples, confidence, seed)
        learned_interval = bootstrap(
            learned_metric_deltas(candidate_learned, baseline_learned, ids),
            samples,
            confidence,
            seed + 1,
        )
        per_case_automated_delta = [
            sum(automated_deltas[case_id]) / len(automated_deltas[case_id]) for case_id in ids
        ]
        automated_interval = bootstrap(
            per_case_automated_delta, samples, confidence, seed + 2
        )
        candidate_automated_mean = sum(
            sum(automated_candidate_scores[case_id]) / len(automated_candidate_scores[case_id])
            for case_id in ids
        ) / len(ids)
        candidate_learned_mean = sum(float(candidate_learned[case_id]["score"]) for case_id in ids) / len(ids)
        direction_critical = sorted(set(ids) & critical_cases)
        gates = [
            gate("minimum-cases", len(ids) >= manifest["minimumCasesPerDirection"], len(ids), manifest["minimumCasesPerDirection"]),
            gate("all-cases-claim-eligible", all(candidate[case_id]["claimEligible"] for case_id in ids), sum(bool(candidate[case_id]["claimEligible"]) for case_id in ids), len(ids)),
            gate("absolute-chrF++", candidate_score["chrFPlusPlus"] >= promotion["minimumChrFPlusPlus"][direction], candidate_score["chrFPlusPlus"], promotion["minimumChrFPlusPlus"][direction]),
            gate("absolute-learned-metric", candidate_learned_mean >= promotion["minimumLearnedMetric"][direction], candidate_learned_mean, promotion["minimumLearnedMetric"][direction]),
            gate("chrF++-paired-bootstrap-lower", chrf_interval["lower"] > promotion["minimumChrFDeltaLowerBound"], chrf_interval["lower"], promotion["minimumChrFDeltaLowerBound"]),
            gate("learned-metric-paired-bootstrap-lower", learned_interval["lower"] > promotion["minimumLearnedMetricDeltaLowerBound"], learned_interval["lower"], promotion["minimumLearnedMetricDeltaLowerBound"]),
            gate("automated-pairwise-paired-bootstrap-lower", automated_interval["lower"] > promotion["minimumAutomatedPairwiseDeltaLowerBound"], automated_interval["lower"], promotion["minimumAutomatedPairwiseDeltaLowerBound"]),
            gate("automated-candidate-mean-score", candidate_automated_mean >= promotion["minimumAutomatedCandidateMeanScore"], candidate_automated_mean, promotion["minimumAutomatedCandidateMeanScore"]),
            gate("candidate-critical-errors", len(direction_critical) <= promotion["maximumCriticalMeaningErrors"], len(direction_critical), promotion["maximumCriticalMeaningErrors"]),
            gate("warm-p95-latency", candidate_score["warmP95LatencySeconds"] <= promotion["maximumWarmP95LatencySeconds"], candidate_score["warmP95LatencySeconds"], promotion["maximumWarmP95LatencySeconds"]),
        ]
        passed = all(value["passed"] for value in gates)
        direction_passes.append(passed)
        direction_reports[direction] = {
            "passed": passed,
            "candidate": candidate_score,
            "baseline": baseline_score,
            "candidateLearnedMetricMean": candidate_learned_mean,
            "candidateAutomatedMeanScore": candidate_automated_mean,
            "chrFPlusPlusPairedDelta": chrf_interval,
            "learnedMetricPairedDelta": learned_interval,
            "automatedPairwisePairedDelta": automated_interval,
            "candidateCriticalCaseIDs": direction_critical,
            "gates": gates,
        }

    promote = all(value["passed"] for value in global_gates) and all(direction_passes)
    output = {
        "schemaVersion": 1,
        "status": "promotion-approved" if promote else "promotion-rejected",
        "promote": promote,
        "suiteID": manifest["suiteID"],
        "candidateEngine": candidate_report["engine"],
        "candidateModelRevision": candidate_report["modelRevision"],
        "baselineEngine": baseline_report["engine"],
        "baselineModelRevision": baseline_report["modelRevision"],
        "appleComparisonRole": "diagnostic-only-not-a-promotion-input",
        "preferredSizeTargetMet": isinstance(model_bytes, int)
        and model_bytes <= promotion["preferredModelBytes"],
        "globalGates": global_gates,
        "directions": direction_reports,
        "inputs": {
            name: sha256(path)
            for name, path in {
                "suiteSHA256": args.suite,
                "manifestSHA256": args.manifest,
                "suiteValidationSHA256": args.suite_validation,
                "candidateReportSHA256": args.candidate_report,
                "baselineReportSHA256": args.baseline_report,
                "candidateLearnedMetricSHA256": args.candidate_learned_metric,
                "baselineLearnedMetricSHA256": args.baseline_learned_metric,
                "pairwiseJudgeASHA256": args.pairwise_judge_a,
                "pairwiseJudgeBSHA256": args.pairwise_judge_b,
                "deterministicCriticalReportSHA256": args.deterministic_critical_report,
                "failurePathVerificationSHA256": args.failure_path_verification,
                "parityVerificationSHA256": args.parity_verification,
                "distributionVerificationSHA256": args.distribution_verification,
            }.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if promote else 2)


if __name__ == "__main__":
    main()
