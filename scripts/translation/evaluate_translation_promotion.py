#!/usr/bin/env python3
"""Evaluate every fail-closed gate before Mimi may promote a local translator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import sacrebleu


RESULT_FIELDS = (
    "sourceLanguage",
    "targetLanguage",
    "domain",
    "source",
    "references",
    "claimEligible",
)
FALLBACK_ASSERTIONS = (
    "appleDefaultWhenExperimentalDisabled",
    "candidateFailureDoesNotUseApple",
    "candidateFailurePreservesLocalResults",
    "candidateFailureShowsRetryableError",
    "applePartialsWhenExperimentalDisabled",
    "experimentalPartialsDoNotUseApple",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def report_index(report: dict, name: str) -> dict[str, dict]:
    if report.get("schemaVersion") != 1:
        raise SystemExit(f"{name} report has unsupported schema")
    output: dict[str, dict] = {}
    for row in report.get("results", []):
        case_id = str(row.get("caseID", "")).strip()
        if not case_id or case_id in output:
            raise SystemExit(f"{name} report has empty or duplicate case ID: {case_id}")
        output[case_id] = row
    if not output:
        raise SystemExit(f"{name} report has no results")
    return output


def learned_metric_index(
    report: dict,
    name: str,
    expected_engine: str,
    engine_report_path: Path,
    suite_path: Path,
    configuration: dict,
) -> dict[str, dict]:
    if report.get("schemaVersion") != 1 or report.get("engine") != expected_engine:
        raise SystemExit(f"{name} learned-metric report has invalid schema or engine")
    expected = {
        "metric": configuration["name"],
        "modelRepository": configuration["modelRepository"],
        "modelRevision": configuration["modelRevision"],
        "modelLicense": configuration["modelLicense"],
        "package": configuration["package"],
        "packageVersion": configuration["packageVersion"],
        "setuptoolsVersion": configuration["setuptoolsVersion"],
        "precision": configuration["precision"],
        "multipleReferenceAggregation": configuration["multipleReferenceAggregation"],
    }
    for field, value in expected.items():
        if report.get(field) != value:
            raise SystemExit(f"{name} learned-metric report is not pinned for {field}")
    signature = hashlib.sha256(
        json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if report.get("signatureSHA256") != signature:
        raise SystemExit(f"{name} learned-metric signature is invalid")
    if report.get("engineReportSHA256") != sha256(engine_report_path):
        raise SystemExit(f"{name} learned-metric report is not bound to its engine report")
    if report.get("suiteSHA256") != sha256(suite_path):
        raise SystemExit(f"{name} learned-metric report is not bound to the suite")
    output: dict[str, dict] = {}
    for row in report.get("results", []):
        case_id = str(row.get("caseID", "")).strip()
        score = row.get("score")
        if (
            not case_id
            or case_id in output
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise SystemExit(f"{name} learned-metric report has invalid result: {case_id}")
        output[case_id] = row
    if not output:
        raise SystemExit(f"{name} learned-metric report has no results")
    return output


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil((len(ordered) - 1) * fraction)]


def corpus_score(rows: list[dict]) -> dict:
    hypotheses = [str(row["hypothesis"]) for row in rows]
    reference_count = max(len(row["references"]) for row in rows)
    references = [
        [row["references"][min(index, len(row["references"]) - 1)] for row in rows]
        for index in range(reference_count)
    ]
    latencies = [
        float(value)
        for row in rows
        for value in (row.get("warmLatencySeconds") or [row["latencySeconds"]])
    ]
    return {
        "cases": len(rows),
        "chrFPlusPlus": sacrebleu.corpus_chrf(hypotheses, references, word_order=2).score,
        "sacreBLEUIntl": sacrebleu.corpus_bleu(hypotheses, references, tokenize="intl").score,
        "warmP50LatencySeconds": percentile(latencies, 0.50),
        "warmP95LatencySeconds": percentile(latencies, 0.95),
    }


def bootstrap(values: list[float], samples: int, confidence: float, seed: int) -> dict:
    if not values or samples < 1 or not 0 < confidence < 1:
        raise SystemExit("invalid bootstrap inputs")
    rng = random.Random(seed)
    means = sorted(
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(samples)
    )
    alpha = (1 - confidence) / 2
    lower_index = min(samples - 1, max(0, math.floor(samples * alpha)))
    upper_index = min(samples - 1, max(0, math.ceil(samples * (1 - alpha)) - 1))
    return {
        "mean": sum(values) / len(values),
        "lower": means[lower_index],
        "upper": means[upper_index],
        "samples": samples,
        "confidence": confidence,
    }


def metric_deltas(candidate: dict[str, dict], apple: dict[str, dict], case_ids: list[str]) -> list[float]:
    output: list[float] = []
    for case_id in case_ids:
        references = candidate[case_id]["references"]
        candidate_score = sacrebleu.sentence_chrf(
            candidate[case_id]["hypothesis"], references, word_order=2
        ).score
        apple_score = sacrebleu.sentence_chrf(
            apple[case_id]["hypothesis"], references, word_order=2
        ).score
        output.append(candidate_score - apple_score)
    return output


def learned_metric_deltas(
    candidate: dict[str, dict], apple: dict[str, dict], case_ids: list[str]
) -> list[float]:
    return [
        float(candidate[case_id]["score"]) - float(apple[case_id]["score"])
        for case_id in case_ids
    ]


def assignment_map(path: Path, case_ids: set[str]) -> tuple[set[str], dict[tuple[str, str], dict]]:
    output: dict[tuple[str, str], dict] = {}
    reviewers: set[str] = set()
    for row in load_jsonl(path):
        case_id = str(row.get("caseID", "")).strip()
        reviewer = str(row.get("reviewerID", "")).strip()
        key = (reviewer, case_id)
        if case_id not in case_ids or not reviewer or key in output:
            raise SystemExit(f"invalid or duplicate sealed assignment: {key}")
        if {row.get("outputAEngine"), row.get("outputBEngine")} != {"candidate", "apple"}:
            raise SystemExit(f"assignment does not map candidate and Apple exactly once: {key}")
        reviewers.add(reviewer)
        output[key] = row
    if len(reviewers) != 2 or len(output) != len(case_ids) * 2:
        raise SystemExit("sealed assignments must cover every case for two reviewers")
    return reviewers, output


def validate_score(value: object, maximum: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise SystemExit(f"invalid human score for {label}: {value}")
    return value


def response_map(path: Path, case_ids: set[str]) -> tuple[str, dict[str, dict]]:
    output: dict[str, dict] = {}
    reviewers: set[str] = set()
    for row in load_jsonl(path):
        case_id = str(row.get("caseID", "")).strip()
        reviewer = str(row.get("reviewerID", "")).strip()
        if case_id not in case_ids or case_id in output or not reviewer:
            raise SystemExit(f"invalid or duplicate human response: {case_id}")
        if row.get("blinded") is not True:
            raise SystemExit(f"human response is not marked blind: {case_id}")
        for output_name in ("outputA", "outputB"):
            scores = row.get(output_name, {})
            validate_score(scores.get("adequacy"), 4, f"{case_id}/{output_name}/adequacy")
            validate_score(scores.get("fluency"), 4, f"{case_id}/{output_name}/fluency")
            validate_score(scores.get("terminology"), 2, f"{case_id}/{output_name}/terminology")
            if not isinstance(scores.get("criticalError"), bool):
                raise SystemExit(f"criticalError must be boolean: {case_id}/{output_name}")
        reviewers.add(reviewer)
        output[case_id] = row
    if len(reviewers) != 1 or set(output) != case_ids:
        raise SystemExit(f"response file must cover every case for one reviewer: {path}")
    return next(iter(reviewers)), output


def total_score(value: dict) -> int:
    return int(value["adequacy"]) + int(value["fluency"]) + int(value["terminology"])


def gate(name: str, passed: bool, actual: object, requirement: object) -> dict:
    return {"name": name, "passed": passed, "actual": actual, "requirement": requirement}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("suite_validation", type=Path)
    parser.add_argument("candidate_report", type=Path)
    parser.add_argument("apple_report", type=Path)
    parser.add_argument("candidate_learned_metric", type=Path)
    parser.add_argument("apple_learned_metric", type=Path)
    parser.add_argument("sealed_assignments", type=Path)
    parser.add_argument("human_review_a", type=Path)
    parser.add_argument("human_review_b", type=Path)
    parser.add_argument("fallback_verification", type=Path)
    parser.add_argument("parity_verification", type=Path)
    parser.add_argument("distribution_verification", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    manifest, validation = load(args.manifest), load(args.suite_validation)
    suite_rows = load_jsonl(args.suite)
    suite = {str(row["id"]): row for row in suite_rows}
    if not suite or len(suite) != len(suite_rows):
        raise SystemExit("suite must have non-empty unique IDs")
    candidate_report, apple_report = load(args.candidate_report), load(args.apple_report)
    candidate = report_index(candidate_report, "candidate")
    apple = report_index(apple_report, "Apple")
    if set(candidate) != set(suite) or set(apple) != set(suite):
        raise SystemExit("suite, candidate, and Apple case IDs must match exactly")
    if apple_report.get("engine") != "apple-translation-high-fidelity":
        raise SystemExit("Apple report is not the high-fidelity Apple engine")
    if candidate_report.get("engine") == apple_report.get("engine"):
        raise SystemExit("candidate and Apple reports identify the same engine")
    if not str(candidate_report.get("modelRevision", "")).strip():
        raise SystemExit("candidate report is missing modelRevision")
    learned_configuration = manifest["measurement"]["learnedMetric"]
    candidate_learned = learned_metric_index(
        load(args.candidate_learned_metric),
        "candidate",
        candidate_report["engine"],
        args.candidate_report,
        args.suite,
        learned_configuration,
    )
    apple_learned = learned_metric_index(
        load(args.apple_learned_metric),
        "Apple",
        apple_report["engine"],
        args.apple_report,
        args.suite,
        learned_configuration,
    )
    if set(candidate_learned) != set(suite) or set(apple_learned) != set(suite):
        raise SystemExit("learned-metric reports must cover the exact suite")

    required_warm_runs = int(manifest["measurement"]["warmRuns"])
    for case_id, suite_row in suite.items():
        for report_name, result in (("candidate", candidate[case_id]), ("Apple", apple[case_id])):
            for field in RESULT_FIELDS:
                expected = suite_row["id"] if field == "caseID" else suite_row.get(field)
                if result.get(field) != expected:
                    raise SystemExit(f"{report_name} result disagrees with suite {field}: {case_id}")
            if len(result.get("warmLatencySeconds", [])) < required_warm_runs:
                raise SystemExit(f"{report_name} result lacks warm runs: {case_id}")
        for field in RESULT_FIELDS:
            if candidate[case_id].get(field) != apple[case_id].get(field):
                raise SystemExit(f"candidate and Apple disagree on {field}: {case_id}")

    reviewers, assignments = assignment_map(args.sealed_assignments, set(suite))
    reviewer_a, reviews_a = response_map(args.human_review_a, set(suite))
    reviewer_b, reviews_b = response_map(args.human_review_b, set(suite))
    if {reviewer_a, reviewer_b} != reviewers or reviewer_a == reviewer_b:
        raise SystemExit("review response identities do not match sealed assignments")

    human_deltas: dict[str, list[float]] = defaultdict(list)
    candidate_critical: dict[str, set[str]] = defaultdict(set)
    for reviewer, reviews in ((reviewer_a, reviews_a), (reviewer_b, reviews_b)):
        for case_id, review in reviews.items():
            assignment = assignments[(reviewer, case_id)]
            expected_hypotheses = {
                "candidate": str(candidate[case_id]["hypothesis"]),
                "apple": str(apple[case_id]["hypothesis"]),
            }
            for label in ("A", "B"):
                engine = assignment[f"output{label}Engine"]
                if assignment[f"output{label}SHA256"] != text_hash(expected_hypotheses[engine]):
                    raise SystemExit(f"sealed output hash no longer matches report: {reviewer}/{case_id}/{label}")
            decoded = {
                assignment["outputAEngine"]: review["outputA"],
                assignment["outputBEngine"]: review["outputB"],
            }
            human_deltas[case_id].append(
                float(total_score(decoded["candidate"]) - total_score(decoded["apple"]))
            )
            if decoded["candidate"]["criticalError"]:
                direction = f"{suite[case_id]['sourceLanguage']}>{suite[case_id]['targetLanguage']}"
                candidate_critical[direction].add(case_id)

    suite_validation_gate = (
        validation.get("status") == "claim-ready-suite-validated"
        and validation.get("suiteID") == manifest.get("suiteID")
        and validation.get("suite", {}).get("sha256") == sha256(args.suite)
    )
    fallback = load(args.fallback_verification)
    fallback_gate = fallback.get("status") == "passed" and all(
        fallback.get(name) is True for name in FALLBACK_ASSERTIONS
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
        and {str(row.get("caseID", "")) for row in parity_results} == set(suite)
        and all(row.get("exactMatch") is True for row in parity_results)
    )
    distribution = load(args.distribution_verification)
    archive_record = distribution.get("archive", {})
    archive_path = Path(str(archive_record.get("path", "")))
    archive_current = archive_path.is_file()
    distribution_gate = (
        distribution.get("schemaVersion") == 1
        and distribution.get("status") == "passed"
        and distribution.get("modelRevision") == candidate_report.get("modelRevision")
        and archive_record.get("maximumBytes")
        == manifest["promotionGate"]["maximumDistributionArchiveBytes"]
        and isinstance(archive_record.get("bytes"), int)
        and 0 < archive_record["bytes"]
        <= manifest["promotionGate"]["maximumDistributionArchiveBytes"]
        and archive_current
        and archive_path.stat().st_size == archive_record.get("bytes")
        and sha256(archive_path) == archive_record.get("sha256")
        and distribution.get("modelBundle", {}).get("bytes")
        == candidate_report.get("modelBytes")
    )
    hardware_match = (
        candidate_report.get("hardware") == apple_report.get("hardware")
        and candidate_report.get("operatingSystem") == apple_report.get("operatingSystem")
    )
    promotion = manifest["promotionGate"]
    model_bytes = candidate_report.get("modelBytes")
    peak_bytes = candidate_report.get("peakResidentBytes")
    global_gates = [
        gate("claim-ready-suite-validation", suite_validation_gate, validation.get("status"), "validated artifact bound to suite SHA-256"),
        gate("same-hardware-and-os", hardware_match, {
            "candidate": [candidate_report.get("hardware"), candidate_report.get("operatingSystem")],
            "apple": [apple_report.get("hardware"), apple_report.get("operatingSystem")],
        }, "exact match"),
        gate("model-bytes", isinstance(model_bytes, int) and 0 < model_bytes <= promotion["maximumModelBytes"], model_bytes, promotion["maximumModelBytes"]),
        gate("peak-resident-bytes", isinstance(peak_bytes, int) and 0 < peak_bytes <= promotion["maximumPeakResidentBytes"], peak_bytes, promotion["maximumPeakResidentBytes"]),
        gate("apple-fallback-contract", fallback_gate, fallback, "all fallback assertions true"),
        gate("swift-python-mlx-exact-output-parity", parity_gate, {
            "status": parity.get("status"),
            "cases": parity.get("cases"),
            "exactMatches": parity.get("exactMatches"),
            "modelRevision": parity.get("modelRevision"),
        }, "every frozen-suite output matches for the exact model revision"),
        gate("combined-distribution-archive", distribution_gate, {
            "status": distribution.get("status"),
            "bytes": archive_record.get("bytes"),
            "sha256": archive_record.get("sha256"),
            "modelRevision": distribution.get("modelRevision"),
        }, {
            "maximumBytes": promotion["maximumDistributionArchiveBytes"],
            "currentArchiveAndModelHashesMustMatch": True,
        }),
    ]

    samples = int(promotion["pairedBootstrapSamples"])
    confidence = float(promotion["confidenceLevel"])
    seed = int(manifest["randomSeed"])
    direction_reports: dict[str, dict] = {}
    direction_passes: list[bool] = []
    for direction in manifest["directions"]:
        case_ids = sorted(
            case_id
            for case_id, row in suite.items()
            if f"{row['sourceLanguage']}>{row['targetLanguage']}" == direction
        )
        candidate_rows = [candidate[case_id] for case_id in case_ids]
        apple_rows = [apple[case_id] for case_id in case_ids]
        candidate_score, apple_score = corpus_score(candidate_rows), corpus_score(apple_rows)
        metric_interval = bootstrap(
            metric_deltas(candidate, apple, case_ids), samples, confidence, seed
        )
        learned_interval = bootstrap(
            learned_metric_deltas(candidate_learned, apple_learned, case_ids),
            samples,
            confidence,
            seed + 2,
        )
        per_case_human = [
            sum(human_deltas[case_id]) / len(human_deltas[case_id]) for case_id in case_ids
        ]
        if any(len(human_deltas[case_id]) != 2 for case_id in case_ids):
            raise SystemExit(f"direction lacks two human scores per case: {direction}")
        human_interval = bootstrap(per_case_human, samples, confidence, seed + 1)
        latency_ratio = candidate_score["warmP95LatencySeconds"] / apple_score["warmP95LatencySeconds"]
        critical_count = len(candidate_critical[direction])
        direction_gates = [
            gate("minimum-cases", len(case_ids) >= manifest["minimumCasesPerDirection"], len(case_ids), manifest["minimumCasesPerDirection"]),
            gate("all-cases-claim-eligible", all(candidate[case_id]["claimEligible"] for case_id in case_ids), sum(bool(candidate[case_id]["claimEligible"]) for case_id in case_ids), len(case_ids)),
            gate("chrF++-paired-bootstrap-lower", metric_interval["lower"] > promotion["minimumChrFDeltaLowerBound"], metric_interval["lower"], promotion["minimumChrFDeltaLowerBound"]),
            gate(
                "learned-metric-paired-bootstrap-lower",
                learned_interval["lower"]
                > promotion["minimumLearnedMetricDeltaLowerBound"],
                learned_interval["lower"],
                promotion["minimumLearnedMetricDeltaLowerBound"],
            ),
            gate("human-score-paired-bootstrap-lower", human_interval["lower"] > promotion["minimumHumanQualityDeltaLowerBound"], human_interval["lower"], promotion["minimumHumanQualityDeltaLowerBound"]),
            gate("candidate-critical-errors", critical_count <= promotion["maximumCriticalMeaningErrors"], critical_count, promotion["maximumCriticalMeaningErrors"]),
            gate("warm-p95-ratio-to-apple", latency_ratio <= promotion["maximumWarmP95LatencyRatioToApple"], latency_ratio, promotion["maximumWarmP95LatencyRatioToApple"]),
        ]
        passed = all(value["passed"] for value in direction_gates)
        direction_passes.append(passed)
        domains: dict[str, dict] = {}
        for domain in manifest["domains"]:
            domain_ids = [case_id for case_id in case_ids if suite[case_id]["domain"] == domain]
            domain_candidate = corpus_score([candidate[case_id] for case_id in domain_ids])
            domain_apple = corpus_score([apple[case_id] for case_id in domain_ids])
            domain_human = [
                sum(human_deltas[case_id]) / len(human_deltas[case_id]) for case_id in domain_ids
            ]
            domains[domain] = {
                "candidate": domain_candidate,
                "apple": domain_apple,
                "chrFPlusPlusPairedDelta": bootstrap(
                    metric_deltas(candidate, apple, domain_ids),
                    samples,
                    confidence,
                    seed,
                ),
                "learnedMetricPairedDelta": bootstrap(
                    learned_metric_deltas(candidate_learned, apple_learned, domain_ids),
                    samples,
                    confidence,
                    seed + 2,
                ),
                "humanScorePairedDelta": bootstrap(
                    domain_human,
                    samples,
                    confidence,
                    seed + 1,
                ),
                "candidateCriticalCaseIDs": sorted(
                    set(domain_ids) & candidate_critical[direction]
                ),
            }
        direction_reports[direction] = {
            "passed": passed,
            "candidate": candidate_score,
            "apple": apple_score,
            "chrFPlusPlusPairedDelta": metric_interval,
            "learnedMetric": {
                "name": learned_configuration["name"],
                "signatureSHA256": load(args.candidate_learned_metric)["signatureSHA256"],
                "candidateMean": sum(
                    float(candidate_learned[case_id]["score"]) for case_id in case_ids
                ) / len(case_ids),
                "appleMean": sum(
                    float(apple_learned[case_id]["score"]) for case_id in case_ids
                ) / len(case_ids),
                "pairedDelta": learned_interval,
            },
            "humanScorePairedDelta": human_interval,
            "candidateCriticalCaseIDs": sorted(candidate_critical[direction]),
            "domains": domains,
            "gates": direction_gates,
        }

    promote = all(value["passed"] for value in global_gates) and all(direction_passes)
    output = {
        "schemaVersion": 1,
        "status": "promotion-approved" if promote else "promotion-rejected",
        "promote": promote,
        "suiteID": manifest["suiteID"],
        "candidateEngine": candidate_report["engine"],
        "candidateModelRevision": candidate_report["modelRevision"],
        "appleEngine": apple_report["engine"],
        "globalGates": global_gates,
        "directions": direction_reports,
        "inputs": {
            "suiteSHA256": sha256(args.suite),
            "candidateReportSHA256": sha256(args.candidate_report),
            "appleReportSHA256": sha256(args.apple_report),
            "candidateLearnedMetricSHA256": sha256(args.candidate_learned_metric),
            "appleLearnedMetricSHA256": sha256(args.apple_learned_metric),
            "sealedAssignmentsSHA256": sha256(args.sealed_assignments),
            "humanReviewASHA256": sha256(args.human_review_a),
            "humanReviewBSHA256": sha256(args.human_review_b),
            "fallbackVerificationSHA256": sha256(args.fallback_verification),
            "parityVerificationSHA256": sha256(args.parity_verification),
            "distributionVerificationSHA256": sha256(args.distribution_verification),
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
