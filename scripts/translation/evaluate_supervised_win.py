#!/usr/bin/env python3
"""Authorize DQO only after a hash-bound supervised development win."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import sacrebleu


LANGUAGES = {
    "en-ja": ("en-US", "ja-JP"),
    "ja-en": ("ja-JP", "en-US"),
}
RESULT_FIELDS = (
    "sourceLanguage",
    "targetLanguage",
    "domain",
    "source",
    "references",
    "claimEligible",
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


def index_suite(path: Path, direction: str, name: str) -> dict[str, dict]:
    expected = LANGUAGES[direction]
    rows = load_jsonl(path)
    output: dict[str, dict] = {}
    for row in rows:
        case_id = str(row.get("id", "")).strip()
        if not case_id or case_id in output:
            raise SystemExit(f"{name} suite has empty or duplicate case ID: {case_id}")
        if (row.get("sourceLanguage"), row.get("targetLanguage")) != expected:
            raise SystemExit(f"{name} suite contains the wrong direction: {case_id}")
        if row.get("claimEligible") is not False:
            raise SystemExit(f"{name} suite must be development-only and non-claimable: {case_id}")
        references = row.get("references")
        if not isinstance(references, list) or not references or not all(
            isinstance(value, str) and value.strip() for value in references
        ):
            raise SystemExit(f"{name} suite case lacks references: {case_id}")
        output[case_id] = row
    if not output:
        raise SystemExit(f"{name} suite is empty")
    return output


def index_report(path: Path, suite: dict[str, dict], name: str) -> tuple[dict, dict[str, dict]]:
    report = load(path)
    if report.get("schemaVersion") != 1:
        raise SystemExit(f"{name} report has unsupported schema")
    output: dict[str, dict] = {}
    for row in report.get("results", []):
        case_id = str(row.get("caseID", "")).strip()
        if not case_id or case_id in output:
            raise SystemExit(f"{name} report has empty or duplicate case ID: {case_id}")
        output[case_id] = row
    if set(output) != set(suite):
        raise SystemExit(f"{name} report must cover the exact suite")
    for case_id, expected in suite.items():
        for field in RESULT_FIELDS:
            if output[case_id].get(field) != expected.get(field):
                raise SystemExit(f"{name} report disagrees with suite {field}: {case_id}")
        if not str(output[case_id].get("hypothesis", "")).strip():
            raise SystemExit(f"{name} report has an empty hypothesis: {case_id}")
    return report, output


def corpus_chrf(rows: dict[str, dict]) -> float:
    ordered = [rows[key] for key in sorted(rows)]
    hypotheses = [row["hypothesis"] for row in ordered]
    reference_count = max(len(row["references"]) for row in ordered)
    references = [
        [row["references"][min(index, len(row["references"]) - 1)] for row in ordered]
        for index in range(reference_count)
    ]
    return float(sacrebleu.corpus_chrf(hypotheses, references, word_order=2).score)


def sentence_chrf(row: dict) -> float:
    return float(
        sacrebleu.sentence_chrf(
            row["hypothesis"], row["references"], word_order=2
        ).score
    )


def bootstrap(values: list[float], samples: int, confidence: float, seed: int) -> dict:
    if not values or samples < 1 or not 0 < confidence < 1:
        raise SystemExit("invalid bootstrap configuration")
    rng = random.Random(seed)
    means = sorted(
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(samples)
    )
    alpha = (1 - confidence) / 2
    return {
        "mean": sum(values) / len(values),
        "lower": means[max(0, math.floor(samples * alpha))],
        "upper": means[min(samples - 1, math.ceil(samples * (1 - alpha)) - 1)],
        "samples": samples,
        "confidence": confidence,
    }


def assignment_map(path: Path, case_ids: set[str]) -> tuple[set[str], dict[tuple[str, str], dict]]:
    output: dict[tuple[str, str], dict] = {}
    reviewers: set[str] = set()
    for row in load_jsonl(path):
        reviewer, case_id = str(row.get("reviewerID", "")).strip(), str(row.get("caseID", "")).strip()
        key = (reviewer, case_id)
        if not reviewer or case_id not in case_ids or key in output:
            raise SystemExit(f"invalid or duplicate sealed assignment: {reviewer}/{case_id}")
        if {row.get("outputAEngine"), row.get("outputBEngine")} != {"candidate", "base"}:
            raise SystemExit(f"sealed assignment does not compare candidate and baseline: {case_id}")
        reviewers.add(reviewer)
        output[key] = row
    if len(reviewers) != 2 or set(output) != {(reviewer, case) for reviewer in reviewers for case in case_ids}:
        raise SystemExit("sealed assignments must cover every case for two reviewers")
    return reviewers, output


def validate_score(value: object, maximum: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise SystemExit(f"invalid human score: {label}")
    return value


def response_map(path: Path, case_ids: set[str]) -> tuple[str, dict[str, dict]]:
    output: dict[str, dict] = {}
    reviewers: set[str] = set()
    for row in load_jsonl(path):
        reviewer, case_id = str(row.get("reviewerID", "")).strip(), str(row.get("caseID", "")).strip()
        if not reviewer or case_id not in case_ids or case_id in output or row.get("blinded") is not True:
            raise SystemExit(f"invalid, duplicate, or unblinded response: {reviewer}/{case_id}")
        for label in ("outputA", "outputB"):
            score = row.get(label, {})
            validate_score(score.get("adequacy"), 4, f"{case_id}/{label}/adequacy")
            validate_score(score.get("fluency"), 4, f"{case_id}/{label}/fluency")
            validate_score(score.get("terminology"), 2, f"{case_id}/{label}/terminology")
            if not isinstance(score.get("criticalError"), bool):
                raise SystemExit(f"invalid critical-error flag: {case_id}/{label}")
        reviewers.add(reviewer)
        output[case_id] = row
    if len(reviewers) != 1 or set(output) != case_ids:
        raise SystemExit(f"human response must cover every case for one reviewer: {path}")
    return next(iter(reviewers)), output


def human_total(value: dict) -> int:
    return int(value["adequacy"]) + int(value["fluency"]) + int(value["terminology"])


def validate_bundle(bundle: Path, checkpoint: Path, direction: str) -> str:
    root_path = bundle / "manifest.json"
    direction_path = bundle / direction / "manifest.json"
    checkpoint_path = checkpoint / "model.safetensors"
    for path in (root_path, direction_path, checkpoint_path):
        if not path.is_file():
            raise SystemExit(f"missing candidate integrity input: {path}")
    root = load(root_path)
    if root.get("format") != "mimi-mlx-marian-pair-v1":
        raise SystemExit("candidate bundle has an unsupported format")
    for relative, expected in root.get("files", {}).items():
        path = bundle / relative
        if (
            not path.is_file()
            or path.stat().st_size != expected.get("bytes")
            or sha256(path) != expected.get("sha256")
        ):
            raise SystemExit(f"candidate bundle integrity failure: {relative}")
    direction_manifest = load(direction_path)
    if direction_manifest.get("direction") != direction:
        raise SystemExit("candidate direction manifest disagrees with requested direction")
    checkpoint_hash = sha256(checkpoint_path)
    if direction_manifest.get("source_weights_sha256") != checkpoint_hash:
        raise SystemExit("candidate bundle was not quantized from the supervised checkpoint")
    return f"pair-manifest-sha256:{sha256(root_path)}"


def gate(name: str, passed: bool, actual: object, requirement: object) -> dict:
    return {"name": name, "passed": passed, "actual": actual, "requirement": requirement}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("development_suite", type=Path)
    parser.add_argument("candidate_development_report", type=Path)
    parser.add_argument("base_development_report", type=Path)
    parser.add_argument("retention_suite", type=Path)
    parser.add_argument("candidate_retention_report", type=Path)
    parser.add_argument("base_retention_report", type=Path)
    parser.add_argument("sealed_assignments", type=Path)
    parser.add_argument("human_review_a", type=Path)
    parser.add_argument("human_review_b", type=Path)
    parser.add_argument("candidate_bundle", type=Path)
    parser.add_argument("supervised_checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--direction", choices=tuple(LANGUAGES), required=True)
    parser.add_argument("--paired-bootstrap-samples", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--maximum-retention-regression", type=float, default=0.5)
    args = parser.parse_args()

    development = index_suite(args.development_suite, args.direction, "development")
    retention = index_suite(args.retention_suite, args.direction, "retention")
    if set(development) & set(retention):
        raise SystemExit("development and retention case IDs overlap")
    candidate_dev_report, candidate_dev = index_report(
        args.candidate_development_report, development, "candidate development"
    )
    base_dev_report, base_dev = index_report(
        args.base_development_report, development, "base development"
    )
    candidate_retention_report, candidate_retention = index_report(
        args.candidate_retention_report, retention, "candidate retention"
    )
    base_retention_report, base_retention = index_report(
        args.base_retention_report, retention, "base retention"
    )
    if candidate_dev_report.get("engine") == base_dev_report.get("engine"):
        raise SystemExit("candidate and base development reports identify the same engine")
    if candidate_retention_report.get("engine") != candidate_dev_report.get("engine"):
        raise SystemExit("candidate engine identity differs across development and retention")
    if base_retention_report.get("engine") != base_dev_report.get("engine"):
        raise SystemExit("base engine identity differs across development and retention")

    exact_revision = validate_bundle(
        args.candidate_bundle, args.supervised_checkpoint, args.direction
    )
    if candidate_dev_report.get("modelRevision") != exact_revision or candidate_retention_report.get("modelRevision") != exact_revision:
        raise SystemExit("candidate reports are not bound to the exact supervised bundle")

    reviewers, assignments = assignment_map(args.sealed_assignments, set(development))
    reviewer_a, reviews_a = response_map(args.human_review_a, set(development))
    reviewer_b, reviews_b = response_map(args.human_review_b, set(development))
    if {reviewer_a, reviewer_b} != reviewers or reviewer_a == reviewer_b:
        raise SystemExit("human reviewer identities do not match sealed assignments")
    human_deltas: list[float] = []
    critical_cases: set[str] = set()
    for reviewer, responses in ((reviewer_a, reviews_a), (reviewer_b, reviews_b)):
        for case_id, response in responses.items():
            assignment = assignments[(reviewer, case_id)]
            expected = {
                "candidate": str(candidate_dev[case_id]["hypothesis"]),
                "base": str(base_dev[case_id]["hypothesis"]),
            }
            for label in ("A", "B"):
                engine = assignment[f"output{label}Engine"]
                if assignment[f"output{label}SHA256"] != text_hash(expected[engine]):
                    raise SystemExit(f"sealed output hash mismatch: {reviewer}/{case_id}/{label}")
            decoded = {
                assignment["outputAEngine"]: response["outputA"],
                assignment["outputBEngine"]: response["outputB"],
            }
            human_deltas.append(float(human_total(decoded["candidate"]) - human_total(decoded["base"])))
            if decoded["candidate"]["criticalError"]:
                critical_cases.add(case_id)

    metric_deltas = [
        sentence_chrf(candidate_dev[case_id]) - sentence_chrf(base_dev[case_id])
        for case_id in sorted(development)
    ]
    metric_interval = bootstrap(
        metric_deltas, args.paired_bootstrap_samples, args.confidence, args.seed
    )
    human_interval = bootstrap(
        human_deltas, args.paired_bootstrap_samples, args.confidence, args.seed + 1
    )
    candidate_retention_score = corpus_chrf(candidate_retention)
    base_retention_score = corpus_chrf(base_retention)
    retention_delta = candidate_retention_score - base_retention_score
    gates = [
        gate("reviewed-development-chrf-win", metric_interval["lower"] > 0, metric_interval, "paired lower confidence bound > 0"),
        gate("blind-human-development-win", human_interval["lower"] > 0, human_interval, "paired lower confidence bound > 0"),
        gate("no-new-critical-errors", not critical_cases, sorted(critical_cases), []),
        gate("general-retention", retention_delta >= -args.maximum_retention_regression, retention_delta, f">= {-args.maximum_retention_regression}"),
        gate("exact-bundle-checkpoint-binding", True, exact_revision, sha256(args.supervised_checkpoint / "model.safetensors")),
    ]
    approved = all(value["passed"] for value in gates)
    output = {
        "schemaVersion": 1,
        "status": "supervised-win-approved" if approved else "supervised-win-rejected",
        "approved": approved,
        "direction": args.direction,
        "candidateEngine": candidate_dev_report["engine"],
        "baseEngine": base_dev_report["engine"],
        "candidateModelRevision": exact_revision,
        "supervisedCheckpoint": {
            "path": str(args.supervised_checkpoint),
            "modelSHA256": sha256(args.supervised_checkpoint / "model.safetensors"),
        },
        "development": {
            "cases": len(development),
            "candidateChrFPlusPlus": corpus_chrf(candidate_dev),
            "baseChrFPlusPlus": corpus_chrf(base_dev),
            "pairedChrFDelta": metric_interval,
            "pairedHumanScoreDelta": human_interval,
            "candidateCriticalCaseIDs": sorted(critical_cases),
        },
        "retention": {
            "cases": len(retention),
            "candidateChrFPlusPlus": candidate_retention_score,
            "baseChrFPlusPlus": base_retention_score,
            "delta": retention_delta,
        },
        "gates": gates,
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in {
                "developmentSuite": args.development_suite,
                "candidateDevelopmentReport": args.candidate_development_report,
                "baseDevelopmentReport": args.base_development_report,
                "retentionSuite": args.retention_suite,
                "candidateRetentionReport": args.candidate_retention_report,
                "baseRetentionReport": args.base_retention_report,
                "sealedAssignments": args.sealed_assignments,
                "humanReviewA": args.human_review_a,
                "humanReviewB": args.human_review_b,
                "candidateBundleManifest": args.candidate_bundle / "manifest.json",
            }.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if approved else 2)


if __name__ == "__main__":
    main()
