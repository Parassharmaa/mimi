#!/usr/bin/env python3
"""Compare two case-level learned-metric reports with paired bootstrap intervals."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values: list[float]) -> float:
    if not values:
        raise SystemExit("cannot average an empty comparison slice")
    return sum(values) / len(values)


def bootstrap_interval(
    values: list[float],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    if not values:
        raise SystemExit("cannot bootstrap an empty comparison slice")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    count = len(values)
    estimates = sorted(
        mean([values[rng.randrange(count)] for _ in range(count)])
        for _ in range(samples)
    )
    tail = (1.0 - confidence) / 2.0
    lower = estimates[min(int(tail * samples), samples - 1)]
    upper = estimates[min(int((1.0 - tail) * samples), samples - 1)]
    return lower, upper


def validate_reports(candidate: dict, baseline: dict) -> list[tuple[dict, dict]]:
    immutable_fields = (
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
        "suiteSHA256",
    )
    for field in immutable_fields:
        if candidate.get(field) != baseline.get(field):
            raise SystemExit(f"learned-metric reports disagree on {field}")
    candidate_rows = {
        str(row.get("caseID", "")): row for row in candidate.get("results", [])
    }
    baseline_rows = {
        str(row.get("caseID", "")): row for row in baseline.get("results", [])
    }
    if (
        not candidate_rows
        or len(candidate_rows) != len(candidate.get("results", []))
        or len(baseline_rows) != len(baseline.get("results", []))
        or set(candidate_rows) != set(baseline_rows)
    ):
        raise SystemExit("reports must contain identical unique non-empty case IDs")
    pairs: list[tuple[dict, dict]] = []
    for case_id in sorted(candidate_rows):
        candidate_row = candidate_rows[case_id]
        baseline_row = baseline_rows[case_id]
        for field in ("sourceLanguage", "targetLanguage", "domain"):
            if candidate_row.get(field) != baseline_row.get(field):
                raise SystemExit(f"reports disagree on {field}: {case_id}")
        for row in (candidate_row, baseline_row):
            if not isinstance(row.get("score"), (int, float)):
                raise SystemExit(f"report lacks a numeric score: {case_id}")
        pairs.append((candidate_row, baseline_row))
    return pairs


def summarize(
    pairs: list[tuple[dict, dict]],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> dict:
    deltas = [float(candidate["score"]) - float(baseline["score"]) for candidate, baseline in pairs]
    candidate_scores = [float(candidate["score"]) for candidate, _ in pairs]
    baseline_scores = [float(baseline["score"]) for _, baseline in pairs]
    lower, upper = bootstrap_interval(
        deltas,
        samples=samples,
        confidence=confidence,
        seed=seed,
    )
    return {
        "cases": len(pairs),
        "candidateMeanScore": mean(candidate_scores),
        "baselineMeanScore": mean(baseline_scores),
        "meanPairedDelta": mean(deltas),
        "pairedBootstrapInterval": {"lower": lower, "upper": upper},
    }


def build_report(
    candidate_path: Path,
    baseline_path: Path,
    candidate: dict,
    baseline: dict,
    pairs: list[tuple[dict, dict]],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> dict:
    directions: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    domains: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for candidate_row, baseline_row in pairs:
        direction = f"{candidate_row['sourceLanguage']}>{candidate_row['targetLanguage']}"
        directions[direction].append((candidate_row, baseline_row))
        domains[f"{direction}/{candidate_row['domain']}"].append(
            (candidate_row, baseline_row)
        )
    return {
        "schemaVersion": 1,
        "metric": candidate["metric"],
        "signatureSHA256": candidate["signatureSHA256"],
        "suiteSHA256": candidate["suiteSHA256"],
        "candidateEngine": candidate.get("engine"),
        "baselineEngine": baseline.get("engine"),
        "candidateReportSHA256": sha256(candidate_path),
        "baselineReportSHA256": sha256(baseline_path),
        "bootstrap": {
            "samples": samples,
            "confidence": confidence,
            "seed": seed,
            "method": "paired-case-resampling-with-replacement",
        },
        "directions": {
            key: summarize(
                value,
                samples=samples,
                confidence=confidence,
                seed=seed,
            )
            for key, value in sorted(directions.items())
        },
        "domains": {
            key: summarize(
                value,
                samples=samples,
                confidence=confidence,
                seed=seed,
            )
            for key, value in sorted(domains.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20_260_718)
    args = parser.parse_args()

    if args.output.exists() and args.output.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    if args.bootstrap_samples < 1:
        raise SystemExit("bootstrap-samples must be positive")
    if not 0.0 < args.confidence < 1.0:
        raise SystemExit("confidence must be between zero and one")
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    pairs = validate_reports(candidate, baseline)
    report = build_report(
        args.candidate,
        args.baseline,
        candidate,
        baseline,
        pairs,
        samples=args.bootstrap_samples,
        confidence=args.confidence,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "metric": report["metric"]}))


if __name__ == "__main__":
    main()
