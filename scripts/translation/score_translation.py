#!/usr/bin/env python3
"""Score one benchmark report or compare two authenticated reports."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
from pathlib import Path

import sacrebleu


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def suite_content_digest(results: list[dict]) -> str:
    """Identify the aligned benchmark content without hashing hypotheses/timings."""
    fields = (
        "caseID",
        "sourceLanguage",
        "targetLanguage",
        "domain",
        "source",
        "references",
        "claimEligible",
    )
    cases = [
        {field: row.get(field) for field in fields}
        for row in sorted(results, key=lambda row: row["caseID"])
    ]
    payload = json.dumps(
        cases, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def direction(result: dict) -> str:
    return f"{result['sourceLanguage']}>{result['targetLanguage']}"


def score(results: list[dict]) -> dict:
    hypotheses = [row["hypothesis"] for row in results]
    reference_count = max(len(row["references"]) for row in results)
    references = [
        [row["references"][min(index, len(row["references"]) - 1)] for row in results]
        for index in range(reference_count)
    ]
    latencies = sorted(
        latency
        for row in results
        for latency in (row.get("warmLatencySeconds") or [row["latencySeconds"]])
    )
    def percentile(proportion: float) -> float:
        return latencies[int((len(latencies) - 1) * proportion + 0.999999)]
    chrf = sacrebleu.metrics.CHRF(word_order=2)
    bleu = sacrebleu.metrics.BLEU(tokenize="intl")
    chrf_score = chrf.corpus_score(hypotheses, references).score
    bleu_score = bleu.corpus_score(hypotheses, references).score
    return {
        "cases": len(results),
        "claimEligibleCases": sum(bool(row["claimEligible"]) for row in results),
        "chrfPlusPlus": chrf_score,
        "sacreBLEUIntl": bleu_score,
        "p50LatencySeconds": percentile(0.50),
        "p95LatencySeconds": percentile(0.95),
    }


def compare(candidate: list[dict], baseline: list[dict], samples: int, seed: int) -> dict:
    candidate_by_id = {row["caseID"]: row for row in candidate}
    baseline_by_id = {row["caseID"]: row for row in baseline}
    if len(candidate_by_id) != len(candidate) or len(baseline_by_id) != len(baseline):
        raise SystemExit("reports contain duplicate case IDs")
    if set(candidate_by_id) != set(baseline_by_id):
        raise SystemExit(
            "candidate and baseline reports must contain exactly the same case IDs"
        )
    common_ids = sorted(candidate_by_id)
    deltas: list[float] = []
    for case_id in common_ids:
        candidate_row = candidate_by_id[case_id]
        baseline_row = baseline_by_id[case_id]
        for field in (
            "sourceLanguage", "targetLanguage", "domain", "source", "references", "claimEligible"
        ):
            if candidate_row.get(field) != baseline_row.get(field):
                raise SystemExit(
                    f"candidate and baseline reports disagree on {field}: {case_id}"
                )
        references = candidate_row["references"]
        candidate_score = sacrebleu.sentence_chrf(
            candidate_row["hypothesis"], references, word_order=2
        ).score
        baseline_score = sacrebleu.sentence_chrf(
            baseline_row["hypothesis"], references, word_order=2
        ).score
        deltas.append(candidate_score - baseline_score)
    if not deltas:
        raise SystemExit("reports have no aligned cases")
    rng = random.Random(seed)
    boot = sorted(
        sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        for _ in range(samples)
    )
    return {
        "alignedCases": len(deltas),
        "meanSentenceChrFPlusPlusDelta": sum(deltas) / len(deltas),
        "pairedBootstrap95Lower": boot[int(samples * 0.025)],
        "pairedBootstrap95Upper": boot[min(samples - 1, int(samples * 0.975))],
        "bootstrapSamples": samples,
        "bootstrapSeed": seed,
    }


def scoring_contract(results: list[dict]) -> dict:
    hypotheses = [row["hypothesis"] for row in results]
    references = [[row["references"][0] for row in results]]
    chrf = sacrebleu.metrics.CHRF(word_order=2)
    bleu = sacrebleu.metrics.BLEU(tokenize="intl")
    chrf.corpus_score(hypotheses, references)
    bleu.corpus_score(hypotheses, references)
    return {
        "sacrebleuVersion": importlib.metadata.version("sacrebleu"),
        "chrfPlusPlusSignature": str(chrf.get_signature()),
        "bleuIntlSignature": str(bleu.get_signature()),
    }


def report_identity(path: Path, report: dict) -> dict:
    return {
        "path": str(path),
        "sha256": digest(path),
        "engine": report["engine"],
        "modelRevision": report.get("modelRevision"),
        "runtimeImplementation": report.get("runtimeImplementation"),
        "benchmarkConfiguration": report.get("benchmarkConfiguration"),
        "suiteContentSha256": suite_content_digest(report["results"]),
    }


def validate_matched_runtime(candidate: dict, baseline: dict) -> None:
    """Reject same-engine comparisons that do not share one runtime contract."""
    if candidate.get("engine") != baseline.get("engine"):
        return
    for field in (
        "benchmarkConfiguration",
        "decoderSelfKVCache",
        "positionEmbeddings",
    ):
        if candidate.get(field) != baseline.get(field):
            raise SystemExit(
                f"same-engine reports use different {field}; regenerate a matched control"
            )
    candidate_runtime = candidate.get("runtimeImplementation")
    baseline_runtime = baseline.get("runtimeImplementation")
    if candidate_runtime is None and baseline_runtime is None:
        return
    if candidate_runtime is None or baseline_runtime is None:
        raise SystemExit(
            "same-engine reports do not both authenticate runtimeImplementation; "
            "regenerate a matched control"
        )
    if candidate_runtime != baseline_runtime:
        raise SystemExit(
            "same-engine reports use different runtimeImplementation; "
            "regenerate a matched control"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument(
        "--compare-report",
        type=Path,
        help="compare against a generic baseline report",
    )
    comparison.add_argument(
        "--compare-apple",
        type=Path,
        help="compare specifically against an Apple Translation report",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.bootstrap_samples <= 0:
        raise SystemExit("--bootstrap-samples must be positive")

    report = load(args.report)
    baseline_path = args.compare_report or args.compare_apple
    baseline = load(baseline_path) if baseline_path else None
    comparison_key = "versusApple" if args.compare_apple else "versusBaseline"
    output = {
        "schemaVersion": 2,
        "engine": report["engine"],
        "modelRevision": report.get("modelRevision"),
        "preparationSeconds": report["preparationSeconds"],
        "peakResidentBytes": report.get("peakResidentBytes"),
        "modelBytes": report.get("modelBytes"),
        "candidateReport": report_identity(args.report, report),
        "scoringContract": scoring_contract(report["results"]),
        "directions": {},
    }
    if baseline is not None and baseline_path is not None:
        if args.compare_report is not None:
            validate_matched_runtime(report, baseline)
        baseline_identity = report_identity(baseline_path, baseline)
        if (
            output["candidateReport"]["suiteContentSha256"]
            != baseline_identity["suiteContentSha256"]
        ):
            raise SystemExit("candidate and baseline reports use different suite content")
        output["baselineReport"] = baseline_identity
        output["comparisonContract"] = {
            "label": comparison_key,
            "bootstrapSamples": args.bootstrap_samples,
            "bootstrapSeed": args.seed,
        }
    for name in sorted({direction(row) for row in report["results"]}):
        current = [row for row in report["results"] if direction(row) == name]
        direction_output = score(current)
        if baseline:
            baseline_current = [
                row for row in baseline["results"] if direction(row) == name
            ]
            direction_output[comparison_key] = compare(
                current, baseline_current, args.bootstrap_samples, args.seed
            )
        direction_output["domains"] = {}
        for domain in sorted({row["domain"] for row in current}):
            domain_rows = [row for row in current if row["domain"] == domain]
            domain_output = score(domain_rows)
            if baseline:
                baseline_domain_rows = [
                    row
                    for row in baseline_current
                    if row["domain"] == domain
                ]
                domain_output[comparison_key] = compare(
                    domain_rows,
                    baseline_domain_rows,
                    args.bootstrap_samples,
                    args.seed,
                )
            direction_output["domains"][domain] = domain_output
        output["directions"][name] = direction_output
    serialized = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
