#!/usr/bin/env python3
"""Seal the fail-closed protected result for the JA-to-EN Claude-5 v7 arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

import sacrebleu

from evaluate_gpt56_student_continuation import (
    computed_structure,
    generation_failures,
)


ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "Research/translation/results"
M = ROOT / "Research/translation/models"
CONTRACT = ROOT / "Research/translation/canonical-pairwise-v7-contract-v2-2026-07-25.json"
INTERNAL_RESULT = ROOT / "Research/translation/canonical-pairwise-v7-result-2026-07-25.json"
CANDIDATE_Q4 = M / "elanmt-canonical-pairwise-v7-ja-en-mlx-4bit"
CURRENT_EN_JA_Q4 = M / "elanmt-release-clean-full-depth-en-ja-v1-avg3-mlx-4bit"
REPORTS = {
    "candidate_documents": R / "development-accuracy-v1-canonical-v7-ja-en.json",
    "candidate_segments": R / "development-accuracy-v1-canonical-v7-ja-en-segments.json",
    "candidate_direct": R / "development-accuracy-v1-canonical-v7-ja-en-direct-under-192.json",
    "baseline_documents": R / "development-accuracy-v1-candidate-clean-pair.json",
    "baseline_segments": R / "development-accuracy-v1-candidate-clean-pair-segments.json",
    "baseline_direct": R / "development-accuracy-v1-candidate-clean-pair-direct-under-192.json",
}


def sha256(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"missing result input: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def record(path: Path) -> dict:
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def ja_en_rows(report: dict, label: str) -> list[dict]:
    rows = [
        row
        for row in report.get("results", [])
        if (row.get("sourceLanguage"), row.get("targetLanguage"))
        == ("ja-JP", "en-US")
    ]
    if not rows:
        raise SystemExit(f"{label} has no JA-to-EN rows")
    identifiers = [str(row.get("caseID", "")) for row in rows]
    if any(not value for value in identifiers) or len(identifiers) != len(set(identifiers)):
        raise SystemExit(f"{label} has invalid case IDs")
    return rows


def aligned(candidate: list[dict], baseline: list[dict], label: str) -> list[tuple[dict, dict]]:
    baseline_by_id = {row["caseID"]: row for row in baseline}
    if set(baseline_by_id) != {row["caseID"] for row in candidate}:
        raise SystemExit(f"{label} candidate/baseline case IDs differ")
    pairs = []
    for row in candidate:
        previous = baseline_by_id[row["caseID"]]
        for field in (
            "sourceLanguage",
            "targetLanguage",
            "domain",
            "source",
            "references",
            "claimEligible",
        ):
            if row.get(field) != previous.get(field):
                raise SystemExit(f"{label} disagrees on {field}: {row['caseID']}")
        pairs.append((row, previous))
    return pairs


def corpus_metrics(rows: list[dict]) -> dict:
    hypotheses = [str(row["hypothesis"]) for row in rows]
    references = [[str(row["references"][0]) for row in rows]]
    return {
        "cases": len(rows),
        "chrFPlusPlus": sacrebleu.corpus_chrf(
            hypotheses, references, word_order=2
        ).score,
        "sacreBLEUIntl": sacrebleu.corpus_bleu(
            hypotheses, references, tokenize="intl"
        ).score,
    }


def sentence_delta(pair: tuple[dict, dict]) -> float:
    candidate, baseline = pair
    return (
        sacrebleu.sentence_chrf(
            str(candidate["hypothesis"]), candidate["references"], word_order=2
        ).score
        - sacrebleu.sentence_chrf(
            str(baseline["hypothesis"]), baseline["references"], word_order=2
        ).score
    )


def paired_interval(deltas: list[float]) -> dict:
    samples = 10_000
    confidence = 0.90
    rng = random.Random(20_260_725)
    estimates = sorted(
        sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        for _ in range(samples)
    )
    tail = (1.0 - confidence) / 2.0
    return {
        "mean": sum(deltas) / len(deltas),
        "lower": estimates[min(samples - 1, int(samples * tail))],
        "upper": estimates[min(samples - 1, int(samples * (1.0 - tail)))],
        "confidence": confidence,
        "samples": samples,
        "seed": 20_260_725,
    }


def comparison(candidate: list[dict], baseline: list[dict], label: str) -> dict:
    pairs = aligned(candidate, baseline, label)
    deltas = [sentence_delta(pair) for pair in pairs]
    domains = {}
    for domain in sorted({row["domain"] for row in candidate}):
        domain_deltas = [
            sentence_delta(pair) for pair in pairs if pair[0]["domain"] == domain
        ]
        domains[domain] = {
            "cases": len(domain_deltas),
            "meanSentenceChrFPlusPlusDelta": sum(domain_deltas) / len(domain_deltas),
        }
    candidate_metrics = corpus_metrics(candidate)
    baseline_metrics = corpus_metrics(baseline)
    return {
        "candidate": candidate_metrics,
        "baseline": baseline_metrics,
        "pairedSentenceChrFPlusPlus": paired_interval(deltas),
        "sacreBLEUIntlDelta": (
            candidate_metrics["sacreBLEUIntl"] - baseline_metrics["sacreBLEUIntl"]
        ),
        "domains": domains,
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil((len(ordered) - 1) * fraction)]


def gate(name: str, passed: bool, actual: object, required: object) -> dict:
    return {"name": name, "passed": passed, "actual": actual, "required": required}


def safety_delta(
    candidate_reports: dict[str, dict],
    baseline_reports: dict[str, dict],
) -> dict:
    new_exact: set[str] = set()
    new_typed: set[str] = set()
    new_negation: set[str] = set()
    new_union: set[str] = set()
    new_generation: set[str] = set()
    modes = {
        "documents": "segment-then-join",
        "segments": "segments",
        "direct": "direct",
    }
    for label, mode in modes.items():
        candidate_report = candidate_reports[label]
        baseline_report = baseline_reports[label]
        candidate_rows = {
            row["caseID"]: row for row in ja_en_rows(candidate_report, f"candidate {label}")
        }
        baseline_rows = {
            row["caseID"]: row for row in ja_en_rows(baseline_report, f"baseline {label}")
        }
        if set(candidate_rows) != set(baseline_rows):
            raise SystemExit(f"safety {label} case IDs differ")
        current = computed_structure(candidate_rows)
        previous = computed_structure(baseline_rows)
        prefix = f"{label}:"
        new_exact.update(prefix + value for value in current["exact"] - previous["exact"])
        new_typed.update(prefix + value for value in current["typed"] - previous["typed"])
        new_negation.update(
            prefix + value for value in current["negation"] - previous["negation"]
        )
        current_union = current["exact"] | current["typed"] | current["negation"]
        previous_union = previous["exact"] | previous["typed"] | previous["negation"]
        new_union.update(prefix + value for value in current_union - previous_union)
        new_generation.update(
            generation_failures(candidate_report, candidate_rows, mode)
            - generation_failures(baseline_report, baseline_rows, mode)
        )
    return {
        "newExactCritical": sorted(new_exact),
        "newTypedCritical": sorted(new_typed),
        "newNegation": sorted(new_negation),
        "newUnionCritical": sorted(new_union),
        "newRepetitionOrGenerationLimit": sorted(new_generation),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite protected result: {output}")

    contract = load(CONTRACT)
    internal = load(INTERNAL_RESULT)
    if (
        contract.get("experiment") != "canonical-pairwise-preference-claude5-v7-ja-en"
        or contract.get("status") != "preregistered-ready-for-training"
        or internal.get("status") != "internal-preference-gate-passed"
        or internal.get("internal_gate", {}).get("passed") is not True
        or internal.get("quantization_authorized") is not True
    ):
        raise SystemExit("v7 internal gate does not authorize protected finalization")

    q4_manifest_path = CANDIDATE_Q4 / "manifest.json"
    q4_manifest = load(q4_manifest_path)
    expected_source_hash = internal["candidate"]["model"]["sha256"]
    if (
        q4_manifest.get("direction") != "ja-en"
        or q4_manifest.get("bits") != 4
        or q4_manifest.get("group_size") != 64
        or q4_manifest.get("source_weights_sha256") != expected_source_hash
    ):
        raise SystemExit("v7 q4 conversion does not match the selected checkpoint")

    reports = {name: load(path) for name, path in REPORTS.items()}
    candidates = {
        "documents": reports["candidate_documents"],
        "segments": reports["candidate_segments"],
        "direct": reports["candidate_direct"],
    }
    baselines = {
        "documents": reports["baseline_documents"],
        "segments": reports["baseline_segments"],
        "direct": reports["baseline_direct"],
    }
    document = comparison(
        ja_en_rows(candidates["documents"], "candidate documents"),
        ja_en_rows(baselines["documents"], "baseline documents"),
        "documents",
    )
    direct = comparison(
        ja_en_rows(candidates["direct"], "candidate direct"),
        ja_en_rows(baselines["direct"], "baseline direct"),
        "direct",
    )
    segment_rows = ja_en_rows(candidates["segments"], "candidate segments")
    segment_latencies = [
        float(value)
        for row in segment_rows
        for value in (row.get("warmLatencySeconds") or [row["latencySeconds"]])
    ]
    segment_p95 = percentile(segment_latencies, 0.95)
    peak_rss = max(
        int(candidates[name].get("peakResidentBytes", 0))
        for name in ("documents", "segments", "direct")
    )
    bundle_bytes = directory_bytes(CURRENT_EN_JA_Q4) + directory_bytes(CANDIDATE_Q4)
    safety = safety_delta(candidates, baselines)
    minimum_domain_delta = min(
        value["meanSentenceChrFPlusPlusDelta"]
        for value in document["domains"].values()
    )
    minimum_direct_domain_delta = min(
        value["meanSentenceChrFPlusPlusDelta"] for value in direct["domains"].values()
    )
    paired = document["pairedSentenceChrFPlusPlus"]
    chr_signal = paired["mean"] >= 0.25
    quality_gates = [
        gate("minimum-two-improvement-signals", False, {
            "passedSignals": int(chr_signal),
            "chrF++": chr_signal,
            "COMET-22": "not-run-after-fixed-rejection",
            "independentJudge": "not-run-after-fixed-rejection",
        }, 2),
        gate("mean-sentence-chrf++-signal", chr_signal, paired["mean"], 0.25),
        gate("chrf++-paired-90pct-lower", paired["lower"] >= -0.25, paired["lower"], -0.25),
        gate(
            "sacrebleu-corpus-regression",
            document["sacreBLEUIntlDelta"] >= -0.10,
            document["sacreBLEUIntlDelta"],
            ">= -0.10",
        ),
        gate(
            "maximum-domain-chrf++-regression",
            minimum_domain_delta >= -0.50,
            minimum_domain_delta,
            ">= -0.50",
        ),
        gate(
            "maximum-direct-chrf++-regression",
            direct["pairedSentenceChrFPlusPlus"]["mean"] >= -0.50,
            direct["pairedSentenceChrFPlusPlus"]["mean"],
            ">= -0.50",
        ),
        gate(
            "maximum-direct-domain-chrf++-regression",
            minimum_direct_domain_delta >= -0.50,
            minimum_direct_domain_delta,
            ">= -0.50",
        ),
    ]
    safety_gates = [
        gate("new-union-critical-errors", not safety["newUnionCritical"], safety["newUnionCritical"], 0),
        gate("new-negation-errors", not safety["newNegation"], safety["newNegation"], 0),
        gate("new-typed-critical-errors", not safety["newTypedCritical"], safety["newTypedCritical"], 0),
        gate(
            "new-repetition-or-generation-limit-errors",
            not safety["newRepetitionOrGenerationLimit"],
            safety["newRepetitionOrGenerationLimit"],
            0,
        ),
    ]
    operational_gates = [
        gate("warm-segment-p95-seconds", segment_p95 <= 0.175, segment_p95, 0.175),
        gate("peak-resident-bytes", peak_rss <= 250_000_000, peak_rss, 250_000_000),
        gate("preferred-two-direction-bundle-bytes", bundle_bytes <= 150_000_000, bundle_bytes, 150_000_000),
        gate("hard-two-direction-bundle-bytes", bundle_bytes <= 500_000_000, bundle_bytes, 500_000_000),
        gate(
            "complete-distribution-provenance",
            q4_manifest.get("training_data") is not None,
            q4_manifest.get("training_data"),
            "complete per-row provenance and attribution sidecar",
        ),
    ]
    all_gates = [*quality_gates, *safety_gates, *operational_gates]
    if all(value["passed"] for value in all_gates):
        raise SystemExit("unexpected v7 protected pass; run the full judge/COMET path")

    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "experiment": contract["experiment"],
        "status": "protected-promotion-gate-rejected",
        "promotion_authorized": False,
        "app_change_authorized": False,
        "bundle_creation_authorized": False,
        "public_upload_authorized": False,
        "current_app_bundle_changed": False,
        "contract": record(CONTRACT),
        "internal_result": record(INTERNAL_RESULT),
        "candidate_q4": {
            "directory": str(CANDIDATE_Q4.relative_to(ROOT)),
            "bytes": directory_bytes(CANDIDATE_Q4),
            "manifest": record(q4_manifest_path),
            "weights": record(CANDIDATE_Q4 / "model.safetensors"),
        },
        "would_be_two_direction_bundle_bytes": bundle_bytes,
        "evaluation": {
            "documents": document,
            "direct": direct,
            "warmSegmentP95Seconds": segment_p95,
            "peakResidentBytes": peak_rss,
            "safety": safety,
            "skippedAfterFixedRejection": [
                "COMET-22",
                "independent blinded quality judge",
            ],
        },
        "reports": {name: record(path) for name, path in REPORTS.items()},
        "gates": {
            "quality": quality_gates,
            "safety": safety_gates,
            "operationalAndDistribution": operational_gates,
        },
        "decision": (
            "Do not bundle or integrate v7. It passes internal preference, exact-q4, "
            "latency, memory, and size gates but fails mandatory protected quality, "
            "safety, and distribution-provenance gates."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "sha256": sha256(output), "status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
