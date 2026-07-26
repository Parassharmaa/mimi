#!/usr/bin/env python3
"""Build the fail-closed result for the canonical-target v3 student screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import sacrebleu

from evaluate_gpt56_student_continuation import (
    computed_structure,
    generation_failures,
)


ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "Research/translation/results"
M = ROOT / "Research/translation/models"
DIRECTIONS = {
    "en-ja": ("en-US", "ja-JP"),
    "ja-en": ("ja-JP", "en-US"),
}


def digest(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"missing result input: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path: Path) -> dict:
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": digest(path),
    }


def load(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid JSON input {path}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def index(report: dict) -> dict[str, dict]:
    rows = {str(row.get("caseID", "")): row for row in report.get("results", [])}
    if not rows or len(rows) != len(report.get("results", [])):
        raise SystemExit("report has missing or duplicate case IDs")
    return rows


def metric(rows: list[dict]) -> dict:
    hypotheses = [str(row["hypothesis"]) for row in rows]
    references = [[str(row["references"][0]) for row in rows]]
    chrf = sacrebleu.metrics.CHRF(word_order=2).corpus_score(
        hypotheses, references
    ).score
    bleu = sacrebleu.metrics.BLEU(tokenize="intl").corpus_score(
        hypotheses, references
    ).score
    latencies = sorted(
        float(value)
        for row in rows
        for value in (row.get("warmLatencySeconds") or [row["latencySeconds"]])
    )
    return {
        "cases": len(rows),
        "chrfPlusPlus": chrf,
        "sacreBLEUIntl": bleu,
        "warmP50Seconds": latencies[(len(latencies) - 1) // 2],
        "warmP95Seconds": latencies[
            min(len(latencies) - 1, int(len(latencies) * 0.95))
        ],
    }


def paired_chrf(
    candidate: list[dict],
    baseline: list[dict],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> dict:
    candidate_by_id = {row["caseID"]: row for row in candidate}
    baseline_by_id = {row["caseID"]: row for row in baseline}
    if set(candidate_by_id) != set(baseline_by_id):
        raise SystemExit("candidate and baseline case IDs differ")
    deltas = []
    for case_id in sorted(candidate_by_id):
        current = candidate_by_id[case_id]
        previous = baseline_by_id[case_id]
        if current["references"] != previous["references"]:
            raise SystemExit(f"candidate and baseline references differ: {case_id}")
        references = current["references"]
        deltas.append(
            sacrebleu.sentence_chrf(
                current["hypothesis"], references, word_order=2
            ).score
            - sacrebleu.sentence_chrf(
                previous["hypothesis"], references, word_order=2
            ).score
        )
    rng = random.Random(seed)
    estimates = sorted(
        sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        for _ in range(samples)
    )
    tail = (1.0 - confidence) / 2.0
    return {
        "cases": len(deltas),
        "meanDelta": sum(deltas) / len(deltas),
        "pairedBootstrapInterval": {
            "confidence": confidence,
            "lower": estimates[min(samples - 1, int(samples * tail))],
            "upper": estimates[
                min(samples - 1, int(samples * (1.0 - tail)))
            ],
        },
        "samples": samples,
        "seed": seed,
    }


def filtered(report: dict, direction: tuple[str, str]) -> list[dict]:
    rows = [
        row
        for row in report["results"]
        if (row["sourceLanguage"], row["targetLanguage"]) == direction
    ]
    if not rows:
        raise SystemExit(f"report has no {direction} rows")
    return rows


def gate(name: str, passed: bool, actual: object, required: object) -> dict:
    return {
        "name": name,
        "passed": passed,
        "actual": actual,
        "required": required,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and output.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {output}")

    contract_path = (
        ROOT / "Research/translation/canonical-target-student-v3-contract-2026-07-25.json"
    )
    contract = load(contract_path)
    if contract.get("status") != "preregistered-ready-for-training":
        raise SystemExit("student contract is not the frozen preregistration")

    paths = {
        "candidate_documents": R
        / "development-accuracy-v1-canonical-v3-step250.json",
        "candidate_segments": R
        / "development-accuracy-v1-canonical-v3-step250-segments.json",
        "candidate_direct": R
        / "development-accuracy-v1-canonical-v3-step250-direct-under-192.json",
        "candidate_comet": R
        / "development-accuracy-v1-canonical-v3-step250-comet22.json",
        "comet_comparison": R
        / "development-accuracy-v1-canonical-v3-step250-vs-current-comet22.json",
        "candidate_structure": R
        / "development-accuracy-v1-canonical-v3-step250-structure-audit.json",
        "candidate_typed": R
        / "development-accuracy-v1-canonical-v3-step250-typed-critical.json",
        "baseline_documents": R
        / "development-accuracy-v1-candidate-clean-pair.json",
        "baseline_segments": R
        / "development-accuracy-v1-candidate-clean-pair-segments.json",
        "baseline_direct": R
        / "development-accuracy-v1-candidate-clean-pair-direct-under-192.json",
        "baseline_comet": R
        / "development-accuracy-v1-candidate-clean-pair-comet22.json",
        "baseline_structure": R
        / "development-accuracy-v1-candidate-clean-pair-structure-audit.json",
    }
    reports = {name: load(path) for name, path in paths.items()}

    candidate_docs = index(reports["candidate_documents"])
    baseline_docs = index(reports["baseline_documents"])
    candidate_direct = index(reports["candidate_direct"])
    baseline_direct = index(reports["baseline_direct"])
    candidate_structure = computed_structure(candidate_docs)
    baseline_structure = computed_structure(baseline_docs)
    new_exact = candidate_structure["exact"] - baseline_structure["exact"]
    new_typed = candidate_structure["typed"] - baseline_structure["typed"]
    new_negation = candidate_structure["negation"] - baseline_structure["negation"]
    candidate_union = (
        candidate_structure["exact"]
        | candidate_structure["typed"]
        | candidate_structure["negation"]
    )
    baseline_union = (
        baseline_structure["exact"]
        | baseline_structure["typed"]
        | baseline_structure["negation"]
    )
    new_union = candidate_union - baseline_union
    candidate_generation = generation_failures(
        reports["candidate_documents"], candidate_docs, "segment-then-join"
    ) | generation_failures(
        reports["candidate_direct"], candidate_direct, "direct"
    )
    baseline_generation = generation_failures(
        reports["baseline_documents"], baseline_docs, "segment-then-join"
    ) | generation_failures(
        reports["baseline_direct"], baseline_direct, "direct"
    )
    new_generation = candidate_generation - baseline_generation

    comet_comparison = reports["comet_comparison"]["directions"]
    direction_results = {}
    direction_passes = []
    for name, language_pair in DIRECTIONS.items():
        candidate_document_rows = filtered(
            reports["candidate_documents"], language_pair
        )
        baseline_document_rows = filtered(
            reports["baseline_documents"], language_pair
        )
        candidate_segment_rows = filtered(
            reports["candidate_segments"], language_pair
        )
        baseline_segment_rows = filtered(
            reports["baseline_segments"], language_pair
        )
        candidate_document_metric = metric(candidate_document_rows)
        baseline_document_metric = metric(baseline_document_rows)
        candidate_segment_metric = metric(candidate_segment_rows)
        baseline_segment_metric = metric(baseline_segment_rows)
        paired = paired_chrf(
            candidate_document_rows,
            baseline_document_rows,
            samples=10_000,
            confidence=0.90,
            seed=20_260_725,
        )
        comet = comet_comparison[
            f"{language_pair[0]}>{language_pair[1]}"
        ]
        chr_signal = paired["meanDelta"] >= 0.25
        comet_signal = comet["meanPairedDelta"] >= 0.002
        improvement_signals = int(chr_signal) + int(comet_signal)
        quality_gates = [
            gate(
                "minimum-two-improvement-signals",
                improvement_signals >= 2,
                {
                    "passedSignals": improvement_signals,
                    "chrF++": chr_signal,
                    "COMET-22": comet_signal,
                    "independentJudge": "not-run-decision-already-fixed",
                },
                2,
            ),
            gate(
                "chrF++-paired-90pct-lower",
                paired["pairedBootstrapInterval"]["lower"] >= -0.25,
                paired["pairedBootstrapInterval"]["lower"],
                -0.25,
            ),
            gate(
                "COMET-22-paired-90pct-lower",
                comet["pairedBootstrapInterval"]["lower"] >= -0.005,
                comet["pairedBootstrapInterval"]["lower"],
                -0.005,
            ),
            gate(
                "BLEU-corpus-regression",
                candidate_document_metric["sacreBLEUIntl"]
                >= baseline_document_metric["sacreBLEUIntl"] - 0.10,
                candidate_document_metric["sacreBLEUIntl"]
                - baseline_document_metric["sacreBLEUIntl"],
                ">= -0.10",
            ),
            gate(
                "warm-segment-p95",
                candidate_segment_metric["warmP95Seconds"] <= 0.175,
                candidate_segment_metric["warmP95Seconds"],
                "<= 0.175",
            ),
        ]
        direction_pass = all(value["passed"] for value in quality_gates)
        direction_passes.append(direction_pass)
        direction_results[name] = {
            "passed": direction_pass,
            "document": {
                "candidate": candidate_document_metric,
                "baseline": baseline_document_metric,
                "pairedChrFPlusPlus": paired,
            },
            "segments": {
                "candidate": candidate_segment_metric,
                "baseline": baseline_segment_metric,
            },
            "comet22": comet,
            "gates": quality_gates,
        }

    q4_paths = {
        "en-ja": M
        / "elanmt-canonical-target-scale-v3-en-ja-step250-mlx-4bit",
        "ja-en": M
        / "elanmt-canonical-target-scale-v3-ja-en-step250-mlx-4bit",
    }
    q4 = {}
    bundle_bytes = 0
    distribution_statuses = {}
    for name, path in q4_paths.items():
        manifest_path = path / "manifest.json"
        manifest = load(manifest_path)
        if manifest.get("bits") != 4 or manifest.get("group_size") != 64:
            raise SystemExit(f"{name} is not exact q4/group-64")
        size = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        bundle_bytes += size
        status = manifest.get("training_data", {}).get("distribution_status")
        distribution_statuses[name] = status
        q4[name] = {
            "directory": str(path.relative_to(ROOT)),
            "bytes": size,
            "manifest": record(manifest_path),
            "weights": record(path / "model.safetensors"),
            "sourceWeightsSha256": manifest.get("source_weights_sha256"),
            "distributionStatus": status,
        }

    peak_rss = max(
        int(reports[name].get("peakResidentBytes", 0))
        for name in ("candidate_documents", "candidate_segments", "candidate_direct")
    )
    global_gates = [
        gate("new-union-critical-errors", not new_union, sorted(new_union), 0),
        gate("new-negation-errors", not new_negation, sorted(new_negation), 0),
        gate("new-typed-critical-errors", not new_typed, sorted(new_typed), 0),
        gate(
            "new-repetition-or-generation-limit-errors",
            not new_generation,
            sorted(new_generation),
            0,
        ),
        gate("peak-resident-bytes", peak_rss <= 250_000_000, peak_rss, 250_000_000),
        gate(
            "preferred-bundle-bytes",
            bundle_bytes <= 150_000_000,
            bundle_bytes,
            150_000_000,
        ),
        gate(
            "hard-bundle-bytes",
            bundle_bytes <= 500_000_000,
            bundle_bytes,
            500_000_000,
        ),
        gate(
            "distribution-provenance",
            all(
                value not in (None, "provenance-incomplete-not-approved-for-distribution")
                for value in distribution_statuses.values()
            ),
            distribution_statuses,
            "complete distributable provenance and attribution review",
        ),
    ]
    continue_training = (
        all(direction_passes) and all(value["passed"] for value in global_gates)
    )
    if continue_training:
        raise SystemExit("unexpected pass: this report is the registered stop result")

    training_paths = {
        "en-ja": M / "elanmt-canonical-target-scale-v3-en-ja-step250",
        "ja-en": M
        / "elanmt-canonical-target-scale-v3-ja-en-step250-checkpoints/step-0000250",
    }
    training = {}
    for name, path in training_paths.items():
        manifest_path = path / "mimi_training_manifest.json"
        manifest = load(manifest_path)
        history = manifest.get("history", [])
        if len(history) != 2 or history[0].get("step") != 0 or history[1].get("step") != 250:
            raise SystemExit(f"{name} training manifest lacks steps 0 and 250")
        training[name] = {
            "directory": str(path.relative_to(ROOT)),
            "weights": record(path / "model.safetensors"),
            "manifest": record(manifest_path),
            "history": history,
            "fullPrecisionChrFPlusPlusDelta": (
                float(history[1]["chrf_pp"]) - float(history[0]["chrf_pp"])
            ),
        }

    report = {
        "schemaVersion": 1,
        "experiment": contract["experiment"],
        "status": "phase-1-continuation-rejected",
        "continueTraining": False,
        "promotionAuthorized": False,
        "appChangeAuthorized": False,
        "publicUploadAuthorized": False,
        "teacherAuthentication": contract["teacher_authentication"],
        "privateReasoningTracesUsed": False,
        "contract": record(contract_path),
        "training": training,
        "q4": q4,
        "bundleBytes": bundle_bytes,
        "peakResidentBytes": peak_rss,
        "directions": direction_results,
        "safety": {
            "baselineExactFailures": len(baseline_structure["exact"]),
            "candidateExactFailures": len(candidate_structure["exact"]),
            "newExactCaseIDs": sorted(new_exact),
            "baselineTypedFailures": len(baseline_structure["typed"]),
            "candidateTypedFailures": len(candidate_structure["typed"]),
            "newTypedCaseIDs": sorted(new_typed),
            "baselineNegationFailures": len(baseline_structure["negation"]),
            "candidateNegationFailures": len(candidate_structure["negation"]),
            "newNegationCaseIDs": sorted(new_negation),
            "baselineUnionFailures": len(baseline_union),
            "candidateUnionFailures": len(candidate_union),
            "newUnionCaseIDs": sorted(new_union),
            "baselineGenerationFailures": len(baseline_generation),
            "candidateGenerationFailures": len(candidate_generation),
            "newGenerationFailureIDs": sorted(new_generation),
        },
        "globalGates": global_gates,
        "independentQualityJudge": {
            "run": False,
            "reason": (
                "JA-to-EN missed both pre-judge improvement signals; one remaining "
                "judge signal cannot reach the required two-of-three gate"
            ),
        },
        "decision": (
            "Stop this 250-step recipe. Preserve the current Mimi translator; do not "
            "continue training, integrate, publish, or claim bidirectional improvement."
        ),
        "inputs": {name: record(path) for name, path in paths.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record(output), indent=2))


if __name__ == "__main__":
    main()
