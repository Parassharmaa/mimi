#!/usr/bin/env python3
"""Fail closed on every preregistered phase-1 GPT-5.6 student continuation gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import sacrebleu

from audit_translation_structures import critical_tokens, tokens
from typed_critical_token_policy import typed_preserves


DIRECTIONS = {
    "en-ja": ("en-US", "ja-JP"),
    "ja-en": ("ja-JP", "en-US"),
}
REPORT_FIELDS = (
    "sourceLanguage",
    "targetLanguage",
    "domain",
    "source",
    "references",
    "claimEligible",
)
COMET_FIELDS = (
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
)
OPEN_MODEL_LICENSES = {
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "CC-BY-4.0",
    "CC-BY-SA-3.0",
    "CC-BY-SA-4.0",
    "MIT",
}


def sha256(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"missing required input: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid JSON input {path}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid JSONL input {path}: {error}") from error
    if not all(isinstance(row, dict) for row in rows):
        raise SystemExit(f"JSONL rows must be objects: {path}")
    return rows


def index_rows(rows: list[dict[str, Any]], field: str, label: str) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for row in rows:
        identifier = str(row.get(field, "")).strip()
        if not identifier or identifier in output:
            raise SystemExit(f"{label} has an empty or duplicate ID: {identifier}")
        output[identifier] = row
    if not output:
        raise SystemExit(f"{label} is empty")
    return output


def direction(row: dict[str, Any]) -> str:
    pair = (row.get("sourceLanguage"), row.get("targetLanguage"))
    for name, expected in DIRECTIONS.items():
        if pair == expected:
            return name
    raise SystemExit(f"unsupported translation direction: {pair}")


def validate_report(
    path: Path,
    suite: dict[str, dict],
    label: str,
) -> tuple[dict[str, Any], dict[str, dict]]:
    report = load(path)
    if report.get("schemaVersion") != 1:
        raise SystemExit(f"{label} has unsupported schema")
    indexed = index_rows(report.get("results", []), "caseID", label)
    if set(indexed) != set(suite):
        raise SystemExit(f"{label} must cover the exact requested suite")
    for case_id, expected in suite.items():
        row = indexed[case_id]
        for field in REPORT_FIELDS:
            if row.get(field) != expected.get(field):
                raise SystemExit(f"{label} disagrees with suite {field}: {case_id}")
        if not str(row.get("hypothesis", "")).strip():
            raise SystemExit(f"{label} has an empty hypothesis: {case_id}")
        warm = row.get("warmLatencySeconds")
        if not isinstance(warm, list) or not warm:
            raise SystemExit(f"{label} lacks a warm latency: {case_id}")
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= 0
            for value in warm
        ):
            raise SystemExit(f"{label} has an invalid warm latency: {case_id}")
    if not str(report.get("engine", "")).strip() or not str(
        report.get("modelRevision", "")
    ).strip():
        raise SystemExit(f"{label} lacks a pinned engine/model revision")
    return report, indexed


def validate_composed_report(
    report: dict[str, Any],
    report_path: Path,
    segment_report_path: Path,
    case_suite_path: Path,
    label: str,
) -> None:
    if (
        report.get("sourceSegmentReportSHA256") != sha256(segment_report_path)
        or report.get("caseSuiteSHA256") != sha256(case_suite_path)
        or report.get("documentAggregation", {}).get("strategy")
        != "translate-segments-independently-then-join-with-newline"
        or report.get("documentAggregation", {}).get("crossSegmentContext") is not False
    ):
        raise SystemExit(f"{label} is not bound to the supplied segment report/suite")
    if not str(report.get("engine", "")).endswith(":segment-then-join"):
        raise SystemExit(f"{label} does not identify segment-then-join mode")


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise SystemExit("cannot calculate percentile of an empty slice")
    ordered = sorted(values)
    return ordered[math.ceil((len(ordered) - 1) * fraction)]


def references(rows: list[dict]) -> list[list[str]]:
    count = max(len(row["references"]) for row in rows)
    return [
        [row["references"][min(index, len(row["references"]) - 1)] for row in rows]
        for index in range(count)
    ]


def corpus_metrics(rows: list[dict]) -> dict[str, float | int]:
    hypotheses = [str(row["hypothesis"]) for row in rows]
    refs = references(rows)
    return {
        "cases": len(rows),
        "chrFPlusPlus": float(
            sacrebleu.corpus_chrf(hypotheses, refs, word_order=2).score
        ),
        "sacreBLEUIntl": float(
            sacrebleu.corpus_bleu(hypotheses, refs, tokenize="intl").score
        ),
    }


def sentence_chrf(row: dict) -> float:
    return float(
        sacrebleu.sentence_chrf(
            str(row["hypothesis"]), row["references"], word_order=2
        ).score
    )


def bootstrap(
    values: list[float],
    *,
    samples: int,
    confidence: float,
    seed: int,
    clusters: list[str] | None = None,
) -> dict[str, float | int | str]:
    if not values or samples < 1 or not 0 < confidence < 1:
        raise SystemExit("invalid bootstrap inputs")
    rng = random.Random(seed)
    estimates: list[float] = []
    method = "paired-case-resampling-with-replacement"
    if clusters is None:
        for _ in range(samples):
            estimates.append(
                sum(values[rng.randrange(len(values))] for _ in values) / len(values)
            )
    else:
        if len(clusters) != len(values):
            raise SystemExit("cluster labels and values have different lengths")
        grouped: dict[str, list[float]] = defaultdict(list)
        for cluster, value in zip(clusters, values):
            grouped[cluster].append(value)
        keys = sorted(grouped)
        method = "paired-parent-document-cluster-resampling-with-replacement"
        for _ in range(samples):
            sampled = [
                item
                for _ in keys
                for item in grouped[keys[rng.randrange(len(keys))]]
            ]
            estimates.append(sum(sampled) / len(sampled))
    estimates.sort()
    alpha = (1 - confidence) / 2
    return {
        "mean": sum(values) / len(values),
        "lower": estimates[max(0, math.floor(samples * alpha))],
        "upper": estimates[min(samples - 1, math.ceil(samples * (1 - alpha)) - 1)],
        "samples": samples,
        "confidence": confidence,
        "method": method,
    }


def validate_comet(
    path: Path,
    engine_report_path: Path,
    suite_path: Path,
    engine: str,
    case_ids: set[str],
    label: str,
) -> tuple[dict[str, Any], dict[str, dict]]:
    report = load(path)
    if (
        report.get("schemaVersion") != 1
        or report.get("engine") != engine
        or report.get("engineReportSHA256") != sha256(engine_report_path)
        or report.get("suiteSHA256") != sha256(suite_path)
    ):
        raise SystemExit(f"{label} COMET report is not bound to its raw inputs")
    indexed = index_rows(report.get("results", []), "caseID", f"{label} COMET")
    if set(indexed) != case_ids:
        raise SystemExit(f"{label} COMET report must cover the exact case suite")
    for case_id, row in indexed.items():
        score = row.get("score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise SystemExit(f"{label} COMET score is invalid: {case_id}")
    return report, indexed


def computed_structure(rows: dict[str, dict]) -> dict[str, Any]:
    failures: dict[str, list[str]] = {}
    exact: set[str] = set()
    typed: set[str] = set()
    negation: set[str] = set()
    for case_id, row in rows.items():
        source = tokens(str(row["source"]))
        hypothesis = tokens(str(row["hypothesis"]))
        reasons = []
        if source["structural"] != hypothesis["structural"]:
            reasons.append("url-placeholder-markup")
        if source["numbers"] != hypothesis["numbers"]:
            reasons.append("atomic-number")
        if source["negative"] != hypothesis["negative"]:
            reasons.append("negation-marker")
            negation.add(case_id)
        if reasons:
            failures[case_id] = reasons
        if critical_tokens(str(row["source"])) != critical_tokens(
            str(row["hypothesis"])
        ):
            exact.add(case_id)
        if not typed_preserves(
            str(row["source"]),
            str(row["hypothesis"]),
            str(row["sourceLanguage"]),
            str(row["targetLanguage"]),
        ):
            typed.add(case_id)
    return {
        "failures": failures,
        "exact": exact,
        "typed": typed,
        "negation": negation,
    }


def validate_structure_audit(
    path: Path, rows: dict[str, dict], label: str
) -> dict[str, Any]:
    audit = load(path)
    computed = computed_structure(rows)
    recorded = {
        str(row.get("caseID", "")): sorted(str(value) for value in row.get("reasons", []))
        for row in audit.get("failures", [])
    }
    recorded_exact = {
        str(row.get("caseID", ""))
        for row in audit.get("exactCriticalTokenAudit", {}).get("failures", [])
    }
    if (
        audit.get("schemaVersion") != 1
        or audit.get("cases") != len(rows)
        or recorded
        != {
            case_id: sorted(reasons)
            for case_id, reasons in computed["failures"].items()
        }
        or recorded_exact != computed["exact"]
    ):
        raise SystemExit(f"{label} structure audit disagrees with the raw report")
    return computed


def validate_judge_scores(value: object, case_id: str, label: str) -> tuple[float, bool]:
    if not isinstance(value, dict):
        raise SystemExit(f"invalid automated judge scores: {case_id}/{label}")
    total = 0
    for name, maximum in (("adequacy", 4), ("fluency", 4), ("terminology", 2)):
        score = value.get(name)
        if (
            isinstance(score, bool)
            or not isinstance(score, int)
            or not 0 <= score <= maximum
        ):
            raise SystemExit(f"invalid automated judge score: {case_id}/{label}/{name}")
        total += score
    if not isinstance(value.get("criticalError"), bool) or not isinstance(
        value.get("errorTags"), list
    ):
        raise SystemExit(f"invalid automated critical verdict: {case_id}/{label}")
    return float(total), bool(value["criticalError"] or value["errorTags"])


def validate_quality_judge(
    path: Path,
    suite_path: Path,
    candidate_report_path: Path,
    baseline_report_path: Path,
    candidate: dict[str, dict],
    baseline: dict[str, dict],
    teacher_model: str,
) -> tuple[
    dict[str, Any],
    set[str],
    set[str],
    str,
    str,
    dict[str, float],
]:
    report = load(path)
    if (
        report.get("schemaVersion") != 1
        or report.get("purpose")
        != "blinded-automated-quality-and-critical-comparison"
        or report.get("suiteSHA256") != sha256(suite_path)
        or report.get("candidateReportSHA256") != sha256(candidate_report_path)
        or report.get("baselineReportSHA256") != sha256(baseline_report_path)
        or report.get("reasoningTracesStored") is not False
        or report.get("blinded") is not True
    ):
        raise SystemExit("automated critical judge metadata/hashes are invalid")
    model = str(report.get("judgeModel", "")).strip()
    family = str(report.get("judgeModelFamily", "")).strip()
    revision = str(report.get("judgeRevision", "")).strip()
    prompt_hash = str(report.get("promptSHA256", "")).strip()
    if (
        not model
        or not family
        or not revision
        or model == teacher_model
        or not re.fullmatch(r"[0-9a-f]{64}", prompt_hash)
    ):
        raise SystemExit("automated critical judge is not independent and pinned")
    rows = index_rows(report.get("results", []), "caseID", "automated critical judge")
    if set(rows) != set(candidate) or set(rows) != set(baseline):
        raise SystemExit("automated critical judge must cover the exact case suite")
    candidate_failures: set[str] = set()
    baseline_failures: set[str] = set()
    deltas: dict[str, float] = {}
    for case_id, row in rows.items():
        if (
            row.get("candidateHypothesisSHA256")
            != text_sha256(str(candidate[case_id]["hypothesis"]))
            or row.get("baselineHypothesisSHA256")
            != text_sha256(str(baseline[case_id]["hypothesis"]))
        ):
            raise SystemExit(f"automated critical judge output hash mismatch: {case_id}")
        candidate_score, candidate_critical = validate_judge_scores(
            row.get("candidate"), case_id, "candidate"
        )
        baseline_score, baseline_critical = validate_judge_scores(
            row.get("baseline"), case_id, "baseline"
        )
        deltas[case_id] = candidate_score - baseline_score
        if candidate_critical:
            candidate_failures.add(case_id)
        if baseline_critical:
            baseline_failures.add(case_id)
    return (
        report,
        candidate_failures,
        baseline_failures,
        model,
        family,
        deltas,
    )


def repeated_token_loop(ids: list[Any]) -> bool:
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in ids):
        raise SystemExit("output token IDs must be integers")
    for width in range(3, min(16, len(ids) // 3) + 1):
        for start in range(0, len(ids) - width * 3 + 1):
            phrase = ids[start : start + width]
            if (
                ids[start + width : start + width * 2] == phrase
                and ids[start + width * 2 : start + width * 3] == phrase
            ):
                return True
    return False


def generation_failures(
    report: dict[str, Any], rows: dict[str, dict], mode: str
) -> set[str]:
    maximum = int(report.get("benchmarkConfiguration", {}).get("maximumGeneratedTokens", 0))
    if maximum < 1:
        raise SystemExit(f"{mode} report lacks maximumGeneratedTokens")
    failures: set[str] = set()
    for case_id, row in rows.items():
        reasons: list[str] = []
        if row.get("failureReason") or row.get("runtimeAccepted") is False:
            reasons.append("runtime-failure")
        if int(row.get("emptySegmentCount", 0)) > 0:
            reasons.append("empty-segment")
        segment_hypotheses = row.get("segmentHypotheses")
        if isinstance(segment_hypotheses, list):
            normalized = [" ".join(str(value).split()) for value in segment_hypotheses]
            if any(
                left and left == right
                for left, right in zip(normalized, normalized[1:])
            ):
                reasons.append("adjacent-duplicate-segment")
        token_sequences = row.get("segmentOutputTokenIDs")
        if not isinstance(token_sequences, list):
            token_sequences = [row.get("outputTokenIDs", [])]
        for sequence in token_sequences:
            if not isinstance(sequence, list):
                raise SystemExit(f"{mode} has invalid token sequence: {case_id}")
            if len(sequence) >= maximum:
                reasons.append("generation-limit")
            if repeated_token_loop(sequence):
                reasons.append("repeated-token-loop")
        if reasons:
            failures.add(f"{mode}:{case_id}")
    return failures


def validate_dataset(
    path: Path,
    direction_name: str,
    protected_suite_hashes: set[str],
) -> tuple[dict[str, Any], set[str]]:
    manifest = load(path)
    policy = manifest.get("synthetic_policy", {})
    if (
        manifest.get("schema_version") != 1
        or manifest.get("direction") != direction_name
        or manifest.get("promotion_eligible") is not True
        or manifest.get("private_reasoning_traces_used") is not False
        or policy.get("review_status") != "two-judge-reference-anchored"
        or policy.get("source_only_provisional_rows_allowed") is not False
        or policy.get("licensed_reference_anchor_required") is not True
        or policy.get("same_source_human_anchor_required") is not True
        or policy.get("human_replay_per_synthetic") != 3
        or float(policy.get("actual_synthetic_fraction", 1)) > 0.25
    ):
        raise SystemExit(f"{direction_name} dataset violates the registered policy")
    base = path.parent
    all_rows: list[dict] = []
    train_rows: list[dict] = []
    for split in ("train", "valid"):
        record = manifest.get("outputs", {}).get(split, {})
        declared = Path(str(record.get("path", "")))
        candidates = [declared, base / declared.name]
        actual = next((item for item in candidates if item.is_file()), None)
        if actual is None or record.get("sha256") != sha256(actual):
            raise SystemExit(f"{direction_name} dataset {split} is not authenticated")
        rows = load_jsonl(actual)
        all_rows.extend(rows)
        if split == "train":
            train_rows = rows
    recorded_protected = {
        str(record.get("sha256", ""))
        for record in manifest.get("decontamination", {}).get(
            "protected_suites", []
        )
        if isinstance(record, dict)
    }
    if (
        not protected_suite_hashes.issubset(recorded_protected)
        or manifest.get("decontamination", {}).get(
            "train_validation_exact_overlap"
        )
        is not False
    ):
        raise SystemExit(
            f"{direction_name} dataset does not authenticate development decontamination"
        )
    for row in all_rows:
        if row.get("promotion_eligible") is not True:
            raise SystemExit(f"{direction_name} dataset contains ineligible rows")
        if not str(row.get("source_license") or row.get("license") or "").strip():
            raise SystemExit(f"{direction_name} dataset row lacks source license")
        if not str(
            row.get("source_provenance") or row.get("provenance") or ""
        ).strip():
            raise SystemExit(f"{direction_name} dataset row lacks source provenance")
    admission_judges = {
        str(model).strip()
        for row in train_rows
        for model in row.get("judge_model_ids", [])
        if str(model).strip()
    }
    if not admission_judges:
        raise SystemExit(f"{direction_name} dataset lacks admission-judge lineage")
    return manifest, admission_judges


def validate_checkpoint(
    path: Path,
    direction_name: str,
    dataset_manifest_path: Path,
    contract: dict[str, Any],
) -> str:
    weights = path / "model.safetensors"
    manifest_path = path / "mimi_training_manifest.json"
    manifest = load(manifest_path)
    expected_initial = contract["student"]["initial_checkpoints"][direction_name]["files"][
        "model.safetensors"
    ]["sha256"]
    phase = contract["student"]["phase_1"]
    hyperparameters = manifest.get("hyperparameters", {})
    expected_hyperparameters = {
        "seed": phase["seed"],
        "batch_size": phase["batch_size"],
        "gradient_accumulation": phase["gradient_accumulation"],
        "max_steps": phase["max_steps"],
        "learning_rate": phase["learning_rate"],
        "weight_decay": phase["weight_decay"],
        "warmup_steps": phase["warmup_steps"],
        "evaluation_steps": phase["evaluation_steps"],
        "max_source_tokens": phase["max_source_tokens"],
        "max_target_tokens": phase["max_target_tokens"],
        "frozen_base_kl_weight": phase["frozen_base_kl_weight"],
        "l2_to_base_weight": phase["l2_to_base_weight"],
        "mlx_fake_quantization_bits": None,
    }
    if (
        manifest.get("schema_version") != 1
        or manifest.get("direction") != direction_name
        or manifest.get("initial_checkpoint", {}).get("model_sha256")
        != expected_initial
        or manifest.get("dataset_manifest", {}).get("sha256")
        != sha256(dataset_manifest_path)
        or manifest.get("dataset_manifest", {}).get("outputs_authenticated") is not True
        or any(hyperparameters.get(key) != value for key, value in expected_hyperparameters.items())
    ):
        raise SystemExit(f"{direction_name} checkpoint violates phase-1 lineage/hyperparameters")
    return sha256(weights)


def validate_bundle(
    path: Path,
    checkpoint_hashes: dict[str, str],
    candidate_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    root_path = path / "manifest.json"
    root = load(root_path)
    if (
        root.get("format") != "mimi-mlx-marian-pair-v1"
        or root.get("engines") != ["en-ja", "ja-en"]
        or root.get("quantization") != {"bits": 4, "group_size": 64}
        or root.get("license") not in OPEN_MODEL_LICENSES
    ):
        raise SystemExit("candidate bundle is not an open-license q4 group-64 Marian pair")
    declared = root.get("files")
    if not isinstance(declared, dict) or not declared:
        raise SystemExit("candidate bundle has no authenticated file inventory")
    actual_relative = {
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file() and item != root_path
    }
    if actual_relative != set(declared):
        raise SystemExit("candidate bundle has missing or undeclared payload files")
    payload_bytes = 0
    for relative, record in declared.items():
        item = path / relative
        if (
            not isinstance(record, dict)
            or record.get("bytes") != item.stat().st_size
            or record.get("sha256") != sha256(item)
        ):
            raise SystemExit(f"candidate bundle integrity failure: {relative}")
        payload_bytes += item.stat().st_size
    for direction_name, checkpoint_hash in checkpoint_hashes.items():
        manifest = load(path / direction_name / "manifest.json")
        if (
            manifest.get("direction") != direction_name
            or manifest.get("bits") != 4
            or manifest.get("group_size") != 64
            or manifest.get("source_weights_sha256") != checkpoint_hash
        ):
            raise SystemExit(f"{direction_name} quantized model is not derived from phase 1")
    revision = f"pair-manifest-sha256:{sha256(root_path)}"
    for report in candidate_reports:
        if (
            report.get("modelRevision") != revision
            or report.get("modelBytes") != payload_bytes
        ):
            raise SystemExit("candidate report is not bound to the exact q4 bundle")
    return {
        "revision": revision,
        "payloadBytes": payload_bytes,
        "bundleBytes": payload_bytes + root_path.stat().st_size,
        "license": root["license"],
        "manifestSHA256": sha256(root_path),
    }


def gate(name: str, passed: bool, actual: Any, requirement: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "actual": actual,
        "requirement": requirement,
    }


def assert_contract_file(contract: dict, section: str, path: Path) -> None:
    record = contract["benchmarks"][section]
    if record.get("sha256") != sha256(path):
        raise SystemExit(f"input differs from preregistered benchmark: {section}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("case_suite", type=Path)
    parser.add_argument("segment_suite", type=Path)
    parser.add_argument("direct_suite", type=Path)
    parser.add_argument("baseline_document_report", type=Path)
    parser.add_argument("candidate_document_report", type=Path)
    parser.add_argument("baseline_segment_report", type=Path)
    parser.add_argument("candidate_segment_report", type=Path)
    parser.add_argument("baseline_direct_report", type=Path)
    parser.add_argument("candidate_direct_report", type=Path)
    parser.add_argument("baseline_comet_report", type=Path)
    parser.add_argument("candidate_comet_report", type=Path)
    parser.add_argument("baseline_structure_audit", type=Path)
    parser.add_argument("candidate_structure_audit", type=Path)
    parser.add_argument("automated_quality_judge_a", type=Path)
    parser.add_argument("automated_quality_judge_b", type=Path)
    parser.add_argument("candidate_bundle", type=Path)
    parser.add_argument("candidate_en_ja_checkpoint", type=Path)
    parser.add_argument("candidate_ja_en_checkpoint", type=Path)
    parser.add_argument("en_ja_dataset_manifest", type=Path)
    parser.add_argument("ja_en_dataset_manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--development-gate-use", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists() and args.output.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")

    contract = load(args.contract)
    if (
        contract.get("schema_version") != 1
        or contract.get("experiment")
        != "gpt56-final-translation-intact-marian-pilot-v1"
        or contract.get("promotion_authorized") is not False
        or contract.get("app_change_authorized") is not False
    ):
        raise SystemExit("invalid student experiment contract")
    assert_contract_file(contract, "development_cases", args.case_suite)
    assert_contract_file(contract, "development_segments", args.segment_suite)
    assert_contract_file(contract, "direct_within_limit_cases", args.direct_suite)
    for name, path in (
        ("development_baseline_document_report", args.baseline_document_report),
        ("development_baseline_segment_report", args.baseline_segment_report),
        ("development_baseline_direct_report", args.baseline_direct_report),
        ("development_baseline_comet_report", args.baseline_comet_report),
        ("development_baseline_structure_audit", args.baseline_structure_audit),
    ):
        assert_contract_file(contract, name, path)
    implementation = contract.get("implementation", {}).get(
        "scripts/translation/evaluate_gpt56_student_continuation.py", {}
    )
    if implementation.get("sha256") != sha256(Path(__file__)):
        raise SystemExit("continuation evaluator differs from the preregistered implementation")

    cases = index_rows(load_jsonl(args.case_suite), "id", "case suite")
    segments = index_rows(load_jsonl(args.segment_suite), "id", "segment suite")
    direct_cases = index_rows(load_jsonl(args.direct_suite), "id", "direct suite")
    if not set(direct_cases).issubset(cases):
        raise SystemExit("direct suite is not a subset of the frozen case suite")

    baseline_document_report, baseline_documents = validate_report(
        args.baseline_document_report, cases, "baseline document report"
    )
    candidate_document_report, candidate_documents = validate_report(
        args.candidate_document_report, cases, "candidate document report"
    )
    baseline_segment_report, baseline_segments = validate_report(
        args.baseline_segment_report, segments, "baseline segment report"
    )
    candidate_segment_report, candidate_segments = validate_report(
        args.candidate_segment_report, segments, "candidate segment report"
    )
    baseline_direct_report, baseline_direct = validate_report(
        args.baseline_direct_report, direct_cases, "baseline direct report"
    )
    candidate_direct_report, candidate_direct = validate_report(
        args.candidate_direct_report, direct_cases, "candidate direct report"
    )
    validate_composed_report(
        baseline_document_report,
        args.baseline_document_report,
        args.baseline_segment_report,
        args.case_suite,
        "baseline document report",
    )
    validate_composed_report(
        candidate_document_report,
        args.candidate_document_report,
        args.candidate_segment_report,
        args.case_suite,
        "candidate document report",
    )
    for label, report_set in (
        (
            "baseline",
            (baseline_document_report, baseline_segment_report, baseline_direct_report),
        ),
        (
            "candidate",
            (candidate_document_report, candidate_segment_report, candidate_direct_report),
        ),
    ):
        revisions = {str(report.get("modelRevision")) for report in report_set}
        hardware = {str(report.get("hardware")) for report in report_set}
        systems = {str(report.get("operatingSystem")) for report in report_set}
        if len(revisions) != 1 or len(hardware) != 1 or len(systems) != 1:
            raise SystemExit(f"{label} reports disagree on model/hardware/OS")
    if (
        candidate_document_report.get("hardware")
        != baseline_document_report.get("hardware")
        or candidate_document_report.get("operatingSystem")
        != baseline_document_report.get("operatingSystem")
    ):
        raise SystemExit("candidate and baseline were not measured on the same hardware/OS")
    expected_baseline_revision = contract["benchmarks"]["development_baseline"][
        "model_revision"
    ]
    if baseline_document_report.get("modelRevision") != expected_baseline_revision:
        raise SystemExit("baseline report is not the preregistered best local model")

    baseline_comet_report, baseline_comet = validate_comet(
        args.baseline_comet_report,
        args.baseline_document_report,
        args.case_suite,
        str(baseline_document_report["engine"]),
        set(cases),
        "baseline",
    )
    candidate_comet_report, candidate_comet = validate_comet(
        args.candidate_comet_report,
        args.candidate_document_report,
        args.case_suite,
        str(candidate_document_report["engine"]),
        set(cases),
        "candidate",
    )
    if any(
        candidate_comet_report.get(field) != baseline_comet_report.get(field)
        for field in COMET_FIELDS
    ):
        raise SystemExit("candidate and baseline COMET reports use different contracts")

    baseline_structure = validate_structure_audit(
        args.baseline_structure_audit, baseline_documents, "baseline"
    )
    candidate_structure = validate_structure_audit(
        args.candidate_structure_audit, candidate_documents, "candidate"
    )
    protected_suite_hashes = {sha256(args.case_suite), sha256(args.segment_suite)}
    _, en_ja_admission_judges = validate_dataset(
        args.en_ja_dataset_manifest, "en-ja", protected_suite_hashes
    )
    _, ja_en_admission_judges = validate_dataset(
        args.ja_en_dataset_manifest, "ja-en", protected_suite_hashes
    )
    admission_judges = en_ja_admission_judges | ja_en_admission_judges
    judge_a = validate_quality_judge(
        args.automated_quality_judge_a,
        args.case_suite,
        args.candidate_document_report,
        args.baseline_document_report,
        candidate_documents,
        baseline_documents,
        str(contract["teacher"]["requests"]["model"]),
    )
    judge_b = validate_quality_judge(
        args.automated_quality_judge_b,
        args.case_suite,
        args.candidate_document_report,
        args.baseline_document_report,
        candidate_documents,
        baseline_documents,
        str(contract["teacher"]["requests"]["model"]),
    )
    if (
        judge_a[3] == judge_b[3]
        or judge_a[4] == judge_b[4]
        or {judge_a[3], judge_b[3]} & admission_judges
    ):
        raise SystemExit(
            "evaluation judges must be distinct model families and disjoint from admission judges"
        )
    candidate_judge_critical = judge_a[1] | judge_b[1]
    baseline_judge_critical = judge_a[2] | judge_b[2]
    judge_deltas = {
        case_id: (judge_a[5][case_id] + judge_b[5][case_id]) / 2
        for case_id in cases
    }
    checkpoint_hashes = {
        "en-ja": validate_checkpoint(
            args.candidate_en_ja_checkpoint,
            "en-ja",
            args.en_ja_dataset_manifest,
            contract,
        ),
        "ja-en": validate_checkpoint(
            args.candidate_ja_en_checkpoint,
            "ja-en",
            args.ja_en_dataset_manifest,
            contract,
        ),
    }
    bundle = validate_bundle(
        args.candidate_bundle,
        checkpoint_hashes,
        [
            candidate_document_report,
            candidate_segment_report,
            candidate_direct_report,
        ],
    )

    statistics = contract["continuation_gates"]["statistics"]
    samples = int(statistics["paired_bootstrap_samples"])
    confidence = float(statistics["early_stop_confidence"])
    seed = int(statistics["seed"])
    quality = contract["continuation_gates"]["quality_each_direction"]
    runtime = contract["continuation_gates"]["runtime"]
    directions: dict[str, Any] = {}
    direction_passes: list[bool] = []
    for offset, (direction_name, languages) in enumerate(DIRECTIONS.items()):
        segment_ids = sorted(
            case_id
            for case_id, row in segments.items()
            if (row.get("sourceLanguage"), row.get("targetLanguage")) == languages
        )
        document_ids = sorted(
            case_id
            for case_id, row in cases.items()
            if (row.get("sourceLanguage"), row.get("targetLanguage")) == languages
        )
        direct_ids = sorted(
            case_id
            for case_id, row in direct_cases.items()
            if (row.get("sourceLanguage"), row.get("targetLanguage")) == languages
        )
        segment_chrf_deltas = [
            sentence_chrf(candidate_segments[case_id])
            - sentence_chrf(baseline_segments[case_id])
            for case_id in segment_ids
        ]
        by_parent: dict[str, list[float]] = defaultdict(list)
        for case_id, value in zip(segment_ids, segment_chrf_deltas):
            by_parent[str(segments[case_id].get("parentCaseID") or case_id)].append(
                value
            )
        parent_chrf_deltas = [
            sum(values) / len(values) for _, values in sorted(by_parent.items())
        ]
        chrf_interval = bootstrap(
            parent_chrf_deltas,
            samples=samples,
            confidence=confidence,
            seed=seed + offset,
        )
        chrf_interval["method"] = (
            "paired-parent-document-resampling-of-parent-mean-segment-deltas"
        )
        comet_deltas = [
            float(candidate_comet[case_id]["score"])
            - float(baseline_comet[case_id]["score"])
            for case_id in document_ids
        ]
        comet_interval = bootstrap(
            comet_deltas,
            samples=samples,
            confidence=confidence,
            seed=seed + 10 + offset,
        )
        judge_interval = bootstrap(
            [judge_deltas[case_id] for case_id in document_ids],
            samples=samples,
            confidence=confidence,
            seed=seed + 20 + offset,
        )
        candidate_document_metrics = corpus_metrics(
            [candidate_documents[case_id] for case_id in document_ids]
        )
        baseline_document_metrics = corpus_metrics(
            [baseline_documents[case_id] for case_id in document_ids]
        )
        candidate_segment_metrics = corpus_metrics(
            [candidate_segments[case_id] for case_id in segment_ids]
        )
        baseline_segment_metrics = corpus_metrics(
            [baseline_segments[case_id] for case_id in segment_ids]
        )
        candidate_direct_metrics = corpus_metrics(
            [candidate_direct[case_id] for case_id in direct_ids]
        )
        baseline_direct_metrics = corpus_metrics(
            [baseline_direct[case_id] for case_id in direct_ids]
        )
        domains = {}
        domain_passes = []
        for domain_name in sorted({str(cases[case_id]["domain"]) for case_id in document_ids}):
            ids = [
                case_id
                for case_id in document_ids
                if str(cases[case_id]["domain"]) == domain_name
            ]
            candidate_value = float(
                corpus_metrics([candidate_documents[case_id] for case_id in ids])[
                    "chrFPlusPlus"
                ]
            )
            baseline_value = float(
                corpus_metrics([baseline_documents[case_id] for case_id in ids])[
                    "chrFPlusPlus"
                ]
            )
            delta = candidate_value - baseline_value
            passed = delta >= -float(quality["maximum_domain_chrf_pp_regression"])
            domain_passes.append(passed)
            domains[domain_name] = {
                "cases": len(ids),
                "candidateChrFPlusPlus": candidate_value,
                "baselineChrFPlusPlus": baseline_value,
                "delta": delta,
                "passed": passed,
            }
        candidate_latencies = [
            float(value)
            for case_id in segment_ids
            for value in candidate_segments[case_id]["warmLatencySeconds"]
        ]
        baseline_latencies = [
            float(value)
            for case_id in segment_ids
            for value in baseline_segments[case_id]["warmLatencySeconds"]
        ]
        candidate_p95 = percentile(candidate_latencies, 0.95)
        baseline_p95 = percentile(baseline_latencies, 0.95)
        improvement_signals = {
            "parentBalancedSentenceChrFPlusPlus": {
                "passed": float(chrf_interval["mean"])
                >= float(quality["mean_sentence_chrf_pp_delta_minimum"]),
                "actual": chrf_interval["mean"],
                "minimum": quality["mean_sentence_chrf_pp_delta_minimum"],
            },
            "COMET22": {
                "passed": float(comet_interval["mean"])
                >= float(quality["comet22_mean_delta_minimum_for_signal"]),
                "actual": comet_interval["mean"],
                "minimum": quality["comet22_mean_delta_minimum_for_signal"],
            },
            "automatedJudge": {
                "passed": float(judge_interval["mean"])
                >= float(quality["automated_judge_mean_delta_minimum_for_signal"]),
                "actual": judge_interval["mean"],
                "minimum": quality[
                    "automated_judge_mean_delta_minimum_for_signal"
                ],
            },
        }
        signal_count = sum(
            bool(value["passed"]) for value in improvement_signals.values()
        )
        gates = [
            gate(
                "multi-metric-improvement-signals",
                signal_count >= int(quality["minimum_improvement_signals"]),
                {
                    "passed": signal_count,
                    "signals": improvement_signals,
                },
                f">= {quality['minimum_improvement_signals']} of 3",
            ),
            gate(
                "parent-balanced-sentence-chrF++-noninferiority-lower",
                float(chrf_interval["lower"])
                >= float(quality["chrf_pp_paired_lower_minimum"]),
                chrf_interval["lower"],
                quality["chrf_pp_paired_lower_minimum"],
            ),
            gate(
                "COMET-22-noninferiority-lower",
                float(comet_interval["lower"])
                >= float(quality["comet22_paired_lower_minimum"]),
                comet_interval["lower"],
                quality["comet22_paired_lower_minimum"],
            ),
            gate(
                "automated-judge-noninferiority-lower",
                float(judge_interval["lower"])
                >= float(quality["automated_judge_paired_lower_minimum"]),
                judge_interval["lower"],
                quality["automated_judge_paired_lower_minimum"],
            ),
            gate(
                "document-sacreBLEU-regression",
                float(candidate_document_metrics["sacreBLEUIntl"])
                >= float(baseline_document_metrics["sacreBLEUIntl"])
                - float(quality["sacrebleu_corpus_regression_maximum"]),
                float(candidate_document_metrics["sacreBLEUIntl"])
                - float(baseline_document_metrics["sacreBLEUIntl"]),
                f">= -{quality['sacrebleu_corpus_regression_maximum']}",
            ),
            gate(
                "all-document-domain-chrF++-retention",
                all(domain_passes),
                domains,
                f"every delta >= -{quality['maximum_domain_chrf_pp_regression']}",
            ),
            gate(
                "direct-chrF++-retention",
                float(candidate_direct_metrics["chrFPlusPlus"])
                >= float(baseline_direct_metrics["chrFPlusPlus"])
                - float(quality["maximum_direct_chrf_pp_regression"]),
                float(candidate_direct_metrics["chrFPlusPlus"])
                - float(baseline_direct_metrics["chrFPlusPlus"]),
                f">= -{quality['maximum_direct_chrf_pp_regression']}",
            ),
            gate(
                "direct-sacreBLEU-retention",
                float(candidate_direct_metrics["sacreBLEUIntl"])
                >= float(baseline_direct_metrics["sacreBLEUIntl"])
                - float(quality["sacrebleu_corpus_regression_maximum"]),
                float(candidate_direct_metrics["sacreBLEUIntl"])
                - float(baseline_direct_metrics["sacreBLEUIntl"]),
                f">= -{quality['sacrebleu_corpus_regression_maximum']}",
            ),
            gate(
                "warm-segment-p95-absolute",
                candidate_p95
                <= float(runtime["warm_segment_p95_seconds_maximum_each_direction"]),
                candidate_p95,
                runtime["warm_segment_p95_seconds_maximum_each_direction"],
            ),
        ]
        passed = all(value["passed"] for value in gates)
        direction_passes.append(passed)
        directions[direction_name] = {
            "passed": passed,
            "sentenceChrFPlusPlusPairedDelta": chrf_interval,
            "COMET22PairedDelta": comet_interval,
            "automatedJudgePairedDelta": judge_interval,
            "improvementSignals": improvement_signals,
            "segment": {
                "candidate": candidate_segment_metrics,
                "baseline": baseline_segment_metrics,
            },
            "documentSegmentThenJoin": {
                "candidate": candidate_document_metrics,
                "baseline": baseline_document_metrics,
            },
            "documentDirectWithinLimit": {
                "candidate": candidate_direct_metrics,
                "baseline": baseline_direct_metrics,
            },
            "domains": domains,
            "latency": {
                "candidateWarmSegmentP95Seconds": candidate_p95,
                "baselineWarmSegmentP95Seconds": baseline_p95,
                "relativeRegressionDiagnostic": candidate_p95 / baseline_p95 - 1,
            },
            "gates": gates,
        }

    safety = contract["continuation_gates"]["safety"]
    baseline_union = (
        baseline_structure["exact"]
        | baseline_structure["typed"]
        | baseline_structure["negation"]
        | baseline_judge_critical
    )
    candidate_union = (
        candidate_structure["exact"]
        | candidate_structure["typed"]
        | candidate_structure["negation"]
        | candidate_judge_critical
    )
    new_union = sorted(candidate_union - baseline_union)
    new_negation = sorted(
        candidate_structure["negation"] - baseline_structure["negation"]
    )
    new_typed = sorted(candidate_structure["typed"] - baseline_structure["typed"])
    new_judge = sorted(candidate_judge_critical - baseline_judge_critical)
    baseline_generation = generation_failures(
        baseline_document_report, baseline_documents, "segment-then-join"
    ) | generation_failures(baseline_direct_report, baseline_direct, "direct")
    candidate_generation = generation_failures(
        candidate_document_report, candidate_documents, "segment-then-join"
    ) | generation_failures(candidate_direct_report, candidate_direct, "direct")
    new_generation = sorted(candidate_generation - baseline_generation)
    peak_resident = max(
        int(report.get("peakResidentBytes", 0))
        for report in (
            candidate_document_report,
            candidate_segment_report,
            candidate_direct_report,
        )
    )
    global_gates = [
        gate(
            "development-gate-use-budget",
            1
            <= args.development_gate_use
            <= int(statistics["maximum_development_gate_uses"]),
            args.development_gate_use,
            f"1..{statistics['maximum_development_gate_uses']}",
        ),
        gate(
            "new-union-critical-errors",
            len(new_union) <= int(safety["new_union_critical_errors_maximum"]),
            new_union,
            safety["new_union_critical_errors_maximum"],
        ),
        gate(
            "new-negation-errors",
            len(new_negation) <= int(safety["new_negation_errors_maximum"]),
            new_negation,
            safety["new_negation_errors_maximum"],
        ),
        gate(
            "new-number-date-unit-placeholder-errors",
            len(new_typed)
            <= int(safety["new_number_date_unit_placeholder_errors_maximum"]),
            new_typed,
            safety["new_number_date_unit_placeholder_errors_maximum"],
        ),
        gate(
            "automated-judge-critical-error-increase",
            len(new_judge)
            <= int(safety["automated_judge_critical_error_increase_maximum"]),
            new_judge,
            safety["automated_judge_critical_error_increase_maximum"],
        ),
        gate(
            "new-repetition-or-nontermination-failures",
            len(new_generation)
            <= int(
                contract["continuation_gates"]["long_documents"][
                    "new_repetition_or_nontermination_failures_maximum"
                ]
            ),
            new_generation,
            0,
        ),
        gate(
            "peak-resident-bytes",
            0 < peak_resident <= int(runtime["peak_resident_bytes_maximum"]),
            peak_resident,
            runtime["peak_resident_bytes_maximum"],
        ),
        gate(
            "bundle-hard-maximum",
            0
            < int(bundle["bundleBytes"])
            <= int(contract["continuation_gates"]["bundle"]["hard_maximum_bytes"]),
            bundle["bundleBytes"],
            contract["continuation_gates"]["bundle"]["hard_maximum_bytes"],
        ),
    ]
    continue_training = all(direction_passes) and all(
        value["passed"] for value in global_gates
    )
    output = {
        "schemaVersion": 1,
        "status": (
            "phase-1-continuation-approved"
            if continue_training
            else "phase-1-continuation-rejected"
        ),
        "continueTraining": continue_training,
        "promotionAuthorized": False,
        "appChangeAuthorized": False,
        "candidateModelRevision": bundle["revision"],
        "baselineModelRevision": baseline_document_report["modelRevision"],
        "preferredBundleTargetMet": int(bundle["bundleBytes"])
        <= int(contract["continuation_gates"]["bundle"]["target_bytes"]),
        "bundle": bundle,
        "globalGates": global_gates,
        "directions": directions,
        "safetyEvidence": {
            "baselineUnionCriticalCaseIDs": sorted(baseline_union),
            "candidateUnionCriticalCaseIDs": sorted(candidate_union),
            "newUnionCriticalCaseIDs": new_union,
            "newGenerationFailureIDs": new_generation,
        },
        "inputs": {
            name: (
                sha256(path)
                if path.is_file()
                else f"directory:{path.resolve()}"
            )
            for name, path in vars(args).items()
            if isinstance(path, Path) and name != "output"
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if continue_training else 2)


if __name__ == "__main__":
    main()
