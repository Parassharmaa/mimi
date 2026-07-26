#!/usr/bin/env python3
"""Freeze Mimi's first GPT-5.6-distilled intact-Marian experiment contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCRIPT_PATHS = [
    Path("scripts/translation/filter_synthetic_batch.py"),
    Path("scripts/translation/prepare_distillation_judge_batch.py"),
    Path("scripts/translation/run_synthetic_batch.py"),
    Path("scripts/translation/prioritize_distillation_judgments.py"),
    Path("scripts/translation/approve_automated_consensus.py"),
    Path("scripts/translation/build_distillation_dataset.py"),
    Path("scripts/translation/build_reference_anchored_distillation_dataset.py"),
    Path("scripts/translation/prepare_gpt56_phase1_judge_batch.py"),
    Path("scripts/translation/collect_gpt56_phase1_judge.py"),
    Path("scripts/translation/prepare_direct_within_limit_suite.py"),
    Path("scripts/translation/train_marian_distillation.py"),
    Path("scripts/translation/prepare_elanmt_mlx.py"),
    Path("scripts/translation/run_mlx_marian_benchmark.py"),
    Path("scripts/translation/score_comet.py"),
    Path("scripts/translation/evaluate_automated_translation_promotion.py"),
    Path("scripts/translation/audit_automated_claim_reference_structures.py"),
    Path("scripts/translation/evaluate_gpt56_student_continuation.py"),
    Path("scripts/translation/prepare_gpt56_student_experiment_contract.py"),
]
MODEL_FILES = (
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "source.spm",
    "target.spm",
    "vocab.json",
    "mimi_training_manifest.json",
    "mimi_checkpoint_averaging_manifest.json",
)


def sha256(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"missing contract input: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid JSON input {path}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"expected a JSON object: {path}")
    return value


def file_contract(path: Path) -> dict:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def resolve_declared_path(declared: str, declaring_file: Path) -> Path:
    path = Path(declared)
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, declaring_file.parent / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SystemExit(
        f"declared input path does not exist relative to cwd or manifest: {declared}"
    )


def seed_contract(path: Path) -> dict:
    manifest = read_json(path)
    output = manifest.get("output")
    if not isinstance(output, dict):
        raise SystemExit("seed manifest has no output contract")
    seed_path = resolve_declared_path(str(output.get("path", "")), path)
    actual = sha256(seed_path)
    if output.get("sha256") != actual:
        raise SystemExit("seed JSONL hash does not match its manifest")
    counts = manifest.get("counts", {})
    if (
        counts.get("rows") != 16_000
        or counts.get("directions") != {"en-ja": 8_000, "ja-en": 8_000}
    ):
        raise SystemExit("student contract requires the frozen 8k-per-direction pilot")
    return {
        "manifest": file_contract(path),
        "seeds": file_contract(seed_path),
        "rows": counts["rows"],
        "directions": counts["directions"],
        "rows_with_human_reference": counts.get("rows_with_human_reference"),
        "rows_without_human_reference": counts.get("rows_without_human_reference"),
    }


def request_contract(path: Path) -> dict:
    contract = read_json(path)
    request_path = resolve_declared_path(str(contract.get("request_path", "")), path)
    if contract.get("request_sha256") != sha256(request_path):
        raise SystemExit("request JSONL hash does not match its contract")
    required = {
        "request_count": 16_000,
        "model": "gpt-5.6-sol",
        "reasoning_effort": "none",
        "pipeline": "mimi-translation-v1",
    }
    for key, expected in required.items():
        if contract.get(key) != expected:
            raise SystemExit(f"request contract {key} is not frozen to {expected!r}")
    return {
        "contract": file_contract(path),
        "requests": file_contract(request_path),
        **{key: contract[key] for key in required},
        "prompt_sha256": contract.get("prompt_sha256"),
    }


def cost_contract(path: Path, request_sha256: str) -> dict:
    cost = read_json(path)
    if cost.get("submitted") is not False:
        raise SystemExit("cost input unexpectedly says the teacher batch was submitted")
    nested = cost.get("contract", {})
    if nested.get("request_sha256") != request_sha256:
        raise SystemExit("cost estimate does not bind the frozen request file")
    pricing = cost.get("pricing_usd_per_million_tokens", {})
    if pricing.get("operator_must_refresh_before_submission") is not True:
        raise SystemExit("cost estimate does not require a pre-submission price refresh")
    return {
        "report": file_contract(path),
        "estimated_cost_usd": cost.get("estimated_cost_usd"),
        "retention_warning": cost.get("retention_warning"),
        "pricing": pricing,
    }


def checkpoint_contract(path: Path, direction: str) -> dict:
    model_path = path / "model.safetensors"
    if not model_path.is_file():
        raise SystemExit(f"initial {direction} checkpoint is incomplete: {path}")
    files = {
        name: file_contract(path / name)
        for name in MODEL_FILES
        if (path / name).is_file()
    }
    tokenizer_complete = (
        "tokenizer_config.json" in files
        and (
            "tokenizer.json" in files
            or {"source.spm", "target.spm", "vocab.json"}.issubset(files)
        )
    )
    if "config.json" not in files or not tokenizer_complete:
        raise SystemExit(f"initial {direction} checkpoint lacks config/tokenizer")
    config = read_json(path / "config.json")
    if (
        int(config.get("encoder_layers", -1)) != 6
        or int(config.get("decoder_layers", -1)) != 6
    ):
        raise SystemExit(f"initial {direction} checkpoint is not intact 6e/6d Marian")
    return {
        "path": str(path),
        "direction": direction,
        "architecture": "Marian encoder-decoder 6e/6d",
        "files": files,
    }


def baseline_contract(path: Path) -> dict:
    baseline = read_json(path)
    metrics = baseline.get("metrics", {})
    for direction in ("en-ja", "ja-en"):
        candidate = metrics.get(direction, {}).get("candidate", {})
        for metric in ("chrfPlusPlus", "sacreBLEUIntl", "comet22"):
            if not isinstance(candidate.get(metric), (int, float)):
                raise SystemExit(f"baseline summary lacks {direction} {metric}")
    return {
        "summary": file_contract(path),
        "model_revision": baseline.get("models", {})
        .get("candidate", {})
        .get("modelRevision"),
        "candidate_metrics": {
            direction: metrics[direction]["candidate"]
            for direction in ("en-ja", "ja-en")
        },
        "bundle_bytes": baseline.get("models", {})
        .get("candidate", {})
        .get("minimalBundleBytes"),
        "peak_resident_bytes": baseline.get("models", {})
        .get("candidate", {})
        .get("peakResidentBytes"),
    }


def direct_suite_contract(path: Path, manifest_path: Path) -> tuple[dict, dict]:
    manifest = read_json(manifest_path)
    output = manifest.get("output", {})
    if (
        manifest.get("schemaVersion") != 1
        or manifest.get("selection", {}).get("manualInclusions") != []
        or manifest.get("selection", {}).get("manualExclusions") != []
        or output.get("sha256") != sha256(path)
        or output.get("cases") != 197
        or output.get("directions") != {"en-ja": 98, "ja-en": 99}
    ):
        raise SystemExit("direct-within-limit suite is not the frozen automatic slice")
    return file_contract(path), file_contract(manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed_manifest", type=Path)
    parser.add_argument("request_contract", type=Path)
    parser.add_argument("cost_estimate", type=Path)
    parser.add_argument("initial_en_ja", type=Path)
    parser.add_argument("initial_ja_en", type=Path)
    parser.add_argument("development_cases", type=Path)
    parser.add_argument("development_segments", type=Path)
    parser.add_argument("development_summary", type=Path)
    parser.add_argument("automated_claim_sources", type=Path)
    parser.add_argument("direct_within_limit_cases", type=Path)
    parser.add_argument("direct_within_limit_manifest", type=Path)
    parser.add_argument("development_baseline_document_report", type=Path)
    parser.add_argument("development_baseline_segment_report", type=Path)
    parser.add_argument("development_baseline_direct_report", type=Path)
    parser.add_argument("development_baseline_comet_report", type=Path)
    parser.add_argument("development_baseline_structure_audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists() and args.output.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")

    seeds = seed_contract(args.seed_manifest)
    requests = request_contract(args.request_contract)
    cost = cost_contract(args.cost_estimate, requests["requests"]["sha256"])
    baseline = baseline_contract(args.development_summary)
    development_cases = file_contract(args.development_cases)
    development_segments = file_contract(args.development_segments)
    automated_sources = file_contract(args.automated_claim_sources)
    direct_cases, direct_manifest = direct_suite_contract(
        args.direct_within_limit_cases, args.direct_within_limit_manifest
    )
    if development_cases["sha256"] != "0684350a4c941a7fc87801444c027e0f5c02ba6f77418c792428d4200b521605":
        raise SystemExit("development case suite is not the frozen v1 suite")
    if development_segments["sha256"] != "b464dfbfe128b3e0cbac2652d8db6ddb976536638676a1dca6e07507eb32a58f":
        raise SystemExit("development segment suite is not the frozen v1 suite")
    if automated_sources["sha256"] != "f039ce456c55f051e8bbcc13ed9bc8270a722819308e008b39da7f30327ec16c":
        raise SystemExit("automated-claim source suite is not the frozen 400+400 draft")
    baseline_artifacts = {
        "development_baseline_document_report": file_contract(
            args.development_baseline_document_report
        ),
        "development_baseline_segment_report": file_contract(
            args.development_baseline_segment_report
        ),
        "development_baseline_direct_report": file_contract(
            args.development_baseline_direct_report
        ),
        "development_baseline_comet_report": file_contract(
            args.development_baseline_comet_report
        ),
        "development_baseline_structure_audit": file_contract(
            args.development_baseline_structure_audit
        ),
    }
    baseline_report = read_json(args.development_baseline_document_report)
    if baseline_report.get("modelRevision") != baseline["model_revision"]:
        raise SystemExit("baseline document report differs from the frozen best local model")

    script_contracts = {
        str(path): file_contract(path)
        for path in SCRIPT_PATHS
    }
    contract = {
        "schema_version": 1,
        "experiment": "gpt56-final-translation-intact-marian-pilot-v1",
        "status": "preregistered-waiting-for-teacher-and-judge-outputs",
        "promotion_authorized": False,
        "app_change_authorized": False,
        "hypothesis": (
            "reference-anchored GPT-5.6 final translations can improve the intact "
            "6e/6d Marian student after exact q4 conversion without increasing "
            "critical errors or harming document/domain retention"
        ),
        "teacher": {
            "seeds": seeds,
            "requests": requests,
            "cost": cost,
            "private_reasoning_traces_used": False,
            "submission_required": True,
        },
        "admission": {
            "mode": "blinded-two-distinct-judge-reference-anchored-consensus",
            "promotion_training_rows": (
                "only two-judge-reference-anchored teacher wins; exclude all "
                "three-candidate source-only provisional rows"
            ),
            "licensed_reference_is_anonymous_candidate": True,
            "reference_or_reference_equivalent_win_emits_no_synthetic_row": True,
            "teacher_model_must_differ_from_both_judges": True,
            "no_private_reasoning_traces": True,
        },
        "student": {
            "architecture": "two intact direction-specific Marian 6e/6d models",
            "initial_checkpoints": {
                "en-ja": checkpoint_contract(args.initial_en_ja, "en-ja"),
                "ja-en": checkpoint_contract(args.initial_ja_en, "ja-en"),
            },
            "phase_1": {
                "max_steps": 250,
                "seed": 20260725,
                "batch_size": 8,
                "gradient_accumulation": 4,
                "effective_batch_size": 32,
                "learning_rate": 0.000002,
                "weight_decay": 0.01,
                "warmup_steps": 25,
                "evaluation_steps": 250,
                "max_source_tokens": 192,
                "max_target_tokens": 192,
                "frozen_base_kl_weight": 0.10,
                "l2_to_base_weight": 0.01,
                "synthetic_target_fraction_maximum": 0.25,
                "human_replay": (
                    "at least three licensed human-reference rows per admitted "
                    "teacher row, balanced across original pilot corpora"
                ),
                "fake_quantization": False,
            },
            "continuation": {
                "maximum_total_steps": 1_000,
                "evaluation_every_steps": 250,
                "condition": "phase-1 exact-q4 candidate passes every continuation gate",
            },
        },
        "conversion": {
            "required_after_phase_1": True,
            "format": "MLX affine q4 group-size 64",
            "single_bidirectional_bundle": True,
        },
        "benchmarks": {
            "development_cases": development_cases,
            "development_segments": development_segments,
            "direct_within_limit_cases": direct_cases,
            "direct_within_limit_manifest": direct_manifest,
            "development_baseline": baseline,
            **baseline_artifacts,
            "automated_claim_sources": automated_sources,
            "sealed_promotion_references": "pending-independent-two-judge-assembly",
            "apple_translation_role": "diagnostic comparison only; never runtime dependency",
        },
        "continuation_gates": {
            "statistics": {
                "paired_bootstrap_samples": 10_000,
                "early_stop_confidence": 0.90,
                "seed": 20260725,
                "maximum_development_gate_uses": 4,
            },
            "quality_each_direction": {
                "minimum_improvement_signals": 2,
                "mean_sentence_chrf_pp_delta_minimum": 0.25,
                "chrf_pp_paired_lower_minimum": -0.25,
                "comet22_mean_delta_minimum_for_signal": 0.002,
                "comet22_paired_lower_minimum": -0.005,
                "automated_judge_mean_delta_minimum_for_signal": 0.10,
                "automated_judge_paired_lower_minimum": -0.25,
                "sacrebleu_corpus_regression_maximum": 0.10,
                "report_all_sentence_and_document_slices": True,
                "maximum_domain_chrf_pp_regression": 0.50,
                "maximum_direct_chrf_pp_regression": 0.50,
            },
            "safety": {
                "new_union_critical_errors_maximum": 0,
                "new_negation_errors_maximum": 0,
                "new_number_date_unit_placeholder_errors_maximum": 0,
                "automated_judge_critical_error_increase_maximum": 0,
            },
            "long_documents": {
                "new_repetition_or_nontermination_failures_maximum": 0,
                "evaluate_segment_then_join_and_direct_within_token_limit": True,
            },
            "runtime": {
                "warm_segment_p95_seconds_maximum_each_direction": 0.175,
                "relative_latency_regression_diagnostic_fraction": 0.10,
                "peak_resident_bytes_maximum": 250_000_000,
            },
            "bundle": {
                "target_bytes": 150_000_000,
                "hard_maximum_bytes": 500_000_000,
            },
            "licensing": {
                "per_row_source_license_and_attribution_required": True,
                "model_and_dataset_publication_requires_distribution_review": True,
                "hugging_face_visibility_when_eligible": "public",
            },
        },
        "promotion_gates_after_continuation": {
            "sealed_400_per_direction_suite_complete": True,
            "report_metrics": ["chrF++", "sacreBLEU", "COMET-22"],
            "paired_quality_lower_bounds_strictly_positive_each_direction": True,
            "two_blinded_judges_no_critical_regression": True,
            "runtime_bundle_memory_and_license_gates_all_pass": True,
            "swift_mlx_parity_required": True,
            "current_translator_unchanged_until_all_pass": True,
        },
        "implementation": script_contracts,
        "pending_inputs": [
            "teacher batch output with all 16000 request IDs accounted for",
            "two complete judgment files from distinct non-teacher models",
            "reference-anchored approved targets and rejected/no-op audit",
            "promotion benchmark reference assembly",
            "two complete blinded quality/critical comparisons from distinct non-teacher, non-admission judge families",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "sha256": sha256(args.output),
        "status": contract["status"],
        "promotion_authorized": False,
        "app_change_authorized": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
