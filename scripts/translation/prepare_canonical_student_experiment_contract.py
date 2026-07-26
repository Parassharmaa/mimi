#!/usr/bin/env python3
"""Freeze the canonical-target v3 Marian student experiment before training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION = (
    "scripts/translation/train_marian_distillation.py",
    "scripts/translation/prepare_elanmt_mlx.py",
    "scripts/translation/run_mlx_marian_benchmark.py",
    "scripts/translation/score_comet.py",
    "scripts/translation/evaluate_gpt56_student_continuation.py",
    "scripts/translation/build_reference_anchored_distillation_dataset.py",
    "scripts/translation/approve_canonical_quality_consensus.py",
    "scripts/translation/prepare_canonical_student_experiment_contract.py",
)
BENCHMARKS = (
    "Research/translation/benchmark/development-accuracy-v1.jsonl",
    "Research/translation/benchmark/development-accuracy-v1.segments.jsonl",
    "Research/translation/benchmark/development-accuracy-v1.direct-under-192.jsonl",
    "Research/translation/benchmark/development-accuracy-v1.direct-under-192.manifest.json",
    "Research/translation/benchmark/automated-claim-v1.sources.jsonl",
    "Research/translation/results/development-accuracy-v1-candidate-clean-pair.json",
    "Research/translation/results/development-accuracy-v1-candidate-clean-pair-segments.json",
    "Research/translation/results/development-accuracy-v1-candidate-clean-pair-direct-under-192.json",
    "Research/translation/results/development-accuracy-v1-candidate-clean-pair-comet22.json",
    "Research/translation/results/development-accuracy-v1-candidate-clean-pair-structure-audit.json",
)
PRESERVATION_ORIGINS = (
    "finalized-japanese-law-translation",
    "human-alt-document-window",
    "human-alt-parallel",
    "human-kftt-replay",
    "human-tatoeba-bidirectional-agreement-filtered",
    "licensed-human-reference-anchor",
    "mimi-shipped-ui-pair",
)
IDENTITIES = {
    "en-ja": (
        "Mitsua/elan-mt-bt-en-ja",
        "02c48e7031386cd2d41974b0ff1aaf52f010c5fa",
    ),
    "ja-en": (
        "Mitsua/elan-mt-bt-ja-en",
        "539f80eb05306e27a166b45e4264c7fa2eb4de97",
    ),
}


def digest(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"missing contract input: {path}")
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


def dataset_contract(path: Path, direction: str, approved_sha256: str) -> dict:
    manifest_path = path / "manifest.json"
    manifest = load(manifest_path)
    train_path = path / "train.jsonl"
    valid_path = path / "valid.jsonl"
    expected_train = manifest.get("outputs", {}).get("train", {}).get("sha256")
    expected_valid = manifest.get("outputs", {}).get("valid", {}).get("sha256")
    if (
        manifest.get("direction") != direction
        or manifest.get("private_reasoning_traces_used") is not False
        or manifest.get("promotion_eligible") is not True
        or manifest.get("synthetic_policy", {}).get("actual_synthetic_fraction") != 0.25
        or manifest.get("synthetic_policy", {}).get("maximum_synthetic_fraction") != 0.25
        or manifest.get("inputs", {}).get("approved", {}).get("sha256")
        != approved_sha256
        or expected_train != digest(train_path)
        or expected_valid != digest(valid_path)
        or manifest.get("decontamination", {}).get("train_validation_exact_overlap")
        is not False
    ):
        raise SystemExit(f"dataset manifest failed frozen invariants: {manifest_path}")
    counts = manifest.get("counts", {})
    approved = counts.get("approved_teacher_targets")
    if not isinstance(approved, int) or approved < 50:
        raise SystemExit(f"dataset has too few approved targets: {manifest_path}")
    if counts.get("train") != approved * 4:
        raise SystemExit(f"dataset is not exact 1:3 synthetic/human: {manifest_path}")
    return {
        "directory": str(path.relative_to(ROOT)),
        "manifest": record(manifest_path),
        "train": record(train_path),
        "valid": record(valid_path),
        "counts": counts,
        "synthetic_policy": manifest["synthetic_policy"],
        "licenses": counts.get("licenses", {}),
        "protected_suites": manifest.get("decontamination", {}).get(
            "protected_suites", []
        ),
    }


def checkpoint_contract(path: Path, direction: str) -> dict:
    config = load(path / "config.json")
    if (
        int(config.get("encoder_layers", -1)) != 6
        or int(config.get("decoder_layers", -1)) != 6
    ):
        raise SystemExit(f"checkpoint is not intact Marian 6e/6d: {path}")
    files = {}
    for name in (
        "model.safetensors",
        "config.json",
        "generation_config.json",
        "source.spm",
        "target.spm",
        "tokenizer_config.json",
        "vocab.json",
        "special_tokens_map.json",
        "mimi_training_manifest.json",
        "mimi_checkpoint_averaging_manifest.json",
    ):
        candidate = path / name
        if candidate.is_file():
            files[name] = record(candidate)
    repository, revision = IDENTITIES[direction]
    return {
        "path": str(path.relative_to(ROOT)),
        "direction": direction,
        "architecture": "Marian encoder-decoder 6e/6d",
        "repository": repository,
        "revision": revision,
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and output.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {output}")

    scale_contract_path = ROOT / (
        "Research/translation/canonical-target-scale-v3-contract-2026-07-25.json"
    )
    approved_path = ROOT / (
        "Research/translation/work/canonical-target-scale-v3.approved.jsonl"
    )
    scale_contract = load(scale_contract_path)
    scale_gate = scale_contract.get("goGateBeforeScaling", {})
    if (
        scale_gate.get("acceptancePolicy") != "dual-absolute-canonical"
        or scale_gate.get("trainingAuthorizedByPilotAlone") is not False
        or scale_gate.get("minimumApprovedTotal") != 120
        or scale_gate.get("minimumApprovedEachDirection") != 50
    ):
        raise SystemExit("scale contract does not carry the frozen absolute gate")
    approved_sha256 = digest(approved_path)

    datasets = {
        direction: dataset_contract(
            ROOT
            / f"Research/translation/work/canonical-target-scale-v3-dataset-{direction}",
            direction,
            approved_sha256,
        )
        for direction in ("en-ja", "ja-en")
    }
    parents = {
        "en-ja": checkpoint_contract(
            ROOT
            / "Research/translation/models/elanmt-release-clean-full-depth-en-ja-v1-avg3",
            "en-ja",
        ),
        "ja-en": checkpoint_contract(
            ROOT
            / "Research/translation/models/elanmt-release-clean-legal-specialist-ja-en-v1",
            "ja-en",
        ),
    }
    baseline_summary_path = (
        ROOT
        / "Research/translation/benchmark/development-accuracy-v1.results-summary.json"
    )
    baseline_summary = load(baseline_summary_path)

    contract = {
        "schema_version": 1,
        "experiment": "canonical-target-scale-v3-intact-marian-step250",
        "status": "preregistered-ready-for-training",
        "promotion_authorized": False,
        "app_change_authorized": False,
        "public_upload_authorized": False,
        "teacher_authentication": (
            "Codex cached ChatGPT authentication; no OpenAI API key used"
        ),
        "private_reasoning_traces_used": False,
        "hypothesis": (
            "two-judge absolute-quality canonical teacher targets can improve the "
            "current intact Marian students after exact MLX q4/group-64 conversion "
            "without new critical, document, latency, memory, size, or license failures"
        ),
        "admission": {
            "scale_contract": record(scale_contract_path),
            "approved_targets": record(approved_path),
            "approved_total": 180,
            "approved_by_direction": {"en-ja": 97, "ja-en": 83},
            "required_review_status": (
                "two-judge-reference-anchored-canonical-absolute"
            ),
            "teacher_model": "gpt-5.6-sol via Codex session",
            "judge_families": ["Claude Fable 5", "Qwen3-8B-4bit-MLX"],
        },
        "datasets": datasets,
        "student": {
            "architecture": "two intact direction-specific Marian 6e/6d models",
            "parents": parents,
            "phase_1": {
                "seed": 20260725,
                "batch_size": 8,
                "gradient_accumulation": 4,
                "effective_batch_size": 32,
                "max_steps": 250,
                "learning_rate": 0.000002,
                "weight_decay": 0.01,
                "warmup_steps": 25,
                "evaluation_steps": 250,
                "max_source_tokens": 192,
                "max_target_tokens": 192,
                "frozen_parent_kl_weight": 0.10,
                "l2_to_parent_weight": 0.01,
                "preservation_origins": list(PRESERVATION_ORIGINS),
                "training_description": (
                    "sequence-level distillation from accepted canonical final "
                    "translations with licensed human-reference anchors and replay; "
                    "no private chain-of-thought"
                ),
            },
            "continuation": {
                "authorized_by_this_contract": False,
                "maximum_total_steps_if_separately_authorized": 1000,
            },
        },
        "conversion": {
            "required_before_decision": True,
            "format": "MLX affine q4",
            "bits": 4,
            "group_size": 64,
            "hard_two_direction_bundle_maximum_bytes": 500_000_000,
            "preferred_two_direction_bundle_maximum_bytes": 150_000_000,
        },
        "benchmarks": {
            "inputs": {
                path: record(ROOT / path)
                for path in BENCHMARKS
            },
            "baseline_summary": record(baseline_summary_path),
            "baseline_metrics": {
                direction: baseline_summary["metrics"][direction]["candidate"]
                for direction in ("en-ja", "ja-en")
            },
            "apple_translation_role": (
                "diagnostic historical comparison only; never a runtime dependency"
            ),
        },
        "continuation_gates": {
            "quality_each_direction": {
                "minimum_improvement_signals": 2,
                "mean_sentence_chrf_pp_delta_minimum": 0.25,
                "comet22_mean_delta_minimum_for_signal": 0.002,
                "independent_judge_mean_delta_minimum_for_signal": 0.10,
                "chrf_pp_paired_90pct_lower_minimum": -0.25,
                "comet22_paired_90pct_lower_minimum": -0.005,
                "independent_judge_paired_90pct_lower_minimum": -0.25,
                "sacrebleu_corpus_regression_maximum": 0.10,
                "maximum_domain_chrf_pp_regression": 0.50,
                "maximum_direct_chrf_pp_regression": 0.50,
            },
            "safety": {
                "new_union_critical_errors_maximum": 0,
                "new_negation_errors_maximum": 0,
                "new_number_date_unit_placeholder_errors_maximum": 0,
                "new_judge_critical_errors_maximum": 0,
            },
            "documents": {
                "segment_then_join_and_direct_evaluation_required": True,
                "new_repetition_or_nontermination_failures_maximum": 0,
            },
            "runtime": {
                "warm_segment_p95_seconds_maximum_each_direction": 0.175,
                "peak_resident_bytes_maximum": 250_000_000,
            },
            "licensing": {
                "distribution_review_required": True,
                "per_row_license_and_attribution_sidecar_required": True,
                "hugging_face_visibility_when_eligible": "public",
            },
        },
        "implementation": {
            path: record(ROOT / path)
            for path in IMPLEMENTATION
        },
        "pending_before_any_promotion": [
            "exact q4 held-out development and safety evaluation",
            "COMET-22 comparison in both directions",
            "two-family blinded quality and critical-error comparison",
            "Swift MLX parity",
            "bundle, latency, peak-RSS, and license-distribution gates",
            "sealed promotion evaluation with positive paired lower bounds",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record(output), indent=2))


if __name__ == "__main__":
    main()
