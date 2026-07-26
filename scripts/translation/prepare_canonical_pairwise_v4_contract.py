#!/usr/bin/env python3
"""Freeze the canonical JA-to-EN pairwise-preference v4 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "canonical-pairwise-preference-adapter-v4-ja-en"
PARENT = Path(
    "Research/translation/models/elanmt-release-clean-legal-specialist-ja-en-v1"
)
DATASET = Path("Research/translation/work/canonical-pairwise-v4-ja-en")
TRAINING = {
    "seed": 20260726,
    "rank": 8,
    "alpha": 16.0,
    "dropout": 0.0,
    "preset": "consultation-v1",
    "encoder_top_layers": 3,
    "batch_size": 4,
    "gradient_accumulation": 4,
    "max_steps": 40,
    "learning_rate": 0.00001,
    "weight_decay": 0.01,
    "warmup_steps": 4,
    "evaluation_steps": 10,
    "beta": 0.10,
    "chosen_sft_weight": 0.02,
    "max_source_tokens": 192,
    "max_target_tokens": 192,
}
BENCHMARKS = (
    Path("Research/translation/benchmark/development-accuracy-v1.jsonl"),
    Path("Research/translation/benchmark/development-accuracy-v1.segments.jsonl"),
    Path("Research/translation/benchmark/development-accuracy-v1.direct-under-192.jsonl"),
    Path("Research/translation/benchmark/development-accuracy-v1.results-summary.json"),
)
PROTECTED = (
    Path("Research/translation/benchmark/canary.jsonl"),
    Path("Research/translation/benchmark/public-stress-v1.jsonl"),
    Path("Research/translation/benchmark/public-stress-v2.jsonl"),
    Path("Research/translation/benchmark/public-stress-v3.jsonl"),
    Path("Research/translation/benchmark/legal-safety-validation-v1.jsonl"),
    Path("Research/translation/benchmark/legal-safety-test-v1.jsonl"),
    Path("Research/translation/benchmark/m2m100-418m-feasibility-v1.jsonl"),
    Path("Research/translation/benchmark/development-accuracy-v1.jsonl"),
    Path("Research/translation/benchmark/development-accuracy-v1.segments.jsonl"),
)
IMPLEMENTATION = (
    Path("scripts/translation/build_canonical_pairwise_preference_dataset.py"),
    Path("scripts/translation/train_marian_automated_preference_adapter.py"),
    Path("scripts/translation/prepare_elanmt_mlx.py"),
    Path("scripts/translation/run_mlx_marian_benchmark.py"),
    Path("scripts/translation/compose_segmented_document_benchmark.py"),
    Path("scripts/translation/score_translation.py"),
    Path("scripts/translation/score_comet.py"),
    Path("scripts/translation/evaluate_typed_critical_token_policy.py"),
)


def sha256(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"missing contract input: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path: Path) -> dict:
    return {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid JSON input {path}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path(
            "Research/translation/canonical-pairwise-v4-contract-2026-07-25.json"
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen contract: {args.output}")

    dataset_manifest = load_json(DATASET / "manifest.json")
    if (
        dataset_manifest.get("experiment")
        != "canonical teacher-over-current preference v4"
        or dataset_manifest.get("direction") != "ja-en"
        or dataset_manifest.get("promotion_eligible") is not True
        or dataset_manifest.get("claim_eligible") is not False
        or dataset_manifest.get("private_reasoning_traces_used") is not False
        or dataset_manifest.get("counts", {}).get("selected") != 61
        or dataset_manifest.get("counts", {}).get("train") != 51
        or dataset_manifest.get("counts", {}).get("valid") != 10
    ):
        raise SystemExit("canonical pairwise dataset does not match the v4 admission")
    review = dataset_manifest.get("review_policy", {})
    if (
        review.get("human_reviewer_required") is not False
        or review.get("independent_judge_models") != 2
        or review.get("unanimous_pareto_preference_over_current_required") is not True
        or review.get("strict_total_improvement_per_judge") is not True
    ):
        raise SystemExit("canonical pairwise review policy is weaker than v4")
    effective_licenses = dataset_manifest.get("effective_licenses")
    if not isinstance(effective_licenses, dict) or any(
        not isinstance(effective_licenses.get(split), dict)
        or not effective_licenses[split]
        for split in ("train", "valid")
    ):
        raise SystemExit("canonical pairwise dataset lacks effective licenses")
    for split in ("train", "valid"):
        path = DATASET / f"{split}.jsonl"
        if dataset_manifest["outputs"][split]["sha256"] != sha256(path):
            raise SystemExit(f"canonical pairwise {split} hash differs")
    declared_protected = {
        value["path"]: value["sha256"]
        for value in dataset_manifest.get("decontamination", {}).get(
            "protected_suites", []
        )
    }
    for path in PROTECTED:
        if declared_protected.get(str(path)) != sha256(path):
            raise SystemExit(f"dataset does not bind protected suite: {path}")

    parent_manifest = load_json(PARENT / "mimi_training_manifest.json")
    if (
        parent_manifest.get("direction") != "ja-en"
        or parent_manifest.get("student_repository") != "Mitsua/elan-mt-bt-ja-en"
        or parent_manifest.get("student_revision")
        != "539f80eb05306e27a166b45e4264c7fa2eb4de97"
    ):
        raise SystemExit("JA-to-EN parent identity differs")
    v3_result_path = Path(
        "Research/translation/canonical-target-student-v3-result-2026-07-25.json"
    )
    v3_result = load_json(v3_result_path)
    if (
        v3_result.get("status") != "phase-1-continuation-rejected"
        or v3_result.get("continueTraining") is not False
        or v3_result.get("directions", {}).get("ja-en", {}).get("passed") is not False
    ):
        raise SystemExit("v3 evidence does not authorize this new hypothesis")

    contract = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": "preregistered-ready-for-training",
        "direction": "ja-en",
        "hypothesis": (
            "A rank-8 adapter trained only on final translations unanimously "
            "Pareto-preferred over authenticated current Mimi by both independent "
            "automated judges will transfer higher-precision JA-to-EN corrections "
            "with less general and safety drift than v3 absolute-target SFT."
        ),
        "teacher_authentication": (
            "Codex cached ChatGPT authentication; no OpenAI API key used"
        ),
        "human_reviewers_used": False,
        "private_reasoning_traces_used": False,
        "parent": {
            "path": str(PARENT),
            "repository": parent_manifest["student_repository"],
            "revision": parent_manifest["student_revision"],
            "license": parent_manifest["license"],
            "model_sha256": sha256(PARENT / "model.safetensors"),
            "training_manifest": record(PARENT / "mimi_training_manifest.json"),
        },
        "dataset": {
            "directory": str(DATASET),
            "files": {
                name: record(DATASET / name)
                for name in ("manifest.json", "train.jsonl", "valid.jsonl")
            },
            "pairs": {
                "selected": 61,
                "train": 51,
                "valid": 10,
            },
            "effective_licenses": effective_licenses,
            "review_policy": review,
            "decontamination": dataset_manifest["decontamination"],
        },
        "training": TRAINING,
        "internal_selection_gate": {
            "selection": (
                "highest validation relative-pair accuracy, then relative margin, "
                "then lower loss"
            ),
            "relative_pair_accuracy_minimum": 0.8,
            "relative_margin_minimum_exclusive": 0.0,
            "checkpoint_steps": [10, 20, 30, 40],
            "step_zero_is_not_a_trained_candidate": True,
        },
        "conversion": {
            "format": "MLX affine q4",
            "bits": 4,
            "group_size": 64,
            "exact_shipping_quantization_required_before_claim": True,
            "unchanged_en_ja_current_model_plus_candidate_ja_en": True,
            "preferred_two_direction_bundle_maximum_bytes": 150_000_000,
            "hard_two_direction_bundle_maximum_bytes": 500_000_000,
        },
        "held_out_evaluation": {
            "benchmarks": {str(path): record(path) for path in BENCHMARKS},
            "baseline": "current Mimi exact q4 pair",
            "candidate": (
                "current Mimi EN-to-JA exact q4 plus selected v4 JA-to-EN exact q4"
            ),
            "quality_each_direction": {
                "minimum_improvement_signals": 2,
                "mean_sentence_chrf_pp_delta_minimum": 0.25,
                "chrf_pp_paired_90pct_lower_minimum": -0.25,
                "comet22_mean_delta_minimum_for_signal": 0.002,
                "comet22_paired_90pct_lower_minimum": -0.005,
                "independent_judge_mean_delta_minimum_for_signal": 0.1,
                "independent_judge_paired_90pct_lower_minimum": -0.25,
                "sacrebleu_corpus_regression_maximum": 0.1,
                "maximum_domain_chrf_pp_regression": 0.5,
                "maximum_direct_chrf_pp_regression": 0.5,
            },
            "safety": {
                "new_union_critical_errors_maximum": 0,
                "new_negation_errors_maximum": 0,
                "new_number_date_unit_placeholder_errors_maximum": 0,
                "new_repetition_or_nontermination_failures_maximum": 0,
                "new_judge_critical_errors_maximum": 0,
            },
            "runtime": {
                "warm_segment_p95_seconds_maximum_each_direction": 0.175,
                "peak_resident_bytes_maximum": 250_000_000,
            },
            "documents": {
                "segment_then_join_and_direct_evaluation_required": True,
                "long_document_slices_required": True,
            },
            "licensing": {
                "complete_distribution_provenance_required": True,
                "per_row_attribution_sidecar_required": True,
                "public_hugging_face_only_after_all_gates": True,
            },
        },
        "implementation": {
            path.stem.replace(
                "train_marian_automated_preference_adapter", "trainer"
            ): record(path)
            for path in IMPLEMENTATION
        },
        "evidence": {
            "rejected_v3_result": record(v3_result_path),
            "v3_ja_en_document_chrf_delta": v3_result["directions"]["ja-en"][
                "document"
            ]["pairedChrFPlusPlus"]["meanDelta"],
            "v3_ja_en_bleu_delta": (
                v3_result["directions"]["ja-en"]["document"]["candidate"][
                    "sacreBLEUIntl"
                ]
                - v3_result["directions"]["ja-en"]["document"]["baseline"][
                    "sacreBLEUIntl"
                ]
            ),
        },
        "promotion_authorized": False,
        "app_change_authorized": False,
        "public_upload_authorized": False,
        "next_step_if_internal_gate_passes": (
            "merge the selected adapter into the frozen parent, convert once to exact "
            "MLX q4/group-64, and run the unchanged held-out promotion suite"
        ),
        "stop_rule": (
            "Stop without held-out evaluation if no trained checkpoint passes the "
            "internal preference gate. Stop without integration or upload if any "
            "held-out quality, safety, runtime, size, or licensing gate fails."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output)}, indent=2))


if __name__ == "__main__":
    main()
