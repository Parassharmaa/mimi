#!/usr/bin/env python3
"""End-to-end contract test for the phase-1 student continuation evaluator."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = ROOT / "scripts/translation/evaluate_gpt56_student_continuation.py"
AUDITOR = ROOT / "scripts/translation/audit_translation_structures.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def report(
    rows: list[dict],
    hypotheses: dict[str, str],
    *,
    engine: str,
    revision: str,
    model_bytes: int,
    direct: bool = False,
) -> dict:
    results = []
    for row in rows:
        result = {
            "caseID": row["id"],
            **{key: row[key] for key in (
                "sourceLanguage",
                "targetLanguage",
                "domain",
                "source",
                "references",
                "claimEligible",
            )},
            "hypothesis": hypotheses[row["id"]],
            "latencySeconds": 0.1,
            "warmLatencySeconds": [0.1],
            "outputTokenIDs": [2, 3, 4],
        }
        if not direct and row.get("sourceUnit") == "document":
            result.update(
                {
                    "sourceUnit": "document",
                    "segmentCount": 1,
                    "segmentHypotheses": [hypotheses[row["id"]]],
                    "emptySegmentCount": 0,
                    "emptySegmentIndexes": [],
                    "segmentBenchmarkIDs": [f"{row['id']}:segment-01"],
                    "segmentOutputTokenIDs": [[2, 3, 4]],
                }
            )
        results.append(result)
    return {
        "schemaVersion": 1,
        "engine": engine,
        "modelRevision": revision,
        "hardware": "fixture-apple-silicon",
        "operatingSystem": "fixture-macos",
        "peakResidentBytes": 1_000_000,
        "modelBytes": model_bytes,
        "benchmarkConfiguration": {
            "maximumGeneratedTokens": 192,
            "warmRunsPerCase": 1,
        },
        "results": results,
    }


def dataset(root: Path, direction: str, protected_hashes: list[str]) -> Path:
    directory = root / f"dataset-{direction}"
    directory.mkdir()
    common = {
        "source_license": "CC-BY-4.0",
        "source_provenance": "fixture licensed corpus",
        "promotion_eligible": True,
    }
    train = [
        {
            **common,
            "id": f"{direction}-synthetic",
            "source": "source",
            "target": "target",
            "origin": "automated-gpt-teacher-reference-anchored",
            "judge_model_ids": ["fixture-admission-a", "fixture-admission-b"],
        },
        *[
            {
                **common,
                "id": f"{direction}-human-{index}",
                "source": f"human source {index}",
                "target": f"human target {index}",
                "origin": "licensed-human-reference-anchor",
            }
            for index in range(3)
        ],
    ]
    valid = [
        {
            **common,
            "id": f"{direction}-valid",
            "source": "valid source",
            "target": "valid target",
            "origin": "licensed-human-reference",
        }
    ]
    train_path, valid_path = directory / "train.jsonl", directory / "valid.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(valid_path, valid)
    manifest = {
        "schema_version": 1,
        "direction": direction,
        "promotion_eligible": True,
        "private_reasoning_traces_used": False,
        "synthetic_policy": {
            "review_status": "two-judge-reference-anchored",
            "source_only_provisional_rows_allowed": False,
            "licensed_reference_anchor_required": True,
            "same_source_human_anchor_required": True,
            "human_replay_per_synthetic": 3,
            "actual_synthetic_fraction": 0.25,
        },
        "outputs": {
            "train": {"path": str(train_path), "sha256": sha256(train_path)},
            "valid": {"path": str(valid_path), "sha256": sha256(valid_path)},
        },
        "decontamination": {
            "protected_suites": [
                {"path": f"fixture-{index}", "sha256": value}
                for index, value in enumerate(protected_hashes)
            ],
            "train_validation_exact_overlap": False,
        },
    }
    path = directory / "manifest.json"
    write_json(path, manifest)
    return path


def checkpoint(
    root: Path,
    direction: str,
    dataset_manifest: Path,
    initial_hash: str,
    phase: dict,
) -> Path:
    directory = root / f"checkpoint-{direction}"
    directory.mkdir()
    weights = directory / "model.safetensors"
    weights.write_bytes(f"phase-1-{direction}".encode())
    hyperparameters = {
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
    write_json(
        directory / "mimi_training_manifest.json",
        {
            "schema_version": 1,
            "direction": direction,
            "initial_checkpoint": {"model_sha256": initial_hash},
            "dataset_manifest": {
                "sha256": sha256(dataset_manifest),
                "outputs_authenticated": True,
            },
            "hyperparameters": hyperparameters,
        },
    )
    return directory


def bundle(root: Path, checkpoint_paths: dict[str, Path]) -> tuple[Path, int, str]:
    directory = root / "bundle"
    directory.mkdir()
    for direction, checkpoint_path in checkpoint_paths.items():
        child = directory / direction
        child.mkdir()
        (child / "model.safetensors").write_bytes(f"q4-{direction}".encode())
        write_json(
            child / "manifest.json",
            {
                "direction": direction,
                "bits": 4,
                "group_size": 64,
                "source_weights_sha256": sha256(
                    checkpoint_path / "model.safetensors"
                ),
            },
        )
    files = {
        item.relative_to(directory).as_posix(): {
            "bytes": item.stat().st_size,
            "sha256": sha256(item),
        }
        for item in sorted(directory.rglob("*"))
        if item.is_file()
    }
    write_json(
        directory / "manifest.json",
        {
            "format": "mimi-mlx-marian-pair-v1",
            "engines": ["en-ja", "ja-en"],
            "quantization": {"bits": 4, "group_size": 64},
            "license": "CC-BY-SA-4.0",
            "files": files,
        },
    )
    return (
        directory,
        sum(record["bytes"] for record in files.values()),
        f"pair-manifest-sha256:{sha256(directory / 'manifest.json')}",
    )


def comet(
    path: Path,
    report_path: Path,
    suite_path: Path,
    engine: str,
    rows: list[dict],
    score: float,
) -> None:
    fields = {
        "metric": "COMET-22",
        "modelRepository": "fixture/comet",
        "modelRevision": "fixture-revision",
        "modelLicense": "Apache-2.0",
        "package": "unbabel-comet",
        "packageVersion": "2.2.7",
        "setuptoolsVersion": "80.9.0",
        "precision": "float32",
        "multipleReferenceAggregation": "mean",
    }
    signature = hashlib.sha256(
        json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    write_json(
        path,
        {
            "schemaVersion": 1,
            **fields,
            "signatureSHA256": signature,
            "engine": engine,
            "suiteSHA256": sha256(suite_path),
            "engineReportSHA256": sha256(report_path),
            "results": [
                {
                    "caseID": row["id"],
                    "sourceLanguage": row["sourceLanguage"],
                    "targetLanguage": row["targetLanguage"],
                    "domain": row["domain"],
                    "score": score,
                }
                for row in rows
            ],
        },
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-gpt56-continuation-") as temporary:
        root = Path(temporary)
        cases = [
            {
                "id": "en-a",
                "sourceLanguage": "en-US",
                "targetLanguage": "ja-JP",
                "domain": "news",
                "source": "Good morning.",
                "references": ["おはようございます。"],
                "claimEligible": False,
                "sourceUnit": "document",
            },
            {
                "id": "en-b",
                "sourceLanguage": "en-US",
                "targetLanguage": "ja-JP",
                "domain": "legal",
                "source": "Please proceed.",
                "references": ["続行してください。"],
                "claimEligible": False,
                "sourceUnit": "document",
            },
            {
                "id": "ja-a",
                "sourceLanguage": "ja-JP",
                "targetLanguage": "en-US",
                "domain": "news",
                "source": "おはようございます。",
                "references": ["Good morning."],
                "claimEligible": False,
                "sourceUnit": "document",
            },
            {
                "id": "ja-b",
                "sourceLanguage": "ja-JP",
                "targetLanguage": "en-US",
                "domain": "legal",
                "source": "続行してください。",
                "references": ["Please proceed."],
                "claimEligible": False,
                "sourceUnit": "document",
            },
        ]
        segments = [
            {
                **row,
                "id": f"{row['id']}:segment-01",
                "parentCaseID": row["id"],
                "sourceUnit": "document-segment",
            }
            for row in cases
        ]
        direct = [{**row, "directSourceTokenCount": 4} for row in cases]
        case_path, segment_path, direct_path = (
            root / "cases.jsonl",
            root / "segments.jsonl",
            root / "direct.jsonl",
        )
        write_jsonl(case_path, cases)
        write_jsonl(segment_path, segments)
        write_jsonl(direct_path, direct)

        phase = {
            "seed": 20260725,
            "batch_size": 8,
            "gradient_accumulation": 4,
            "max_steps": 250,
            "learning_rate": 0.000002,
            "weight_decay": 0.01,
            "warmup_steps": 25,
            "evaluation_steps": 250,
            "max_source_tokens": 192,
            "max_target_tokens": 192,
            "frozen_base_kl_weight": 0.1,
            "l2_to_base_weight": 0.01,
        }
        initial_hashes = {
            direction: hashlib.sha256(f"initial-{direction}".encode()).hexdigest()
            for direction in ("en-ja", "ja-en")
        }
        datasets = {
            direction: dataset(
                root, direction, [sha256(case_path), sha256(segment_path)]
            )
            for direction in ("en-ja", "ja-en")
        }
        checkpoints = {
            direction: checkpoint(
                root, direction, datasets[direction], initial_hashes[direction], phase
            )
            for direction in ("en-ja", "ja-en")
        }
        bundle_path, model_bytes, candidate_revision = bundle(root, checkpoints)
        baseline_revision = "fixture-baseline-revision"
        candidate_hypotheses = {row["id"]: row["references"][0] for row in cases}
        baseline_hypotheses = {
            row["id"]: ("不適切。" if row["targetLanguage"] == "ja-JP" else "Poor.")
            for row in cases
        }
        candidate_segment_hypotheses = {
            f"{row['id']}:segment-01": candidate_hypotheses[row["id"]]
            for row in cases
        }
        baseline_segment_hypotheses = {
            f"{row['id']}:segment-01": baseline_hypotheses[row["id"]]
            for row in cases
        }

        paths = {
            name: root / f"{name}.json"
            for name in (
                "baseline_document",
                "candidate_document",
                "baseline_segment",
                "candidate_segment",
                "baseline_direct",
                "candidate_direct",
                "baseline_comet",
                "candidate_comet",
                "baseline_structure",
                "candidate_structure",
                "judge",
                "judge_b",
                "contract",
                "result",
            )
        }
        write_json(
            paths["baseline_segment"],
            report(
                segments,
                baseline_segment_hypotheses,
                engine="baseline",
                revision=baseline_revision,
                model_bytes=10,
                direct=True,
            ),
        )
        write_json(
            paths["candidate_segment"],
            report(
                segments,
                candidate_segment_hypotheses,
                engine="candidate",
                revision=candidate_revision,
                model_bytes=model_bytes,
                direct=True,
            ),
        )
        for label, hypotheses, revision, engine, bytes_value, segment_path_value in (
            (
                "baseline",
                baseline_hypotheses,
                baseline_revision,
                "baseline:segment-then-join",
                10,
                paths["baseline_segment"],
            ),
            (
                "candidate",
                candidate_hypotheses,
                candidate_revision,
                "candidate:segment-then-join",
                model_bytes,
                paths["candidate_segment"],
            ),
        ):
            value = report(
                cases,
                hypotheses,
                engine=engine,
                revision=revision,
                model_bytes=bytes_value,
            )
            value["sourceSegmentReportSHA256"] = sha256(segment_path_value)
            value["caseSuiteSHA256"] = sha256(case_path)
            value["documentAggregation"] = {
                "strategy": "translate-segments-independently-then-join-with-newline",
                "crossSegmentContext": False,
            }
            write_json(paths[f"{label}_document"], value)
            write_json(
                paths[f"{label}_direct"],
                report(
                    direct,
                    hypotheses,
                    engine=f"{label}:direct",
                    revision=revision,
                    model_bytes=bytes_value,
                    direct=True,
                ),
            )

        comet(
            paths["baseline_comet"],
            paths["baseline_document"],
            case_path,
            "baseline:segment-then-join",
            cases,
            0.5,
        )
        comet(
            paths["candidate_comet"],
            paths["candidate_document"],
            case_path,
            "candidate:segment-then-join",
            cases,
            0.9,
        )
        for label in ("baseline", "candidate"):
            audit = subprocess.run(
                [
                    "python3",
                    str(AUDITOR),
                    str(paths[f"{label}_document"]),
                    str(paths[f"{label}_structure"]),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            assert audit.returncode == 0, audit.stderr

        def judge(
            path: Path,
            *,
            model: str,
            family: str,
            candidate_critical: bool,
        ) -> None:
            write_json(
                path,
                {
                    "schemaVersion": 1,
                    "purpose": "blinded-automated-quality-and-critical-comparison",
                    "suiteSHA256": sha256(case_path),
                    "candidateReportSHA256": sha256(paths["candidate_document"]),
                    "baselineReportSHA256": sha256(paths["baseline_document"]),
                    "reasoningTracesStored": False,
                    "blinded": True,
                    "judgeModel": model,
                    "judgeModelFamily": family,
                    "judgeRevision": "fixture-revision",
                    "promptSHA256": "a" * 64,
                    "results": [
                        {
                            "caseID": row["id"],
                            "candidateHypothesisSHA256": text_hash(
                                candidate_hypotheses[row["id"]]
                            ),
                            "baselineHypothesisSHA256": text_hash(
                                baseline_hypotheses[row["id"]]
                            ),
                            "candidate": {
                                "adequacy": 4,
                                "fluency": 4,
                                "terminology": 2,
                                "criticalError": candidate_critical
                                and row["id"] == "en-a",
                                "errorTags": (
                                    ["meaning"]
                                    if candidate_critical and row["id"] == "en-a"
                                    else []
                                ),
                            },
                            "baseline": {
                                "adequacy": 2,
                                "fluency": 2,
                                "terminology": 1,
                                "criticalError": False,
                                "errorTags": [],
                            },
                        }
                        for row in cases
                    ],
                },
            )

        judge(
            paths["judge"],
            model="fixture-eval-a",
            family="fixture-eval-family-a",
            candidate_critical=False,
        )
        judge(
            paths["judge_b"],
            model="fixture-eval-b",
            family="fixture-eval-family-b",
            candidate_critical=False,
        )
        contract = {
            "schema_version": 1,
            "experiment": "gpt56-final-translation-intact-marian-pilot-v1",
            "promotion_authorized": False,
            "app_change_authorized": False,
            "teacher": {"requests": {"model": "fixture-teacher"}},
            "student": {
                "phase_1": phase,
                "initial_checkpoints": {
                    direction: {
                        "files": {
                            "model.safetensors": {"sha256": initial_hashes[direction]}
                        }
                    }
                    for direction in ("en-ja", "ja-en")
                },
            },
            "benchmarks": {
                "development_cases": {"sha256": sha256(case_path)},
                "development_segments": {"sha256": sha256(segment_path)},
                "direct_within_limit_cases": {"sha256": sha256(direct_path)},
                "development_baseline_document_report": {
                    "sha256": sha256(paths["baseline_document"])
                },
                "development_baseline_segment_report": {
                    "sha256": sha256(paths["baseline_segment"])
                },
                "development_baseline_direct_report": {
                    "sha256": sha256(paths["baseline_direct"])
                },
                "development_baseline_comet_report": {
                    "sha256": sha256(paths["baseline_comet"])
                },
                "development_baseline_structure_audit": {
                    "sha256": sha256(paths["baseline_structure"])
                },
                "development_baseline": {"model_revision": baseline_revision},
            },
            "implementation": {
                "scripts/translation/evaluate_gpt56_student_continuation.py": {
                    "sha256": sha256(EVALUATOR)
                }
            },
            "continuation_gates": {
                "statistics": {
                    "paired_bootstrap_samples": 200,
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
                    "sacrebleu_corpus_regression_maximum": 0.1,
                    "maximum_domain_chrf_pp_regression": 0.5,
                    "maximum_direct_chrf_pp_regression": 0.5,
                },
                "runtime": {
                    "warm_segment_p95_seconds_maximum_each_direction": 0.175,
                    "relative_latency_regression_diagnostic_fraction": 0.1,
                    "peak_resident_bytes_maximum": 250_000_000,
                },
                "safety": {
                    "new_union_critical_errors_maximum": 0,
                    "new_negation_errors_maximum": 0,
                    "new_number_date_unit_placeholder_errors_maximum": 0,
                    "automated_judge_critical_error_increase_maximum": 0,
                },
                "long_documents": {
                    "new_repetition_or_nontermination_failures_maximum": 0
                },
                "bundle": {
                    "target_bytes": 150_000_000,
                    "hard_maximum_bytes": 500_000_000,
                },
            },
        }
        write_json(paths["contract"], contract)

        command = [
            "python3",
            str(EVALUATOR),
            str(paths["contract"]),
            str(case_path),
            str(segment_path),
            str(direct_path),
            str(paths["baseline_document"]),
            str(paths["candidate_document"]),
            str(paths["baseline_segment"]),
            str(paths["candidate_segment"]),
            str(paths["baseline_direct"]),
            str(paths["candidate_direct"]),
            str(paths["baseline_comet"]),
            str(paths["candidate_comet"]),
            str(paths["baseline_structure"]),
            str(paths["candidate_structure"]),
            str(paths["judge"]),
            str(paths["judge_b"]),
            str(bundle_path),
            str(checkpoints["en-ja"]),
            str(checkpoints["ja-en"]),
            str(datasets["en-ja"]),
            str(datasets["ja-en"]),
            str(paths["result"]),
        ]
        passed = subprocess.run(
            command, cwd=ROOT, text=True, capture_output=True, check=False
        )
        assert passed.returncode == 0, passed.stderr + passed.stdout
        result = json.loads(paths["result"].read_text())
        assert result["status"] == "phase-1-continuation-approved"
        assert result["continueTraining"] is True
        assert result["promotionAuthorized"] is False

        judge(
            paths["judge_b"],
            model="fixture-eval-a",
            family="fixture-eval-family-a",
            candidate_critical=False,
        )
        collision_path = root / "judge-collision.json"
        collision = subprocess.run(
            [*command[:-1], str(collision_path)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert collision.returncode == 1
        assert "distinct model families" in collision.stderr
        judge(
            paths["judge_b"],
            model="fixture-eval-b",
            family="fixture-eval-family-b",
            candidate_critical=False,
        )

        judge(
            paths["judge"],
            model="fixture-eval-a",
            family="fixture-eval-family-a",
            candidate_critical=True,
        )
        failed_path = root / "failed.json"
        failed = subprocess.run(
            [*command[:-1], str(failed_path)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert failed.returncode == 2, failed.stderr + failed.stdout
        rejected = json.loads(failed_path.read_text())
        assert rejected["continueTraining"] is False
        union_gate = next(
            gate
            for gate in rejected["globalGates"]
            if gate["name"] == "new-union-critical-errors"
        )
        assert union_gate["passed"] is False

    print("GPT-5.6 phase-1 student continuation contract passed.")


if __name__ == "__main__":
    main()
