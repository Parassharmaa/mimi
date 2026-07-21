#!/usr/bin/env python3
"""End-to-end contract test for preregistered legal/safety checkpoint selection."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/translation/select_marian_legal_safety_checkpoint.py"
BENCHMARK_SCRIPT = ROOT / "scripts/translation/run_mlx_marian_benchmark.py"
MARIAN_RUNTIME = ROOT / "scripts/translation/marian_mlx.py"
STRUCTURE_AUDIT_SCRIPT = ROOT / "scripts/translation/audit_translation_structures.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def report_row(hypothesis: str) -> dict:
    return {
        "caseID": "legal-1",
        "sourceLanguage": "en-US",
        "targetLanguage": "ja-JP",
        "domain": "legal",
        "source": "The limit is 12 percent.",
        "references": ["上限は12パーセントとする。"],
        "claimEligible": False,
        "hypothesis": hypothesis,
    }


def audit(path: Path, report: Path) -> None:
    write_json(
        path,
        {
            "report": str(report),
            "cases": 1,
            "flaggedCases": 0,
            "exactCriticalTokenAudit": {"flaggedCases": 0},
            "reasonCounts": {},
        },
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-checkpoint-selection-") as temporary:
        root = Path(temporary)
        suite = root / "suite.jsonl"
        suite.write_text(
            json.dumps(
                {
                    "id": "legal-1",
                    "sourceLanguage": "en-US",
                    "targetLanguage": "ja-JP",
                    "domain": "legal",
                    "source": "The limit is 12 percent.",
                    "references": ["上限は12パーセントとする。"],
                    "claimEligible": False,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        suite_manifest = root / "suite.manifest.json"
        write_json(suite_manifest, {"fixture": True})

        baseline_report = root / "baseline.json"
        runtime = {
            "benchmarkScriptSha256": sha256(BENCHMARK_SCRIPT),
            "marianRuntimeSha256": sha256(MARIAN_RUNTIME),
            "packages": {"mlx": "0.30.6"},
        }
        write_json(
            baseline_report,
            {
                "runtimeImplementation": runtime,
                "results": [report_row("上限は12パーセントとする。")],
            },
        )
        baseline_audit = root / "baseline-audit.json"
        audit(baseline_audit, baseline_report)
        baseline_training = root / "baseline-training.json"
        write_json(baseline_training, {"best": {"chrf_pp": 31.0}})

        initial = root / "initial"
        initial.mkdir()
        (initial / "model.safetensors").write_bytes(b"clean initial fixture")
        write_json(initial / "mimi_training_manifest.json", {"fixture": "clean"})
        initial_model_sha = sha256(initial / "model.safetensors")
        initial_manifest_sha = sha256(initial / "mimi_training_manifest.json")
        dataset = root / "dataset"
        dataset.mkdir()
        write_json(dataset / "manifest.json", {"target_source": "licensed-human-reference"})
        dataset_manifest_sha = sha256(dataset / "manifest.json")

        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "model.safetensors").write_bytes(b"full precision fixture")
        write_json(
            checkpoint / "mimi_training_manifest.json",
            {
                "checkpoint_step": 250,
                "checkpoint_metrics": {"chrf_pp": 30.95},
                "initial_checkpoint": {
                    "model_sha256": initial_model_sha,
                    "training_manifest_sha256": initial_manifest_sha,
                },
                "preservation_checkpoint": {
                    "model_sha256": initial_model_sha,
                    "training_manifest_sha256": initial_manifest_sha,
                },
                "dataset_manifest": {
                    "sha256": dataset_manifest_sha,
                    "target_source": "licensed-human-reference",
                },
                "objective": {
                    "sequence_target": "licensed human-authored or project-owned parallel reference translation"
                },
                "hyperparameters": {
                    "seed": 7,
                    "batch_size": 8,
                    "gradient_accumulation": 2,
                    "max_steps": 250,
                    "learning_rate": 0.000001,
                    "weight_decay": 0.01,
                    "warmup_steps": 10,
                    "evaluation_steps": 250,
                    "max_source_tokens": 128,
                    "max_target_tokens": 128,
                    "frozen_base_kl_weight": 0.25,
                    "l2_to_base_weight": 0.00001,
                    "domain_loss_weight_start": 1.0,
                    "domain_loss_weight_end": 1.0,
                    "curriculum_ramp_steps": 250,
                    "preservation_origins": ["human-kftt-replay"],
                },
            },
        )
        checkpoint_sha = sha256(checkpoint / "model.safetensors")

        q4 = root / "q4"
        q4.mkdir()
        weights = q4 / "model.safetensors"
        weights.write_bytes(b"quantized fixture")
        q4_manifest = q4 / "manifest.json"
        write_json(
            q4_manifest,
            {
                "bits": 4,
                "group_size": 64,
                "source_weights_sha256": checkpoint_sha,
                "files": {
                    "model.safetensors": {
                        "bytes": weights.stat().st_size,
                        "sha256": sha256(weights),
                    }
                },
            },
        )
        candidate_report = root / "candidate.json"
        write_json(
            candidate_report,
            {
                "declaredModels": {
                    "en-ja": {
                        "path": str(q4),
                        "manifestSha256": sha256(q4_manifest),
                        "sourceWeightsSha256": checkpoint_sha,
                        "quantizedWeightsSha256": sha256(weights),
                    }
                },
                "runtimeImplementation": runtime,
                "results": [report_row("上限は12パーセントとする。")],
            },
        )
        candidate_audit = root / "candidate-audit.json"
        audit(candidate_audit, candidate_report)

        contract = root / "contract.json"
        write_json(
            contract,
            {
                "candidateSteps": [250],
                "selectionImplementation": {
                    "path": str(SCRIPT),
                    "sha256": sha256(SCRIPT),
                },
                "evaluationImplementation": {
                    "benchmarkScriptSha256": sha256(BENCHMARK_SCRIPT),
                    "marianRuntimeSha256": sha256(MARIAN_RUNTIME),
                    "structureAuditScriptSha256": sha256(STRUCTURE_AUDIT_SCRIPT),
                },
                "continuation": {
                    "initialCheckpointPath": str(initial),
                    "initialCheckpointModelSha256": initial_model_sha,
                    "initialCheckpointTrainingManifestSha256": initial_manifest_sha,
                    "preservationCheckpoint": "same-as-initial-checkpoint",
                    "datasetDirectory": str(dataset),
                    "datasetManifestSha256": dataset_manifest_sha,
                    "targetSource": "licensed-human-reference",
                    "syntheticTargetsUsed": False,
                    "privateReasoningTracesUsed": False,
                    "seed": 7,
                    "batchSize": 8,
                    "gradientAccumulation": 2,
                    "maximumContinuationSteps": 250,
                    "learningRate": 0.000001,
                    "weightDecay": 0.01,
                    "warmupSteps": 10,
                    "evaluationSteps": 250,
                    "maximumSourceTokens": 128,
                    "maximumTargetTokens": 128,
                    "frozenBaseKlWeight": 0.25,
                    "l2ToBaseWeight": 0.00001,
                    "domainLossWeightStart": 1.0,
                    "domainLossWeightEnd": 1.0,
                    "curriculumRampSteps": 250,
                    "preservationOrigins": ["human-kftt-replay"],
                },
                "suite": {
                    "path": str(suite),
                    "sha256": sha256(suite),
                    "manifestPath": str(suite_manifest),
                    "manifestSha256": sha256(suite_manifest),
                    "cases": 1,
                },
                "baseline": {
                    "reportPath": str(baseline_report),
                    "reportSha256": sha256(baseline_report),
                    "structureAuditPath": str(baseline_audit),
                    "structureAuditSha256": sha256(baseline_audit),
                    "trainingManifestPath": str(baseline_training),
                    "trainingManifestSha256": sha256(baseline_training),
                    "bestGeneralDevelopmentChrFPlusPlus": 31.0,
                    "exactCriticalTokenMismatches": 0,
                    "negationMarkerMismatches": 0,
                },
                "candidateRequirements": {
                    "pairedSentenceChrFPlusPlusBootstrap": {
                        "samples": 100,
                        "seed": 7,
                        "minimumLowerBound": -0.1,
                    },
                    "maximumExactCriticalTokenMismatches": 0,
                    "maximumNegationMarkerMismatches": 0,
                    "minimumGeneralDevelopmentChrFPlusPlus": 30.9,
                    "quantization": {
                        "bits": 4,
                        "groupSize": 64,
                        "mlxVersion": "0.30.6",
                    },
                },
                "selectionUsesPublicStressV3Outputs": False,
                "doesNotAuthorizeModelPromotion": True,
            },
        )
        output = root / "selection.json"
        command = [
            sys.executable,
            str(SCRIPT),
            str(contract),
            str(output),
            "--candidate",
            "250",
            str(checkpoint),
            str(q4),
            str(candidate_report),
            str(candidate_audit),
        ]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        selection = json.loads(output.read_text(encoding="utf-8"))
        assert selection["status"] == "research-checkpoint-selected"
        assert selection["selectedStep"] == 250
        assert selection["candidates"][0]["eligible"] is True
        assert selection["doesNotAuthorizeModelPromotion"] is True

        weights.write_bytes(b"tampered quantized fixture")
        tampered = subprocess.run(
            command[:3] + [str(root / "tampered.json")] + command[4:],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        assert tampered.returncode != 0
        assert "weight integrity differs" in tampered.stderr

    print("Marian legal/safety checkpoint selection contracts passed.")


if __name__ == "__main__":
    main()
