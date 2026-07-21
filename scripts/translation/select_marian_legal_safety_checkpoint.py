#!/usr/bin/env python3
"""Apply a preregistered legal/safety contract to clean Marian checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import sacrebleu


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def authenticate(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256(path) != expected:
        raise SystemExit(f"{label} hash mismatch: {path}")


def suite_rows(path: Path) -> dict[str, dict]:
    output = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (row.get("sourceLanguage"), row.get("targetLanguage")) != (
            "en-US",
            "ja-JP",
        ):
            continue
        case_id = str(row.get("id", ""))
        if not case_id or case_id in output:
            raise SystemExit(f"suite has empty or duplicate EN-to-JA ID: {case_id}")
        if row.get("claimEligible") is not False or not row.get("references"):
            raise SystemExit(f"suite row is not reference-bearing development data: {case_id}")
        output[case_id] = row
    if not output:
        raise SystemExit("suite has no EN-to-JA rows")
    return output


def report_rows(path: Path, suite: dict[str, dict]) -> tuple[dict, dict[str, dict]]:
    report = load(path)
    rows = {}
    for row in report.get("results", []):
        case_id = str(row.get("caseID", ""))
        if not case_id or case_id in rows:
            raise SystemExit(f"report has empty or duplicate case ID: {path}: {case_id}")
        rows[case_id] = row
    if set(rows) != set(suite):
        raise SystemExit(f"report does not cover the exact EN-to-JA suite: {path}")
    for case_id, expected in suite.items():
        for field in (
            "sourceLanguage",
            "targetLanguage",
            "domain",
            "source",
            "references",
            "claimEligible",
        ):
            if rows[case_id].get(field) != expected.get(field):
                raise SystemExit(f"report disagrees with suite {field}: {path}: {case_id}")
        if not str(rows[case_id].get("hypothesis", "")).strip():
            raise SystemExit(f"report has empty hypothesis: {path}: {case_id}")
    return report, rows


def corpus_chrf(rows: dict[str, dict]) -> float:
    ordered = [rows[key] for key in sorted(rows)]
    hypotheses = [row["hypothesis"] for row in ordered]
    reference_count = max(len(row["references"]) for row in ordered)
    references = [
        [row["references"][min(index, len(row["references"]) - 1)] for row in ordered]
        for index in range(reference_count)
    ]
    return float(sacrebleu.corpus_chrf(hypotheses, references, word_order=2).score)


def sentence_delta(candidate: dict, baseline: dict) -> float:
    references = candidate["references"]
    return float(
        sacrebleu.sentence_chrf(
            candidate["hypothesis"], references, word_order=2
        ).score
        - sacrebleu.sentence_chrf(
            baseline["hypothesis"], references, word_order=2
        ).score
    )


def bootstrap(values: list[float], samples: int, seed: int) -> dict:
    rng = random.Random(seed)
    means = sorted(
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(samples)
    )
    return {
        "mean": sum(values) / len(values),
        "lower": means[math.floor(samples * 0.025)],
        "upper": means[min(samples - 1, math.ceil(samples * 0.975) - 1)],
        "samples": samples,
        "seed": seed,
        "confidence": 0.95,
    }


def audit_counts(path: Path, report_path: Path, cases: int) -> dict[str, int]:
    audit = load(path)
    if Path(str(audit.get("report"))).resolve() != report_path.resolve():
        raise SystemExit(f"structure audit is bound to another report: {path}")
    if audit.get("cases") != cases:
        raise SystemExit(f"structure audit case count differs: {path}")
    return {
        "exactCriticalTokenMismatches": int(
            audit.get("exactCriticalTokenAudit", {}).get("flaggedCases", -1)
        ),
        "negationMarkerMismatches": int(
            audit.get("reasonCounts", {}).get("negation-marker", 0)
        ),
        "flaggedCases": int(audit.get("flaggedCases", -1)),
    }


def validate_continuation_manifest(manifest: dict, continuation: dict) -> None:
    initial = manifest.get("initial_checkpoint", {})
    preservation = manifest.get("preservation_checkpoint", {})
    if (
        initial.get("model_sha256") != continuation["initialCheckpointModelSha256"]
        or initial.get("training_manifest_sha256")
        != continuation["initialCheckpointTrainingManifestSha256"]
        or preservation.get("model_sha256")
        != continuation["initialCheckpointModelSha256"]
        or preservation.get("training_manifest_sha256")
        != continuation["initialCheckpointTrainingManifestSha256"]
        or manifest.get("dataset_manifest", {}).get("sha256")
        != continuation["datasetManifestSha256"]
        or manifest.get("dataset_manifest", {}).get("target_source")
        != continuation["targetSource"]
        or manifest.get("objective", {}).get("sequence_target")
        != "licensed human-authored or project-owned parallel reference translation"
    ):
        raise SystemExit("candidate continuation lineage or target provenance differs")
    hyperparameters = manifest.get("hyperparameters", {})
    expected = {
        "seed": continuation["seed"],
        "batch_size": continuation["batchSize"],
        "gradient_accumulation": continuation["gradientAccumulation"],
        "max_steps": continuation["maximumContinuationSteps"],
        "learning_rate": continuation["learningRate"],
        "weight_decay": continuation["weightDecay"],
        "warmup_steps": continuation["warmupSteps"],
        "evaluation_steps": continuation["evaluationSteps"],
        "max_source_tokens": continuation["maximumSourceTokens"],
        "max_target_tokens": continuation["maximumTargetTokens"],
        "frozen_base_kl_weight": continuation["frozenBaseKlWeight"],
        "l2_to_base_weight": continuation["l2ToBaseWeight"],
        "domain_loss_weight_start": continuation["domainLossWeightStart"],
        "domain_loss_weight_end": continuation["domainLossWeightEnd"],
        "curriculum_ramp_steps": continuation["curriculumRampSteps"],
        "preservation_origins": continuation["preservationOrigins"],
    }
    if any(hyperparameters.get(key) != value for key, value in expected.items()):
        raise SystemExit("candidate continuation hyperparameters differ from contract")


def checkpoint_metrics(
    path: Path, expected_step: int, continuation: dict | None
) -> tuple[dict, float]:
    manifest_path = path / "mimi_training_manifest.json"
    model_path = path / "model.safetensors"
    manifest = load(manifest_path)
    step = int(manifest.get("checkpoint_step", manifest.get("best", {}).get("step", -1)))
    metrics = manifest.get("checkpoint_metrics") or manifest.get("best")
    if step != expected_step or not isinstance(metrics, dict):
        raise SystemExit(f"checkpoint step or metrics differ: {path}")
    if continuation is not None:
        validate_continuation_manifest(manifest, continuation)
    return {
        "path": str(path),
        "modelSha256": sha256(model_path),
        "trainingManifestSha256": sha256(manifest_path),
    }, float(metrics["chrf_pp"])


def validate_quantized_model(
    path: Path,
    checkpoint_model_sha: str,
    report: dict,
    quantization: dict,
    evaluation_implementation: dict | None,
) -> dict:
    manifest_path = path / "manifest.json"
    manifest = load(manifest_path)
    if (
        manifest.get("source_weights_sha256") != checkpoint_model_sha
        or manifest.get("bits") != quantization["bits"]
        or manifest.get("group_size") != quantization["groupSize"]
    ):
        raise SystemExit(f"quantized checkpoint binding differs: {path}")
    weights_path = path / "model.safetensors"
    weights_record = manifest.get("files", {}).get("model.safetensors")
    if not weights_path.is_file() or not isinstance(weights_record, dict) or (
        weights_record.get("bytes") != weights_path.stat().st_size
        or weights_record.get("sha256") != sha256(weights_path)
    ):
        raise SystemExit(f"quantized checkpoint weight integrity differs: {path}")
    declared = report.get("declaredModels", {}).get("en-ja")
    if not isinstance(declared, dict) or (
        Path(str(declared.get("path", ""))).resolve() != path.resolve()
        or declared.get("manifestSha256") != sha256(manifest_path)
        or declared.get("sourceWeightsSha256") != checkpoint_model_sha
        or declared.get("quantizedWeightsSha256") != weights_record["sha256"]
    ):
        raise SystemExit(f"report does not bind the quantized EN-to-JA checkpoint: {path}")
    mlx_version = report.get("runtimeImplementation", {}).get("packages", {}).get("mlx")
    if mlx_version != quantization["mlxVersion"]:
        raise SystemExit(f"report uses an unregistered MLX version: {path}")
    if evaluation_implementation is not None:
        runtime = report.get("runtimeImplementation", {})
        if (
            runtime.get("benchmarkScriptSha256")
            != evaluation_implementation["benchmarkScriptSha256"]
            or runtime.get("marianRuntimeSha256")
            != evaluation_implementation["marianRuntimeSha256"]
        ):
            raise SystemExit(f"report uses an unregistered benchmark implementation: {path}")
    return {
        "path": str(path),
        "manifestSha256": sha256(manifest_path),
        "quantizedWeightsSha256": weights_record["sha256"],
    }


def input_record(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("contract", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=5,
        metavar=("STEP", "CHECKPOINT", "Q4_MODEL", "REPORT", "STRUCTURE_AUDIT"),
        required=True,
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")
    contract = load(args.contract)
    if (
        contract.get("doesNotAuthorizeModelPromotion") is not True
        or contract.get("selectionUsesPublicStressV3Outputs") is not False
    ):
        raise SystemExit("selection contract lacks fail-closed policy")
    selection_implementation = contract.get("selectionImplementation")
    if selection_implementation is not None:
        current_script = Path(__file__).resolve()
        if (
            Path(selection_implementation["path"]).resolve() != current_script
            or selection_implementation["sha256"] != sha256(current_script)
        ):
            raise SystemExit("selection implementation differs from contract")
    evaluation_implementation = contract.get("evaluationImplementation")
    if evaluation_implementation is not None:
        script_directory = Path(__file__).resolve().parent
        authenticate(
            script_directory / "run_mlx_marian_benchmark.py",
            evaluation_implementation["benchmarkScriptSha256"],
            "benchmark implementation",
        )
        authenticate(
            script_directory / "marian_mlx.py",
            evaluation_implementation["marianRuntimeSha256"],
            "Marian runtime implementation",
        )
        authenticate(
            script_directory / "audit_translation_structures.py",
            evaluation_implementation["structureAuditScriptSha256"],
            "structure audit implementation",
        )
    suite_path = Path(contract["suite"]["path"])
    suite_manifest_path = Path(contract["suite"]["manifestPath"])
    authenticate(suite_path, contract["suite"]["sha256"], "suite")
    authenticate(
        suite_manifest_path, contract["suite"]["manifestSha256"], "suite manifest"
    )
    suite = suite_rows(suite_path)
    if len(suite) != contract["suite"]["cases"]:
        raise SystemExit("selection suite case count differs")

    continuation = contract.get("continuation")
    if continuation is not None:
        initial_path = Path(continuation["initialCheckpointPath"])
        authenticate(
            initial_path / "model.safetensors",
            continuation["initialCheckpointModelSha256"],
            "continuation initial weights",
        )
        authenticate(
            initial_path / "mimi_training_manifest.json",
            continuation["initialCheckpointTrainingManifestSha256"],
            "continuation initial training manifest",
        )
        authenticate(
            Path(continuation["datasetDirectory"]) / "manifest.json",
            continuation["datasetManifestSha256"],
            "continuation dataset manifest",
        )
        if (
            continuation.get("syntheticTargetsUsed") is not False
            or continuation.get("privateReasoningTracesUsed") is not False
            or continuation.get("preservationCheckpoint") != "same-as-initial-checkpoint"
        ):
            raise SystemExit("continuation contract lacks clean fail-closed policy")

    baseline_spec = contract["baseline"]
    baseline_report_path = Path(baseline_spec["reportPath"])
    baseline_audit_path = Path(baseline_spec["structureAuditPath"])
    baseline_training_path = Path(baseline_spec["trainingManifestPath"])
    authenticate(baseline_report_path, baseline_spec["reportSha256"], "baseline report")
    authenticate(
        baseline_audit_path,
        baseline_spec["structureAuditSha256"],
        "baseline structure audit",
    )
    authenticate(
        baseline_training_path,
        baseline_spec["trainingManifestSha256"],
        "baseline training manifest",
    )
    baseline_training = load(baseline_training_path)
    if float(baseline_training.get("best", {}).get("chrf_pp", float("nan"))) != float(
        baseline_spec["bestGeneralDevelopmentChrFPlusPlus"]
    ):
        raise SystemExit("baseline general-development score differs from contract")
    baseline_report, baseline = report_rows(baseline_report_path, suite)
    if evaluation_implementation is not None:
        runtime = baseline_report.get("runtimeImplementation", {})
        if (
            runtime.get("benchmarkScriptSha256")
            != evaluation_implementation["benchmarkScriptSha256"]
            or runtime.get("marianRuntimeSha256")
            != evaluation_implementation["marianRuntimeSha256"]
        ):
            raise SystemExit("baseline report uses an unregistered implementation")
    baseline_counts = audit_counts(baseline_audit_path, baseline_report_path, len(suite))
    if (
        baseline_counts["exactCriticalTokenMismatches"]
        != baseline_spec["exactCriticalTokenMismatches"]
        or baseline_counts["negationMarkerMismatches"]
        != baseline_spec["negationMarkerMismatches"]
    ):
        raise SystemExit("baseline structure counts differ from contract")

    requirements = contract["candidateRequirements"]
    bootstrap_contract = requirements["pairedSentenceChrFPlusPlusBootstrap"]
    expected_steps = {int(value) for value in contract["candidateSteps"]}
    candidates = []
    seen_steps: set[int] = set()
    for step_value, checkpoint_value, q4_value, report_value, audit_value in args.candidate:
        step = int(step_value)
        if step not in expected_steps or step in seen_steps:
            raise SystemExit(f"unexpected or duplicate candidate step: {step}")
        seen_steps.add(step)
        checkpoint = Path(checkpoint_value)
        q4_model = Path(q4_value)
        report_path = Path(report_value)
        audit_path = Path(audit_value)
        report, rows = report_rows(report_path, suite)
        checkpoint_record, general_chrf = checkpoint_metrics(
            checkpoint, step, continuation
        )
        q4_record = validate_quantized_model(
            q4_model,
            checkpoint_record["modelSha256"],
            report,
            requirements["quantization"],
            evaluation_implementation,
        )
        counts = audit_counts(audit_path, report_path, len(suite))
        deltas = [
            sentence_delta(rows[case_id], baseline[case_id])
            for case_id in sorted(suite)
        ]
        interval = bootstrap(
            deltas,
            int(bootstrap_contract["samples"]),
            int(bootstrap_contract["seed"]),
        )
        gates = {
            "legalChrFNonInferiority": interval["lower"]
            >= float(bootstrap_contract["minimumLowerBound"]),
            "exactCriticalTokenNonRegression": counts[
                "exactCriticalTokenMismatches"
            ]
            <= int(requirements["maximumExactCriticalTokenMismatches"]),
            "negationNonRegression": counts["negationMarkerMismatches"]
            <= int(requirements["maximumNegationMarkerMismatches"]),
            "generalDevelopmentRetention": general_chrf
            >= float(requirements["minimumGeneralDevelopmentChrFPlusPlus"]),
        }
        candidates.append(
            {
                "step": step,
                "checkpoint": checkpoint_record,
                "quantizedModel": q4_record,
                "report": input_record(report_path),
                "structureAudit": input_record(audit_path),
                "generalDevelopmentChrFPlusPlus": general_chrf,
                "legalValidationCorpusChrFPlusPlus": corpus_chrf(rows),
                "pairedSentenceChrFPlusPlusDelta": interval,
                "structure": counts,
                "gates": gates,
                "eligible": all(gates.values()),
            }
        )
    if seen_steps != expected_steps:
        raise SystemExit(f"candidate steps differ: expected {sorted(expected_steps)}")

    eligible = [value for value in candidates if value["eligible"]]
    ranked = sorted(
        eligible,
        key=lambda value: (
            -value["legalValidationCorpusChrFPlusPlus"],
            value["structure"]["exactCriticalTokenMismatches"],
            value["structure"]["negationMarkerMismatches"],
            -value["generalDevelopmentChrFPlusPlus"],
            value["step"],
        ),
    )
    selected = ranked[0] if ranked else None
    payload = {
        "schemaVersion": 1,
        "status": "research-checkpoint-selected" if selected else "clean-checkpoint-family-rejected",
        "contract": input_record(args.contract),
        "selectionImplementation": input_record(Path(__file__).resolve()),
        "baseline": {
            "report": input_record(baseline_report_path),
            "structureAudit": input_record(baseline_audit_path),
            "generalDevelopmentChrFPlusPlus": baseline_spec[
                "bestGeneralDevelopmentChrFPlusPlus"
            ],
            "legalValidationCorpusChrFPlusPlus": corpus_chrf(baseline),
            "structure": baseline_counts,
        },
        "candidates": sorted(candidates, key=lambda value: value["step"]),
        "selectedStep": selected["step"] if selected else None,
        "claimEligible": False,
        "doesNotAuthorizeModelPromotion": True,
        "doesNotAuthorizeAppIntegration": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "selectedStep": payload["selectedStep"],
                "eligibleSteps": [value["step"] for value in ranked],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
