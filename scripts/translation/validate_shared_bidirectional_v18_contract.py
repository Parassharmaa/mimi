#!/usr/bin/env python3
"""Validate the frozen V18 shared-bidirectional phase-one contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def authenticated_file(record: dict, label: str, *, required: bool = True) -> dict:
    path = resolve(str(record.get("path", "")))
    expected = str(record.get("sha256", ""))
    if not path.is_file() and not required:
        return {}
    if not path.is_file() or not expected or sha256(path) != expected:
        raise SystemExit(f"{label} is missing or has the wrong SHA-256: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def authenticated_weights(record: dict, label: str, *, required: bool) -> None:
    directory = resolve(str(record.get("path", "")))
    path = directory / "model.safetensors"
    if not path.is_file() and not required:
        return
    if (
        not path.is_file()
        or sha256(path) != str(record.get("weights_sha256", ""))
    ):
        raise SystemExit(f"{label} weights are missing or unauthenticated: {path}")


def validate(
    contract: dict,
    *,
    require_materialized_dataset: bool,
    require_local_model_artifacts: bool,
) -> None:
    if contract.get("experiment") != "shared-bidirectional-v18-wide-dense-phase1":
        raise SystemExit("unexpected V18 experiment identity")
    if contract.get("status") != "phase1-capacity-control-authorized":
        raise SystemExit("V18 phase one is not in the authorized state")
    if contract.get("training_authorized") is not True:
        raise SystemExit("V18 phase-one training authorization is absent")
    for field in (
        "app_change_authorized",
        "promotion_authorized",
        "public_upload_authorized",
        "private_reasoning_traces_used",
    ):
        if contract.get(field) is not False:
            raise SystemExit(f"V18 contract must keep {field}=false")

    architecture = contract["architecture"]
    if (
        int(architecture["parameters"]) != 92_043_009
        or int(architecture["encoder_ffn_dim"]) != 4_608
        or int(architecture["decoder_ffn_dim"]) != 4_608
        or int(architecture["projected_q4_one_model_pack_bytes"]) > 150_000_000
    ):
        raise SystemExit("V18 architecture or size boundary differs")

    dataset = contract["dataset"]
    if (
        dataset.get("licensed_human_targets_only") is not True
        or dataset.get("promotion_eligible") is not True
        or dataset.get("v17_generated_candidates_used") is not False
        or int(dataset["train_rows"]) != 12_066
        or int(dataset["valid_rows"]) != 2_370
        or len(dataset.get("protected_suites", [])) != 10
    ):
        raise SystemExit("V18 dataset policy/count boundary differs")
    dataset_result = authenticated_file(dataset["result"], "dataset result")
    if (
        dataset_result.get("promotion_eligible") is not True
        or dataset_result.get("licensed_human_targets_only") is not True
        or dataset_result.get("v17_generated_candidates_used") is not False
        or dataset_result.get("training_authorized") is not False
        or dataset_result.get("outputs", {}).get("train", {}).get("sha256")
        != dataset.get("train_sha256")
        or dataset_result.get("outputs", {}).get("valid", {}).get("sha256")
        != dataset.get("valid_sha256")
    ):
        raise SystemExit("V18 dataset result disagrees with the contract")

    materialized = resolve(dataset["manifest"]["materialized_path"])
    if require_materialized_dataset and not materialized.is_file():
        raise SystemExit("V18 materialized dataset is required but absent")
    if materialized.is_file():
        if sha256(materialized) != dataset["manifest"]["sha256"]:
            raise SystemExit("V18 materialized dataset manifest hash differs")
        manifest = json.loads(materialized.read_text(encoding="utf-8"))
        if (
            manifest.get("outputs", {}).get("train", {}).get("sha256")
            != dataset["train_sha256"]
            or manifest.get("outputs", {}).get("valid", {}).get("sha256")
            != dataset["valid_sha256"]
        ):
            raise SystemExit("V18 materialized dataset outputs differ")

    initialization = contract["initialization"]
    authenticated_weights(
        initialization["source"],
        "initialization source",
        required=require_local_model_artifacts,
    )
    parity = authenticated_file(initialization["parity_report"], "parity report")
    if (
        parity.get("status") != "initialization-parity-passed"
        or parity.get("candidate", {}).get("weights_sha256")
        != initialization["expected_widened_weights_sha256"]
        or parity.get("candidate", {}).get("transformation_manifest_sha256")
        != initialization["widening_manifest_sha256"]
    ):
        raise SystemExit("V18 initialization parity evidence differs")

    for direction in ("en-ja", "ja-en"):
        teacher = contract["teachers"][direction]
        authenticated_weights(
            teacher,
            f"{direction} teacher",
            required=require_local_model_artifacts,
        )
        authenticated_file(
            teacher["lineage_manifest"],
            f"{direction} teacher lineage",
            required=require_local_model_artifacts,
        )

    smoke = authenticated_file(contract["training_path_smoke"], "training smoke")
    if (
        smoke.get("status") != "one-update-training-path-passed"
        or smoke.get("promotion_authorized") is not False
        or smoke.get("training_authorized") is not False
    ):
        raise SystemExit("V18 training-path smoke evidence differs")

    for name, record in contract["tools"].items():
        path = resolve(record["path"])
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise SystemExit(f"V18 {name} tool hash differs: {path}")

    phase = contract["phase1_training"]
    parameters = phase["hyperparameters"]
    if (
        int(phase["authorized_updates"]) != 1_000
        or phase["checkpoint_steps"] != [250, 500, 750, 1000]
        or int(parameters["batch_size"]) * int(parameters["gradient_accumulation"])
        != int(parameters["effective_batch_size"])
        or float(parameters["teacher_kl_weight"]) != 0.25
        or float(parameters["encoder_alignment_weight"]) != 0.05
        or phase.get("source_only_teacher_sequences_as_positive_targets") is not False
    ):
        raise SystemExit("V18 phase-one recipe differs")

    gates = contract["evaluation_gates"]
    if (
        gates["full_precision_internal"].get(
            "zero_new_critical_meaning_failures"
        )
        is not True
        or int(gates["q4"]["maximum_one_model_bundle_bytes"]) > 150_000_000
        or gates["promotion"].get("requires_zero_critical_error_union") is not True
        or gates["promotion"].get("requires_non_apple_failure_path") is not True
    ):
        raise SystemExit("V18 quality/promotion gates are weaker than frozen")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("--require-materialized-dataset", action="store_true")
    parser.add_argument("--require-local-model-artifacts", action="store_true")
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    validate(
        contract,
        require_materialized_dataset=args.require_materialized_dataset,
        require_local_model_artifacts=args.require_local_model_artifacts,
    )
    print("V18 shared-bidirectional phase-one contract passed")


if __name__ == "__main__":
    main()
