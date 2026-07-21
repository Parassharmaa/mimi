#!/usr/bin/env python3
"""Authenticate training datasets and derive truthful target provenance."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


REFERENCE_TEACHER_EXPERIMENT = "strict local Qwen reference-filtered Marian ablation"
QWEN_ORIGIN = "strict-local-qwen-reference-distillation"
HUMAN_REFERENCE_ORIGIN = "matched-licensed-human-reference-control"

QWEN_TRAINING_DESCRIPTION = (
    "sequence-level distillation from provisional Qwen final translations admitted by "
    "hidden-reference metric filters; training-only; no reasoning traces"
)
QWEN_SEQUENCE_TARGET = "provisional hidden-reference-filtered Qwen final translation"
HUMAN_REFERENCE_TRAINING_DESCRIPTION = (
    "supervised control adaptation on matched licensed human-reference translations; "
    "no synthetic targets; no reasoning traces"
)
HUMAN_REFERENCE_SEQUENCE_TARGET = "matched licensed human-reference translation"
LICENSED_PARALLEL_TRAINING_DESCRIPTION = (
    "supervised adaptation on licensed human-authored and project-owned parallel "
    "references; no synthetic targets; no reasoning traces"
)
LICENSED_PARALLEL_SEQUENCE_TARGET = (
    "licensed human-authored or project-owned parallel reference translation"
)
LICENSED_HUMAN_REFERENCE_SOURCE = "licensed-human-reference"
LICENSED_PARALLEL_ORIGINS = frozenset(
    {
        "finalized-japanese-law-translation",
        "human-alt-parallel",
        "human-kftt-replay",
        "human-tatoeba-bidirectional-agreement-filtered",
        "mimi-shipped-ui-pair",
    }
)
LEGACY_SEQUENCE_TARGET = "reviewed canonical translation"
MARIAN_SEQUENCE_TARGET_SOURCE = "marian-source-only-sequence-distillation"
MARIAN_SEQUENCE_TARGET = (
    "source-only greedy final translation from the authenticated frozen Marian teacher"
)
MARIAN_SEQUENCE_TRAINING_DESCRIPTION = (
    "canonical sequence distillation from source-only final translations emitted by "
    "the authenticated frozen Marian teacher; licensed references were not exposed; "
    "no reasoning traces"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"could not read dataset manifest {path}: {error}") from error
    if not isinstance(manifest, dict):
        raise SystemExit(f"dataset manifest must be a JSON object: {path}")
    return manifest


def authenticate_structural_pruning_manifest(
    checkpoint_directory: Path,
) -> dict[str, Any] | None:
    manifest_path = checkpoint_directory / "mimi_structural_pruning_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = _load_manifest(manifest_path)
    model_path = checkpoint_directory / "model.safetensors"
    model_record = manifest.get("files", {}).get("model.safetensors", {})
    if not model_path.is_file() or model_record.get("sha256") != sha256(model_path):
        raise SystemExit("structural pruning manifest does not authenticate model weights")
    source_layers = manifest.get("source_decoder_layers")
    kept_layers = manifest.get("kept_decoder_layers")
    decoder_layers = manifest.get("decoder_layers")
    if (
        not isinstance(source_layers, list)
        or not isinstance(kept_layers, list)
        or decoder_layers != len(kept_layers)
        or len(kept_layers) >= len(source_layers)
    ):
        raise SystemExit("structural pruning manifest has an invalid decoder depth")
    if manifest.get("promotion_eligible") is not False:
        raise SystemExit("structural pruning checkpoint must be promotion-ineligible")
    if manifest.get("private_reasoning_traces_used") is not False:
        raise SystemExit("structural pruning checkpoint must exclude reasoning traces")
    return {
        "path": str(manifest_path.resolve()),
        "sha256": sha256(manifest_path),
        "method": manifest.get("method"),
        "encoder_layers": manifest.get("encoder_layers"),
        "source_decoder_layers": source_layers,
        "kept_decoder_layers": kept_layers,
        "decoder_layers": decoder_layers,
        "promotion_eligible": False,
        "private_reasoning_traces_used": False,
        "model_sha256": sha256(model_path),
    }


def authenticate_dataset_manifest(
    dataset_directory: Path,
    *,
    direction: str,
    train_path: Path,
    valid_path: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Load a dataset manifest and bind its declared outputs to the actual files.

    Older datasets without a manifest remain supported. Older manifest schemas that
    do not declare both output hashes are recorded but are not represented as fully
    output-authenticated. The reference-teacher experiment is held to the stricter
    current contract and must authenticate both splits.
    """

    manifest_path = dataset_directory / "manifest.json"
    if not manifest_path.is_file():
        return None, None

    manifest = _load_manifest(manifest_path)
    manifest_direction = manifest.get("direction")
    if manifest_direction not in (None, "", direction):
        raise SystemExit(
            f"dataset manifest direction differs: expected {direction}, "
            f"found {manifest_direction}"
        )

    outputs = manifest.get("outputs")
    authenticated_outputs: list[str] = []
    for split, actual_path in (("train", train_path), ("valid", valid_path)):
        record = outputs.get(split) if isinstance(outputs, dict) else None
        if not isinstance(record, dict) or not record.get("sha256"):
            continue
        actual_sha256 = sha256(actual_path)
        if record["sha256"] != actual_sha256:
            raise SystemExit(
                f"dataset manifest {split} hash differs: expected {record['sha256']}, "
                f"found {actual_sha256}"
            )
        authenticated_outputs.append(split)

    outputs_authenticated = authenticated_outputs == ["train", "valid"]
    if (
        manifest.get("experiment") == REFERENCE_TEACHER_EXPERIMENT
        and not outputs_authenticated
    ):
        raise SystemExit(
            "reference-teacher dataset manifest must authenticate train and valid outputs"
        )

    metadata = {
        "path": str(manifest_path.resolve()),
        "sha256": sha256(manifest_path),
        "schema_version": manifest.get("schema_version"),
        "direction": manifest_direction,
        "experiment": manifest.get("experiment"),
        "target_source": manifest.get("target_source"),
        "effective_licenses": manifest.get("effective_licenses"),
        "promotion_eligible": manifest.get("promotion_eligible"),
        "authenticated_outputs": authenticated_outputs,
        "outputs_authenticated": outputs_authenticated,
    }
    return manifest, metadata


def derive_target_provenance(
    dataset_manifest: dict[str, Any] | None,
    train_rows: list[dict[str, Any]],
    *,
    fallback_training_description: str,
) -> dict[str, str]:
    """Derive target wording from the authenticated dataset declaration.

    The explicit reviewed-target fallback preserves historical behavior for
    datasets outside the reference-teacher ablation contract.
    """

    if dataset_manifest is None:
        return {
            "training_description": fallback_training_description,
            "sequence_target": LEGACY_SEQUENCE_TARGET,
        }

    if dataset_manifest.get("target_source") == MARIAN_SEQUENCE_TARGET_SOURCE:
        if dataset_manifest.get("promotion_eligible") is not False:
            raise SystemExit("Marian sequence-distillation datasets must be promotion-ineligible")
        if dataset_manifest.get("references_exposed_to_teacher") is not False:
            raise SystemExit("Marian sequence teacher must not receive reference translations")
        if dataset_manifest.get("private_reasoning_traces_used") is not False:
            raise SystemExit("Marian sequence targets must exclude reasoning traces")
        teacher = dataset_manifest.get("teacher")
        if not isinstance(teacher, dict) or not all(
            teacher.get(field)
            for field in ("weights_sha256", "repository", "revision", "license")
        ):
            raise SystemExit("Marian sequence dataset lacks authenticated teacher identity")
        effective_licenses = dataset_manifest.get("effective_licenses")
        if not isinstance(effective_licenses, dict) or not isinstance(
            effective_licenses.get("train"), dict
        ):
            raise SystemExit("Marian sequence dataset lacks effective train licenses")
        expected_revision = f"{teacher['repository']}@{teacher['revision']}"
        if any(
            row.get("target_source") != MARIAN_SEQUENCE_TARGET_SOURCE
            or row.get("teacher_model_revision") != expected_revision
            or len(str(row.get("reference_target_sha256", ""))) != 64
            for row in train_rows
        ):
            raise SystemExit("Marian sequence rows contradict declared target provenance")
        return {
            "training_description": MARIAN_SEQUENCE_TRAINING_DESCRIPTION,
            "sequence_target": MARIAN_SEQUENCE_TARGET,
        }

    if dataset_manifest.get("experiment") != REFERENCE_TEACHER_EXPERIMENT:
        if dataset_manifest.get("target_source") == LICENSED_HUMAN_REFERENCE_SOURCE:
            if dataset_manifest.get("promotion_eligible") is not False:
                raise SystemExit(
                    "licensed human-reference datasets must remain promotion-ineligible"
                )
            effective_licenses = dataset_manifest.get("effective_licenses")
            train_licenses = (
                effective_licenses.get("train")
                if isinstance(effective_licenses, dict)
                else None
            )
            if not isinstance(train_licenses, dict) or not train_licenses:
                raise SystemExit(
                    "licensed human-reference dataset is missing effective train licenses"
                )
            actual_origins = Counter(row.get("origin") for row in train_rows)
            if not actual_origins:
                raise SystemExit("licensed human-reference dataset contains no train rows")
            unexpected = set(actual_origins) - LICENSED_PARALLEL_ORIGINS
            if unexpected:
                raise SystemExit(
                    "licensed human-reference dataset contains unapproved origins: "
                    f"{sorted(str(value) for value in unexpected)}"
                )
            if any(
                not row.get("source_license")
                or not row.get("source_provenance")
                or not row.get("attribution")
                for row in train_rows
            ):
                raise SystemExit(
                    "licensed human-reference rows must retain license, provenance, and "
                    "attribution"
                )
            licensed_row_count = sum(
                count
                for count in train_licenses.values()
                if isinstance(count, int) and count >= 0
            )
            if licensed_row_count != len(train_rows):
                raise SystemExit(
                    "licensed human-reference train license count differs: declared "
                    f"{licensed_row_count}, found {len(train_rows)}"
                )
            return {
                "training_description": LICENSED_PARALLEL_TRAINING_DESCRIPTION,
                "sequence_target": LICENSED_PARALLEL_SEQUENCE_TARGET,
            }

        declared_origins = dataset_manifest.get("origins", {}).get("train", {})
        counts = dataset_manifest.get("counts", {})
        synthetic_count = counts.get("synthetic_train")
        licensed_origin_names = (
            set(declared_origins)
            if isinstance(declared_origins, dict)
            and declared_origins
            and all(
                name.startswith("human-") or name.startswith("mimi-")
                for name in declared_origins
            )
            else set()
        )
        if licensed_origin_names and synthetic_count == 0:
            actual_origins = Counter(row.get("origin") for row in train_rows)
            for origin, declared_count in declared_origins.items():
                if actual_origins[origin] != declared_count:
                    raise SystemExit(
                        f"dataset {origin} row count differs: declared "
                        f"{declared_count}, found {actual_origins[origin]}"
                    )
            unexpected = set(actual_origins) - licensed_origin_names
            if unexpected:
                raise SystemExit(
                    "licensed parallel dataset contains undeclared origins: "
                    f"{sorted(str(value) for value in unexpected)}"
                )
            return {
                "training_description": LICENSED_PARALLEL_TRAINING_DESCRIPTION,
                "sequence_target": LICENSED_PARALLEL_SEQUENCE_TARGET,
            }
        return {
            "training_description": fallback_training_description,
            "sequence_target": LEGACY_SEQUENCE_TARGET,
        }

    if dataset_manifest.get("promotion_eligible") is not False:
        raise SystemExit("reference-teacher datasets must remain promotion-ineligible")

    effective_licenses = dataset_manifest.get("effective_licenses")
    if not isinstance(effective_licenses, dict) or not isinstance(
        effective_licenses.get("train"), dict
    ):
        raise SystemExit("reference-teacher dataset is missing effective train licenses")

    target_source = dataset_manifest.get("target_source")
    if target_source == "qwen":
        origin = QWEN_ORIGIN
        contradictory_origin = HUMAN_REFERENCE_ORIGIN
        training_description = QWEN_TRAINING_DESCRIPTION
        sequence_target = QWEN_SEQUENCE_TARGET
    elif target_source == "human-reference":
        origin = HUMAN_REFERENCE_ORIGIN
        contradictory_origin = QWEN_ORIGIN
        training_description = HUMAN_REFERENCE_TRAINING_DESCRIPTION
        sequence_target = HUMAN_REFERENCE_SEQUENCE_TARGET
    else:
        raise SystemExit(
            f"unknown reference-teacher target_source: {target_source!r}"
        )

    actual_origins = Counter(row.get("origin") for row in train_rows)
    declared_origins = dataset_manifest.get("origins", {}).get("train", {})
    declared_count = declared_origins.get(origin) if isinstance(declared_origins, dict) else None
    if not isinstance(declared_count, int) or declared_count < 1:
        raise SystemExit(f"dataset manifest does not declare any {origin} rows")
    if actual_origins[origin] != declared_count:
        raise SystemExit(
            f"dataset {origin} row count differs: declared {declared_count}, "
            f"found {actual_origins[origin]}"
        )
    if actual_origins[contradictory_origin]:
        raise SystemExit(
            f"dataset target_source {target_source} contains contradictory "
            f"{contradictory_origin} rows"
        )

    selected_rows = [row for row in train_rows if row.get("origin") == origin]
    if any(row.get("training_only") is not True for row in selected_rows):
        raise SystemExit("reference-teacher rows must be marked training_only")
    if any(row.get("promotion_eligible") is not False for row in selected_rows):
        raise SystemExit("reference-teacher rows must remain promotion-ineligible")
    if any(
        row.get("quality_control", {}).get("reasoning_trace_requested_or_stored")
        is not False
        for row in selected_rows
    ):
        raise SystemExit("reference-teacher rows must explicitly exclude reasoning traces")
    if target_source == "qwen":
        if any(
            row.get("review_status") != "hidden-reference-metric-filtered-provisional"
            or row.get("quality_control", {}).get("reference_exposed_to_teacher")
            is not False
            for row in selected_rows
        ):
            raise SystemExit(
                "Qwen target rows lack the declared hidden-reference filtering controls"
            )
        if any(
            row.get("target") != row.get("qwen_candidate") for row in selected_rows
        ):
            raise SystemExit(
                "Qwen target rows do not contain the declared Qwen final translation"
            )
    if target_source == "human-reference" and any(
        not row.get("source_license") or not row.get("reference_provenance")
        for row in selected_rows
    ):
        raise SystemExit("human-reference target rows lack licensed reference provenance")

    return {
        "training_description": training_description,
        "sequence_target": sequence_target,
    }
