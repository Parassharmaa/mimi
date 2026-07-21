#!/usr/bin/env python3
"""Build a hash-bound release/provenance contract for Mimi's routed Marian MoE."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from collections import Counter
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any


KFTT_ATTRIBUTION = (
    "English contents translated by NICT from Japanese Wikipedia; CC-BY-SA-3.0; "
    "https://alaginrc.nict.go.jp/WikiCorpus/"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"could not read JSON object {path}: {error}") from error
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return payload


def resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def authenticate(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"missing {label}: {path}")
    actual = sha256(path)
    if actual != expected_sha256:
        raise SystemExit(
            f"{label} hash mismatch: expected {expected_sha256}, found {actual}: {path}"
        )


class ReleaseTrace:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.training_manifests: dict[str, dict[str, Any]] = {}
        self.lineage_manifests: dict[str, dict[str, Any]] = {}
        self.dataset_files: dict[str, dict[str, Any]] = {}
        self.dataset_manifests: dict[str, dict[str, Any]] = {}
        self.dataset_manifest_by_file: dict[str, set[str]] = {}
        self.upstream_models: dict[str, dict[str, str]] = {}
        self.visited_models: set[tuple[str, str]] = set()

    def visit_model(self, directory_value: str, expected_model_sha256: str) -> None:
        """Visit either a trained checkpoint or an authenticated transform."""

        directory = resolve_path(directory_value, self.workspace)
        training_manifest = directory / "mimi_training_manifest.json"
        if training_manifest.is_file():
            self.visit_training_model(str(directory), expected_model_sha256)
            return
        candidates = (
            directory / "mimi_checkpoint_interpolation_manifest.json",
            directory / "mimi_checkpoint_averaging_manifest.json",
        )
        for candidate in candidates:
            if candidate.is_file():
                visit_generalist_lineage(self, candidate, expected_model_sha256)
                return
        raise SystemExit(f"model lineage manifest is missing: {directory}")

    def record_dataset(
        self,
        declared_path: str,
        declared_sha256: str,
        declared_rows: int | None,
        *,
        source_manifest: Path,
        split: str,
    ) -> None:
        path = resolve_path(declared_path, self.workspace)
        authenticate(path, declared_sha256, f"{split} dataset")
        rows = sum(1 for line in path.open(encoding="utf-8") if line.strip())
        if declared_rows is not None and rows != declared_rows:
            raise SystemExit(
                f"{split} row count mismatch: expected {declared_rows}, found {rows}: {path}"
            )
        key = str(path)
        record = {
            "path": key,
            "sha256": declared_sha256,
            "rows": rows,
            "declaredBy": str(source_manifest),
            "split": split,
        }
        previous = self.dataset_files.get(key)
        if previous and (
            previous["sha256"] != record["sha256"]
            or previous["rows"] != record["rows"]
        ):
            raise SystemExit(f"dataset has conflicting declarations: {path}")
        self.dataset_files[key] = record

    def record_dataset_manifest(
        self,
        path_value: str | Path,
        expected_sha256: str | None = None,
    ) -> None:
        """Authenticate a dataset manifest against every output it declares.

        Dataset rows alone cannot authorize promotion. The dataset-level policy
        must explicitly opt in, and its output hashes must cover the exact files
        visited through the selected model lineage.
        """

        path = resolve_path(path_value, self.workspace)
        if expected_sha256 is not None:
            authenticate(path, expected_sha256, "dataset manifest")
        elif not path.is_file():
            return
        manifest = load_object(path)
        outputs = manifest.get("outputs")
        if not isinstance(outputs, dict):
            raise SystemExit(f"dataset manifest lacks outputs: {path}")
        authenticated_outputs: list[str] = []
        output_files: dict[str, dict[str, Any]] = {}
        for split, output in outputs.items():
            if not isinstance(output, dict) or not output.get("path") or not output.get(
                "sha256"
            ):
                raise SystemExit(f"dataset manifest has invalid {split} output: {path}")
            output_path = resolve_path(str(output["path"]), self.workspace)
            authenticate(output_path, str(output["sha256"]), f"{split} dataset output")
            output_key = str(output_path)
            output_files[str(split)] = {
                "path": output_key,
                "sha256": str(output["sha256"]),
            }
            authenticated_outputs.append(str(split))
            self.dataset_manifest_by_file.setdefault(output_key, set()).add(str(path))
        actual_sha256 = sha256(path)
        record = {
            "path": str(path),
            "sha256": actual_sha256,
            "promotionEligible": manifest.get("promotion_eligible") is True,
            "declaredPromotionEligible": manifest.get("promotion_eligible"),
            "authenticatedOutputs": sorted(authenticated_outputs),
            "outputs": dict(sorted(output_files.items())),
        }
        previous = self.dataset_manifests.get(str(path))
        if previous and previous != record:
            raise SystemExit(f"dataset manifest has conflicting declarations: {path}")
        self.dataset_manifests[str(path)] = record

    def visit_training_model(self, directory_value: str, expected_model_sha256: str) -> None:
        directory = resolve_path(directory_value, self.workspace)
        identity = (str(directory), expected_model_sha256)
        if identity in self.visited_models:
            return
        self.visited_models.add(identity)
        authenticate(
            directory / "model.safetensors",
            expected_model_sha256,
            "lineage model weights",
        )
        manifest_path = directory / "mimi_training_manifest.json"
        manifest = load_object(manifest_path)
        manifest_sha = sha256(manifest_path)
        declared_manifest_sha = manifest.get("manifest_sha256")
        if declared_manifest_sha and declared_manifest_sha != manifest_sha:
            raise SystemExit(f"training manifest self hash differs: {manifest_path}")
        self.training_manifests[str(manifest_path)] = {
            "sha256": manifest_sha,
            "modelPath": str(directory / "model.safetensors"),
            "modelSha256": expected_model_sha256,
        }
        dataset = manifest.get("dataset")
        if not isinstance(dataset, dict):
            raise SystemExit(f"training manifest lacks dataset: {manifest_path}")
        for split in ("train", "valid"):
            self.record_dataset(
                str(dataset[f"{split}_path"]),
                str(dataset[f"{split}_sha256"]),
                int(dataset[f"{split}_rows"]),
                source_manifest=manifest_path,
                split=split,
            )
        dataset_parents = {
            resolve_path(str(dataset[f"{split}_path"]), self.workspace).parent
            for split in ("train", "valid")
        }
        for parent in sorted(dataset_parents):
            self.record_dataset_manifest(parent / "manifest.json")
        repository = str(manifest.get("student_repository", ""))
        revision = str(manifest.get("student_revision", ""))
        license_name = str(manifest.get("license", ""))
        if not repository or not revision or not license_name:
            raise SystemExit(f"training manifest lacks upstream identity: {manifest_path}")
        upstream_key = f"{repository}@{revision}"
        self.upstream_models[upstream_key] = {
            "repository": repository,
            "revision": revision,
            "license": license_name,
            "url": f"https://huggingface.co/{repository}/tree/{revision}",
        }
        for field in ("initial_checkpoint", "preservation_checkpoint"):
            checkpoint = manifest.get(field)
            if isinstance(checkpoint, dict) and checkpoint.get("path"):
                checkpoint_manifest = resolve_path(
                    str(checkpoint["path"]), self.workspace
                ) / "mimi_training_manifest.json"
                declared_checkpoint_manifest_sha = checkpoint.get(
                    "training_manifest_sha256"
                )
                if declared_checkpoint_manifest_sha:
                    authenticate(
                        checkpoint_manifest,
                        str(declared_checkpoint_manifest_sha),
                        f"{field} training manifest",
                    )
                self.visit_model(
                    str(checkpoint["path"]), str(checkpoint["model_sha256"])
                )

    def visit_expert_dataset(self, training_data: dict[str, Any]) -> None:
        dataset_record = training_data.get("dataset_manifest")
        if not isinstance(dataset_record, dict):
            raise SystemExit("expert engine lacks dataset manifest provenance")
        path = resolve_path(str(dataset_record["path"]), self.workspace)
        authenticate(path, str(dataset_record["sha256"]), "expert dataset manifest")
        self.record_dataset_manifest(path, str(dataset_record["sha256"]))
        manifest = load_object(path)
        outputs = manifest.get("outputs")
        counts = manifest.get("counts", {})
        if not isinstance(outputs, dict):
            raise SystemExit(f"expert dataset manifest lacks outputs: {path}")
        for split in ("train", "valid"):
            record = outputs.get(split)
            if not isinstance(record, dict):
                raise SystemExit(f"expert dataset manifest lacks {split}: {path}")
            self.record_dataset(
                str(record["path"]),
                str(record["sha256"]),
                int(counts[split]) if split in counts else None,
                source_manifest=path,
                split=split,
            )


def validate_pack(pack: Path) -> tuple[dict[str, Any], int]:
    manifest_path = pack / "manifest.json"
    manifest = load_object(manifest_path)
    pack_format = manifest.get("format")
    if pack_format not in {
        "mimi-mlx-marian-moe-v1",
        "mimi-mlx-marian-moe-v2",
    }:
        raise SystemExit("unsupported routed Marian pack")
    if pack_format == "mimi-mlx-marian-moe-v2":
        shared = manifest.get("sharedTokenizer")
        shared_path = PurePosixPath(shared) if isinstance(shared, str) else None
        if (
            shared_path is None
            or shared_path.is_absolute()
            or any(part in ("", ".", "..") for part in shared_path.parts)
            or shared not in manifest.get("files", {})
        ):
            raise SystemExit("invalid shared-tokenizer Marian pack contract")
    for relative, record in manifest.get("files", {}).items():
        path = pack / relative
        if not isinstance(record, dict):
            raise SystemExit(f"invalid pack file record: {relative}")
        if (
            not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or sha256(path) != record.get("sha256")
        ):
            raise SystemExit(f"pack integrity failure: {relative}")
    size = sum(item.stat().st_size for item in pack.rglob("*") if item.is_file())
    return manifest, size


def validate_translation_memory(
    pack: Path,
    pack_manifest: dict[str, Any],
    audit_path: Path | None,
    training_data_path: Path | None,
) -> dict[str, Any] | None:
    metadata = pack_manifest.get("translationMemory")
    if metadata is None:
        if audit_path is not None or training_data_path is not None:
            raise SystemExit("translation-memory evidence supplied for a pack without memory")
        return None
    if not isinstance(metadata, dict):
        raise SystemExit("invalid translation-memory metadata")
    if audit_path is None or training_data_path is None:
        raise SystemExit(
            "a memory-bearing pack requires --translation-memory-audit and "
            "--translation-memory-training-data"
        )
    runtime_path = pack / str(metadata.get("path", ""))
    runtime = load_object(runtime_path)
    if (
        runtime.get("schemaVersion") != 1
        or runtime.get("doesNotAuthorizeAppIntegration") is not True
        or runtime.get("normalization") != metadata.get("normalization")
        or runtime.get("sourceLicense") != metadata.get("sourceLicense")
        or runtime.get("trainingDataSHA256") != metadata.get("trainingDataSHA256")
        or runtime.get("auditSHA256") != metadata.get("auditSHA256")
    ):
        raise SystemExit("translation-memory runtime and pack metadata differ")
    entries = runtime.get("entries")
    if not isinstance(entries, dict) or set(entries) != {"en-ja", "ja-en"}:
        raise SystemExit("translation-memory runtime has invalid directions")
    entry_count = sum(len(values) for values in entries.values())
    if entry_count != metadata.get("entries"):
        raise SystemExit("translation-memory entry count differs")

    audit_path = audit_path.resolve()
    training_data_path = training_data_path.resolve()
    authenticate(audit_path, str(metadata["auditSHA256"]), "translation-memory audit")
    authenticate(
        training_data_path,
        str(metadata["trainingDataSHA256"]),
        "translation-memory training data",
    )
    try:
        with gzip.open(audit_path, "rt", encoding="utf-8") as archive:
            audit = json.load(archive)
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"could not read translation-memory audit: {error}") from error
    if (
        not isinstance(audit, dict)
        or audit.get("schemaVersion") != 1
        or audit.get("doesNotAuthorizeAppIntegration") is not True
        or audit.get("sourceLicense") != metadata.get("sourceLicense")
        or audit.get("trainingData", {}).get("sha256")
        != metadata.get("trainingDataSHA256")
        or audit.get("counts", {}).get("entries") != entry_count
    ):
        raise SystemExit("translation-memory audit differs from the runtime contract")

    source_rows = 0
    training_only_rows = 0
    promotion_false_rows = 0
    for line_number, line in enumerate(training_data_path.open(encoding="utf-8"), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        source_rows += 1
        if (
            row.get("source_license") != metadata.get("sourceLicense")
            or not row.get("attribution")
            or not row.get("source_provenance")
            or not row.get("source_id")
        ):
            raise SystemExit(
                f"translation-memory source row lacks provenance: "
                f"{training_data_path}:{line_number}"
            )
        training_only_rows += row.get("training_only") is True
        promotion_false_rows += row.get("promotion_eligible") is False
    if source_rows != audit.get("counts", {}).get("trainingRows"):
        raise SystemExit("translation-memory source row count differs from audit")
    if training_only_rows == 0 or promotion_false_rows == 0:
        raise SystemExit("translation-memory source is not explicitly training-only")

    return {
        "runtime": {
            "path": str(runtime_path.resolve()),
            "bytes": runtime_path.stat().st_size,
            "sha256": sha256(runtime_path),
        },
        "auditSource": {
            "path": str(audit_path),
            "bytes": audit_path.stat().st_size,
            "sha256": sha256(audit_path),
        },
        "trainingData": {
            "path": str(training_data_path),
            "sha256": sha256(training_data_path),
            "rows": source_rows,
            "trainingOnlyRows": training_only_rows,
            "promotionEligibleFalseRows": promotion_false_rows,
        },
        "entries": entry_count,
        "directions": {name: len(values) for name, values in sorted(entries.items())},
        "normalization": metadata["normalization"],
        "sourceLicense": metadata["sourceLicense"],
        "policy": audit["policy"],
        "promotionEligible": False,
        "blocker": "runtime entries derive from rows explicitly marked training-only and promotion_eligible=false",
        "doesNotAuthorizeAppIntegration": True,
    }


def visit_generalist_lineage(
    trace: ReleaseTrace,
    manifest_path: Path,
    expected_output_sha256: str,
) -> None:
    if manifest_path.is_dir():
        trace.visit_training_model(str(manifest_path), expected_output_sha256)
        return
    if manifest_path.name == "mimi_training_manifest.json":
        trace.visit_training_model(str(manifest_path.parent), expected_output_sha256)
        return
    manifest = load_object(manifest_path)
    output = manifest.get("output", {})
    if output.get("model_sha256") != expected_output_sha256:
        raise SystemExit(f"generalist lineage output differs: {manifest_path}")
    trace.lineage_manifests[str(manifest_path)] = {"sha256": sha256(manifest_path)}
    operation = manifest.get("operation")
    if operation == "linear-checkpoint-interpolation":
        for field in ("parent", "adapted"):
            model = manifest.get(field)
            if not isinstance(model, dict):
                raise SystemExit(f"interpolation lacks {field}: {manifest_path}")
            trace.visit_model(str(model["path"]), str(model["model_sha256"]))
        return
    if operation == "arithmetic-mean-of-best-adjacent-full-precision-checkpoints":
        checkpoints = manifest.get("selected_checkpoints")
        if not isinstance(checkpoints, list) or not checkpoints:
            raise SystemExit(f"averaging manifest lacks checkpoints: {manifest_path}")
        for checkpoint in checkpoints:
            trace.visit_model(
                str(checkpoint["path"]), str(checkpoint["model_sha256"])
            )
        return
    raise SystemExit(f"unsupported generalist lineage operation: {operation}")


def direct_engine_lineage_record(
    trace: ReleaseTrace,
    source: Path,
    expected_model_sha256: str,
    expected_training_manifest_sha256: str,
) -> dict[str, Any]:
    """Authenticate and record a selected direct expert checkpoint."""

    source = source.resolve()
    if not source.is_dir():
        raise SystemExit(f"expert lineage must be a checkpoint directory: {source}")
    model_path = source / "model.safetensors"
    manifest_path = source / "mimi_training_manifest.json"
    authenticate(model_path, expected_model_sha256, "expert lineage model")
    authenticate(
        manifest_path,
        expected_training_manifest_sha256,
        "expert lineage training manifest",
    )
    trace.visit_training_model(str(source), expected_model_sha256)
    return {
        "kind": "direct-training-checkpoint",
        "path": str(manifest_path),
        "sha256": sha256(manifest_path),
        "modelPath": str(model_path),
        "modelSha256": expected_model_sha256,
    }


def conversion_provenance_status(
    pack: Path,
    pack_manifest: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Validate explicit full-precision-to-MLX records; absence is a blocker."""

    authenticated: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for engine_name in (
        "generalist-en-ja",
        "generalist-ja-en",
        "formal-en-ja",
        "legal-ja-en",
    ):
        engine_record = pack_manifest.get("engines", {}).get(engine_name)
        manifest_path = pack / "engines" / engine_name / "manifest.json"
        engine_manifest = load_object(manifest_path)
        conversion = engine_manifest.get("conversion")
        model_record = engine_manifest.get("files", {}).get("model.safetensors")
        if not isinstance(engine_record, dict) or not isinstance(model_record, dict):
            raise SystemExit(f"engine conversion identity is incomplete: {engine_name}")
        if not isinstance(conversion, dict):
            missing.append(engine_name)
            continue
        tool = conversion.get("tool")
        if (
            conversion.get("sourceWeightsSha256")
            != engine_record.get("sourceWeightsSha256")
            or conversion.get("outputWeightsSha256") != model_record.get("sha256")
            or not isinstance(tool, dict)
            or not tool.get("path")
            or not tool.get("sha256")
        ):
            raise SystemExit(f"invalid conversion provenance: {engine_name}")
        tool_path = resolve_path(str(tool["path"]), Path.cwd().resolve())
        authenticate(tool_path, str(tool["sha256"]), "MLX conversion tool")
        authenticated[engine_name] = conversion
    return authenticated, missing


def extract_attributions(trace: ReleaseTrace) -> tuple[dict[str, Any], bytes]:
    license_counts: Counter[str] = Counter()
    origin_counts: Counter[str] = Counter()
    promotion_exclusion_reasons: Counter[str] = Counter()
    promotion_excluded_origins: Counter[str] = Counter()
    promotion_excluded_rows = 0
    tatoeba: dict[tuple[str, str], dict[str, str]] = {}
    for dataset in trace.dataset_files.values():
        path = Path(dataset["path"])
        for line_number, line in enumerate(path.open(encoding="utf-8"), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            license_name = str(row.get("source_license") or metadata.get("license") or "")
            attribution = str(row.get("attribution") or metadata.get("attribution") or "")
            source_id = str(row.get("source_id") or metadata.get("source_id") or "")
            provenance = str(
                row.get("source_provenance") or metadata.get("source") or ""
            )
            origin = str(row.get("origin") or metadata.get("source") or "unknown")
            if not license_name:
                raise SystemExit(f"row lacks license: {path}:{line_number}")
            if not attribution and license_name == "project-owned" and provenance:
                attribution = f"Mimi project-owned source: {provenance}"
            if (
                not attribution
                and license_name == "CC-BY-SA-3.0"
                and "Kyoto Free Translation Task" in provenance
            ):
                attribution = KFTT_ATTRIBUTION
            if not attribution:
                raise SystemExit(f"row lacks attribution: {path}:{line_number}")
            license_counts[license_name] += 1
            origin_counts[origin] += 1
            excluded_reasons: list[str] = []
            if row.get("promotion_eligible") is False:
                excluded_reasons.append("promotion_eligible=false")
            if row.get("training_only") is True:
                excluded_reasons.append("training_only=true")
            if excluded_reasons:
                promotion_excluded_rows += 1
                promotion_excluded_origins[origin] += 1
                promotion_exclusion_reasons.update(excluded_reasons)
            if license_name == "CC-BY-2.0-FR":
                if not source_id or "tatoeba.org #" not in attribution:
                    raise SystemExit(
                        f"Tatoeba row lacks contributor identity: {path}:{line_number}"
                    )
                tatoeba[(source_id, attribution)] = {
                    "sourceID": source_id,
                    "attribution": attribution,
                    "sourceProvenance": provenance,
                    "license": license_name,
                    "sourceURL": f"https://tatoeba.org/en/sentences/show/{source_id}",
                }
    lines = [
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in sorted(
            tatoeba.values(),
            key=lambda row: (row["sourceID"], row["attribution"]),
        )
    ]
    payload = (("\n".join(lines) + "\n") if lines else "").encode()
    return {
        "datasetRows": sum(record["rows"] for record in trace.dataset_files.values()),
        "licenses": dict(sorted(license_counts.items())),
        "origins": dict(sorted(origin_counts.items())),
        "promotionExcludedRows": promotion_excluded_rows,
        "promotionExclusionReasons": dict(sorted(promotion_exclusion_reasons.items())),
        "promotionExcludedOrigins": dict(sorted(promotion_excluded_origins.items())),
        "uniqueTatoebaAttributions": len(lines),
    }, payload


def dataset_policy_status(trace: ReleaseTrace) -> dict[str, Any]:
    files_without_manifest = sorted(
        path
        for path in trace.dataset_files
        if path not in trace.dataset_manifest_by_file
    )
    promotion_ineligible_manifests = sorted(
        path
        for path, record in trace.dataset_manifests.items()
        if record["promotionEligible"] is not True
    )
    return {
        "promotionEligible": bool(trace.dataset_files)
        and not files_without_manifest
        and not promotion_ineligible_manifests,
        "filesWithoutManifest": files_without_manifest,
        "promotionIneligibleManifests": promotion_ineligible_manifests,
    }


def attribution_markdown(
    pack_manifest_sha256: str,
    jlt_access_date: str,
    tatoeba_sha256: str,
    tatoeba_count: int,
    includes_translation_memory: bool,
) -> str:
    translation_memory_notice = """

The development pack also contains a deterministic exact-source translation
memory derived only from repeated, human Japanese Law Translation pairs. Its
hash-bound audit records cross-document evidence, source selection, critical-
token checks, and all rejected candidates. These verbatim runtime entries remain
blocked from app integration because their source rows are explicitly marked
training-only and promotion-ineligible; the memory is evaluation evidence, not
an authorization to distribute or enable it.
""" if includes_translation_memory else ""
    return f"""# Mimi local translation model attributions

This notice applies to the routed Marian MLX model pack whose manifest SHA-256 is
`{pack_manifest_sha256}`. Mimi fine-tuned, interpolated, averaged, quantized, and
packaged the identified upstream weights; no upstream author or public agency
endorses Mimi or these adapted translations.

## ElanMT model weights

The four engines derive from ElanMT by the ELAN MITSUA Project / Abstract Engine,
licensed CC BY-SA 4.0. Upstream revisions and transformation hashes are recorded
in `release-contract.json`. The proposed license for the adapted model weights is
CC BY-SA 4.0; distribution remains blocked pending final compatibility and app-
distribution review.

## Kyoto Free Translation Task

The data used in this service contains English contents which is translated by
the National Institute of Information and Communications Technology (NICT) from
Japanese sentences on Wikipedia. Our use of this data is licensed by the Creative
Commons Attribution-Share-Alike License 3.0. Please refer to
http://creativecommons.org/licenses/by-sa/3.0/ or
http://alaginrc.nict.go.jp/WikiCorpus/ for details.

## NICT Asian Language Treebank

NICT Asian Language Treebank Parallel Corpus; NICT translations CC BY 4.0;
English Wikinews source text CC BY 2.5. Cite Riza et al. (2016), “Introduction of
the Asian Language Treebank.” Source: https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/

## Tatoeba via ManyThings

Retained sentence IDs, contributor names, source links, and CC BY 2.0 France
notices for {tatoeba_count:,} unique attributions are in
`tatoeba-attributions.jsonl.gz` (uncompressed content SHA-256
`{tatoeba_sha256}`). Source: https://www.manythings.org/anki/

## Japanese Law Translation Database System

Created by Mimi based on Japanese Law Translation Database System content
published by the Ministry of Justice, Japan, accessed {jlt_access_date}. PDL 1.0:
https://www.digital.go.jp/en/resources/open_data/public_data_license_v1.0
Source: https://www.japaneselawtranslation.go.jp/en/laws

Mimi filtered, normalized, selected, and converted the source content into
parallel training rows. The English translations are not official texts; only
the original Japanese laws and regulations have legal effect. The translations
are reference material, may include tentative versions, and carry the accuracy,
reliability, currency, and interpretation disclaimers in the database terms:
https://www.japaneselawtranslation.go.jp/en/index/terms
{translation_memory_notice}

## Mimi project-owned parallel copy

Small English/Japanese UI pairs authored and shipped by Mimi are project-owned.
The authenticated source revision and source rows are recorded in
`release-contract.json`.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pack", type=Path)
    parser.add_argument("en_ja_generalist_lineage", type=Path)
    parser.add_argument("ja_en_generalist_lineage", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--jlt-access-date", type=date.fromisoformat, required=True)
    parser.add_argument("--translation-memory-audit", type=Path)
    parser.add_argument("--translation-memory-training-data", type=Path)
    parser.add_argument("--formal-en-ja-lineage", type=Path)
    parser.add_argument("--legal-ja-en-lineage", type=Path)
    args = parser.parse_args()
    expert_lineage_inputs = {
        "formal-en-ja": args.formal_en_ja_lineage,
        "legal-ja-en": args.legal_ja_en_lineage,
    }
    if any(expert_lineage_inputs.values()) and not all(expert_lineage_inputs.values()):
        raise SystemExit("both expert lineage inputs must be supplied together")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    workspace = Path.cwd().resolve()
    pack = args.pack.resolve()
    pack_manifest, pack_bytes = validate_pack(pack)
    translation_memory = validate_translation_memory(
        pack,
        pack_manifest,
        args.translation_memory_audit,
        args.translation_memory_training_data,
    )
    trace = ReleaseTrace(workspace)
    visit_generalist_lineage(
        trace,
        args.en_ja_generalist_lineage.resolve(),
        str(pack_manifest["engines"]["generalist-en-ja"]["sourceWeightsSha256"]),
    )
    visit_generalist_lineage(
        trace,
        args.ja_en_generalist_lineage.resolve(),
        str(pack_manifest["engines"]["generalist-ja-en"]["sourceWeightsSha256"]),
    )
    engine_lineages: dict[str, dict[str, Any]] = {
        name: dict(record["releaseLineage"])
        for name, record in pack_manifest["engines"].items()
        if isinstance(record.get("releaseLineage"), dict)
    }
    missing_engine_lineages: list[str] = []
    for engine_name in ("formal-en-ja", "legal-ja-en"):
        engine_record = pack_manifest["engines"][engine_name]
        training_data = engine_record.get("trainingData")
        if not isinstance(training_data, dict):
            raise SystemExit(f"expert engine lacks training data: {engine_name}")
        lineage_source = expert_lineage_inputs[engine_name]
        if lineage_source is None:
            missing_engine_lineages.append(engine_name)
        else:
            declared_manifest_sha256 = training_data.get("training_manifest_sha256")
            if not isinstance(declared_manifest_sha256, str):
                raise SystemExit(
                    f"expert engine lacks training-manifest hash: {engine_name}"
                )
            engine_lineages[engine_name] = direct_engine_lineage_record(
                trace,
                lineage_source,
                str(engine_record["sourceWeightsSha256"]),
                declared_manifest_sha256,
            )
        trace.visit_expert_dataset(training_data)

    conversion_provenance, missing_conversion_provenance = (
        conversion_provenance_status(pack, pack_manifest)
    )

    attribution_summary, tatoeba_payload = extract_attributions(trace)
    dataset_policy = dataset_policy_status(trace)
    promotion_eligible = (
        attribution_summary["promotionExcludedRows"] == 0
        and translation_memory is None
        and dataset_policy["promotionEligible"]
    )
    if translation_memory is not None:
        distribution_status = "blocked-training-only-runtime-memory-and-final-review"
    elif promotion_eligible:
        distribution_status = "blocked-pending-final-license-and-app-distribution-review"
    else:
        distribution_status = "blocked-promotion-ineligible-training-data"
    blockers = []
    if translation_memory is not None:
        blockers.append("translation-memory-is-training-only-and-promotion-ineligible")
    if attribution_summary["promotionExcludedRows"]:
        blockers.append("training-lineage-contains-promotion-ineligible-rows")
    if dataset_policy["filesWithoutManifest"]:
        blockers.append("training-lineage-has-dataset-files-without-policy-manifest")
    if dataset_policy["promotionIneligibleManifests"]:
        blockers.append("dataset-policy-manifest-does-not-authorize-promotion")
    if missing_engine_lineages:
        blockers.append(
            "missing-selected-engine-lineage:" + ",".join(sorted(missing_engine_lineages))
        )
    if missing_conversion_provenance:
        blockers.append(
            "missing-full-precision-to-mlx-conversion-provenance:"
            + ",".join(sorted(missing_conversion_provenance))
        )
    blockers.extend(
        [
            "sealed-automated-400x2-quality-evaluation-pending",
            "license-compatibility-and-distribution-review-pending",
            "portable-release-inventory-pending",
            "app-distribution-terms-review-pending",
        ]
    )
    tatoeba_content_sha = hashlib.sha256(tatoeba_payload).hexdigest()
    tatoeba_path = args.output / "tatoeba-attributions.jsonl.gz"
    with tatoeba_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            compressed.write(tatoeba_payload)
    notices_path = args.output / "ATTRIBUTIONS.md"
    notices_path.write_text(
        attribution_markdown(
            sha256(pack / "manifest.json"),
            args.jlt_access_date.isoformat(),
            tatoeba_content_sha,
            attribution_summary["uniqueTatoebaAttributions"],
            translation_memory is not None,
        ),
        encoding="utf-8",
    )
    memory_audit_release_path = None
    if translation_memory is not None:
        memory_audit_release_path = args.output / "exact-translation-memory-audit.json.gz"
        shutil.copyfile(args.translation_memory_audit, memory_audit_release_path)
    contract = {
        "schemaVersion": 1,
        "purpose": "hash-bound release provenance and attribution contract",
        "pack": {
            "path": str(pack),
            "manifestSha256": sha256(pack / "manifest.json"),
            "bytes": pack_bytes,
        },
        "lineageManifests": dict(sorted(trace.lineage_manifests.items())),
        "engineLineages": dict(sorted(engine_lineages.items())),
        "conversionProvenance": dict(sorted(conversion_provenance.items())),
        "trainingManifests": dict(sorted(trace.training_manifests.items())),
        "datasetManifests": dict(sorted(trace.dataset_manifests.items())),
        "datasetFiles": dict(sorted(trace.dataset_files.items())),
        "upstreamModels": dict(sorted(trace.upstream_models.items())),
        "attributionSummary": attribution_summary,
        "releaseFiles": {
            "ATTRIBUTIONS.md": {
                "bytes": notices_path.stat().st_size,
                "sha256": sha256(notices_path),
            },
            "tatoeba-attributions.jsonl.gz": {
                "bytes": tatoeba_path.stat().st_size,
                "sha256": sha256(tatoeba_path),
                "uncompressedBytes": len(tatoeba_payload),
                "uncompressedSha256": tatoeba_content_sha,
            },
        },
        "jltAccessDate": args.jlt_access_date.isoformat(),
        "provenanceComplete": not missing_engine_lineages
        and not missing_conversion_provenance,
        "modelPromotionEligible": promotion_eligible,
        "datasetPolicy": dataset_policy,
        "qualityAuthorization": "blocked-pending-sealed-automated-400x2-evaluation",
        "distributionStatus": distribution_status,
        "releaseAuthorization": "blocked",
        "blockers": blockers,
        "doesNotAuthorizeDistribution": True,
        "doesNotAuthorizeAppIntegration": True,
    }
    if translation_memory is not None and memory_audit_release_path is not None:
        translation_memory["auditReleaseFile"] = {
            "path": memory_audit_release_path.name,
            "bytes": memory_audit_release_path.stat().st_size,
            "sha256": sha256(memory_audit_release_path),
        }
        contract["translationMemory"] = translation_memory
        contract["releaseFiles"][memory_audit_release_path.name] = {
            "bytes": memory_audit_release_path.stat().st_size,
            "sha256": sha256(memory_audit_release_path),
        }
    contract_path = args.output / "release-contract.json"
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "releaseContractSha256": sha256(contract_path),
                "datasetFiles": len(trace.dataset_files),
                "datasetManifests": len(trace.dataset_manifests),
                "trainingManifests": len(trace.training_manifests),
                "uniqueTatoebaAttributions": attribution_summary[
                    "uniqueTatoebaAttributions"
                ],
                "releaseBytes": sum(
                    item.stat().st_size
                    for item in args.output.rglob("*")
                    if item.is_file()
                ),
                "distributionStatus": contract["distributionStatus"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
