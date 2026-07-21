#!/usr/bin/env python3
"""Freeze Mimi's exact controlled exposure and bounded upstream provenance.

The output deliberately does not claim that opaque upstream pretraining rows are
available. It binds every project-controlled text-bearing input to hashes and
uses pinned upstream model revisions, created before the private source suite,
as the only upstream temporal exclusion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Iterator

from validate_benchmark_suite import iter_training_text


ROOT = Path(__file__).resolve().parents[2]
TEXT_SCALAR_KEYS = {
    "source",
    "target",
    "translation",
    "reference",
    "reference_translation",
    "student_hypothesis",
    "hypothesis",
    "inputText",
    "outputText",
    "candidateTranslation",
    "acceptedTranslation",
}
TEXT_LIST_KEYS = {"references", "targets", "translations", "candidates"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def load(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing input: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"missing input: {path}")
    values = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(value, dict) for value in values):
        raise SystemExit(f"expected JSON objects: {path}")
    return values


def write_new(path: Path, content: str) -> None:
    if path.exists() and path.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def relative(path: Path, base: Path) -> str:
    return os.path.relpath(path.resolve(), base.resolve())


def verified_path(path_value: object, digest: object, label: str) -> Path:
    path = Path(str(path_value or ""))
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    expected = str(digest or "").lower()
    if not path.is_file() or len(expected) != 64 or sha256(path) != expected:
        raise SystemExit(f"missing or changed release-lineage input: {label} {path}")
    return path


def parse_timestamp(value: object, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise SystemExit(f"invalid timestamp for {label}") from error
    if parsed.tzinfo is None:
        raise SystemExit(f"timestamp must include timezone for {label}")
    return parsed


def result_text(value: object, path: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            if key in TEXT_SCALAR_KEYS and isinstance(child, str) and child.strip():
                yield child_path, child.strip()
            elif key == "content" and path.endswith("messages") and isinstance(child, str) and child.strip():
                yield child_path, child.strip()
            elif key in TEXT_LIST_KEYS and isinstance(child, list):
                for index, item in enumerate(child):
                    item_path = f"{child_path}[{index}]"
                    if isinstance(item, str) and item.strip():
                        yield item_path, item.strip()
                    else:
                        yield from result_text(item, item_path)
            elif isinstance(child, (dict, list)):
                yield from result_text(child, child_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from result_text(item, f"{path}[{index}]")


def result_documents(path: Path) -> Iterator[object]:
    if path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)
    else:
        yield json.loads(path.read_text(encoding="utf-8"))


def jsonl_text_count(path: Path) -> int:
    count = 0
    for row in load_jsonl(path):
        count += sum(1 for _ in iter_training_text(row))
    return count


def add_asset(
    assets: dict[str, dict],
    path: Path,
    extraction: Path,
    scopes: set[str],
    manifest_base: Path,
    text_count: int,
) -> None:
    key = str(path.resolve())
    existing = assets.get(key)
    if existing is not None:
        if existing["textExtractionJSONL"] != relative(extraction, manifest_base):
            raise SystemExit(f"conflicting extraction for exposure asset: {path}")
        existing["scopes"] = sorted(set(existing["scopes"]) | scopes)
        return
    assets[key] = {
        "path": relative(path, manifest_base),
        "sha256": sha256(path),
        "projectControlled": True,
        "scopes": sorted(scopes),
        "textExtractionJSONL": relative(extraction, manifest_base),
        "textExtractionSHA256": sha256(extraction),
        "textCount": text_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", type=Path)
    parser.add_argument("claim_manifest", type=Path)
    parser.add_argument("release_contract", type=Path)
    parser.add_argument("upstream_attestations", type=Path)
    parser.add_argument("output_manifest", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--protected-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--router-report", type=Path, action="append", default=[])
    parser.add_argument("--evidence", type=Path, action="append", default=[])
    parser.add_argument("--results-directory", type=Path)
    args = parser.parse_args()

    for output in (args.output_manifest, args.output_directory / "exact-memory.jsonl"):
        if output.exists() and output.stat().st_size:
            raise SystemExit(f"refusing to overwrite non-empty output: {output}")

    sources = load_jsonl(args.sources)
    claim = load(args.claim_manifest)
    release = load(args.release_contract)
    attestation_file = load(args.upstream_attestations)
    if sha256(args.sources) != claim.get("frozenSources", {}).get("sha256"):
        raise SystemExit("claim manifest does not authenticate the frozen source suite")
    if not sources or any(row.get("references") != [] for row in sources):
        raise SystemExit("expected a non-empty source-only frozen suite")
    minimum_source_date = date.fromisoformat(claim["sourcePolicy"]["minimumCreationDate"])
    if any(date.fromisoformat(str(row.get("sourceCreatedAt"))) < minimum_source_date for row in sources):
        raise SystemExit("frozen source predates the registered source cutoff")

    upstream = release.get("upstreamModels")
    supplied = attestation_file.get("models")
    if not isinstance(upstream, dict) or not isinstance(supplied, list):
        raise SystemExit("missing upstream release or attestation records")
    attestations: list[dict] = []
    seen_upstream: set[str] = set()
    for record in supplied:
        repository = str(record.get("repository", ""))
        revision = str(record.get("revision", ""))
        key = f"{repository}@{revision}"
        if key in seen_upstream or key not in upstream:
            raise SystemExit(f"unknown or duplicate upstream attestation: {key}")
        seen_upstream.add(key)
        metadata = record.get("revisionMetadata")
        if (
            not isinstance(metadata, dict)
            or metadata.get("id") != repository
            or metadata.get("sha") != revision
            or upstream[key].get("license") != record.get("license")
        ):
            raise SystemExit(f"upstream attestation mismatch: {key}")
        if parse_timestamp(metadata.get("createdAt"), key).date() >= minimum_source_date:
            raise SystemExit(f"upstream revision is not temporally excluded: {key}")
        parse_timestamp(metadata.get("lastModified"), key)
        rendered = dict(record)
        rendered["revisionMetadataSHA256"] = canonical_sha256(metadata)
        attestations.append(rendered)
    if seen_upstream != set(upstream):
        raise SystemExit("upstream attestations do not cover the exact release lineage")

    args.output_directory.mkdir(parents=True, exist_ok=True)
    manifest_base = args.output_manifest.parent
    assets: dict[str, dict] = {}
    for label, record in release.get("datasetFiles", {}).items():
        path = verified_path(record.get("path"), record.get("sha256"), label)
        scopes = {"training"} if record.get("split") == "train" else {"development", "model-selection"}
        add_asset(assets, path, path, scopes, manifest_base, jsonl_text_count(path))

    memory_record = release.get("translationMemory", {}).get("trainingData", {})
    memory_training = verified_path(
        memory_record.get("path"), memory_record.get("sha256"), "translation-memory training"
    )
    add_asset(
        assets,
        memory_training,
        memory_training,
        {"training", "exact-memory"},
        manifest_base,
        jsonl_text_count(memory_training),
    )

    for protected in args.protected_jsonl:
        path = protected.resolve()
        add_asset(
            assets,
            path,
            path,
            {"development", "router", "model-selection"},
            manifest_base,
            jsonl_text_count(path),
        )

    runtime_record = release.get("translationMemory", {}).get("runtime", {})
    runtime_memory = verified_path(
        runtime_record.get("path"), runtime_record.get("sha256"), "runtime exact memory"
    )
    runtime = load(runtime_memory)
    memory_rows: list[dict] = []
    for direction, pairs in sorted(runtime.get("entries", {}).items()):
        if not isinstance(pairs, dict):
            raise SystemExit(f"invalid exact-memory direction: {direction}")
        for source, target in sorted(pairs.items()):
            memory_rows.append({"direction": direction, "source": source, "target": target})
    memory_extraction = args.output_directory / "exact-memory.jsonl"
    write_new(
        memory_extraction,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in memory_rows),
    )
    add_asset(
        assets,
        runtime_memory,
        memory_extraction,
        {"exact-memory"},
        manifest_base,
        len(memory_rows) * 2,
    )

    generated_evidence: list[Path] = []
    if args.results_directory is not None:
        result_files = sorted(
            path.resolve()
            for path in args.results_directory.iterdir()
            if path.is_file() and (path.suffix == ".json" or path.suffix == ".jsonl")
        )
        inventory = {
            "schemaVersion": 1,
            "purpose": "pre-freeze project model-selection text inventory",
            "files": [
                {"path": relative(path, ROOT), "sha256": sha256(path), "bytes": path.stat().st_size}
                for path in result_files
            ],
        }
        inventory_path = args.output_directory / "model-selection-results.inventory.json"
        write_new(
            inventory_path,
            json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        extraction_path = args.output_directory / "model-selection-results.jsonl"
        seen_text: set[str] = set()
        extraction_rows: list[dict] = []
        for result_path in result_files:
            for document in result_documents(result_path):
                for json_path, text in result_text(document):
                    if text in seen_text:
                        continue
                    seen_text.add(text)
                    extraction_rows.append(
                        {
                            "source": text,
                            "origin": relative(result_path, ROOT),
                            "jsonPath": json_path,
                        }
                    )
        write_new(
            extraction_path,
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in extraction_rows
            ),
        )
        add_asset(
            assets,
            inventory_path,
            extraction_path,
            {"model-selection"},
            manifest_base,
            len(extraction_rows),
        )
        generated_evidence.append(inventory_path)

    evidence_paths: list[Path] = [
        args.release_contract.resolve(),
        args.upstream_attestations.resolve(),
        *[path.resolve() for path in args.router_report],
        *[path.resolve() for path in args.evidence],
        *generated_evidence,
    ]
    for label, record in release.get("trainingManifests", {}).items():
        evidence_paths.append(
            verified_path(
                label,
                record.get("sha256"),
                "training manifest",
            )
        )
    for label, record in release.get("lineageManifests", {}).items():
        evidence_paths.append(verified_path(label, record.get("sha256"), "lineage manifest"))
    unique_evidence: dict[str, Path] = {}
    for evidence in evidence_paths:
        if not evidence.is_file():
            raise SystemExit(f"missing evidence asset: {evidence}")
        unique_evidence[str(evidence.resolve())] = evidence.resolve()
    evidence_assets = [
        {
            "path": relative(path, manifest_base),
            "sha256": sha256(path),
            "purpose": "hash-bound release-lineage or exposure evidence",
        }
        for path in sorted(unique_evidence.values())
    ]
    teacher_evidence_hashes = sorted(
        {
            record["sha256"]
            for record in release.get("trainingManifests", {}).values()
        }
    )
    if not teacher_evidence_hashes:
        teacher_evidence_hashes = [sha256(args.release_contract)]

    output = {
        "schemaVersion": 2,
        "purpose": "claim benchmark contamination exposure contract",
        "coverageBasis": "exact-project-controlled-plus-upstream-revision-temporal-exclusion",
        "projectControlledExposureComplete": True,
        "upstreamExactRowsComplete": False,
        "upstreamLimitation": (
            "Exact upstream ElanMT pretraining rows are not locally enumerable; only the pinned "
            "model revisions are temporally excluded because they predate creation of this "
            "private project-owned source suite."
        ),
        "frozenSourcesSHA256": sha256(args.sources),
        "releaseContract": {
            "path": relative(args.release_contract, manifest_base),
            "sha256": sha256(args.release_contract),
        },
        "trainingTeacherModelsComplete": True,
        "trainingTeacherModels": [],
        "upstreamRevisionAttestations": sorted(
            attestations, key=lambda value: (value["repository"], value["revision"])
        ),
        "assetCount": len(assets),
        "assets": sorted(assets.values(), key=lambda value: value["path"]),
        "evidenceAssetCount": len(evidence_assets),
        "evidenceAssets": evidence_assets,
        "zeroTextScopeAttestations": [
            {
                "scope": scope,
                "reason": (
                    "The current release contract declares licensed human-reference targets and "
                    "no synthetic or model teacher inputs/outputs in the shipped model lineage."
                ),
                "evidenceAssetSHA256s": teacher_evidence_hashes,
            }
            for scope in ("teacher-input", "teacher-output")
        ],
    }
    write_new(
        args.output_manifest,
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(
        json.dumps(
            {
                "manifest": str(args.output_manifest),
                "manifestSHA256": sha256(args.output_manifest),
                "assets": len(assets),
                "evidenceAssets": len(evidence_assets),
                "textExtractions": sum(asset["textCount"] for asset in assets.values()),
                "upstreamExactRowsComplete": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
