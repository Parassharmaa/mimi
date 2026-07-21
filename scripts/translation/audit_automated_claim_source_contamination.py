#!/usr/bin/env python3
"""Audit frozen claim sources against the exact current release lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from validate_benchmark_suite import scan_training


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_path(record: dict, label: str) -> Path:
    path = Path(str(record.get("path", "")))
    if not path.is_file() or sha256(path) != record.get("sha256"):
        raise SystemExit(f"missing or changed release-lineage input: {label} {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", type=Path)
    parser.add_argument("release_contract", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--protected-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--character-ngram-size", type=int, default=5)
    parser.add_argument("--maximum-jaccard", type=float, default=0.65)
    args = parser.parse_args()
    if args.output.exists() and args.output.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    if not 0 <= args.maximum_jaccard < 1 or args.character_ngram_size < 1:
        raise SystemExit("invalid contamination threshold")

    source_rows = rows(args.sources)
    if not source_rows:
        raise SystemExit("source suite is empty")
    source_text = []
    document_ids = set()
    for row in source_rows:
        identifier = str(row.get("id", "")).strip()
        source = str(row.get("source", "")).strip()
        document_id = str(row.get("documentID", "")).strip()
        if not identifier or not source or not document_id or row.get("references") != []:
            raise SystemExit(f"invalid source-only row: {identifier}")
        source_text.append((identifier, source))
        document_ids.add(document_id)

    contract = load(args.release_contract)
    if contract.get("schemaVersion") != 1:
        raise SystemExit("unsupported release contract")
    records: list[tuple[str, dict]] = list(contract.get("datasetFiles", {}).items())
    memory_training = contract.get("translationMemory", {}).get("trainingData")
    if not isinstance(memory_training, dict):
        raise SystemExit("release contract lacks translation-memory training lineage")
    records.append(("translation-memory-training", memory_training))
    scan_paths: list[Path] = []
    seen: set[str] = set()
    lineage: list[dict] = []
    for label, record in records:
        path = verified_path(record, label)
        if str(path) in seen:
            continue
        seen.add(str(path))
        scan_paths.append(path)
        lineage.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "rows": record.get("rows"),
                "split": record.get("split"),
            }
        )
    protected: list[dict] = []
    for path in args.protected_jsonl:
        if not path.is_file():
            raise SystemExit(f"missing protected JSONL: {path}")
        if str(path.resolve()) not in seen:
            scan_paths.append(path.resolve())
            seen.add(str(path.resolve()))
        protected.append({"path": str(path.resolve()), "sha256": sha256(path)})

    scanned = scan_training(
        scan_paths,
        source_text,
        document_ids,
        args.character_ngram_size,
        args.maximum_jaccard,
        True,
    )
    output = {
        "schemaVersion": 1,
        "status": "release-lineage-source-contamination-scan-passed",
        "scope": "source-side exact current release lineage plus explicit protected suites; references pending",
        "sources": {
            "path": str(args.sources.resolve()),
            "sha256": sha256(args.sources),
            "cases": len(source_rows),
        },
        "releaseContract": {
            "path": str(args.release_contract.resolve()),
            "sha256": sha256(args.release_contract),
        },
        "lineageFiles": lineage,
        "protectedSuites": protected,
        "policy": {
            "normalization": "NFKC lowercase whitespace collapse",
            "exactMatch": True,
            "characterNgramSize": args.character_ngram_size,
            "maximumJaccard": args.maximum_jaccard,
            "documentIDOverlapForbidden": True,
        },
        "filesScanned": len(scan_paths),
        "textsScanned": scanned,
        "semanticNeighborScanComplete": False,
        "completeExposureManifest": False,
        "claimEligible": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
