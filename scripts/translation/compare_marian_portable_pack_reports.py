#!/usr/bin/env python3
"""Prove a portable Marian metadata clone preserves routed model behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CASE_FIELDS = (
    "caseID",
    "sourceLanguage",
    "targetLanguage",
    "domain",
    "source",
    "references",
    "hypothesis",
    "outputTokenIDs",
    "selectedEngine",
    "selectedNeuralEngine",
    "routedToExpert",
    "routerScore",
    "criticalFallbackDirection",
    "criticalFallbackUsed",
    "criticalTokenGuardPasses",
    "plausibilityGuardPasses",
    "runtimeAccepted",
    "failureReason",
    "outputShortlistTokens",
    "claimEligible",
)
MANIFEST_PATHS = {"manifest.json"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path: Path) -> dict[str, int | str]:
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def files(root: Path) -> dict[str, dict[str, int | str]]:
    return {
        item.relative_to(root).as_posix(): record(item)
        for item in sorted(root.rglob("*"))
        if item.is_file()
    }


def indexed(report: dict[str, Any], path: Path) -> dict[str, dict[str, Any]]:
    results = report.get("results")
    if report.get("schemaVersion") != 1 or not isinstance(results, list):
        raise ValueError(f"invalid Marian runtime report: {path}")
    rows = {str(row.get("caseID", "")): row for row in results}
    if "" in rows or len(rows) != len(results):
        raise ValueError(f"empty or duplicate case ID: {path}")
    return rows


def case_view(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in CASE_FIELDS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_pack", type=Path)
    parser.add_argument("portable_pack", type=Path)
    parser.add_argument("source_report", type=Path)
    parser.add_argument("portable_report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source_manifest_path = args.source_pack / "manifest.json"
    portable_manifest_path = args.portable_pack / "manifest.json"
    source_manifest = load(source_manifest_path)
    portable_manifest = load(portable_manifest_path)
    if portable_manifest.get("portableMetadata", {}).get(
        "sourceManifestSha256"
    ) != sha256(source_manifest_path):
        raise SystemExit("portable pack does not identify the source manifest")
    if portable_manifest.get("portableMetadata", {}).get(
        "weightPayloadUnchanged"
    ) is not True:
        raise SystemExit("portable pack does not declare unchanged model payload")

    source_files = files(args.source_pack)
    portable_files = files(args.portable_pack)
    if set(source_files) != set(portable_files):
        raise SystemExit("source and portable packs contain different file paths")
    changed = sorted(
        relative
        for relative in source_files
        if source_files[relative] != portable_files[relative]
    )
    expected_changed = sorted(
        relative
        for relative in source_files
        if relative == "manifest.json" or relative.endswith("/manifest.json")
        if source_files[relative] != portable_files[relative]
    )
    if changed != expected_changed or not changed:
        raise SystemExit("portable pack changes non-manifest payload or no metadata")

    source_report = load(args.source_report)
    portable_report = load(args.portable_report)
    expected_source_revision = f"moe-manifest-sha256:{sha256(source_manifest_path)}"
    expected_portable_revision = f"moe-manifest-sha256:{sha256(portable_manifest_path)}"
    if source_report.get("modelRevision") != expected_source_revision:
        raise SystemExit("source report is not bound to source pack")
    if portable_report.get("modelRevision") != expected_portable_revision:
        raise SystemExit("portable report is not bound to portable pack")
    source_rows = indexed(source_report, args.source_report)
    portable_rows = indexed(portable_report, args.portable_report)
    if set(source_rows) != set(portable_rows):
        raise SystemExit("runtime reports cover different cases")
    mismatches = [
        case_id
        for case_id in sorted(source_rows)
        if case_view(source_rows[case_id]) != case_view(portable_rows[case_id])
    ]
    summary_fields = (
        "status",
        "cases",
        "failures",
        "selectedEngineCounts",
        "failureCounts",
        "runtimeAcceptedCases",
        "directionShortlistTokens",
    )
    summary_exact = all(
        source_report.get("summary", {}).get(field)
        == portable_report.get("summary", {}).get(field)
        for field in summary_fields
    ) and source_report.get("status") == portable_report.get("status")
    status = "passed" if not mismatches and summary_exact else "failed"
    output = {
        "schemaVersion": 1,
        "status": status,
        "purpose": "exact non-timing equivalence of repository-relative Marian metadata clone",
        "sourcePack": {
            "path": str(args.source_pack),
            "bytes": sum(value["bytes"] for value in source_files.values()),
            "manifestSha256": sha256(source_manifest_path),
        },
        "portablePack": {
            "path": str(args.portable_pack),
            "bytes": sum(value["bytes"] for value in portable_files.values()),
            "manifestSha256": sha256(portable_manifest_path),
        },
        "payload": {
            "filePathsExact": True,
            "changedManifestFiles": changed,
            "unchangedFiles": len(source_files) - len(changed),
            "nonManifestPayloadExact": True,
        },
        "runtime": {
            "cases": len(source_rows),
            "exactCases": len(source_rows) - len(mismatches),
            "mismatchCaseIDs": mismatches,
            "summaryExactExcludingTimingAndMemory": summary_exact,
            "sourceReport": {
                "path": str(args.source_report),
                "sha256": sha256(args.source_report),
            },
            "portableReport": {
                "path": str(args.portable_report),
                "sha256": sha256(args.portable_report),
            },
        },
        "claimEligible": False,
        "claimBlocker": "metadata-equivalence-only",
        "doesNotAuthorizeDistribution": True,
        "doesNotAuthorizeAppIntegration": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": status,
                "cases": len(source_rows),
                "changedManifestFiles": changed,
                "unchangedFiles": output["payload"]["unchangedFiles"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
