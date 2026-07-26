#!/usr/bin/env python3
"""Convert canonical Codex teacher output into Mimi's metric-report contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from filter_synthetic_batch import response_payload


def rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("teacher_output", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists() and args.output.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")

    suite_rows = rows(args.suite)
    suite = {str(row.get("id", "")): row for row in suite_rows}
    if not suite or len(suite) != len(suite_rows):
        raise SystemExit("suite IDs are empty or duplicated")
    teacher: dict[str, tuple[dict, dict]] = {}
    for batch_row in rows(args.teacher_output):
        source_id = str(batch_row.get("custom_id", ""))
        if source_id not in suite or source_id in teacher:
            raise SystemExit(f"invalid teacher output ID: {source_id}")
        payload, body = response_payload(batch_row)
        if (
            set(payload) != {"source_id", "canonical_translation", "risk_tags"}
            or payload["source_id"] != source_id
            or not isinstance(payload["canonical_translation"], str)
            or not payload["canonical_translation"].strip()
        ):
            raise SystemExit(f"invalid canonical payload: {source_id}")
        teacher[source_id] = (payload, body)
    if set(teacher) != set(suite):
        raise SystemExit("teacher output does not cover the exact suite")

    models = {str(body.get("model", "")) for _, body in teacher.values()}
    if len(models) != 1 or not next(iter(models)):
        raise SystemExit("canonical output must contain exactly one teacher model")
    results = []
    for source_id in sorted(suite):
        case = suite[source_id]
        payload, _ = teacher[source_id]
        results.append({
            "caseID": source_id,
            "sourceLanguage": case["sourceLanguage"],
            "targetLanguage": case["targetLanguage"],
            "domain": case["domain"],
            "source": case["source"],
            "references": case["references"],
            "claimEligible": False,
            "hypothesis": payload["canonical_translation"].strip(),
        })
    report = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "engine": f"{next(iter(models))}:canonical-target-pilot",
        "modelRevision": next(iter(models)),
        "claimEligible": False,
        "trainingReferenceDiagnosticOnly": True,
        "reasoningTraceStored": False,
        "suiteSHA256": sha256(args.suite),
        "teacherOutputSHA256": sha256(args.teacher_output),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "rows": len(results),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
