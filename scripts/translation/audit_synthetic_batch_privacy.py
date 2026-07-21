#!/usr/bin/env python3
"""Write a content-free completeness and reasoning-retention audit for Batch output."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trace_markers(value: object) -> tuple[bool, bool]:
    summary = False
    encrypted = False
    if isinstance(value, dict):
        if value.get("type") == "reasoning" and value.get("summary"):
            summary = True
        for key, child in value.items():
            if key == "summary_text" and child:
                summary = True
            if key == "encrypted_content" and child:
                encrypted = True
            child_summary, child_encrypted = trace_markers(child)
            summary = summary or child_summary
            encrypted = encrypted or child_encrypted
    elif isinstance(value, list):
        for child in value:
            child_summary, child_encrypted = trace_markers(child)
            summary = summary or child_summary
            encrypted = encrypted or child_encrypted
    return summary, encrypted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_output", type=Path)
    parser.add_argument("audit_output", type=Path)
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args()
    if args.audit_output.exists() and args.audit_output.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty audit: {args.audit_output}")

    rows = [
        json.loads(line)
        for line in args.batch_output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise SystemExit("Batch output must contain non-empty JSON objects")
    identifiers = [str(row.get("custom_id", "")).strip() for row in rows]
    if any(not value for value in identifiers) or len(identifiers) != len(set(identifiers)):
        raise SystemExit("Batch output contains an empty or duplicate custom_id")

    http_statuses: Counter[str] = Counter()
    body_statuses: Counter[str] = Counter()
    incomplete_reasons: Counter[str] = Counter()
    batch_error_rows = 0
    final_message_rows = 0
    summary_rows = 0
    encrypted_rows = 0
    for row in rows:
        if row.get("error") not in (None, {}):
            batch_error_rows += 1
        response = row.get("response") if isinstance(row.get("response"), dict) else {}
        http_statuses[str(response.get("status_code"))] += 1
        body = response.get("body") if isinstance(response.get("body"), dict) else {}
        body_statuses[str(body.get("status"))] += 1
        reason = (body.get("incomplete_details") or {}).get("reason")
        if reason:
            incomplete_reasons[str(reason)] += 1
        output = body.get("output") if isinstance(body.get("output"), list) else []
        if any(isinstance(item, dict) and item.get("type") == "message" for item in output):
            final_message_rows += 1
        summary, encrypted = trace_markers(body)
        summary_rows += int(summary)
        encrypted_rows += int(encrypted)

    expected_ok = args.expected_count is None or len(rows) == args.expected_count
    admissible = (
        expected_ok
        and batch_error_rows == 0
        and http_statuses == Counter({"200": len(rows)})
        and body_statuses == Counter({"completed": len(rows)})
        and final_message_rows == len(rows)
        and summary_rows == 0
        and encrypted_rows == 0
    )
    audit = {
        "schemaVersion": 1,
        "purpose": "batch-privacy-and-completeness-audit",
        "input": {
            "path": args.batch_output.as_posix(),
            "sha256": sha256(args.batch_output),
            "bytes": args.batch_output.stat().st_size,
        },
        "expectedCount": args.expected_count,
        "rows": len(rows),
        "uniqueCustomIDs": len(set(identifiers)),
        "batchErrorRows": batch_error_rows,
        "httpStatuses": dict(sorted(http_statuses.items())),
        "bodyStatuses": dict(sorted(body_statuses.items())),
        "incompleteReasons": dict(sorted(incomplete_reasons.items())),
        "rowsWithFinalMessage": final_message_rows,
        "rowsWithReasoningSummary": summary_rows,
        "rowsWithEncryptedReasoning": encrypted_rows,
        "admissible": admissible,
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
