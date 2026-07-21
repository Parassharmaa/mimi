#!/usr/bin/env python3
"""Compare authenticated v1/v2 Swift Marian MoE smoke reports exactly."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


COMPARISON_FIELDS = (
    "schemaVersion",
    "direction",
    "expectedEngine",
    "selectedEngine",
    "source",
    "hypothesis",
    "outputTokenIDs",
    "status",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def compare_pair(label: str, source_path: Path, candidate_path: Path) -> dict[str, Any]:
    source = load_object(source_path)
    candidate = load_object(candidate_path)
    comparable_source = {field: source.get(field) for field in COMPARISON_FIELDS}
    comparable_candidate = {field: candidate.get(field) for field in COMPARISON_FIELDS}
    text_exact = source.get("hypothesis") == candidate.get("hypothesis")
    source_tokens = source.get("outputTokenIDs")
    candidate_tokens = candidate.get("outputTokenIDs")
    selected = source.get("selectedEngine")
    if selected == "translation-memory":
        token_exact = None
        valid_token_evidence = source_tokens is None and candidate_tokens is None
    else:
        token_exact = source_tokens == candidate_tokens
        valid_token_evidence = (
            isinstance(source_tokens, list)
            and bool(source_tokens)
            and all(isinstance(value, int) for value in source_tokens)
            and isinstance(candidate_tokens, list)
        )
    exact = (
        source.get("status") == "passed"
        and candidate.get("status") == "passed"
        and comparable_source == comparable_candidate
        and text_exact
        and valid_token_evidence
        and token_exact is not False
    )
    return {
        "label": label,
        "direction": source.get("direction"),
        "selectedEngine": selected,
        "source": source.get("source"),
        "hypothesis": source.get("hypothesis"),
        "outputTokenIDs": source_tokens,
        "textExactMatch": text_exact,
        "tokenExactMatch": token_exact,
        "exactMatch": exact,
        "sourceReport": {"path": str(source_path), "sha256": sha256(source_path)},
        "candidateReport": {
            "path": str(candidate_path),
            "sha256": sha256(candidate_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_pack", type=Path)
    parser.add_argument("candidate_pack", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--pair",
        action="append",
        nargs=3,
        metavar=("LABEL", "SOURCE_REPORT", "CANDIDATE_REPORT"),
        required=True,
    )
    args = parser.parse_args()
    source_manifest = args.source_pack / "manifest.json"
    candidate_manifest = args.candidate_pack / "manifest.json"
    cases = [
        compare_pair(label, Path(source), Path(candidate))
        for label, source, candidate in args.pair
    ]
    status = "passed" if cases and all(case["exactMatch"] for case in cases) else "failed"
    report = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": status,
        "comparison": "exact Swift routed text and output-token parity after tokenizer deduplication",
        "sourcePack": {
            "path": str(args.source_pack),
            "manifestSHA256": sha256(source_manifest),
        },
        "candidatePack": {
            "path": str(args.candidate_pack),
            "manifestSHA256": sha256(candidate_manifest),
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "status": status, "cases": len(cases)}))
    if status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
