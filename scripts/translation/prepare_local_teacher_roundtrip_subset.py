#!/usr/bin/env python3
"""Build forward/reverse benchmark-shaped suites from surface-filtered candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in values),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", type=Path)
    parser.add_argument("forward_output", type=Path)
    parser.add_argument("reverse_output", type=Path)
    args = parser.parse_args()
    for path in (args.forward_output, args.reverse_output):
        if path.exists() and path.stat().st_size:
            raise SystemExit(f"refusing to overwrite non-empty output: {path}")

    candidates = rows(args.candidates)
    identifiers: set[str] = set()
    forward, reverse = [], []
    for row in candidates:
        identifier = str(row.get("source_id", "")).strip()
        if not identifier or identifier in identifiers:
            raise SystemExit("candidate rows have missing or duplicate source IDs")
        identifiers.add(identifier)
        if row.get("promotion_eligible") is not False:
            raise SystemExit(f"candidate is not explicitly claim-ineligible: {identifier}")
        common = {
            "id": identifier,
            "domain": row.get("domain", "unknown"),
            "references": [],
            "claimEligible": False,
        }
        forward.append({
            **common,
            "sourceLanguage": row["source_language"],
            "targetLanguage": row["target_language"],
            "source": row["source"],
            "sourceLicense": row.get("source_license"),
            "sourceProvenance": row.get("source_provenance"),
        })
        reverse.append({
            **common,
            "sourceLanguage": row["target_language"],
            "targetLanguage": row["source_language"],
            "source": row["target"],
            "roundTripOriginalSource": row["source"],
        })

    write_jsonl(args.forward_output, forward)
    write_jsonl(args.reverse_output, reverse)
    manifest = {
        "schema_version": 1,
        "purpose": "surface-consensus subset for local round-trip filtering",
        "claim_eligible": False,
        "rows": len(candidates),
        "input": {"path": str(args.candidates.resolve()), "sha256": sha256(args.candidates)},
        "outputs": {
            "forward": {"path": str(args.forward_output.resolve()), "sha256": sha256(args.forward_output)},
            "reverse": {"path": str(args.reverse_output.resolve()), "sha256": sha256(args.reverse_output)},
        },
    }
    args.forward_output.with_suffix(args.forward_output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
