#!/usr/bin/env python3
"""Adapt a blinded candidate queue for Mimi's pinned local bilingual judge."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("review_queue", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists() and args.output.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")

    candidates = []
    seen: set[str] = set()
    for row in rows(args.review_queue):
        candidate_id = str(row.get("candidate_id", "")).strip()
        source_id = str(row.get("source_id", "")).strip()
        if not candidate_id or not source_id or candidate_id in seen:
            raise SystemExit(f"invalid or duplicate candidate: {source_id}/{candidate_id}")
        seen.add(candidate_id)
        if not str(row.get("critical_token_admission", "")).strip():
            raise SystemExit(
                f"candidate lacks deterministic critical admission: {candidate_id}"
            )
        candidates.append(
            {
                "id": candidate_id,
                "source_id": source_id,
                "source_language": row["source_language"],
                "target_language": row["target_language"],
                "domain": row["domain"],
                "source": row["source"],
                "target": row["translation"],
                "promotion_eligible": False,
            }
        )
    if not candidates:
        raise SystemExit("review queue has no candidates")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x", encoding="utf-8") as handle:
            for candidate in candidates:
                handle.write(
                    json.dumps(
                        candidate,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
    except FileExistsError as error:
        raise SystemExit(f"refusing to overwrite existing output: {args.output}") from error
    manifest = {
        "schema_version": 1,
        "purpose": "blinded local bilingual judge input for two-model consensus",
        "promotion_eligible": False,
        "review_queue": {
            "path": str(args.review_queue.resolve()),
            "sha256": sha256(args.review_queue),
        },
        "output": {
            "path": str(args.output.resolve()),
            "sha256": sha256(args.output),
            "rows": len(candidates),
        },
        "source_only_judgment": True,
        "candidate_origin_exposed": False,
        "reference_provenance_exposed": False,
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
