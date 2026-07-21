#!/usr/bin/env python3
"""Prepare claim-ineligible exact-memory evaluation cases from a held-out split."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("memory", type=Path)
    parser.add_argument("evaluation_data", type=Path)
    parser.add_argument("output_suite", type=Path)
    parser.add_argument("output_manifest", type=Path)
    args = parser.parse_args()
    memory = json.loads(args.memory.read_text(encoding="utf-8"))
    if memory.get("schemaVersion") != 1:
        raise SystemExit("unsupported translation-memory schema")
    entries = memory["entries"]
    rows = [
        json.loads(line)
        for line in args.evaluation_data.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    directions = {("en-US", "ja-JP"): "en-ja", ("ja-JP", "en-US"): "ja-en"}
    selected = []
    for row in rows:
        languages = (row["source_language"], row["target_language"])
        if languages not in directions:
            raise SystemExit(f"unsupported direction: {languages}")
        current_direction = directions[languages]
        if normalize(row["source"]) not in entries[current_direction]:
            continue
        selected.append(
            {
                "id": f"exact-memory-valid:{row['id']}",
                "sourceLanguage": row["source_language"],
                "targetLanguage": row["target_language"],
                "domain": row["domain"],
                "source": row["source"],
                "references": [row["target"]],
                "claimEligible": False,
                "split": "translation-memory-validation",
                "license": row.get("source_license"),
                "provenance": row.get("source_provenance"),
                "originalID": row["id"],
            }
        )
    if not selected:
        raise SystemExit("translation memory has no matches in evaluation data")
    args.output_suite.parent.mkdir(parents=True, exist_ok=True)
    args.output_suite.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    counts = {
        name: sum(
            row["sourceLanguage"] == source_language
            for row in selected
        )
        for name, source_language in (("en-ja", "en-US"), ("ja-en", "ja-JP"))
    }
    manifest = {
        "schemaVersion": 1,
        "purpose": "claim-ineligible threshold-independent translation-memory validation subset",
        "memory": {"path": str(args.memory), "sha256": sha256(args.memory)},
        "evaluationData": {"path": str(args.evaluation_data), "sha256": sha256(args.evaluation_data)},
        "outputSuite": {"path": str(args.output_suite), "sha256": sha256(args.output_suite)},
        "counts": {"cases": len(selected), "directions": counts},
        "claimEligible": False,
    }
    args.output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["counts"]))


if __name__ == "__main__":
    main()
