#!/usr/bin/env python3
"""Merge independently prepared teacher seeds with provenance and contamination checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path


ALLOWED_LICENSES = {
    "CC0",
    "CC0-1.0",
    "CC-BY-3.0",
    "CC-BY-4.0",
    "CC-BY-SA-3.0",
    "CC-BY-SA-4.0",
    "CC BY 3.0",
    "CC BY 4.0",
    "CC BY-SA 3.0",
    "CC BY-SA 4.0",
    "Public Domain",
    "project-owned",
}
LANGUAGES = {
    ("en-US", "ja-JP"): "en-ja",
    ("ja-JP", "en-US"): "ja-en",
}


def load_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def ngrams(text: str, size: int = 5) -> set[str]:
    value = normalized(text).replace(" ", "")
    return {value[index:index + size] for index in range(max(1, len(value) - size + 1))}


def near_protected(text: str, protected: list[set[str]], threshold: float) -> bool:
    candidate = ngrams(text)
    return any(
        len(candidate & heldout) / max(1, len(candidate | heldout)) > threshold
        for heldout in protected
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("protected_benchmark", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--maximum-jaccard", type=float, default=0.80)
    args = parser.parse_args()

    if args.output.exists() and args.output.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    protected = [
        ngrams(text)
        for row in load_rows(args.protected_benchmark)
        for text in (row["source"], *row.get("references", []))
    ]
    output: list[dict] = []
    identifiers: set[str] = set()
    sources: set[tuple[str, str]] = set()
    input_records: list[dict] = []
    for path in args.inputs:
        rows = load_rows(path)
        input_records.append({"path": str(path), "sha256": sha256(path), "rows": len(rows)})
        for row in rows:
            row_id = str(row.get("id", "")).strip()
            if not row_id or row_id in identifiers:
                raise SystemExit(f"empty or duplicate seed ID: {row_id!r}")
            identifiers.add(row_id)
            split = str(row.get("split", "")).lower()
            if split in {"benchmark", "heldout", "test", "canary"} or row.get("claimEligible"):
                raise SystemExit(f"refusing protected evaluation seed: {row_id} ({split})")
            languages = (row.get("source_language"), row.get("target_language"))
            if languages not in LANGUAGES:
                raise SystemExit(f"unsupported direction: {row_id}")
            license_name = str(row.get("license", "")).strip()
            if license_name not in ALLOWED_LICENSES:
                raise SystemExit(f"unknown or non-distributable seed license: {row_id} / {license_name}")
            source = str(row.get("source", "")).strip()
            if not source or not str(row.get("provenance", "")).strip():
                raise SystemExit(f"seed lacks source or provenance: {row_id}")
            source_key = (LANGUAGES[languages], normalized(source))
            if source_key in sources:
                raise SystemExit(f"duplicate normalized source in one direction: {row_id}")
            sources.add(source_key)
            texts = [source]
            if row.get("reference_translation"):
                texts.append(str(row["reference_translation"]))
            if any(near_protected(text, protected, args.maximum_jaccard) for text in texts):
                raise SystemExit(f"seed is near the protected benchmark: {row_id}")
            output.append(row)

    output.sort(key=lambda row: str(row["id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output),
        encoding="utf-8",
    )
    counts = Counter(
        LANGUAGES[(row["source_language"], row["target_language"])] for row in output
    )
    manifest = {
        "schema_version": 1,
        "inputs": input_records,
        "protected_benchmark": {
            "path": str(args.protected_benchmark),
            "sha256": sha256(args.protected_benchmark),
        },
        "maximum_jaccard": args.maximum_jaccard,
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "rows": len(output),
        "directions": dict(sorted(counts.items())),
        "duplicate_source_policy": "fail closed within each direction",
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
