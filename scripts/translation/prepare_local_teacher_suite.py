#!/usr/bin/env python3
"""Convert source-only licensed seeds into a non-claimable local-teacher suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path


LANGUAGES = {"en-US", "ja-JP"}


def rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"missing input: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalized(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def ngrams(text: str, size: int = 5) -> set[str]:
    value = normalized(text)
    if len(value) < size:
        return {value}
    return {value[index:index + size] for index in range(len(value) - size + 1)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("seeds", type=Path)
    parser.add_argument("protected_suite", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--additional-protected-suite", type=Path, action="append", default=[])
    parser.add_argument("--maximum-jaccard", type=float, default=0.8)
    args = parser.parse_args()
    if args.output.exists() and args.output.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    if not 0 <= args.maximum_jaccard <= 1:
        raise SystemExit("maximum-jaccard must be between zero and one")

    protected_suites = [args.protected_suite, *args.additional_protected_suite]
    protected = [
        ngrams(text)
        for path in protected_suites
        for row in rows(path)
        for text in [row.get("source", ""), *row.get("references", [])]
        if str(text).strip()
    ]
    output: list[dict] = []
    identifiers: set[str] = set()
    rejected = 0
    for seed in rows(args.seeds):
        identifier = str(seed.get("id", "")).strip()
        source = str(seed.get("source", "")).strip()
        source_language = str(seed.get("source_language", ""))
        target_language = str(seed.get("target_language", ""))
        if not identifier or not source or {source_language, target_language} != LANGUAGES:
            raise SystemExit(f"invalid local-teacher seed: {identifier or '<missing-id>'}")
        if identifier in identifiers:
            raise SystemExit(f"duplicate seed ID: {identifier}")
        identifiers.add(identifier)
        split = str(seed.get("split", "")).lower()
        if split in {"benchmark", "heldout", "test", "canary"} or seed.get("claimEligible"):
            raise SystemExit(f"protected seed cannot enter local-teacher suite: {identifier}")
        candidate = ngrams(source)
        if any(
            len(candidate & value) / max(1, len(candidate | value)) > args.maximum_jaccard
            for value in protected
        ):
            rejected += 1
            continue
        output.append({
            "id": identifier,
            "sourceLanguage": source_language,
            "targetLanguage": target_language,
            "domain": seed.get("domain", "unknown"),
            "source": source,
            "references": [],
            "claimEligible": False,
            "sourceLicense": seed.get("license"),
            "sourceProvenance": seed.get("provenance"),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output),
        encoding="utf-8",
    )
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest = {
        "schema_version": 1,
        "purpose": "source-only local teacher generation; never evaluation evidence",
        "claim_eligible": False,
        "inputs": {
            "seeds": {"path": str(args.seeds.resolve()), "sha256": sha256(args.seeds)},
            "protected_suites": [
                {"path": str(path.resolve()), "sha256": sha256(path)}
                for path in protected_suites
            ],
        },
        "maximum_five_gram_jaccard": args.maximum_jaccard,
        "rows": len(output),
        "rejected_near_protected": rejected,
        "output": {"path": str(args.output.resolve()), "sha256": sha256(args.output)},
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
