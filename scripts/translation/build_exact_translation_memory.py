#!/usr/bin/env python3
"""Build a compact exact-source translation memory from licensed training rows."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from sacrebleu import sentence_chrf


CRITICAL_TOKEN_RE = re.compile(
    r"https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+"
    r"|\{[^{}]+\}|%[A-Za-z]|<[A-Za-z][^<>]*>|%"
    r"|(?<![\d.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)*(?!\d|\.\d)"
)
LANGUAGE_DIRECTIONS = {
    ("en-US", "ja-JP"): "en-ja",
    ("ja-JP", "en-US"): "ja-en",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def critical_tokens(value: str) -> list[str]:
    return sorted(token.replace(",", "") for token in CRITICAL_TOKEN_RE.findall(normalize(value)))


def direction(row: dict) -> str:
    key = (str(row.get("source_language")), str(row.get("target_language")))
    if key not in LANGUAGE_DIRECTIONS:
        raise SystemExit(f"unsupported language direction: {key}")
    return LANGUAGE_DIRECTIONS[key]


def document_id(row: dict) -> str:
    source_id = str(row.get("source_id", ""))
    if ":" not in source_id or not source_id.split(":", 1)[0]:
        raise SystemExit(f"row lacks a document-grouped source_id: {row.get('id')}")
    return source_id.split(":", 1)[0]


def target_medoid(target_counts: Counter[str]) -> tuple[str, float, float]:
    targets = sorted(target_counts)
    similarities = {
        (left, right): sentence_chrf(left, [right], word_order=2).score
        for left, right in itertools.combinations(targets, 2)
    }

    def similarity(left: str, right: str) -> float:
        if left == right:
            return 100.0
        if (left, right) in similarities:
            return similarities[(left, right)]
        return similarities[(right, left)]

    def rank(target: str) -> tuple[float, int, int, str]:
        mean_similarity = sum(similarity(target, other) for other in targets) / len(targets)
        return mean_similarity, target_counts[target], -len(target), target

    selected = max(targets, key=rank)
    pairwise = list(similarities.values()) or [100.0]
    return selected, min(pairwise), sum(pairwise) / len(pairwise)


def write_deterministic_gzip(path: Path, payload: dict) -> None:
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as archive:
        archive.write(encoded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("training_data", type=Path)
    parser.add_argument("runtime_output", type=Path)
    parser.add_argument("audit_output", type=Path)
    parser.add_argument("--minimum-documents", type=int, default=2)
    parser.add_argument("--maximum-source-characters", type=int, default=64)
    parser.add_argument("--maximum-target-characters", type=int, default=128)
    args = parser.parse_args()
    if args.minimum_documents < 2:
        raise SystemExit("--minimum-documents must be at least 2")
    if args.maximum_source_characters < 1 or args.maximum_target_characters < 1:
        raise SystemExit("character limits must be positive")

    rows = [
        json.loads(line)
        for line in args.training_data.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    groups: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    conflicting_documents: set[tuple[str, str, str]] = set()
    for row in rows:
        source = normalize(str(row.get("source", "")))
        target = normalize(str(row.get("target", "")))
        if not source or not target:
            raise SystemExit(f"empty source or target: {row.get('id')}")
        doc = document_id(row)
        current_direction = direction(row)
        conflict_key = (current_direction, source, doc)
        if conflict_key in conflicting_documents:
            continue
        group = groups[(current_direction, source)]
        previous = group.get(doc)
        if previous is not None and normalize(str(previous["target"])) != target:
            group.pop(doc)
            conflicting_documents.add(conflict_key)
            continue
        group.setdefault(doc, row)

    entries = {"en-ja": {}, "ja-en": {}}
    evidence = []
    rejected = Counter()
    rejected["conflicting-document-targets"] = len(conflicting_documents)
    for (current_direction, source), documents in sorted(groups.items()):
        if len(documents) < args.minimum_documents:
            rejected["insufficient-distinct-documents"] += 1
            continue
        target_counts = Counter(normalize(str(row["target"])) for row in documents.values())
        target, minimum_similarity, mean_similarity = target_medoid(target_counts)
        if len(source) > args.maximum_source_characters:
            rejected["source-too-long"] += 1
            continue
        if len(target) > args.maximum_target_characters:
            rejected["target-too-long"] += 1
            continue
        if critical_tokens(source) != critical_tokens(target):
            rejected["critical-token-mismatch"] += 1
            continue
        entries[current_direction][source] = target
        evidence.append(
            {
                "direction": current_direction,
                "source": source,
                "selectedTarget": target,
                "distinctDocuments": len(documents),
                "minimumPairwiseTargetChrFPlusPlus": minimum_similarity,
                "meanPairwiseTargetChrFPlusPlus": mean_similarity,
                "observations": [
                    {
                        "documentID": doc,
                        "sourceID": row["source_id"],
                        "target": normalize(str(row["target"])),
                    }
                    for doc, row in sorted(documents.items())
                ],
            }
        )

    policy = {
        "normalization": "NFKC then Unicode-whitespace collapse",
        "minimumDistinctDocuments": args.minimum_documents,
        "maximumSourceCharacters": args.maximum_source_characters,
        "maximumTargetCharacters": args.maximum_target_characters,
        "selection": "observed human target medoid by mean sentence chrF++",
        "criticalTokens": "exact NFKC multiset of ASCII URLs, placeholders, markup, percent, and digits",
    }
    audit = {
        "schemaVersion": 1,
        "purpose": "training-only cross-document exact translation-memory evidence",
        "trainingData": {"path": str(args.training_data), "sha256": sha256(args.training_data)},
        "sourceLicense": "PDL-1.0-compatible-CC-BY-4.0",
        "policy": policy,
        "counts": {
            "trainingRows": len(rows),
            "entries": sum(len(values) for values in entries.values()),
            "directions": {name: len(values) for name, values in entries.items()},
            "rejected": dict(sorted(rejected.items())),
        },
        "evidence": evidence,
        "doesNotAuthorizeAppIntegration": True,
    }
    write_deterministic_gzip(args.audit_output, audit)
    runtime = {
        "schemaVersion": 1,
        "normalization": policy["normalization"],
        "trainingDataSHA256": sha256(args.training_data),
        "auditSHA256": sha256(args.audit_output),
        "sourceLicense": audit["sourceLicense"],
        "doesNotAuthorizeAppIntegration": True,
        "entries": entries,
    }
    args.runtime_output.parent.mkdir(parents=True, exist_ok=True)
    args.runtime_output.write_text(
        json.dumps(runtime, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "entries": audit["counts"]["entries"],
                "directions": audit["counts"]["directions"],
                "runtimeBytes": args.runtime_output.stat().st_size,
                "runtimeSHA256": sha256(args.runtime_output),
                "auditBytes": args.audit_output.stat().st_size,
                "auditSHA256": sha256(args.audit_output),
            }
        )
    )


if __name__ == "__main__":
    main()
