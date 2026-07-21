#!/usr/bin/env python3
"""Freeze a model-independent, stratified M2M-100 development screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DIRECTIONS = {("en-US", "ja-JP"), ("ja-JP", "en-US")}
DOMAINS = {
    "human-translated-news",
    "ministry-published-legal",
    "professional-wikipedia",
    "everyday-conversation",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(seed: str, identifier: str) -> str:
    return hashlib.sha256(f"{seed}\0{identifier}".encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--per-domain", type=int, default=10)
    parser.add_argument("--seed", default="mimi-m2m100-418m-feasibility-v1")
    args = parser.parse_args()

    if args.per_domain <= 0:
        raise SystemExit("--per-domain must be positive")
    if args.output.exists() or args.manifest.exists():
        raise SystemExit("refusing to overwrite a frozen M2M-100 feasibility suite")
    if not args.source.is_file() or args.source.is_symlink():
        raise SystemExit("source suite is missing or a symlink")

    rows = [
        json.loads(line)
        for line in args.source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    identifiers = [row.get("id") for row in rows]
    if not rows or any(not isinstance(value, str) or not value for value in identifiers):
        raise SystemExit("source suite has missing case IDs")
    if len(identifiers) != len(set(identifiers)):
        raise SystemExit("source suite has duplicate case IDs")

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        direction = (row.get("sourceLanguage"), row.get("targetLanguage"))
        domain = row.get("domain")
        references = row.get("references")
        if direction not in DIRECTIONS or domain not in DOMAINS:
            raise SystemExit(f"unexpected source stratum: {direction} / {domain}")
        if (
            not isinstance(row.get("source"), str)
            or not row["source"].strip()
            or not isinstance(references, list)
            or not references
            or any(not isinstance(value, str) or not value.strip() for value in references)
        ):
            raise SystemExit(f"source case lacks source or human reference: {row.get('id')}")
        groups[(direction[0], direction[1], domain)].append(row)

    expected_groups = {
        (source, target, domain)
        for source, target in DIRECTIONS
        for domain in DOMAINS
    }
    if set(groups) != expected_groups:
        raise SystemExit("source suite does not cover every required direction/domain stratum")

    selected: list[dict[str, Any]] = []
    selection: dict[str, dict[str, Any]] = {}
    for key in sorted(groups):
        candidates = sorted(groups[key], key=lambda row: (rank(args.seed, row["id"]), row["id"]))
        if len(candidates) < args.per_domain:
            raise SystemExit(f"too few cases for stratum {key}: {len(candidates)}")
        chosen = candidates[: args.per_domain]
        direction = f"{key[0]}>{key[1]}"
        selection[f"{direction}/{key[2]}"] = {
            "available": len(candidates),
            "selected": len(chosen),
            "ids": [row["id"] for row in chosen],
        }
        for original in chosen:
            row = dict(original)
            row["split"] = "m2m100-418m-feasibility-v1"
            row["claimEligible"] = False
            row["screenRole"] = "model-independent-development-architecture-gate"
            row["selectionRankSha256"] = rank(args.seed, row["id"])
            selected.append(row)

    selected.sort(
        key=lambda row: (
            row["sourceLanguage"],
            row["targetLanguage"],
            row["domain"],
            row["selectionRankSha256"],
            row["id"],
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in selected
        ),
        encoding="utf-8",
    )
    counts = Counter(
        f"{row['sourceLanguage']}>{row['targetLanguage']}" for row in selected
    )
    manifest = {
        "schemaVersion": 1,
        "suiteID": "mimi-m2m100-418m-feasibility-v1",
        "status": "frozen-before-m2m100-inference",
        "role": "diagnostic-development-screen-not-promotion-evidence",
        "seed": args.seed,
        "source": {
            "path": args.source.as_posix(),
            "sha256": sha256(args.source),
            "publicBenchmarkMayOverlapPretraining": True,
        },
        "output": {
            "path": args.output.as_posix(),
            "sha256": sha256(args.output),
            "cases": len(selected),
        },
        "directions": dict(sorted(counts.items())),
        "perDomainPerDirection": args.per_domain,
        "selection": selection,
        "model": {
            "repository": "facebook/m2m100_418M",
            "revision": "55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636",
            "license": "MIT",
            "pytorchCheckpointBytes": 1935796948,
            "architecture": "single-bidirectional-dense-encoder-decoder",
        },
        "advanceGate": {
            "maximumChrFPlusPlusRegressionPerDirection": 0.25,
            "requiresFewerStrictCriticalFailuresPerDirection": True,
            "maximumDomainChrFPlusPlusRegression": 1.0,
            "maximumEstimatedFourBitPackBytes": 500000000,
            "requiresPinnedMITModelAndTokenizerEvidence": True,
        },
        "limitations": [
            "small development screen with only ten cases per direction/domain",
            "public source corpora may overlap M2M-100 pretraining",
            "not a substitute for the sealed 400+400 promotion benchmark",
            "PyTorch latency is not native Swift/MLX latency",
        ],
        "claimEligible": False,
        "selectedWithoutCandidateOutputs": True,
    }
    write_json(args.manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
