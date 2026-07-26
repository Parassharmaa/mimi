#!/usr/bin/env python3
"""Evaluate a frozen source-only expert router without tuning on the target suite."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from evaluate_expert_router import (
    align,
    bootstrap_interval,
    load_suite,
    report_rows,
    routed_summary,
    sha256,
)
from source_expert_router import SourceExpertRouter


DIRECTIONS = {
    "en-ja": ("en-US", "ja-JP"),
    "ja-en": ("ja-JP", "en-US"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("baseline_report", type=Path)
    parser.add_argument("expert_report", type=Path)
    parser.add_argument("router", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--direction", choices=tuple(DIRECTIONS), required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        raise SystemExit("bootstrap sample count must be positive")

    router_payload = json.loads(args.router.read_text(encoding="utf-8"))
    if router_payload.get("direction") != args.direction:
        raise SystemExit("router direction differs from requested evaluation")
    router = SourceExpertRouter(router_payload)
    rows = align(
        load_suite(args.suite, DIRECTIONS[args.direction]),
        report_rows(args.baseline_report),
        report_rows(args.expert_report),
        f"{args.direction}-frozen-router-evaluation",
    )
    predictions = np.array([router.score(row["source"]) for row in rows])
    threshold = float(router_payload["routing"]["scoreThreshold"])
    minimum_source_characters = int(
        router_payload["routing"]["minimumSourceCharacters"]
    )
    summary = routed_summary(
        rows,
        predictions,
        threshold,
        minimum_source_characters,
    )
    summary["pairedBootstrap95"] = bootstrap_interval(
        rows,
        predictions,
        threshold,
        minimum_source_characters,
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    summary["domains"] = {}
    for domain in sorted({row["domain"] for row in rows}):
        indices = [
            index for index, row in enumerate(rows) if row["domain"] == domain
        ]
        summary["domains"][domain] = routed_summary(
            [rows[index] for index in indices],
            predictions[indices],
            threshold,
            minimum_source_characters,
        )
    route_domains = Counter(
        row["domain"]
        for row in rows
        if router.routes_to_expert(row["source"])
    )

    report = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "purpose": "frozen source-only expert router external evaluation",
        "promotionEligible": False,
        "direction": args.direction,
        "inputs": {
            "suite": {"path": str(args.suite.resolve()), "sha256": sha256(args.suite)},
            "baselineReport": {
                "path": str(args.baseline_report.resolve()),
                "sha256": sha256(args.baseline_report),
            },
            "expertReport": {
                "path": str(args.expert_report.resolve()),
                "sha256": sha256(args.expert_report),
            },
            "router": {
                "path": str(args.router.resolve()),
                "sha256": sha256(args.router),
            },
        },
        "contract": {
            "routerFrozenBeforeSuiteEvaluation": True,
            "routerUsesReferences": False,
            "bootstrapGrouping": "source/document group_id with replacement",
            "bootstrapSamples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "routeDomains": dict(sorted(route_domains.items())),
        "summary": summary,
        "decision": {
            "passesExternalEvaluation": summary["pairedBootstrap95"][0] > 0,
            "doesNotAuthorizeAppIntegration": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
