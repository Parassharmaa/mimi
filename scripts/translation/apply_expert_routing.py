#!/usr/bin/env python3
"""Materialize a routed Marian benchmark report from pinned engine reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

from source_expert_router import SourceExpertRouter


CRITICAL_TOKEN_RE = re.compile(
    r"https?://[^\s]+|\{[^{}]+\}|%[A-Za-z]|<[A-Za-z][^<>]*>|%|"
    r"(?<![\d.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)*(?!\d|\.\d)"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return payload


def indexed(report: dict, path: Path) -> dict[str, dict]:
    rows = report.get("results")
    if not isinstance(rows, list):
        raise SystemExit(f"report lacks results: {path}")
    output = {str(row["caseID"]): row for row in rows}
    if len(output) != len(rows):
        raise SystemExit(f"report has duplicate case IDs: {path}")
    return output


def critical_tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", value)
    return sorted(token.replace(",", "") for token in CRITICAL_TOKEN_RE.findall(normalized))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("generalist_report", type=Path)
    parser.add_argument("en_ja_expert_report", type=Path)
    parser.add_argument("ja_en_expert_report", type=Path)
    parser.add_argument("en_ja_router", type=Path)
    parser.add_argument("ja_en_router", type=Path)
    parser.add_argument("moe_pack", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    generalist = load(args.generalist_report)
    reports = {
        "generalist": indexed(generalist, args.generalist_report),
        "en-ja": indexed(load(args.en_ja_expert_report), args.en_ja_expert_report),
        "ja-en": indexed(load(args.ja_en_expert_report), args.ja_en_expert_report),
    }
    if not (set(reports["generalist"]) == set(reports["en-ja"]) == set(reports["ja-en"])):
        raise SystemExit("generalist and expert reports do not cover identical cases")
    routers = {
        "en-ja": SourceExpertRouter.load(args.en_ja_router),
        "ja-en": SourceExpertRouter.load(args.ja_en_router),
    }
    direction_for_source = {"en-US": "en-ja", "ja-JP": "ja-en"}
    selected = []
    route_counts = {"en-ja": 0, "ja-en": 0}
    critical_token_fallbacks = {"en-ja": 0, "ja-en": 0}
    unrecoverable_critical_token_cases = []
    for row in generalist["results"]:
        direction = direction_for_source[str(row["sourceLanguage"])]
        use_expert = routers[direction].routes_to_expert(str(row["source"]))
        expert = reports[direction][str(row["caseID"])]
        chosen = dict(expert if use_expert else row)
        selected_engine = "expert" if use_expert else "generalist"
        source_tokens = critical_tokens(str(row["source"]))
        if use_expert and critical_tokens(str(expert["hypothesis"])) != source_tokens:
            if critical_tokens(str(row["hypothesis"])) == source_tokens:
                chosen = dict(row)
                selected_engine = "generalist-critical-token-fallback"
                critical_token_fallbacks[direction] += 1
            else:
                unrecoverable_critical_token_cases.append(str(row["caseID"]))
                chosen["criticalTokenFailure"] = True
        chosen["selectedEngine"] = selected_engine
        chosen["routerScore"] = routers[direction].score(str(row["source"]))
        selected.append(chosen)
        route_counts[direction] += int(use_expert)
    manifest_path = args.moe_pack / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing MoE manifest: {manifest_path}")
    output = {
        **{key: value for key, value in generalist.items() if key != "results"},
        "engine": "mlx:ElanMT-BT:4bit-g64-source-routed-moe-simulation",
        "modelRevision": f"moe-manifest-sha256:{sha256(manifest_path)}",
        "modelBytes": sum(
            path.stat().st_size for path in args.moe_pack.rglob("*") if path.is_file()
        ),
        "physicalModelCount": 4,
        "routeCounts": route_counts,
        "criticalTokenSafety": {
            "generalistFallbacks": critical_token_fallbacks,
            "unrecoverableCases": unrecoverable_critical_token_cases,
        },
        "doesNotAuthorizeAppIntegration": True,
        "inputs": {
            "generalistReport": {"path": str(args.generalist_report), "sha256": sha256(args.generalist_report)},
            "enJAExpertReport": {"path": str(args.en_ja_expert_report), "sha256": sha256(args.en_ja_expert_report)},
            "jaENExpertReport": {"path": str(args.ja_en_expert_report), "sha256": sha256(args.ja_en_expert_report)},
            "enJARouter": {"path": str(args.en_ja_router), "sha256": sha256(args.en_ja_router)},
            "jaENRouter": {"path": str(args.ja_en_router), "sha256": sha256(args.ja_en_router)},
        },
        "results": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "cases": len(selected),
                "routeCounts": route_counts,
                "criticalTokenFallbacks": critical_token_fallbacks,
                "unrecoverableCriticalTokenCases": len(
                    unrecoverable_critical_token_cases
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
