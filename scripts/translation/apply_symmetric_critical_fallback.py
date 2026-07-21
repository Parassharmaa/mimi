#!/usr/bin/env python3
"""Evaluate a symmetric safety fallback across an authenticated Marian MoE.

The shipping-shaped router already falls back from an unsafe expert output to a
safe generalist output. This ablation also tries the expert when a router-chosen
generalist fails the exact critical-token policy. It consumes saved model
reports, so quality and safety can be rejected before any Swift runtime change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from source_expert_router import SourceExpertRouter  # noqa: E402
from typed_critical_token_policy import single_percentage_preserves  # noqa: E402


STRICT_CRITICAL_TOKEN_RE = re.compile(
    r"https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+"
    r"|\{[^{}]+\}|%[A-Za-z]|<[A-Za-z][^<>]*>|%"
    r"|(?<![\d.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)*(?!\d|\.\d)"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise SystemExit(f"translation report is invalid: {path}")
    return payload


def indexed(report: dict, path: Path) -> dict[str, dict]:
    output = {str(row.get("caseID", "")): row for row in report["results"]}
    if "" in output or len(output) != len(report["results"]):
        raise SystemExit(f"translation report has empty or duplicate case IDs: {path}")
    return output


def strict_tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", value)
    return sorted(
        token.replace(",", "")
        for token in STRICT_CRITICAL_TOKEN_RE.findall(normalized)
    )


def preserves(source: str, output: str) -> bool:
    return strict_tokens(source) == strict_tokens(output) or single_percentage_preserves(
        source, output
    )


def summed_latency(first: dict, second: dict | None, key: str):
    first_values = first.get(key)
    if key == "latencySeconds":
        return float(first_values) + (float(second[key]) if second is not None else 0.0)
    values = list(first_values or [first["latencySeconds"]])
    if second is None:
        return values
    second_values = list(second.get(key) or [second["latencySeconds"]])
    if len(values) != len(second_values):
        raise SystemExit("fallback reports use different warm repetition counts")
    return [left + right for left, right in zip(values, second_values, strict=True)]


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

    generalist_report = load(args.generalist_report)
    generalist = indexed(generalist_report, args.generalist_report)
    experts = {
        "en-ja": indexed(load(args.en_ja_expert_report), args.en_ja_expert_report),
        "ja-en": indexed(load(args.ja_en_expert_report), args.ja_en_expert_report),
    }
    if not (set(generalist) == set(experts["en-ja"]) == set(experts["ja-en"])):
        raise SystemExit("generalist and expert reports do not cover identical cases")
    routers = {
        "en-ja": SourceExpertRouter.load(args.en_ja_router),
        "ja-en": SourceExpertRouter.load(args.ja_en_router),
    }
    manifest_path = args.moe_pack / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit("Marian MoE pack lacks its root manifest")

    selected = []
    counts: Counter[str] = Counter()
    recovered = []
    remaining_failures = []
    direction_for_source = {"en-US": "en-ja", "ja-JP": "ja-en"}
    for base_row in generalist_report["results"]:
        case_id = str(base_row["caseID"])
        direction = direction_for_source[str(base_row["sourceLanguage"])]
        source = str(base_row["source"])
        expert_row = experts[direction][case_id]
        use_expert = routers[direction].routes_to_expert(source)
        first = expert_row if use_expert else base_row
        alternate = base_row if use_expert else expert_row
        first_safe = preserves(source, str(first["hypothesis"]))
        alternate_safe = preserves(source, str(alternate["hypothesis"]))
        fallback = not first_safe and alternate_safe
        chosen = dict(alternate if fallback else first)
        if use_expert:
            initial_engine = "expert"
            selected_engine = "generalist-critical-token-fallback" if fallback else "expert"
        else:
            initial_engine = "generalist"
            selected_engine = "expert-critical-token-fallback" if fallback else "generalist"
        selected_engine = selected_engine if first_safe or fallback else f"{initial_engine}-unsafe"
        chosen["selectedEngine"] = selected_engine
        chosen["initialEngine"] = initial_engine
        chosen["routerScore"] = routers[direction].score(source)
        chosen["criticalTokenGuardPasses"] = first_safe or alternate_safe
        chosen["symmetricCriticalFallbackUsed"] = fallback
        chosen["latencySeconds"] = summed_latency(
            first, alternate if fallback else None, "latencySeconds"
        )
        chosen["warmLatencySeconds"] = summed_latency(
            first, alternate if fallback else None, "warmLatencySeconds"
        )
        counts[f"{direction}:{selected_engine}"] += 1
        if fallback:
            recovered.append(case_id)
        elif not first_safe:
            remaining_failures.append(case_id)
        selected.append(chosen)

    pack_bytes = sum(path.stat().st_size for path in args.moe_pack.rglob("*") if path.is_file())
    output = {
        **{key: value for key, value in generalist_report.items() if key != "results"},
        "engine": "mlx:ElanMT-BT:4bit-g64-source-routed-moe-symmetric-critical-fallback-simulation",
        "modelRevision": f"moe-manifest-sha256:{sha256(manifest_path)}",
        "modelBytes": pack_bytes,
        "physicalModelCount": 4,
        "ablation": {
            "policy": (
                "route once; if selected output fails exact critical-token policy, "
                "try the opposite bundled role and accept it only if it passes"
            ),
            "selectionCounts": dict(sorted(counts.items())),
            "recoveredCases": recovered,
            "remainingUnsafeCases": remaining_failures,
            "latencyAccounting": "sum authenticated first and fallback model latencies",
        },
        "inputs": {
            "generalistReport": {
                "path": str(args.generalist_report),
                "sha256": sha256(args.generalist_report),
            },
            "enJAExpertReport": {
                "path": str(args.en_ja_expert_report),
                "sha256": sha256(args.en_ja_expert_report),
            },
            "jaENExpertReport": {
                "path": str(args.ja_en_expert_report),
                "sha256": sha256(args.ja_en_expert_report),
            },
            "enJARouter": {"path": str(args.en_ja_router), "sha256": sha256(args.en_ja_router)},
            "jaENRouter": {"path": str(args.ja_en_router), "sha256": sha256(args.ja_en_router)},
        },
        "claimEligible": False,
        "doesNotAuthorizeAppIntegration": True,
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
                "recovered": len(recovered),
                "remainingUnsafe": len(remaining_failures),
                "selectionCounts": dict(sorted(counts.items())),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
