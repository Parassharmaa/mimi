#!/usr/bin/env python3
"""Simulate a source-only guarded EN→JA canonical expert over current Mimi."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from audit_translation_structures import critical_tokens, tokens
from typed_critical_token_policy import typed_preserves


IMMUTABLE_FIELDS = (
    "caseID",
    "sourceLanguage",
    "targetLanguage",
    "domain",
    "source",
    "references",
    "claimEligible",
)


def sha256(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"missing router input: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid JSON input {path}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def indexed(report: dict, path: Path) -> dict[str, dict]:
    rows = report.get("results")
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"report has no results: {path}")
    output = {str(row.get("caseID", "")): row for row in rows}
    if len(output) != len(rows):
        raise SystemExit(f"report has duplicate or missing case IDs: {path}")
    return output


def repeated_token_loop(ids: list[object]) -> bool:
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in ids):
        raise SystemExit("output token IDs must be integers")
    for width in range(3, min(16, len(ids) // 3) + 1):
        for start in range(0, len(ids) - width * 3 + 1):
            phrase = ids[start : start + width]
            if (
                ids[start + width : start + width * 2] == phrase
                and ids[start + width * 2 : start + width * 3] == phrase
            ):
                return True
    return False


def source_risky(row: dict) -> bool:
    source = str(row["source"])
    return bool(critical_tokens(source) or tokens(source)["negative"])


def output_safe(row: dict, maximum_tokens: int) -> bool:
    source = str(row["source"])
    hypothesis = str(row["hypothesis"])
    source_language = str(row["sourceLanguage"])
    target_language = str(row["targetLanguage"])
    output_ids = row.get("outputTokenIDs", [])
    segment_ids = row.get("segmentOutputTokenIDs")
    sequences = segment_ids if isinstance(segment_ids, list) else [output_ids]
    if (
        critical_tokens(source) != critical_tokens(hypothesis)
        or tokens(source)["negative"] != tokens(hypothesis)["negative"]
        or not typed_preserves(
            source, hypothesis, source_language, target_language
        )
    ):
        return False
    for sequence in sequences:
        if (
            not isinstance(sequence, list)
            or len(sequence) >= maximum_tokens
            or repeated_token_loop(sequence)
        ):
            return False
    return not (
        row.get("failureReason")
        or row.get("runtimeAccepted") is False
        or int(row.get("emptySegmentCount", 0)) > 0
    )


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_report", type=Path)
    parser.add_argument("expert_report", type=Path)
    parser.add_argument("baseline_en_ja_model", type=Path)
    parser.add_argument("baseline_ja_en_model", type=Path)
    parser.add_argument("expert_en_ja_model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists() and args.output.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")

    baseline = load(args.baseline_report)
    expert = load(args.expert_report)
    baseline_rows = indexed(baseline, args.baseline_report)
    expert_rows = indexed(expert, args.expert_report)
    if set(baseline_rows) != set(expert_rows):
        raise SystemExit("baseline and expert reports cover different case IDs")
    maximum_tokens = int(
        baseline.get("benchmarkConfiguration", {}).get(
            "maximumGeneratedTokens", 0
        )
    )
    if (
        maximum_tokens < 1
        or expert.get("benchmarkConfiguration", {}).get(
            "maximumGeneratedTokens"
        )
        != maximum_tokens
    ):
        raise SystemExit("baseline and expert maximum-token contracts differ")

    route_counts = {
        "baseline-ja-en": 0,
        "baseline-en-ja-preflight": 0,
        "expert-en-ja": 0,
        "baseline-en-ja-output-fallback": 0,
        "baseline-en-ja-both-unsafe": 0,
    }
    selected = []
    for baseline_row in baseline["results"]:
        case_id = str(baseline_row["caseID"])
        expert_row = expert_rows[case_id]
        for field in IMMUTABLE_FIELDS:
            if baseline_row.get(field) != expert_row.get(field):
                raise SystemExit(f"reports disagree on {field}: {case_id}")
        if baseline_row["sourceLanguage"] == "ja-JP":
            chosen = dict(baseline_row)
            route = "baseline-ja-en"
        elif source_risky(baseline_row):
            chosen = dict(baseline_row)
            route = "baseline-en-ja-preflight"
        elif output_safe(expert_row, maximum_tokens):
            chosen = dict(expert_row)
            route = "expert-en-ja"
        elif output_safe(baseline_row, maximum_tokens):
            chosen = dict(baseline_row)
            route = "baseline-en-ja-output-fallback"
        else:
            chosen = dict(baseline_row)
            route = "baseline-en-ja-both-unsafe"
        chosen["selectedEngine"] = route
        route_counts[route] += 1
        selected.append(chosen)

    model_paths = (
        args.baseline_en_ja_model,
        args.baseline_ja_en_model,
        args.expert_en_ja_model,
    )
    manifests = []
    for path in model_paths:
        manifest_path = path / "manifest.json"
        manifests.append(
            {
                "path": str(manifest_path),
                "sha256": sha256(manifest_path),
            }
        )
    policy = {
        "status": "posthoc-exploratory-not-claim-eligible",
        "jaEN": "always current Mimi",
        "enJAPreflight": (
            "current Mimi when source has an exact critical token or deterministic "
            "negation marker; otherwise attempt canonical expert"
        ),
        "enJAOutputFallback": (
            "fall back to current Mimi when expert violates exact/typed critical "
            "tokens, negation parity, generation limit, repetition, empty-output, "
            "or runtime guards"
        ),
        "bothUnsafe": "retain current Mimi behavior; never introduce a new route",
        "usesSourceOnlyBeforeGeneration": True,
        "usesReferenceAtRuntime": False,
        "usesMetricAtRuntime": False,
    }
    policy_sha256 = hashlib.sha256(
        json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    report = {
        **{
            key: value
            for key, value in baseline.items()
            if key
            not in (
                "results",
                "engine",
                "modelRevision",
                "modelBytes",
                "physicalModelCount",
                "peakResidentBytes",
            )
        },
        "engine": "mlx:Mimi-Marian-canonical-en-ja-safety-router-simulation",
        "modelRevision": f"routing-policy-sha256:{policy_sha256}",
        "modelBytes": sum(directory_bytes(path) for path in model_paths),
        "physicalModelCount": 3,
        "peakResidentBytes": None,
        "routeCounts": route_counts,
        "routingPolicy": policy,
        "modelManifests": manifests,
        "simulationLimitations": {
            "latencyAndRSSClaimEligible": False,
            "reason": (
                "selected-output replay does not include attempted-expert plus fallback "
                "latency or simultaneous residency; measure in Swift/MLX before use"
            ),
        },
        "claimEligible": False,
        "doesNotAuthorizeAppIntegration": True,
        "inputs": {
            "baselineReport": {
                "path": str(args.baseline_report),
                "sha256": sha256(args.baseline_report),
            },
            "expertReport": {
                "path": str(args.expert_report),
                "sha256": sha256(args.expert_report),
            },
        },
        "results": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cases": len(selected),
                "routeCounts": route_counts,
                "modelBytes": report["modelBytes"],
                "policySHA256": policy_sha256,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
