#!/usr/bin/env python3
"""Freeze a source-only canonical teacher corpus for one translation direction."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from tokenizers import Tokenizer

from build_canonical_target_pilot import (
    DEVELOPER_PROMPT,
    DIRECTIONS,
    MAXIMUM_SOURCE_TOKENS,
    MODEL,
    QUOTAS,
    exclusive_json,
    exclusive_jsonl,
    request_for,
    rows,
    sha256,
    sha256_text,
    validate_seed,
)


DIRECTION_LANGUAGES = {
    "en-ja": ("en-US", "ja-JP"),
    "ja-en": ("ja-JP", "en-US"),
}


def rank(row: dict, selection_seed: str, direction: str) -> bytes:
    return hashlib.sha256(
        f"{selection_seed}\0{direction}\0{row['domain']}\0{row['id']}".encode(
            "utf-8"
        )
    ).digest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_seeds", type=Path)
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("incumbent_model", type=Path)
    parser.add_argument("pilot_seeds", type=Path)
    parser.add_argument("teacher_requests", type=Path)
    parser.add_argument("incumbent_suite", type=Path)
    parser.add_argument("pilot_contract", type=Path)
    parser.add_argument("--direction", choices=tuple(DIRECTION_LANGUAGES), required=True)
    parser.add_argument("--exclude-seeds", type=Path, action="append", default=[])
    parser.add_argument("--selection-seed", required=True)
    parser.add_argument("--quota-multiplier", type=int, default=20)
    parser.add_argument("--minimum-approved", type=int, default=120)
    args = parser.parse_args()
    if (
        not args.selection_seed.strip()
        or args.quota_multiplier < 1
        or args.minimum_approved < 1
    ):
        raise SystemExit("selection seed, quota multiplier, and approval gate must be positive")

    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    if (
        source_manifest.get("output", {}).get("sha256") != sha256(args.source_seeds)
        or source_manifest.get("counts", {}).get("directions", {}).get(
            args.direction
        )
        != 8000
        or source_manifest.get("private_reasoning_traces_used") is not False
        or not source_manifest.get("decontamination", {}).get("protected_suites")
    ):
        raise SystemExit("source seed manifest is not the authenticated decontaminated pool")

    tokenizer_path = args.incumbent_model / "tokenizer.json"
    model_manifest_path = args.incumbent_model / "manifest.json"
    if not tokenizer_path.is_file() or not model_manifest_path.is_file():
        raise SystemExit("incumbent model lacks tokenizer or manifest")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    if model_manifest.get("direction") != args.direction:
        raise SystemExit("incumbent model direction differs")

    excluded_ids: set[str] = set()
    exclusion_records: list[dict] = []
    for path in args.exclude_seeds:
        excluded = rows(path)
        identifiers = {str(row.get("id", "")).strip() for row in excluded}
        if not identifiers or "" in identifiers or len(identifiers) != len(excluded):
            raise SystemExit(f"invalid exclusion seed file: {path}")
        excluded_ids.update(identifiers)
        exclusion_records.append(
            {"path": str(path), "sha256": sha256(path), "source_ids": len(identifiers)}
        )

    expected_languages = DIRECTION_LANGUAGES[args.direction]
    scaled_quotas = {
        domain: count * args.quota_multiplier for domain, count in QUOTAS.items()
    }
    grouped: dict[str, list[dict]] = defaultdict(list)
    seen_ids: set[str] = set()
    for row in rows(args.source_seeds):
        if (
            row.get("source_language"),
            row.get("target_language"),
        ) != expected_languages or row.get("domain") not in scaled_quotas:
            continue
        validate_seed(row)
        identifier = str(row["id"])
        if identifier in excluded_ids:
            continue
        if identifier in seen_ids:
            raise SystemExit(f"duplicate source ID: {identifier}")
        seen_ids.add(identifier)
        source_tokens = len(tokenizer.encode(str(row["source"])).ids)
        if source_tokens > MAXIMUM_SOURCE_TOKENS:
            continue
        grouped[str(row["domain"])].append(
            {**row, "canonical_pilot_source_tokens": source_tokens}
        )

    selected: list[dict] = []
    for domain, quota in scaled_quotas.items():
        available = sorted(
            grouped[domain],
            key=lambda row: rank(row, args.selection_seed, args.direction),
        )
        if len(available) < quota:
            raise SystemExit(
                f"insufficient {args.direction}/{domain} rows after exclusions: "
                f"{len(available)} < {quota}"
            )
        selected.extend(available[:quota])
    selected.sort(
        key=lambda row: (
            row["domain"],
            rank(row, args.selection_seed, args.direction),
        )
    )
    expected_count = sum(scaled_quotas.values())
    if len(selected) != expected_count or {
        (row["source_language"], row["target_language"]) for row in selected
    } != {expected_languages}:
        raise SystemExit("directional canonical selection is invalid")

    prompt_sha = sha256_text(DEVELOPER_PROMPT)
    requests = [request_for(row, prompt_sha) for row in selected]
    suite = [
        {
            "id": row["id"],
            "sourceLanguage": row["source_language"],
            "targetLanguage": row["target_language"],
            "domain": row["domain"],
            "source": row["source"],
            "references": [row["reference_translation"]],
            "claimEligible": False,
        }
        for row in selected
    ]
    exclusive_jsonl(args.pilot_seeds, selected)
    exclusive_jsonl(args.teacher_requests, requests)
    exclusive_jsonl(args.incumbent_suite, suite)
    contract = {
        "schema_version": 1,
        "experiment": f"canonical-target-{args.direction}-scale-v7",
        "status": "source-only-teacher-requests-frozen",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "direction": args.direction,
        "selection_seed": args.selection_seed,
        "selection_uses_previous_teacher_or_judge_outputs": False,
        "excluded_prior_source_ids": len(excluded_ids),
        "exclusion_seeds": exclusion_records,
        "source_only_teacher": True,
        "teacher_model": MODEL,
        "teacher_authentication": (
            "Codex cached ChatGPT authentication; no OpenAI API key"
        ),
        "teacher_reasoning_effort": "none",
        "private_reasoning_traces_used": False,
        "teacher_prompt_sha256": prompt_sha,
        "maximum_source_tokens": MAXIMUM_SOURCE_TOKENS,
        "quota_multiplier": args.quota_multiplier,
        "quotas": scaled_quotas,
        "counts": {
            "total": len(selected),
            "domains": dict(
                sorted(Counter(str(row["domain"]) for row in selected).items())
            ),
            "licenses": dict(
                sorted(Counter(str(row["license"]) for row in selected).items())
            ),
        },
        "admission": {
            "candidate_design": [
                "one fresh canonical Codex teacher translation",
                "one licensed human reference",
                "one authenticated current Mimi translation",
            ],
            "independent_judges_required": 2,
            "human_reviewers_required": False,
            "absolute_teacher_thresholds_required": True,
            "unanimous_pareto_preference_over_current_required_for_preference_training": True,
            "minimum_approved": args.minimum_approved,
            "training_authorized_by_teacher_output_alone": False,
        },
        "decontamination": source_manifest["decontamination"],
        "inputs": {
            "source_seeds": {
                "path": str(args.source_seeds),
                "sha256": sha256(args.source_seeds),
            },
            "source_manifest": {
                "path": str(args.source_manifest),
                "sha256": sha256(args.source_manifest),
            },
            "incumbent_model": {
                "path": str(args.incumbent_model),
                "manifest_sha256": sha256(model_manifest_path),
                "tokenizer_sha256": sha256(tokenizer_path),
            },
        },
        "outputs": {
            "pilot_seeds": {
                "path": str(args.pilot_seeds),
                "sha256": sha256(args.pilot_seeds),
            },
            "teacher_requests": {
                "path": str(args.teacher_requests),
                "sha256": sha256(args.teacher_requests),
            },
            "incumbent_suite": {
                "path": str(args.incumbent_suite),
                "sha256": sha256(args.incumbent_suite),
            },
        },
        "promotion_authorized": False,
        "app_change_authorized": False,
        "public_upload_authorized": False,
    }
    exclusive_json(args.pilot_contract, contract)
    print(
        json.dumps(
            {
                "selected": len(selected),
                "domains": contract["counts"]["domains"],
                "pilot_seeds_sha256": sha256(args.pilot_seeds),
                "teacher_requests_sha256": sha256(args.teacher_requests),
                "incumbent_suite_sha256": sha256(args.incumbent_suite),
                "contract": str(args.pilot_contract),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
