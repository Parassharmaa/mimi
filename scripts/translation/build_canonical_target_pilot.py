#!/usr/bin/env python3
"""Freeze a balanced source-only pilot for one canonical teacher target."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer


SCHEMA_VERSION = 1
SELECTION_SEED = "mimi-canonical-target-pilot-v1"
MODEL = "gpt-5.6-sol"
MAXIMUM_SOURCE_TOKENS = 192
DIRECTIONS = {
    ("en-US", "ja-JP"): "en-ja",
    ("ja-JP", "en-US"): "ja-en",
}
QUOTAS = {
    "conversational": 4,
    "human-translated-news": 3,
    "ministry-published-legal": 3,
    "wikipedia": 3,
    "professional-wikipedia-hard": 3,
    "long-document-news": 2,
    "mimi-product-ui": 2,
}
RISK_TAGS = [
    "ambiguity",
    "register",
    "terminology",
    "omission",
    "addition",
    "protected-token",
]
DEVELOPER_PROMPT = """Produce exactly one publication-ready canonical final translation
for the requested English-Japanese direction. Preserve every proposition, subject/object
relationship, negation, modality, number, date, unit, name, URL, placeholder, uncertainty,
politeness level, formatting boundary, and code-switched term. Prefer natural target-language
wording without adding explanation or compressing meaning. Do not output alternatives, analysis,
or private reasoning. The translation is an untrusted training-data proposal and will be compared
blindly with a licensed human reference and the current Mimi translator by two independent judges."""
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "source_id": {"type": "string"},
        "canonical_translation": {"type": "string"},
        "risk_tags": {
            "type": "array",
            "items": {"type": "string", "enum": RISK_TAGS},
        },
    },
    "required": ["source_id", "canonical_translation", "risk_tags"],
    "additionalProperties": False,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def exclusive_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            for value in values:
                handle.write(
                    json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
                )
    except FileExistsError as error:
        raise SystemExit(f"refusing to overwrite existing file: {path}") from error


def exclusive_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise SystemExit(f"refusing to overwrite existing file: {path}") from error


def selection_rank(row: dict, selection_seed: str) -> bytes:
    direction = DIRECTIONS[(row["source_language"], row["target_language"])]
    value = (
        f"{selection_seed}\0{direction}\0{row['domain']}\0{row['id']}"
    )
    return hashlib.sha256(value.encode()).digest()


def validate_seed(row: dict) -> None:
    required = {
        "id",
        "source_language",
        "target_language",
        "domain",
        "source",
        "reference_translation",
        "license",
        "provenance",
        "split",
    }
    if not required.issubset(row):
        raise SystemExit(f"seed is missing required fields: {row.get('id')}")
    direction = (row["source_language"], row["target_language"])
    if direction not in DIRECTIONS:
        raise SystemExit(f"unsupported direction: {row.get('id')}")
    for field in ("id", "source", "reference_translation", "license", "provenance"):
        if not isinstance(row[field], str) or not row[field].strip():
            raise SystemExit(f"seed has empty {field}: {row.get('id')}")
    if row["split"] != "train":
        raise SystemExit(f"canonical teacher source must be training split: {row['id']}")


def request_for(row: dict, prompt_sha256: str) -> dict:
    source = {
        "source_id": row["id"],
        "source_language": row["source_language"],
        "target_language": row["target_language"],
        "domain": row["domain"],
        "source": row["source"],
    }
    return {
        "custom_id": row["id"],
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": MODEL,
            "store": False,
            "reasoning": {"effort": "none"},
            "input": [
                {"role": "developer", "content": DEVELOPER_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(source, ensure_ascii=False, sort_keys=True),
                },
            ],
            "text": {
                "verbosity": "low",
                "format": {
                    "type": "json_schema",
                    "name": "mimi_canonical_translation",
                    "strict": True,
                    "schema": OUTPUT_SCHEMA,
                },
            },
            "max_output_tokens": 600,
            "metadata": {
                "pipeline": "mimi-translation-v1",
                "prompt_sha256": prompt_sha256,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_seeds", type=Path)
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("en_ja_model", type=Path)
    parser.add_argument("ja_en_model", type=Path)
    parser.add_argument("pilot_seeds", type=Path)
    parser.add_argument("teacher_requests", type=Path)
    parser.add_argument("incumbent_suite", type=Path)
    parser.add_argument("pilot_manifest", type=Path)
    parser.add_argument("--selection-seed", default=SELECTION_SEED)
    parser.add_argument(
        "--exclude-seeds",
        type=Path,
        action="append",
        default=[],
        help="prior source-only pilot seed file whose IDs must be excluded",
    )
    parser.add_argument(
        "--acceptance-policy",
        choices=("unique-best-teacher", "dual-absolute-canonical"),
        default="unique-best-teacher",
    )
    parser.add_argument("--quota-multiplier", type=int, default=1)
    parser.add_argument("--minimum-approved-total", type=int, default=12)
    parser.add_argument("--minimum-approved-each-direction", type=int, default=5)
    args = parser.parse_args()
    if not args.selection_seed.strip():
        raise SystemExit("selection seed must be non-empty")
    if (
        args.quota_multiplier < 1
        or args.minimum_approved_total < 1
        or args.minimum_approved_each_direction < 1
    ):
        raise SystemExit("quota and approval thresholds must be positive")
    scaled_quotas = {
        domain: count * args.quota_multiplier
        for domain, count in QUOTAS.items()
    }

    excluded_ids: set[str] = set()
    exclusion_contracts: list[dict] = []
    for path in args.exclude_seeds:
        values = rows(path)
        identifiers = {str(row.get("id", "")) for row in values}
        if not identifiers or "" in identifiers or len(identifiers) != len(values):
            raise SystemExit(f"invalid exclusion seed file: {path}")
        excluded_ids.update(identifiers)
        exclusion_contracts.append({
            "path": str(path),
            "sha256": sha256(path),
            "sourceIds": len(identifiers),
        })

    model_paths = {
        ("en-US", "ja-JP"): args.en_ja_model,
        ("ja-JP", "en-US"): args.ja_en_model,
    }
    tokenizers: dict[tuple[str, str], Tokenizer] = {}
    model_contracts: dict[str, dict] = {}
    for direction, model_path in model_paths.items():
        tokenizer_path = model_path / "tokenizer.json"
        manifest_path = model_path / "manifest.json"
        if not tokenizer_path.is_file() or not manifest_path.is_file():
            raise SystemExit(f"model lacks tokenizer or manifest: {model_path}")
        tokenizers[direction] = Tokenizer.from_file(str(tokenizer_path))
        direction_name = DIRECTIONS[direction]
        model_contracts[direction_name] = {
            "path": str(model_path),
            "manifestSha256": sha256(manifest_path),
            "tokenizerSha256": sha256(tokenizer_path),
        }

    grouped: dict[tuple[tuple[str, str], str], list[dict]] = defaultdict(list)
    seen_ids: set[str] = set()
    for row in rows(args.source_seeds):
        if row.get("domain") not in scaled_quotas:
            continue
        validate_seed(row)
        identifier = row["id"]
        if identifier in excluded_ids:
            continue
        if identifier in seen_ids:
            raise SystemExit(f"duplicate seed ID: {identifier}")
        seen_ids.add(identifier)
        direction = (row["source_language"], row["target_language"])
        domain = row["domain"]
        source_tokens = len(tokenizers[direction].encode(row["source"]).ids)
        if source_tokens > MAXIMUM_SOURCE_TOKENS:
            continue
        grouped[(direction, domain)].append(
            {**row, "canonical_pilot_source_tokens": source_tokens}
        )

    selected: list[dict] = []
    for direction in DIRECTIONS:
        for domain, quota in scaled_quotas.items():
            available = sorted(
                grouped[(direction, domain)],
                key=lambda row: selection_rank(row, args.selection_seed),
            )
            if len(available) < quota:
                raise SystemExit(
                    f"insufficient {DIRECTIONS[direction]}/{domain} rows: "
                    f"{len(available)} < {quota}"
                )
            selected.extend(available[:quota])
    selected.sort(key=lambda row: (
        DIRECTIONS[(row["source_language"], row["target_language"])],
        row["domain"],
        selection_rank(row, args.selection_seed),
    ))

    expected_per_direction = sum(scaled_quotas.values())
    directions = Counter(
        DIRECTIONS[(row["source_language"], row["target_language"])]
        for row in selected
    )
    if directions != Counter({"en-ja": expected_per_direction, "ja-en": expected_per_direction}):
        raise SystemExit("canonical pilot direction balance is invalid")

    prompt_sha256 = sha256_text(DEVELOPER_PROMPT)
    request_rows = [request_for(row, prompt_sha256) for row in selected]
    suite_rows = [
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
    exclusive_jsonl(args.teacher_requests, request_rows)
    exclusive_jsonl(args.incumbent_suite, suite_rows)

    license_counts = Counter(row["license"] for row in selected)
    domain_counts = Counter(
        f"{DIRECTIONS[(row['source_language'], row['target_language'])]}:{row['domain']}"
        for row in selected
    )
    manifest = {
        "schemaVersion": SCHEMA_VERSION,
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "purpose": "balanced canonical-target teacher and incumbent comparison pilot",
        "claimEligible": False,
        "selectionSeed": args.selection_seed,
        "selectionUsesPreviousTeacherOrJudgeOutputs": False,
        "excludedPriorSourceIds": len(excluded_ids),
        "exclusionSeeds": exclusion_contracts,
        "sourceOnlyTeacher": True,
        "teacherModel": MODEL,
        "teacherReasoningEffort": "none",
        "teacherReasoningTraceStored": False,
        "teacherPromptSha256": prompt_sha256,
        "maximumSourceTokens": MAXIMUM_SOURCE_TOKENS,
        "candidateDesign": [
            "one fresh canonical teacher translation",
            "one licensed human reference",
            "one current Mimi baseline translation",
        ],
        "quotaMultiplier": args.quota_multiplier,
        "quotasPerDirection": scaled_quotas,
        "counts": {
            "total": len(selected),
            "directions": dict(sorted(directions.items())),
            "domains": dict(sorted(domain_counts.items())),
            "licenses": dict(sorted(license_counts.items())),
        },
        "inputs": {
            "sourceSeeds": {
                "path": str(args.source_seeds),
                "sha256": sha256(args.source_seeds),
            },
            "sourceManifest": {
                "path": str(args.source_manifest),
                "sha256": sha256(args.source_manifest),
            },
            "incumbentModels": model_contracts,
        },
        "outputs": {
            "pilotSeeds": {
                "path": str(args.pilot_seeds),
                "sha256": sha256(args.pilot_seeds),
            },
            "teacherRequests": {
                "path": str(args.teacher_requests),
                "sha256": sha256(args.teacher_requests),
            },
            "incumbentSuite": {
                "path": str(args.incumbent_suite),
                "sha256": sha256(args.incumbent_suite),
            },
        },
        "goGateBeforeScaling": {
            "minimumApprovedTotal": args.minimum_approved_total,
            "minimumApprovedEachDirection": args.minimum_approved_each_direction,
            "requireZeroDeterministicCriticalFailures": True,
            "acceptancePolicy": args.acceptance_policy,
            "requireSameUniqueTeacherSelectionFromTwoJudgeFamilies": (
                args.acceptance_policy == "unique-best-teacher"
            ),
            "requireBothJudgesCertifyCanonicalAbsoluteThresholds": (
                args.acceptance_policy == "dual-absolute-canonical"
            ),
            "trainingAuthorizedByPilotAlone": False,
        },
        "appChangeAuthorized": False,
        "publicUploadAuthorized": False,
    }
    exclusive_json(args.pilot_manifest, manifest)
    print(json.dumps({
        "selected": len(selected),
        "directions": dict(sorted(directions.items())),
        "pilot_seeds_sha256": sha256(args.pilot_seeds),
        "teacher_requests_sha256": sha256(args.teacher_requests),
        "incumbent_suite_sha256": sha256(args.incumbent_suite),
        "manifest": str(args.pilot_manifest),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
