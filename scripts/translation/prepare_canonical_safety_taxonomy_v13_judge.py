#!/usr/bin/env python3
"""Freeze blinded Sonnet 5 + Opus 5 taxonomy requests for v12 failures."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPERIMENT = "canonical-safety-taxonomy-v13-ja-en"
SOURCE_EXPERIMENT = "canonical-safety-repair-v12-ja-en"
MODELS = ("claude-sonnet-5", "claude-opus-5")
PIPELINE = "mimi-translation-judge-v1"
RESULT_SHA256 = "0db46aee124cb3d9e61898630349e61cfa2a1fcecc5d04f591d3f5c5a2a9dcd1"
DIAGNOSTIC_SHA256 = "4c3a651f5a14ade02da41c64a077401d9627c0645cc8a33a8b9c1e329ce83c10"
ERROR_TAGS = [
    "polarity",
    "number-or-quantity",
    "legal-citation",
    "omission",
    "addition",
    "repetition-nontermination",
    "terminology",
    "named-entity",
    "untranslated-source",
    "other-critical",
]
SCHEMA = {
    "type": "object",
    "properties": {
        "source_id": {"type": "string"},
        "assessments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string"},
                    "adequacy": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 4,
                    },
                    "fluent_and_complete": {"type": "boolean"},
                    "critical_semantic_error": {"type": "boolean"},
                    "representation_only_difference": {"type": "boolean"},
                    "error_tags": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ERROR_TAGS,
                        },
                    },
                    "brief_evidence": {"type": "string"},
                },
                "required": [
                    "candidate_id",
                    "adequacy",
                    "fluent_and_complete",
                    "critical_semantic_error",
                    "representation_only_difference",
                    "error_tags",
                    "brief_evidence",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["source_id", "assessments"],
    "additionalProperties": False,
}
DEVELOPER_PROMPT = """Assess each anonymous English translation against the Japanese source.
Judge each candidate independently; candidate identity and origin are intentionally hidden.
Adequacy is 0 to 4. A critical semantic error changes polarity, amount, number, date,
legal article/item/paragraph identity, named entity, obligation, permission, or scope;
omits an essential clause; leaves essential Japanese untranslated; or repeats/does not terminate.
Treat Roman, Arabic, and spelled-out legal numbering as representation-only only when they
preserve the same legal identity and hierarchy. The abbreviation "No." means "number",
not negation. Japanese upper-bound language may correctly become "less than", "not more
than", or an equivalent phrase when the mathematical boundary is preserved.
Set representation_only_difference true only when the candidate is semantically sound and
differs merely in an equivalent surface form. Use error_tags only for observable errors.
brief_evidence must be a short error fragment or "none"; do not provide reasoning steps.
Return compact structured judgments only and never reveal chain-of-thought."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def display_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": display_path(path, root),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def opaque_candidate_id(source_id: str, role: str) -> str:
    digest = hashlib.sha256(
        f"mimi-v13-taxonomy\0{source_id}\0{role}".encode()
    ).hexdigest()[:16]
    return f"candidate-{digest}"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def request_row(
    source: dict[str, Any],
    *,
    model: str,
    prompt_sha256: str,
) -> dict[str, Any]:
    return {
        "custom_id": source["source_id"],
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "store": False,
            "reasoning": {"effort": "low"},
            "input": [
                {"role": "developer", "content": DEVELOPER_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(source, ensure_ascii=False),
                },
            ],
            "text": {
                "verbosity": "low",
                "format": {
                    "type": "json_schema",
                    "name": "mimi_safety_taxonomy_v13",
                    "strict": True,
                    "schema": SCHEMA,
                },
            },
            "max_output_tokens": 1_200,
            "metadata": {
                "pipeline": PIPELINE,
                "prompt_sha256": prompt_sha256,
            },
        },
    }


def normalized_requests(rows: list[dict[str, Any]]) -> str:
    normalized = []
    for row in rows:
        body = {**row["body"], "model": "<frozen-judge-model>"}
        normalized.append({**row, "body": body})
    return hashlib.sha256(canonical(normalized)).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("v12_result", type=Path)
    parser.add_argument("v12_diagnostic", type=Path)
    parser.add_argument("work_directory", type=Path)
    parser.add_argument("contract_output", type=Path)
    args = parser.parse_args()
    if args.work_directory.exists() and any(args.work_directory.iterdir()):
        raise SystemExit(
            f"refusing to overwrite non-empty work directory: {args.work_directory}"
        )
    if args.contract_output.exists():
        raise SystemExit(f"refusing to overwrite contract: {args.contract_output}")
    if (
        sha256(args.v12_result) != RESULT_SHA256
        or sha256(args.v12_diagnostic) != DIAGNOSTIC_SHA256
    ):
        raise SystemExit("v12 result or diagnostic bytes differ")

    root = Path(__file__).resolve().parents[2]
    result = load_json(args.v12_result)
    diagnostic = load_json(args.v12_diagnostic)
    if (
        result.get("experiment") != SOURCE_EXPERIMENT
        or result.get("status") != "internal-gate-rejected"
        or result.get("selected_step") is not None
        or diagnostic.get("experiment") != SOURCE_EXPERIMENT
        or diagnostic.get("status") != "post-result-diagnostic-only"
        or diagnostic.get("selection_changed") is not False
        or diagnostic.get("case_count") != 27
    ):
        raise SystemExit("v12 source evidence safety state differs")

    mapping_rows = []
    queue_rows = []
    sources = []
    for case in diagnostic["cases"]:
        source_id = str(case["id"])
        candidates = {
            "licensed-reference": str(case["reference"]),
            "safe-parent": str(case["baseline"]["hypothesis"]),
            "v12-step-50": str(case["candidates"]["50"]["hypothesis"]),
            "v12-step-100": str(case["candidates"]["100"]["hypothesis"]),
        }
        blinded = []
        for role, translation in candidates.items():
            candidate_id = opaque_candidate_id(source_id, role)
            mapping_rows.append(
                {
                    "source_id": source_id,
                    "candidate_id": candidate_id,
                    "role": role,
                    "translation_sha256": hashlib.sha256(
                        translation.encode()
                    ).hexdigest(),
                    "registered_new_failure_types": (
                        []
                        if role in {"licensed-reference", "safe-parent"}
                        else case["candidates"][
                            "50" if role == "v12-step-50" else "100"
                        ]["registered_new_failure_types"]
                    ),
                }
            )
            queue_rows.append(
                {
                    "source_id": source_id,
                    "candidate_id": candidate_id,
                    "source_language": "ja-JP",
                    "target_language": "en-US",
                    "domain": case["stratum"],
                    "source": case["source"],
                    "translation": translation,
                }
            )
            blinded.append(
                {
                    "candidate_id": candidate_id,
                    "translation": translation,
                }
            )
        sources.append(
            {
                "source_id": source_id,
                "source_language": "ja-JP",
                "target_language": "en-US",
                "domain": case["stratum"],
                "source": case["source"],
                "candidates": sorted(
                    blinded,
                    key=lambda item: item["candidate_id"],
                ),
            }
        )
    sources.sort(key=lambda item: item["source_id"])
    mapping_rows.sort(key=lambda item: (item["source_id"], item["candidate_id"]))
    queue_rows.sort(key=lambda item: (item["source_id"], item["candidate_id"]))
    if len(sources) != 27 or len(mapping_rows) != 108:
        raise SystemExit("v13 taxonomy case or candidate count differs")

    args.work_directory.mkdir(parents=True, exist_ok=True)
    mapping_path = args.work_directory / "candidate-mapping.jsonl"
    queue_path = args.work_directory / "review-queue.jsonl"
    write_jsonl(mapping_path, mapping_rows)
    write_jsonl(queue_path, queue_rows)
    prompt_sha256 = hashlib.sha256(DEVELOPER_PROMPT.encode()).hexdigest()
    request_paths = {}
    request_rows_by_model = {}
    for model in MODELS:
        path = args.work_directory / f"{model}-requests.jsonl"
        rows = [
            request_row(
                source,
                model=model,
                prompt_sha256=prompt_sha256,
            )
            for source in sources
        ]
        write_jsonl(path, rows)
        request_paths[model] = path
        request_rows_by_model[model] = rows
    if normalized_requests(request_rows_by_model[MODELS[0]]) != normalized_requests(
        request_rows_by_model[MODELS[1]]
    ):
        raise SystemExit("Sonnet 5 and Opus 5 blinded payloads differ")

    implementation_paths = [
        Path(__file__).resolve(),
        root / "scripts/translation/collect_canonical_safety_taxonomy_v13_judge.py",
        root / "scripts/translation/run_claude_consensus_judge.py",
        root / "scripts/translation/prioritize_distillation_judgments.py",
    ]
    contract = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": "frozen-before-any-taxonomy-judgment",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "purpose": (
            "future-evaluator taxonomy calibration only; never reinterpret or "
            "rescue the final v12 rejection"
        ),
        "inputs": {
            "v12_result": record(args.v12_result, root),
            "v12_diagnostic": record(args.v12_diagnostic, root),
        },
        "queue": {
            "mapping": record(mapping_path, root),
            "review_queue": record(queue_path, root),
            "sources": len(sources),
            "candidates": len(mapping_rows),
            "candidate_origin_exposed_to_judges": False,
            "licensed_reference_origin_exposed_to_judges": False,
        },
        "judge_policy": {
            "required_exact_models": list(MODELS),
            "same_blinded_payload_required": True,
            "actual_canonical_model_usage_required_per_shard": True,
            "fallback_model_allowed": False,
            "human_reviewers_required": False,
            "reasoning_trace_requested_or_stored": False,
            "brief_observable_evidence_only": True,
            "fail_closed_on_judge_disagreement": True,
        },
        "judge_requests": {
            model: {
                **record(request_paths[model], root),
                "model": model,
                "prompt_sha256": prompt_sha256,
                "model_independent_payload_sha256": normalized_requests(
                    request_rows_by_model[model]
                ),
            }
            for model in MODELS
        },
        "consensus_policy": {
            "critical_semantic_error": (
                "true if either judge marks critical_semantic_error"
            ),
            "related_error_supported": (
                "true if either judge applies the error tag mapped to the "
                "registered deterministic failure type"
            ),
            "representation_only": (
                "true only if both judges mark representation-only, neither "
                "marks critical, both adequacy scores are at least 3, and both "
                "mark fluent_and_complete"
            ),
            "unresolved": (
                "true on any critical, representation, or related-tag disagreement"
            ),
            "applies_to": "future v13+ evaluators only",
            "v12_selection_or_decision_changed": False,
        },
        "implementation": {
            path.name: record(path, root) for path in implementation_paths
        },
        "v12_rejection_final": True,
        "training_authorized": False,
        "protected_evaluation_authorized": False,
        "app_change_authorized": False,
        "public_upload_authorized": False,
    }
    args.contract_output.parent.mkdir(parents=True, exist_ok=True)
    args.contract_output.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "contract": display_path(args.contract_output, root),
                "contract_sha256": sha256(args.contract_output),
                "sources": len(sources),
                "candidates": len(mapping_rows),
                "models": list(MODELS),
                "prompt_sha256": prompt_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
