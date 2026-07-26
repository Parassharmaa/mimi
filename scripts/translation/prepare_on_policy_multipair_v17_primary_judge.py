#!/usr/bin/env python3
"""Freeze V17's blinded primary Sonnet 5 + Opus 5 semantic audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

EXPERIMENT = "faithful-on-policy-multipair-v17-primary-semantic-audit"
SOURCE_EXPERIMENT = "faithful-on-policy-multipair-v17-prediagnostic"
MODELS = ("claude-sonnet-5", "claude-opus-5")
PIPELINE = "mimi-translation-judge-v1"
COMET_MODEL = "Unbabel/wmt22-comet-da"
COMET_REVISION = "371e9839ca4e213dde891b066cf3080f75ec7e72"
COMET_PACKAGE_VERSION = "2.2.7"
ERROR_TAGS = [
    "polarity",
    "number-or-quantity",
    "legal-citation",
    "omission",
    "addition",
    "repetition-nontermination",
    "terminology",
    "named-entity",
    "particle-or-case-role",
    "register-or-honorific",
    "untranslated-source",
    "other-critical",
]
ASSESSMENT = {
    "type": "object",
    "properties": {
        "candidate_id": {"type": "string"},
        "adequacy": {"type": "integer", "minimum": 0, "maximum": 4},
        "fluent_and_complete": {"type": "boolean"},
        "critical_semantic_error": {"type": "boolean"},
        "error_tags": {
            "type": "array",
            "items": {"type": "string", "enum": ERROR_TAGS},
            "uniqueItems": True,
        },
        "brief_evidence": {"type": "string"},
    },
    "required": [
        "candidate_id",
        "adequacy",
        "fluent_and_complete",
        "critical_semantic_error",
        "error_tags",
        "brief_evidence",
    ],
    "additionalProperties": False,
}
SCHEMA = {
    "type": "object",
    "properties": {
        "source_id": {"type": "string"},
        "assessments": {"type": "array", "items": ASSESSMENT},
        "preferred_candidate_ids": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 2,
            "uniqueItems": True,
        },
    },
    "required": ["source_id", "assessments", "preferred_candidate_ids"],
    "additionalProperties": False,
}
DEVELOPER_PROMPT = """Act as an exact bilingual English-Japanese translation auditor.
For each source, assess both anonymous candidates independently, then list the candidate
ID or IDs that are best translations. Candidate identity and origin are hidden.
Prioritize meaning, faithfulness, and completeness before fluency. Do not favor literal
wording when an idiomatic rendering preserves the same meaning. Adequacy is 0 to 4.
A critical semantic error changes polarity, number, quantity, date, legal identity,
named entity, obligation, permission, scope, causal relation, or participant role;
omits or adds an essential proposition; leaves essential source text untranslated; or
repeats/does not terminate. For Japanese, explicitly check particles and case roles,
negation, terminology, and honorific/register meaning. If candidates are semantically
tied, list all tied IDs. Use error_tags only for observable errors. brief_evidence must
be a short source/translation fragment or "none", never reasoning steps. Return only
the structured result and never reveal chain-of-thought."""


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


def rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def opaque_id(kind: str, pair_id: str, role: str = "") -> str:
    digest = hashlib.sha256(
        f"mimi-v17-primary\0{kind}\0{pair_id}\0{role}".encode()
    ).hexdigest()[:20]
    return f"v17-{kind}-{digest}"


def request_row(source: dict[str, Any], model: str, prompt_sha256: str) -> dict:
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
                {"role": "user", "content": json.dumps(source, ensure_ascii=False)},
            ],
            "text": {
                "verbosity": "low",
                "format": {
                    "type": "json_schema",
                    "name": "mimi_v17_primary_semantic_audit",
                    "strict": True,
                    "schema": SCHEMA,
                },
            },
            "max_output_tokens": 700,
            "metadata": {
                "pipeline": PIPELINE,
                "prompt_sha256": prompt_sha256,
            },
        },
    }


def normalized_requests(values: list[dict[str, Any]]) -> str:
    normalized = []
    for value in values:
        body = {**value["body"], "model": "<frozen-judge-model>"}
        normalized.append({**value, "body": body})
    return hashlib.sha256(canonical(normalized)).hexdigest()


def validate_comet(
    report: dict[str, Any],
    *,
    pairs: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, float]:
    expected_ids = {str(pair["pair_id"]) for pair in pairs}
    results = report.get("results")
    if (
        report.get("metric") != "COMET-22"
        or report.get("modelRepository") != COMET_MODEL
        or report.get("modelRevision") != COMET_REVISION
        or report.get("modelLicense") != "Apache-2.0"
        or report.get("packageVersion") != COMET_PACKAGE_VERSION
        or report.get("precision") != "float32"
        or report.get("suiteSHA256")
        != manifest["outputs"]["comet_suite"]["sha256"]
        or report.get("engineReportSHA256")
        != manifest["outputs"]["comet_engine_report"]["sha256"]
        or not isinstance(results, list)
    ):
        raise SystemExit("V17 pinned COMET evidence differs")
    by_id: dict[str, float] = {}
    for result in results:
        pair_id = str(result.get("caseID", ""))
        score = result.get("score")
        if (
            pair_id in by_id
            or pair_id not in expected_ids
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
        ):
            raise SystemExit("V17 COMET result coverage or score differs")
        by_id[pair_id] = float(score)
    if set(by_id) != expected_ids:
        raise SystemExit("V17 COMET does not cover the exact primary judge pool")
    return by_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediagnostic_directory", type=Path)
    parser.add_argument("comet_report", type=Path)
    parser.add_argument("work_directory", type=Path)
    parser.add_argument("contract_output", type=Path)
    args = parser.parse_args()
    if args.work_directory.exists() and any(args.work_directory.iterdir()):
        raise SystemExit(
            f"refusing to overwrite non-empty work directory: {args.work_directory}"
        )
    if args.contract_output.exists():
        raise SystemExit(f"refusing to overwrite contract: {args.contract_output}")

    root = Path(__file__).resolve().parents[2]
    manifest_path = args.prediagnostic_directory / "manifest.json"
    pairs_path = args.prediagnostic_directory / "preselected-pairs.jsonl"
    manifest = load_json(manifest_path)
    pairs = rows(pairs_path)
    if (
        manifest.get("experiment") != SOURCE_EXPERIMENT
        or manifest.get("status") != "presemantic-candidate-availability-passed"
        or manifest.get("training_authorized") is not False
        or manifest.get("semantic_audit_authorized") is not True
        or not all(manifest.get("stage_one_gates", {}).values())
        or manifest.get("outputs", {}).get("preselected_pairs", {}).get("sha256")
        != sha256(pairs_path)
        or manifest.get("counts", {})
        .get("preselected_pairs", {})
        .get("total")
        != len(pairs)
    ):
        raise SystemExit("V17 prediagnostic is not authorized for semantic audit")
    direction_counts = Counter(str(pair.get("direction")) for pair in pairs)
    if (
        len(pairs) < 1_500
        or direction_counts["en-ja"] < 600
        or direction_counts["ja-en"] < 600
    ):
        raise SystemExit("V17 primary judge pool is below frozen minimums")

    comet = load_json(args.comet_report)
    comet_scores = validate_comet(comet, pairs=pairs, manifest=manifest)
    mapping_rows = []
    queue_rows = []
    sources = []
    seen_pair_ids: set[str] = set()
    for pair in pairs:
        pair_id = str(pair.get("pair_id", ""))
        direction = str(pair.get("direction", ""))
        if (
            not pair_id
            or pair_id in seen_pair_ids
            or direction not in {"en-ja", "ja-en"}
            or pair.get("positive_target_source") != "licensed-human-reference"
            or pair.get("generated_strings_are_positive_targets") is not False
            or pair.get("semantic_status") != "not-yet-judged"
            or not str(pair.get("source_license", "")).strip()
            or not str(pair.get("source_provenance", "")).strip()
        ):
            raise SystemExit(f"invalid V17 preselected pair: {pair_id}")
        seen_pair_ids.add(pair_id)
        judge_source_id = opaque_id("source", pair_id)
        candidate_values = []
        ids = {}
        for role, field in (
            ("licensed-reference", "reference"),
            ("generated-rollout", "hypothesis"),
        ):
            translation = str(pair.get(field, "")).strip()
            if not translation:
                raise SystemExit(f"empty V17 translation: {pair_id}: {role}")
            candidate_id = opaque_id("candidate", pair_id, role)
            ids[role] = candidate_id
            candidate_values.append(
                {"candidate_id": candidate_id, "translation": translation}
            )
            queue_rows.append(
                {
                    "judge_source_id": judge_source_id,
                    "pair_id": pair_id,
                    "candidate_id": candidate_id,
                    "source_language": pair["source_language"],
                    "target_language": pair["target_language"],
                    "domain": pair.get("domain") or "unknown",
                    "source": pair["source"],
                    "translation": translation,
                }
            )
        candidate_values.sort(key=lambda value: str(value["candidate_id"]))
        mapping_rows.append(
            {
                "judge_source_id": judge_source_id,
                "pair_id": pair_id,
                "original_source_id": pair["source_id"],
                "original_candidate_id": pair["candidate_id"],
                "direction": direction,
                "domain": pair.get("domain") or "unknown",
                "origin": pair.get("origin"),
                "long_source": bool(
                    pair.get("origin") == "human-alt-document-window"
                    or len(str(pair["source"])) >= 256
                ),
                "reference_candidate_id": ids["licensed-reference"],
                "generated_candidate_id": ids["generated-rollout"],
                "reference_sha256": hashlib.sha256(
                    str(pair["reference"]).encode()
                ).hexdigest(),
                "generated_sha256": hashlib.sha256(
                    str(pair["hypothesis"]).encode()
                ).hexdigest(),
                "deterministic_risk_tags": sorted(
                    str(tag) for tag in pair["deterministic_risk_tags"]
                ),
                "reference_minus_candidate_margin": float(
                    pair["reference_minus_candidate_margin"]
                ),
                "chrf_plus_plus": float(pair["chrf_plus_plus"]),
                "sacrebleu_intl": float(pair["sacrebleu_intl"]),
                "comet22": comet_scores[pair_id],
            }
        )
        sources.append(
            {
                "source_id": judge_source_id,
                "source_language": pair["source_language"],
                "target_language": pair["target_language"],
                "domain": pair.get("domain") or "unknown",
                "source": pair["source"],
                "candidates": candidate_values,
            }
        )
    mapping_rows.sort(key=lambda value: str(value["pair_id"]))
    queue_rows.sort(
        key=lambda value: (str(value["pair_id"]), str(value["candidate_id"]))
    )
    sources.sort(key=lambda value: str(value["source_id"]))

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
        values = [
            request_row(source, model=model, prompt_sha256=prompt_sha256)
            for source in sources
        ]
        write_jsonl(path, values)
        request_paths[model] = path
        request_rows_by_model[model] = values
    payload_hashes = {
        normalized_requests(request_rows_by_model[model]) for model in MODELS
    }
    if len(payload_hashes) != 1:
        raise SystemExit("Sonnet 5 and Opus 5 primary payloads differ")

    implementation_paths = [
        Path(__file__).resolve(),
        root / "scripts/translation/collect_on_policy_multipair_v17_primary_judge.py",
        root / "scripts/translation/run_claude_consensus_judge.py",
        root / "scripts/translation/run_synthetic_batch.py",
    ]
    contract = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": "frozen-before-any-primary-semantic-judgment",
        "purpose": (
            "primary semantic admission audit for V17 hard pairs; not model "
            "evaluation and not training authorization"
        ),
        "inputs": {
            "prediagnostic_manifest": record(manifest_path, root),
            "preselected_pairs": record(pairs_path, root),
            "comet22": record(args.comet_report, root),
        },
        "queue": {
            "mapping": record(mapping_path, root),
            "review_queue": record(queue_path, root),
            "pairs": len(mapping_rows),
            "directions": dict(sorted(direction_counts.items())),
            "candidates_per_pair": 2,
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
        "admission_policy": {
            "minimum_total_verified_pairs": 1_500,
            "minimum_verified_pairs_per_direction": 600,
            "minimum_fraction_each_required_category": 0.15,
            "required_categories": [
                "omission",
                "repetition",
                "japanese-sensitive",
            ],
            "both_judges_must_prefer_reference_without_tie": True,
            "reference_must_be_noncritical_complete_and_adequate": True,
            "at_least_one_judge_error_label_must_explain_rejection": True,
            "generated_rollout_can_be_positive_target": False,
            "comet_is_candidate_mining_only": True,
        },
        "implementation": {
            path.name: record(path, root) for path in implementation_paths
        },
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
                "pairs": len(mapping_rows),
                "directions": dict(sorted(direction_counts.items())),
                "models": list(MODELS),
                "prompt_sha256": prompt_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
