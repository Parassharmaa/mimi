#!/usr/bin/env python3
"""Collect exact Claude-5 judgments into a fail-closed v13 taxonomy audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from prioritize_distillation_judgments import (
    response_payload,
    verify_claude_run,
)

EXPERIMENT = "canonical-safety-taxonomy-v13-ja-en"
MODELS = ("claude-sonnet-5", "claude-opus-5")
ASSESSMENT_KEYS = {
    "candidate_id",
    "adequacy",
    "fluent_and_complete",
    "critical_semantic_error",
    "representation_only_difference",
    "error_tags",
    "brief_evidence",
}
ERROR_TAGS = {
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
}
RELATED_TAGS = {
    "exact": {
        "number-or-quantity",
        "legal-citation",
        "named-entity",
        "untranslated-source",
        "other-critical",
    },
    "typed": {
        "number-or-quantity",
        "legal-citation",
        "named-entity",
        "untranslated-source",
        "other-critical",
    },
    "negation": {"polarity"},
    "generation": {"repetition-nontermination"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def validate_assessment(
    value: Any,
    *,
    expected_candidate_ids: set[str],
    source_id: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != ASSESSMENT_KEYS:
        raise SystemExit(f"invalid assessment fields: {source_id}")
    candidate_id = str(value.get("candidate_id", ""))
    if candidate_id not in expected_candidate_ids:
        raise SystemExit(f"unknown assessment candidate: {source_id}: {candidate_id}")
    adequacy = value.get("adequacy")
    if (
        isinstance(adequacy, bool)
        or not isinstance(adequacy, int)
        or not 0 <= adequacy <= 4
    ):
        raise SystemExit(f"invalid assessment adequacy: {source_id}: {candidate_id}")
    for field in (
        "fluent_and_complete",
        "critical_semantic_error",
        "representation_only_difference",
    ):
        if not isinstance(value.get(field), bool):
            raise SystemExit(f"invalid assessment {field}: {source_id}: {candidate_id}")
    tags = value.get("error_tags")
    if (
        not isinstance(tags, list)
        or len(tags) != len(set(tags))
        or any(not isinstance(tag, str) or tag not in ERROR_TAGS for tag in tags)
    ):
        raise SystemExit(f"invalid assessment tags: {source_id}: {candidate_id}")
    evidence = str(value.get("brief_evidence", "")).strip()
    if not evidence or len(evidence) > 240:
        raise SystemExit(f"invalid brief evidence: {source_id}: {candidate_id}")
    if value["representation_only_difference"] and value["critical_semantic_error"]:
        raise SystemExit(
            f"representation-only candidate is also critical: "
            f"{source_id}: {candidate_id}"
        )
    return value


def collect_one_model(
    *,
    model: str,
    request_path: Path,
    run_directory: Path,
    output_path: Path,
    expected_by_source: dict[str, set[str]],
    root: Path,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    verified = verify_claude_run(
        request_path,
        run_directory,
        output_path,
        model,
    )
    by_source: dict[str, dict[str, dict[str, Any]]] = {}
    response_ids: set[str] = set()
    for row in rows(output_path):
        source_id = str(row.get("custom_id", ""))
        if source_id not in expected_by_source or source_id in by_source:
            raise SystemExit(f"unknown or duplicate judge source: {source_id}")
        try:
            payload, body = response_payload(row)
        except (ValueError, json.JSONDecodeError) as error:
            raise SystemExit(
                f"invalid judge payload for {source_id}: {error}"
            ) from error
        if payload.get("source_id") != source_id or body.get("model") != model:
            raise SystemExit(f"judge source or model differs: {source_id}")
        response_id = str(body.get("id", "")).strip()
        if not response_id or response_id in response_ids:
            raise SystemExit(f"judge response ID differs: {source_id}")
        response_ids.add(response_id)
        assessments = payload.get("assessments")
        expected_ids = expected_by_source[source_id]
        if not isinstance(assessments, list) or len(assessments) != len(expected_ids):
            raise SystemExit(f"judge assessment coverage differs: {source_id}")
        indexed = {}
        for assessment in assessments:
            value = validate_assessment(
                assessment,
                expected_candidate_ids=expected_ids,
                source_id=source_id,
            )
            candidate_id = str(value["candidate_id"])
            if candidate_id in indexed:
                raise SystemExit(
                    f"duplicate judge candidate: {source_id}: {candidate_id}"
                )
            indexed[candidate_id] = {
                **value,
                "error_tags": sorted(value["error_tags"]),
                "judge_response_id": response_id,
            }
        if set(indexed) != expected_ids:
            raise SystemExit(f"judge candidate set differs: {source_id}")
        by_source[source_id] = indexed
    if set(by_source) != set(expected_by_source):
        raise SystemExit(f"{model} does not cover the exact v13 source set")

    run_manifest = run_directory / "manifest.json"
    output_manifest = output_path.with_suffix(output_path.suffix + ".manifest.json")
    return by_source, {
        "model": model,
        "actual_canonical_model_usage_verified": verified[
            "actual_canonical_model_usage_verified"
        ],
        "verified_shards": verified["verified_shards"],
        "requests": record(request_path, root),
        "run_manifest": record(run_manifest, root),
        "output": record(output_path, root),
        "output_manifest": record(output_manifest, root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("sonnet_run_directory", type=Path)
    parser.add_argument("sonnet_output", type=Path)
    parser.add_argument("opus_run_directory", type=Path)
    parser.add_argument("opus_output", type=Path)
    parser.add_argument("result_output", type=Path)
    args = parser.parse_args()
    if args.result_output.exists():
        raise SystemExit(f"refusing to overwrite result: {args.result_output}")

    root = Path(__file__).resolve().parents[2]
    contract = load_json(args.contract)
    policy = contract.get("judge_policy", {})
    if (
        contract.get("experiment") != EXPERIMENT
        or contract.get("status") != "frozen-before-any-taxonomy-judgment"
        or contract.get("v12_rejection_final") is not True
        or contract.get("training_authorized") is not False
        or policy.get("required_exact_models") != list(MODELS)
        or policy.get("actual_canonical_model_usage_required_per_shard") is not True
        or policy.get("fallback_model_allowed") is not False
        or policy.get("reasoning_trace_requested_or_stored") is not False
    ):
        raise SystemExit("v13 taxonomy contract safety state differs")
    for item in contract["implementation"].values():
        path = root / item["path"]
        if sha256(path) != item["sha256"]:
            raise SystemExit(f"contract-bound script differs: {path}")

    mapping_path = root / contract["queue"]["mapping"]["path"]
    if sha256(mapping_path) != contract["queue"]["mapping"]["sha256"]:
        raise SystemExit("v13 candidate mapping differs")
    mapping = rows(mapping_path)
    by_source_mapping: defaultdict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in mapping:
        source_id = str(row["source_id"])
        candidate_id = str(row["candidate_id"])
        if candidate_id in by_source_mapping[source_id]:
            raise SystemExit(f"duplicate v13 mapping candidate: {source_id}")
        by_source_mapping[source_id][candidate_id] = row
    expected_by_source = {
        source_id: set(candidates)
        for source_id, candidates in by_source_mapping.items()
    }
    if (
        len(expected_by_source) != contract["queue"]["sources"]
        or sum(len(value) for value in expected_by_source.values())
        != contract["queue"]["candidates"]
    ):
        raise SystemExit("v13 mapping coverage differs")

    collected = {}
    evidence = {}
    for model, run_directory, output_path in (
        (MODELS[0], args.sonnet_run_directory, args.sonnet_output),
        (MODELS[1], args.opus_run_directory, args.opus_output),
    ):
        request_path = root / contract["judge_requests"][model]["path"]
        if sha256(request_path) != contract["judge_requests"][model]["sha256"]:
            raise SystemExit(f"v13 {model} requests differ")
        collected[model], evidence[model] = collect_one_model(
            model=model,
            request_path=request_path,
            run_directory=run_directory,
            output_path=output_path,
            expected_by_source=expected_by_source,
            root=root,
        )

    detailed_cases = []
    registered_events = []
    agreement_counts: Counter[str] = Counter()
    role_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for source_id in sorted(expected_by_source):
        candidates = []
        for candidate_id, mapping_row in sorted(by_source_mapping[source_id].items()):
            judgments = {
                model: collected[model][source_id][candidate_id] for model in MODELS
            }
            critical_values = [
                bool(judgments[model]["critical_semantic_error"]) for model in MODELS
            ]
            representation_values = [
                bool(judgments[model]["representation_only_difference"])
                for model in MODELS
            ]
            complete_values = [
                bool(judgments[model]["fluent_and_complete"]) for model in MODELS
            ]
            adequacy_values = [int(judgments[model]["adequacy"]) for model in MODELS]
            consensus = {
                "critical_semantic_error_fail_closed": any(critical_values),
                "critical_semantic_error_unanimous": all(critical_values),
                "representation_only_unanimous": (
                    all(representation_values)
                    and not any(critical_values)
                    and all(complete_values)
                    and min(adequacy_values) >= 3
                ),
                "clean_translation_unanimous": (
                    not any(critical_values)
                    and all(complete_values)
                    and min(adequacy_values) >= 3
                ),
                "critical_disagreement": len(set(critical_values)) > 1,
                "representation_disagreement": (len(set(representation_values)) > 1),
                "error_tags_union": sorted(
                    {tag for model in MODELS for tag in judgments[model]["error_tags"]}
                ),
                "error_tags_intersection": sorted(
                    set(judgments[MODELS[0]]["error_tags"])
                    & set(judgments[MODELS[1]]["error_tags"])
                ),
            }
            agreement_counts["candidates"] += 1
            agreement_counts["critical_agreement"] += int(
                not consensus["critical_disagreement"]
            )
            agreement_counts["representation_agreement"] += int(
                not consensus["representation_disagreement"]
            )
            role = str(mapping_row["role"])
            role_counts[role]["candidates"] += 1
            role_counts[role]["critical_fail_closed"] += int(
                consensus["critical_semantic_error_fail_closed"]
            )
            role_counts[role]["representation_only"] += int(
                consensus["representation_only_unanimous"]
            )
            role_counts[role]["clean_unanimous"] += int(
                consensus["clean_translation_unanimous"]
            )
            registered = []
            for failure_type in mapping_row["registered_new_failure_types"]:
                related = RELATED_TAGS[failure_type]
                support = {
                    model: bool(related & set(judgments[model]["error_tags"]))
                    for model in MODELS
                }
                event = {
                    "source_id": source_id,
                    "candidate_id": candidate_id,
                    "role": role,
                    "registered_failure_type": failure_type,
                    "related_error_tags": sorted(related),
                    "judge_support": support,
                    "supported_fail_closed": any(support.values()),
                    "supported_unanimously": all(support.values()),
                    "unanimous_no_related_tag": not any(support.values()),
                    "related_tag_disagreement": len(set(support.values())) > 1,
                    "representation_only_unanimous": consensus[
                        "representation_only_unanimous"
                    ],
                    "critical_semantic_error_fail_closed": consensus[
                        "critical_semantic_error_fail_closed"
                    ],
                }
                registered.append(event)
                registered_events.append(event)
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "role": role,
                    "translation_sha256": mapping_row["translation_sha256"],
                    "judgments": judgments,
                    "consensus": consensus,
                    "registered_failure_audit": registered,
                }
            )
        detailed_cases.append(
            {
                "source_id": source_id,
                "candidates": candidates,
            }
        )

    by_role_and_failure: defaultdict[tuple[str, str], Counter[str]] = defaultdict(
        Counter
    )
    for event in registered_events:
        counter = by_role_and_failure[(event["role"], event["registered_failure_type"])]
        counter["registered"] += 1
        counter["supported_fail_closed"] += int(event["supported_fail_closed"])
        counter["supported_unanimously"] += int(event["supported_unanimously"])
        counter["unanimous_no_related_tag"] += int(event["unanimous_no_related_tag"])
        counter["related_tag_disagreement"] += int(event["related_tag_disagreement"])
        counter["representation_only_unanimous"] += int(
            event["representation_only_unanimous"]
        )
        counter["critical_semantic_error_fail_closed"] += int(
            event["critical_semantic_error_fail_closed"]
        )
    registered_summary = {
        role: {
            failure_type: dict(
                sorted(by_role_and_failure[(role, failure_type)].items())
            )
            for candidate_role, failure_type in sorted(by_role_and_failure)
            if candidate_role == role
        }
        for role in sorted(
            {candidate_role for candidate_role, _ in by_role_and_failure}
        )
    }
    output = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": "dual-claude5-taxonomy-complete",
        "purpose": (
            "future-evaluator calibration only; v12 remains rejected and no "
            "model selection is changed"
        ),
        "contract": record(args.contract, root),
        "judges": evidence,
        "counts": {
            "sources": len(detailed_cases),
            "candidates": agreement_counts["candidates"],
            "registered_failure_events": len(registered_events),
            "critical_agreement": agreement_counts["critical_agreement"],
            "representation_agreement": agreement_counts["representation_agreement"],
            "by_role": {
                role: dict(sorted(counts.items()))
                for role, counts in sorted(role_counts.items())
            },
            "registered_failure_audit": registered_summary,
        },
        "cases": detailed_cases,
        "v12_selection_or_decision_changed": False,
        "v12_rejection_final": True,
        "training_authorized": False,
        "protected_evaluation_authorized": False,
        "app_change_authorized": False,
        "public_upload_authorized": False,
        "reasoning_traces_stored": False,
        "human_reviewers_used": False,
    }
    args.result_output.parent.mkdir(parents=True, exist_ok=True)
    args.result_output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "result": display_path(args.result_output, root),
                "result_sha256": sha256(args.result_output),
                "counts": output["counts"],
                "v12_decision_changed": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
