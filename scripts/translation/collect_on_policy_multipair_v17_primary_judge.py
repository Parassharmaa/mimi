#!/usr/bin/env python3
"""Collect V17's exact dual-Claude primary semantic audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from prioritize_distillation_judgments import response_payload, verify_claude_run

EXPERIMENT = "faithful-on-policy-multipair-v17-primary-semantic-audit"
MODELS = ("claude-sonnet-5", "claude-opus-5")
ASSESSMENT_KEYS = {
    "candidate_id",
    "adequacy",
    "fluent_and_complete",
    "critical_semantic_error",
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
    "particle-or-case-role",
    "register-or-honorific",
    "untranslated-source",
    "other-critical",
}
JAPANESE_SENSITIVE_TAGS = {
    "polarity",
    "terminology",
    "particle-or-case-role",
    "register-or-honorific",
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
    candidate_ids: set[str],
    source_id: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != ASSESSMENT_KEYS:
        raise SystemExit(f"invalid V17 assessment fields: {source_id}")
    candidate_id = str(value.get("candidate_id", ""))
    adequacy = value.get("adequacy")
    if (
        candidate_id not in candidate_ids
        or isinstance(adequacy, bool)
        or not isinstance(adequacy, int)
        or not 0 <= adequacy <= 4
    ):
        raise SystemExit(f"invalid V17 assessment identity/adequacy: {source_id}")
    for field in ("fluent_and_complete", "critical_semantic_error"):
        if not isinstance(value.get(field), bool):
            raise SystemExit(f"invalid V17 assessment {field}: {source_id}")
    tags = value.get("error_tags")
    if (
        not isinstance(tags, list)
        or len(tags) != len(set(tags))
        or any(not isinstance(tag, str) or tag not in ERROR_TAGS for tag in tags)
    ):
        raise SystemExit(f"invalid V17 assessment tags: {source_id}")
    evidence = str(value.get("brief_evidence", "")).strip()
    if not evidence or len(evidence) > 240:
        raise SystemExit(f"invalid V17 brief evidence: {source_id}")
    return {
        **value,
        "candidate_id": candidate_id,
        "error_tags": sorted(tags),
        "brief_evidence": evidence,
    }


def collect_one_model(
    *,
    model: str,
    request_path: Path,
    run_directory: Path,
    output_path: Path,
    expected: dict[str, set[str]],
    root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    verified = verify_claude_run(
        request_path,
        run_directory,
        output_path,
        model,
    )
    by_source: dict[str, dict[str, Any]] = {}
    response_ids: set[str] = set()
    for row in rows(output_path):
        source_id = str(row.get("custom_id", ""))
        if source_id not in expected or source_id in by_source:
            raise SystemExit(f"unknown or duplicate V17 judge source: {source_id}")
        try:
            payload, body = response_payload(row)
        except (ValueError, json.JSONDecodeError) as error:
            raise SystemExit(
                f"invalid V17 judge payload for {source_id}: {error}"
            ) from error
        if payload.get("source_id") != source_id or body.get("model") != model:
            raise SystemExit(f"V17 judge source or model differs: {source_id}")
        response_id = str(body.get("id", "")).strip()
        if not response_id or response_id in response_ids:
            raise SystemExit(f"V17 judge response ID differs: {source_id}")
        response_ids.add(response_id)
        candidate_ids = expected[source_id]
        assessments = payload.get("assessments")
        preferred = payload.get("preferred_candidate_ids")
        if (
            not isinstance(assessments, list)
            or len(assessments) != len(candidate_ids)
            or not isinstance(preferred, list)
            or not 1 <= len(preferred) <= len(candidate_ids)
            or len(preferred) != len(set(preferred))
            or any(value not in candidate_ids for value in preferred)
        ):
            raise SystemExit(f"V17 judge coverage/preference differs: {source_id}")
        indexed = {}
        for assessment in assessments:
            value = validate_assessment(
                assessment,
                candidate_ids=candidate_ids,
                source_id=source_id,
            )
            candidate_id = value["candidate_id"]
            if candidate_id in indexed:
                raise SystemExit(f"duplicate V17 assessment: {source_id}")
            indexed[candidate_id] = value
        if set(indexed) != candidate_ids:
            raise SystemExit(f"V17 assessment set differs: {source_id}")
        by_source[source_id] = {
            "response_id": response_id,
            "preferred_candidate_ids": sorted(preferred),
            "assessments": indexed,
        }
    if set(by_source) != set(expected):
        raise SystemExit(f"{model} does not cover the exact V17 primary pool")

    return by_source, {
        "model": model,
        "actual_canonical_model_usage_verified": verified[
            "actual_canonical_model_usage_verified"
        ],
        "verified_shards": verified["verified_shards"],
        "requests": record(request_path, root),
        "run_manifest": record(run_directory / "manifest.json", root),
        "output": record(output_path, root),
        "output_manifest": record(
            output_path.with_suffix(output_path.suffix + ".manifest.json"),
            root,
        ),
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
    admission = contract.get("admission_policy", {})
    if (
        contract.get("experiment") != EXPERIMENT
        or contract.get("status")
        != "frozen-before-any-primary-semantic-judgment"
        or contract.get("training_authorized") is not False
        or policy.get("required_exact_models") != list(MODELS)
        or policy.get("actual_canonical_model_usage_required_per_shard") is not True
        or policy.get("fallback_model_allowed") is not False
        or policy.get("reasoning_trace_requested_or_stored") is not False
        or admission.get("minimum_total_verified_pairs") != 1_500
        or admission.get("minimum_verified_pairs_per_direction") != 600
        or admission.get("minimum_fraction_each_required_category") != 0.15
        or admission.get("generated_rollout_can_be_positive_target") is not False
    ):
        raise SystemExit("V17 primary judge contract safety state differs")
    for item in contract["implementation"].values():
        path = root / item["path"]
        if sha256(path) != item["sha256"]:
            raise SystemExit(f"V17 contract-bound script differs: {path}")

    mapping_path = root / contract["queue"]["mapping"]["path"]
    if not mapping_path.is_file():
        mapping_path = Path(contract["queue"]["mapping"]["path"])
    if sha256(mapping_path) != contract["queue"]["mapping"]["sha256"]:
        raise SystemExit("V17 primary candidate mapping differs")
    mapping = rows(mapping_path)
    by_source = {}
    expected = {}
    for row in mapping:
        source_id = str(row["judge_source_id"])
        candidate_ids = {
            str(row["reference_candidate_id"]),
            str(row["generated_candidate_id"]),
        }
        if source_id in by_source or len(candidate_ids) != 2:
            raise SystemExit(f"duplicate/invalid V17 mapping: {source_id}")
        by_source[source_id] = row
        expected[source_id] = candidate_ids
    if len(mapping) != contract["queue"]["pairs"]:
        raise SystemExit("V17 primary mapping count differs")

    collected = {}
    evidence = {}
    for model, run_directory, output_path in (
        (MODELS[0], args.sonnet_run_directory, args.sonnet_output),
        (MODELS[1], args.opus_run_directory, args.opus_output),
    ):
        request_item = contract["judge_requests"][model]
        request_path = root / request_item["path"]
        if not request_path.is_file():
            request_path = Path(request_item["path"])
        if sha256(request_path) != request_item["sha256"]:
            raise SystemExit(f"V17 {model} primary requests differ")
        collected[model], evidence[model] = collect_one_model(
            model=model,
            request_path=request_path,
            run_directory=run_directory,
            output_path=output_path,
            expected=expected,
            root=root,
        )

    cases = []
    verified_pairs = []
    direction_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    long_counts: Counter[str] = Counter()
    for source_id in sorted(by_source):
        mapping_row = by_source[source_id]
        reference_id = str(mapping_row["reference_candidate_id"])
        generated_id = str(mapping_row["generated_candidate_id"])
        judgments = {
            model: collected[model][source_id] for model in MODELS
        }
        reference_assessments = {
            model: judgments[model]["assessments"][reference_id]
            for model in MODELS
        }
        generated_assessments = {
            model: judgments[model]["assessments"][generated_id]
            for model in MODELS
        }
        reference_preferred = all(
            reference_id in judgments[model]["preferred_candidate_ids"]
            and generated_id not in judgments[model]["preferred_candidate_ids"]
            for model in MODELS
        )
        reference_safe = all(
            not reference_assessments[model]["critical_semantic_error"]
            and reference_assessments[model]["fluent_and_complete"]
            and int(reference_assessments[model]["adequacy"]) >= 3
            for model in MODELS
        )
        generated_error_tags = sorted(
            {
                tag
                for model in MODELS
                for tag in generated_assessments[model]["error_tags"]
            }
        )
        rejection_explained = bool(generated_error_tags)
        verified = reference_preferred and reference_safe and rejection_explained
        categories = []
        if "omission" in generated_error_tags:
            categories.append("omission")
        if "repetition-nontermination" in generated_error_tags:
            categories.append("repetition")
        if JAPANESE_SENSITIVE_TAGS & set(generated_error_tags):
            categories.append("japanese-sensitive")
        if verified:
            direction = str(mapping_row["direction"])
            verified_pairs.append(str(mapping_row["pair_id"]))
            direction_counts[direction] += 1
            for category in categories:
                category_counts[category] += 1
            if mapping_row["long_source"]:
                long_counts[direction] += 1
        cases.append(
            {
                **mapping_row,
                "judgments": judgments,
                "primary_consensus": {
                    "reference_preferred_without_tie_unanimous": reference_preferred,
                    "reference_safe_unanimous": reference_safe,
                    "generated_error_tags_union": generated_error_tags,
                    "rejection_explained": rejection_explained,
                    "verified_hard_pair": verified,
                    "categories": categories,
                },
            }
        )

    total = len(verified_pairs)
    fractions = {
        category: category_counts[category] / max(1, total)
        for category in admission["required_categories"]
    }
    gates = {
        "verified_pairs_at_least_1500": total >= 1_500,
        "en_ja_verified_pairs_at_least_600": direction_counts["en-ja"] >= 600,
        "ja_en_verified_pairs_at_least_600": direction_counts["ja-en"] >= 600,
        "omission_fraction_at_least_0_15": fractions["omission"] >= 0.15,
        "repetition_fraction_at_least_0_15": fractions["repetition"] >= 0.15,
        "japanese_sensitive_fraction_at_least_0_15": (
            fractions["japanese-sensitive"] >= 0.15
        ),
        "long_sources_present_both_directions": (
            long_counts["en-ja"] > 0 and long_counts["ja-en"] > 0
        ),
    }
    passed = all(gates.values())
    output = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": (
            "primary-semantic-audit-passed"
            if passed
            else "primary-semantic-audit-rejected"
        ),
        "contract": record(args.contract, root),
        "judges": evidence,
        "counts": {
            "audited_pairs": len(cases),
            "verified_hard_pairs": {
                **dict(sorted(direction_counts.items())),
                "total": total,
                "long": dict(sorted(long_counts.items())),
            },
            "required_categories": {
                category: {
                    "pairs": category_counts[category],
                    "fraction": fractions[category],
                }
                for category in admission["required_categories"]
            },
        },
        "gates": gates,
        "reaudit_authorized": passed,
        "next_gate": (
            "deterministic stratified dual-Claude pair-label re-audit with "
            "paired bootstrap lower bound"
            if passed
            else "stop V17 without re-audit, gradients, contract, or training"
        ),
        "verified_pair_ids": sorted(verified_pairs),
        "cases": cases,
        "generated_strings_are_positive_targets": False,
        "private_reasoning_traces_stored": False,
        "human_reviewers_used": False,
        "training_authorized": False,
        "protected_evaluation_authorized": False,
        "app_change_authorized": False,
        "public_upload_authorized": False,
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
                "status": output["status"],
                "counts": output["counts"],
                "gates": gates,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
