#!/usr/bin/env python3
"""Build a decontaminated teacher-over-current preference dataset.

This builder is deliberately stricter than the canonical-target v3 admission
rule.  A teacher target is retained only when both independent automated judges
score it no worse than current Mimi on adequacy, fluency, and terminology and
strictly better in total.  The teacher target must also pass the existing
absolute-quality and protected-token requirements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from build_reference_anchored_distillation_dataset import (
    ALLOWED_LICENSES,
    LANGUAGES,
    license_name,
    normalized,
    provenance,
    protected_text,
    sha256,
)
from filter_training_dataset_against_protected import ProtectedIndex


PURPOSE = (
    "preregistered automated-consensus teacher-over-current preference experiment"
)
ORIGIN = "two-model-automated-consensus-preference"
REVIEW_STATUS = "two-model-unanimous-pareto-preferred-over-current"
SCORE_FIELDS = ("adequacy", "fluency", "terminology")


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"missing JSONL input: {path}")
    rows: list[dict] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise SystemExit(f"invalid JSONL {path}:{line_number}: {error}") from error
        if not isinstance(value, dict):
            raise SystemExit(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def assessments_by_candidate(judgment: dict) -> dict[str, dict]:
    value = judgment.get("assessments")
    if isinstance(value, dict):
        return {
            str(candidate_id): assessment
            for candidate_id, assessment in value.items()
            if isinstance(assessment, dict)
        }
    if isinstance(value, list):
        return {
            str(assessment.get("candidate_id", "")): assessment
            for assessment in value
            if isinstance(assessment, dict)
            and str(assessment.get("candidate_id", "")).strip()
        }
    return {}


def absolute_teacher_pass(assessment: dict) -> bool:
    return (
        assessment.get("critical_error") is False
        and assessment.get("protected_tokens_preserved") is True
        and assessment.get("error_tags") == []
        and assessment.get("adequacy", -1) >= 4
        and assessment.get("fluency", -1) >= 3
        and assessment.get("terminology", -1) >= 3
    )


def pareto_preferred(teacher: dict, current: dict) -> bool:
    if not absolute_teacher_pass(teacher):
        return False
    teacher_scores = [teacher.get(field, -1) for field in SCORE_FIELDS]
    current_scores = [current.get(field, -1) for field in SCORE_FIELDS]
    if any(not isinstance(value, int) for value in teacher_scores + current_scores):
        return False
    return all(
        teacher_value >= current_value
        for teacher_value, current_value in zip(teacher_scores, current_scores)
    ) and sum(teacher_scores) > sum(current_scores)


def queue_index(rows: list[dict]) -> dict[str, dict[str, dict]]:
    output: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        source_id = str(row.get("source_id", "")).strip()
        candidate_id = str(row.get("candidate_id", "")).strip()
        if not source_id or not candidate_id:
            raise SystemExit("review queue contains an empty source or candidate ID")
        if candidate_id in output[source_id]:
            raise SystemExit(
                f"review queue contains a duplicate candidate: {source_id}/{candidate_id}"
            )
        output[source_id][candidate_id] = row
    return output


def rank_key(seed: str, split: str, row: dict) -> bytes:
    return hashlib.sha256(
        f"{seed}\0{split}\0{row['source_id']}".encode("utf-8")
    ).digest()


def stratified_split(
    rows: list[dict],
    *,
    seed: str,
    validation_fraction: float,
) -> tuple[list[dict], list[dict]]:
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_domain[str(row.get("domain", "unknown"))].append(row)
    train: list[dict] = []
    valid: list[dict] = []
    for domain, domain_rows in sorted(by_domain.items()):
        ordered = sorted(
            domain_rows,
            key=lambda row: rank_key(seed, f"valid:{domain}", row),
        )
        if len(ordered) == 1:
            train.extend(ordered)
            continue
        valid_count = min(
            len(ordered) - 1,
            max(1, math.floor(len(ordered) * validation_fraction + 0.5)),
        )
        valid.extend(ordered[:valid_count])
        train.extend(ordered[valid_count:])
    train.sort(key=lambda row: rank_key(seed, "train", row))
    valid.sort(key=lambda row: rank_key(seed, "valid", row))
    if not train or not valid:
        raise SystemExit("preference split produced an empty train or validation set")
    return train, valid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("approved", type=Path)
    parser.add_argument("review_queue", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--direction", choices=tuple(LANGUAGES), required=True)
    parser.add_argument("--protected-suite", type=Path, action="append", required=True)
    parser.add_argument("--seed", default="mimi-canonical-pairwise-v4")
    parser.add_argument("--experiment", default="canonical-pairwise-v4")
    parser.add_argument("--id-prefix", default="canonical-pairwise-v4")
    parser.add_argument("--required-judge-model", action="append", default=[])
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--minimum-pairs", type=int, default=20)
    parser.add_argument("--maximum-jaccard", type=float, default=0.8)
    parser.add_argument("--character-ngram-size", type=int, default=5)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    if not 0 < args.validation_fraction < 0.5:
        raise SystemExit("validation-fraction must be greater than zero and below 0.5")
    if args.minimum_pairs < 2:
        raise SystemExit("minimum-pairs must be at least two")
    required_judges = set(args.required_judge_model)
    if args.required_judge_model and (
        len(args.required_judge_model) != 2 or len(required_judges) != 2
    ):
        raise SystemExit("required-judge-model must name exactly two distinct models")
    if not 0 <= args.maximum_jaccard < 1:
        raise SystemExit("maximum-jaccard must be at least zero and below one")

    expected_languages = LANGUAGES[args.direction]
    protected = ProtectedIndex(args.protected_suite, args.character_ngram_size)
    approved_rows = load_jsonl(args.approved)
    queue = queue_index(load_jsonl(args.review_queue))
    rejected: Counter[str] = Counter()
    selected: list[dict] = []

    for approved in approved_rows:
        if (
            approved.get("source_language"),
            approved.get("target_language"),
        ) != expected_languages:
            continue
        source_id = str(approved.get("source_id", "")).strip()
        teacher_id = str(approved.get("candidate_id", "")).strip()
        if (
            approved.get("candidate_origin") != "teacher"
            or approved.get("promotion_eligible") is not True
            or approved.get("review_status")
            != "two-judge-reference-anchored-canonical-absolute"
        ):
            rejected["not-canonical-approved-teacher"] += 1
            continue
        candidates = queue.get(source_id)
        if not candidates or teacher_id not in candidates:
            raise SystemExit(f"approved teacher is absent from review queue: {source_id}")
        baselines = [
            row
            for row in candidates.values()
            if row.get("candidate_origin") == "current-mimi-baseline"
        ]
        if len(baselines) != 1:
            raise SystemExit(f"expected exactly one current Mimi candidate: {source_id}")
        baseline = baselines[0]
        baseline_id = str(baseline["candidate_id"])
        teacher_candidate = candidates[teacher_id]
        if normalized(str(teacher_candidate.get("translation", ""))) != normalized(
            str(approved.get("translation", ""))
        ):
            raise SystemExit(f"approved teacher text differs from queue: {source_id}")
        if normalized(str(baseline.get("source", ""))) != normalized(
            str(approved.get("source", ""))
        ):
            raise SystemExit(f"baseline source differs from approved source: {source_id}")

        judges = approved.get("automated_judgments")
        declared_judges = {
            str(value).strip()
            for value in approved.get("judge_model_ids", [])
            if str(value).strip()
        }
        if (
            not isinstance(judges, list)
            or len(judges) != 2
            or len(declared_judges) != 2
            or str(approved.get("teacher_model", "")).strip() in declared_judges
            or (required_judges and declared_judges != required_judges)
        ):
            raise SystemExit(f"invalid independent judge declaration: {source_id}")
        evidence: list[dict] = []
        observed_judges: set[str] = set()
        unanimous = True
        for judgment in judges:
            judge_model = str(judgment.get("judge_model", "")).strip()
            response_id = str(judgment.get("judge_response_id", "")).strip()
            assessments = assessments_by_candidate(judgment)
            teacher_assessment = assessments.get(teacher_id)
            baseline_assessment = assessments.get(baseline_id)
            if (
                judge_model not in declared_judges
                or judge_model in observed_judges
                or not response_id
                or not isinstance(teacher_assessment, dict)
                or not isinstance(baseline_assessment, dict)
            ):
                raise SystemExit(f"incomplete pairwise judge evidence: {source_id}")
            observed_judges.add(judge_model)
            preferred = pareto_preferred(teacher_assessment, baseline_assessment)
            unanimous = unanimous and preferred
            evidence.append(
                {
                    "judge_model": judge_model,
                    "judge_response_id": response_id,
                    "teacher": {
                        field: teacher_assessment.get(field)
                        for field in (
                            *SCORE_FIELDS,
                            "critical_error",
                            "protected_tokens_preserved",
                            "error_tags",
                        )
                    },
                    "current": {
                        field: baseline_assessment.get(field)
                        for field in (
                            *SCORE_FIELDS,
                            "critical_error",
                            "protected_tokens_preserved",
                            "error_tags",
                        )
                    },
                    "pareto_preferred": preferred,
                }
            )
        if observed_judges != declared_judges:
            raise SystemExit(f"judge model mismatch: {source_id}")
        if not unanimous:
            rejected["not-unanimous-pareto-preferred"] += 1
            continue

        source = str(approved.get("source", "")).strip()
        chosen = str(approved.get("translation", "")).strip()
        current = str(baseline.get("translation", "")).strip()
        if not source or not chosen or not current or normalized(chosen) == normalized(current):
            raise SystemExit(f"empty or identical preference text: {source_id}")
        if any(
            protected_text(
                protected,
                text,
                args.maximum_jaccard,
                args.character_ngram_size,
            )
            for text in (source, chosen, current)
        ):
            rejected["protected-overlap"] += 1
            continue
        try:
            source_license = license_name(approved)
            source_provenance = provenance(approved)
        except ValueError as error:
            raise SystemExit(f"invalid preference license/provenance: {error}") from error
        if source_license not in ALLOWED_LICENSES:
            raise AssertionError("license_name returned a disallowed license")
        selected.append(
            {
                "id": f"{args.id_prefix}:{teacher_id}:{baseline_id}",
                "source_id": source_id,
                "source": source,
                "chosen": chosen,
                "rejected": current,
                "source_language": approved["source_language"],
                "target_language": approved["target_language"],
                "domain": approved.get("domain", "unknown"),
                "origin": ORIGIN,
                "review_status": REVIEW_STATUS,
                "chosen_candidate_id": teacher_id,
                "rejected_candidate_id": baseline_id,
                "teacher_model": approved.get("teacher_model"),
                "teacher_response_id": approved.get("teacher_response_id"),
                "judge_model_ids": sorted(declared_judges),
                "judge_evidence": sorted(
                    evidence, key=lambda value: value["judge_model"]
                ),
                "source_license": source_license,
                "source_provenance": source_provenance,
                "target_license": "project-owned-generated-output",
                "target_provenance": (
                    f"teacher={approved.get('teacher_model')} response="
                    f"{approved.get('teacher_response_id')} unanimously preferred over "
                    "authenticated current Mimi output by two distinct automated judge models"
                ),
                "private_reasoning_traces_used": False,
                "promotion_eligible": True,
            }
        )

    if len(selected) < args.minimum_pairs:
        raise SystemExit(
            f"need at least {args.minimum_pairs} unanimous preference pairs; "
            f"found {len(selected)}"
        )
    source_ids = [row["source_id"] for row in selected]
    if len(source_ids) != len(set(source_ids)):
        raise SystemExit("selected preference pairs contain duplicate sources")
    train, valid = stratified_split(
        selected,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
    )
    if {row["source_id"] for row in train} & {
        row["source_id"] for row in valid
    }:
        raise AssertionError("preference train/validation sources overlap")

    args.output.mkdir(parents=True, exist_ok=True)
    train_path = args.output / "train.jsonl"
    valid_path = args.output / "valid.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(valid_path, valid)
    manifest = {
        "schema_version": 1,
        "experiment": args.experiment,
        "purpose": PURPOSE,
        "direction": args.direction,
        "required_judge_model_ids": sorted(required_judges),
        "target_source": (
            "gpt-5.6-sol final translations unanimously Pareto-preferred over "
            "authenticated current Mimi output"
        ),
        "promotion_eligible": True,
        "claim_eligible": False,
        "private_reasoning_traces_used": False,
        "teacher_models": {
            "sequence_teachers": dict(
                sorted(Counter(str(row["teacher_model"]) for row in selected).items())
            ),
            "admission_judges": sorted(
                {model for row in selected for model in row["judge_model_ids"]}
            ),
            "rejected_response_model": "authenticated current Mimi JA-to-EN model",
        },
        "review_policy": {
            "human_reviewer_required": False,
            "independent_judge_models": 2,
            "teacher_not_a_judge": True,
            "absolute_teacher_quality_required": True,
            "unanimous_pareto_preference_over_current_required": True,
            "score_fields": list(SCORE_FIELDS),
            "strict_total_improvement_per_judge": True,
        },
        "effective_licenses": {
            "train": dict(
                sorted(Counter(row["source_license"] for row in train).items())
            ),
            "valid": dict(
                sorted(Counter(row["source_license"] for row in valid).items())
            ),
        },
        "decontamination": {
            "protected_suites": [
                {"path": str(path), "sha256": sha256(path)}
                for path in args.protected_suite
            ],
            "character_ngram_size": args.character_ngram_size,
            "maximum_jaccard_exclusive": args.maximum_jaccard,
            "screened_fields": ["source", "chosen", "rejected"],
            "train_validation_source_overlap": False,
        },
        "split": {
            "seed": args.seed,
            "method": "deterministic SHA-256 rank stratified by domain",
            "validation_fraction": args.validation_fraction,
        },
        "counts": {
            "selected": len(selected),
            "train": len(train),
            "valid": len(valid),
            "rejected": dict(sorted(rejected.items())),
            "train_domains": dict(
                sorted(Counter(str(row["domain"]) for row in train).items())
            ),
            "valid_domains": dict(
                sorted(Counter(str(row["domain"]) for row in valid).items())
            ),
        },
        "license_policy": (
            "source licenses and attribution obligations are retained per row; "
            "teacher and current-model outputs are project-generated text"
        ),
        "inputs": {
            "approved": {"path": str(args.approved), "sha256": sha256(args.approved)},
            "review_queue": {
                "path": str(args.review_queue),
                "sha256": sha256(args.review_queue),
            },
        },
        "outputs": {
            "train": {
                "path": str(train_path),
                "rows": len(train),
                "sha256": sha256(train_path),
            },
            "valid": {
                "path": str(valid_path),
                "rows": len(valid),
                "sha256": sha256(valid_path),
            },
        },
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
