#!/usr/bin/env python3
"""Build a provenance-rich Marian dataset from selected teacher output + KFTT.

The promotion-eligible path requires one canonical human-approved target per
synthetic source. An explicit training-only flag can admit conservative
two-model automated consensus rows; those remain permanently ineligible for
DQO and promotion. The builder creates a deterministic train/dev split, samples
only high-quality KFTT replay, and rechecks protected-benchmark contamination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import unicodedata
from collections import Counter
from pathlib import Path


LANGUAGES = {
    "en-ja": ("en-US", "ja-JP"),
    "ja-en": ("ja-JP", "en-US"),
}
ALLOWED_LICENSES = {
    "CC0",
    "CC0-1.0",
    "CC-BY-2.0-FR",
    "CC-BY-3.0",
    "CC-BY-4.0",
    "CC-BY-SA-3.0",
    "CC-BY-SA-4.0",
    "CC BY 3.0",
    "CC BY 4.0",
    "CC BY-SA 3.0",
    "CC BY-SA 4.0",
    "Public Domain",
    "project-owned",
}


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing input: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def ngrams(text: str, size: int = 5) -> set[str]:
    value = normalized(text).replace(" ", "")
    return {value[index:index + size] for index in range(max(1, len(value) - size + 1))}


def jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / max(1, len(left | right))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_bucket(source_id: str, seed: str) -> float:
    digest = hashlib.sha256(f"{seed}\0{source_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def protected_ngrams(path: Path) -> list[set[str]]:
    protected: list[set[str]] = []
    for row in load_rows(path):
        for text in (row["source"], *row.get("references", [])):
            protected.append(ngrams(text))
    return protected


def near_protected(row: dict, protected: list[set[str]], threshold: float) -> bool:
    return any(
        jaccard(candidate, heldout) > threshold
        for candidate in (ngrams(row["source"]), ngrams(row["target"]))
        for heldout in protected
    )


def validate_automated_consensus(
    row: dict,
    source_id: str,
    candidate_id: str,
    judge_models: set[str],
) -> None:
    policy = row.get("automated_consensus_policy")
    if not isinstance(policy, dict):
        raise SystemExit(f"automated consensus policy is missing: {source_id}")
    minimums = {
        "adequacy": int(policy.get("minimum_adequacy", -1)),
        "fluency": int(policy.get("minimum_fluency", -1)),
        "terminology": int(policy.get("minimum_terminology", -1)),
    }
    if (
        minimums["adequacy"] < 4
        or minimums["fluency"] < 3
        or minimums["terminology"] < 3
        or any(
            policy.get(name) is not True
            for name in (
                "require_no_error_tags",
                "require_no_critical_error",
                "require_protected_tokens_preserved",
                "require_unique_best_per_judge",
                "require_matching_selection",
            )
        )
    ):
        raise SystemExit(f"automated consensus policy was weakened: {source_id}")
    judgments = row.get("automated_judgments")
    if not isinstance(judgments, list) or len(judgments) != 2:
        raise SystemExit(f"automated consensus requires two judgments: {source_id}")
    seen_models: set[str] = set()
    for judgment in judgments:
        if not isinstance(judgment, dict) or judgment.get("source_id") != source_id:
            raise SystemExit(f"automated judgment source mismatch: {source_id}")
        judge_model = str(judgment.get("judge_model", "")).strip()
        seen_models.add(judge_model)
        assessments = judgment.get("assessments")
        if not isinstance(assessments, dict) or len(assessments) != 3:
            raise SystemExit(f"automated judgment assessment mismatch: {source_id}")
        eligible: list[tuple[int, str]] = []
        for assessed_id, assessment in assessments.items():
            if not isinstance(assessment, dict):
                raise SystemExit(f"invalid automated assessment: {source_id}")
            if (
                assessment.get("critical_error") is not False
                or assessment.get("protected_tokens_preserved") is not True
                or assessment.get("error_tags") != []
                or any(
                    assessment.get(name, -1) < minimum
                    for name, minimum in minimums.items()
                )
            ):
                continue
            score = sum(int(assessment[name]) for name in minimums)
            eligible.append((score, str(assessed_id)))
        eligible.sort(reverse=True)
        if (
            not eligible
            or (len(eligible) > 1 and eligible[0][0] == eligible[1][0])
            or eligible[0][1] != candidate_id
        ):
            raise SystemExit(f"automated judgment does not uniquely select target: {source_id}")
    if seen_models != judge_models:
        raise SystemExit(f"automated judge identities do not match: {source_id}")
    if str(row.get("teacher_model", "")).strip() in judge_models:
        raise SystemExit(f"automated judge matches teacher: {source_id}")


def synthetic_rows(
    path: Path,
    direction: str,
    target_mode: str,
    allow_automated_consensus: bool = False,
) -> list[dict]:
    expected_source, expected_target = LANGUAGES[direction]
    selected_sources: set[str] = set()
    output: list[dict] = []
    for row in load_rows(path):
        source_id = str(row.get("source_id", "")).strip()
        candidate_id = str(row.get("candidate_id", "")).strip()
        if not source_id or not candidate_id:
            raise SystemExit("approved row is missing source_id or candidate_id")
        if source_id in selected_sources:
            raise SystemExit(f"more than one approved target for source: {source_id}")
        selected_sources.add(source_id)
        if (row.get("source_language"), row.get("target_language")) != (
            expected_source,
            expected_target,
        ):
            continue
        status = row.get("review_status")
        reviewers = {str(value).strip() for value in row.get("reviewer_ids", []) if str(value).strip()}
        if status in {"two-reviewer-accepted", "two-reviewer-selected"} and len(reviewers) < 2:
            raise SystemExit(f"approved row lacks two independent reviewers: {source_id}")
        if status == "adjudicated" and len(reviewers) < 3:
            raise SystemExit(f"adjudicated row has invalid reviewer record: {source_id}")
        automated = status == "two-judge-consensus-provisional"
        judge_models = {
            str(value).strip()
            for value in row.get("judge_model_ids", [])
            if str(value).strip()
        }
        if automated and (
            not allow_automated_consensus
            or len(judge_models) != 2
            or row.get("promotion_eligible") is not False
        ):
            raise SystemExit(f"invalid or unauthorized automated consensus row: {source_id}")
        if automated:
            validate_automated_consensus(row, source_id, candidate_id, judge_models)
        if status not in {
            "two-reviewer-accepted",
            "two-reviewer-selected",
            "adjudicated",
            "two-judge-consensus-provisional",
        }:
            raise SystemExit(f"unapproved synthetic row: {source_id}")
        license_name = str(row.get("source_license", "")).strip()
        if license_name not in ALLOWED_LICENSES:
            raise SystemExit(f"non-distributable or unknown source license: {license_name}")
        source = str(row["source"]).strip()
        target = str(row["translation"]).strip()
        if not source or not target:
            raise SystemExit(f"empty approved pair: {source_id}")
        output_row = {
                "id": f"synthetic:{candidate_id}",
                "source_id": source_id,
                "source_language": expected_source,
                "target_language": expected_target,
                "source": source,
                "target": target,
                "domain": row.get("domain", "unknown"),
                "origin": (
                    "automated-gpt-teacher-provisional"
                    if automated
                    else "reviewed-gpt-teacher"
                ),
                "source_license": license_name,
                "source_provenance": row["source_provenance"],
                "teacher_model": row.get("teacher_model"),
                "teacher_response_id": row.get("teacher_response_id"),
                "teacher_system_fingerprint": row.get("teacher_system_fingerprint"),
                "licensed_reference": row.get("licensed_reference"),
                "reference_provenance": row.get("reference_provenance"),
                "review_status": status,
                "reviewer_ids": sorted(reviewers),
                "judge_model_ids": sorted(judge_models),
                "promotion_eligible": not automated,
                "automated_judgments": row.get("automated_judgments"),
                "source_level_reviews": row.get("source_level_reviews"),
                "adjudication": row.get("adjudication"),
            }
        if target_mode == "sample-approved-diverse":
            alternative = row.get("approved_alternative")
            if alternative is not None:
                alternative_id = str(alternative.get("candidate_id", "")).strip()
                alternative_translation = str(alternative.get("translation", "")).strip()
                alternative_reviewers = {
                    str(value).strip()
                    for value in alternative.get("reviewer_ids", [])
                    if str(value).strip()
                }
                if (
                    not alternative_id
                    or alternative_id == candidate_id
                    or not alternative_translation
                    or normalized(alternative_translation) == normalized(target)
                    or len(alternative_reviewers) < 2
                    or alternative.get("review_status")
                    not in {
                        "two-reviewer-approved-diverse-alternative",
                        "adjudicated-diverse-alternative",
                    }
                ):
                    raise SystemExit(f"invalid reviewed diverse alternative: {source_id}")
                output_row["target_variants"] = [
                    {
                        "candidate_id": candidate_id,
                        "translation": target,
                        "role": "canonical",
                    },
                    {
                        "candidate_id": alternative_id,
                        "translation": alternative_translation,
                        "role": "reviewed-diverse-alternative",
                    },
                ]
        output.append(output_row)
    return output


def kftt_rows(path: Path, direction: str) -> list[dict]:
    output: list[dict] = []
    for row in load_rows(path):
        metadata = row["metadata"]
        if metadata["direction"] != direction:
            continue
        messages = row["messages"]
        if [message["role"] for message in messages] != ["system", "user", "assistant"]:
            raise SystemExit(f"unexpected KFTT chat shape: {metadata['source_id']}")
        source_language, target_language = LANGUAGES[direction]
        output.append(
            {
                "id": f"kftt:{metadata['source_id']}:{direction}",
                "source_id": metadata["source_id"],
                "source_language": source_language,
                "target_language": target_language,
                "source": messages[1]["content"].strip(),
                "target": messages[2]["content"].strip(),
                "domain": "wikipedia",
                "origin": "human-kftt-replay",
                "source_license": metadata["license"],
                "source_provenance": metadata["source"],
                "attribution": metadata["attribution"],
            }
        )
    return output


def parallel_rows(path: Path, direction: str) -> list[dict]:
    expected = LANGUAGES[direction]
    output: list[dict] = []
    for row in load_rows(path):
        if (row.get("source_language"), row.get("target_language")) != expected:
            continue
        license_name = str(row.get("source_license", "")).strip()
        if license_name not in ALLOWED_LICENSES:
            raise SystemExit(f"parallel row has unknown source license: {license_name}")
        if not str(row.get("source", "")).strip() or not str(row.get("target", "")).strip():
            raise SystemExit(f"parallel row is empty: {row.get('id')}")
        output.append(row)
    return output


def deterministic_sample(rows: list[dict], count: int, seed: str) -> list[dict]:
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{seed}\0{row['id']}".encode()).digest(),
    )
    return ranked[: min(count, len(ranked))]


def remove_split_overlap(train: list[dict], valid: list[dict]) -> tuple[list[dict], int]:
    protected_values = {
        normalized(row[field])
        for row in valid
        for field in ("source", "target")
    }
    clean: list[dict] = []
    removed = 0
    for row in train:
        if any(normalized(row[field]) in protected_values for field in ("source", "target")):
            removed += 1
            continue
        clean.append(row)
    return clean, removed


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("approved_synthetic", type=Path)
    parser.add_argument("kftt_directory", type=Path)
    parser.add_argument("protected_benchmark", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--direction", choices=tuple(LANGUAGES), required=True)
    parser.add_argument("--split-seed", default="mimi-distillation-v1")
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--kftt-replay-multiplier", type=float, default=3.0)
    parser.add_argument(
        "--parallel-corpus-directory",
        type=Path,
        action="append",
        default=[],
        help="Repeat for each prepared high-quality parallel corpus.",
    )
    parser.add_argument("--maximum-parallel-train-per-corpus", type=int, default=2_000)
    parser.add_argument("--maximum-parallel-valid-per-corpus", type=int, default=200)
    parser.add_argument("--minimum-synthetic-train", type=int, default=500)
    parser.add_argument("--minimum-synthetic-validation", type=int, default=50)
    parser.add_argument("--maximum-jaccard", type=float, default=0.80)
    parser.add_argument(
        "--reviewed-target-mode",
        choices=("canonical", "sample-approved-diverse"),
        default="canonical",
    )
    parser.add_argument(
        "--allow-automated-consensus",
        action="store_true",
        help=(
            "Admit two-distinct-judge consensus targets for provisional SFT only; "
            "the resulting dataset is not promotion eligible."
        ),
    )
    args = parser.parse_args()

    if not 0 < args.validation_fraction < 0.5:
        raise SystemExit("validation-fraction must be between 0 and 0.5")
    if args.kftt_replay_multiplier < 0:
        raise SystemExit("kftt-replay-multiplier must be non-negative")
    if min(
        args.maximum_parallel_train_per_corpus,
        args.maximum_parallel_valid_per_corpus,
    ) < 0:
        raise SystemExit("parallel corpus limits must be non-negative")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_directory}")

    synthetic = synthetic_rows(
        args.approved_synthetic,
        args.direction,
        args.reviewed_target_mode,
        args.allow_automated_consensus,
    )
    synthetic_train = [
        row
        for row in synthetic
        if split_bucket(row["source_id"], args.split_seed) >= args.validation_fraction
    ]
    synthetic_valid = [
        row
        for row in synthetic
        if split_bucket(row["source_id"], args.split_seed) < args.validation_fraction
    ]
    if len(synthetic_train) < args.minimum_synthetic_train:
        raise SystemExit(
            f"need at least {args.minimum_synthetic_train} reviewed synthetic train rows; "
            f"found {len(synthetic_train)}"
        )
    if len(synthetic_valid) < args.minimum_synthetic_validation:
        raise SystemExit(
            f"need at least {args.minimum_synthetic_validation} reviewed synthetic validation rows; "
            f"found {len(synthetic_valid)}"
        )

    kftt_train_path = args.kftt_directory / "train.jsonl"
    kftt_valid_path = args.kftt_directory / "valid.jsonl"
    parallel_train: list[dict] = []
    parallel_valid: list[dict] = []
    parallel_inputs: list[dict] = []
    for directory in args.parallel_corpus_directory:
        parallel_train_path = directory / "train.jsonl"
        parallel_valid_path = directory / "valid.jsonl"
        available_train = parallel_rows(parallel_train_path, args.direction)
        available_valid = parallel_rows(parallel_valid_path, args.direction)
        selected_train = deterministic_sample(
            available_train,
            args.maximum_parallel_train_per_corpus,
            f"{args.split_seed}:parallel-train",
        )
        selected_valid = deterministic_sample(
            available_valid,
            args.maximum_parallel_valid_per_corpus,
            f"{args.split_seed}:parallel-valid",
        )
        parallel_train.extend(selected_train)
        parallel_valid.extend(selected_valid)
        parallel_inputs.append({
            "directory": str(directory),
            "train": {
                "path": str(parallel_train_path),
                "sha256": sha256(parallel_train_path),
                "available": len(available_train),
                "selected": len(selected_train),
            },
            "valid": {
                "path": str(parallel_valid_path),
                "sha256": sha256(parallel_valid_path),
                "available": len(available_valid),
                "selected": len(selected_valid),
            },
        })
    parallel_ids = [str(row["id"]) for row in parallel_train + parallel_valid]
    if len(parallel_ids) != len(set(parallel_ids)):
        raise SystemExit("parallel corpora contain duplicate row IDs")
    replay_count = math.ceil(
        (len(synthetic_train) + len(parallel_train)) * args.kftt_replay_multiplier
    )
    replay_train = deterministic_sample(
        kftt_rows(kftt_train_path, args.direction), replay_count, args.split_seed
    )
    replay_valid = kftt_rows(kftt_valid_path, args.direction)

    protected = protected_ngrams(args.protected_benchmark)
    train = synthetic_train + replay_train + parallel_train
    valid = synthetic_valid + replay_valid + parallel_valid
    for split_name, split_rows in (("train", train), ("valid", valid)):
        contaminated = [row["id"] for row in split_rows if near_protected(row, protected, args.maximum_jaccard)]
        if contaminated:
            raise SystemExit(
                f"{split_name} has {len(contaminated)} protected-benchmark near matches; "
                f"first: {contaminated[0]}"
            )

    train, overlap_removed = remove_split_overlap(train, valid)
    random.Random(args.split_seed).shuffle(train)
    random.Random(f"{args.split_seed}:valid").shuffle(valid)

    args.output_directory.mkdir(parents=True, exist_ok=True)
    train_path = args.output_directory / "train.jsonl"
    valid_path = args.output_directory / "valid.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(valid_path, valid)
    contains_automated_consensus = any(
        row["origin"] == "automated-gpt-teacher-provisional" for row in synthetic
    )
    manifest = {
        "schema_version": 1,
        "direction": args.direction,
        "split_seed": args.split_seed,
        "validation_fraction": args.validation_fraction,
        "kftt_replay_multiplier": args.kftt_replay_multiplier,
        "parallel_corpus_limits": {
            "maximum_train_per_corpus": args.maximum_parallel_train_per_corpus,
            "maximum_valid_per_corpus": args.maximum_parallel_valid_per_corpus,
        },
        "maximum_protected_jaccard": args.maximum_jaccard,
        "reviewed_target_mode": args.reviewed_target_mode,
        "one_canonical_target_per_source": True,
        "maximum_reviewed_target_variants_per_source": 2,
        "minimum_review": (
            "two distinct automated judge models must uniquely select the same "
            "error-free candidate; provisional SFT only"
            if contains_automated_consensus
            else "the same candidate selected by two independent bilingual reviewers, "
            "or selection by an independent third adjudicator"
        ),
        "contains_automated_consensus": contains_automated_consensus,
        "promotion_eligible": not contains_automated_consensus,
        "human_review_required_for_promotion": True,
        "private_chain_of_thought_stored": False,
        "counts": {
            "synthetic_train": len(synthetic_train),
            "synthetic_valid": len(synthetic_valid),
            "synthetic_train_with_diverse_alternative": sum(
                "target_variants" in row for row in synthetic_train
            ),
            "kftt_replay_train": len(replay_train),
            "kftt_valid": len(replay_valid),
            "parallel_train": len(parallel_train),
            "parallel_valid": len(parallel_valid),
            "cross_split_overlap_removed": overlap_removed,
            "train": len(train),
            "valid": len(valid),
        },
        "origins": {
            "train": dict(Counter(row["origin"] for row in train)),
            "valid": dict(Counter(row["origin"] for row in valid)),
        },
        "inputs": {
            "approved_synthetic": {
                "path": str(args.approved_synthetic),
                "sha256": sha256(args.approved_synthetic),
            },
            "kftt_train": {"path": str(kftt_train_path), "sha256": sha256(kftt_train_path)},
            "kftt_valid": {"path": str(kftt_valid_path), "sha256": sha256(kftt_valid_path)},
            "protected_benchmark": {
                "path": str(args.protected_benchmark),
                "sha256": sha256(args.protected_benchmark),
            },
        },
        "outputs": {
            "train": {"path": str(train_path), "sha256": sha256(train_path)},
            "valid": {"path": str(valid_path), "sha256": sha256(valid_path)},
        },
    }
    if parallel_inputs:
        manifest["inputs"]["parallel_corpora"] = parallel_inputs
    (args.output_directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
