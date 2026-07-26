#!/usr/bin/env python3
"""Build a 25%-synthetic, reference-anchored Marian distillation dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_distillation_dataset import validate_automated_consensus  # noqa: E402
from filter_training_dataset_against_protected import ProtectedIndex  # noqa: E402


LANGUAGES = {
    "en-ja": ("en-US", "ja-JP"),
    "ja-en": ("ja-JP", "en-US"),
}
ALLOWED_LICENSES = {
    "CC-BY-2.0-FR",
    "CC-BY-4.0",
    "CC-BY-SA-3.0",
    "PDL-1.0-compatible-CC-BY-4.0",
    "project-owned",
}
ORIGIN_BUCKETS = {
    "human-kftt-replay": "kftt",
    "human-alt-parallel": "alt",
    "finalized-japanese-law-translation": "jlt",
    "human-tatoeba-bidirectional-agreement-filtered": "tatoeba",
    "mimi-shipped-ui-pair": "ui",
    "human-alt-document-window": "alt-document",
}
REPLAY_SHARES = {
    "kftt": 0.25,
    "alt": 0.22,
    "jlt": 0.20,
    "tatoeba": 0.25,
    "alt-document": 0.07,
    "ui": 0.01,
}


def sha256(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"missing dataset input: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid JSON input {path}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"expected a JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"missing JSONL input: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def authenticated_dataset(directory: Path) -> tuple[dict[str, list[dict]], dict]:
    manifest_path = directory / "manifest.json"
    manifest = load_json(manifest_path)
    rows: dict[str, list[dict]] = {}
    contract = {
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256(manifest_path),
        },
        "outputs": {},
    }
    for split in ("train", "valid"):
        path = directory / f"{split}.jsonl"
        actual = sha256(path)
        expected = manifest.get("outputs", {}).get(split, {}).get("sha256")
        if expected != actual:
            raise SystemExit(
                f"authenticated dataset {directory} has a bad {split} hash"
            )
        rows[split] = load_jsonl(path)
        contract["outputs"][split] = {
            "path": str(path),
            "sha256": actual,
            "rows": len(rows[split]),
        }
    return rows, contract


def direction_matches(row: dict, direction: str) -> bool:
    return (
        row.get("source_language"),
        row.get("target_language"),
    ) == LANGUAGES[direction]


def license_name(row: dict) -> str:
    value = str(row.get("source_license") or row.get("license") or "").strip()
    if value not in ALLOWED_LICENSES:
        raise ValueError(f"license is not distributable: {value!r}")
    return value


def provenance(row: dict) -> str:
    value = str(
        row.get("source_provenance") or row.get("provenance") or ""
    ).strip()
    if not value:
        raise ValueError("missing source provenance")
    return value


def protected_text(
    index: ProtectedIndex,
    text: str,
    maximum_jaccard: float,
    ngram_size: int,
) -> bool:
    return index.match(text, maximum_jaccard, ngram_size) is not None


def validate_approved_row(row: dict, seed: dict) -> None:
    source_id = str(row.get("source_id", "")).strip()
    candidate_id = str(row.get("candidate_id", "")).strip()
    if (
        row.get("review_status")
        not in {
            "two-judge-reference-anchored",
            "two-judge-reference-anchored-canonical-absolute",
        }
        or row.get("promotion_eligible") is not True
        or row.get("candidate_origin") != "teacher"
        or not candidate_id
    ):
        raise SystemExit(f"row is not a promotion-safe reference-anchored win: {source_id}")
    judges = {
        str(value).strip()
        for value in row.get("judge_model_ids", [])
        if str(value).strip()
    }
    if len(judges) != 2 or str(row.get("teacher_model", "")).strip() in judges:
        raise SystemExit(f"row lacks two independent non-teacher judges: {source_id}")
    if row.get("review_status") == "two-judge-reference-anchored":
        policy = row.get("automated_consensus_policy")
        if (
            not isinstance(policy, dict)
            or policy.get("candidate_count") != 4
            or policy.get("licensed_reference_blinded_when_available") is not True
            or policy.get("selected_candidate_must_be_teacher_only") is not True
        ):
            raise SystemExit(
                f"row has no four-candidate reference-anchor policy: {source_id}"
            )
        validate_automated_consensus(row, source_id, candidate_id, judges)
    else:
        policy = row.get("canonical_absolute_consensus_policy")
        if (
            not isinstance(policy, dict)
            or policy.get("preregistered") is not True
            or policy.get("minimum_adequacy") != 4
            or policy.get("minimum_fluency") != 3
            or policy.get("minimum_terminology") != 3
            or policy.get("require_no_error_tags") is not True
            or policy.get("require_no_critical_error") is not True
            or policy.get("require_protected_tokens_preserved") is not True
            or policy.get("require_two_distinct_judge_models") is not True
            or policy.get("canonical_candidate_selected_before_judging") is not True
            or policy.get("licensed_reference_anchor_present") is not True
            or policy.get("current_mimi_baseline_present") is not True
            or policy.get("training_authorized_by_pilot_alone") is not False
        ):
            raise SystemExit(
                f"row has no preregistered canonical absolute policy: {source_id}"
            )
        judgments = row.get("automated_judgments")
        if not isinstance(judgments, list) or len(judgments) != 2:
            raise SystemExit(
                f"canonical row lacks two complete judgments: {source_id}"
            )
        found_judges: set[str] = set()
        response_ids: set[str] = set()
        for judgment in judgments:
            if (
                not isinstance(judgment, dict)
                or judgment.get("source_id") != source_id
                or judgment.get("priority_status")
                != "automated-review-order-only-not-approval"
            ):
                raise SystemExit(f"invalid canonical judgment: {source_id}")
            judge_model = str(judgment.get("judge_model", "")).strip()
            response_id = str(judgment.get("judge_response_id", "")).strip()
            if (
                judge_model not in judges
                or judge_model in found_judges
                or not response_id
                or response_id in response_ids
            ):
                raise SystemExit(
                    f"canonical judgment independence failure: {source_id}"
                )
            found_judges.add(judge_model)
            response_ids.add(response_id)
            assessments = judgment.get("assessments")
            if isinstance(assessments, dict):
                by_candidate = {
                    str(key): value
                    for key, value in assessments.items()
                    if isinstance(value, dict)
                }
            elif isinstance(assessments, list):
                by_candidate = {
                    str(assessment.get("candidate_id", "")): assessment
                    for assessment in assessments
                    if isinstance(assessment, dict)
                }
            else:
                by_candidate = {}
            if len(by_candidate) != 3:
                raise SystemExit(
                    f"canonical judgment must cover three candidates: {source_id}"
                )
            if candidate_id not in by_candidate:
                raise SystemExit(
                    f"canonical judgment candidate coverage failure: {source_id}"
                )
            assessment = by_candidate[candidate_id]
            if (
                assessment.get("critical_error") is not False
                or assessment.get("protected_tokens_preserved") is not True
                or assessment.get("error_tags") != []
                or assessment.get("adequacy", -1) < 4
                or assessment.get("fluency", -1) < 3
                or assessment.get("terminology", -1) < 3
            ):
                raise SystemExit(
                    f"canonical target does not pass both judgments: {source_id}"
                )
        if found_judges != judges:
            raise SystemExit(f"canonical judge model mismatch: {source_id}")
    if normalized(str(row.get("source", ""))) != normalized(str(seed.get("source", ""))):
        raise SystemExit(f"approved source differs from frozen seed: {source_id}")
    if normalized(str(row.get("licensed_reference", ""))) != normalized(
        str(seed.get("reference_translation", ""))
    ):
        raise SystemExit(f"approved reference differs from frozen seed: {source_id}")


def allocate_counts(total: int) -> dict[str, int]:
    raw = {bucket: total * share for bucket, share in REPLAY_SHARES.items()}
    output = {bucket: math.floor(value) for bucket, value in raw.items()}
    remainder = total - sum(output.values())
    order = sorted(
        raw,
        key=lambda bucket: (
            -(raw[bucket] - output[bucket]),
            bucket,
        ),
    )
    for bucket in order[:remainder]:
        output[bucket] += 1
    if sum(output.values()) != total:
        raise AssertionError("replay allocation does not sum to requested total")
    return output


def ranked(seed: str, bucket: str, rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}\0{bucket}\0{row['source']}\0{row['target']}".encode("utf-8")
        ).digest(),
    )


def choose_with_repetition(
    pool: list[dict],
    count: int,
    seed: str,
    bucket: str,
) -> list[dict]:
    if count == 0:
        return []
    if not pool:
        raise SystemExit(f"no eligible human replay rows for required bucket {bucket}")
    ordered = ranked(seed, bucket, pool)
    output: list[dict] = []
    for index in range(count):
        source = ordered[index % len(ordered)]
        output.append(
            {
                **source,
                "id": f"human-replay:{bucket}:{index:06d}:{source['id']}",
                "training_repeat_index": index // len(ordered),
                "replay_bucket": bucket,
            }
        )
    return output


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("approved", type=Path)
    parser.add_argument("pilot_seeds", type=Path)
    parser.add_argument("validation_dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--direction", choices=tuple(LANGUAGES), required=True)
    parser.add_argument("--replay-dataset", type=Path, action="append", required=True)
    parser.add_argument("--protected-suite", type=Path, action="append", required=True)
    parser.add_argument("--seed", default="mimi-gpt56-reference-anchored-dataset-v1")
    parser.add_argument("--maximum-jaccard", type=float, default=0.8)
    parser.add_argument("--character-ngram-size", type=int, default=5)
    parser.add_argument("--minimum-approved", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    if not 0 <= args.maximum_jaccard < 1:
        raise SystemExit("maximum-jaccard must be at least zero and below one")
    if args.minimum_approved < 1:
        raise SystemExit("minimum-approved must be positive")

    protected = ProtectedIndex(args.protected_suite, args.character_ngram_size)
    seeds = {
        str(row["id"]): row
        for row in load_jsonl(args.pilot_seeds)
    }
    approved_rows = [
        row
        for row in load_jsonl(args.approved)
        if direction_matches(row, args.direction)
    ]
    if len(approved_rows) < args.minimum_approved:
        raise SystemExit(
            f"need at least {args.minimum_approved} anchored rows for "
            f"{args.direction}; found {len(approved_rows)}"
        )
    source_ids = [str(row.get("source_id", "")) for row in approved_rows]
    if len(source_ids) != len(set(source_ids)):
        raise SystemExit("approved input contains duplicate sources")

    validation_rows_by_split, validation_input = authenticated_dataset(
        args.validation_dataset
    )
    valid: list[dict] = []
    validation_exclusions: Counter = Counter()
    for row in validation_rows_by_split["valid"]:
        if not direction_matches(row, args.direction):
            continue
        try:
            license_name(row)
            provenance(row)
        except ValueError as error:
            raise SystemExit(f"invalid validation row: {error}") from error
        if any(
            protected_text(
                protected,
                str(row.get(field, "")),
                args.maximum_jaccard,
                args.character_ngram_size,
            )
            for field in ("source", "target")
        ):
            validation_exclusions["protected-overlap"] += 1
            continue
        valid.append({**row, "promotion_eligible": True})
    if not valid:
        raise SystemExit("no human validation rows remain after screening")
    valid_text = {
        normalized(str(row[field]))
        for row in valid
        for field in ("source", "target")
    }

    synthetic: list[dict] = []
    anchor_replay: list[dict] = []
    approved_source_text: set[str] = set()
    for row in approved_rows:
        source_id = str(row["source_id"])
        seed = seeds.get(source_id)
        if seed is None:
            raise SystemExit(f"approved row is absent from frozen pilot seeds: {source_id}")
        validate_approved_row(row, seed)
        source = str(row["source"]).strip()
        target = str(row["translation"]).strip()
        reference = str(row["licensed_reference"]).strip()
        if any(
            protected_text(
                protected,
                text,
                args.maximum_jaccard,
                args.character_ngram_size,
            )
            for text in (source, target, reference)
        ):
            raise SystemExit(f"approved row overlaps a protected suite: {source_id}")
        if any(normalized(text) in valid_text for text in (source, target, reference)):
            raise SystemExit(f"approved row overlaps human validation: {source_id}")
        source_key = normalized(source)
        if source_key in approved_source_text:
            raise SystemExit(f"approved rows duplicate normalized source: {source_id}")
        approved_source_text.add(source_key)
        source_license = license_name(row)
        source_provenance = provenance(row)
        shared = {
            "source_id": source_id,
            "source_language": row["source_language"],
            "target_language": row["target_language"],
            "source": source,
            "domain": row.get("domain", "unknown"),
            "source_license": source_license,
            "source_provenance": source_provenance,
        }
        synthetic.append(
            {
                **shared,
                "id": f"reference-anchored-teacher:{row['candidate_id']}",
                "target": target,
                "origin": "automated-gpt-teacher-reference-anchored",
                "target_license": "project-owned-generated-output",
                "target_provenance": (
                    f"teacher={row.get('teacher_model')} response="
                    f"{row.get('teacher_response_id')} judges="
                    f"{','.join(sorted(row.get('judge_model_ids', [])))}"
                ),
                "teacher_model": row.get("teacher_model"),
                "teacher_response_id": row.get("teacher_response_id"),
                "judge_model_ids": sorted(row.get("judge_model_ids", [])),
                "licensed_reference": reference,
                "reference_provenance": row.get("reference_provenance"),
                "review_status": row["review_status"],
                "promotion_eligible": True,
            }
        )
        anchor_replay.append(
            {
                **shared,
                "id": f"same-source-human-anchor:{source_id}",
                "target": reference,
                "origin": "licensed-human-reference-anchor",
                "target_license": source_license,
                "target_provenance": row.get("reference_provenance"),
                "promotion_eligible": True,
            }
        )

    replay_inputs: list[dict] = []
    replay_pool: dict[str, dict[str, dict]] = defaultdict(dict)
    replay_rejections: Counter = Counter()
    for directory in args.replay_dataset:
        dataset, contract = authenticated_dataset(directory)
        replay_inputs.append({"directory": str(directory), **contract})
        for row in dataset["train"]:
            if not direction_matches(row, args.direction):
                continue
            bucket = ORIGIN_BUCKETS.get(str(row.get("origin", "")))
            if bucket is None:
                replay_rejections["unsupported-origin"] += 1
                continue
            try:
                row_license = license_name(row)
                row_provenance = provenance(row)
            except ValueError as error:
                raise SystemExit(f"invalid replay row: {error}") from error
            source = str(row["source"]).strip()
            target = str(row["target"]).strip()
            if normalized(source) in approved_source_text:
                replay_rejections["approved-source"] += 1
                continue
            if any(normalized(text) in valid_text for text in (source, target)):
                replay_rejections["validation-overlap"] += 1
                continue
            if any(
                protected_text(
                    protected,
                    text,
                    args.maximum_jaccard,
                    args.character_ngram_size,
                )
                for text in (source, target)
            ):
                replay_rejections["protected-overlap"] += 1
                continue
            key = f"{normalized(source)}\0{normalized(target)}"
            replay_pool[bucket].setdefault(
                key,
                {
                    **row,
                    "source_license": row_license,
                    "source_provenance": row_provenance,
                    "promotion_eligible": True,
                },
            )

    general_replay_count = len(synthetic) * 2
    replay_counts = allocate_counts(general_replay_count)
    general_replay: list[dict] = []
    for bucket, count in replay_counts.items():
        general_replay.extend(
            choose_with_repetition(
                list(replay_pool[bucket].values()),
                count,
                args.seed,
                bucket,
            )
        )
    train = synthetic + anchor_replay + general_replay
    train.sort(
        key=lambda row: hashlib.sha256(
            f"{args.seed}\0train\0{row['id']}".encode("utf-8")
        ).digest()
    )
    synthetic_fraction = len(synthetic) / len(train)
    if synthetic_fraction > 0.25 or len(train) != len(synthetic) * 4:
        raise SystemExit("internal error: the registered 25% synthetic ceiling was violated")
    if any(row.get("promotion_eligible") is not True for row in train + valid):
        raise SystemExit("promotion dataset contains an ineligible row")

    args.output.mkdir(parents=True, exist_ok=True)
    train_path = args.output / "train.jsonl"
    valid_path = args.output / "valid.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(valid_path, valid)
    manifest = {
        "schema_version": 1,
        "experiment": "reference-anchored final-translation distillation dataset",
        "direction": args.direction,
        "target_source": (
            "reviewed-canonical-teacher-and-licensed-human-reference-mixture"
        ),
        "promotion_eligible": True,
        "private_reasoning_traces_used": False,
        "teacher_models": {
            "sequence_teachers": dict(
                sorted(
                    Counter(
                        str(row.get("teacher_model", "")).strip()
                        for row in synthetic
                        if str(row.get("teacher_model", "")).strip()
                    ).items()
                )
            ),
            "admission_judges": sorted(
                {
                    str(model).strip()
                    for row in synthetic
                    for model in row.get("judge_model_ids", [])
                    if str(model).strip()
                }
            ),
        },
        "effective_licenses": {
            "train": dict(
                sorted(Counter(license_name(row) for row in train).items())
            ),
            "valid": dict(
                sorted(Counter(license_name(row) for row in valid).items())
            ),
        },
        "synthetic_policy": {
            "review_statuses": sorted(
                {row["review_status"] for row in approved_rows}
            ),
            "source_only_provisional_rows_allowed": False,
            "licensed_reference_anchor_required": True,
            "same_source_human_anchor_required": True,
            "human_replay_per_synthetic": 3,
            "maximum_synthetic_fraction": 0.25,
            "actual_synthetic_fraction": synthetic_fraction,
        },
        "replay_policy": {
            "general_human_replay_per_synthetic": 2,
            "shares": REPLAY_SHARES,
            "counts": replay_counts,
            "sampling": (
                "deterministic SHA-256 rank without replacement, then deterministic "
                "repetition only when a corpus has fewer unique eligible rows"
            ),
        },
        "decontamination": {
            "protected_suites": [
                {"path": str(path), "sha256": sha256(path)}
                for path in args.protected_suite
            ],
            "character_ngram_size": args.character_ngram_size,
            "maximum_jaccard_exclusive": args.maximum_jaccard,
            "human_validation_exclusions": dict(validation_exclusions),
            "replay_rejections": dict(replay_rejections),
            "train_validation_exact_overlap": False,
        },
        "counts": {
            "approved_teacher_targets": len(synthetic),
            "same_source_human_anchors": len(anchor_replay),
            "general_human_replay": len(general_replay),
            "train": len(train),
            "valid_human_only": len(valid),
            "train_origins": dict(sorted(Counter(row["origin"] for row in train).items())),
            "licenses": dict(
                sorted(Counter(license_name(row) for row in train + valid).items())
            ),
        },
        "license_policy": (
            "retain per-row source/target licenses and provenance; generated targets "
            "are project-owned but inherit all source attribution obligations"
        ),
        "inputs": {
            "approved": {
                "path": str(args.approved),
                "sha256": sha256(args.approved),
            },
            "pilot_seeds": {
                "path": str(args.pilot_seeds),
                "sha256": sha256(args.pilot_seeds),
            },
            "validation_dataset": validation_input,
            "replay_datasets": replay_inputs,
        },
        "outputs": {
            "train": {
                "path": str(train_path),
                "sha256": sha256(train_path),
            },
            "valid": {
                "path": str(valid_path),
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
