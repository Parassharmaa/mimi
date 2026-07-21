#!/usr/bin/env python3
"""Build conservative human-preference pairs for post-SFT Marian DQO.

Only sources where two independent bilingual reviewers selected the same
canonical candidate are eligible. A rejected candidate becomes a DQO loser
only when neither reviewer marked it as the canonical selection or as an
approved diverse alternative. Adjudicated disagreements are deliberately
excluded because they do not establish a two-reviewer pairwise preference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import defaultdict
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
    if not path.is_file():
        raise SystemExit(f"missing input: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def ngrams(text: str, size: int = 5) -> set[str]:
    value = normalized(text).replace(" ", "")
    return {value[index:index + size] for index in range(max(1, len(value) - size + 1))}


def jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / max(1, len(left | right))


def split_bucket(source_id: str, seed: str) -> float:
    digest = hashlib.sha256(f"{seed}\0{source_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def approved_ids(review: dict) -> set[str]:
    if review.get("decision") != "select":
        return set()
    return {
        str(value)
        for value in (
            review.get("selected_candidate_id"),
            review.get("approved_alternative_candidate_id"),
        )
        if value is not None
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("review_queue", type=Path)
    parser.add_argument("approved_selections", type=Path)
    parser.add_argument("protected_benchmark", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--direction", choices=tuple(LANGUAGES), required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--split-seed", default="mimi-dqo-preferences-v1")
    parser.add_argument("--maximum-jaccard", type=float, default=0.80)
    parser.add_argument("--minimum-pairs", type=int, default=100)
    args = parser.parse_args()

    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_directory}")
    if not 0 < args.validation_fraction < 1:
        raise SystemExit("validation-fraction must be between zero and one")
    if not 0 <= args.maximum_jaccard <= 1 or args.minimum_pairs < 1:
        raise SystemExit("invalid contamination threshold or minimum-pairs")

    expected_languages = LANGUAGES[args.direction]
    queue: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in load_rows(args.review_queue):
        source_id = str(row.get("source_id", "")).strip()
        candidate_id = str(row.get("candidate_id", "")).strip()
        if not source_id or not candidate_id or candidate_id in queue[source_id]:
            raise SystemExit(f"invalid or duplicate review candidate: {source_id}/{candidate_id}")
        queue[source_id][candidate_id] = row
    if not queue or any(len(values) != 3 for values in queue.values()):
        raise SystemExit("review queue must contain exactly three candidates per source")

    protected = [
        ngrams(text)
        for row in load_rows(args.protected_benchmark)
        for text in (row["source"], *row.get("references", []))
    ]
    seen_sources: set[str] = set()
    pairs: list[dict] = []
    excluded = defaultdict(int)
    for approved in load_rows(args.approved_selections):
        source_id = str(approved.get("source_id", "")).strip()
        if not source_id or source_id in seen_sources or source_id not in queue:
            raise SystemExit(f"invalid, duplicate, or unknown approved source: {source_id}")
        seen_sources.add(source_id)
        if (approved.get("source_language"), approved.get("target_language")) != expected_languages:
            continue
        if approved.get("review_status") != "two-reviewer-selected":
            excluded["not-two-reviewer-consensus"] += 1
            continue
        chosen_id = str(approved.get("candidate_id", "")).strip()
        if chosen_id not in queue[source_id]:
            raise SystemExit(f"approved candidate is not in queue: {source_id}/{chosen_id}")
        source_reviews = approved.get("source_level_reviews")
        if not isinstance(source_reviews, list) or len(source_reviews) != 2:
            raise SystemExit(f"consensus source lacks exactly two review records: {source_id}")
        reviewer_ids = [str(review.get("reviewer_id", "")).strip() for review in source_reviews]
        if not all(reviewer_ids) or len(set(reviewer_ids)) != 2:
            raise SystemExit(f"consensus source lacks independent reviewer identities: {source_id}")
        if any(
            review.get("decision") != "select"
            or review.get("selected_candidate_id") != chosen_id
            or review.get("critical_error") is True
            for review in source_reviews
        ):
            raise SystemExit(f"consensus evidence does not select the canonical candidate: {source_id}")
        if set(reviewer_ids) != set(approved.get("reviewer_ids", [])):
            raise SystemExit(f"approved reviewer IDs disagree with source reviews: {source_id}")

        chosen = queue[source_id][chosen_id]
        license_name = str(chosen.get("source_license", "")).strip()
        if license_name not in ALLOWED_LICENSES:
            raise SystemExit(f"non-distributable or unknown source license: {license_name}")
        source = str(chosen.get("source", "")).strip()
        chosen_text = str(chosen.get("translation", "")).strip()
        if not source or not chosen_text:
            raise SystemExit(f"empty source or chosen translation: {source_id}")
        if any(
            jaccard(candidate, heldout) > args.maximum_jaccard
            for candidate in (ngrams(source), ngrams(chosen_text))
            for heldout in protected
        ):
            excluded["near-protected-benchmark"] += 1
            continue

        human_approved = set().union(*(approved_ids(review) for review in source_reviews))
        if chosen_id not in human_approved:
            raise SystemExit(f"canonical candidate missing from human-approved set: {source_id}")
        for rejected_id, rejected in sorted(queue[source_id].items()):
            if rejected_id in human_approved:
                continue
            rejected_text = str(rejected.get("translation", "")).strip()
            if not rejected_text or normalized(rejected_text) == normalized(chosen_text):
                raise SystemExit(f"empty or duplicate rejected candidate: {source_id}/{rejected_id}")
            if any(
                jaccard(ngrams(rejected_text), heldout) > args.maximum_jaccard
                for heldout in protected
            ):
                excluded["rejected-near-protected-benchmark"] += 1
                continue
            pair_id = hashlib.sha256(
                f"{source_id}\0{chosen_id}\0{rejected_id}".encode()
            ).hexdigest()[:24]
            pairs.append(
                {
                    "id": f"dqo:{pair_id}",
                    "source_id": source_id,
                    "source_language": expected_languages[0],
                    "target_language": expected_languages[1],
                    "source": source,
                    "chosen": chosen_text,
                    "rejected": rejected_text,
                    "chosen_candidate_id": chosen_id,
                    "rejected_candidate_id": rejected_id,
                    "domain": chosen.get("domain", "unknown"),
                    "origin": "two-reviewer-human-preference",
                    "source_license": license_name,
                    "source_provenance": chosen.get("source_provenance"),
                    "teacher_model": chosen.get("teacher_model"),
                    "review_status": "two-reviewer-selected-over-unapproved-candidate",
                    "reviewer_ids": sorted(reviewer_ids),
                }
            )

    if len(pairs) < args.minimum_pairs:
        raise SystemExit(f"need at least {args.minimum_pairs} conservative preference pairs; found {len(pairs)}")
    if len({row["id"] for row in pairs}) != len(pairs):
        raise SystemExit("preference pair IDs are not unique")

    train = [
        row for row in pairs
        if split_bucket(row["source_id"], args.split_seed) >= args.validation_fraction
    ]
    valid = [
        row for row in pairs
        if split_bucket(row["source_id"], args.split_seed) < args.validation_fraction
    ]
    if not train or not valid:
        raise SystemExit("preference split produced an empty train or validation set")
    train_sources = {row["source_id"] for row in train}
    valid_sources = {row["source_id"] for row in valid}
    if train_sources & valid_sources:
        raise SystemExit("source-level preference split leaked across train and validation")

    args.output_directory.mkdir(parents=True, exist_ok=True)
    train_path, valid_path = args.output_directory / "train.jsonl", args.output_directory / "valid.jsonl"
    write_jsonl(train_path, sorted(train, key=lambda row: row["id"]))
    write_jsonl(valid_path, sorted(valid, key=lambda row: row["id"]))
    manifest = {
        "schema_version": 1,
        "purpose": "post-supervised-win human-preference DQO only",
        "direction": args.direction,
        "policy": {
            "consensus_required": "same canonical candidate selected by two independent bilingual reviewers",
            "loser_required": "candidate approved by neither reviewer",
            "adjudicated_preferences_allowed": False,
            "approved_diverse_alternatives_as_losers": False,
            "source_level_split": True,
            "maximum_train_heldout_jaccard": args.maximum_jaccard,
        },
        "inputs": {
            "review_queue": {"path": str(args.review_queue), "sha256": sha256(args.review_queue)},
            "approved_selections": {"path": str(args.approved_selections), "sha256": sha256(args.approved_selections)},
            "protected_benchmark": {"path": str(args.protected_benchmark), "sha256": sha256(args.protected_benchmark)},
        },
        "split_seed": args.split_seed,
        "validation_fraction": args.validation_fraction,
        "pairs": len(pairs),
        "sources": len({row["source_id"] for row in pairs}),
        "train": {"rows": len(train), "sources": len(train_sources), "sha256": sha256(train_path)},
        "valid": {"rows": len(valid), "sources": len(valid_sources), "sha256": sha256(valid_path)},
        "excluded": dict(sorted(excluded.items())),
    }
    (args.output_directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
