#!/usr/bin/env python3
"""Create two independently shuffled, teacher-blinded bilingual review packets."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def grouped_candidates(path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    seen_candidates: set[str] = set()
    for row in rows(path):
        candidate_id = str(row["candidate_id"])
        if candidate_id in seen_candidates:
            raise SystemExit(f"duplicate candidate_id: {candidate_id}")
        seen_candidates.add(candidate_id)
        grouped[str(row["source_id"])].append(row)
    for source_id, candidates in grouped.items():
        if len(candidates) != 3:
            raise SystemExit(f"source must have exactly three review candidates: {source_id}")
        for field in (
            "source_language",
            "target_language",
            "domain",
            "source",
            "licensed_reference",
            "reference_provenance",
        ):
            if len({json.dumps(candidate.get(field), sort_keys=True) for candidate in candidates}) != 1:
                raise SystemExit(f"source candidates disagree on {field}: {source_id}")
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("review_queue", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--reviewer", action="append", required=True)
    parser.add_argument("--shuffle-seed", default="mimi-bilingual-review-v1")
    parser.add_argument("--priority", type=Path)
    args = parser.parse_args()

    reviewer_ids = [value.strip() for value in args.reviewer if value.strip()]
    if len(reviewer_ids) != 2 or len(set(reviewer_ids)) != 2:
        raise SystemExit("provide exactly two distinct --reviewer values")
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_directory}")

    grouped = grouped_candidates(args.review_queue)
    priority: dict[str, int] | None = None
    if args.priority is not None:
        priority = {}
        ranks: set[int] = set()
        for row in rows(args.priority):
            source_id = str(row.get("source_id", ""))
            rank = row.get("priority_rank")
            if (
                source_id not in grouped
                or source_id in priority
                or not isinstance(rank, int)
                or isinstance(rank, bool)
                or rank < 1
                or rank in ranks
                or row.get("priority_status") != "automated-review-order-only-not-approval"
            ):
                raise SystemExit(f"invalid automated priority row: {source_id}")
            priority[source_id] = rank
            ranks.add(rank)
        if set(priority) != set(grouped) or ranks != set(range(1, len(grouped) + 1)):
            raise SystemExit("priority file must rank every review source exactly once")
    args.output_directory.mkdir(parents=True, exist_ok=True)
    for reviewer_id in reviewer_ids:
        packet: list[dict] = []
        template: list[dict] = []
        for source_id, candidates in grouped.items():
            shuffled = sorted(
                candidates,
                key=lambda candidate: hashlib.sha256(
                    f"{args.shuffle_seed}\0{reviewer_id}\0{source_id}\0{candidate['candidate_id']}".encode()
                ).digest(),
            )
            source = candidates[0]
            packet.append(
                {
                    "review_item_id": hashlib.sha256(
                        f"{reviewer_id}\0{source_id}".encode()
                    ).hexdigest()[:24],
                    "reviewer_id": reviewer_id,
                    "source_id": source_id,
                    "source_language": source["source_language"],
                    "target_language": source["target_language"],
                    "domain": source["domain"],
                    "source": source["source"],
                    "licensed_reference": source.get("licensed_reference"),
                    "reference_provenance": source.get("reference_provenance"),
                    "candidates": [
                        {
                            "candidate_id": candidate["candidate_id"],
                            "translation": candidate["translation"],
                        }
                        for candidate in shuffled
                    ],
                    "instruction": (
                        "Choose exactly one canonical faithful candidate, or reject all. "
                        "You may also approve one different candidate as a meaning-equivalent "
                        "but genuinely lexically or structurally distinct training alternative. "
                        "Critical meaning, negation, entity, number, safety, or protected-token "
                        "errors require reject-all unless another candidate is fully correct."
                    ),
                }
            )
            template.append(
                {
                    "source_id": source_id,
                    "reviewer_id": reviewer_id,
                    "decision": "pending",
                    "selected_candidate_id": None,
                    "approved_alternative_candidate_id": None,
                    "critical_error": False,
                    "error_tags": [],
                    "notes": "",
                }
            )
        if priority is None:
            packet.sort(key=lambda row: row["review_item_id"])
        else:
            packet.sort(key=lambda row: (priority[row["source_id"]], row["review_item_id"]))
        template.sort(key=lambda row: row["source_id"])
        write_jsonl(args.output_directory / f"{reviewer_id}.packet.jsonl", packet)
        write_jsonl(args.output_directory / f"{reviewer_id}.responses.jsonl", template)

    manifest = {
        "schema_version": 1,
        "reviewers": reviewer_ids,
        "sources": len(grouped),
        "candidates": sum(len(values) for values in grouped.values()),
        "selection_policy": (
            "one canonical candidate or reject-all; optionally one distinct valid alternative; "
            "an alternative requires matching independent approval"
        ),
        "independence": "candidate order is independently shuffled per reviewer; teacher identity/style/risk tags omitted",
        "review_queue": {"path": str(args.review_queue), "sha256": sha256(args.review_queue)},
        "shuffle_seed": args.shuffle_seed,
        "source_order": (
            "independent deterministic hash"
            if priority is None
            else "automated risk priority; scores and judge identity omitted from reviewer packets"
        ),
    }
    if args.priority is not None:
        manifest["priority"] = {"path": str(args.priority), "sha256": sha256(args.priority)}
    (args.output_directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
