#!/usr/bin/env python3
"""Approve synthetic pairs only after two independent accepts or adjudication."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("review_queue", type=Path)
    parser.add_argument("reviews", type=Path)
    parser.add_argument("approved", type=Path)
    args = parser.parse_args()

    if args.approved.exists() and args.approved.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.approved}")

    queue = {row["candidate_id"]: row for row in rows(args.review_queue)}
    by_candidate: dict[str, list[dict]] = defaultdict(list)
    seen_reviews: set[tuple[str, str]] = set()
    for review in rows(args.reviews):
        if review["candidate_id"] not in queue:
            raise SystemExit(f"review references unknown candidate: {review['candidate_id']}")
        if review["decision"] not in {"accept", "reject", "adjudicated-accept", "adjudicated-reject"}:
            raise SystemExit("invalid review decision")
        reviewer_id = str(review.get("reviewer_id", "")).strip()
        if not reviewer_id:
            raise SystemExit("reviewer_id must be non-empty")
        review_key = (review["candidate_id"], reviewer_id)
        if review_key in seen_reviews:
            raise SystemExit(
                "duplicate review for candidate/reviewer: "
                f"{review['candidate_id']} / {reviewer_id}"
            )
        seen_reviews.add(review_key)
        by_candidate[review["candidate_id"]].append(review)

    approved: list[dict] = []
    for candidate_id, candidate in queue.items():
        reviews = by_candidate.get(candidate_id, [])
        adjudications = [r for r in reviews if r["decision"].startswith("adjudicated-")]
        if adjudications:
            if len(adjudications) != 1:
                raise SystemExit(f"candidate has multiple adjudications: {candidate_id}")
            if adjudications[0]["decision"] != "adjudicated-accept":
                continue
            approval = adjudications[0]
            reviewer_ids = [approval["reviewer_id"]]
            status = "adjudicated"
        else:
            independent = {r["reviewer_id"]: r["decision"] for r in reviews}
            accepts = [reviewer for reviewer, decision in independent.items() if decision == "accept"]
            rejects = [reviewer for reviewer, decision in independent.items() if decision == "reject"]
            if len(accepts) < 2 or rejects:
                continue
            reviewer_ids = sorted(accepts)
            status = "two-reviewer-accepted"
        approved.append({
            **candidate,
            "review_status": status,
            "reviewer_ids": reviewer_ids,
        })

    approved_by_source: dict[str, list[str]] = defaultdict(list)
    for candidate in approved:
        approved_by_source[str(candidate["source_id"])].append(candidate["candidate_id"])
    ambiguous = {
        source_id: candidate_ids
        for source_id, candidate_ids in approved_by_source.items()
        if len(candidate_ids) > 1
    }
    if ambiguous:
        example_source, candidates = next(iter(ambiguous.items()))
        raise SystemExit(
            "multiple approved targets for one source; adjudicate to exactly one: "
            f"{example_source} -> {', '.join(candidates)}"
        )

    args.approved.parent.mkdir(parents=True, exist_ok=True)
    args.approved.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in approved),
        encoding="utf-8",
    )
    print(json.dumps({"approved": len(approved), "review_queue": len(queue)}))


if __name__ == "__main__":
    main()
