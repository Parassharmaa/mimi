#!/usr/bin/env python3
"""Approve one target per source from independent selection or adjudication."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def rows(path: Path | None) -> list[dict]:
    if path is None:
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def selection_map(path: Path, queue: dict[str, dict[str, dict]]) -> tuple[str, dict[str, dict]]:
    output: dict[str, dict] = {}
    reviewer_ids: set[str] = set()
    for row in rows(path):
        source_id = str(row["source_id"])
        reviewer_id = str(row.get("reviewer_id", "")).strip()
        decision = row.get("decision")
        if source_id not in queue:
            raise SystemExit(f"review references unknown source: {source_id}")
        if source_id in output:
            raise SystemExit(f"review file contains duplicate source: {source_id}")
        if not reviewer_id:
            raise SystemExit(f"review is missing reviewer_id: {source_id}")
        reviewer_ids.add(reviewer_id)
        if decision not in {"select", "reject-all"}:
            raise SystemExit(f"review remains pending or invalid: {source_id}")
        selected = row.get("selected_candidate_id")
        alternative = row.get("approved_alternative_candidate_id")
        if decision == "select":
            if selected not in queue[source_id]:
                raise SystemExit(f"review selected a candidate from another source: {source_id}")
            if row.get("critical_error"):
                raise SystemExit(f"review cannot select while declaring a critical error: {source_id}")
            if alternative is not None:
                if alternative not in queue[source_id] or alternative == selected:
                    raise SystemExit(f"review approved an invalid alternative: {source_id}")
        elif selected is not None or alternative is not None:
            raise SystemExit(f"reject-all must not select or approve a candidate: {source_id}")
        output[source_id] = row
    if len(reviewer_ids) != 1:
        raise SystemExit(f"each review file must contain exactly one reviewer ID: {path}")
    missing = set(queue) - set(output)
    if missing:
        raise SystemExit(f"review file is missing {len(missing)} sources; first: {next(iter(missing))}")
    return next(iter(reviewer_ids)), output


def approved_candidate_ids(review: dict) -> set[str]:
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
    parser.add_argument("review_a", type=Path)
    parser.add_argument("review_b", type=Path)
    parser.add_argument("approved", type=Path)
    parser.add_argument("disagreements", type=Path)
    parser.add_argument("--adjudications", type=Path)
    args = parser.parse_args()

    for output in (args.approved, args.disagreements):
        if output.exists() and output.stat().st_size:
            raise SystemExit(f"refusing to overwrite non-empty output: {output}")

    queue: dict[str, dict[str, dict]] = defaultdict(dict)
    for candidate in rows(args.review_queue):
        source_id = str(candidate["source_id"])
        candidate_id = str(candidate["candidate_id"])
        if candidate_id in queue[source_id]:
            raise SystemExit(f"duplicate candidate: {candidate_id}")
        queue[source_id][candidate_id] = candidate
    if any(len(candidates) != 3 for candidates in queue.values()):
        raise SystemExit("every source must have exactly three candidates")

    reviewer_a, selections_a = selection_map(args.review_a, queue)
    reviewer_b, selections_b = selection_map(args.review_b, queue)
    if reviewer_a == reviewer_b:
        raise SystemExit("the two review files must use distinct reviewer IDs")

    adjudications: dict[str, dict] = {}
    for row in rows(args.adjudications):
        source_id = str(row.get("source_id", "")).strip()
        if source_id not in queue:
            raise SystemExit(f"adjudication references unknown source: {source_id}")
        if source_id in adjudications:
            raise SystemExit(f"duplicate adjudication for source: {source_id}")
        adjudications[source_id] = row
    approved: list[dict] = []
    disagreements: list[dict] = []
    used_adjudications: set[str] = set()
    for source_id, candidates in queue.items():
        left, right = selections_a[source_id], selections_b[source_id]
        left_selection = left.get("selected_candidate_id") if left["decision"] == "select" else None
        right_selection = right.get("selected_candidate_id") if right["decision"] == "select" else None
        selected: str | None = None
        status: str | None = None
        reviewer_ids: list[str] = []
        alternative: str | None = None
        alternative_status: str | None = None
        resolved_reject = False
        if left_selection is not None and left_selection == right_selection:
            selected = left_selection
            status = "two-reviewer-selected"
            reviewer_ids = sorted([reviewer_a, reviewer_b])
            left_alternative = left.get("approved_alternative_candidate_id")
            right_alternative = right.get("approved_alternative_candidate_id")
            if left_alternative is not None and left_alternative == right_alternative:
                alternative = str(left_alternative)
                alternative_status = "two-reviewer-approved-diverse-alternative"
        elif left_selection is None and right_selection is None:
            continue
        else:
            adjudication = adjudications.get(source_id)
            if adjudication is not None:
                used_adjudications.add(source_id)
                adjudicator = str(adjudication.get("reviewer_id", "")).strip()
                if not adjudicator or adjudicator in {reviewer_a, reviewer_b}:
                    raise SystemExit(f"adjudicator must be an independent third reviewer: {source_id}")
                if adjudication.get("decision") == "select":
                    selected_candidate = adjudication.get("selected_candidate_id")
                    if selected_candidate not in candidates:
                        raise SystemExit(f"invalid adjudicated candidate: {source_id}")
                    if adjudication.get("critical_error"):
                        raise SystemExit(f"adjudication cannot select with critical error: {source_id}")
                    selected = selected_candidate
                    status = "adjudicated"
                    reviewer_ids = sorted([reviewer_a, reviewer_b, adjudicator])
                    adjudicated_alternative = adjudication.get(
                        "approved_alternative_candidate_id"
                    )
                    if adjudicated_alternative is not None:
                        if (
                            adjudicated_alternative not in candidates
                            or adjudicated_alternative == selected
                            or adjudicated_alternative
                            not in (
                                approved_candidate_ids(left)
                                & approved_candidate_ids(right)
                            )
                        ):
                            raise SystemExit(
                                f"adjudication approved an insufficiently reviewed alternative: {source_id}"
                            )
                        alternative = str(adjudicated_alternative)
                        alternative_status = "adjudicated-diverse-alternative"
                elif adjudication.get("decision") != "reject-all":
                    raise SystemExit(f"invalid adjudication decision: {source_id}")
                else:
                    resolved_reject = True
            if resolved_reject:
                continue
            if status is None and selected is None:
                exemplar = next(iter(candidates.values()))
                disagreements.append(
                    {
                        "source_id": source_id,
                        "source_language": exemplar["source_language"],
                        "target_language": exemplar["target_language"],
                        "source": exemplar["source"],
                        "licensed_reference": exemplar.get("licensed_reference"),
                        "review_a": left,
                        "review_b": right,
                        "candidates": [
                            {
                                "candidate_id": candidate_id,
                                "translation": candidate["translation"],
                            }
                            for candidate_id, candidate in candidates.items()
                        ],
                    }
                )
                continue
        if selected is not None and status is not None:
            alternative_record = None
            if alternative is not None and alternative_status is not None:
                if candidates[alternative]["translation"].strip() == candidates[selected]["translation"].strip():
                    raise SystemExit(f"approved alternative duplicates the canonical target: {source_id}")
                alternative_record = {
                    key: candidates[alternative].get(key)
                    for key in (
                        "candidate_id",
                        "translation",
                        "teacher_model",
                        "teacher_response_id",
                        "teacher_system_fingerprint",
                    )
                }
                alternative_record.update(
                    {
                        "review_status": alternative_status,
                        "reviewer_ids": reviewer_ids,
                    }
                )
            approved.append(
                {
                    **candidates[selected],
                    "review_status": status,
                    "reviewer_ids": reviewer_ids,
                    "source_level_reviews": [left, right],
                    "adjudication": adjudications.get(source_id),
                    "approved_alternative": alternative_record,
                }
            )

    unused_adjudications = set(adjudications) - used_adjudications
    if unused_adjudications:
        raise SystemExit(
            "adjudication supplied for a source without reviewer disagreement; first: "
            f"{next(iter(unused_adjudications))}"
        )

    approved.sort(key=lambda row: row["source_id"])
    disagreements.sort(key=lambda row: row["source_id"])
    write_jsonl(args.approved, approved)
    write_jsonl(args.disagreements, disagreements)
    print(
        json.dumps(
            {
                "approved": len(approved),
                "disagreements": len(disagreements),
                "sources": len(queue),
                "reviewers": [reviewer_a, reviewer_b],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
