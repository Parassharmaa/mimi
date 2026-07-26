#!/usr/bin/env python3
"""Convert the pinned Qwen bilingual judge report to consensus-judge JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ERROR_TAG_MAP = {
    "omission": "omission",
    "addition": "addition",
    "mistranslation": "meaning-reversal",
    "negation": "negation",
    "entity": "named-entity",
    "number": "number-or-date",
    "tense-aspect": "meaning-reversal",
    "pronoun-role": "meaning-reversal",
    "register": "register",
    "unnatural": "disfluency",
    "source-copy": "meaning-reversal",
    "other": "meaning-reversal",
}
CRITICAL_TAGS = {"negation", "entity", "number"}


def rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("review_queue", type=Path)
    parser.add_argument("local_judge_report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists() and args.output.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")

    queue: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows(args.review_queue):
        source_id = str(row.get("source_id", "")).strip()
        candidate_id = str(row.get("candidate_id", "")).strip()
        if not source_id or not candidate_id or candidate_id in queue[source_id]:
            raise SystemExit(f"invalid queue candidate: {source_id}/{candidate_id}")
        queue[source_id][candidate_id] = row
    if not queue or any(len(candidates) not in {3, 4} for candidates in queue.values()):
        raise SystemExit("every source must have three or four candidates")

    report = json.loads(args.local_judge_report.read_text(encoding="utf-8"))
    if (
        report.get("claimEligible") is not False
        or not report.get("judgeModel")
        or not report.get("judgeRevision")
    ):
        raise SystemExit("invalid local bilingual judge report")
    judge_model = f"{report['judgeModel']}@{report['judgeRevision']}"
    expected_ids = {
        candidate_id
        for candidates in queue.values()
        for candidate_id in candidates
    }
    judgments: dict[str, dict] = {}
    for result in report.get("results", []):
        candidate_id = str(result.get("candidateID", "")).strip()
        if (
            not candidate_id
            or candidate_id in judgments
            or candidate_id not in expected_ids
        ):
            raise SystemExit(f"invalid local judgment candidate: {candidate_id}")
        candidate = next(
            value[candidate_id]
            for value in queue.values()
            if candidate_id in value
        )
        if (
            result.get("source") != candidate["source"]
            or result.get("candidate") != candidate["translation"]
        ):
            raise SystemExit(f"local judgment text mismatch: {candidate_id}")
        judgments[candidate_id] = result["judgment"]
    if set(judgments) != expected_ids:
        raise SystemExit("local report does not cover the exact candidate queue")

    output_rows = []
    for source_id, candidates in sorted(queue.items()):
        assessments = []
        for candidate_id, candidate in sorted(candidates.items()):
            judgment = judgments[candidate_id]
            raw_tags = [str(tag) for tag in judgment.get("error_tags", [])]
            mapped_tags = list(
                dict.fromkeys(ERROR_TAG_MAP.get(tag, "meaning-reversal") for tag in raw_tags)
            )
            adequacy = max(0, min(4, int(judgment["adequacy"]) - 1))
            fluency = max(0, min(4, int(judgment["fluency"]) - 1))
            terminology = (
                0
                if "terminology" in raw_tags
                else adequacy
            )
            protected = (
                bool(candidate.get("critical_token_admission"))
                and not {"number", "entity"} & set(raw_tags)
            )
            critical = bool(judgment.get("critical_error")) or bool(
                CRITICAL_TAGS & set(raw_tags)
            )
            assessments.append(
                {
                    "candidate_id": candidate_id,
                    "adequacy": adequacy,
                    "fluency": fluency,
                    "terminology": terminology,
                    "protected_tokens_preserved": protected,
                    "critical_error": critical,
                    "error_tags": mapped_tags,
                }
            )
        response_id = "qwen-local-" + hashlib.sha256(
            (
                source_id
                + "\0"
                + judge_model
                + "\0"
                + json.dumps(assessments, sort_keys=True)
            ).encode()
        ).hexdigest()[:20]
        output_rows.append(
            {
                "source_id": source_id,
                "priority_status": "automated-review-order-only-not-approval",
                "judge_model": judge_model,
                "judge_response_id": response_id,
                "judge_system_fingerprint": report.get("systemPromptSHA256"),
                "assessments": assessments,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x", encoding="utf-8") as handle:
            for row in output_rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
    except FileExistsError as error:
        raise SystemExit(f"refusing to overwrite existing output: {args.output}") from error
    summary = {
        "sources": len(output_rows),
        "candidates": len(expected_ids),
        "judge_model": judge_model,
        "review_queue_sha256": sha256(args.review_queue),
        "local_report_sha256": sha256(args.local_judge_report),
        "output_sha256": sha256(args.output),
        "conversion": {
            "score_mapping": "local 1-5 scores minus one to consensus 0-4",
            "terminology": "adequacy unless the local terminology tag is present",
            "protected": (
                "deterministic admission plus no local number/entity error"
            ),
            "error_tags": ERROR_TAG_MAP,
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
