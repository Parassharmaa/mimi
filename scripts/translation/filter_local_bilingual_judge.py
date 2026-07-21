#!/usr/bin/env python3
"""Admit only strict accepts from a pinned local bilingual judge report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


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
    parser.add_argument("candidates", type=Path)
    parser.add_argument("judge_report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists() and args.output.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")

    candidates = {str(row["id"]): row for row in rows(args.candidates)}
    report = json.loads(args.judge_report.read_text(encoding="utf-8"))
    if report.get("claimEligible") is not False or not report.get("judgeModel"):
        raise SystemExit("invalid local judge report")
    judgments = {}
    for result in report.get("results", []):
        identifier = str(result.get("candidateID", ""))
        if not identifier or identifier in judgments or identifier not in candidates:
            raise SystemExit("judge report has missing, duplicate, or unknown candidate ID")
        if result.get("source") != candidates[identifier]["source"] or result.get("candidate") != candidates[identifier]["target"]:
            raise SystemExit(f"judge report text mismatch: {identifier}")
        judgments[identifier] = result["judgment"]
    if set(judgments) != set(candidates):
        raise SystemExit("judge report does not cover the exact candidate set")

    accepted = []
    for identifier, row in candidates.items():
        judgment = judgments[identifier]
        strict = (
            judgment == {
                "adequacy": 5,
                "fluency": 5,
                "meaning_preserved": True,
                "critical_error": False,
                "error_tags": [],
                "verdict": "accept",
            }
        )
        if strict:
            accepted.append({
                **row,
                "local_judge": {
                    "model": report["judgeModel"],
                    "revision": report["judgeRevision"],
                    "license": report["judgeLicense"],
                    "system_prompt_sha256": report["systemPromptSHA256"],
                    "judgment": judgment,
                },
                "review_status": "local-multimodel-plus-bilingual-judge-provisional",
                "promotion_eligible": False,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in accepted),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "purpose": "reviewer-free local teacher pilot; never promotion evidence",
        "promotion_eligible": False,
        "counts": {"candidates": len(candidates), "accepted": len(accepted)},
        "judge": {
            "model": report["judgeModel"],
            "revision": report["judgeRevision"],
            "license": report["judgeLicense"],
            "strict_policy": "adequacy=5; fluency=5; meaning preserved; no critical/error tags",
        },
        "inputs": {
            "candidates": {"path": str(args.candidates.resolve()), "sha256": sha256(args.candidates)},
            "judge_report": {"path": str(args.judge_report.resolve()), "sha256": sha256(args.judge_report)},
        },
        "output": {"path": str(args.output.resolve()), "sha256": sha256(args.output)},
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
