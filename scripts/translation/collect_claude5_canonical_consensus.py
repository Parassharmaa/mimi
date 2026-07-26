#!/usr/bin/env python3
"""Collect verified Sonnet 5 + Opus 5 judgments into one canonical decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODELS = ("claude-sonnet-5", "claude-opus-5")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def artifact(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def ensure_priority(
    queue: Path,
    requests: Path,
    run_directory: Path,
    output: Path,
    judgments: Path,
) -> None:
    evidence = judgments.with_suffix(judgments.suffix + ".manifest.json")
    if judgments.is_file() and evidence.is_file():
        return
    if judgments.exists() or evidence.exists():
        raise SystemExit(f"incomplete existing judgment evidence: {judgments}")
    subprocess.run(
        [
            "python3",
            "scripts/translation/prioritize_distillation_judgments.py",
            str(queue),
            str(output),
            str(judgments),
            "--judge-requests",
            str(requests),
            "--claude-run-directory",
            str(run_directory),
        ],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("queue_manifest", type=Path)
    parser.add_argument("review_queue", type=Path)
    parser.add_argument("sonnet_requests", type=Path)
    parser.add_argument("sonnet_run_directory", type=Path)
    parser.add_argument("sonnet_output", type=Path)
    parser.add_argument("sonnet_judgments", type=Path)
    parser.add_argument("opus_requests", type=Path)
    parser.add_argument("opus_run_directory", type=Path)
    parser.add_argument("opus_output", type=Path)
    parser.add_argument("opus_judgments", type=Path)
    parser.add_argument("approved", type=Path)
    parser.add_argument("rejected", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    if args.result.exists():
        raise SystemExit(f"refusing to overwrite consensus result: {args.result}")
    if args.approved.exists() or args.rejected.exists():
        raise SystemExit("refusing to overwrite existing canonical consensus outputs")

    contract = load_json(args.contract)
    policy = contract.get("judgePolicy", {})
    gate = contract.get("goGateBeforeScaling", {})
    if (
        policy.get("requiredJudgeModelIds") != list(MODELS)
        or policy.get("actualCanonicalModelUsageRequiredPerShard") is not True
        or policy.get("fallbackModelAllowed") is not False
        or gate.get("minimumApprovedSources") != 120
        or gate.get("preferenceTrainingRequiresUnanimousParetoWinOverCurrentMimi")
        is not True
    ):
        raise SystemExit("Claude 5 consensus contract has unexpected policy")
    for script in contract.get("scripts", {}).values():
        path = Path(script["path"])
        if not path.is_absolute():
            path = ROOT / path
        if sha256(path) != script["sha256"]:
            raise SystemExit(f"contract-bound script has drifted: {path}")

    ensure_priority(
        args.review_queue,
        args.sonnet_requests,
        args.sonnet_run_directory,
        args.sonnet_output,
        args.sonnet_judgments,
    )
    ensure_priority(
        args.review_queue,
        args.opus_requests,
        args.opus_run_directory,
        args.opus_output,
        args.opus_judgments,
    )
    subprocess.run(
        [
            "python3",
            "scripts/translation/approve_canonical_quality_consensus.py",
            str(args.contract),
            str(args.queue_manifest),
            str(args.review_queue),
            str(args.sonnet_judgments),
            str(args.opus_judgments),
            str(args.approved),
            str(args.rejected),
        ],
        cwd=ROOT,
        check=True,
    )

    approved = rows(args.approved)
    rejected = rows(args.rejected)
    queue_sources = {
        str(row["source_id"]) for row in rows(args.review_queue)
    }
    if len(approved) + len(rejected) != len(queue_sources):
        raise SystemExit("canonical consensus does not cover the exact review queue")
    minimum = int(gate["minimumApprovedSources"])
    minimum_passed = len(approved) >= minimum
    judge_evidence = {}
    for model, judgments in (
        (MODELS[0], args.sonnet_judgments),
        (MODELS[1], args.opus_judgments),
    ):
        evidence = judgments.with_suffix(judgments.suffix + ".manifest.json")
        judge_evidence[model] = {
            "judgments": artifact(judgments),
            "identityEvidence": artifact(evidence),
        }
    result = {
        "schemaVersion": 1,
        "experiment": contract.get("experiment"),
        "status": (
            "absolute-consensus-scale-gate-passed"
            if minimum_passed
            else "absolute-consensus-scale-gate-rejected"
        ),
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract": artifact(args.contract),
        "reviewQueue": artifact(args.review_queue),
        "judges": judge_evidence,
        "outputs": {
            "approved": artifact(args.approved),
            "rejected": artifact(args.rejected),
        },
        "counts": {
            "sources": len(queue_sources),
            "approved": len(approved),
            "rejected": len(rejected),
            "approvedDomains": dict(
                sorted(Counter(str(row.get("domain", "unknown")) for row in approved).items())
            ),
        },
        "gates": {
            "minimumApprovedSources": {
                "required": minimum,
                "actual": len(approved),
                "passed": minimum_passed,
            }
        },
        "pairwiseDatasetConstructionAuthorized": minimum_passed,
        "trainingAuthorized": False,
        "protectedEvaluationAuthorized": False,
        "promotionAuthorized": False,
        "appChangeAuthorized": False,
        "publicUploadAuthorized": False,
        "excludedJudgeEvidenceUsed": False,
        "humanReviewersUsed": False,
        "reasoningTracesStored": False,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "approved": len(approved),
                "rejected": len(rejected),
                "minimum_required": minimum,
                "result": str(args.result),
                "result_sha256": sha256(args.result),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
