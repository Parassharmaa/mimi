#!/usr/bin/env python3
"""Freeze the Sonnet 5 + Opus 5 admission contract before judging."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REQUIRED_MODELS = ("claude-sonnet-5", "claude-opus-5")
CONTRACT_SCRIPTS = (
    "scripts/translation/prepare_claude5_dual_judge_contract.py",
    "scripts/translation/prepare_distillation_judge_batch.py",
    "scripts/translation/run_claude_consensus_judge.py",
    "scripts/translation/prioritize_distillation_judgments.py",
    "scripts/translation/approve_canonical_quality_consensus.py",
    "scripts/translation/build_canonical_pairwise_preference_dataset.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def request_contract(path: Path, expected_model: str) -> dict:
    values = rows(path)
    if not values:
        raise SystemExit(f"empty judge request file: {path}")
    custom_ids: list[str] = []
    normalized: list[dict] = []
    prompt_hashes: set[str] = set()
    for value in values:
        custom_id = str(value.get("custom_id", "")).strip()
        body = value.get("body")
        if (
            not custom_id
            or custom_id in custom_ids
            or value.get("method") != "POST"
            or value.get("url") != "/v1/responses"
            or not isinstance(body, dict)
            or body.get("model") != expected_model
            or body.get("store") is not False
            or body.get("metadata", {}).get("pipeline")
            != "mimi-translation-judge-v1"
        ):
            raise SystemExit(f"invalid {expected_model} request: {custom_id}")
        visible = str(body.get("input", [{}, {}])[1].get("content", ""))
        if any(
            forbidden in visible
            for forbidden in (
                "candidate_origin",
                "licensed-reference",
                "current-mimi-baseline",
                "teacher_model",
            )
        ):
            raise SystemExit(f"candidate origin leaked to judge: {custom_id}")
        custom_ids.append(custom_id)
        prompt_hashes.add(str(body["metadata"].get("prompt_sha256", "")))
        normalized_body = {**body, "model": "<frozen-judge-model>"}
        normalized.append({**value, "body": normalized_body})
    if len(prompt_hashes) != 1 or "" in prompt_hashes:
        raise SystemExit(f"non-uniform judge prompt: {path}")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "model": expected_model,
        "requests": len(values),
        "custom_ids": custom_ids,
        "prompt_sha256": next(iter(prompt_hashes)),
        "model_independent_payload_sha256": hashlib.sha256(
            canonical(normalized)
        ).hexdigest(),
    }


def artifact(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256(path)}


def script_artifact(relative: str) -> dict:
    path = ROOT / relative
    return {"path": relative, "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scale_contract", type=Path)
    parser.add_argument("queue_manifest", type=Path)
    parser.add_argument("review_queue", type=Path)
    parser.add_argument("sonnet_requests", type=Path)
    parser.add_argument("opus_requests", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--supersedes-contract", type=Path)
    args = parser.parse_args()
    if args.output.exists() and args.output.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")

    scale = load(args.scale_contract)
    queue_manifest = load(args.queue_manifest)
    admission = scale.get("admission", {})
    if (
        admission.get("human_reviewers_required") is not False
        or admission.get("independent_judges_required") != 2
        or admission.get("minimum_approved") != 120
        or admission.get(
            "unanimous_pareto_preference_over_current_required_for_preference_training"
        )
        is not True
        or queue_manifest.get("candidateOriginExposedToJudges") is not False
        or queue_manifest.get("output", {}).get("sha256") != sha256(args.review_queue)
    ):
        raise SystemExit("scale or review-queue admission contract is not satisfied")

    queue_source_ids = sorted(
        {
            str(row.get("source_id", "")).strip()
            for row in rows(args.review_queue)
            if str(row.get("source_id", "")).strip()
        }
    )
    if len(queue_source_ids) != queue_manifest.get("sources"):
        raise SystemExit("review queue source coverage does not match its manifest")

    sonnet = request_contract(args.sonnet_requests, REQUIRED_MODELS[0])
    opus = request_contract(args.opus_requests, REQUIRED_MODELS[1])
    if (
        sonnet["custom_ids"] != opus["custom_ids"]
        or sorted(sonnet["custom_ids"]) != queue_source_ids
        or sonnet["prompt_sha256"] != opus["prompt_sha256"]
        or sonnet["model_independent_payload_sha256"]
        != opus["model_independent_payload_sha256"]
    ):
        raise SystemExit("Sonnet 5 and Opus 5 requests are not the same blinded batch")

    pilot_seeds = queue_manifest.get("inputs", {}).get("pilotSeeds", {})
    supersedes = None
    if args.supersedes_contract:
        prior = load(args.supersedes_contract)
        prior_requests = prior.get("judgeRequests", {})
        for model, request in (
            (REQUIRED_MODELS[0], sonnet),
            (REQUIRED_MODELS[1], opus),
        ):
            if prior_requests.get(model, {}).get("sha256") != request["sha256"]:
                raise SystemExit("superseded contract used a different judge request")
        if prior.get("goGateBeforeScaling") != {
            "acceptancePolicy": "dual-absolute-canonical",
            "requireBothJudgesCertifyCanonicalAbsoluteThresholds": True,
            "requireZeroDeterministicCriticalFailures": True,
            "trainingAuthorizedByPilotAlone": False,
            "minimumApprovedSources": 120,
            "preferenceTrainingRequiresUnanimousParetoWinOverCurrentMimi": True,
        }:
            raise SystemExit("superseded contract used different admission thresholds")
        supersedes = {
            **artifact(args.supersedes_contract),
            "changeScope": (
                "fail-closed judgment evidence propagation only; judge requests, "
                "models, thresholds, and candidate data are unchanged"
            ),
            "frozenBeforeCompleteCollectionAndBeforeJudgmentContentInspection": True,
        }

    contract = {
        "schemaVersion": 1,
        "experiment": scale.get("experiment"),
        "status": "frozen-before-complete-collection-and-content-inspection",
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "inputs": {
            "scaleContract": artifact(args.scale_contract),
            "reviewQueueManifest": artifact(args.queue_manifest),
            "reviewQueue": artifact(args.review_queue),
        },
        "outputs": {"pilotSeeds": pilot_seeds},
        "judgePolicy": {
            "requiredJudgeModelIds": list(REQUIRED_MODELS),
            "sameBlindedBatchRequired": True,
            "actualCanonicalModelUsageRequiredPerShard": True,
            "fallbackModelAllowed": False,
            "humanReviewersRequired": False,
            "excludedFromAdmissionAndTrainingSelection": [
                "all local Qwen judge outputs",
                "all Claude Fable judge outputs",
            ],
            "excludedArtifactsRetainedForAuditOnly": True,
        },
        "judgeRequests": {
            "claude-sonnet-5": {
                key: value
                for key, value in sonnet.items()
                if key != "custom_ids"
            },
            "claude-opus-5": {
                key: value
                for key, value in opus.items()
                if key != "custom_ids"
            },
        },
        "goGateBeforeScaling": {
            "acceptancePolicy": "dual-absolute-canonical",
            "requireBothJudgesCertifyCanonicalAbsoluteThresholds": True,
            "requireZeroDeterministicCriticalFailures": True,
            "trainingAuthorizedByPilotAlone": False,
            "minimumApprovedSources": 120,
            "preferenceTrainingRequiresUnanimousParetoWinOverCurrentMimi": True,
        },
        "absoluteThresholds": {
            "minimumAdequacy": 4,
            "minimumFluency": 3,
            "minimumTerminology": 3,
            "requireProtectedTokensPreserved": True,
            "requireNoCriticalError": True,
            "requireNoErrorTags": True,
        },
        "scripts": {
            relative: script_artifact(relative)
            for relative in CONTRACT_SCRIPTS
        },
        "reasoningTraceStored": False,
        "candidateOriginExposedToJudges": False,
        "trainingAuthorized": False,
        "protectedEvaluationAuthorized": False,
        "appChangeAuthorized": False,
        "publicUploadAuthorized": False,
    }
    if supersedes:
        contract["supersedes"] = supersedes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(contract, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise SystemExit(f"refusing to overwrite existing output: {args.output}") from error
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": sha256(args.output),
                "sources": len(queue_source_ids),
                "required_models": list(REQUIRED_MODELS),
                "excluded_judges": ["local Qwen", "Claude Fable"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
