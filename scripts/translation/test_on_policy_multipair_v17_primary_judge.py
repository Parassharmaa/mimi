#!/usr/bin/env python3
"""Offline contract checks for V17's primary semantic audit."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from collect_on_policy_multipair_v17_primary_judge import validate_assessment
from prepare_on_policy_multipair_v17_primary_judge import (
    COMET_MODEL,
    COMET_PACKAGE_VERSION,
    COMET_REVISION,
    DEVELOPER_PROMPT,
    normalized_requests,
    opaque_id,
    request_row,
    validate_comet,
)
from run_claude_consensus_judge import read_requests


def must_fail(function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except SystemExit:
        return
    raise AssertionError("unsafe V17 primary judge fixture unexpectedly passed")


def main() -> None:
    pair_id = "v17-pair-test"
    source_id = opaque_id("source", pair_id)
    reference_id = opaque_id("candidate", pair_id, "licensed-reference")
    generated_id = opaque_id("candidate", pair_id, "generated-rollout")
    assert reference_id != generated_id
    source = {
        "source_id": source_id,
        "source_language": "en-US",
        "target_language": "ja-JP",
        "domain": "safety",
        "source": "Do not enter.",
        "candidates": [
            {"candidate_id": reference_id, "translation": "立入禁止。"},
            {"candidate_id": generated_id, "translation": "入ってください。"},
        ],
    }
    prompt_sha256 = hashlib.sha256(DEVELOPER_PROMPT.encode()).hexdigest()
    sonnet = request_row(source, "claude-sonnet-5", prompt_sha256)
    opus = request_row(source, "claude-opus-5", prompt_sha256)
    assert normalized_requests([sonnet]) == normalized_requests([opus])
    with tempfile.TemporaryDirectory() as directory:
        request_path = Path(directory) / "requests.jsonl"
        request_path.write_text(
            json.dumps(sonnet, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        parsed, contract = read_requests(request_path)
        assert parsed[0]["source"]["candidates"] == source["candidates"]
        assert contract["model"] == "claude-sonnet-5"

    pairs = [{"pair_id": pair_id}]
    manifest = {
        "outputs": {
            "comet_suite": {"sha256": "suite-sha"},
            "comet_engine_report": {"sha256": "engine-sha"},
        }
    }
    comet = {
        "metric": "COMET-22",
        "modelRepository": COMET_MODEL,
        "modelRevision": COMET_REVISION,
        "modelLicense": "Apache-2.0",
        "packageVersion": COMET_PACKAGE_VERSION,
        "precision": "float32",
        "suiteSHA256": "suite-sha",
        "engineReportSHA256": "engine-sha",
        "results": [{"caseID": pair_id, "score": 0.75}],
    }
    assert validate_comet(comet, pairs=pairs, manifest=manifest) == {
        pair_id: 0.75
    }
    must_fail(
        validate_comet,
        {**comet, "modelRevision": "moving-main"},
        pairs=pairs,
        manifest=manifest,
    )
    must_fail(
        validate_comet,
        {**comet, "results": []},
        pairs=pairs,
        manifest=manifest,
    )

    assessment = {
        "candidate_id": generated_id,
        "adequacy": 0,
        "fluent_and_complete": False,
        "critical_semantic_error": True,
        "error_tags": ["polarity"],
        "brief_evidence": "禁止 -> ください",
    }
    assert (
        validate_assessment(
            assessment,
            candidate_ids={reference_id, generated_id},
            source_id=source_id,
        )["error_tags"]
        == ["polarity"]
    )
    must_fail(
        validate_assessment,
        {**assessment, "candidate_id": "unknown"},
        candidate_ids={reference_id, generated_id},
        source_id=source_id,
    )
    print("V17 primary semantic judge contract checks passed")


if __name__ == "__main__":
    main()
