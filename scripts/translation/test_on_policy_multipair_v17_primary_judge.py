#!/usr/bin/env python3
"""Offline contract checks for V17's primary semantic audit."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from collect_on_policy_multipair_v17_primary_judge import validate_assessment
from prepare_on_policy_multipair_v17_primary_judge import (
    COMET_MODEL,
    COMET_PACKAGE_VERSION,
    COMET_REVISION,
    DEVELOPER_PROMPT,
    SCHEMA,
    normalized_requests,
    opaque_id,
    request_row,
    validate_comet,
)
from run_claude_consensus_judge import read_requests, validate_payload


def must_fail(function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except (SystemExit, ValueError):
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
        work = Path(directory)
        request_path = Path(directory) / "requests.jsonl"
        second_pair = "v17-pair-test-2"
        second_source = {
            **source,
            "source_id": opaque_id("source", second_pair),
            "source": "Please wait.",
            "candidates": [
                {
                    "candidate_id": opaque_id(
                        "candidate", second_pair, "licensed-reference"
                    ),
                    "translation": "お待ちください。",
                },
                {
                    "candidate_id": opaque_id(
                        "candidate", second_pair, "generated-rollout"
                    ),
                    "translation": "行ってください。",
                },
            ],
        }
        second_request = request_row(
            second_source,
            "claude-sonnet-5",
            prompt_sha256,
        )
        request_path.write_text(
            "".join(
                json.dumps(value, ensure_ascii=False) + "\n"
                for value in (sonnet, second_request)
            ),
            encoding="utf-8",
        )
        parsed, contract = read_requests(request_path)
        assert parsed[0]["source"]["candidates"] == source["candidates"]
        assert len(parsed) == 2
        assert contract["model"] == "claude-sonnet-5"
        fake_claude = work / "fake-claude"
        fake_claude.write_text(
            """#!/usr/bin/env python3
import json
import sys

model = sys.argv[sys.argv.index("--model") + 1]
sources = json.loads(sys.argv[-1].split("Blinded judge inputs:\\n", 1)[1])
results = []
for source in sources:
    assessments = [
        {
            "candidate_id": candidate["candidate_id"],
            "adequacy": 4 if index == 0 else 1,
            "fluent_and_complete": index == 0,
            "critical_semantic_error": index != 0,
            "error_tags": [] if index == 0 else ["other-critical"],
            "brief_evidence": "none" if index == 0 else "mismatch",
        }
        for index, candidate in enumerate(source["candidates"])
    ]
    results.append(
        {
            "source_id": source["source_id"],
            "assessments": assessments,
            "preferred_candidate_ids": [source["candidates"][0]["candidate_id"]],
        }
    )
print(json.dumps({
    "structured_output": {"results": results},
    "modelUsage": {
        model: {
            "canonicalModel": model,
            "outputTokens": 50,
            "provider": "firstParty",
        }
    },
    "duration_api_ms": 1,
    "duration_ms": 1,
    "total_cost_usd": 0,
    "usage": {},
}))
""",
            encoding="utf-8",
        )
        fake_claude.chmod(0o755)
        run_directory = work / "run"
        output = work / "output.jsonl"
        completed = subprocess.run(
            [
                "python3",
                str(
                    Path(__file__).with_name(
                        "run_claude_consensus_judge.py"
                    )
                ),
                str(request_path),
                str(run_directory),
                str(output),
                "--maximum-items",
                "1",
                "--parallelism",
                "2",
                "--claude-executable",
                str(fake_claude),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert len(output.read_text(encoding="utf-8").splitlines()) == 2
        run_manifest = json.loads(
            (run_directory / "manifest.json").read_text(encoding="utf-8")
        )
        assert run_manifest["shard_count"] == 2
        assert all(
            (run_directory / "shards" / f"{index:05d}.metadata.json").is_file()
            for index in range(2)
        )

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
    payload = {
        "source_id": source_id,
        "assessments": [
            {
                "candidate_id": reference_id,
                "adequacy": 4,
                "fluent_and_complete": True,
                "critical_semantic_error": False,
                "error_tags": [],
                "brief_evidence": "none",
            },
            assessment,
        ],
        "preferred_candidate_ids": [reference_id],
    }
    assert validate_payload(payload, source, SCHEMA) == payload
    must_fail(
        validate_payload,
        {**payload, "preferred_candidate_ids": ["unknown"]},
        source,
        SCHEMA,
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
