#!/usr/bin/env python3
"""Offline validation tests for the local multi-candidate judge contract."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path


SCRIPT = Path(__file__).with_name("run_local_consensus_judge.py")
SPEC = importlib.util.spec_from_file_location("run_local_consensus_judge", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def source() -> dict:
    return {
        "source_id": "s1",
        "source_language": "en-US",
        "target_language": "ja-JP",
        "domain": "general",
        "source": "Keep https://example.com unchanged.",
        "candidates": [
            {"candidate_id": "a", "translation": "https://example.com を変更しないでください。"},
            {"candidate_id": "b", "translation": "そのリンクを変更しないでください。"},
            {"candidate_id": "c", "translation": "変更しないでください。"},
        ],
    }


def assessment(candidate_id: str) -> list:
    return [
        candidate_id,
        4,
        4,
        4,
        candidate_id == "a",
        False,
        [] if candidate_id == "a" else ["omission"],
    ]


def expect_error(callback) -> None:
    try:
        callback()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def main() -> None:
    item = source()
    assert MODULE.validate_source(item, "s1") == item
    payload = {
        "source_id": "s1",
        "assessments": [assessment("c"), assessment("a"), assessment("b")],
    }
    parsed = MODULE.parse_payload(json.dumps(payload), item)
    assert [value["candidate_id"] for value in parsed["assessments"]] == ["a", "b", "c"]

    wrong_source = {**payload, "source_id": "s2"}
    expect_error(lambda: MODULE.parse_payload(json.dumps(wrong_source), item))
    corrected = MODULE.parse_payload(
        json.dumps(wrong_source),
        item,
        allow_source_id_correction=True,
    )
    assert corrected["source_id"] == "s1"
    assert corrected.pop("_mimi_source_id_corrected") is True
    duplicate = {**payload, "assessments": [assessment("a")] * 3}
    expect_error(lambda: MODULE.parse_payload(json.dumps(duplicate), item))
    invalid_tag = assessment("a")
    invalid_tag[-1] = [{"explanation": "not an allowed tag"}]
    tagged = {**payload, "assessments": [invalid_tag, assessment("b"), assessment("c")]}
    expect_error(lambda: MODULE.parse_payload(json.dumps(tagged), item))
    repetition = assessment("b")
    repetition[-1] = ["repetition"]
    aliased = MODULE.parse_payload(
        json.dumps(
            {
                **payload,
                "assessments": [assessment("a"), repetition, assessment("c")],
            }
        ),
        item,
    )
    assert aliased.pop("_mimi_normalized_error_tags") == 1
    assert aliased["assessments"][1]["error_tags"] == ["disfluency"]
    leaked = {**item, "source": "licensed-reference must stay hidden"}
    expect_error(lambda: MODULE.validate_source(leaked, "s1"))

    request = {
        "custom_id": "s1",
        "body": {
            "metadata": {"pipeline": MODULE.JUDGE_PIPELINE},
            "input": [
                {"role": "developer", "content": "judge"},
                {"role": "user", "content": json.dumps(item)},
            ],
        },
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "requests.jsonl"
        path.write_text(json.dumps(request) + "\n", encoding="utf-8")
        assert MODULE.read_requests(path)[0]["source"] == item

    print("local consensus judge contract tests passed")


if __name__ == "__main__":
    main()
