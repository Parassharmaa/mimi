#!/usr/bin/env python3
"""Fast contracts for V18 full-precision internal evaluation."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from evaluate_shared_bidirectional_v18_internal import (
    DIRECTIONS,
    SUITE_ORDER,
    aggregate,
    authenticate_checkpoint,
    checkpoint_decision,
    computed_failures,
    direction,
    normalized_row,
)

ROOT = Path(__file__).resolve().parents[2]
EVALUATION_CONTRACT = (
    ROOT
    / "Research/translation/shared-bidirectional-v18-internal-evaluation-contract-2026-07-26.json"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert direction(
    {"id": "a", "sourceLanguage": "en-US", "targetLanguage": "ja-JP"}
) == "en-ja"
assert direction({"id": "b", "direction": "ja-en"}) == "ja-en"
normalized = normalized_row(
    {
        "id": "c",
        "direction": "en-ja",
        "source": "Open {FILE}.",
        "target": "{FILE}を開きます。",
    }
)
assert normalized["references"] == ["{FILE}を開きます。"]
assert normalized["segments"] == ["Open {FILE}."]
assert normalized["sourceLanguage"] == "en-US"

generated = {
    "c": {
        "id": "c",
        "direction": "en-ja",
        "sourceLanguage": "en-US",
        "targetLanguage": "ja-JP",
        "source": "Open {FILE}.",
        "source_segments": ["Open {FILE}."],
        "references": ["{FILE}を開きます。"],
        "hypothesis": "{FILE}を開きます。",
        "segment_hypotheses": ["{FILE}を開きます。"],
        "segment_input_token_ids": [[1]],
        "segment_output_token_ids": [[2]],
        "source_truncated": False,
        "reached_generation_limit": False,
        "repeated_token_loop": False,
        "empty_segment": False,
        "adjacent_duplicate_segment": False,
    }
}
metrics = aggregate([normalized], generated)
assert metrics["cases"] == 1
assert metrics["chrFPlusPlus"] == 100.0
assert all(not values for values in metrics["failures"].values())

bad = {
    **generated["c"],
    "hypothesis": "",
    "empty_segment": True,
}
failures = computed_failures([bad])
assert failures["generation"] == ["c"]
assert failures["exact"] == []

swapped = {
    **generated["c"],
    "id": "swapped",
    "source": "Pay 10.\nPay 20.",
    "source_segments": ["Pay 10.", "Pay 20."],
    "hypothesis": "20を支払う。\n10を支払う。",
    "segment_hypotheses": ["20を支払う。", "10を支払う。"],
}
swapped_failures = computed_failures([swapped])
assert swapped_failures["exact"] == ["swapped"]
assert swapped_failures["typed"] == ["swapped"]


def metric(chrf: float) -> dict:
    return {
        "cases": 1,
        "chrFPlusPlus": chrf,
        "BLEU": 1.0,
        "meanSentenceChrFPlusPlus": chrf,
        "sentenceChrFPlusPlus": {"case": chrf},
        "failures": {
            "exact": [],
            "typed": [],
            "negation": [],
            "generation": [],
        },
        "outputs": {},
    }


teachers = {
    suite: {
        direction_name: metric(31.0 if direction_name == "en-ja" else 51.0)
        for direction_name in DIRECTIONS
    }
    for suite in SUITE_ORDER
}
candidate = {
    suite: {
        direction_name: metric(31.0 if direction_name == "en-ja" else 51.0)
        for direction_name in DIRECTIONS
    }
    for suite in SUITE_ORDER
}
requirements = {
    "minimum_chrf_pp": {"en-ja": 30.0, "ja-en": 50.0},
    "minimum_macro_direction_chrf_pp": 40.0,
    "maximum_directional_chrf_pp_regression_from_teacher": 1.0,
}
regression_suites = list(SUITE_ORDER[1:])
gates, decision = checkpoint_decision(
    candidate,
    teachers,
    requirements,
    regression_suites,
)
assert all(item["passed"] for item in gates)
assert decision["selector_macro_direction_chrf_pp"] == 41.0

candidate["legal-safety-test-v1"]["ja-en"]["failures"]["negation"] = ["case"]
gates, decision = checkpoint_decision(
    candidate,
    teachers,
    requirements,
    regression_suites,
)
assert not all(item["passed"] for item in gates)
assert decision["new_failures"]["legal-safety-test-v1"]["ja-en"]["negation"] == [
    "case"
]

with tempfile.TemporaryDirectory(prefix="mimi-v18-eval-") as temporary:
    checkpoint = Path(temporary)
    weights = checkpoint / "model.safetensors"
    weights.write_bytes(b"authenticated fixture")
    config = checkpoint / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "experiment": "shared-bidirectional-v18-wide-dense-phase1",
        "step": 250,
        "contract_sha256": "a" * 64,
        "immutable_scheduled_checkpoint": True,
        "inference_config_use_cache": True,
        "files": {
            "config.json": {
                "bytes": config.stat().st_size,
                "sha256": digest(config),
            },
            "model.safetensors": {
                "bytes": weights.stat().st_size,
                "sha256": digest(weights),
            },
        },
    }
    (checkpoint / "mimi_checkpoint_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    assert authenticate_checkpoint(
        checkpoint,
        step=250,
        contract_sha256="a" * 64,
    )["step"] == 250
    weights.write_bytes(b"tampered fixture")
    try:
        authenticate_checkpoint(
            checkpoint,
            step=250,
            contract_sha256="a" * 64,
        )
    except SystemExit:
        pass
    else:
        raise AssertionError("tampered checkpoint unexpectedly authenticated")

evaluation_contract = json.loads(EVALUATION_CONTRACT.read_text(encoding="utf-8"))
assert evaluation_contract["implementation"]["sha256"] == digest(
    ROOT / evaluation_contract["implementation"]["path"]
)
assert evaluation_contract["training_contract"]["sha256"] == digest(
    ROOT / evaluation_contract["training_contract"]["path"]
)
assert evaluation_contract["observed_before_evaluation_freeze"][
    "maximum_observed_step"
] == 250
assert evaluation_contract["required_regression_suites"] == regression_suites
assert evaluation_contract["runtime"]["batch_size"] == 4
assert evaluation_contract["semantic_failure_behavior"].startswith(
    "stop-experiment"
)
for suite in evaluation_contract["suites"].values():
    suite_path = ROOT / suite["path"]
    assert suite["sha256"] == digest(suite_path)
    assert suite["cases"] == sum(
        bool(line.strip())
        for line in suite_path.read_text(encoding="utf-8").splitlines()
    )
for direction_name, teacher in evaluation_contract["teachers"].items():
    directory = (
        ROOT
        / json.loads(
            (
                ROOT
                / evaluation_contract["training_contract"]["path"]
            ).read_text(encoding="utf-8")
        )["teachers"][direction_name]["path"]
    )
    assert set(teacher["files"]) == {
        path.name for path in directory.iterdir() if path.is_file()
    }
    for name, item in teacher["files"].items():
        path = directory / name
        assert item["bytes"] == path.stat().st_size
        assert item["sha256"] == digest(path)

print("V18 full-precision evaluator contracts passed.")
