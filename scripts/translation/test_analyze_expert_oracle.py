#!/usr/bin/env python3
"""Contract tests for the reference-aware expert-oracle analyzer."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("analyze_expert_oracle.py")
SPEC = importlib.util.spec_from_file_location("analyze_expert_oracle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


assert (
    MODULE.group_id(
        "development-accuracy-v1:document:jlt:law-123:ja-en:segment-04"
    )
    == "development-accuracy-v1:document:jlt:law-123:ja-en"
)
assert (
    MODULE.group_id(
        "development-accuracy-v1:sentence:tatoeba:123:ja-en:segment-01"
    )
    == "development-accuracy-v1:sentence:tatoeba:123:ja-en:segment-01"
)

rows = [
    {"caseID": "development-accuracy-v1:document:a:segment-01", "oracleDelta": 1.0},
    {"caseID": "development-accuracy-v1:document:a:segment-02", "oracleDelta": 3.0},
    {"caseID": "development-accuracy-v1:document:b:segment-01", "oracleDelta": 5.0},
]
first = MODULE.bootstrap_grouped_mean(rows, samples=1_000, seed=7)
second = MODULE.bootstrap_grouped_mean(rows, samples=1_000, seed=7)
assert first == second
assert 1.0 <= first[0] <= first[1] <= 5.0
assert MODULE.parse_expert("formal=/tmp/report.json") == (
    "formal",
    Path("/tmp/report.json"),
)

print("expert oracle contracts passed")
