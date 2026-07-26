#!/usr/bin/env python3
"""Fast contract tests for V18's frozen stratified selector."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "build_shared_bidirectional_v18_selection",
    ROOT / "scripts/translation/build_shared_bidirectional_v18_selection.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


rows = []
for direction, quotas in MODULE.QUOTAS.items():
    for stratum, count in quotas.items():
        for index in range(count + 2):
            domain = stratum
            source = f"{direction}-{stratum}-source-{index}"
            target = f"{direction}-{stratum}-target-{index}"
            if stratum.startswith("wikipedia-"):
                domain = "wikipedia"
                if stratum.endswith("long-pair"):
                    source += "x" * MODULE.LONG_PAIR_CHARACTERS
            rows.append(
                {
                    "id": f"{direction}:{stratum}:{index}",
                    "direction": direction,
                    "domain": domain,
                    "source": source,
                    "target": target,
                }
            )

first = MODULE.select(rows, MODULE.SEED)
second = MODULE.select(list(reversed(rows)), MODULE.SEED)
assert [row["id"] for row in first] == [row["id"] for row in second]
assert len(first) == 512
assert len({row["id"] for row in first}) == 512
for direction, quotas in MODULE.QUOTAS.items():
    current = [row for row in first if row["direction"] == direction]
    assert len(current) == 256
    for stratum, count in quotas.items():
        assert sum(row["selection_stratum"] == stratum for row in current) == count

print("V18 stratified selection contracts passed")
