#!/usr/bin/env python3
"""Static contracts for the V18 validation cache parity harness."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
source = (
    ROOT / "scripts/translation/verify_marian_validation_cache_parity.py"
).read_text(encoding="utf-8")
tree = ast.parse(source)
calls = [
    node
    for node in ast.walk(tree)
    if isinstance(node, ast.Call)
    and isinstance(node.func, ast.Name)
    and node.func.id == "evaluate"
]
assert len(calls) == 2
policies = []
for call in calls:
    values = {
        keyword.arg: keyword.value
        for keyword in call.keywords
        if keyword.arg is not None
    }
    assert isinstance(values["loss_use_cache"], ast.Constant)
    assert values["loss_use_cache"].value is False
    assert isinstance(values["generation_use_cache"], ast.Constant)
    policies.append(values["generation_use_cache"].value)
    assert isinstance(values["return_generated_token_ids"], ast.Constant)
    assert values["return_generated_token_ids"].value is True
assert policies == [False, True]

print("Marian validation cache parity harness contracts passed")
