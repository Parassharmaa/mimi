#!/usr/bin/env python3
"""Offline checks for the frozen V18 phase-one contract."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = (
    ROOT
    / "Research/translation/shared-bidirectional-v18-phase1-contract-2026-07-26.json"
)
MODULE_PATH = (
    ROOT / "scripts/translation/validate_shared_bidirectional_v18_contract.py"
)
SPEC = importlib.util.spec_from_file_location("validate_v18_contract", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
MODULE.validate(
    contract,
    require_materialized_dataset=False,
    require_local_model_artifacts=False,
)

weakened = copy.deepcopy(contract)
weakened["evaluation_gates"]["promotion"]["requires_zero_critical_error_union"] = False
try:
    MODULE.validate(
        weakened,
        require_materialized_dataset=False,
        require_local_model_artifacts=False,
    )
except SystemExit as error:
    assert "gates" in str(error)
else:
    raise AssertionError("weakened promotion gate must fail")

unauthorized = copy.deepcopy(contract)
unauthorized["training_authorized"] = False
try:
    MODULE.validate(
        unauthorized,
        require_materialized_dataset=False,
        require_local_model_artifacts=False,
    )
except SystemExit as error:
    assert "authorization" in str(error)
else:
    raise AssertionError("missing phase-one authorization must fail")

print("V18 shared-bidirectional contract checks passed")
