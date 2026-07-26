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

distribution_misclaimed = copy.deepcopy(contract)
distribution_misclaimed["dataset"]["distribution_eligible"] = True
try:
    MODULE.validate(
        distribution_misclaimed,
        require_materialized_dataset=False,
        require_local_model_artifacts=False,
    )
except SystemExit as error:
    assert "dataset" in str(error)
else:
    raise AssertionError("unaudited distribution eligibility must fail")

non_immutable = copy.deepcopy(contract)
non_immutable["phase1_training"]["checkpoint_policy"][
    "immutable_model_and_tokenizer_at_every_scheduled_step"
] = False
try:
    MODULE.validate(
        non_immutable,
        require_materialized_dataset=False,
        require_local_model_artifacts=False,
    )
except SystemExit as error:
    assert "recipe" in str(error)
else:
    raise AssertionError("non-immutable scheduled checkpoints must fail")

wrong_selection = copy.deepcopy(contract)
wrong_selection["phase1_training"]["selection_artifact"]["sha256"] = "0" * 64
try:
    MODULE.validate(
        wrong_selection,
        require_materialized_dataset=False,
        require_local_model_artifacts=False,
    )
except SystemExit as error:
    assert "selection" in str(error)
else:
    raise AssertionError("unauthenticated selection must fail")

print("V18 shared-bidirectional contract checks passed")
