from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / (
    "run_causal_inner_face36_fixed_q_heldout_continuation_"
    "wp10c9d6c7c3b5c4f24e14t.py"
)
SPEC = importlib.util.spec_from_file_location("e14t_execution", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_root_accounting_counts_only_accepted_roots():
    metrics = {
        "cold_1": {
            "accepted": True,
            "maximum_reaction_ledger_relative_defect": 1.0e-16,
            "maximum_constraint_action_ledger_relative_defect": 2.0e-16,
        },
        "warm_1": {
            "accepted": False,
            "maximum_reaction_ledger_relative_defect": 3.0e-16,
            "maximum_constraint_action_ledger_relative_defect": 4.0e-16,
        },
    }
    accounting = MODULE._root_accounting(metrics)
    assert accounting["accepted_roots"] == ["cold_1"]
    assert accounting["rejected_roots"] == ["warm_1"]
    assert accounting["accepted_trajectory_horizon_seconds"] == 1.0e-7
    assert accounting["accepted_trajectory_cumulative_ledger"] == 2.0e-16
    assert accounting["planned_ladder_complete"] is False


def test_frozen_manifest_validation_when_available():
    if not MODULE.MANIFEST_DIRECTORY.exists():
        pytest.skip("prospective held-out manifest is not frozen yet")
    frozen = MODULE._validate_manifest()
    assert frozen["summary"]["heldout_continuation_execution_authorized"] is True


def test_canonical_heldout_result_when_available():
    path = MODULE.CANONICAL_DIRECTORY / "summary.json"
    if not path.exists():
        pytest.skip("prospective held-out execution has not completed")
    summary = MODULE._read(path)
    assert summary["classification"] in {
        "heldout_bounded_continuation_certified",
        "heldout_bounded_continuation_failed",
    }
    MODULE._checksums(MODULE.CANONICAL_DIRECTORY)
