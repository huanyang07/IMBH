from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / (
    "run_causal_inner_face36_fixed_q_operational_timestep_rung_"
    "wp10c9d6c7c3b5c4f24e14v.py"
)
SPEC = importlib.util.spec_from_file_location("e14v_rung", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_policy_is_cold_variable_step_control():
    policy = MODULE._policy("coarse_2e7")
    assert policy["cold"] is True
    assert policy["initial_exact_jacobian_required"] is True
    assert policy["maximum_exact_jacobian_refreshes"] == 2
    assert policy["use_carried_solver_state"] is False


def test_manifest_validation_when_frozen():
    if not MODULE.MANIFEST_DIRECTORY.exists():
        pytest.skip("prospective operational-timestep manifest is not frozen yet")
    frozen = MODULE._validate_manifest()
    assert frozen["summary"]["operational_timestep_rung_2e7_execution_authorized"]
    assert frozen["contract"]["matched_endpoint"]["coarse_timestep_seconds"] == 2.0e-7


def test_canonical_result_when_available():
    summary_path = MODULE.CANONICAL_DIRECTORY / "summary.json"
    if not summary_path.exists():
        pytest.skip("prospective operational-timestep rung has not executed")
    summary = MODULE._read(summary_path)
    metrics = MODULE._read(MODULE.CANONICAL_DIRECTORY / "metrics.json")
    assert summary["passed"] == metrics["scientific_passed"]
    if summary["passed"]:
        assert metrics["coarse_root"]["accepted"]
        assert metrics["matched_endpoint"]["passed"]
        assert metrics["replay"]["passed"]
    MODULE._checksums(MODULE.CANONICAL_DIRECTORY)
