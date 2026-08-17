from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / (
    "run_causal_inner_face36_fixed_q_operational_timestep_predictor_manifest_"
    "wp10c9d6c7c3b5c4f24e14w.py"
)
SPEC = importlib.util.spec_from_file_location("e14w_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_predictor_repair_changes_no_scientific_gate():
    contract = MODULE.CONTRACT
    assert contract["supersedes_before_nonlinear_root"] == "WP10c9d6c7c3b5c4f24e14u"
    assert contract["binding_gates"] == MODULE.base.CONTRACT["binding_gates"]
    assert contract["matched_endpoint"] == MODULE.base.CONTRACT["matched_endpoint"]


def test_predictor_is_explicit_and_within_bound():
    solver = MODULE.CONTRACT["solver_contract"]
    assert solver["initial_predictor"] == "previous_accepted_scaled_primitive_increment"
    assert solver["last_rate_extrapolation_forbidden"] is True
    assert solver["previous_accepted_scaled_increment_maximum"] < solver["maximum_scaled_primitive_change"]


def test_parents_preserve_pre_root_failure():
    parents = MODULE._validate_parents()
    failure = parents["pre_root_execution_failure"]
    assert failure["nonlinear_root_solved"] is False
    assert failure["trajectory_horizon_seconds_added"] == 0.0
    assert failure["unbounded_predictor_maximum"] > failure["bound"]


def test_frozen_manifest_when_available():
    path = MODULE.ARTIFACT_DIRECTORY / "summary.json"
    if not path.exists():
        pytest.skip("predictor-repair manifest is not frozen yet")
    summary = MODULE.base._read(path)
    assert summary["operational_timestep_rung_2e7_execution_authorized"]
    MODULE.base._checksums(MODULE.ARTIFACT_DIRECTORY)
