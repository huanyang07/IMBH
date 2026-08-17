from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / (
    "run_causal_inner_face36_fixed_q_operational_timestep_manifest_"
    "wp10c9d6c7c3b5c4f24e14u.py"
)
SPEC = importlib.util.spec_from_file_location("e14u_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_contract_freezes_first_doubled_step_only():
    contract = MODULE.CONTRACT
    assert contract["definitions_only"] is True
    assert contract["state"] == "primary_20ms"
    matched = contract["matched_endpoint"]
    assert matched["coarse_timestep_seconds"] == 2.0e-7
    assert matched["fine_timestep_seconds"] == 1.0e-7
    assert matched["fine_reference_roots"] == ["cold_1", "warm_1"]


def test_contract_preserves_root_and_physical_gates():
    gates = MODULE.CONTRACT["binding_gates"]
    assert gates["maximum_scaled_residual"] == 1.0e-10
    assert gates["minimum_path_reconstruction_factor"] == 1.0 - 1.0e-12
    assert gates["maximum_scaled_primitive_change"] == 5.0e-3
    assert all(MODULE.CONTRACT["hard_stops"].values())


def test_contract_binds_matched_endpoint_and_replay():
    matched = MODULE.CONTRACT["matched_endpoint"]
    assert matched["state_difference_relative_to_coarse_change_maximum"] == 0.1
    assert matched["reaction_action_relative_difference_maximum"] == 0.1
    assert MODULE.CONTRACT["solver_contract"]["bitwise_root_and_continuation_replay"]


def test_parents_authorize_manifest():
    parents = MODULE._validate_parents()
    assert parents["primary_summary"]["passed"]
    assert parents["heldout_summary"]["operational_timestep_manifest_authorized"]
    assert parents["fine_reference"]["cold_1"]["accepted"]
    assert parents["fine_reference"]["warm_1"]["accepted"]


def test_frozen_manifest_when_available():
    summary_path = MODULE.ARTIFACT_DIRECTORY / "summary.json"
    if not summary_path.exists():
        pytest.skip("prospective operational-timestep manifest is not frozen yet")
    summary = MODULE._read(summary_path)
    assert summary["operational_timestep_rung_2e7_execution_authorized"] is True
    assert summary["operational_timestep_rung_4e7_execution_authorized"] is False
    MODULE._checksums(MODULE.ARTIFACT_DIRECTORY)
