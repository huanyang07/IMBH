from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / (
    "run_causal_inner_face36_fixed_q_heldout_continuation_manifest_"
    "wp10c9d6c7c3b5c4f24e14s.py"
)
SPEC = importlib.util.spec_from_file_location("e14s_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_contract_is_short_heldout_continuation():
    contract = MODULE.CONTRACT
    assert contract["definitions_only"] is True
    assert contract["state"] == "heldout_16ms"
    assert contract["timestep_seconds"] == 1.0e-7
    assert contract["trajectory"]["root_order"] == ["cold_1", "warm_1", "warm_2"]
    assert contract["trajectory"]["replay_roots"] == ["warm_2"]


def test_contract_reuses_certified_warm_policy():
    solver = MODULE.CONTRACT["solver_contract"]
    assert solver["warm_carried_matrix_at_iteration_zero"] is True
    assert solver["warm_maximum_exact_assemblies"] == 1
    assert solver["warm_iteration_reserve_trigger"] == 6
    assert solver["warm_failed_relative_backtrack_trigger"] == 4


def test_contract_preserves_scientific_gates_and_hard_stops():
    gates = MODULE.CONTRACT["binding_gates"]
    assert gates["maximum_scaled_residual"] == 1.0e-10
    assert gates["minimum_path_reconstruction_factor"] == 1.0 - 1.0e-12
    assert gates["maximum_cumulative_absolute_ledger_defect"] == 3.0e-12
    assert all(MODULE.CONTRACT["hard_stops"].values())


def test_parent_authorizes_manifest_and_seed_is_accepted():
    parents = MODULE._validate_parents()
    assert parents["primary_evidence_summary"]["heldout_continuation_manifest_authorized"]
    assert parents["heldout_seed_summary"]["passed"]


def test_frozen_manifest_and_seed_when_available():
    summary_path = MODULE.ARTIFACT_DIRECTORY / "summary.json"
    if not summary_path.exists():
        pytest.skip("prospective held-out manifest is not frozen yet")
    summary = MODULE._read(summary_path)
    seed = MODULE._read(MODULE.ARTIFACT_DIRECTORY / "seed_reconstruction.json")
    assert summary["heldout_continuation_execution_authorized"] is True
    assert seed["bdf1_residual_bitwise"] is True
    assert seed["bdf2_residual_bitwise"] is True
    assert seed["continuation_roundtrip_bitwise"] is True
    MODULE._checksums(MODULE.ARTIFACT_DIRECTORY)
