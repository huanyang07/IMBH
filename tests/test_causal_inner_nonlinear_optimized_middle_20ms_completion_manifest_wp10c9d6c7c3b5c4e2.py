from __future__ import annotations

import json

import run_causal_inner_nonlinear_optimized_middle_20ms_completion_manifest_wp10c9d6c7c3b5c4e2 as c4e2


def test_manifest_freezes_cost_reduction_without_relaxing_science() -> None:
    manifest = json.loads(c4e2.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["passed"]
    assert manifest["definitions_only"]
    assert not manifest["operator_changed"]
    assert manifest["optimization_contract"]["routine_base_step"] == (
        "one_full_nonlinear_BDF2_solve"
    )
    assert tuple(manifest["optimization_contract"]["timestep_cap_candidates_seconds"]) == (
        4.0e-4,
        8.0e-4,
        1.2e-3,
    )
    assert manifest["method_gates"]["maximum_scaled_nonlinear_residual"] == 1.0e-10
    assert manifest["method_gates"]["maximum_discrete_ledger_defect"] == 1.0e-12


def test_extraction_partition_tangent_is_binding() -> None:
    manifest = json.loads(c4e2.MANIFEST_PATH.read_text(encoding="utf-8"))
    contract = manifest["extraction_tangent_contract"]
    assert contract["observable"] == "certified_conservative_exterior_partition"
    assert contract["all_five_profile_directions_required"]
    assert contract["instantaneous_response_required"]
    assert contract["cumulative_response_required"]
    assert contract["maximum_generic_discrepancy_fraction_of_response"] == 0.01
    assert manifest["scientific_scope"]["raw_inner_face_is_not_a_binding_slow_export"]


def test_only_optimized_middle_propagation_is_authorized() -> None:
    summary = json.loads(c4e2.SUMMARY_PATH.read_text(encoding="utf-8"))
    assert summary["middle_twenty_ms_optimized_propagation_authorized"]
    assert not summary["fine_twenty_ms_propagation_authorized"]
    assert not summary["fifty_ms_propagation_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
