from __future__ import annotations

import hashlib
import json

import pytest

import run_causal_inner_reduced_cycle_architecture_manifest_wp10c9d6c7c3b5c4f25 as f25


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_locked_inputs_reproduce_the_architecture_evidence():
    evidence = f25._validate_inputs()
    assert evidence["latest"]["classification"] == "operational_timestep_rung_2e7_failed"
    assert evidence["warm"]["root_passed"]
    assert evidence["memory"]["cross_resolution"]["minimum_principal_cosine"] > 0.99998
    assert evidence["two_mode"]["minimum_passing_output_oriented_dimension"] == 6
    assert not evidence["six_mode"]["passed"]
    assert evidence["hybrid"]["leading_two_state_coordinate_supported"]


def test_cost_contract_forbids_online_truth_calls():
    cost = f25._contract()["runtime_contract"]
    assert cost["online_truth_solver_calls_per_macrostep"] == 0
    assert cost["direct_truth_wall_hours_per_microsecond"] == pytest.approx(3.5822401927085594)
    assert cost["direct_truth_wall_days_per_millisecond"] > 149.0
    assert cost["direct_truth_wall_years_per_cycle"] > 2.0e8
    assert cost["minimum_required_end_to_end_speedup"] > 2.0e10
    assert cost["minimum_average_macrostep_seconds"] > 5.0


def test_primary_state_size_and_cross_grid_roles_are_explicit():
    contract = f25._contract()
    state = contract["macro_state"]
    cost = contract["runtime_contract"]
    assert state["radial_cell_candidates"] == (8, 16, 32)
    assert state["primary_radial_cells"] == 16
    assert state["finite_memory_dimensions_to_screen"] == (0, 2, 4, 6)
    assert cost["primary_continuous_state_dimension"] == 88
    assert cost["primary_continuous_state_dimension"] <= cost["maximum_primary_continuous_state_dimension"]
    assert cost["validation_continuous_state_dimensions"]["32"] == 168


def test_semidiscrete_form_is_exactly_conservative():
    conservation = f25._contract()["conservative_semidiscrete_form"]
    assert conservation["interior_face_flux_is_single_valued"]
    assert conservation["interior_fluxes_telescope_exactly"]
    assert conservation["binding_global_ledgers"] == (
        "mass",
        "angular_momentum",
        "killing_energy",
    )
    assert conservation["raw_horizon_face_flux_forbidden"]


def test_finite_memory_and_hybrid_switches_preserve_physics():
    contract = f25._contract()
    memory = contract["finite_memory_boundary_closure"]
    branch = contract["hybrid_branch_contract"]
    assert memory["stable_poles_required"]
    assert memory["passivity_or_declared_dissipation_gate_required"]
    assert branch["hysteresis_required"]
    assert branch["mass_angular_momentum_energy_continuous_across_switch"]
    assert branch["reset_impulse_ledger_must_close"]


def test_cycle_claim_is_explicitly_exploratory():
    scope = f25._contract()["cycle_physics_scope"]
    assert not scope["current_cycle_is_predictive_from_certified_truth"]
    assert "certified_hot_branch" in scope["predictive_route_requires"]
    assert "held_out_cycle_validation" in scope["predictive_route_requires"]


def test_next_step_is_evidence_only_and_cannot_fit():
    contract = f25._contract()
    screen = contract["evidence_only_screen"]
    assert screen["new_nonlinear_trajectory_count"] == 0
    assert screen["new_fixed_Q_root_count"] == 0
    assert screen["new_tangent_propagation_count"] == 0
    assert screen["may_select_architecture_but_not_fit_coefficients"]
    assert screen["may_not_authorize_online_solver_implementation"]
    assert contract["authorized_next"].endswith("evidence_only_identifiability_screen")


def test_frozen_package_when_available():
    summary_path = f25.ARTIFACT_DIRECTORY / "summary.json"
    if not summary_path.exists():
        pytest.skip("architecture manifest not frozen yet")
    summary = _read(summary_path)
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["evidence_only_identifiability_authorized"]
    assert not summary["online_reduced_solver_implementation_authorized"]
    for line in (f25.ARTIFACT_DIRECTORY / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((f25.ARTIFACT_DIRECTORY / name).read_bytes()).hexdigest() == expected
