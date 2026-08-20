from __future__ import annotations

import numpy as np

import run_causal_inner_certified_0p015_departure_rate_design_manifest_wp10c9d6c7c3b5c4f25ce as f25ce


def test_parent_authorizes_rate_design_within_certified_bound():
    frozen = f25ce._validate_parent(require_clean=False)
    assert frozen["summary"]["classification"] == f25ce.parent.PARTIAL_CLASSIFICATION
    assert frozen["summary"]["largest_passing_component_bound"] == 0.015
    assert frozen["summary"]["authorized_next"] == (
        "definitions_only_rate_design_within_largest_passing_departure_bound"
    )


def test_target_selection_is_exact_deterministic_and_signed():
    first, first_design = f25ce._selected_design()
    second, second_design = f25ce._selected_design()
    assert first_design == second_design
    assert np.array_equal(first["parent_candidate_indices"], second["parent_candidate_indices"])
    assert tuple(first["parent_candidate_indices"]) == (8, 9, 12, 13, 16, 17, 18, 19)
    assert tuple(first["direction_indices"]) == (4, 4, 6, 6, 8, 8, 9, 9)
    assert tuple(first["signs"]) == (-1, 1, -1, 1, -1, 1, -1, 1)
    assert first["departure_coordinates"].shape == (8, 28)


def test_contract_freezes_truth_gates_and_forward_decision():
    contract = f25ce._contract()
    assert contract["candidate_design"]["component_bound"] == 0.015
    assert contract["candidate_design"]["planned_exact_rate_evaluations"] == 8
    assert contract["candidate_design"]["forward_positive_local_index"] == 7
    assert contract["binding_exact_rate_gates"]["maximum_raw_Schur_condition_number"] == 1.0e6
    assert contract["frozen_old_field_diagnostic"]["model_refit_during_screen"] is False
    assert contract["forward_boundary_decision"]["radial_direction_cosine_threshold"] == 0.02


def test_architecture_keeps_memory_dynamic_and_truth_offline():
    architecture = f25ce._contract()["target_mathematical_architecture"]
    assert architecture["active_state"] == "q162_physical_coordinates"
    assert architecture["stable_memory"].startswith("z280_dynamic")
    assert architecture["departure"].startswith("a28_nonlinear_transient_atlas")
    assert architecture["online_truth_calls_per_macrostep"] == 0
    assert architecture["maximum_macrosteps_per_cycle"] == 100_000


def test_no_outcome_authorizes_trajectory_or_cycle_directly():
    boundaries = f25ce._contract()["scientific_boundaries"]
    assert not boundaries["propagated_state"]
    assert not boundaries["stationary_stable_memory_elimination"]
    assert not boundaries["amplitude_above_certified_bound"]
    assert not boundaries["physical_microburst_authorized"]
    assert not boundaries["predictive_cycle_authorized"]
    assert not boundaries["reduced_slow_evolution_authorized"]
