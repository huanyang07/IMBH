from __future__ import annotations

import numpy as np

import run_causal_inner_shell_gated_atlas_geometry_manifest_wp10c9d6c7c3b5c4f25ch as f25ch


def test_parent_authorizes_geometry_first_holdout():
    frozen = f25ch._validate_parent(require_clean=False)
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authorized_next"] == f25ch.WORK_PACKAGE
    assert frozen["holdout"]["directions"].shape == (4, 28)


def test_contract_freezes_fail_fast_geometry_rungs():
    contract = f25ch._contract()
    family = contract["candidate_family"]
    assert family["component_bounds"] == [0.0125, 0.015]
    assert family["rung_order"] == "strictly_increasing_fail_fast"
    assert family["signs"] == [1]
    assert family["maximum_planned_candidates"] == 8
    assert contract["exact_geometric_retraction"]["rate_reaction_lift_used"] is False


def test_geometry_gates_preserve_prior_exact_contract():
    gates = f25ch._contract()["binding_per_rung_gates"]
    assert gates["maximum_coordinate_residual_infinity"] == 1.0e-10
    assert gates["maximum_normalized_Q3_defect"] == 1.0e-10
    assert gates["minimum_departure_direction_alignment_cosine"] == 0.99
    assert gates["maximum_departure_transverse_fraction"] == 0.05
    assert gates["maximum_coordinate_Jacobian_condition_number"] == 1.0e4


def test_no_rate_or_trajectory_is_authorized():
    contract = f25ch._contract()
    assert contract["cost_budget"]["new_nonbase_continuous_rate_evaluations_equal"] == 0
    boundaries = contract["authorization_boundaries"]
    assert not boundaries["rate_truth_authorized_by_this_manifest"]
    assert not boundaries["geometry_candidate_may_become_atlas_center"]
    assert not boundaries["trajectory_authorized"]
    assert not boundaries["predictive_cycle_authorized"]
    assert not boundaries["reduced_slow_evolution_authorized"]
