from __future__ import annotations

import numpy as np

import run_causal_inner_guarded_departure_amplitude_expansion_manifest_wp10c9d6c7c3b5c4f25cc as f25cc


def test_parent_authorizes_only_guarded_amplitude_manifest():
    frozen = f25cc._validate_parent(require_clean=False)
    assert frozen["summary"]["classification"] == f25cc.parent.NONCLOSURE_CLASSIFICATION
    assert frozen["summary"]["authorized_next"] == (
        "definitions_only_guarded_departure_amplitude_expansion_manifest"
    )
    assert frozen["summary"]["accepted_root_count"] == 0


def test_direction_design_is_deterministic_normalized_and_targeted():
    first, labels, diagnostics = f25cc._direction_design()
    second, second_labels, second_diagnostics = f25cc._direction_design()
    assert np.array_equal(first, second)
    assert labels == second_labels
    assert diagnostics == second_diagnostics
    assert first.shape == (10, 28)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0, rtol=0.0, atol=1.0e-14)
    assert labels[-2:] == ("screen_escape", "accepted_forward_rate")


def test_contract_is_geometry_first_and_fail_fast():
    contract = f25cc._contract()
    assert contract["candidate_family"]["component_bound_rungs"] == [0.015, 0.02, 0.03]
    assert contract["candidate_family"]["rung_order"] == "strictly_increasing_fail_fast"
    assert contract["cost_and_truth_budget"]["new_nonbase_continuous_rate_evaluations_equal"] == 0
    assert contract["scientific_interpretation"]["old_polynomial_extrapolation_is_binding"] is False
    assert contract["scientific_interpretation"]["stable_memory_must_remain_dynamic"]


def test_cycle_architecture_boundary_is_explicit():
    architecture = f25cc._contract()["required_mathematical_architecture"]
    assert architecture["active_state"] == "q162_physical_coordinates"
    assert architecture["stable_memory"].startswith("z280_dynamic_exponential")
    assert architecture["departure"].startswith("a28_nonlinear_multichart")
    assert architecture["maximum_online_truth_calls_per_macrostep"] == 0
    assert architecture["target_maximum_cycle_macrosteps"] == 100_000


def test_no_outcome_directly_authorizes_cycle_evolution():
    decision = f25cc._contract()["fail_fast_decision"]
    assert not decision["physical_microburst_authorized"]
    assert not decision["predictive_cycle_authorized"]
    assert not decision["reduced_slow_evolution_authorized"]
