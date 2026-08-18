from __future__ import annotations

import run_causal_inner_stable_parametric_online_manifest_wp10c9d6c7c3b5c4f25ah as f25ah


def test_selected_470_state_parent_is_locked():
    summary, hashes = f25ah._validate_parent()
    assert summary["passed"]
    assert summary["selected_hidden_order"] == 280
    assert summary["selected_online_dimension"] == 470
    assert "decisive_model.npz" in hashes


def test_descriptor_interpolation_preserves_structure_by_construction():
    contract = f25ah._contract()
    interpolation = contract["stable_descriptor_interpolation"]
    assert interpolation["metric"].startswith("G_theta")
    assert interpolation["operator"] == "A_theta_equals_solve_G_theta_K_theta"
    assert "strictly_negative" in interpolation["proof"]
    assert len(interpolation["parameter_grid"]) == 101


def test_unstable_bundle_is_not_misclassified_as_memory():
    contract = f25ah._contract()
    unstable = contract["unstable_bundle_interpolation"]
    assert unstable["linear_macro_propagation_forbidden"] is True
    assert "nonlinear_saturation" in unstable["required_online_replacement"]
    assert contract["state_partition"]["unstable_bundle_may_be_treated_as_stable_memory"] is False


def test_runtime_target_retains_cycle_and_truth_free_contract():
    runtime = f25ah._contract()["runtime_contract"]
    assert runtime["fiducial_cycle_seconds"] == 578_880.0
    assert runtime["maximum_macrosteps"] == 100_000
    assert runtime["online_truth_calls_per_macrostep"] == 0
    assert runtime["minimum_average_macrostep_seconds"] == 5.7888
