from __future__ import annotations

import run_causal_inner_nonlinear_bundle_database_manifest_wp10c9d6c7c3b5c4f25aj as f25aj


def test_parent_stable_kernel_is_locked():
    summary, hashes = f25aj._validate_parent()
    assert summary["passed"]
    assert summary["stable_descriptor_dimension"] == 442
    assert summary["unstable_bundle_dimension"] == 28
    assert "parametric_diagnostics.npz" in hashes


def test_state_partition_distinguishes_true_conservation():
    partition = f25aj._contract()["corrected_state_partition"]
    assert partition["coarse_true_conservative_M_J_E_storage"] == 96
    assert partition["coarse_constitutive_storage"] == 64
    assert partition["explicit_stable_coordinates"] == 2
    assert sum(
        partition[name]
        for name in (
            "coarse_true_conservative_M_J_E_storage",
            "coarse_constitutive_storage",
            "explicit_stable_coordinates",
            "stable_hidden_memory",
            "exact_unstable_bundle",
        )
    ) == 470


def test_finite_amplitude_screen_is_prospective_and_bounded():
    screen = f25aj._contract()["prospective_screen"]
    assert screen["anchors"] == ["primary", "heldout"]
    assert screen["direction_count_per_anchor"] == 8
    assert screen["total_nonbase_rate_evaluations"] == 96
    assert screen["maximum_scaled_component_amplitudes"][-1] == 5.0e-3
    assert screen["heldout_results_may_not_change_directions_or_amplitudes"]


def test_hybrid_fallback_is_conservative_and_truth_free_online():
    contract = f25aj._contract()
    fallback = contract["architecture_selection"]["otherwise"]
    database = contract["hybrid_database_requirements"]
    assert fallback["selection"] == "conservative_hybrid_branch_and_event_map"
    assert "Q3_preserving" in database["reset_map"]
    assert database["online_truth_calls"] == 0
    assert contract["claim_boundary"]["branch_existence_assumed"] is False
