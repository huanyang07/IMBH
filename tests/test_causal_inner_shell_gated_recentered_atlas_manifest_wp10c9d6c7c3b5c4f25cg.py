from __future__ import annotations

import numpy as np

import run_causal_inner_shell_gated_recentered_atlas_manifest_wp10c9d6c7c3b5c4f25cg as f25cg


def test_parent_requires_local_extension_and_recentered_chart():
    frozen = f25cg._validate_parent(require_clean=False)
    assert frozen["summary"]["classification"] == f25cg.parent.OUTWARD_CLASSIFICATION
    assert frozen["summary"]["forward_boundary_behavior"] == "outward"
    assert not frozen["summary"]["old_departure28_field_supported_to_0p015"]
    assert frozen["summary"]["authorized_next"] == (
        "definitions_only_local_rate_extension_and_recentered_chart_manifest"
    )


def test_shell_gate_exactly_preserves_inner_model_and_is_C2():
    assert f25cg._shell_weight(0.0) == 0.0
    assert f25cg._shell_weight(f25cg.INNER_PRESERVATION_LOAD) == 0.0
    assert f25cg._shell_weight(f25cg.FULL_EXTENSION_LOAD) == 1.0
    assert f25cg._shell_weight(1.0) == 1.0
    samples = np.asarray(
        [f25cg._shell_weight(value) for value in np.linspace(0.01, 0.013, 51)]
    )
    assert np.all(np.diff(samples) >= 0.0)


def test_degree45_extension_fit_is_deterministic_and_small_on_training():
    first, first_metrics = f25cg._fit_extension(include_physical_audits=False)
    assert first["extension_center_directions"].shape == (4, 28)
    assert first["decoder_even4_coefficients"].shape == (4, 560)
    assert first["full_state_rate_odd5_coefficients"].shape == (4, 560)
    assert first_metrics["maximum_inner_certificate_shell_weight"] == 0.0
    assert first_metrics["extended_decoder_maximum_relative_error"] <= 1.0e-3
    assert first_metrics["extended_full_state_rate_maximum_relative_error"] <= 5.0e-3


def test_holdout_design_is_normalized_mixed_and_independent():
    directions, labels, metrics = f25cg._holdout_design()
    assert directions.shape == (4, 28)
    assert len(labels) == 4
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0, rtol=0.0, atol=1.0e-14)
    assert len(set(labels)) == 4
    assert metrics["minimum_holdout_pair_separation"] > 0.0


def test_contract_freezes_recentered_atlas_without_trajectory():
    contract = f25cg._contract()
    assert contract["shell_gate"]["recenter_trigger"] < contract["shell_gate"]["hard_chart_limit"]
    assert contract["atlas_transition"]["center_source"] == (
        "accepted_authentic_propagated_state_only"
    )
    assert not contract["atlas_transition"]["geometry_only_holdout_may_become_center"]
    assert contract["target_cycle_architecture"]["z280"].startswith("dynamic_exponential")
    assert contract["target_cycle_architecture"]["online_truth_calls_per_macrostep"] == 0
    boundaries = contract["authorization_boundaries"]
    assert boundaries["new_truth_calls"] == 0
    assert not boundaries["trajectory_authorized"]
    assert not boundaries["predictive_cycle_authorized"]
    assert not boundaries["reduced_slow_evolution_authorized"]
