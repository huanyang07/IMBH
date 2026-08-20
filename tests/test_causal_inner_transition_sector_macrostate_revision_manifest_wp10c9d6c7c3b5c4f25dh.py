from __future__ import annotations

import numpy as np

import run_causal_inner_transition_sector_macrostate_revision_manifest_wp10c9d6c7c3b5c4f25dh as f25dh


def test_anchor_rejection_authorizes_only_a_revision_manifest() -> None:
    frozen = f25dh._validate_parent(require_clean=False)
    assert frozen["parent_hashes"]
    assert "root_not_attempted" in frozen["parent_classification"]


def test_exact_hidden_action_reconciles_with_prior_hybrid_coordinates() -> None:
    metrics, arrays = f25dh._reconciliation()
    assert all(f25dh._checks(metrics).values())
    assert arrays["macro_restriction_R82"].shape == (82, 470)
    assert arrays["exact_anchor_hidden_action470_per_s"].shape == (470,)
    assert metrics["hidden_block_energy_fractions"]["memory280"] > 0.6
    assert metrics["prior_transition_total_energy_capture"]["2"] < 0.04
    assert metrics["prior_transition_total_energy_capture"]["3"] < 0.37


def test_projection_capture_is_monotone() -> None:
    metrics, _ = f25dh._reconciliation()
    capture = metrics["prior_transition_total_energy_capture"]
    assert 0.0 <= capture["1"] <= capture["2"] <= capture["3"] <= 1.0


def test_revised_architecture_separates_online_and_offline_states() -> None:
    metrics, _ = f25dh._reconciliation()
    architecture = f25dh._architecture(metrics)
    hybrid = architecture["two_level_hybrid_architecture"]
    screen = architecture["prospective_hidden_basis_screen"]
    boundary = architecture["authorization_boundaries"]
    assert hybrid["online_transition_ODE"] is False
    assert hybrid["offline_transition_reference_state"] == "full_exact_y470_chart"
    assert "memory280" in hybrid["offline_hidden_state"]
    assert screen["new_exact_rate_calls_equal"] == 0
    assert screen["current_primary_role"] == "mandatory_heldout_transition_direction"
    assert "fieldwise_and_radial_capture" in screen["physical_structure_audit"]
    assert not boundary["complete_tangent_authorized_in_this_package"]
    assert not boundary["reduced_slow_evolution_authorized"]


def test_canonical_manifest_if_present() -> None:
    if not f25dh.CANONICAL_DIRECTORY.exists():
        return
    f25dh._checksums(f25dh.CANONICAL_DIRECTORY)
    summary = f25dh._read(f25dh.CANONICAL_DIRECTORY / "summary.json")
    payload = f25dh._read(
        f25dh.CANONICAL_DIRECTORY / "transition_reconciliation_metrics.json"
    )
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["parent_anchor_rejection_preserved"]
    assert summary["prior_three_coordinate_transition_internal_model_rejected"]
    assert summary["authorized_next"] == f25dh.AUTHORIZED_NEXT
    assert summary["new_exact_fixed_Q_rate_calls"] == 0
    assert summary["new_complete_generator_assemblies"] == 0
    assert summary["new_nonlinear_roots"] == 0
    assert not summary["sealed_16ms_opened"]
    assert all(payload["checks"].values())
