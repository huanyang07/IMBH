from __future__ import annotations

import run_causal_inner_explicit_nonlinear_470_architecture_audit_wp10c9d6c7c3b5c4f25aw as f25aw


def test_frozen_470_architecture_manifest_is_locked():
    frozen = f25aw._validate_manifest(require_clean=False)
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authorized_next"] == f25aw.WORK_PACKAGE


def test_online_coordinate_map_has_470_rank_and_90_hidden_dimensions():
    metrics, _ = f25aw._coordinate_audit()
    assert metrics["online_coordinate_rank"] == 470
    assert metrics["hidden_remainder_dimension"] == 90
    assert metrics["stable_memory_projected_rank"] == 280
    assert metrics["departure_projected_rank"] == 28


def test_explicit_departure_coordinates_reduce_unresolved_anchor_rate():
    metrics, _ = f25aw._coordinate_audit()
    assert metrics["online_470_hidden_rate_relative_fraction"] <= 0.05
    assert metrics["online_470_hidden_rate_norm_per_second"] < metrics[
        "physical_memory_442_hidden_rate_norm_per_second"
    ]


def test_inherited_stable_kernel_and_online_algebra_fit_budget():
    metrics = f25aw._cost_and_inheritance_audit()
    assert metrics["inherited_stable_energy_amplification_max"] <= 1.0
    assert metrics["projected_online_cycle_wall_seconds"] <= 1.5 * 86_400.0
    assert metrics["online_truth_calls_per_macrostep"] == 0
