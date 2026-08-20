from __future__ import annotations

import numpy as np

import run_causal_inner_exact_geometric_470_chart_preflight_wp10c9d6c7c3b5c4f25de as f25de


def test_schedule_and_dimensions_are_frozen() -> None:
    assert f25de.PHYSICAL_DIMENSION == 560
    assert f25de.COORDINATE_DIMENSION == 470
    assert f25de.GAUGE_DIMENSION == 90
    assert f25de.MACRO_DIMENSION == 82
    assert f25de.HIDDEN_DIMENSION == 388
    assert f25de.PLANNED_RETRACTIONS == 18


def test_augmented_chart_is_locally_invertible_in_synthetic_geometry() -> None:
    rng = np.random.default_rng(11)
    raw = rng.normal(size=(f25de.COORDINATE_DIMENSION, f25de.PHYSICAL_DIMENSION))
    q, _ = np.linalg.qr(raw.T, mode="complete")
    coordinate = q[:, : f25de.COORDINATE_DIMENSION].T
    gauge = q[:, f25de.COORDINATE_DIMENSION :]
    augmented = np.vstack((coordinate, gauge.T))
    assert np.linalg.matrix_rank(augmented) == f25de.PHYSICAL_DIMENSION
    right = rng.normal(size=f25de.PHYSICAL_DIMENSION)
    correction = np.linalg.solve(augmented, right)
    assert np.allclose(augmented @ correction, right)


def test_direction_indices_cover_hidden_and_macro_without_holdout_tuning() -> None:
    assert len(f25de.HIDDEN_DIRECTION_INDICES) == 4
    assert len(f25de.MACRO_DIRECTION_INDICES) == 4
    assert min(f25de.HIDDEN_DIRECTION_INDICES) >= 0
    assert max(f25de.HIDDEN_DIRECTION_INDICES) < f25de.HIDDEN_DIMENSION
    assert max(f25de.MACRO_DIRECTION_INDICES) < f25de.MACRO_DIMENSION
    assert f25de.SEALED_INDEX not in f25de.HIDDEN_DIRECTION_INDICES


def test_checks_fail_closed() -> None:
    metrics = {
        "completed_retraction_count": 18,
        "failed_retraction_count": 0,
        "coordinate_geometry": {"rank": 470},
        "augmented_geometry": {"augmented_rank": 560},
        "maximum_coordinate_residual_infinity": 1.0e-12,
        "maximum_gauge_residual_infinity": 1.0e-12,
        "maximum_augmented_condition_number": 1.0e3,
        "implicit_derivative": {"maximum_relative_defect": 1.0e-8},
        "maximum_scaled_anchor_departure": 0.003,
        "minimum_reconstruction_factor": 1.0,
        "maximum_height_ratio": 0.1,
        "minimum_scattering_optical_depth": 10.0,
        "all_physical_audits_passed": True,
        "anchor_exact_seed_roundtrip_bitwise": True,
        "raw_decoder_repaired_to_anchor_scaled_infinity": 1.0e-12,
        "sealed_state_was_not_evaluated": True,
        "new_exact_fixed_Q_rate_evaluations": 0,
        "new_complete_generator_assemblies": 0,
        "new_intrinsic_hidden_roots": 0,
        "propagated_states": 0,
    }
    assert all(f25de._checks(metrics).values())
    metrics["new_exact_fixed_Q_rate_evaluations"] = 1
    assert not f25de._checks(metrics)["rate_budget"]


def test_canonical_preflight_if_present() -> None:
    if not f25de.CANONICAL_DIRECTORY.exists():
        return
    f25de._checksums(f25de.CANONICAL_DIRECTORY)
    summary = f25de._read(f25de.CANONICAL_DIRECTORY / "summary.json")
    metrics = f25de._read(
        f25de.CANONICAL_DIRECTORY / "exact_chart_metrics.json"
    )
    assert not summary["passed"]
    assert summary["classification"] == f25de.FAIL_CLASSIFICATION
    assert summary["authorized_next"] is None
    assert not summary["branch_root_execution_authorized"]
    assert not summary["sealed_16ms_opened"]
    assert metrics["metrics"]["completed_retraction_count"] == 18
    assert metrics["metrics"]["failed_retraction_count"] == 0
    assert not metrics["checks"]["implicit_derivative"]
    assert all(
        passed
        for name, passed in metrics["checks"].items()
        if name != "implicit_derivative"
    )
