from __future__ import annotations

import numpy as np
import pytest

import run_causal_inner_active8_projective_kernel_rate_validation_wp10c9d6c7c3b5c4f25bt as f25bt


def test_geometry_certificate_authorizes_only_projective_rate_validation():
    frozen = f25bt._validate_geometry(require_clean=False)
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authorized_next"] == f25bt.WORK_PACKAGE
    assert frozen["summary"]["completed_candidate_count"] == 48
    assert frozen["summary"]["nonbase_continuous_rate_evaluations"] == 0
    assert frozen["contract"]["leakage_control"][
        "all_coefficients_frozen_and_hashed_before_new_rate_truth"
    ]


def test_evaluation_order_is_new_holdout_then_radial_without_training_truth():
    inputs = f25bt._load_inputs()
    assert f25bt._evaluation_order() == tuple(range(48))
    assert [item["split"] for item in inputs["candidates"][:32]] == [
        "holdout"
    ] * 32
    assert [item["split"] for item in inputs["candidates"][32:]] == [
        "tuning_low"
    ] * 16


def test_revealed_fit_has_frozen_dimensions_and_valid_design():
    inputs = f25bt._load_inputs()
    targets = f25bt._training_targets(inputs)
    metrics, coefficients = f25bt._fit_coefficients(targets)
    assert targets["directions"].shape == (144, 8)
    assert metrics["regularized_even_kernel_rank"] == 144
    assert metrics["odd_cubic_feature_rank"] == 120
    assert metrics["stored_nonlinear_coefficient_count"] == 7_872
    assert coefficients["even_kernel_coefficients"].shape == (144, 28)
    assert coefficients["odd_cubic_coefficients"].shape == (120, 28)
    assert coefficients["curvature_cubic_coefficients"].shape == (120, 4)


def test_prediction_has_even_quadratic_and_odd_cubic_parity():
    rng = np.random.default_rng(83)
    directions = rng.normal(size=(144, 8))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    coefficients = {
        "training_directions_active8": directions,
        "even_kernel_coefficients": rng.normal(size=(144, 28)),
        "odd_cubic_coefficients": rng.normal(size=(120, 28)),
        "curvature_cubic_coefficients": rng.normal(size=(120, 4)),
    }
    active = rng.normal(size=8) * 0.004
    rate_plus, curvature_plus = f25bt._predict(active, coefficients)
    rate_minus, curvature_minus = f25bt._predict(-active, coefficients)
    radius = np.linalg.norm(active)
    direction = active / radius
    even = (
        radius**2
        * f25bt._even_kernel(direction[None], directions)[0]
        @ coefficients["even_kernel_coefficients"]
    )
    assert np.allclose(0.5 * (rate_plus + rate_minus), even)
    assert np.allclose(curvature_plus, -curvature_minus)


def test_coefficient_lock_precedes_any_new_truth(tmp_path, monkeypatch):
    geometry_database = tmp_path / "geometry.npz"
    previous_closure = tmp_path / "previous.npz"
    previous_frozen = tmp_path / "frozen.npz"
    geometry_database.write_bytes(b"geometry")
    previous_closure.write_bytes(b"previous")
    previous_frozen.write_bytes(b"frozen")
    monkeypatch.setattr(f25bt, "DATABASE_PATH", geometry_database)
    monkeypatch.setattr(f25bt, "PREVIOUS_CLOSURE_PATH", previous_closure)
    monkeypatch.setattr(f25bt, "PREVIOUS_FROZEN_PATH", previous_frozen)
    monkeypatch.setattr(f25bt, "SCRATCH_DIRECTORY", tmp_path / "scratch")
    monkeypatch.setattr(f25bt, "FIT_ARRAY_PATH", tmp_path / "scratch" / "fit.npz")
    monkeypatch.setattr(f25bt, "FIT_LOCK_PATH", tmp_path / "scratch" / "fit.json")
    targets = {
        "directions": np.zeros((144, 8)),
        "radii": np.ones(144),
        "rate_quadratic_targets": np.zeros((144, 28)),
        "rate_cubic_targets": np.zeros((144, 28)),
        "curvature_cubic_targets": np.zeros((144, 4)),
    }
    fit_metrics = {
        "regularized_even_kernel_rank": 144,
        "odd_cubic_feature_rank": 120,
        "stored_nonlinear_coefficient_count": 7_872,
    }
    fit_arrays = {
        "training_directions_active8": targets["directions"],
        "training_radii": targets["radii"],
        "rate_quadratic_targets": targets["rate_quadratic_targets"],
        "rate_cubic_targets": targets["rate_cubic_targets"],
        "curvature_cubic_targets": targets["curvature_cubic_targets"],
        "even_kernel_coefficients": np.zeros((144, 28)),
        "odd_cubic_coefficients": np.zeros((120, 28)),
        "curvature_cubic_coefficients": np.zeros((120, 4)),
    }
    monkeypatch.setattr(f25bt, "_training_targets", lambda _inputs: targets)
    monkeypatch.setattr(
        f25bt, "_fit_coefficients", lambda _targets: (fit_metrics, fit_arrays)
    )
    f25bt.SCRATCH_DIRECTORY.mkdir()
    (f25bt.SCRATCH_DIRECTORY / "progress.json").write_text("{}")
    with pytest.raises(RuntimeError, match="truth exists before coefficient freeze"):
        f25bt._freeze_or_validate_coefficients({})
    (f25bt.SCRATCH_DIRECTORY / "progress.json").unlink()
    metrics, arrays = f25bt._freeze_or_validate_coefficients({})
    lock = f25bt._read(f25bt.FIT_LOCK_PATH)
    assert metrics == fit_metrics
    assert arrays.keys() == fit_arrays.keys()
    assert lock["training_direction_count"] == 144
    assert lock["validation_rate_evaluations_at_freeze"] == 0
    assert lock["coefficient_sha256"] == f25bt._sha(f25bt.FIT_ARRAY_PATH)


def test_binding_gates_are_unchanged_independent_holdout_limits():
    contract = f25bt.manifest._contract()
    gates = contract["binding_independent_model_gates"]
    assert gates["holdout_median_nonlinear_departure_rate_relative_error"] == 0.10
    assert gates["holdout_maximum_nonlinear_departure_rate_relative_error"] == 0.25
    assert gates["holdout_median_full_departure_rate_relative_error"] == 0.02
    assert gates["holdout_maximum_full_departure_rate_relative_error"] == 0.05
    assert contract["decision"]["predictive_cycle_authorized"] is False
    assert contract["decision"]["reduced_slow_evolution_authorized"] is False
