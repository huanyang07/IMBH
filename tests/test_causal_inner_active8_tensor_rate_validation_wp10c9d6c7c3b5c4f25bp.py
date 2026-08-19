from __future__ import annotations

import numpy as np
import pytest

import run_causal_inner_active8_tensor_rate_validation_wp10c9d6c7c3b5c4f25bp as f25bp


def test_geometry_certificate_authorizes_only_tensor_rate_validation():
    frozen = f25bp._validate_geometry(require_clean=False)
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authorized_next"] == f25bp.WORK_PACKAGE
    assert frozen["summary"]["completed_candidate_count"] == 192
    assert frozen["summary"]["nonbase_continuous_rate_evaluations"] == 0


def test_evaluation_order_freezes_fit_before_any_validation_truth():
    order = f25bp._evaluation_order()
    assert len(order) == 192
    assert order[:128] == tuple(range(128))
    assert order[128:144] == tuple(range(128, 144))
    assert order[144:160] == tuple(range(176, 192))
    assert order[160:] == tuple(range(144, 176))


def test_pair_targets_recover_exact_homogeneous_tensors():
    rng = np.random.default_rng(31)
    directions = rng.normal(size=(12, 8))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    radii = np.linspace(0.004, 0.01, 12)
    qcoef = rng.normal(size=(36, 28))
    ccoef = rng.normal(size=(120, 28))
    kcoef = rng.normal(size=(120, 4))
    energy = np.eye(28, 8)
    curvature_basis = np.eye(560, 4)
    deltas = []
    coordinates = []
    rates = []
    linear = []
    for direction, radius in zip(directions, radii):
        for sign in (-1, 1):
            active = sign * radius * direction
            coordinate = np.zeros(28)
            coordinate[:8] = active
            delta = np.zeros(560)
            delta[:4] = f25bp.architecture._cubic_features(active[None])[0] @ kcoef
            nonlinear = (
                f25bp.architecture.parent.manifest._quadratic_features(active[None])[0]
                @ qcoef
                + f25bp.architecture._cubic_features(active[None])[0] @ ccoef
            )
            deltas.append(delta)
            coordinates.append(coordinate)
            rates.append(nonlinear)
            linear.append(np.zeros(28))
    targets = f25bp._pair_targets(
        deltas=np.asarray(deltas),
        coordinates=np.asarray(coordinates),
        departure_increments=np.asarray(rates),
        departure_linear=np.asarray(linear),
        energy_directions=energy,
        curvature_basis=curvature_basis,
    )
    assert np.allclose(targets["directions"], directions)
    assert np.allclose(
        targets["rate_quadratic_targets"],
        f25bp.architecture.parent.manifest._quadratic_features(directions) @ qcoef,
    )
    assert np.allclose(
        targets["rate_cubic_targets"],
        f25bp.architecture._cubic_features(directions) @ ccoef,
    )
    assert np.allclose(
        targets["curvature_cubic_targets"],
        f25bp.architecture._cubic_features(directions) @ kcoef,
    )


def test_full_tensor_fit_and_prediction_are_exact_on_a_full_rank_design():
    design = f25bp._load_npz(f25bp.DESIGN_PATH)
    directions = design["total_training_directions_active8"].T
    rng = np.random.default_rng(37)
    qcoef = rng.normal(size=(36, 28))
    ccoef = rng.normal(size=(120, 28))
    kcoef = rng.normal(size=(120, 4))
    targets = {
        "directions": directions,
        "radii": np.ones(120),
        "rate_quadratic_targets": (
            f25bp.architecture.parent.manifest._quadratic_features(directions)
            @ qcoef
        ),
        "rate_cubic_targets": f25bp.architecture._cubic_features(directions)
        @ ccoef,
        "curvature_cubic_targets": f25bp.architecture._cubic_features(directions)
        @ kcoef,
    }
    metrics, fitted = f25bp._fit_coefficients(targets)
    assert metrics["actual_quadratic_feature_rank"] == 36
    assert metrics["actual_cubic_feature_rank"] == 120
    point = rng.normal(size=8) * 0.003
    rate, curvature = f25bp._predict(point, fitted)
    expected_rate = (
        f25bp.architecture.parent.manifest._quadratic_features(point[None])[0]
        @ qcoef
        + f25bp.architecture._cubic_features(point[None])[0] @ ccoef
    )
    expected_curvature = f25bp.architecture._cubic_features(point[None])[0] @ kcoef
    assert np.allclose(rate, expected_rate, rtol=1.0e-10, atol=1.0e-10)
    assert np.allclose(curvature, expected_curvature, rtol=1.0e-10, atol=1.0e-10)


def test_coefficient_lock_refuses_partial_training_and_records_zero_validation(
    tmp_path, monkeypatch
):
    database = tmp_path / "geometry.npz"
    old = tmp_path / "old.npz"
    database.write_bytes(b"geometry")
    old.write_bytes(b"old")
    monkeypatch.setattr(f25bp, "DATABASE_PATH", database)
    monkeypatch.setattr(f25bp, "OLD_CLOSURE_PATH", old)
    monkeypatch.setattr(f25bp, "FIT_ARRAY_PATH", tmp_path / "fit.npz")
    monkeypatch.setattr(f25bp, "FIT_LOCK_PATH", tmp_path / "fit.json")
    monkeypatch.setattr(
        f25bp,
        "_training_targets",
        lambda _inputs, _progress: {"directions": np.zeros((120, 8))},
    )
    fit_metrics = {
        "actual_quadratic_feature_rank": 36,
        "actual_quadratic_feature_condition_number": 2.0,
        "actual_cubic_feature_rank": 120,
        "actual_cubic_feature_condition_number": 20.0,
    }
    fit_arrays = {
        "rate_quadratic_coefficients": np.zeros((36, 28)),
        "rate_cubic_coefficients": np.zeros((120, 28)),
        "curvature_cubic_coefficients": np.zeros((120, 4)),
    }
    monkeypatch.setattr(
        f25bp,
        "_fit_coefficients",
        lambda _targets: (fit_metrics, fit_arrays),
    )
    progress = {"evaluations": [{}] * 127}
    with pytest.raises(RuntimeError, match="before training completion"):
        f25bp._freeze_or_validate_coefficients({}, progress)
    progress["evaluations"].append({})
    metrics, arrays = f25bp._freeze_or_validate_coefficients({}, progress)
    lock = f25bp._read(f25bp.FIT_LOCK_PATH)
    assert metrics == fit_metrics
    assert arrays.keys() == fit_arrays.keys()
    assert lock["training_candidate_count"] == 128
    assert lock["validation_rate_evaluations_at_freeze"] == 0
    assert lock["coefficient_sha256"] == f25bp._sha(f25bp.FIT_ARRAY_PATH)


def test_binding_gates_measure_online_rate_and_state_not_hidden90():
    gates = f25bp.manifest._contract()["binding_independent_model_gates"]
    assert gates["holdout_maximum_full_departure_rate_relative_error"] == 0.05
    assert gates["maximum_full_scaled_state_decoder_relative_error"] == 2.5e-3
    assert not any("hidden_decoder_relative_error" in name for name in gates)
