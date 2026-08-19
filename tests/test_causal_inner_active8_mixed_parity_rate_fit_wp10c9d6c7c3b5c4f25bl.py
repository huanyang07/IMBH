from __future__ import annotations

import numpy as np

import run_causal_inner_active8_mixed_parity_rate_fit_wp10c9d6c7c3b5c4f25bl as f25bl


def test_geometry_lineage_and_authorization_are_frozen():
    assert f25bl.GEOMETRY_COMMIT == (
        "ca0d5daebd30ba3bfde518ce75d3b380cd4f56b6"
    )
    assert f25bl.GEOMETRY_PARENT == (
        "d4a7453aed3cfe29675fb340842028a9283a8aea"
    )
    assert f25bl.GEOMETRY_TREE == (
        "7b722df4cff21d04693c525533d905d0f83d6950"
    )
    frozen = f25bl._validate_geometry(require_clean=False)
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authorized_next"] == f25bl.WORK_PACKAGE
    assert all(frozen["metrics"]["checks"].values())


def test_quadratic_cubic_and_decoder_models_reproduce_synthetic_training_data():
    rng = np.random.default_rng(19)
    directions = rng.normal(size=(40, 8))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    quadratic = f25bl.manifest._quadratic_features(directions)
    q_coefficients = rng.normal(size=(36, 28))
    cubic_coefficients = rng.normal(size=(40, 28))
    decoder_cubic_coefficients = rng.normal(size=(40, 90))
    decoder_quartic_coefficients = rng.normal(size=(40, 90))
    cubic_kernel = (directions @ directions.T) ** 3
    quartic_kernel = (directions @ directions.T) ** 4
    fitted = f25bl._fit_coefficients(
        directions,
        quadratic @ q_coefficients,
        cubic_kernel @ cubic_coefficients,
        cubic_kernel @ decoder_cubic_coefficients,
        quartic_kernel @ decoder_quartic_coefficients,
    )
    index = 7
    rate, hidden = f25bl._predict(directions[index], directions, fitted)
    expected_rate = (
        quadratic[index] @ q_coefficients
        + cubic_kernel[index] @ cubic_coefficients
    )
    expected_hidden = (
        cubic_kernel[index] @ decoder_cubic_coefficients
        + quartic_kernel[index] @ decoder_quartic_coefficients
    )
    assert np.allclose(rate, expected_rate, rtol=1.0e-10, atol=1.0e-10)
    assert np.allclose(hidden, expected_hidden, rtol=1.0e-10, atol=1.0e-10)


def test_model_gates_are_fail_closed_and_keep_holdout_binding():
    gates = f25bl.manifest._contract()["binding_model_validation_gates"]
    passing = {
        name: (threshold if not name.startswith("minimum_") else 1.0)
        for name, threshold in gates.items()
    }
    assert all(f25bl._model_gate_checks(passing, gates).values())
    failing = dict(passing)
    failing["holdout_maximum_departure_rate_relative_error"] = 0.5
    checks = f25bl._model_gate_checks(failing, gates)
    assert not checks["holdout_maximum_departure_rate_relative_error"]


def test_rate_progress_roundtrip_is_lossless(tmp_path, monkeypatch):
    monkeypatch.setattr(f25bl, "SCRATCH_DIRECTORY", tmp_path / "scratch")
    monkeypatch.setattr(f25bl, "_progress_identity", lambda: {"identity": "test"})
    progress = f25bl._empty_progress({"identity": "test"})
    f25bl._save_progress(progress)
    loaded = f25bl._load_or_create_progress()
    assert loaded["identity"] == progress["identity"]
    assert loaded["evaluations"] == []
    assert loaded["total_rates_per_second"].shape == (0, 560)
