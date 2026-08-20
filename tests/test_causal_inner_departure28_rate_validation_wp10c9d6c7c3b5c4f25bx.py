from __future__ import annotations

import numpy as np
import pytest

import run_causal_inner_departure28_rate_validation_wp10c9d6c7c3b5c4f25bx as f25bx


def test_geometry_certificate_authorizes_only_departure28_rate_validation():
    frozen = f25bx._validate_geometry(require_clean=False)
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authorized_next"] == f25bx.WORK_PACKAGE
    assert frozen["summary"]["completed_candidate_count"] == 48
    assert frozen["summary"]["nonbase_continuous_rate_evaluations"] == 0
    assert frozen["contract"]["leakage_control"][
        "all_rate_coefficients_frozen_and_hashed_before_new_rate_truth"
    ]


def test_evaluation_order_is_independent_holdout_then_radial():
    inputs = f25bx._load_inputs()
    assert f25bx._evaluation_order() == tuple(range(48))
    assert [item["split"] for item in inputs["candidates"][:32]] == [
        "holdout"
    ] * 32
    assert [item["split"] for item in inputs["candidates"][32:]] == [
        "tuning_low"
    ] * 16
    assert inputs["coordinates"].shape == (48, 28)


def test_revealed_fit_has_frozen_dimensions_conditioning_and_decoder():
    inputs = f25bx._load_inputs()
    targets = f25bx._training_targets(inputs)
    metrics, coefficients = f25bx._fit_coefficients(targets)
    gates = f25bx.manifest._contract()["binding_fit_gates"]
    assert targets["directions"].shape == (160, 28)
    assert metrics["even_system_rank"] == 160
    assert metrics["odd_system_rank"] == 160
    assert metrics["even_system_condition_number"] <= gates[
        "even_system_condition_number"
    ]
    assert metrics["odd_system_condition_number"] <= gates[
        "odd_system_condition_number"
    ]
    assert metrics["stored_total_nonlinear_coefficient_count"] == 9_440
    assert coefficients["even_dual_coefficients"].shape == (160, 28)
    assert coefficients["odd_dual_coefficients"].shape == (160, 28)
    assert coefficients["frozen_rank4_curvature_decoder_coefficients"].shape == (
        120,
        4,
    )


def test_prediction_has_full_departure28_even_quadratic_and_odd_cubic_parity():
    rng = np.random.default_rng(89)
    directions = rng.normal(size=(160, 28))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    coefficients = {
        "training_directions_departure28": directions,
        "even_dual_coefficients": rng.normal(size=(160, 28)),
        "odd_dual_coefficients": rng.normal(size=(160, 28)),
    }
    coordinate = rng.normal(size=28) * 0.004
    plus = f25bx._predict_rate(coordinate, coefficients)
    minus = f25bx._predict_rate(-coordinate, coefficients)
    radius = np.linalg.norm(coordinate)
    direction = coordinate / radius
    even = (
        radius**2
        * f25bx.diagnosis._kernel(
            direction[None], directions, f25bx.EVEN_KERNEL_POWER
        )[0]
        @ coefficients["even_dual_coefficients"]
    )
    odd = (
        radius**3
        * f25bx.diagnosis._kernel(
            direction[None], directions, f25bx.ODD_KERNEL_POWER
        )[0]
        @ coefficients["odd_dual_coefficients"]
    )
    assert np.allclose(0.5 * (plus + minus), even)
    assert np.allclose(0.5 * (plus - minus), odd)


def test_frozen_curvature_decoder_remains_odd_and_algebraic():
    rng = np.random.default_rng(97)
    coefficients = {
        "frozen_rank4_curvature_decoder_coefficients": rng.normal(size=(120, 4))
    }
    active = rng.normal(size=8) * 0.004
    plus = f25bx._predict_curvature(active, coefficients)
    minus = f25bx._predict_curvature(-active, coefficients)
    assert np.allclose(plus, -minus)


def test_certified_truth_engine_is_rebound_to_departure28_contract():
    engine = f25bx._fresh_engine()
    assert engine.manifest is f25bx.manifest
    assert engine.geometry is f25bx.geometry
    assert engine.WORK_PACKAGE == f25bx.WORK_PACKAGE
    assert engine.DATABASE_PATH == f25bx.DATABASE_PATH
    assert engine.FIT_ARRAY_PATH == f25bx.FIT_ARRAY_PATH
    assert engine._freeze_or_validate_coefficients is f25bx._freeze_or_validate_coefficients
    assert engine._evaluation_order() == tuple(range(48))


def test_coefficient_lock_precedes_any_new_truth(tmp_path, monkeypatch):
    geometry_database = tmp_path / "geometry.npz"
    architecture_database = tmp_path / "architecture.npz"
    prior_decoder = tmp_path / "decoder.npz"
    geometry_database.write_bytes(b"geometry")
    architecture_database.write_bytes(b"architecture")
    prior_decoder.write_bytes(b"decoder")
    monkeypatch.setattr(f25bx, "DATABASE_PATH", geometry_database)
    monkeypatch.setattr(f25bx, "ARCHITECTURE_PATH", architecture_database)
    monkeypatch.setattr(f25bx, "PRIOR_DECODER_PATH", prior_decoder)
    monkeypatch.setattr(f25bx, "SCRATCH_DIRECTORY", tmp_path / "scratch")
    monkeypatch.setattr(f25bx, "FIT_ARRAY_PATH", tmp_path / "scratch" / "fit.npz")
    monkeypatch.setattr(f25bx, "FIT_LOCK_PATH", tmp_path / "scratch" / "fit.json")
    targets = {
        "directions": np.zeros((160, 28)),
        "radii": np.ones(160),
        "quadratic_targets": np.zeros((160, 28)),
        "cubic_targets": np.zeros((160, 28)),
        "curvature_decoder": np.zeros((120, 4)),
    }
    fit_metrics = {
        "even_system_rank": 160,
        "odd_system_rank": 160,
        "stored_total_nonlinear_coefficient_count": 9_440,
    }
    fit_arrays = {
        "training_directions_departure28": targets["directions"],
        "training_radii": targets["radii"],
        "rate_quadratic_targets": targets["quadratic_targets"],
        "rate_cubic_targets": targets["cubic_targets"],
        "frozen_rank4_curvature_decoder_coefficients": targets[
            "curvature_decoder"
        ],
    }
    monkeypatch.setattr(f25bx, "_training_targets", lambda _inputs: targets)
    monkeypatch.setattr(
        f25bx, "_fit_coefficients", lambda _targets: (fit_metrics, fit_arrays)
    )
    f25bx.SCRATCH_DIRECTORY.mkdir()
    (f25bx.SCRATCH_DIRECTORY / "progress.json").write_text("{}")
    with pytest.raises(RuntimeError, match="truth exists before coefficient freeze"):
        f25bx._freeze_or_validate_coefficients({})
    (f25bx.SCRATCH_DIRECTORY / "progress.json").unlink()
    metrics, arrays = f25bx._freeze_or_validate_coefficients({})
    lock = f25bx._read(f25bx.FIT_LOCK_PATH)
    assert metrics == fit_metrics
    assert arrays.keys() == fit_arrays.keys()
    assert lock["training_direction_count"] == 160
    assert lock["validation_rate_evaluations_at_freeze"] == 0
    assert lock["coefficient_sha256"] == f25bx._sha(f25bx.FIT_ARRAY_PATH)


def test_binding_gates_and_authorization_boundary_are_unchanged():
    contract = f25bx.manifest._contract()
    gates = contract["binding_independent_model_gates"]
    assert gates["holdout_median_nonlinear_departure_rate_relative_error"] == 0.10
    assert gates["holdout_maximum_nonlinear_departure_rate_relative_error"] == 0.25
    assert gates["holdout_median_full_departure_rate_relative_error"] == 0.02
    assert gates["holdout_maximum_full_departure_rate_relative_error"] == 0.05
    assert contract["decision"]["predictive_cycle_authorized"] is False
    assert contract["decision"]["reduced_slow_evolution_authorized"] is False
