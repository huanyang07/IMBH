from __future__ import annotations

import json

import numpy as np

import run_causal_inner_forward_quadratic_field_revision_manifest_wp10c9d6c7c3b5c4f25cx as f25cx


def test_affine_failure_is_preserved_and_localized():
    frozen = f25cx._validate_parent(require_clean=False)
    assert not frozen["summary"]["passed"]
    assert frozen["summary"]["classification"] == f25cx.parent.FAIL_CLASSIFICATION
    assert frozen["summary"]["completed_exact_rate_calls"] == 4
    assert frozen["metrics"]["checks"]["decoder"]
    assert frozen["metrics"]["checks"]["q162_Jacobian"]
    assert not frozen["metrics"]["checks"]["full_state"]
    assert not frozen["metrics"]["checks"]["q162"]


def test_minimal_features_add_only_forward_curvature():
    active = np.asarray(((0.0, 0.0, 0.0), (2.0, -3.0, 4.0)))
    full = f25cx._full_features(active)
    q = f25cx._q_jacobian_features(active)
    assert np.array_equal(full[:, :4], q)
    assert np.array_equal(full[:, 4], np.asarray((0.0, 4.0)))
    assert full.shape == (2, 5)
    assert q.shape == (2, 4)


def test_blind_directions_vary_cone_radius_and_are_prospective():
    frozen = f25cx._validate_parent(require_clean=False)
    directions = f25cx._blind_directions(frozen["closure"]["active_departure_basis"])
    assert directions.shape == (4, 28)
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)
    assert np.linalg.matrix_rank(directions) == 3
    assert np.array_equal(
        f25cx.BLIND_MIXING_MAGNITUDES, np.asarray((0.0, 0.25, 0.50, 0.75))
    )
    assert f25cx._contract()["geometry_preflight"]["directions_may_not_change"]


def test_forward_quadratic_field_matches_frozen_equations():
    rng = np.random.default_rng(2523)

    class FakeDirect:
        def decoded_delta(self, coordinate):
            return np.resize(np.asarray(coordinate), 560) + 0.2

        def full_state_rate(self, coordinate):
            return 1.5 * np.resize(np.asarray(coordinate), 560) - 0.4

        def field(self, coordinate):
            return 2.0 * np.asarray(coordinate) + 0.3

    center = rng.normal(scale=1.0e-3, size=470)
    direct = FakeDirect()
    restriction = rng.normal(scale=1.0e-2, size=(470, 560))
    closure = {
        "authentic_center_absolute_coordinate": center,
        "authentic_center_scaled_delta": rng.normal(size=560),
        "authentic_center_direct_decoded_scaled_delta": direct.decoded_delta(center),
        "authentic_center_fixed_restriction": restriction,
        "active_departure_basis": np.eye(28, 3),
        "decoder_affine_coefficients": rng.normal(size=(3, 560)),
        "full_rate_forward_quadratic_coefficients": rng.normal(size=(5, 560)),
        "q162_Jacobian_affine_coefficients": rng.normal(
            scale=1.0e-2, size=(4, 162, 560)
        ),
    }
    field = f25cx.ForwardQuadraticAuthenticCenterField(
        closure, model=object(), direct=direct
    )
    local = np.zeros(470)
    local[-28:] = (2.0e-2, -3.0e-3, 4.0e-3) + (0.0,) * 25
    active = field._active(local)
    full_features = np.r_[1.0, active, active[0] ** 2]
    q_features = np.r_[1.0, active]
    absolute = center + local
    old_full = direct.full_state_rate(absolute)
    new_full = old_full + full_features @ closure[
        "full_rate_forward_quadratic_coefficients"
    ]
    weight = field.weight(local)
    assert np.allclose(
        field.full_state_rate(local), old_full + weight * (new_full - old_full)
    )
    new_coordinate = restriction @ new_full
    jacobian = np.einsum(
        "f,fij->ij", q_features, closure["q162_Jacobian_affine_coefficients"]
    )
    new_coordinate[:162] = jacobian @ new_full
    old_coordinate = direct.field(absolute)
    assert np.allclose(
        field.field(local), old_coordinate + weight * (new_coordinate - old_coordinate)
    )


def test_contract_keeps_truth_and_evolution_blocked():
    contract = f25cx._contract()
    assert contract["geometry_preflight"]["count"] == 4
    assert contract["blind_rate_validation"]["count"] == 4
    assert contract["blind_rate_validation"]["maximum_q162_rate_relative_error"] == 0.075
    assert contract["authorization_boundaries"] == {
        "physical_microburst_authorized": False,
        "predictive_cycle_authorized": False,
        "fast_average_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }


def test_canonical_manifest_if_present():
    if not f25cx.CANONICAL_DIRECTORY.exists():
        return
    f25cx._checksums(f25cx.CANONICAL_DIRECTORY)
    summary = json.loads(
        (f25cx.CANONICAL_DIRECTORY / "summary.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (f25cx.CANONICAL_DIRECTORY / "design_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    closure = f25cx._load_npz(
        f25cx.CANONICAL_DIRECTORY / "forward_quadratic_local_field.npz"
    )
    assert summary["passed"] and summary["definitions_only"]
    assert summary["classification"] == f25cx.CLASSIFICATION
    assert summary["authorized_next"] == f25cx.AUTHORIZED_NEXT
    assert metrics["passed"] and all(metrics["checks"].values())
    assert closure["full_rate_forward_quadratic_coefficients"].shape == (5, 560)
    assert closure["q162_Jacobian_affine_coefficients"].shape == (4, 162, 560)
    assert closure["blind_directions"].shape == (4, 28)
