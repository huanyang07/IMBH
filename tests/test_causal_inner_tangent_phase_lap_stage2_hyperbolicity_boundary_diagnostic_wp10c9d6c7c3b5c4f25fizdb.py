from __future__ import annotations

import numpy as np

import run_causal_inner_tangent_phase_lap_stage2_hyperbolicity_boundary_diagnostic_wp10c9d6c7c3b5c4f25fizdb as target


def test_manifest_authorizes_only_a_nonpropagating_diagnostic() -> None:
    lock = target._validate_manifest(require_clean=False)
    scope = lock["contract"]["diagnostic_scope"]
    assert lock["summary"]["authorized_next"] == target.WORK_PACKAGE
    assert scope["nonpropagating"]
    assert scope["maximum_new_free_field_calls"] == 0
    assert scope["maximum_new_retractions"] == 0


def test_boundary_seed_separates_accepted_and_failed_states() -> None:
    seed = target._seed()
    assert seed["combined_accepted_endpoint_coordinates470"].shape == (71, 470)
    assert seed["combined_accepted_endpoint_primitive_states"].shape == (
        71,
        112,
        5,
    )
    assert not np.array_equal(
        seed["current_primitive_state"], seed["failed_retracted_primitive_state"]
    )


def test_scaled_eigensystem_reports_real_and_complex_pencils() -> None:
    temporal = np.eye(2)
    real = target._scaled_eigensystem(
        temporal,
        np.diag([-1.0, 1.0]),
        np.ones(2),
    )
    complex_pair = target._scaled_eigensystem(
        temporal,
        np.asarray([[0.0, -1.0], [1.0, 0.0]]),
        np.ones(2),
    )
    assert real["maximum_imaginary_speed"] == 0.0
    assert complex_pair["maximum_imaginary_speed"] == 1.0
    assert real["maximum_eigenpair_defect"] < 1.0e-14
    assert complex_pair["maximum_eigenpair_defect"] < 1.0e-14


def test_face_chart_reconstruction_has_all_113_faces(monkeypatch) -> None:
    class Grid:
        edges = np.arange(4.0)

    class Context:
        grid = Grid()

    state = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    left_weights = np.asarray(
        [[1.0, 0.0], [0.75, 0.25], [0.25, 0.75], [0.0, 1.0]]
    )
    right_weights = np.asarray(
        [[0.0, 1.0], [0.25, 0.75], [0.75, 0.25], [1.0, 0.0]]
    )
    monkeypatch.setattr(
        target.radial,
        "_frozen_quadratic_reconstruction_weights",
        lambda _context, _state: (left_weights, right_weights, 0.0),
    )
    charts = target._face_charts(Context(), state)
    left = left_weights @ state
    right = right_weights @ state
    assert charts.shape == (4, 2)
    np.testing.assert_array_equal(charts[0], right[0])
    np.testing.assert_array_equal(charts[1:], 0.5 * (left[1:] + right[1:]))


def test_genuine_complex_pair_only_authorizes_half_step_bracket() -> None:
    lock = target._validate_manifest(require_clean=False)
    branches = lock["contract"]["classification_branches"]
    forbidden = lock["contract"]["forbidden"]
    assert "two-half-step boundary bracket" in branches[
        "genuine_local_complex_pair"
    ]
    assert "replace the complex pair by its real parts" in forbidden
    assert "propagate the saved failed candidate" in forbidden
