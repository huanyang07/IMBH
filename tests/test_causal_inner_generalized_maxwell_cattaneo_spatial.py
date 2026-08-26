from __future__ import annotations

import numpy as np

from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_spatial import (
    generalized_maxwell_cattaneo_path_jump,
    generalized_maxwell_cattaneo_signed_fluctuations,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_geometry import (
    kerr_schild_column_geometry,
)


OMEGA = 2.7491520839259703
ALPHA = 0.1


def _fixture():
    geometry = kerr_schild_column_geometry(5.599841633135499e9, 1.48e9)
    left = np.asarray(
        [
            4.74082887,
            -0.330628060,
            0.662598339,
            14.9471713,
            2.13041458e-4,
            20.1048472,
            0.0,
        ],
        dtype=float,
    )
    delta = np.asarray(
        [2.0e-3, 4.0e-4, -3.0e-4, 1.5e-3, -1.0e-6, 8.0e-4, 2.0e-4],
        dtype=float,
    )
    return geometry, left, left + delta


def _jump(geometry, left, right, order=8):
    return generalized_maxwell_cattaneo_path_jump(
        geometry,
        left,
        right,
        proper_vertical_frequency=OMEGA,
        alpha=ALPHA,
        quadrature_order=order,
    )


def _split(geometry, left, right, order=8):
    return generalized_maxwell_cattaneo_signed_fluctuations(
        geometry,
        left,
        right,
        proper_vertical_frequency=OMEGA,
        alpha=ALPHA,
        quadrature_order=order,
    )


def test_constant_state_has_zero_path_jump_and_dissipation() -> None:
    geometry, left, _right = _fixture()
    split = _split(geometry, left, left)
    np.testing.assert_allclose(split.path_jump.total_principal_jump_over_c, 0.0)
    np.testing.assert_allclose(split.dissipation_over_c, 0.0)
    np.testing.assert_allclose(split.negative_fluctuation_over_c, 0.0)
    np.testing.assert_allclose(split.positive_fluctuation_over_c, 0.0)
    assert split.characteristic_quadratic_dissipation == 0.0


def test_exact_flux_rows_close_and_define_one_shared_flux() -> None:
    geometry, left, right = _fixture()
    split = _split(geometry, left, right)
    assert split.path_jump.exact_flux_parity_relative_defect <= 1.0e-8
    assert split.split_closure_relative_defect <= 1.0e-14
    assert split.shared_exact_flux_relative_defect <= 1.0e-8
    assert split.characteristic_quadratic_dissipation >= 0.0


def test_path_reversal_is_antisymmetric_for_each_signed_fluctuation() -> None:
    geometry, left, right = _fixture()
    forward = _split(geometry, left, right)
    reverse = _split(geometry, right, left)
    scale = max(
        float(np.max(np.abs(forward.path_jump.total_principal_jump_over_c))),
        1.0,
    )
    np.testing.assert_allclose(
        reverse.path_jump.total_principal_jump_over_c,
        -forward.path_jump.total_principal_jump_over_c,
        rtol=1.0e-10,
        atol=1.0e-12 * scale,
    )
    np.testing.assert_allclose(
        reverse.negative_fluctuation_over_c,
        -forward.negative_fluctuation_over_c,
        rtol=1.0e-9,
        atol=1.0e-12 * scale,
    )
    np.testing.assert_allclose(
        reverse.positive_fluctuation_over_c,
        -forward.positive_fluctuation_over_c,
        rtol=1.0e-9,
        atol=1.0e-12 * scale,
    )


def test_path_quadrature_converges_on_finite_jump() -> None:
    geometry, left, right = _fixture()
    jumps = [_jump(geometry, left, right, order) for order in (4, 8, 16)]
    scale = max(float(np.linalg.norm(jumps[-1].total_principal_jump_over_c)), 1.0)
    middle_fine = float(
        np.linalg.norm(
            jumps[1].total_principal_jump_over_c
            - jumps[2].total_principal_jump_over_c
        )
        / scale
    )
    assert middle_fine <= 1.0e-7


def test_smooth_jump_limit_matches_midpoint_principal() -> None:
    geometry, base, endpoint = _fixture()
    direction = endpoint - base
    errors = []
    for amplitude in (1.0e-3, 5.0e-4, 2.5e-4):
        left = base - 0.5 * amplitude * direction
        right = base + 0.5 * amplitude * direction
        split = _split(geometry, left, right)
        linear = split.midpoint_principal.radial_matrix @ (right - left)
        scale = max(float(np.max(np.abs(linear))), np.finfo(float).tiny)
        errors.append(
            float(
                np.max(
                    np.abs(split.path_jump.total_principal_jump_over_c - linear)
                )
                / scale
            )
        )
    # This is the prospectively frozen tiny-amplitude ladder.  Its analytic
    # midpoint error is already below the finite-difference principal floor,
    # so monotone ratios would measure roundoff rather than consistency.
    assert max(errors) <= 1.0e-6
