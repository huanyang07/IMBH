from __future__ import annotations

import numpy as np

from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo import (
    GENERALIZED_MAXWELL_CATTANEO_PRIMITIVE_NAMES,
    _sixth_order_centered_jacobian,
    audit_generalized_maxwell_cattaneo_source_ledger,
    audit_specialized_nonlinear_causality,
    default_primitive_steps,
    generalized_maxwell_cattaneo_local_state,
    generalized_maxwell_cattaneo_principal,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_geometry import (
    kerr_schild_column_geometry,
)


def _fixture():
    geometry = kerr_schild_column_geometry(5.599841633135499e9, 1.48e9)
    chart = np.asarray(
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
    return geometry, chart


def test_local_state_has_exact_conservative_and_transient_split() -> None:
    geometry, chart = _fixture()
    state = generalized_maxwell_cattaneo_local_state(
        geometry,
        chart,
        proper_vertical_frequency=2.7491520839259703,
        alpha=0.1,
    )
    assert len(GENERALIZED_MAXWELL_CATTANEO_PRIMITIVE_NAMES) == 7
    assert state.conservative_state6.shape == (6,)
    assert state.conservative_flux6_over_c.shape == (6,)
    np.testing.assert_allclose(
        state.conservative_flux6_over_c[[0, 4, 5]],
        state.transport_velocity_over_c * state.conservative_state6[[0, 4, 5]],
        rtol=0.0,
        atol=0.0,
    )
    assert state.specific_viscosity_seconds > 0.0
    assert state.relaxation_time_seconds > 0.0
    np.testing.assert_allclose(
        state.shear_ratio,
        0.1 * state.sound_speed_over_c**2,
        rtol=2.0e-14,
        atol=0.0,
    )


def test_representative_principal_is_real_causal_and_diagonalizable() -> None:
    geometry, chart = _fixture()
    principal = generalized_maxwell_cattaneo_principal(
        geometry,
        chart,
        proper_vertical_frequency=2.7491520839259703,
        alpha=0.1,
    )
    assert principal.temporal_matrix.shape == (7, 7)
    assert principal.radial_matrix.shape == (7, 7)
    assert principal.maximum_imaginary_speed_over_c <= 1.0e-10
    assert principal.maximum_eigenpair_relative_defect <= 1.0e-8
    assert principal.eigenvector_condition_number <= 1.0e8
    assert principal.maximum_biorthogonality_defect <= 1.0e-10
    assert principal.maximum_projector_idempotence_defect <= 1.0e-10
    assert principal.scaled_temporal_condition_number <= 1.0e8
    assert principal.maximum_light_cone_excess_over_c <= 1.0e-10
    assert abs(principal.temporal_matrix[4, 1]) > 1.0e-4
    assert abs(principal.temporal_matrix[4, 2]) > 1.0e-4
    assert (
        principal.local_state.four_velocity_normalization_relative_defect
        <= 1.0e-12
    )
    assert (
        principal.local_state.shear_tensor_trace_relative_defect <= 1.0e-12
    )
    assert (
        principal.local_state.shear_tensor_orthogonality_relative_defect
        <= 1.0e-12
    )


def test_complete_principal_is_stable_under_derivative_step_halving() -> None:
    geometry, chart = _fixture()
    pencils = [
        generalized_maxwell_cattaneo_principal(
            geometry,
            chart,
            proper_vertical_frequency=2.7491520839259703,
            alpha=0.1,
            derivative_step_factor=factor,
        )
        for factor in (2.0, 1.0, 0.5)
    ]
    for name in ("temporal_matrix", "radial_matrix"):
        coarse, middle, fine = [getattr(item, name) for item in pencils]
        scale = max(float(np.linalg.norm(fine, ord=np.inf)), 1.0)
        assert np.linalg.norm(coarse - middle, ord=np.inf) / scale <= 1.0e-8
        assert np.linalg.norm(middle - fine, ord=np.inf) / scale <= 1.0e-8


def test_material_current_principal_rows_use_exact_product_rule() -> None:
    geometry, chart = _fixture()
    omega = 2.7491520839259703
    alpha = 0.1

    def evaluate(candidate):
        return generalized_maxwell_cattaneo_local_state(
            geometry,
            candidate,
            proper_vertical_frequency=omega,
            alpha=alpha,
        )

    base = evaluate(chart)
    steps = default_primitive_steps(
        chart,
        equilibrium_specific_stress=base.equilibrium_specific_stress,
    )
    _, state_jacobian = _sixth_order_centered_jacobian(
        lambda candidate: evaluate(candidate).conservative_state6,
        chart,
        steps,
    )
    _, transport_jacobian = _sixth_order_centered_jacobian(
        lambda candidate: np.atleast_1d(
            geometry.base.lapse
            * float(candidate[1])
            / np.sqrt(geometry.base.gamma_rr)
            - geometry.base.radial_shift_over_c
        ),
        chart,
        steps,
    )
    expected = np.asarray(
        [
            base.transport_velocity_over_c * state_jacobian[index]
            + base.conservative_state6[index] * transport_jacobian.ravel()
            for index in (0, 4, 5)
        ]
    )
    principal = generalized_maxwell_cattaneo_principal(
        geometry,
        chart,
        proper_vertical_frequency=omega,
        alpha=alpha,
    )
    actual = principal.radial_matrix[[0, 5, 6]]
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_representative_nonlinear_causality_margins_are_positive() -> None:
    geometry, chart = _fixture()
    state = generalized_maxwell_cattaneo_local_state(
        geometry,
        chart,
        proper_vertical_frequency=2.7491520839259703,
        alpha=0.1,
    )
    audit = audit_specialized_nonlinear_causality(state)
    assert audit.E_plus_Lambda_minimum > 0.0
    assert len(audit.inequality_margins) == 18
    assert audit.minimum_margin > 1.0e-8


def test_representative_source_ledger_closes_and_produces_entropy() -> None:
    geometry, chart = _fixture()
    state = generalized_maxwell_cattaneo_local_state(
        geometry,
        chart,
        proper_vertical_frequency=2.7491520839259703,
        alpha=0.1,
    )
    ledger = audit_generalized_maxwell_cattaneo_source_ledger(
        state,
        alpha=0.1,
    )
    assert ledger.vertical_total_energy_relative_defect <= 1.0e-12
    assert ledger.vertical_reversible_exchange_relative_defect <= 1.0e-12
    assert ledger.minimum_entropy_production_rate >= 0.0
