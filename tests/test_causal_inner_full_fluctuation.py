from __future__ import annotations

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    causal_five_field_complete_principal_path_jump,
    causal_five_field_coordinate_principal_components,
    causal_five_field_periodic_cell_fluctuation_ledger,
    causal_five_field_signed_principal_fluctuations,
    causal_five_field_straight_principal_path_jump,
    make_causal_five_field_regression_context,
    make_causal_five_field_seed,
)


def _inputs():
    context = make_causal_five_field_regression_context(8)
    primitives = make_causal_five_field_seed(context).primitives
    face = 3
    return (
        context,
        float(context.grid.edges[face]),
        np.asarray(primitives[face - 1], dtype=float),
        np.asarray(primitives[face], dtype=float),
    )


def test_complete_path_jump_preserves_sign_and_component_contract() -> None:
    context, radius, left, right = _inputs()
    result = causal_five_field_complete_principal_path_jump(
        context,
        radius,
        left,
        right,
    )
    existing = causal_five_field_straight_principal_path_jump(
        context,
        radius,
        left,
        right,
    )

    np.testing.assert_allclose(
        result.total_principal_jump_over_c,
        existing,
        rtol=0.0,
        atol=1.0e-14
        * max(float(np.max(np.abs(existing))), np.finfo(float).tiny),
    )
    np.testing.assert_allclose(
        result.principal_source_path_integral_over_c,
        result.shear_source_path_integral_over_c
        + result.vertical_source_path_integral_over_c,
        rtol=0.0,
        atol=1.0e-14
        * max(
            float(
                np.max(
                    np.abs(
                        result.principal_source_path_integral_over_c
                    )
                )
            ),
            np.finfo(float).tiny,
        ),
    )
    assert result.source_partition_defect <= 1.0e-14
    assert result.principal_closure_defect <= 1.0e-14


def test_complete_path_reverses_and_signed_fluctuations_close() -> None:
    context, radius, left, right = _inputs()
    forward = causal_five_field_signed_principal_fluctuations(
        context,
        radius,
        left,
        right,
    )
    reverse = causal_five_field_signed_principal_fluctuations(
        context,
        radius,
        right,
        left,
    )
    constant = causal_five_field_signed_principal_fluctuations(
        context,
        radius,
        left,
        left,
    )

    scale = max(
        float(
            np.max(
                np.abs(
                    forward.path_jump.total_principal_jump_over_c
                )
            )
        ),
        np.finfo(float).tiny,
    )
    np.testing.assert_allclose(
        reverse.path_jump.total_principal_jump_over_c,
        -forward.path_jump.total_principal_jump_over_c,
        rtol=0.0,
        atol=1.0e-12 * scale,
    )
    np.testing.assert_allclose(
        forward.negative_fluctuation_over_c
        + forward.stationary_fluctuation_over_c
        + forward.positive_fluctuation_over_c,
        forward.path_jump.total_principal_jump_over_c,
        rtol=0.0,
        atol=1.0e-12 * scale,
    )
    assert forward.split_closure_defect <= 1.0e-12
    assert forward.minimum_speed_gap_over_c > 0.0
    assert np.array_equal(
        constant.path_jump.total_principal_jump_over_c,
        np.zeros(5),
    )
    assert np.array_equal(constant.negative_fluctuation_over_c, np.zeros(5))
    assert np.array_equal(constant.stationary_fluctuation_over_c, np.zeros(5))
    assert np.array_equal(constant.positive_fluctuation_over_c, np.zeros(5))


def test_complete_path_small_jump_linearizes_to_implemented_matrix() -> None:
    context, radius, left, right = _inputs()
    chart = 0.5 * (left + right)
    components = causal_five_field_coordinate_principal_components(
        context,
        radius,
        chart,
    )
    direction = np.asarray([0.25, -0.2, 0.15, 0.3, -0.1], dtype=float)
    direction *= (
        components.primitive_column_scales / np.linalg.norm(direction)
    )
    epsilon = 1.0e-6
    result = causal_five_field_complete_principal_path_jump(
        context,
        radius,
        chart - 0.5 * epsilon * direction,
        chart + 0.5 * epsilon * direction,
    )
    expected = (
        epsilon * components.spatial_principal_matrix @ direction
    )
    scale = max(
        float(np.max(np.abs(expected))),
        float(np.max(np.abs(result.total_principal_jump_over_c))),
        np.finfo(float).tiny,
    )
    defect = float(
        np.max(
            np.abs(result.total_principal_jump_over_c - expected)
        )
        / scale
    )
    assert defect <= 2.0e-6


def test_periodic_cell_ledger_closes_and_preserves_constant_state() -> None:
    context, radius, left, right = _inputs()
    chart = 0.5 * (left + right)
    constant = np.repeat(chart[None, :], 6, axis=0)
    constant_ledger = causal_five_field_periodic_cell_fluctuation_ledger(
        context,
        radius,
        constant,
        constant,
    )
    assert np.array_equal(
        constant_ledger.cell_principal_residuals_over_c,
        np.zeros((6, 5)),
    )
    assert constant_ledger.global_conservative_cycle_defect == 0.0
    assert constant_ledger.global_fluctuation_assembly_defect == 0.0

    edges = np.linspace(0.0, 2.0 * np.pi, 9)
    direction = np.asarray([0.03, -0.02, 0.01, 0.04, -0.01])
    trace = chart[None, :] + np.sin(edges)[:, None] * direction[None, :]
    trace[-1] = trace[0]
    ledger = causal_five_field_periodic_cell_fluctuation_ledger(
        context,
        radius,
        trace[:-1],
        trace[1:],
    )
    assert np.array_equal(
        ledger.interface_total_jumps_over_c,
        np.zeros((8, 5)),
    )
    assert ledger.global_conservative_cycle_defect <= 1.0e-14
    assert ledger.global_fluctuation_assembly_defect <= 1.0e-14
