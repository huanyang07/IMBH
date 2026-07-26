import numpy as np
import pytest

from imri_qpe.layer3_minidisk_1d import (
    causal_exact_coordinate_projection,
    causal_exact_equal_coordinate_lift_pair,
    causal_gate_normalized_pair_half_spread,
    causal_rescale_descriptor_matrix,
    causal_weighted_constraint_normal_basis,
    causal_weighted_constraint_fiber_null_projection,
)


def _nonlinear_coordinate_fixture():
    base = np.asarray((0.2, -0.1, 0.3, -0.4, 0.5), dtype=float)
    scales = np.asarray((2.0, 0.5, 1.5, 0.75, 1.25), dtype=float)
    weights = np.asarray((1.0, 2.0, 0.5, 3.0, 1.5), dtype=float)
    amplitudes = 4.0 * scales

    def coordinates(primitives):
        delta = (np.asarray(primitives) - base) / scales
        return np.asarray(
            (
                7.0 + delta[0] + 0.2 * delta[2] ** 2,
                -3.0 + delta[1] - 0.15 * delta[3] ** 2,
            )
        )

    target = coordinates(base)
    coordinate_scales = np.asarray((2.0, 5.0), dtype=float)
    raw_constraint = np.asarray(
        (
            (1.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0, 0.0),
        )
    )
    constraint = raw_constraint / coordinate_scales[:, None]
    seed = np.asarray((0.4, -0.3, 1.0, -0.5, 0.25), dtype=float)
    return (
        base,
        scales,
        weights,
        amplitudes,
        coordinates,
        target,
        coordinate_scales,
        constraint,
        seed,
    )


def test_weighted_constraint_normal_basis_and_null_projection():
    (
        _base,
        _scales,
        weights,
        _amplitudes,
        _coordinates,
        _target,
        _coordinate_scales,
        constraint,
        seed,
    ) = _nonlinear_coordinate_fixture()
    result = causal_weighted_constraint_normal_basis(constraint, weights)
    basis = result.basis
    assert result.numerical_rank == 2
    assert result.weighted_orthogonality_defect < 1.0e-14
    assert result.row_space_reconstruction_defect < 1.0e-14
    assert np.max(
        np.abs(basis.T @ (weights[:, None] * basis) - np.eye(2))
    ) < 1.0e-14
    projected = causal_weighted_constraint_fiber_null_projection(
        seed, weights, result
    )
    assert np.max(np.abs(constraint @ projected)) < 1.0e-14
    assert np.max(
        np.abs(basis.T @ (weights * projected))
    ) < 1.0e-14


def test_dense_weighted_constraint_null_projection_is_geometric():
    constraints = np.asarray(
        (
            (1.0, 2.0, -0.5, 0.25),
            (0.3, -0.2, 1.0, 1.5),
        ),
        dtype=float,
    )
    weights = np.asarray((0.5, 3.0, 1.25, 4.0), dtype=float)
    direction = np.asarray((0.7, -0.4, 1.1, 0.2), dtype=float)
    normal = causal_weighted_constraint_normal_basis(constraints, weights)
    projected = causal_weighted_constraint_fiber_null_projection(
        direction,
        weights,
        normal,
    )
    assert np.max(
        np.abs(
            normal.basis.T
            @ (weights[:, None] * normal.basis)
            - np.eye(2)
        )
    ) < 1.0e-13
    assert np.max(np.abs(constraints @ projected)) < 1.0e-13
    removed = direction - projected
    coefficients = np.linalg.lstsq(
        constraints.T,
        weights * removed,
        rcond=None,
    )[0]
    assert np.max(
        np.abs(weights * removed - constraints.T @ coefficients)
    ) < 1.0e-13


def test_descriptor_rescaling_preserves_the_physical_operator():
    physical = np.asarray(((2.0, -0.5), (0.25, 3.0)), dtype=float)
    source_columns = np.asarray((4.0, 0.5), dtype=float)
    source_rows = np.asarray((2.0, 5.0), dtype=float)
    target_columns = np.asarray((1.5, 3.0), dtype=float)
    target_rows = np.asarray((0.75, 6.0), dtype=float)
    source_scaled = (
        physical
        * source_columns[None, :]
        / source_rows[:, None]
    )
    expected = (
        physical
        * target_columns[None, :]
        / target_rows[:, None]
    )
    converted = causal_rescale_descriptor_matrix(
        source_scaled,
        source_primitive_scales=source_columns,
        source_conservation_scales=source_rows,
        target_primitive_scales=target_columns,
        target_conservation_scales=target_rows,
    )
    assert np.array_equal(converted, expected)


def test_exact_equal_coordinate_pair_preserves_seed_projection():
    (
        base,
        scales,
        weights,
        amplitudes,
        coordinates,
        target,
        coordinate_scales,
        constraint,
        seed,
    ) = _nonlinear_coordinate_fixture()
    pair = causal_exact_equal_coordinate_lift_pair(
        base_primitive_vector=base,
        primitive_column_scales=scales,
        state_weights=weights,
        physical_input_amplitudes=amplitudes,
        target_coordinate_values=target,
        target_coordinate_scales=coordinate_scales,
        constraint_matrix=constraint,
        seed_direction=seed,
        seed_multiplier=0.2,
        coordinate_evaluator=coordinates,
    )
    assert pair.minus.optimizer_success
    assert pair.plus.optimizer_success
    assert pair.minus.maximum_coordinate_defect < 1.0e-11
    assert pair.plus.maximum_coordinate_defect < 1.0e-11
    assert pair.maximum_pairwise_coordinate_defect < 1.0e-11
    assert pair.minus.retained_seed_multiplier == pytest.approx(-0.2)
    assert pair.plus.retained_seed_multiplier == pytest.approx(0.2)
    assert pair.minus.retained_seed_multiplier_defect < 1.0e-14
    assert pair.plus.retained_seed_multiplier_defect < 1.0e-14
    assert pair.minus.weighted_direction_cosine > 0.99
    assert pair.plus.weighted_direction_cosine > 0.99
    assert not np.array_equal(
        pair.minus.primitive_vector, pair.plus.primitive_vector
    )


def test_exact_coordinate_projection_reaches_target_minimally():
    (
        base,
        scales,
        weights,
        amplitudes,
        coordinates,
        target,
        coordinate_scales,
        constraint,
        _seed,
    ) = _nonlinear_coordinate_fixture()
    displaced = base + scales * np.asarray(
        (0.05, -0.03, 0.2, -0.1, 0.15),
        dtype=float,
    )
    result = causal_exact_coordinate_projection(
        base_primitive_vector=displaced,
        primitive_column_scales=scales,
        state_weights=weights,
        physical_input_amplitudes=amplitudes,
        target_coordinate_values=target,
        target_coordinate_scales=coordinate_scales,
        constraint_matrix=constraint,
        coordinate_evaluator=coordinates,
    )
    assert result.optimizer_success
    assert result.maximum_coordinate_defect < 1.0e-11
    assert result.normal_basis.numerical_rank == 2
    assert result.weighted_radius > 0.0
    np.testing.assert_allclose(
        coordinates(result.primitive_vector),
        target,
        rtol=0.0,
        atol=5.0e-11,
    )
    expected_component = np.asarray(
        (-0.2 * 0.2**2, 0.15 * 0.1**2, 0.2, -0.1, 0.15)
    )
    np.testing.assert_allclose(
        (result.primitive_vector - base) / scales,
        expected_component,
        rtol=0.0,
        atol=5.0e-10,
    )


def test_exact_equal_coordinate_pair_is_deterministic():
    (
        base,
        scales,
        weights,
        amplitudes,
        coordinates,
        target,
        coordinate_scales,
        constraint,
        seed,
    ) = _nonlinear_coordinate_fixture()
    arguments = dict(
        base_primitive_vector=base,
        primitive_column_scales=scales,
        state_weights=weights,
        physical_input_amplitudes=amplitudes,
        target_coordinate_values=target,
        target_coordinate_scales=coordinate_scales,
        constraint_matrix=constraint,
        seed_direction=seed,
        seed_multiplier=0.1,
        coordinate_evaluator=coordinates,
    )
    first = causal_exact_equal_coordinate_lift_pair(**arguments)
    second = causal_exact_equal_coordinate_lift_pair(**arguments)
    assert np.array_equal(
        first.minus.primitive_vector, second.minus.primitive_vector
    )
    assert np.array_equal(
        first.plus.primitive_vector, second.plus.primitive_vector
    )


def test_gate_normalized_pair_half_spread_is_sign_invariant():
    minus = np.asarray((1.0, -2.0, 3.0))
    plus = np.asarray((1.4, -1.0, 2.0))
    gates = np.asarray((0.2, 0.5, 2.0))
    spread = causal_gate_normalized_pair_half_spread(minus, plus, gates)
    swapped = causal_gate_normalized_pair_half_spread(plus, minus, gates)
    assert np.array_equal(spread, swapped)
    assert spread == pytest.approx((1.0, 1.0, 0.25))


def test_weighted_constraint_normal_basis_rejects_rank_deficiency():
    with pytest.raises(ValueError, match="not full row rank"):
        causal_weighted_constraint_normal_basis(
            np.asarray(((1.0, 0.0), (2.0, 0.0))),
            np.ones(2),
        )
