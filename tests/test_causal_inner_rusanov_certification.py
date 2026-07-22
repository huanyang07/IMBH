from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from imri_qpe.layer3_minidisk_1d.causal_inner_rusanov_certification import (
    certify_cached_rusanov_finite_neighborhood,
    certify_rusanov_candidate_coverage,
    certify_rusanov_finite_neighborhood,
    quadratic_taylor_remainder_bound,
    rusanov_gap_variation_bounds,
    rusanov_structured_possible_winner_closure,
    rusanov_structured_zero_remainder_decomposition,
    rusanov_structured_zero_remainder_preflight,
    rusanov_weighted_anchor_gap_radius_screen,
)


def test_gap_variation_bound_includes_first_and_second_derivatives() -> None:
    result = rusanov_gap_variation_bounds(
        0.2,
        np.asarray([2.0, 3.0]),
        gap_hessian_norm_bounds=np.asarray([5.0, 7.0]),
    )

    assert np.isclose(quadratic_taylor_remainder_bound(0.2, 5.0), 0.1)

    np.testing.assert_allclose(
        result,
        np.asarray([0.5, 0.74]),
        rtol=0.0,
        atol=2.0e-16,
    )


def test_candidate_coverage_requires_every_possible_competitor() -> None:
    covered = certify_rusanov_candidate_coverage(
        face_index=1,
        candidate_absolute_speeds=np.asarray([2.0, 1.9, 0.5]),
        gap_variation_bounds=np.asarray([0.0, 0.15, 0.2]),
        neighborhood_radius=0.2,
        represented_candidates=np.asarray([1]),
        variation_bounds_certified=True,
    )
    missing = certify_rusanov_candidate_coverage(
        face_index=1,
        candidate_absolute_speeds=np.asarray([2.0, 1.9, 0.5]),
        gap_variation_bounds=np.asarray([0.0, 0.15, 0.2]),
        neighborhood_radius=0.2,
        represented_candidates=np.asarray([], dtype=int),
        variation_bounds_certified=True,
    )

    assert covered.binding
    assert covered.passed
    assert np.array_equal(covered.possible_candidates, np.asarray([0, 1]))
    assert np.array_equal(covered.represented_candidates, np.asarray([0, 1]))
    assert missing.binding
    assert not missing.passed
    assert np.array_equal(missing.unresolved_candidates, np.asarray([1]))


def test_candidate_suppression_must_itself_be_certified() -> None:
    diagnostic = certify_rusanov_candidate_coverage(
        face_index=1,
        candidate_absolute_speeds=np.asarray([2.0, 1.95]),
        gap_variation_bounds=np.asarray([0.0, 0.1]),
        neighborhood_radius=0.2,
        represented_candidates=np.asarray([], dtype=int),
        suppressed_candidates=np.asarray([1]),
        suppressed_candidate_effect_bounds=np.asarray([0.0, 0.0]),
        variation_bounds_certified=True,
        suppression_certified=False,
    )
    certified = certify_rusanov_candidate_coverage(
        face_index=1,
        candidate_absolute_speeds=np.asarray([2.0, 1.95]),
        gap_variation_bounds=np.asarray([0.0, 0.1]),
        neighborhood_radius=0.2,
        represented_candidates=np.asarray([], dtype=int),
        suppressed_candidates=np.asarray([1]),
        suppressed_candidate_effect_bounds=np.asarray([0.0, 0.0]),
        variation_bounds_certified=True,
        suppression_certified=True,
    )

    assert not diagnostic.binding
    assert not diagnostic.passed
    assert np.array_equal(diagnostic.unresolved_candidates, np.asarray([1]))
    assert certified.binding
    assert certified.passed
    assert certified.unresolved_candidates.size == 0


def test_nonzero_candidate_suppression_requires_explicit_remainder_reserve() -> None:
    missing_reserve = certify_rusanov_candidate_coverage(
        face_index=1,
        candidate_absolute_speeds=np.asarray([2.0, 1.95]),
        gap_variation_bounds=np.asarray([0.0, 0.1]),
        neighborhood_radius=0.2,
        represented_candidates=np.asarray([], dtype=int),
        suppressed_candidates=np.asarray([1]),
        suppressed_candidate_effect_bounds=np.asarray([0.0, 0.03]),
        variation_bounds_certified=True,
        suppression_certified=True,
    )
    reserved = certify_rusanov_candidate_coverage(
        face_index=1,
        candidate_absolute_speeds=np.asarray([2.0, 1.95]),
        gap_variation_bounds=np.asarray([0.0, 0.1]),
        neighborhood_radius=0.2,
        represented_candidates=np.asarray([], dtype=int),
        suppressed_candidates=np.asarray([1]),
        suppressed_candidate_effect_bounds=np.asarray([0.0, 0.03]),
        suppression_remainder_rate_reserved=0.03,
        variation_bounds_certified=True,
        suppression_certified=True,
    )

    assert not missing_reserve.binding
    assert not missing_reserve.passed
    assert np.array_equal(
        missing_reserve.unresolved_candidates,
        np.asarray([1]),
    )
    assert reserved.binding
    assert reserved.passed
    assert reserved.suppression_remainder_rate_required == 0.03


def test_scalar_fixed_branch_bound_matches_exact_solution() -> None:
    horizon = 0.2
    result = certify_rusanov_finite_neighborhood(
        base_generator_per_s=np.asarray([[-2.0]]),
        output_operator=np.asarray([[3.0]]),
        generator_left_factors=np.asarray([[0.5]]),
        generator_right_factors=np.asarray([[1.0]]),
        horizon_seconds=horizon,
        output_gates=np.asarray([1.0]),
        neighborhood_bounds_global=True,
        candidate_coverage_certified=True,
        nonlinear_remainder_certified=True,
        maximum_gate_fraction=1.0,
    )
    exact_state_difference = np.exp(-1.5 * horizon) - np.exp(-2.0 * horizon)

    assert np.isclose(
        result.branch_state_deviation_bound,
        exact_state_difference,
        rtol=2.0e-15,
    )
    assert np.isclose(
        result.per_output_total_bounds[0],
        3.0 * exact_state_difference,
        rtol=2.0e-15,
    )
    assert result.binding
    assert result.passed


def test_bound_encloses_arbitrary_simultaneous_piecewise_switching() -> None:
    generator = np.asarray([[-1.0, 2.0], [0.0, -0.5]])
    left = np.asarray([[0.2, 0.0], [0.1, 0.3]])
    right = np.asarray([[0.4, -0.2], [0.5, 0.1]])
    outputs = np.asarray([[1.0, -0.3], [0.2, 0.7]])
    horizon = 0.3
    duration = horizon / 3.0
    coefficients = (
        np.asarray([1.0, 0.0]),
        np.asarray([-1.0, 1.0]),
        np.asarray([0.5, -0.5]),
    )
    propagator = np.eye(2)
    for values in coefficients:
        switched = generator + sum(
            values[index] * np.outer(left[:, index], right[:, index])
            for index in range(2)
        )
        propagator = expm(duration * switched) @ propagator
    initial = np.asarray([0.6, -0.8])
    exact = np.abs(
        outputs @ (propagator - expm(horizon * generator)) @ initial
    )
    result = certify_rusanov_finite_neighborhood(
        base_generator_per_s=generator,
        output_operator=outputs,
        generator_left_factors=left,
        generator_right_factors=right,
        horizon_seconds=horizon,
        neighborhood_bounds_global=True,
        candidate_coverage_certified=True,
        nonlinear_remainder_certified=True,
    )

    assert np.all(exact <= result.per_output_total_bounds * (1.0 + 1.0e-13))


def test_bound_includes_nonlinear_forcing_and_direct_output_switch() -> None:
    horizon = 0.2
    remainder_rate = 0.1
    result = certify_rusanov_finite_neighborhood(
        base_generator_per_s=np.asarray([[-2.0]]),
        output_operator=np.asarray([[3.0]]),
        generator_left_factors=np.asarray([[0.5]]),
        generator_right_factors=np.asarray([[1.0]]),
        direct_output_deltas=np.asarray([[[0.2]]]),
        horizon_seconds=horizon,
        output_gates=np.asarray([2.0]),
        neighborhood_bounds_global=True,
        nonlinear_remainder_rate=remainder_rate,
        nonlinear_output_remainder_bounds=np.asarray([0.03]),
        candidate_coverage_certified=True,
        nonlinear_remainder_certified=True,
        maximum_gate_fraction=1.0,
    )
    switched_state = (
        np.exp(-1.5 * horizon)
        + remainder_rate * np.expm1(-1.5 * horizon) / -1.5
    )
    state_difference = (
        np.exp(-1.5 * horizon)
        - np.exp(-2.0 * horizon)
        + remainder_rate * np.expm1(-1.5 * horizon) / -1.5
    )
    expected = 3.0 * state_difference + 0.2 * switched_state + 0.03

    assert np.isclose(
        result.switched_state_radius_bound,
        switched_state,
        rtol=2.0e-15,
    )
    assert np.isclose(
        result.per_output_total_bounds[0],
        expected,
        rtol=2.0e-15,
    )
    assert np.isclose(result.per_output_gate_fractions[0], expected / 2.0)


def test_uncertified_remainder_or_candidate_set_cannot_pass() -> None:
    common = {
        "base_generator_per_s": np.asarray([[-2.0]]),
        "output_operator": np.asarray([[1.0]]),
        "generator_left_factors": np.asarray([[1.0e-6]]),
        "generator_right_factors": np.asarray([[1.0]]),
        "horizon_seconds": 0.01,
        "output_gates": np.asarray([1.0]),
        "neighborhood_bounds_global": True,
    }
    missing_candidates = certify_rusanov_finite_neighborhood(
        **common,
        candidate_coverage_certified=False,
        nonlinear_remainder_certified=True,
    )
    missing_remainder = certify_rusanov_finite_neighborhood(
        **common,
        candidate_coverage_certified=True,
        nonlinear_remainder_certified=False,
    )

    assert missing_candidates.maximum_gate_fraction < 1.0e-2
    assert not missing_candidates.binding
    assert not missing_candidates.passed
    assert not missing_remainder.binding
    assert not missing_remainder.passed


def test_trajectory_must_remain_inside_certified_neighborhood() -> None:
    result = certify_rusanov_finite_neighborhood(
        base_generator_per_s=np.asarray([[1.0]]),
        output_operator=np.asarray([[0.0]]),
        generator_left_factors=np.empty((1, 0)),
        generator_right_factors=np.empty((1, 0)),
        horizon_seconds=1.0,
        initial_state_radius=1.0,
        certified_neighborhood_radius=2.0,
        candidate_coverage_certified=True,
        nonlinear_remainder_certified=True,
    )

    assert result.switched_state_radius_bound > 2.0
    assert not result.neighborhood_containment_passed
    assert not result.binding
    assert not result.passed


def test_metric_similarity_matches_explicitly_transformed_problem() -> None:
    generator = np.asarray([[-1.0, 2.0], [0.0, -3.0]])
    output = np.asarray([[2.0, -1.0]])
    left = np.asarray([[0.3], [0.4]])
    right = np.asarray([[0.5], [-0.2]])
    metric = np.asarray([4.0, 0.25])
    scale = np.sqrt(metric)
    transformed = certify_rusanov_finite_neighborhood(
        base_generator_per_s=(
            scale[:, None] * generator / scale[None, :]
        ),
        output_operator=output / scale[None, :],
        generator_left_factors=scale[:, None] * left,
        generator_right_factors=right / scale[:, None],
        horizon_seconds=0.1,
        neighborhood_bounds_global=True,
        candidate_coverage_certified=True,
        nonlinear_remainder_certified=True,
    )
    weighted = certify_rusanov_finite_neighborhood(
        base_generator_per_s=generator,
        output_operator=output,
        generator_left_factors=left,
        generator_right_factors=right,
        horizon_seconds=0.1,
        state_metric_diagonal=metric,
        neighborhood_bounds_global=True,
        candidate_coverage_certified=True,
        nonlinear_remainder_certified=True,
    )

    assert np.isclose(
        weighted.logarithmic_norm_per_s,
        transformed.logarithmic_norm_per_s,
    )
    assert np.isclose(
        weighted.aggregate_switching_radius_per_s,
        transformed.aggregate_switching_radius_per_s,
    )
    np.testing.assert_allclose(
        weighted.per_output_total_bounds,
        transformed.per_output_total_bounds,
    )


def test_direct_output_switches_are_summed_for_simultaneous_branches() -> None:
    result = certify_rusanov_finite_neighborhood(
        base_generator_per_s=np.asarray([[0.0]]),
        output_operator=np.asarray([[0.0]]),
        generator_left_factors=np.asarray([[0.0, 0.0]]),
        generator_right_factors=np.asarray([[1.0, 1.0]]),
        direct_output_deltas=np.asarray([[[2.0]], [[-3.0]]]),
        coefficient_bounds=np.asarray([0.5, 0.25]),
        horizon_seconds=0.4,
        output_gates=np.asarray([2.0]),
        neighborhood_bounds_global=True,
        candidate_coverage_certified=True,
        nonlinear_remainder_certified=True,
        maximum_gate_fraction=1.0,
    )

    assert result.switched_state_radius_bound == 1.0
    assert result.per_output_direct_bounds[0] == 1.75
    assert result.per_output_gate_fractions[0] == 0.875


def test_structured_preflight_preserves_per_face_mutual_exclusivity() -> None:
    common = {
        "base_generator_per_s": np.asarray([[0.0]]),
        "output_operator": np.asarray([[0.0]]),
        "generator_left_factors": np.asarray([[0.0, 0.0]]),
        "generator_right_factors": np.asarray([[1.0, 1.0]]),
        "direct_output_deltas": np.asarray([[[2.0]], [[-3.0]]]),
        "initial_basis": np.asarray([[1.0]]),
        "horizon_seconds": 0.0,
        "output_gates": np.asarray([10.0]),
    }
    same_face = rusanov_structured_zero_remainder_preflight(
        branch_face_indices=np.asarray([1, 1]),
        **common,
    )
    different_faces = rusanov_structured_zero_remainder_preflight(
        branch_face_indices=np.asarray([1, 2]),
        **common,
    )

    assert same_face.per_output_direct_bounds[0] == 3.0
    assert different_faces.per_output_direct_bounds[0] == 5.0
    assert same_face.face_count == 1
    assert different_faces.face_count == 2


def test_face_compressed_direct_left_factors_match_dense_operators() -> None:
    right = np.asarray([[1.0, 2.0]])
    direct_left = np.asarray([[2.0], [-3.0]])
    dense_direct = direct_left[:, :, None] * right.T[:, None, :]
    common = {
        "base_generator_per_s": np.asarray([[-1.0]]),
        "output_operator": np.asarray([[0.5]]),
        "generator_left_factors": np.asarray([[0.2, 0.2]]),
        "generator_right_factors": right,
        "branch_face_indices": np.asarray([1, 1]),
        "initial_basis": np.asarray([[1.0]]),
        "horizon_seconds": 0.1,
        "output_gates": np.asarray([10.0]),
        "time_steps": 32,
    }
    dense = rusanov_structured_zero_remainder_preflight(
        direct_output_deltas=dense_direct,
        **common,
    )
    factored = rusanov_structured_zero_remainder_preflight(
        direct_output_left_factors=direct_left,
        **common,
    )

    assert dense.used_face_compression
    assert factored.used_face_compression
    assert dense.left_compression_relative_defect == 0.0
    assert dense.direct_factor_relative_defect == 0.0
    np.testing.assert_allclose(
        factored.per_output_total_bounds, dense.per_output_total_bounds
    )


def test_structured_preflight_converges_for_scalar_switch() -> None:
    common = {
        "base_generator_per_s": np.asarray([[-2.0]]),
        "output_operator": np.asarray([[3.0]]),
        "generator_left_factors": np.asarray([[0.5]]),
        "generator_right_factors": np.asarray([[1.0]]),
        "branch_face_indices": np.asarray([1]),
        "initial_basis": np.asarray([[1.0]]),
        "horizon_seconds": 0.2,
        "output_gates": np.asarray([1.0]),
    }
    coarse = rusanov_structured_zero_remainder_preflight(
        time_steps=64,
        **common,
    )
    fine = rusanov_structured_zero_remainder_preflight(
        time_steps=256,
        **common,
    )
    exact = 3.0 * (np.exp(-1.5 * 0.2) - np.exp(-2.0 * 0.2))

    assert abs(fine.per_output_dynamic_bounds[0] - exact) < (
        abs(coarse.per_output_dynamic_bounds[0] - exact)
    )
    assert np.isclose(
        fine.per_output_dynamic_bounds[0],
        exact,
        rtol=6.0e-3,
    )
    assert "nonbinding" in fine.semantics


def test_structured_preflight_uses_declared_initial_subspace() -> None:
    result = rusanov_structured_zero_remainder_preflight(
        base_generator_per_s=np.asarray([[-1.0]]),
        output_operator=np.asarray([[1.0]]),
        generator_left_factors=np.asarray([[1.0]]),
        generator_right_factors=np.asarray([[1.0]]),
        branch_face_indices=np.asarray([1]),
        initial_basis=np.empty((1, 0)),
        horizon_seconds=0.1,
        output_gates=np.asarray([1.0]),
    )

    assert result.maximum_branch_coordinate_bound == 0.0
    assert result.per_output_total_bounds[0] == 0.0


def test_structured_preflight_decomposition_sums_to_aggregate_bound() -> None:
    result = rusanov_structured_zero_remainder_decomposition(
        base_generator_per_s=np.asarray([[-1.0, 0.2], [0.0, -0.5]]),
        output_operator=np.asarray([[1.0, -0.3], [0.2, 0.7]]),
        generator_left_factors=np.asarray(
            [[0.2, -0.1, 0.3], [0.1, 0.4, -0.2]]
        ),
        generator_right_factors=np.asarray(
            [[0.5, -0.2, 0.1], [0.3, 0.6, -0.4]]
        ),
        branch_face_indices=np.asarray([1, 1, 2]),
        initial_basis=np.eye(2),
        direct_output_deltas=np.asarray(
            [
                [[0.2, 0.0], [0.0, 0.1]],
                [[-0.3, 0.0], [0.0, 0.2]],
                [[0.1, -0.1], [0.2, 0.0]],
            ]
        ),
        horizon_seconds=0.1,
        time_steps=32,
    )

    np.testing.assert_allclose(
        np.sum(result.per_branch_dynamic_bounds, axis=0),
        result.bound.per_output_dynamic_bounds,
    )
    np.testing.assert_allclose(
        np.sum(result.per_branch_direct_bounds, axis=0),
        result.bound.per_output_direct_bounds,
    )
    np.testing.assert_allclose(
        np.sum(result.per_face_total_bounds, axis=0),
        result.bound.per_output_total_bounds,
    )
    assert np.array_equal(result.face_indices, np.asarray([1, 2]))
    assert np.all(result.direct_winning_branch_indices >= 0)


def test_structured_decomposition_is_invariant_to_candidate_order() -> None:
    common = {
        "base_generator_per_s": np.asarray([[-1.0]]),
        "output_operator": np.asarray([[1.0]]),
        "generator_left_factors": np.asarray([[0.2, 0.2]]),
        "generator_right_factors": np.asarray([[0.3, 0.3]]),
        "branch_face_indices": np.asarray([1, 1]),
        "initial_basis": np.asarray([[1.0]]),
        "direct_output_deltas": np.asarray([[[0.4]], [[0.4]]]),
        "horizon_seconds": 0.1,
        "time_steps": 16,
    }
    forward = rusanov_structured_zero_remainder_decomposition(**common)
    reverse = rusanov_structured_zero_remainder_decomposition(
        **{
            **common,
            "generator_left_factors": common["generator_left_factors"][:, ::-1],
            "generator_right_factors": common[
                "generator_right_factors"
            ][:, ::-1],
            "direct_output_deltas": common["direct_output_deltas"][::-1],
        }
    )

    np.testing.assert_allclose(
        forward.per_face_total_bounds, reverse.per_face_total_bounds
    )
    np.testing.assert_allclose(
        forward.per_branch_dynamic_bounds,
        reverse.per_branch_dynamic_bounds[::-1],
    )
    np.testing.assert_allclose(
        forward.per_branch_direct_bounds,
        reverse.per_branch_direct_bounds[::-1],
    )


def test_zero_horizon_has_no_dynamic_winner_counts() -> None:
    result = rusanov_structured_zero_remainder_decomposition(
        base_generator_per_s=np.asarray([[0.0]]),
        output_operator=np.asarray([[1.0]]),
        generator_left_factors=np.asarray([[1.0, 1.0]]),
        generator_right_factors=np.asarray([[1.0, 1.0]]),
        branch_face_indices=np.asarray([1, 1]),
        initial_basis=np.asarray([[1.0]]),
        horizon_seconds=0.0,
    )

    assert not np.any(result.per_branch_dynamic_winner_counts)


def test_weighted_anchor_gap_screen_uses_dual_state_norm() -> None:
    result = rusanov_weighted_anchor_gap_radius_screen(
        base_speed_gaps=np.asarray([0.5, 1.5]),
        gap_gradient_vectors=np.asarray([[2.0, 0.0], [0.0, 0.5]]),
        state_weights=np.asarray([4.0, 0.25]),
        neighborhood_radii=np.asarray([0.4, 0.5, 2.0]),
        branch_face_indices=np.asarray([1, 2]),
    )

    np.testing.assert_allclose(
        result.weighted_dual_gradient_norms, np.ones(2)
    )
    np.testing.assert_allclose(
        result.candidate_threshold_radii, np.asarray([0.5, 1.5])
    )
    assert np.array_equal(
        result.possible_candidate_masks,
        np.asarray([[False, False], [True, False], [True, True]]),
    )
    assert np.array_equal(result.possible_face_counts, np.asarray([0, 1, 2]))
    assert "nonbinding" in result.semantics


def test_weighted_gap_screen_is_nested_forced_and_metric_invariant() -> None:
    original = rusanov_weighted_anchor_gap_radius_screen(
        base_speed_gaps=np.asarray([0.5, 1.5]),
        gap_gradient_vectors=np.asarray([[2.0, 0.0], [0.0, 0.5]]),
        state_weights=np.asarray([4.0, 0.25]),
        neighborhood_radii=np.asarray([0.0, 0.5, 2.0]),
        branch_face_indices=np.asarray([1, 2]),
        forced_candidate_mask=np.asarray([False, True]),
    )
    transformed = rusanov_weighted_anchor_gap_radius_screen(
        base_speed_gaps=np.asarray([0.5, 1.5]),
        gap_gradient_vectors=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        state_weights=np.ones(2),
        neighborhood_radii=np.asarray([0.0, 0.5, 2.0]),
        branch_face_indices=np.asarray([1, 2]),
        forced_candidate_mask=np.asarray([False, True]),
    )

    np.testing.assert_allclose(
        original.weighted_dual_gradient_norms,
        transformed.weighted_dual_gradient_norms,
    )
    assert np.all(
        original.possible_candidate_masks[:-1]
        <= original.possible_candidate_masks[1:]
    )
    assert np.all(original.possible_candidate_masks[:, 1])
    assert original.possible_candidate_masks[1, 0]


def test_structured_possible_winner_closure_adds_reachable_candidate() -> None:
    result = rusanov_structured_possible_winner_closure(
        base_generator_per_s=np.asarray([[0.0]]),
        generator_left_factors=np.asarray([[1.0, 1.0]]),
        generator_right_factors=np.asarray([[1.0, 1.0]]),
        branch_face_indices=np.asarray([1, 1]),
        base_speed_gaps=np.asarray([0.0, 1.1]),
        initial_basis=np.asarray([[1.0]]),
        horizon_seconds=0.2,
        seed_candidate_mask=np.asarray([True, False]),
        time_steps=256,
    )

    assert result.converged
    assert result.iteration_count == 2
    assert np.array_equal(
        result.possible_candidate_mask, np.asarray([True, True])
    )
    assert result.closed_maximum_gap_variations[1] > 1.1
    assert "nonbinding" in result.semantics


def test_possible_winner_closure_reports_incomplete_monotone_set() -> None:
    result = rusanov_structured_possible_winner_closure(
        base_generator_per_s=np.asarray([[0.0]]),
        generator_left_factors=np.asarray([[1.0, 1.0]]),
        generator_right_factors=np.asarray([[1.0, 1.0]]),
        branch_face_indices=np.asarray([1, 1]),
        base_speed_gaps=np.asarray([0.0, 1.1]),
        initial_basis=np.asarray([[1.0]]),
        horizon_seconds=0.2,
        seed_candidate_mask=np.asarray([True, False]),
        time_steps=256,
        maximum_iterations=1,
    )

    assert not result.converged
    assert np.array_equal(
        result.possible_candidate_mask, np.asarray([True, True])
    )
    assert result.closed_maximum_gap_variations[1] > 1.1


def test_possible_winner_closure_handles_zero_branches_and_weighted_radius() -> None:
    result = rusanov_structured_possible_winner_closure(
        base_generator_per_s=np.asarray([[-1.0]]),
        generator_left_factors=np.empty((1, 0)),
        generator_right_factors=np.empty((1, 0)),
        branch_face_indices=np.empty(0, dtype=int),
        base_speed_gaps=np.empty(0),
        initial_basis=np.asarray([[0.5]]),
        state_metric_diagonal=np.asarray([4.0]),
        horizon_seconds=0.2,
        time_steps=16,
    )

    assert result.converged
    assert result.possible_candidate_count == 0
    assert result.maximum_nominal_state_radius == 1.0
    assert result.maximum_branch_state_deviation == 0.0
    assert result.maximum_total_state_radius == 1.0


def test_cached_wrapper_refuses_incomplete_coverage_or_remainder() -> None:
    arrays = {
        "dynamic": -np.eye(10),
        "production_rusanov_kink_generator_left_factors": np.ones((10, 1)),
        "production_rusanov_kink_generator_right_factors": np.ones((10, 1)),
        "production_rusanov_kink_face_indices": np.asarray([1]),
        "production_rusanov_kink_competitor_codes": np.asarray([1]),
        "state_weights": np.ones(10),
    }
    coverage = certify_rusanov_candidate_coverage(
        face_index=1,
        candidate_absolute_speeds=np.asarray([2.0, 1.95]),
        gap_variation_bounds=np.asarray([0.0, 0.1]),
        neighborhood_radius=2.0,
        represented_candidates=np.asarray([1]),
        variation_bounds_certified=True,
    )
    missing_coverage = certify_cached_rusanov_finite_neighborhood(
        arrays,
        output_operator=np.zeros((1, 10)),
        output_gates=np.ones(1),
        horizon_seconds=0.025,
        nonlinear_remainder_rate=0.0,
        nonlinear_output_remainder_bounds=np.zeros(1),
        nonlinear_remainder_certified=True,
        certified_neighborhood_radius=2.0,
    )
    missing_remainder = certify_cached_rusanov_finite_neighborhood(
        arrays,
        output_operator=np.zeros((1, 10)),
        output_gates=np.ones(1),
        horizon_seconds=0.025,
        candidate_coverages=(coverage,),
        certified_neighborhood_radius=2.0,
    )
    complete = certify_cached_rusanov_finite_neighborhood(
        arrays,
        output_operator=np.zeros((1, 10)),
        output_gates=np.ones(1),
        horizon_seconds=0.025,
        candidate_coverages=(coverage,),
        nonlinear_remainder_rate=0.0,
        nonlinear_output_remainder_bounds=np.zeros(1),
        nonlinear_remainder_certified=True,
        certified_neighborhood_radius=2.0,
    )

    assert not missing_coverage.binding
    assert not missing_coverage.passed
    assert not missing_remainder.binding
    assert not missing_remainder.passed
    assert complete.binding
    assert complete.passed


def test_cached_wrapper_rejects_duplicate_face_coverage() -> None:
    arrays = {
        "dynamic": -np.eye(15),
        "production_rusanov_kink_generator_left_factors": np.empty((15, 0)),
        "production_rusanov_kink_generator_right_factors": np.empty((15, 0)),
        "state_weights": np.ones(15),
    }
    face_one = certify_rusanov_candidate_coverage(
        face_index=1,
        candidate_absolute_speeds=np.asarray([2.0, 1.0]),
        gap_variation_bounds=np.asarray([0.0, 0.1]),
        neighborhood_radius=2.0,
        represented_candidates=np.asarray([], dtype=int),
        variation_bounds_certified=True,
    )

    result = certify_cached_rusanov_finite_neighborhood(
        arrays,
        output_operator=np.zeros((1, 15)),
        output_gates=np.ones(1),
        horizon_seconds=0.025,
        candidate_coverages=(face_one, face_one),
        nonlinear_remainder_rate=0.0,
        nonlinear_output_remainder_bounds=np.zeros(1),
        nonlinear_remainder_certified=True,
        certified_neighborhood_radius=2.0,
    )

    assert not result.binding
    assert not result.passed


def test_cached_wrapper_rejects_represented_candidate_without_factor() -> None:
    arrays = {
        "dynamic": -np.eye(10),
        "production_rusanov_kink_generator_left_factors": np.empty((10, 0)),
        "production_rusanov_kink_generator_right_factors": np.empty((10, 0)),
        "state_weights": np.ones(10),
    }
    coverage = certify_rusanov_candidate_coverage(
        face_index=1,
        candidate_absolute_speeds=np.asarray([2.0, 1.95]),
        gap_variation_bounds=np.asarray([0.0, 0.1]),
        neighborhood_radius=2.0,
        represented_candidates=np.asarray([1]),
        variation_bounds_certified=True,
    )

    result = certify_cached_rusanov_finite_neighborhood(
        arrays,
        output_operator=np.zeros((1, 10)),
        output_gates=np.ones(1),
        horizon_seconds=0.025,
        candidate_coverages=(coverage,),
        nonlinear_remainder_rate=0.0,
        nonlinear_output_remainder_bounds=np.zeros(1),
        nonlinear_remainder_certified=True,
        certified_neighborhood_radius=2.0,
    )

    assert not result.binding
    assert not result.passed


def test_cached_wrapper_requires_nonzero_suppression_inside_remainder() -> None:
    arrays = {
        "dynamic": -np.eye(10),
        "production_rusanov_kink_generator_left_factors": np.empty((10, 0)),
        "production_rusanov_kink_generator_right_factors": np.empty((10, 0)),
        "state_weights": np.ones(10),
    }
    coverage = certify_rusanov_candidate_coverage(
        face_index=1,
        candidate_absolute_speeds=np.asarray([2.0, 1.95]),
        gap_variation_bounds=np.asarray([0.0, 0.1]),
        neighborhood_radius=2.0,
        represented_candidates=np.asarray([], dtype=int),
        suppressed_candidates=np.asarray([1]),
        suppressed_candidate_effect_bounds=np.asarray([0.0, 0.03]),
        suppression_remainder_rate_reserved=0.03,
        variation_bounds_certified=True,
        suppression_certified=True,
    )
    under_reserved = certify_cached_rusanov_finite_neighborhood(
        arrays,
        output_operator=np.zeros((1, 10)),
        output_gates=np.ones(1),
        horizon_seconds=0.025,
        candidate_coverages=(coverage,),
        nonlinear_remainder_rate=0.02,
        nonlinear_output_remainder_bounds=np.zeros(1),
        nonlinear_remainder_certified=True,
        certified_neighborhood_radius=2.0,
    )
    fully_reserved = certify_cached_rusanov_finite_neighborhood(
        arrays,
        output_operator=np.zeros((1, 10)),
        output_gates=np.ones(1),
        horizon_seconds=0.025,
        candidate_coverages=(coverage,),
        nonlinear_remainder_rate=0.03,
        nonlinear_output_remainder_bounds=np.zeros(1),
        nonlinear_remainder_certified=True,
        certified_neighborhood_radius=2.0,
    )

    assert not under_reserved.binding
    assert not under_reserved.passed
    assert fully_reserved.binding
    assert fully_reserved.passed
