from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d import (
    CausalFiveFieldMonolithicBDFHistory,
    causal_five_field_analytic_local_maps,
    causal_five_field_dae_scaling,
    causal_five_field_monolithic_frozen_tangent,
    causal_five_field_monolithic_storage_increment,
    causal_five_field_state_from_primitives,
    evaluate_causal_five_field_dae,
    make_causal_five_field_regression_context,
    make_causal_five_field_seed,
    pack_causal_five_field_state,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_fixed_q import (
    causal_five_field_fixed_q_accepted_history,
    causal_five_field_fixed_q_augmented_step_matrix,
    causal_five_field_fixed_q_bdf_restart,
    causal_five_field_fixed_q_bdf_restarts_equal,
    causal_five_field_exterior_q3,
    causal_five_field_fixed_q_reaction,
    causal_five_field_fixed_q_reaction_jvp,
    evaluate_causal_five_field_fixed_q_bdf,
    load_causal_five_field_fixed_q_bdf_restart,
    save_causal_five_field_fixed_q_bdf_restart,
    solve_causal_five_field_fixed_q_bdf,
    solve_causal_five_field_fixed_q_backward_euler,
)


def _problem():
    context = make_causal_five_field_regression_context(5)
    base = np.asarray(
        make_causal_five_field_seed(context).primitives,
        dtype=float,
    )
    context = replace(
        context,
        spatial_reconstruction="quadratic_admissible",
        boundary_trace_reconstruction="plm_one_sided",
        cell_storage_quadrature="gauss_legendre_4",
        outer_boundary_flux_mode="frozen_exterior_rusanov",
        outer_boundary_frozen_exterior_chart=np.array(base[-1], copy=True),
    ).validated()
    state = causal_five_field_state_from_primitives(context, base)
    evaluation = evaluate_causal_five_field_dae(
        pack_causal_five_field_state(state),
        context,
    )
    scaling = causal_five_field_dae_scaling(state, evaluation)
    dimensions = int(base.size)
    columns = np.asarray(
        scaling.column_scales[dimensions : 2 * dimensions],
        dtype=float,
    ).reshape(base.shape)
    rows = np.asarray(
        scaling.row_scales[:dimensions],
        dtype=float,
    ).reshape(base.shape)
    parents = np.arange(base.shape[0], dtype=int)
    keywords = {
        "primitive_column_scales": columns,
        "conservation_row_scales": rows,
        "parent_cell_indices": parents,
        "refinement_ratio": 1,
        "exterior_parent_face": 2,
        "guard_end_parent_face": 3,
        "parent_cell_count": 5,
    }
    return context, base, columns, rows, keywords


def _relative(first: np.ndarray, second: np.ndarray) -> float:
    scale = max(
        float(np.linalg.norm(first)),
        float(np.linalg.norm(second)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(first - second) / scale)


def _scaled_direction(base: np.ndarray) -> np.ndarray:
    phase = np.sin(np.linspace(0.0, np.pi, base.shape[0]))[:, None]
    direction = phase * np.asarray(
        [0.5, -0.25, 0.125, -0.375, 0.25],
        dtype=float,
    )[None, :]
    direction = direction.ravel()
    return direction / np.linalg.norm(direction)


def test_analytic_local_map_exposes_coordinate_angular_velocity_jacobian():
    context, base, columns, _rows, _keywords = _problem()
    cell = 2
    local = causal_five_field_analytic_local_maps(
        context,
        float(context.grid.centers[cell]),
        base[cell],
    )
    direction = columns[cell] * np.asarray(
        [0.2, -0.1, 0.3, -0.4, 0.1],
        dtype=float,
    )
    step = 1.0e-5
    plus = causal_five_field_analytic_local_maps(
        context,
        float(context.grid.centers[cell]),
        base[cell] + step * direction,
    )
    minus = causal_five_field_analytic_local_maps(
        context,
        float(context.grid.centers[cell]),
        base[cell] - step * direction,
    )
    direct = (
        plus.coordinate_angular_velocity
        - minus.coordinate_angular_velocity
    ) / (2.0 * step)
    analytic = float(local.coordinate_angular_velocity_jacobian @ direction)
    assert abs(direct - analytic) / max(abs(direct), abs(analytic), 1.0) <= 1.0e-9


def test_raw_reaction_jvp_matches_independent_five_point_reference():
    context, base, columns, _rows, keywords = _problem()
    reaction = causal_five_field_fixed_q_reaction(context, base, **keywords)
    direction = _scaled_direction(base)
    analytic = causal_five_field_fixed_q_reaction_jvp(
        context,
        base,
        direction,
        reaction=reaction,
        **keywords,
    )
    physical_direction = columns * direction.reshape(base.shape)
    step = 1.0e-4
    values = []
    for coefficient in (1.0, -1.0, 2.0, -2.0):
        candidate = causal_five_field_fixed_q_reaction(
            context,
            base + coefficient * step * physical_direction,
            **keywords,
        )
        values.append(candidate.raw_reaction_scaled_rows)
    direct = (
        -values[2] + 8.0 * values[0] - 8.0 * values[1] + values[3]
    ) / (12.0 * step)
    assert _relative(
        analytic.raw_reaction_scaled_row_derivatives[0],
        direct,
    ) <= 1.0e-8
    assert analytic.maximum_identity_directional_defect <= 1.0e-10
    assert (
        analytic.maximum_reaction_ledger_directional_relative_defect
        <= 1.0e-12
    )


def test_raw_and_normalized_reaction_channels_have_the_same_state_residual():
    context, base, _columns, _rows, keywords = _problem()
    reaction = causal_five_field_fixed_q_reaction(context, base, **keywords)
    normalized_multiplier = np.asarray([0.2, -0.1, 0.05], dtype=float)
    raw_multiplier = reaction.raw_schur_inverse @ normalized_multiplier
    common = {
        "old_primitive_charts": base,
        "new_primitive_charts": base,
        "q3_target": reaction.q3_value,
        "timestep_seconds": 1.0e-5,
        "context": context,
        "order": 1,
        "constraint_row_scales": reaction.q3_derivative_norms,
        **keywords,
    }
    normalized = evaluate_causal_five_field_fixed_q_bdf(
        multipliers=normalized_multiplier,
        reaction_channel_basis="normalized",
        **common,
    )
    raw = evaluate_causal_five_field_fixed_q_bdf(
        multipliers=raw_multiplier,
        reaction_channel_basis="raw",
        **common,
    )
    frozen = evaluate_causal_five_field_fixed_q_bdf(
        multipliers=normalized_multiplier,
        reaction_channel_basis="frozen_normalized",
        reaction_channel_transform=reaction.raw_schur_inverse,
        **common,
    )
    np.testing.assert_allclose(
        raw.augmented_scaled_residual,
        normalized.augmented_scaled_residual,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        frozen.augmented_scaled_residual,
        normalized.augmented_scaled_residual,
        rtol=2.0e-13,
        atol=2.0e-13,
    )


def test_augmented_matrix_contains_state_dependent_raw_reaction_derivative():
    context, base, _columns, _rows, keywords = _problem()
    reaction = causal_five_field_fixed_q_reaction(context, base, **keywords)
    multiplier = np.asarray([0.2, -0.1, 0.05], dtype=float)
    matrix = causal_five_field_fixed_q_augmented_step_matrix(
        context,
        base,
        base,
        multiplier,
        1.0e-5,
        None,
        order=1,
        constraint_row_scales=reaction.q3_derivative_norms,
        reaction_channel_basis="frozen_normalized",
        reaction_channel_transform=reaction.raw_schur_inverse,
        reaction=reaction,
        **keywords,
    )
    dimensions = base.size
    assert matrix.scaled_matrix.shape == (dimensions + 3, dimensions + 3)
    assert np.linalg.norm(matrix.reaction_state_scaled_matrix) > 0.0
    np.testing.assert_array_equal(
        matrix.scaled_matrix[:dimensions, dimensions:],
        -matrix.reaction_multiplier_scaled_matrix,
    )
    np.testing.assert_array_equal(
        matrix.scaled_matrix[dimensions:, :dimensions],
        matrix.constraint_scaled_matrix,
    )
    assert matrix.maximum_block_closure_defect == 0.0
    assert matrix.maximum_reaction_ledger_relative_defect <= 1.0e-12


def test_backward_euler_correction_preserves_q3_and_roundtrips_restart(
    tmp_path,
):
    context, base, _columns, _rows, keywords = _problem()
    tangent = causal_five_field_monolithic_frozen_tangent(
        context,
        base,
        primitive_column_scales=keywords["primitive_column_scales"],
        conservation_row_scales=keywords["conservation_row_scales"],
    )
    reaction = causal_five_field_fixed_q_reaction(context, base, **keywords)
    normalized_multiplier = (
        -reaction.q3_scaled_derivative @ tangent.scaled_base_rate_per_s
    )
    constrained_rate = (
        tangent.scaled_base_rate_per_s
        + reaction.reaction_lift @ normalized_multiplier
    )
    timestep = 1.0e-8
    top_left = (
        reaction.descriptor_scaled_matrix / timestep
        + tangent.evolving_scaled_jacobian
    )
    result = solve_causal_five_field_fixed_q_backward_euler(
        context,
        base,
        timestep,
        constrained_rate,
        normalized_multiplier,
        top_left,
        q3_target=reaction.q3_value,
        constraint_row_scales=reaction.q3_derivative_norms,
        reaction_channel_basis="frozen_normalized",
        reaction_channel_transform=reaction.raw_schur_inverse,
        maximum_scaled_primitive_change=1.0e-2,
        residual_tolerance=1.0e-2,
        storage_parity_tolerance=2.0e-6,
        maximum_newton_iterations=1,
        maximum_line_search_iterations=1,
        **keywords,
    )
    assert result.accepted, result.message
    assert result.maximum_scaled_residual <= 1.0e-2
    assert result.evaluation.maximum_constraint_relative_defect <= 1.0e-12
    assert _relative(result.scaled_rate_per_s, constrained_rate) <= 2.0e-2
    assert not (
        result.evaluation.monolithic_evaluation
        .temporal_storage_uses_direct_rate_action
    )
    assert (
        result.direct_rate_evaluation.monolithic_evaluation
        .temporal_storage_uses_direct_rate_action
    )
    history = causal_five_field_fixed_q_accepted_history(result)
    np.testing.assert_array_equal(
        history.previous_primitive_increment,
        result.primitive_charts - base,
    )
    restart = causal_five_field_fixed_q_bdf_restart(
        result,
        context,
        base,
        primitive_column_scales=keywords["primitive_column_scales"],
        conservation_row_scales=keywords["conservation_row_scales"],
        parent_cell_indices=keywords["parent_cell_indices"],
        refinement_ratio=keywords["refinement_ratio"],
        exterior_parent_face=keywords["exterior_parent_face"],
        guard_end_parent_face=keywords["guard_end_parent_face"],
        parent_cell_count=keywords["parent_cell_count"],
        q3_target=reaction.q3_value,
        constraint_row_scales=reaction.q3_derivative_norms,
        reaction_channel_basis="frozen_normalized",
        elapsed_time_seconds=timestep,
        completed_steps=1,
        provenance={"fixture": "fixed_q"},
    )
    path = tmp_path / "fixed_q_restart.npz"
    save_causal_five_field_fixed_q_bdf_restart(path, context, restart)
    loaded = load_causal_five_field_fixed_q_bdf_restart(
        path,
        context,
        expected_provenance={"fixture": "fixed_q"},
    )
    assert causal_five_field_fixed_q_bdf_restarts_equal(restart, loaded)


def test_synthetic_equal_q_history_does_not_certify_increment_primary_bdf2() -> None:
    context, base, columns, _rows, keywords = _problem()
    tangent = causal_five_field_monolithic_frozen_tangent(
        context,
        base,
        primitive_column_scales=keywords["primitive_column_scales"],
        conservation_row_scales=keywords["conservation_row_scales"],
    )
    reaction = causal_five_field_fixed_q_reaction(context, base, **keywords)
    multiplier = -reaction.q3_scaled_derivative @ tangent.scaled_base_rate_per_s
    constrained_rate = (
        tangent.scaled_base_rate_per_s
        + reaction.reaction_lift @ multiplier
    )
    timestep = 1.0e-8
    previous_scaled_increment = -timestep * constrained_rate
    target_scale = np.maximum(
        np.abs(reaction.q3_value),
        np.finfo(float).tiny,
    )
    exterior_face = keywords["exterior_parent_face"]
    for _iteration in range(8):
        previous = base + columns * previous_scaled_increment.reshape(base.shape)
        q3, _factors = causal_five_field_exterior_q3(
            context,
            previous,
            exterior_face_index=exterior_face,
        )
        if np.max(np.abs((q3 - reaction.q3_value) / target_scale)) <= 1.0e-13:
            break
        previous_scaled_increment -= reaction.reaction_lift @ (
            (q3 - reaction.q3_value) / reaction.q3_derivative_norms
        )
    previous = base + columns * previous_scaled_increment.reshape(base.shape)
    previous_q3, _factors = causal_five_field_exterior_q3(
        context,
        previous,
        exterior_face_index=exterior_face,
    )
    assert (
        np.max(
            np.abs((previous_q3 - reaction.q3_value) / target_scale)
        )
        <= 1.0e-12
    )
    previous_storage = causal_five_field_monolithic_storage_increment(
        context,
        previous,
        base,
    )
    history = CausalFiveFieldMonolithicBDFHistory(
        previous_primitive_increment=base - previous,
        previous_mapped_storage_increment=(
            previous_storage.mapped_path_increment
        ),
        previous_responsive_height_storage_increment=(
            previous_storage.responsive_height_path_increment
        ),
        previous_timestep_seconds=timestep,
    )
    top_left = (
        1.5 * reaction.descriptor_scaled_matrix / timestep
        + tangent.evolving_scaled_jacobian
    )
    result = solve_causal_five_field_fixed_q_bdf(
        context,
        base,
        timestep,
        constrained_rate,
        multiplier,
        top_left,
        order=2,
        history=history,
        q3_target=reaction.q3_value,
        constraint_row_scales=reaction.q3_derivative_norms,
        reaction_channel_basis="frozen_normalized",
        reaction_channel_transform=reaction.raw_schur_inverse,
        maximum_scaled_primitive_change=1.0e-2,
        residual_tolerance=1.0e-8,
        maximum_newton_iterations=3,
        maximum_line_search_iterations=2,
        refresh_exact_jacobian=True,
        maximum_exact_jacobian_refreshes=2,
        **keywords,
    )
    assert not result.accepted
    assert result.order == 2
    assert "nonlinear_root" in result.acceptance.failure_reasons
    with pytest.raises(ValueError, match="rejected fixed-Q step"):
        causal_five_field_fixed_q_accepted_history(result)


def test_binding_solver_rejects_state_normalized_reaction_channels() -> None:
    context, base, _columns, _rows, keywords = _problem()
    dimensions = base.size
    with pytest.raises(ValueError, match="raw or frozen-normalized"):
        solve_causal_five_field_fixed_q_bdf(
            context,
            base,
            1.0e-8,
            np.zeros(dimensions),
            np.zeros(3),
            np.eye(dimensions),
            order=1,
            history=None,
            reaction_channel_basis="normalized",  # type: ignore[arg-type]
            **keywords,
        )


def test_reaction_schur_condition_gate_fails_closed() -> None:
    context, base, _columns, _rows, keywords = _problem()
    with pytest.raises(np.linalg.LinAlgError, match="condition gate"):
        causal_five_field_fixed_q_reaction(
            context,
            base,
            maximum_schur_condition_number=5.0e4,
            **keywords,
        )
