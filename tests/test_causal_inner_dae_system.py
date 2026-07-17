from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from imri_qpe.constants import C, G
from imri_qpe.layer3_minidisk_1d import (
    KERR_SCHILD_HILL_ENERGY_ZERO,
    CausalFiveFieldDAEContext,
    GasRadiationHillRocheNozzleProvider,
    KerrSchildCellSourceRates,
    SchwarzschildCurvatureVerticalFrequency,
    audit_causal_five_field_consistent_initial_data,
    audit_causal_five_field_dae_jacobian,
    audit_causal_five_field_reduced_stationary_response,
    causal_five_field_dae_count,
    causal_five_field_dae_scaling,
    causal_five_field_endpoint_temporal_storage_increment,
    causal_five_field_path_temporal_storage_increment,
    causal_five_field_reduced_backward_euler_residual,
    causal_five_field_reduced_stationary_residual,
    causal_five_field_state_from_primitives,
    evaluate_causal_five_field_dae,
    evaluate_causal_five_field_increment_backward_euler,
    fiducial_hill_roche_nozzle_geometry,
    make_causal_five_field_seed,
    make_kerr_schild_column_grid,
    pack_causal_five_field_state,
    unpack_causal_five_field_state,
)
from imri_qpe.parameters import FiducialParams


def _context(
    n_cells: int,
    *,
    cooling: bool = False,
) -> CausalFiveFieldDAEContext:
    mass = FiducialParams().M2_g
    gravitational_radius = G * mass / C**2
    grid = make_kerr_schild_column_grid(
        1.8 * gravitational_radius,
        335.0 * gravitational_radius,
        n_cells,
        gravitational_radius,
    )
    geometry = replace(
        fiducial_hill_roche_nozzle_geometry(),
        energy_zero=KERR_SCHILD_HILL_ENERGY_ZERO,
    )
    provider = GasRadiationHillRocheNozzleProvider(
        geometry,
        transverse_quadrature_zones=24,
    )
    return CausalFiveFieldDAEContext(
        grid=grid,
        vertical_frequency=SchwarzschildCurvatureVerticalFrequency(
            gravitational_radius
        ),
        outer_boundary_provider=provider,
        include_radiative_cooling=cooling,
    ).validated()


def test_flux_primary_state_pack_is_exact_and_round_trips() -> None:
    context = _context(4)
    state = make_causal_five_field_seed(context)
    packed = pack_causal_five_field_state(state)
    recovered = unpack_causal_five_field_state(packed, 4)
    count = causal_five_field_dae_count(4)

    assert packed.shape == (count.total_unknowns,)
    assert recovered.conserved == pytest.approx(state.conserved)
    assert recovered.primitives == pytest.approx(state.primitives)
    assert recovered.weighted_face_fluxes_over_c == pytest.approx(
        state.weighted_face_fluxes_over_c
    )


def test_seed_closes_primitive_and_all_face_maps_exactly() -> None:
    context = _context(8)
    state = make_causal_five_field_seed(context)
    evaluation = evaluate_causal_five_field_dae(
        pack_causal_five_field_state(state),
        context,
    )

    assert evaluation.primitive_map_rows == pytest.approx(
        np.zeros((8, 5)),
        abs=0.0,
    )
    assert evaluation.interior_flux_rows == pytest.approx(
        np.zeros((7, 5)),
        abs=0.0,
    )
    assert evaluation.inner_flux_rows == pytest.approx(
        np.zeros(5),
        abs=0.0,
    )
    assert evaluation.outer_flux_rows == pytest.approx(
        np.zeros(5),
        abs=0.0,
    )
    assert evaluation.outer_incoming_characteristics == 2
    assert not evaluation.outer_boundary_choked
    assert (
        evaluation.numerical_weighted_face_fluxes_over_c[-1, 4]
        == 0.0
    )


def test_stationary_conservation_rows_telescope_componentwise() -> None:
    context = _context(8)
    state = make_causal_five_field_seed(context)
    evaluation = evaluate_causal_five_field_dae(
        pack_causal_five_field_state(state),
        context,
    )
    expected = (
        state.weighted_face_fluxes_over_c[-1]
        - state.weighted_face_fluxes_over_c[0]
        - np.sum(evaluation.integrated_sources_per_ct, axis=0)
    )

    assert np.sum(evaluation.conservation_rows, axis=0) == pytest.approx(
        expected,
        rel=3.0e-15,
        abs=1.0e-12,
    )
    assert np.all(np.isfinite(evaluation.proper_shear_rates))
    assert np.all(evaluation.proper_shear_rates > 0.0)
    assert np.all(np.isfinite(evaluation.proper_log_height_rates))


def test_backward_euler_adds_all_killing_vertical_storage_components() -> None:
    context = _context(6)
    old_state = make_causal_five_field_seed(context)
    old_vector = pack_causal_five_field_state(old_state)
    new_primitives = np.array(old_state.primitives, copy=True)
    new_primitives[:, 3] += 1.0e-3
    new_state = causal_five_field_state_from_primitives(
        context,
        new_primitives,
    )
    evaluation = evaluate_causal_five_field_dae(
        pack_causal_five_field_state(new_state),
        context,
        old_vector=old_vector,
        timestep_seconds=2.0,
    )

    assert evaluation.temporal_vertical_storage[:, 0] == pytest.approx(
        np.zeros(6),
        abs=0.0,
    )
    assert np.all(
        np.linalg.norm(
            evaluation.temporal_vertical_storage[:, 1:],
            axis=1,
        )
        > 0.0
    )
    expected = (
        new_state.weighted_face_fluxes_over_c[-1]
        - new_state.weighted_face_fluxes_over_c[0]
        - np.sum(evaluation.integrated_sources_per_ct, axis=0)
        + np.sum(
            context.grid.cell_measures[:, None]
            * (new_state.conserved - old_state.conserved)
            / (2.0 * C),
            axis=0,
        )
    )
    expected[:4] += np.sum(
        evaluation.temporal_vertical_storage,
        axis=0,
    )
    assert np.sum(evaluation.conservation_rows, axis=0) == pytest.approx(
        expected,
        rel=5.0e-14,
        abs=1.0e-8,
    )


def test_diffusion_cooling_is_included_without_breaking_the_ledger() -> None:
    context = _context(4, cooling=True)
    state = make_causal_five_field_seed(context)
    evaluation = evaluate_causal_five_field_dae(
        pack_causal_five_field_state(state),
        context,
    )
    no_cooling = evaluate_causal_five_field_dae(
        pack_causal_five_field_state(state),
        replace(context, include_radiative_cooling=False),
    )

    assert np.all(evaluation.scattering_optical_depths > 1.0)
    assert np.all(np.isfinite(evaluation.integrated_sources_per_ct))
    assert np.all(
        evaluation.integrated_sources_per_ct[:, 3]
        < no_cooling.integrated_sources_per_ct[:, 3]
    )


def test_exact_stream_moments_enter_only_the_four_killing_rows() -> None:
    context = _context(4)
    state = make_causal_five_field_seed(context)
    vector = pack_causal_five_field_state(state)
    baseline = evaluate_causal_five_field_dae(vector, context)
    mass = np.asarray([0.0, 2.0e20, 3.0e20, 0.0])
    stream = KerrSchildCellSourceRates(
        rest_mass=mass,
        radial_momentum_over_c=0.25 * mass,
        angular_momentum_over_c=1.5e9 * mass,
        killing_energy_over_c2=1.01 * mass,
    )
    sourced = evaluate_causal_five_field_dae(
        vector,
        replace(context, stream_sources=stream),
    )
    expected = stream.weighted_killing_source_per_ct

    assert (
        sourced.integrated_sources_per_ct[:, :4]
        - baseline.integrated_sources_per_ct[:, :4]
        == pytest.approx(expected)
    )
    assert sourced.integrated_sources_per_ct[:, 4] == pytest.approx(
        baseline.integrated_sources_per_ct[:, 4]
    )
    assert (
        sourced.conservation_rows[:, :4]
        - baseline.conservation_rows[:, :4]
        == pytest.approx(-expected)
    )


def test_small_assembled_scaled_jacobian_is_numerically_full_rank() -> None:
    context = _context(2)
    state = make_causal_five_field_seed(context)
    vector = pack_causal_five_field_state(state)
    evaluation = evaluate_causal_five_field_dae(vector, context)
    scaling = causal_five_field_dae_scaling(state, evaluation)
    audit = audit_causal_five_field_dae_jacobian(
        lambda trial: evaluate_causal_five_field_dae(
            trial,
            context,
        ).residual,
        vector,
        scaling,
        rank_relative_threshold=1.0e-11,
    )

    assert audit.dimensions == (35, 35)
    assert audit.full_rank
    assert audit.smallest_singular_value > 1.0e-8


def test_reduced_stationary_residual_eliminates_exact_map_rows() -> None:
    context = _context(4)
    state = make_causal_five_field_seed(context)
    vector = pack_causal_five_field_state(state)
    evaluation = evaluate_causal_five_field_dae(vector, context)
    reduced = causal_five_field_reduced_stationary_residual(
        state.primitives.ravel(),
        context,
    )

    assert reduced == pytest.approx(evaluation.conservation_rows.ravel())
    assert evaluation.primitive_map_rows == pytest.approx(
        np.zeros((4, 5)),
        abs=0.0,
    )
    assert evaluation.interior_flux_rows == pytest.approx(
        np.zeros((3, 5)),
        abs=0.0,
    )
    assert evaluation.inner_flux_rows == pytest.approx(np.zeros(5), abs=0.0)
    assert evaluation.outer_flux_rows == pytest.approx(np.zeros(5), abs=0.0)


def test_reduced_stationary_response_matches_full_schur_complement() -> None:
    context = _context(2)
    state = make_causal_five_field_seed(context)
    vector = pack_causal_five_field_state(state)
    evaluation = evaluate_causal_five_field_dae(vector, context)
    scaling = causal_five_field_dae_scaling(state, evaluation)
    full = audit_causal_five_field_dae_jacobian(
        lambda trial: evaluate_causal_five_field_dae(
            trial,
            context,
        ).residual,
        vector,
        scaling,
        finite_difference_step=2.0e-6,
        rank_relative_threshold=1.0e-11,
    )
    reduced = audit_causal_five_field_reduced_stationary_response(
        context,
        state,
        full,
        scaling=scaling,
        finite_difference_step=2.0e-6,
        rank_relative_threshold=1.0e-11,
    )

    assert reduced.dimensions == (10, 10)
    assert reduced.algebraic_dimensions == (25, 25)
    assert reduced.algebraic_full_rank
    assert reduced.direct_scaled_jacobian == pytest.approx(
        reduced.schur_scaled_jacobian,
        rel=2.0e-6,
        abs=2.0e-8,
    )
    assert reduced.relative_frobenius_matrix_defect < 2.0e-7
    assert reduced.maximum_directional_relative_defect < 2.0e-5
    assert reduced.reconstructed_algebraic_residual_norm < 2.0e-12
    assert reduced.outer_thermal_stress.response_matrix.shape == (2, 2)
    assert reduced.outer_thermal_stress.interior_full_rank


def test_reduced_stationary_response_preserves_open_roche_active_set() -> None:
    context = _context(2)
    state = make_causal_five_field_seed(
        context,
        outer_surface_density=1.0e4,
        outer_temperature=1.0e6,
    )
    vector = pack_causal_five_field_state(state)
    evaluation = evaluate_causal_five_field_dae(vector, context)
    scaling = causal_five_field_dae_scaling(state, evaluation)
    full = audit_causal_five_field_dae_jacobian(
        lambda trial: evaluate_causal_five_field_dae(
            trial,
            context,
        ).residual,
        vector,
        scaling,
        finite_difference_step=2.0e-6,
        rank_relative_threshold=1.0e-11,
    )
    reduced = audit_causal_five_field_reduced_stationary_response(
        context,
        state,
        full,
        scaling=scaling,
        finite_difference_step=2.0e-6,
        rank_relative_threshold=1.0e-11,
    )

    assert evaluation.outer_boundary_choked
    assert reduced.outer_boundary_choked
    assert reduced.algebraic_full_rank
    assert reduced.relative_frobenius_matrix_defect < 2.0e-7


def test_consistent_initial_tangent_balances_storage_on_constraint_manifold() -> None:
    context = _context(2)
    state = make_causal_five_field_seed(context)
    vector = pack_causal_five_field_state(state)
    evaluation = evaluate_causal_five_field_dae(vector, context)
    scaling = causal_five_field_dae_scaling(state, evaluation)
    stationary = audit_causal_five_field_dae_jacobian(
        lambda trial: evaluate_causal_five_field_dae(
            trial,
            context,
        ).residual,
        vector,
        scaling,
        finite_difference_step=2.0e-6,
        rank_relative_threshold=1.0e-11,
    )
    backward_euler = audit_causal_five_field_dae_jacobian(
        lambda trial: evaluate_causal_five_field_dae(
            trial,
            context,
            old_vector=vector,
            timestep_seconds=1.0,
        ).residual,
        vector,
        scaling,
        finite_difference_step=2.0e-6,
        rank_relative_threshold=1.0e-11,
    )
    consistent = audit_causal_five_field_consistent_initial_data(
        context,
        state,
        stationary,
        backward_euler,
        scaling=scaling,
        descriptor_timestep_seconds=1.0,
        rank_relative_threshold=1.0e-11,
    )
    reduced_backward_euler = (
        causal_five_field_reduced_backward_euler_residual(
            state.primitives.ravel(),
            context,
            old_vector=vector,
            timestep_seconds=1.0,
        )
    )

    assert consistent.dimensions == (35, 35)
    assert consistent.full_rank
    assert consistent.descriptor_dimensions == (10, 35)
    assert consistent.descriptor_full_row_rank
    assert consistent.maximum_initial_algebraic_residual == 0.0
    assert consistent.maximum_scaled_consistency_residual < 2.0e-11
    assert consistent.storage_balance_residual_norm < 2.0e-10
    assert consistent.algebraic_tangent_residual_norm < 2.0e-10
    assert reduced_backward_euler == pytest.approx(
        evaluation.conservation_rows.ravel()
    )


def test_increment_primary_zero_state_preserves_stationary_constraints() -> None:
    context = _context(2)
    old_state = make_causal_five_field_seed(context)
    old_vector = pack_causal_five_field_state(old_state)
    stationary = evaluate_causal_five_field_dae(old_vector, context)
    count = causal_five_field_dae_count(2)
    evaluation = evaluate_causal_five_field_increment_backward_euler(
        np.zeros(count.total_unknowns),
        context,
        old_vector=old_vector,
        timestep_seconds=1.0,
    )

    assert evaluation.conservation_rows == pytest.approx(
        stationary.conservation_rows
    )
    assert evaluation.primitive_map_rows == pytest.approx(
        np.zeros((2, 5)),
        abs=0.0,
    )
    assert evaluation.interior_flux_rows == pytest.approx(
        np.zeros((1, 5)),
        abs=0.0,
    )
    assert evaluation.inner_flux_rows == pytest.approx(
        np.zeros(5),
        abs=0.0,
    )
    assert evaluation.outer_flux_rows == pytest.approx(
        np.zeros(5),
        abs=0.0,
    )
    assert evaluation.temporal_conserved_storage == pytest.approx(
        np.zeros((2, 5)),
        abs=0.0,
    )
    assert evaluation.temporal_vertical_storage == pytest.approx(
        np.zeros((2, 4)),
        abs=0.0,
    )


def test_increment_primary_storage_uses_declared_conserved_increment() -> None:
    context = _context(2)
    old_state = make_causal_five_field_seed(context)
    old_vector = pack_causal_five_field_state(old_state)
    count = causal_five_field_dae_count(2)
    increment = np.zeros(count.total_unknowns)
    declared = abs(old_state.conserved[1, 2]) * 1.0e-12
    increment[7] = declared
    timestep = 3.0e-4
    evaluation = evaluate_causal_five_field_increment_backward_euler(
        increment,
        context,
        old_vector=old_vector,
        timestep_seconds=timestep,
    )
    expected = (
        context.grid.cell_measures[1]
        * declared
        / (C * timestep)
    )

    assert evaluation.temporal_conserved_storage[1, 2] == expected
    assert np.count_nonzero(evaluation.temporal_conserved_storage) == 1
    assert evaluation.temporal_vertical_storage == pytest.approx(
        np.zeros((2, 4)),
        abs=0.0,
    )


def test_increment_primary_matches_endpoint_form_at_resolved_increment() -> None:
    context = _context(2)
    old_state = make_causal_five_field_seed(context)
    old_vector = pack_causal_five_field_state(old_state)
    new_primitives = np.array(old_state.primitives, copy=True)
    new_primitives[:, 0] += 1.0e-4
    new_primitives[:, 1] += 2.0e-5
    new_primitives[:, 2] -= 1.0e-5
    new_primitives[:, 3] += 3.0e-4
    new_primitives[:, 4] *= 1.0002
    new_state = causal_five_field_state_from_primitives(
        context,
        new_primitives,
    )
    new_vector = pack_causal_five_field_state(new_state)
    timestep = 1.0e-3
    endpoint = evaluate_causal_five_field_dae(
        new_vector,
        context,
        old_vector=old_vector,
        timestep_seconds=timestep,
        temporal_storage_scheme="endpoint",
    )
    increment = evaluate_causal_five_field_increment_backward_euler(
        new_vector - old_vector,
        context,
        old_vector=old_vector,
        timestep_seconds=timestep,
        temporal_height_scheme="endpoint",
    )

    assert increment.residual == pytest.approx(
        endpoint.residual,
        rel=2.0e-13,
        abs=1.0e-12,
    )


def test_increment_primary_backward_euler_is_full_rank() -> None:
    context = _context(2)
    old_state = make_causal_five_field_seed(context)
    old_vector = pack_causal_five_field_state(old_state)
    stationary = evaluate_causal_five_field_dae(old_vector, context)
    scaling = causal_five_field_dae_scaling(old_state, stationary)
    count = causal_five_field_dae_count(2)
    audit = audit_causal_five_field_dae_jacobian(
        lambda increment: (
            evaluate_causal_five_field_increment_backward_euler(
                increment,
                context,
                old_vector=old_vector,
                timestep_seconds=1.0,
            ).residual
        ),
        np.zeros(count.total_unknowns),
        scaling,
        finite_difference_step=2.0e-6,
        rank_relative_threshold=1.0e-11,
    )

    assert audit.dimensions == (35, 35)
    assert audit.full_rank


def test_path_storage_recovers_tiny_rest_mass_increment() -> None:
    context = _context(2)
    old = make_causal_five_field_seed(context).primitives
    new = np.array(old, copy=True)
    new[0, 0] += 1.0e-12
    actual_log_increment = new[0, 0] - old[0, 0]
    endpoint = causal_five_field_endpoint_temporal_storage_increment(
        context,
        old,
        new,
    )
    path = causal_five_field_path_temporal_storage_increment(
        context,
        old,
        new,
        quadrature_order=4,
    )
    lorentz = 1.0 / np.sqrt(
        1.0 - old[0, 1] ** 2 - old[0, 2] ** 2
    )
    expected = (
        np.exp(old[0, 0])
        * lorentz
        * np.expm1(actual_log_increment)
    )

    path_error = abs(path.conserved_increment[0, 0] - expected)
    endpoint_error = abs(
        endpoint.conserved_increment[0, 0] - expected
    )
    assert path.conserved_increment[0, 0] == pytest.approx(
        expected,
        rel=3.0e-12,
    )
    assert path_error < 1.0e-6 * endpoint_error


def test_path_storage_is_smooth_under_tiny_endpoint_correction() -> None:
    context = _context(2)
    old = make_causal_five_field_seed(context).primitives
    candidate = np.array(old, copy=True)
    candidate[0, 0] += 1.0e-3
    corrected = np.array(candidate, copy=True)
    corrected[0, 0] += 1.0e-12
    actual_correction = corrected[0, 0] - candidate[0, 0]

    base = causal_five_field_path_temporal_storage_increment(
        context,
        old,
        candidate,
    )
    updated = causal_five_field_path_temporal_storage_increment(
        context,
        old,
        corrected,
    )
    lorentz = 1.0 / np.sqrt(
        1.0 - candidate[0, 1] ** 2 - candidate[0, 2] ** 2
    )
    expected = (
        np.exp(candidate[0, 0])
        * lorentz
        * np.expm1(actual_correction)
    )
    actual = (
        updated.conserved_increment[0, 0]
        - base.conserved_increment[0, 0]
    )

    assert actual == pytest.approx(expected, rel=5.0e-4)


def test_path_storage_tiny_endpoint_change_matches_directional_response() -> None:
    context = _context(2)
    old = make_causal_five_field_seed(context).primitives
    candidate = np.array(old, copy=True)
    candidate[:, 0] += 1.0e-4
    candidate[:, 1] += 2.0e-5
    candidate[:, 2] -= 1.0e-5
    candidate[:, 3] += 3.0e-4
    candidate[:, 4] *= 1.0002
    direction = np.tile(
        np.asarray([0.3, 0.1, -0.2, 0.4, 0.2]),
        (2, 1),
    )
    direction[:, 4] *= np.maximum(np.abs(old[:, 4]), 1.0e-14)
    endpoint_step = 5.0e-12
    directional_step = 2.0e-6

    base = causal_five_field_path_temporal_storage_increment(
        context,
        old,
        candidate,
    )
    corrected = causal_five_field_path_temporal_storage_increment(
        context,
        old,
        candidate + endpoint_step * direction,
    )
    minus = causal_five_field_path_temporal_storage_increment(
        context,
        old,
        candidate - directional_step * direction,
    )
    plus = causal_five_field_path_temporal_storage_increment(
        context,
        old,
        candidate + directional_step * direction,
    )
    actual = corrected.conserved_increment - base.conserved_increment
    predicted = (
        endpoint_step
        * (plus.conserved_increment - minus.conserved_increment)
        / (2.0 * directional_step)
    )
    scale = max(float(np.max(np.abs(predicted))), 1.0e-30)

    assert np.max(np.abs(actual - predicted)) / scale < 2.0e-3


def test_path_storage_quadrature_converges_for_all_components() -> None:
    context = _context(2)
    old = make_causal_five_field_seed(context).primitives
    new = np.array(old, copy=True)
    new[:, 0] += 1.0e-4
    new[:, 1] += 2.0e-5
    new[:, 2] -= 1.0e-5
    new[:, 3] += 3.0e-4
    new[:, 4] *= 1.0002
    path4 = causal_five_field_path_temporal_storage_increment(
        context,
        old,
        new,
        quadrature_order=4,
    )
    path8 = causal_five_field_path_temporal_storage_increment(
        context,
        old,
        new,
        quadrature_order=8,
    )
    conserved_scale = np.maximum(
        np.abs(path8.conserved_increment),
        1.0e-20,
    )
    vertical_scale = np.maximum(
        np.abs(path8.vertical_killing_increment),
        1.0e-30,
    )

    assert np.max(
        np.abs(
            path4.conserved_increment - path8.conserved_increment
        )
        / conserved_scale
    ) < 2.0e-10
    assert np.max(
        np.abs(
            path4.vertical_killing_increment
            - path8.vertical_killing_increment
        )
        / vertical_scale
    ) < 2.0e-10
    assert path4.vertical_killing_increment[:, 0] == pytest.approx(
        np.zeros(2),
        abs=0.0,
    )


def test_path_integrated_backward_euler_preserves_exact_maps() -> None:
    context = _context(2)
    old_state = make_causal_five_field_seed(context)
    old_vector = pack_causal_five_field_state(old_state)
    new_primitives = np.array(old_state.primitives, copy=True)
    new_primitives[:, 3] += 1.0e-4
    new_state = causal_five_field_state_from_primitives(
        context,
        new_primitives,
    )
    evaluation = evaluate_causal_five_field_dae(
        pack_causal_five_field_state(new_state),
        context,
        old_vector=old_vector,
        timestep_seconds=1.0e-3,
        temporal_storage_scheme="path_integrated",
    )

    assert evaluation.primitive_map_rows == pytest.approx(
        np.zeros((2, 5)),
        abs=0.0,
    )
    assert evaluation.interior_flux_rows == pytest.approx(
        np.zeros((1, 5)),
        abs=0.0,
    )
    assert evaluation.inner_flux_rows == pytest.approx(
        np.zeros(5),
        abs=0.0,
    )
    assert evaluation.outer_flux_rows == pytest.approx(
        np.zeros(5),
        abs=0.0,
    )
    assert np.all(
        np.isfinite(evaluation.temporal_conserved_storage)
    )
    assert np.all(np.isfinite(evaluation.temporal_vertical_storage))
