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
    audit_causal_five_field_dae_jacobian,
    causal_five_field_dae_count,
    causal_five_field_dae_scaling,
    causal_five_field_state_from_primitives,
    evaluate_causal_five_field_dae,
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
