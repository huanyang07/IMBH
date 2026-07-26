from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from imri_qpe.constants import C, G
from imri_qpe.layer3_minidisk_1d import (
    CAUSAL_FIVE_FIELD_INNER_TRACE_OVERRIDES,
    CAUSAL_FIVE_FIELD_OUTER_BOUNDARY_FLUX_MODES,
    CAUSAL_FIVE_FIELD_RECONSTRUCTION_PURPOSES,
    CAUSAL_FIVE_FIELD_SPATIAL_RECONSTRUCTIONS,
    KERR_SCHILD_HILL_ENERGY_ZERO,
    CausalFiveFieldDAEContext,
    GasRadiationHillRocheNozzleProvider,
    KerrSchildCellSourceRates,
    SchwarzschildCurvatureVerticalFrequency,
    ValenciaPerfectFluidPrimitive,
    audit_causal_five_field_consistent_initial_data,
    audit_causal_five_field_dae_jacobian,
    audit_causal_five_field_reduced_stationary_response,
    causal_five_field_colored_central_jacobian,
    causal_five_field_dae_count,
    causal_five_field_dae_jacobian_color_groups,
    causal_five_field_dae_jacobian_sparsity,
    causal_five_field_dae_scaling,
    causal_five_field_equilibrated_sparse_solve,
    causal_five_field_endpoint_temporal_storage_increment,
    causal_five_field_face_flux_decomposition,
    causal_five_field_path_temporal_storage_increment,
    causal_five_field_reduced_backward_euler_residual,
    causal_five_field_reduced_stationary_residual,
    causal_five_field_reconstruct_face_charts,
    causal_five_field_state_from_primitives,
    evaluate_causal_five_field_dae,
    evaluate_causal_five_field_increment_backward_euler,
    exact_kerr_schild_compact_stream_sources,
    fiducial_hill_roche_nozzle_geometry,
    kerr_schild_column_geometry,
    kerr_schild_stream_injection,
    make_causal_five_field_seed,
    make_kerr_schild_column_grid,
    pack_causal_five_field_state,
    unpack_causal_five_field_state,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot


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


def test_spatial_reconstruction_mode_is_validated() -> None:
    context = _context(4)

    assert (
        context.spatial_reconstruction
        == CAUSAL_FIVE_FIELD_SPATIAL_RECONSTRUCTIONS[0]
    )
    with pytest.raises(ValueError):
        replace(
            context,
            spatial_reconstruction="unsupported",
        ).validated()
    for name in (
        "boundary_trace_reconstruction",
        "inner_boundary_trace_override",
        "inner_flux_trace_override",
        "inner_storage_trace_override",
        "cell_rate_scheme",
        "cell_source_quadrature",
        "cell_storage_quadrature",
    ):
        with pytest.raises(ValueError):
            replace(context, **{name: "unsupported"}).validated()
    with pytest.raises(
        ValueError,
        match="second-order causal spatial stencils require three cells",
    ):
        replace(
            _context(2),
            spatial_reconstruction="quadratic_admissible",
        ).validated()
    with pytest.raises(
        ValueError,
        match="second-order causal spatial stencils require three cells",
    ):
        causal_five_field_dae_jacobian_sparsity(
            2,
            spatial_reconstruction="quadratic_admissible",
        )


def test_inner_trace_overrides_change_only_the_inner_face() -> None:
    context = replace(
        _context(8),
        spatial_reconstruction="quadratic_admissible",
        boundary_trace_reconstruction="plm_one_sided",
    ).validated()
    seed = make_causal_five_field_seed(context)
    inherited = causal_five_field_reconstruct_face_charts(
        context,
        seed.primitives,
    )
    assert CAUSAL_FIVE_FIELD_INNER_TRACE_OVERRIDES == (
        "inherit",
        "cell_centered",
        "linear_outgoing",
    )
    for mode in ("cell_centered", "linear_outgoing"):
        candidate = causal_five_field_reconstruct_face_charts(
            replace(
                context,
                inner_boundary_trace_override=mode,
            ).validated(),
            seed.primitives,
        )
        assert np.array_equal(
            candidate.left_face_charts[1:],
            inherited.left_face_charts[1:],
        )
        assert np.array_equal(
            candidate.right_face_charts[1:],
            inherited.right_face_charts[1:],
        )
    centered = causal_five_field_reconstruct_face_charts(
        replace(
            context,
            inner_boundary_trace_override="cell_centered",
        ).validated(),
        seed.primitives,
    )
    assert np.array_equal(
        centered.right_face_charts[0],
        seed.primitives[0],
    )


def test_flux_and_storage_inner_trace_overrides_are_independent() -> None:
    context = replace(
        _context(8),
        spatial_reconstruction="quadratic_admissible",
        boundary_trace_reconstruction="plm_one_sided",
        cell_storage_quadrature="gauss_legendre_4",
    ).validated()
    seed = make_causal_five_field_seed(context)
    assert CAUSAL_FIVE_FIELD_RECONSTRUCTION_PURPOSES == (
        "flux",
        "storage",
    )
    inherited = causal_five_field_state_from_primitives(
        context,
        seed.primitives,
    )
    flux_only = causal_five_field_state_from_primitives(
        replace(
            context,
            inner_flux_trace_override="cell_centered",
        ).validated(),
        seed.primitives,
    )
    storage_only = causal_five_field_state_from_primitives(
        replace(
            context,
            inner_storage_trace_override="cell_centered",
        ).validated(),
        seed.primitives,
    )

    np.testing.assert_array_equal(flux_only.conserved, inherited.conserved)
    assert not np.array_equal(
        flux_only.weighted_face_fluxes_over_c[0],
        inherited.weighted_face_fluxes_over_c[0],
    )
    np.testing.assert_array_equal(
        storage_only.weighted_face_fluxes_over_c,
        inherited.weighted_face_fluxes_over_c,
    )
    assert not np.array_equal(storage_only.conserved[0], inherited.conserved[0])

    combined = causal_five_field_state_from_primitives(
        replace(
            context,
            inner_boundary_trace_override="linear_outgoing",
        ).validated(),
        seed.primitives,
    )
    separated = causal_five_field_state_from_primitives(
        replace(
            context,
            inner_flux_trace_override="linear_outgoing",
            inner_storage_trace_override="linear_outgoing",
        ).validated(),
        seed.primitives,
    )
    np.testing.assert_array_equal(combined.conserved, separated.conserved)
    np.testing.assert_array_equal(
        combined.weighted_face_fluxes_over_c,
        separated.weighted_face_fluxes_over_c,
    )

    updated = np.array(seed.primitives, copy=True)
    updated[:, 3] += 1.0e-5
    inherited_path = causal_five_field_path_temporal_storage_increment(
        context,
        seed.primitives,
        updated,
    )
    for candidate_context in (
        replace(
            context,
            inner_flux_trace_override="cell_centered",
        ).validated(),
        replace(
            context,
            inner_storage_trace_override="cell_centered",
        ).validated(),
    ):
        candidate_path = (
            causal_five_field_path_temporal_storage_increment(
                candidate_context,
                seed.primitives,
                updated,
            )
        )
        np.testing.assert_array_equal(
            candidate_path.vertical_killing_increment,
            inherited_path.vertical_killing_increment,
        )
        np.testing.assert_array_equal(
            candidate_path.vertical_work_per_area,
            inherited_path.vertical_work_per_area,
        )


def test_frozen_exterior_rusanov_boundary_is_explicit_and_audit_only() -> None:
    context = _context(4)
    seed = make_causal_five_field_seed(context)
    assert CAUSAL_FIVE_FIELD_OUTER_BOUNDARY_FLUX_MODES == (
        "roche",
        "frozen_exterior_rusanov",
    )
    with pytest.raises(
        ValueError,
        match="unsupported causal five-field outer boundary flux mode",
    ):
        replace(
            context,
            outer_boundary_flux_mode="unsupported",
        ).validated()
    with pytest.raises(
        ValueError,
        match="requires one finite five-field primitive chart",
    ):
        replace(
            context,
            outer_boundary_flux_mode="frozen_exterior_rusanov",
        ).validated()
    with pytest.raises(
        ValueError,
        match="physical Roche boundary cannot carry",
    ):
        replace(
            context,
            outer_boundary_frozen_exterior_chart=np.array(
                seed.primitives[-1],
                copy=True,
            ),
        ).validated()

    audit_context = replace(
        context,
        outer_boundary_flux_mode="frozen_exterior_rusanov",
        outer_boundary_frozen_exterior_chart=np.array(
            seed.primitives[-1],
            copy=True,
        ),
    ).validated()
    audit_state = causal_five_field_state_from_primitives(
        audit_context,
        seed.primitives,
    )
    evaluation = evaluate_causal_five_field_dae(
        pack_causal_five_field_state(audit_state),
        audit_context,
    )

    assert evaluation.outer_boundary_choked is False
    assert evaluation.outer_incoming_characteristics == 5
    assert np.array_equal(evaluation.primitive_map_rows, np.zeros((4, 5)))
    assert np.array_equal(evaluation.interior_flux_rows, np.zeros((3, 5)))
    assert np.array_equal(evaluation.inner_flux_rows, np.zeros(5))
    assert np.array_equal(evaluation.outer_flux_rows, np.zeros(5))


def test_explicit_piecewise_constant_backend_is_bitwise_frozen() -> None:
    context = _context(8)
    explicit = replace(
        context,
        spatial_reconstruction="piecewise_constant",
    ).validated()
    state = make_causal_five_field_seed(context)
    vector = pack_causal_five_field_state(state)

    default_evaluation = evaluate_causal_five_field_dae(vector, context)
    explicit_evaluation = evaluate_causal_five_field_dae(vector, explicit)

    assert np.array_equal(
        default_evaluation.residual,
        explicit_evaluation.residual,
    )
    assert np.array_equal(
        default_evaluation.numerical_weighted_face_fluxes_over_c,
        explicit_evaluation.numerical_weighted_face_fluxes_over_c,
    )
    assert np.array_equal(
        default_evaluation.central_weighted_face_fluxes_over_c,
        explicit_evaluation.central_weighted_face_fluxes_over_c,
    )
    assert np.array_equal(
        default_evaluation.rusanov_dissipation_weighted_face_fluxes_over_c,
        explicit_evaluation.rusanov_dissipation_weighted_face_fluxes_over_c,
    )


@pytest.mark.parametrize(
    "mode",
    (
        "plm_unlimited",
        "plm_smooth",
        "quadratic_admissible",
    ),
)
def test_plm_reconstruction_preserves_constant_charts(mode: str) -> None:
    context = replace(
        _context(8),
        spatial_reconstruction=mode,
    ).validated()
    chart = np.asarray(
        [np.log(120.0), 0.02, 0.12, np.log(4.0e6), 0.0],
        dtype=float,
    )
    charts = np.repeat(chart[None, :], 8, axis=0)
    reconstruction = causal_five_field_reconstruct_face_charts(
        context,
        charts,
    )

    assert reconstruction.unlimited_slopes == pytest.approx(
        np.zeros((8, 5)),
        abs=0.0,
    )
    assert reconstruction.limited_slopes == pytest.approx(
        np.zeros((8, 5)),
        abs=0.0,
    )
    assert reconstruction.left_face_charts == pytest.approx(
        np.repeat(chart[None, :], 9, axis=0),
        abs=0.0,
    )
    assert reconstruction.right_face_charts == pytest.approx(
        np.repeat(chart[None, :], 9, axis=0),
        abs=0.0,
    )
    state = causal_five_field_state_from_primitives(context, charts)
    evaluation = evaluate_causal_five_field_dae(
        pack_causal_five_field_state(state),
        context,
    )
    assert evaluation.rusanov_dissipation_weighted_face_fluxes_over_c[
        1:-1
    ] == pytest.approx(np.zeros((7, 5)), abs=0.0)


def test_unlimited_plm_is_exact_for_linear_log_radius_chart() -> None:
    context = replace(
        _context(8),
        spatial_reconstruction="plm_unlimited",
    ).validated()
    log_centers = np.log(context.grid.centers)
    log_edges = np.log(context.grid.edges)
    origin = float(np.mean(log_centers))
    intercept = np.asarray(
        [np.log(120.0), 0.02, 0.12, np.log(4.0e6), 1.0e12],
        dtype=float,
    )
    slope = np.asarray([0.04, 0.002, -0.003, -0.02, 2.0e10])
    charts = intercept + (log_centers - origin)[:, None] * slope
    reconstruction = causal_five_field_reconstruct_face_charts(
        context,
        charts,
    )
    exact = intercept + (log_edges - origin)[:, None] * slope

    assert reconstruction.unlimited_slopes[1:-1] == pytest.approx(
        np.repeat(slope[None, :], 6, axis=0),
        rel=2.0e-13,
        abs=2.0e-13,
    )
    assert reconstruction.left_face_charts[2:-2] == pytest.approx(
        exact[2:-2],
        rel=2.0e-13,
        abs=2.0e-13,
    )
    assert reconstruction.right_face_charts[2:-2] == pytest.approx(
        exact[2:-2],
        rel=2.0e-13,
        abs=2.0e-13,
    )


def test_one_sided_plm_is_exact_on_every_face_for_linear_log_chart() -> None:
    context = replace(
        _context(8),
        spatial_reconstruction="plm_unlimited",
        boundary_trace_reconstruction="plm_one_sided",
    ).validated()
    log_centers = np.log(context.grid.centers)
    log_edges = np.log(context.grid.edges)
    origin = float(np.mean(log_centers))
    intercept = np.asarray(
        [np.log(120.0), 0.02, 0.12, np.log(4.0e6), 1.0e12],
        dtype=float,
    )
    slope = np.asarray([0.04, 0.002, -0.003, -0.02, 2.0e10])
    charts = intercept + (log_centers - origin)[:, None] * slope
    exact = intercept + (log_edges - origin)[:, None] * slope
    reconstruction = causal_five_field_reconstruct_face_charts(
        context,
        charts,
    )

    np.testing.assert_allclose(
        reconstruction.left_face_charts,
        exact,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        reconstruction.right_face_charts,
        exact,
        rtol=2.0e-13,
        atol=2.0e-13,
    )


def test_quadratic_reconstruction_is_exact_for_quadratic_log_chart() -> None:
    context = replace(
        _context(8),
        spatial_reconstruction="quadratic_admissible",
        boundary_trace_reconstruction="plm_one_sided",
    ).validated()
    log_centers = np.log(context.grid.centers)
    log_edges = np.log(context.grid.edges)
    origin = float(np.mean(log_centers))
    intercept = np.asarray(
        [np.log(120.0), 0.02, 0.12, np.log(4.0e6), 0.0],
        dtype=float,
    )
    slope = np.asarray([0.04, 0.002, -0.003, -0.02, 0.0])
    curvature = np.asarray(
        [0.002, -0.0001, 0.0002, 0.001, 0.0],
    )
    center_offset = log_centers - origin
    edge_offset = log_edges - origin
    charts = (
        intercept
        + center_offset[:, None] * slope
        + center_offset[:, None] ** 2 * curvature
    )
    exact = (
        intercept
        + edge_offset[:, None] * slope
        + edge_offset[:, None] ** 2 * curvature
    )
    reconstruction = causal_five_field_reconstruct_face_charts(
        context,
        charts,
    )

    np.testing.assert_allclose(
        reconstruction.left_face_charts,
        exact,
        rtol=3.0e-13,
        atol=3.0e-13,
    )
    np.testing.assert_allclose(
        reconstruction.right_face_charts,
        exact,
        rtol=3.0e-13,
        atol=3.0e-13,
    )
    np.testing.assert_array_equal(
        reconstruction.admissibility_factors,
        np.ones(8),
    )


def test_quadratic_reconstruction_couples_active_admissibility() -> None:
    context = replace(
        _context(8),
        spatial_reconstruction="quadratic_admissible",
    ).validated()
    chart = np.asarray(
        [np.log(120.0), 0.0, 0.02, np.log(4.0e6), 0.0],
    )
    charts = np.repeat(chart[None, :], 8, axis=0)
    charts[:, 1] = np.asarray(
        [
            0.5163577636618574,
            0.7707262643831303,
            0.4275776096593312,
            -0.15600738289988691,
            -0.8132799375132267,
            -0.8028076837709259,
            0.7624065872330135,
            -0.6231457513536676,
        ]
    )
    reconstruction = causal_five_field_reconstruct_face_charts(
        context,
        charts,
    )

    assert np.min(reconstruction.admissibility_factors) < 0.99
    assert np.min(reconstruction.admissibility_factors) > 0.98
    for faces in (
        reconstruction.left_face_charts,
        reconstruction.right_face_charts,
    ):
        velocity_squared = faces[:, 1] ** 2 + faces[:, 2] ** 2
        assert np.max(velocity_squared) < 1.0


def test_high_order_endogenous_quadrature_leaves_exact_stream_unchanged() -> None:
    context = _context(16, cooling=True)
    mass = FiducialParams().M2_g
    gravitational_radius = context.grid.gravitational_radius
    radius = 240.0 * gravitational_radius
    geometry = kerr_schild_column_geometry(radius, gravitational_radius)
    primitive = ValenciaPerfectFluidPrimitive(
        surface_density=1.0e5,
        radial_velocity_over_c=2.0 * gravitational_radius / radius,
        azimuthal_velocity_over_c=0.05,
        specific_internal_energy=1.0e18,
        integrated_pressure=1.0e20,
    )
    injection = kerr_schild_stream_injection(
        geometry,
        primitive,
        rest_mass_rate=5.0 * eddington_mdot(mass),
    )
    stream = exact_kerr_schild_compact_stream_sources(
        context.grid,
        injection,
        center=radius,
        log_width=0.08,
        shape="compact_c2",
    )
    midpoint = replace(
        context,
        stream_sources=stream,
        spatial_reconstruction="plm_smooth",
    ).validated()
    high_order = replace(
        midpoint,
        spatial_reconstruction="quadratic_admissible",
        boundary_trace_reconstruction="plm_one_sided",
        cell_source_quadrature="gauss_legendre_4_local_rates",
        cell_storage_quadrature="gauss_legendre_4",
    ).validated()
    state = make_causal_five_field_seed(midpoint)
    midpoint_vector = pack_causal_five_field_state(
        causal_five_field_state_from_primitives(
            midpoint,
            state.primitives,
        )
    )
    high_order_vector = pack_causal_five_field_state(
        causal_five_field_state_from_primitives(
            high_order,
            state.primitives,
        )
    )
    midpoint_evaluation = evaluate_causal_five_field_dae(
        midpoint_vector,
        midpoint,
    )
    high_order_evaluation = evaluate_causal_five_field_dae(
        high_order_vector,
        high_order,
    )

    np.testing.assert_array_equal(
        high_order_evaluation.integrated_source_components_per_ct[
            "stream"
        ],
        midpoint_evaluation.integrated_source_components_per_ct["stream"],
    )
    reconstructed = np.sum(
        np.asarray(
            list(
                high_order_evaluation
                .integrated_source_components_per_ct.values()
            )
        ),
        axis=0,
    )
    np.testing.assert_allclose(
        reconstructed,
        high_order_evaluation.integrated_sources_per_ct,
        rtol=2.0e-15,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "mode",
    (
        "plm_unlimited",
        "plm_smooth",
        "quadratic_admissible",
    ),
)
def test_plm_seed_closes_all_algebraic_maps(mode: str) -> None:
    context = replace(
        _context(8),
        spatial_reconstruction=mode,
    ).validated()
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


def test_rusanov_flux_components_reconstruct_production_flux() -> None:
    context = _context(8)
    state = make_causal_five_field_seed(context)
    evaluation = evaluate_causal_five_field_dae(
        pack_causal_five_field_state(state),
        context,
    )
    reconstructed = (
        evaluation.central_weighted_face_fluxes_over_c
        + evaluation.rusanov_dissipation_weighted_face_fluxes_over_c
    )
    scale = np.maximum(
        np.abs(evaluation.numerical_weighted_face_fluxes_over_c),
        1.0,
    )

    assert np.max(
        np.abs(
            reconstructed
            - evaluation.numerical_weighted_face_fluxes_over_c
        )
        / scale
    ) < 2.0e-16
    assert evaluation.rusanov_dissipation_weighted_face_fluxes_over_c[
        [0, -1]
    ] == pytest.approx(np.zeros((2, 5)), abs=0.0)
    assert np.any(
        evaluation.rusanov_dissipation_weighted_face_fluxes_over_c[
            1:-1
        ]
        != 0.0
    )


def test_physical_face_flux_decomposition_reconstructs_production() -> None:
    context = _context(8)
    state = make_causal_five_field_seed(context)
    vector = pack_causal_five_field_state(state)
    split = causal_five_field_face_flux_decomposition(context, vector)
    reconstructed = (
        split.central_perfect_weighted_face_fluxes_over_c
        + split.central_stress_weighted_face_fluxes_over_c
        + split.rusanov_weighted_face_fluxes_over_c
    )

    np.testing.assert_array_equal(split.face_indices, np.arange(1, 8))
    np.testing.assert_allclose(
        reconstructed,
        split.numerical_weighted_face_fluxes_over_c,
        rtol=2.0e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        split.numerical_weighted_face_fluxes_over_c,
        split.production_weighted_face_fluxes_over_c,
        rtol=2.0e-15,
        atol=0.0,
    )
    assert split.maximum_production_reconstruction_defect < 2.0e-15
    assert np.all(
        split.central_stress_weighted_face_fluxes_over_c[:, 0] == 0.0
    )
    assert np.all(
        split.central_stress_weighted_face_fluxes_over_c[:, 4] == 0.0
    )


def test_fixed_plateau_seed_samples_one_common_continuum_profile() -> None:
    context16 = _context(16)
    context32 = _context(32)
    gravitational_radius = context16.grid.gravitational_radius
    kwargs = {
        "inner_surface_density": 120.0,
        "outer_surface_density": 8.0e4,
        "inner_temperature": 4.2e6,
        "outer_temperature": 9.0e5,
        "profile_inner_plateau_radius": (
            6.0 * gravitational_radius
        ),
        "profile_outer_plateau_radius": (
            240.0 * gravitational_radius
        ),
    }
    state16 = make_causal_five_field_seed(context16, **kwargs)
    state32 = make_causal_five_field_seed(context32, **kwargs)

    np.testing.assert_array_equal(
        state16.primitives[0, :4],
        state32.primitives[0, :4],
    )
    np.testing.assert_array_equal(
        state16.primitives[-1, :4],
        state32.primitives[-1, :4],
    )
    for context, state in (
        (context16, state16),
        (context32, state32),
    ):
        coordinate = np.clip(
            (
                np.log(
                    context.grid.centers
                    / kwargs["profile_inner_plateau_radius"]
                )
                / np.log(
                    kwargs["profile_outer_plateau_radius"]
                    / kwargs["profile_inner_plateau_radius"]
                )
            ),
            0.0,
            1.0,
        )
        fraction = coordinate**3 * (
            10.0 - 15.0 * coordinate + 6.0 * coordinate**2
        )
        expected_log_sigma = (
            (1.0 - fraction)
            * np.log(kwargs["inner_surface_density"])
            + fraction * np.log(kwargs["outer_surface_density"])
        )
        expected_log_temperature = (
            (1.0 - fraction)
            * np.log(kwargs["inner_temperature"])
            + fraction * np.log(kwargs["outer_temperature"])
        )
        np.testing.assert_allclose(
            state.primitives[:, 0],
            expected_log_sigma,
            rtol=0.0,
            atol=2.0e-15,
        )
        np.testing.assert_allclose(
            state.primitives[:, 3],
            expected_log_temperature,
            rtol=0.0,
            atol=2.0e-15,
        )


def test_fixed_plateau_seed_can_interpolate_one_bounded_h_over_r_profile() -> (
    None
):
    context16 = _context(16)
    context32 = _context(32)
    gravitational_radius = context16.grid.gravitational_radius
    kwargs = {
        "inner_surface_density": 120.0,
        "outer_surface_density": 8.0e4,
        "inner_temperature": 4.2e6,
        "outer_temperature": 9.0e5,
        "profile_inner_plateau_radius": (
            6.0 * gravitational_radius
        ),
        "profile_outer_plateau_radius": (
            240.0 * gravitational_radius
        ),
        "profile_interpolate_log_h_over_r": True,
    }
    state16 = make_causal_five_field_seed(context16, **kwargs)
    state32 = make_causal_five_field_seed(context32, **kwargs)

    np.testing.assert_array_equal(
        state16.primitives[0, :3],
        state32.primitives[0, :3],
    )
    np.testing.assert_array_equal(
        state16.primitives[-1, :3],
        state32.primitives[-1, :3],
    )
    for context, state in (
        (context16, state16),
        (context32, state32),
    ):
        inner_radius = float(context.grid.edges[0])
        outer_radius = float(context.grid.edges[-1])
        inner_h_over_r = (
            context.vertical_frequency.eos(
                inner_radius
            ).from_surface_density_temperature(
                kwargs["inner_surface_density"],
                kwargs["inner_temperature"],
            ).proper_half_thickness
            / inner_radius
        )
        outer_h_over_r = (
            context.vertical_frequency.eos(
                outer_radius
            ).from_surface_density_temperature(
                kwargs["outer_surface_density"],
                kwargs["outer_temperature"],
            ).proper_half_thickness
            / outer_radius
        )
        coordinate = np.clip(
            (
                np.log(
                    context.grid.centers
                    / kwargs["profile_inner_plateau_radius"]
                )
                / np.log(
                    kwargs["profile_outer_plateau_radius"]
                    / kwargs["profile_inner_plateau_radius"]
                )
            ),
            0.0,
            1.0,
        )
        fraction = coordinate**3 * (
            10.0 - 15.0 * coordinate + 6.0 * coordinate**2
        )
        expected = np.exp(
            (1.0 - fraction) * np.log(inner_h_over_r)
            + fraction * np.log(outer_h_over_r)
        )
        actual = np.asarray(
            [
                context.vertical_frequency.eos(
                    float(radius)
                ).from_surface_density_temperature(
                    float(np.exp(primitive[0])),
                    float(np.exp(primitive[3])),
                ).proper_half_thickness
                / float(radius)
                for radius, primitive in zip(
                    context.grid.centers,
                    state.primitives,
                    strict=True,
                )
            ]
        )
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=2.0e-14,
            atol=0.0,
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


def test_integrated_source_components_reconstruct_production_source() -> None:
    context = _context(4, cooling=True)
    mass = np.asarray([0.0, 2.0e20, 3.0e20, 0.0])
    stream = KerrSchildCellSourceRates(
        rest_mass=mass,
        radial_momentum_over_c=0.25 * mass,
        angular_momentum_over_c=1.5e9 * mass,
        killing_energy_over_c2=1.01 * mass,
    )
    context = replace(context, stream_sources=stream)
    state = make_causal_five_field_seed(context)
    evaluation = evaluate_causal_five_field_dae(
        pack_causal_five_field_state(state),
        context,
    )
    components = evaluation.integrated_source_components_per_ct

    assert set(components) == {
        "perfect_fluid_geometry",
        "stress_geometry",
        "radiative_cooling",
        "vertical_work",
        "stress_relaxation",
        "stream",
    }
    reconstructed = np.sum(
        np.asarray(list(components.values()), dtype=float),
        axis=0,
    )
    scale = np.maximum(
        np.abs(evaluation.integrated_sources_per_ct),
        1.0,
    )
    assert np.max(
        np.abs(
            reconstructed - evaluation.integrated_sources_per_ct
        )
        / scale
    ) < 1.0e-14
    assert components["stream"][:, :4] == pytest.approx(
        stream.weighted_killing_source_per_ct
    )
    assert np.count_nonzero(components["stream"][:, 4]) == 0
    assert np.count_nonzero(
        components["stress_relaxation"][:, :4]
    ) == 0


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


def test_exact_stream_moments_enter_increment_primary_rows_exactly() -> None:
    context = _context(16, cooling=True)
    mass = FiducialParams().M2_g
    gravitational_radius = G * mass / C**2
    radius = 240.0 * gravitational_radius
    geometry = kerr_schild_column_geometry(
        radius,
        gravitational_radius,
    )
    thermodynamics = context.vertical_frequency.eos(
        radius
    ).from_surface_density_temperature(1.0e5, 1.0e6)
    primitive = ValenciaPerfectFluidPrimitive(
        surface_density=1.0e5,
        radial_velocity_over_c=2.0 * gravitational_radius / radius,
        azimuthal_velocity_over_c=float(
            np.sqrt(gravitational_radius / radius)
            / geometry.base.lapse
        ),
        specific_internal_energy=(
            thermodynamics.specific_internal_energy
        ),
        integrated_pressure=thermodynamics.integrated_pressure,
    )
    injection = kerr_schild_stream_injection(
        geometry,
        primitive,
        rest_mass_rate=5.0 * eddington_mdot(mass),
    )
    stream = exact_kerr_schild_compact_stream_sources(
        context.grid,
        injection,
        center=radius,
        log_width=0.08,
        shape="compact_c2",
    )
    sourced_context = replace(context, stream_sources=stream).validated()
    state = make_causal_five_field_seed(context)
    old_vector = pack_causal_five_field_state(state)
    zero_increment = np.zeros_like(old_vector)
    baseline = evaluate_causal_five_field_increment_backward_euler(
        zero_increment,
        context,
        old_vector=old_vector,
        timestep_seconds=1.0,
    )
    sourced = evaluate_causal_five_field_increment_backward_euler(
        zero_increment,
        sourced_context,
        old_vector=old_vector,
        timestep_seconds=1.0,
    )
    expected = stream.weighted_killing_source_per_ct

    assert np.count_nonzero(stream.rest_mass) > 0
    assert np.sum(stream.matrix, axis=0) == pytest.approx(
        np.asarray(
            [
                injection.rest_mass_rate,
                injection.rest_mass_rate
                * injection.moments.radial_momentum_over_c,
                injection.rest_mass_rate
                * injection.moments.angular_momentum_over_c,
                injection.rest_mass_rate
                * injection.moments.killing_energy_over_c2,
            ]
        ),
        rel=2.0e-15,
    )
    assert (
        sourced.conservation_rows[:, :4]
        - baseline.conservation_rows[:, :4]
        == pytest.approx(-expected)
    )
    assert sourced.conservation_rows[:, 4] == pytest.approx(
        baseline.conservation_rows[:, 4]
    )
    n_differential = 5 * context.grid.centers.size
    assert sourced.residual[n_differential:] == pytest.approx(
        baseline.residual[n_differential:]
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


@pytest.mark.parametrize(
    "spatial_options",
    (
        {"spatial_reconstruction": "piecewise_constant"},
        {"spatial_reconstruction": "plm_smooth"},
        {
            "spatial_reconstruction": "plm_smooth",
            "boundary_trace_reconstruction": "plm_one_sided",
            "cell_rate_scheme": "quadratic_log_radius",
            "cell_source_quadrature": "gauss_legendre_4",
            "cell_storage_quadrature": "gauss_legendre_4",
        },
        {
            "spatial_reconstruction": "quadratic_admissible",
            "boundary_trace_reconstruction": "plm_one_sided",
            "cell_source_quadrature": "gauss_legendre_4_local_rates",
            "cell_storage_quadrature": "gauss_legendre_4",
        },
    ),
)
def test_colored_increment_jacobian_matches_every_dense_column(
    spatial_options: dict[str, str],
) -> None:
    context = replace(
        _context(4, cooling=True),
        **spatial_options,
    ).validated()
    state = make_causal_five_field_seed(context)
    old_vector = pack_causal_five_field_state(state)
    stationary = evaluate_causal_five_field_dae(
        old_vector,
        context,
    )
    scaling = causal_five_field_dae_scaling(state, stationary)
    values = np.zeros_like(old_vector)
    step = 2.0e-6

    def residual(scaled_increment):
        return (
            evaluate_causal_five_field_increment_backward_euler(
                scaling.column_scales * scaled_increment,
                context,
                old_vector=old_vector,
                timestep_seconds=2.0e-8,
            ).residual
            / scaling.row_scales
        )

    dense = np.empty((values.size, values.size), dtype=float)
    for column in range(values.size):
        plus = np.array(values, copy=True)
        minus = np.array(values, copy=True)
        plus[column] += step
        minus[column] -= step
        dense[:, column] = (
            residual(plus) - residual(minus)
        ) / (2.0 * step)
    pattern = causal_five_field_dae_jacobian_sparsity(
        4,
        spatial_reconstruction=context.spatial_reconstruction,
        boundary_trace_reconstruction=(
            context.boundary_trace_reconstruction
        ),
        cell_rate_scheme=context.cell_rate_scheme,
        cell_source_quadrature=context.cell_source_quadrature,
        cell_storage_quadrature=context.cell_storage_quadrature,
    )
    groups = causal_five_field_dae_jacobian_color_groups(pattern)
    colored = causal_five_field_colored_central_jacobian(
        residual,
        values,
        pattern,
        finite_difference_step=step,
    ).toarray()
    allowed = pattern.toarray().astype(bool)
    row_scale = np.maximum(
        np.max(np.abs(dense), axis=1),
        1.0e-14,
    )
    omitted_relative = np.abs(
        np.where(allowed, 0.0, dense)
    ) / row_scale[:, None]
    colored_relative = np.abs(colored - dense) / row_scale[:, None]

    assert pattern.shape == dense.shape
    assert pattern.nnz < dense.size // 3
    assert len(groups) < values.size // 2
    for group in groups:
        assert np.max(
            np.asarray(pattern[:, group].sum(axis=1))
        ) <= 1
    assert np.max(omitted_relative) < 1.0e-11
    assert np.max(colored_relative) < 1.0e-11

    right_hand_side = -residual(values)
    sparse_solution, audit = (
        causal_five_field_equilibrated_sparse_solve(
            causal_five_field_colored_central_jacobian(
                residual,
                values,
                pattern,
                finite_difference_step=step,
            ),
            right_hand_side,
        )
    )
    dense_solution = np.linalg.solve(dense, right_hand_side)
    correction_scale = max(
        np.max(np.abs(dense_solution)),
        1.0e-14,
    )

    assert audit.dimensions == dense.shape
    assert audit.nonzeros <= pattern.nnz
    assert audit.relative_linear_residual < 1.0e-8
    assert (
        np.max(np.abs(sparse_solution - dense_solution))
        / correction_scale
        < 1.0e-7
    )


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
