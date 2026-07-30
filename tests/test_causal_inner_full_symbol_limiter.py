from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scipy.linalg import expm

from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d.causal_inner_full_symbol_limiter import (
    CausalNormalizedLocalDAESymbol,
    causal_continuum_normalized_local_dae,
    causal_local_dae_component_stencil,
    causal_match_symbol_eigenvalues_to_tracked_branches,
    causal_symbol_shapley_attribution,
    causal_track_symbol_eigenbranches,
    causal_time_ordered_symbol_propagator,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_resolution import (
    causal_five_field_continuum_local_symbol,
    causal_five_field_local_symbol_stencil,
)


OPERATOR_NAMES = (
    "principal",
    "mapped_storage_rate",
    "height_storage_rate",
    "stress_relaxation",
    "lower_sources",
)


def _synthetic_tangent():
    n_cells = 9
    fields = 5
    dimensions = n_cells * fields
    mapped_descriptor = np.zeros((dimensions, dimensions))
    height_descriptor = np.zeros_like(mapped_descriptor)
    principal = np.zeros_like(mapped_descriptor)
    mapped_storage = np.zeros_like(mapped_descriptor)
    height_storage = np.zeros_like(mapped_descriptor)
    stress = np.zeros_like(mapped_descriptor)
    geometry = np.zeros_like(mapped_descriptor)
    cooling = np.zeros_like(mapped_descriptor)
    stream = np.zeros_like(mapped_descriptor)
    lower_height = np.zeros_like(mapped_descriptor)
    identity = np.eye(fields)
    for cell in range(n_cells):
        row = slice(fields * cell, fields * (cell + 1))
        mapped_descriptor[row, row] = 1.5 * identity
        height_descriptor[row, row] = 0.5 * identity
        mapped_storage[row, row] = 0.05 * identity
        height_storage[row, row] = 0.02 * identity
        stress[row, row] = 0.03 * identity
        geometry[row, row] = 0.08 * identity
        cooling[row, row] = 0.07 * identity
        lower_height[row, row] = 0.05 * identity
        for offset, coefficient in ((-1, -1.0), (0, 0.5), (1, 0.5)):
            column = cell + offset
            if 0 <= column < n_cells:
                target = slice(
                    fields * column,
                    fields * (column + 1),
                )
                principal[row, target] = coefficient * identity
    blocks = {
        "candidate_conservative_transport": principal,
        "candidate_shear_principal": np.zeros_like(principal),
        "candidate_height_principal": np.zeros_like(principal),
        "candidate_local_stress_relaxation": stress,
        "candidate_geometry": geometry,
        "candidate_cooling": cooling,
        "candidate_stream": stream,
        "candidate_lower_height_work": lower_height,
    }
    descriptor = mapped_descriptor + height_descriptor
    storage = mapped_storage + height_storage
    stationary = sum(blocks.values(), start=np.zeros_like(principal))
    evolving = stationary + storage
    spatial = SimpleNamespace(
        block_scaled_jacobians=blocks,
        characteristic_face_radii=np.exp(np.linspace(0.0, 0.9, 10)),
    )
    return SimpleNamespace(
        base_primitives=np.ones((n_cells, fields)),
        primitive_column_scales=np.ones(dimensions),
        conservation_row_scales=np.ones(dimensions),
        mapped_descriptor_scaled_matrix=mapped_descriptor,
        responsive_height_descriptor_scaled_matrix=height_descriptor,
        descriptor_scaled_matrix=descriptor,
        mapped_storage_rate_derivative_scaled_matrix=mapped_storage,
        responsive_height_storage_rate_derivative_scaled_matrix=(
            height_storage
        ),
        evolving_scaled_jacobian=evolving,
        spatial_tangent=spatial,
    )


def test_component_stencil_closes_against_complete_local_symbol() -> None:
    tangent = _synthetic_tangent()
    components = causal_local_dae_component_stencil(
        tangent,
        4,
        np.ones(5),
    )
    complete = causal_five_field_local_symbol_stencil(
        tangent,
        4,
        np.ones(5),
    )
    assert components.maximum_component_closure_defect <= 1.0e-15
    assert components.maximum_omitted_fraction == 0.0
    assert not components.touches_boundary
    for theta in (0.0, 0.3, 0.7):
        np.testing.assert_allclose(
            components.symbol(theta).generator_per_s,
            complete.generators(theta)[0],
            rtol=1.0e-14,
            atol=1.0e-14,
        )


def _constant_background():
    count = 21
    logs = np.linspace(np.log(1.5), np.log(4.5), count)
    radii = np.exp(logs)
    fields = 5
    mapped = np.diag(np.linspace(0.8, 1.0, fields))
    height_storage = np.diag(np.linspace(0.2, 0.4, fields))
    temporal = mapped + height_storage
    flux = np.diag(np.linspace(-0.4, 0.6, fields))
    shear = 0.1 * np.eye(fields)
    height = 0.05 * np.eye(fields)
    zeros_matrix = np.zeros((count, fields, fields))
    zeros_derivative = np.zeros((count, fields, fields, fields))
    lower = {
        "stress_relaxation": np.repeat(
            (0.02 * np.eye(fields))[None, :, :],
            count,
            axis=0,
        ),
        "perfect_fluid_geometry": zeros_matrix,
        "stress_geometry": zeros_matrix,
        "radiative_cooling": zeros_matrix,
        "vertical_work": zeros_matrix,
    }
    return SimpleNamespace(
        log_radii=logs,
        radii=radii,
        face_measures=np.ones(count),
        temporal_storage_matrices=np.repeat(
            temporal[None, :, :],
            count,
            axis=0,
        ),
        physical_flux_jacobians=np.repeat(
            flux[None, :, :],
            count,
            axis=0,
        ),
        shear_principal_matrices=np.repeat(
            shear[None, :, :],
            count,
            axis=0,
        ),
        height_principal_matrices=np.repeat(
            height[None, :, :],
            count,
            axis=0,
        ),
        primitive_radius_derivative=np.zeros((count, fields)),
        base_rate_per_s=np.zeros((count, fields)),
        shear_principal_derivatives=zeros_derivative,
        height_principal_derivatives=zeros_derivative,
        mapped_conserved_hessians=zeros_derivative,
        vertical_storage_derivatives=zeros_derivative,
        lower_source_jacobians=lower,
    )


def test_continuum_component_sum_matches_complete_c6a_symbol() -> None:
    background = _constant_background()
    radius = 2.4
    theta = 0.3
    spacing = 0.04
    scales = np.linspace(0.7, 1.3, 5)
    components = causal_continuum_normalized_local_dae(
        background,
        radius,
        theta,
        spacing,
        scales,
    )
    complete = causal_five_field_continuum_local_symbol(
        background,
        radius,
        theta,
        spacing,
        scales,
    )
    np.testing.assert_allclose(
        components.generator_per_s,
        complete.complete_generator_per_s,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        -components.operator("principal"),
        complete.principal_generator_per_s,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    expected_relaxation = (
        C
        * np.linalg.solve(
            background.temporal_storage_matrices[0],
            0.02 * np.eye(5),
        )
    )
    scale = np.diag(scales)
    inverse_scale = np.diag(1.0 / scales)
    np.testing.assert_allclose(
        components.operator("stress_relaxation"),
        -inverse_scale @ expected_relaxation @ scale,
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def _zero_symbol() -> CausalNormalizedLocalDAESymbol:
    return CausalNormalizedLocalDAESymbol(
        descriptor=np.eye(5, dtype=complex),
        operators={
            name: np.zeros((5, 5), dtype=complex)
            for name in OPERATOR_NAMES
        },
    )


def test_shapley_attribution_closes_and_selects_one_changed_group() -> None:
    continuum = _zero_symbol()
    numerical_operators = {
        name: np.array(values, copy=True)
        for name, values in continuum.operators.items()
    }
    numerical_operators["stress_relaxation"] = 0.2 * np.eye(5)
    numerical = CausalNormalizedLocalDAESymbol(
        descriptor=np.eye(5, dtype=complex),
        operators=numerical_operators,
    )
    audit = causal_symbol_shapley_attribution(
        numerical,
        continuum,
        0.4,
    )
    assert audit.maximum_closure_defect <= 1.0e-14
    stress = audit.player_names.index("stress_relaxation")
    np.testing.assert_allclose(
        audit.contributions[stress],
        audit.total_difference,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    for index, name in enumerate(audit.player_names):
        if name != "stress_relaxation":
            assert np.linalg.norm(audit.contributions[index]) <= 1.0e-14


def test_time_ordered_propagator_recovers_constant_generator() -> None:
    generator = np.diag(
        np.asarray((-0.3 - 2.0j, -0.2 - 1.0j, -0.1, -0.2 + 1.0j, -0.3 + 2.0j))
    )
    steps = np.asarray((0.01, 0.02, 0.03))
    result = causal_time_ordered_symbol_propagator(
        np.repeat(generator[None, :, :], steps.size, axis=0),
        steps,
    )
    np.testing.assert_allclose(
        result,
        expm(float(np.sum(steps)) * generator),
        rtol=2.0e-14,
        atol=2.0e-14,
    )


def test_overlap_tracking_preserves_branches_through_speed_crossing() -> None:
    coordinates = np.linspace(-1.0, 1.0, 11)
    generators = np.asarray(
        [
            np.diag(
                1.0j
                * np.asarray(
                    (
                        -3.0,
                        coordinate,
                        -coordinate,
                        2.0,
                        4.0,
                    )
                )
            )
            for coordinate in coordinates
        ]
    )
    tracked = causal_track_symbol_eigenbranches(generators)
    assert tracked.minimum_consecutive_overlap >= 1.0 - 1.0e-14
    np.testing.assert_allclose(
        np.imag(tracked.eigenvalues[:, 1]),
        coordinates,
    )
    np.testing.assert_allclose(
        np.imag(tracked.eigenvalues[:, 2]),
        -coordinates,
    )
    shifted = generators + 0.01j * np.eye(5)[None, :, :]
    matched = causal_match_symbol_eigenvalues_to_tracked_branches(
        shifted,
        tracked,
    )
    np.testing.assert_allclose(
        np.imag(matched - tracked.eigenvalues),
        0.01,
        atol=1.0e-14,
    )
