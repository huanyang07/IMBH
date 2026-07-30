from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d import (
    causal_five_field_continuum_local_symbol,
    causal_five_field_local_symbol_stencil,
    causal_five_field_symbol_error,
    causal_packet_spectrum,
)


def _synthetic_tangent():
    n_cells = 9
    fields = 5
    dimensions = n_cells * fields
    descriptor = np.zeros((dimensions, dimensions))
    conservative = np.zeros_like(descriptor)
    shear = np.zeros_like(descriptor)
    height = np.zeros_like(descriptor)
    lower = np.zeros_like(descriptor)
    identity = np.eye(fields)
    for cell in range(n_cells):
        row = slice(fields * cell, fields * (cell + 1))
        descriptor[row, row] = 2.0 * identity
        lower[row, row] = 0.3 * identity
        for offset, coefficient in ((-1, -1.0), (0, 0.5), (1, 0.5)):
            column = cell + offset
            if 0 <= column < n_cells:
                target = slice(
                    fields * column,
                    fields * (column + 1),
                )
                conservative[row, target] = coefficient * identity
    stationary = conservative + shear + height + lower
    spatial = SimpleNamespace(
        block_scaled_jacobians={
            "candidate_conservative_transport": conservative,
            "candidate_shear_principal": shear,
            "candidate_height_principal": height,
            "candidate_local_stress_relaxation": lower,
        },
        characteristic_face_radii=np.exp(np.linspace(0.0, 0.9, 10)),
    )
    return SimpleNamespace(
        base_primitives=np.ones((n_cells, fields)),
        primitive_column_scales=np.ones(dimensions),
        conservation_row_scales=np.ones(dimensions),
        descriptor_scaled_matrix=descriptor,
        evolving_scaled_jacobian=stationary,
        spatial_tangent=spatial,
    )


def test_local_symbol_extracts_the_exact_interior_block_row() -> None:
    tangent = _synthetic_tangent()
    stencil = causal_five_field_local_symbol_stencil(
        tangent,
        4,
        np.ones(5),
    )
    assert not stencil.touches_boundary
    np.testing.assert_array_equal(stencil.offsets, (-1, 0, 1))
    descriptor, evolving, principal = stencil.matrices(0.4)
    expected_principal = (
        -np.exp(-0.4j) + 0.5 + 0.5 * np.exp(0.4j)
    ) * np.eye(5)
    np.testing.assert_allclose(descriptor, 2.0 * np.eye(5))
    np.testing.assert_allclose(principal, expected_principal)
    np.testing.assert_allclose(
        evolving,
        expected_principal + 0.3 * np.eye(5),
    )
    assert stencil.maximum_row_symbol_parity_defect <= 1.0e-14
    assert stencil.maximum_evolving_omitted_fraction == 0.0


def _constant_background():
    count = 21
    logs = np.linspace(np.log(1.5), np.log(4.5), count)
    radii = np.exp(logs)
    fields = 5
    temporal = np.diag(np.linspace(1.0, 1.4, fields))
    flux = np.diag(np.linspace(-0.4, 0.6, fields))
    shear = 0.1 * np.eye(fields)
    height = 0.05 * np.eye(fields)
    zeros_matrix = np.zeros((count, fields, fields))
    zeros_derivative = np.zeros((count, fields, fields, fields))
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
        lower_source_jacobians={"zero": zeros_matrix},
    )


def test_continuum_symbol_matches_constant_coefficient_pencil() -> None:
    background = _constant_background()
    radius = 2.4
    theta = 0.3
    spacing = 0.04
    symbol = causal_five_field_continuum_local_symbol(
        background,
        radius,
        theta,
        spacing,
        np.ones(5),
    )
    temporal = background.temporal_storage_matrices[0]
    spatial = (
        background.physical_flux_jacobians[0]
        - background.shear_principal_matrices[0]
        - background.height_principal_matrices[0]
    )
    expected = (
        -1.0j
        * C
        * theta
        / spacing
        / radius
        * np.linalg.solve(temporal, spatial)
    )
    np.testing.assert_allclose(
        symbol.principal_generator_per_s,
        expected,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        symbol.complete_generator_per_s,
        expected,
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_identical_symbols_have_zero_finite_time_error() -> None:
    generator = np.diag(
        np.asarray((-0.3 - 2.0j, -0.2 - 1.0j, -0.1, -0.2 + 1.0j, -0.3 + 2.0j))
    )
    error = causal_five_field_symbol_error(
        generator,
        generator,
        generator,
        generator,
        times=(0.01, 0.05),
    )
    assert error.maximum_complete_semigroup_relative_error <= 1.0e-14
    assert error.maximum_principal_semigroup_relative_error <= 1.0e-14
    assert error.maximum_principal_phase_error_radians == 0.0
    assert error.maximum_principal_log_amplitude_error == 0.0
    assert error.maximum_principal_family_leakage <= 1.0e-14


def test_packet_spectrum_recovers_gaussian_energy_scale() -> None:
    spacing = 0.025
    coordinates = spacing * (np.arange(256) - 127.5)
    values = np.zeros((coordinates.size, 5))
    values[:, 0] = np.exp(-0.5 * (coordinates / 0.2) ** 2)
    spectrum = causal_packet_spectrum(values, spacing, quantile=0.99)
    # For |F|^2 proportional to exp(-sigma^2 k^2), the two-sided 99 percent
    # quantile is erf^{-1}(0.99) / sigma ~= 1.8214 / sigma.
    expected = 1.821386 / 0.2
    assert abs(spectrum.quantile_angular_wavenumber / expected - 1.0) < 0.03
    assert spectrum.nyquist_alias_fraction < 1.0e-3
