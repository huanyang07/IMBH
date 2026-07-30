"""Local symbol and packet-resolution tools for the monolithic inner DAE.

The functions in this module are diagnostic only.  They freeze one block row
of the already-certified monolithic tangent and interpret its exact finite-
volume stencil as a local translation-invariant generalized symbol,

    M_h(theta) q_t + E_h(theta) q = 0.

The continuum comparison is assembled independently from the smooth local
physical maps used by the high-order continuum-truncation reference.  Both
symbols include the temporal descriptor, the derivative of that descriptor
acting on the self-consistent base rate, the complete principal operator,
and all lower-order couplings.

The one-sided excision row is deliberately excluded.  A periodic local symbol
cannot certify a boundary-overlapping packet; such packets require a separate
one-sided DAE-truncation contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.linalg import expm
from scipy.optimize import linear_sum_assignment

from imri_qpe.constants import C

from .causal_inner_continuum_truncation import (
    CausalFiveFieldContinuumBackground,
)
from .causal_inner_monolithic_tangent import (
    CausalFiveFieldMonolithicFrozenTangent,
)


_N_FIELDS = 5
_PRINCIPAL_BLOCKS = (
    "candidate_conservative_transport",
    "candidate_shear_principal",
    "candidate_height_principal",
)


def _relative_norm(difference: np.ndarray, *references: np.ndarray) -> float:
    scale = max(
        *(float(np.linalg.norm(reference)) for reference in references),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(difference) / scale)


def _positive_scales(field_scales: np.ndarray) -> np.ndarray:
    scales = np.asarray(field_scales, dtype=float).ravel()
    if (
        scales.shape != (_N_FIELDS,)
        or np.any(~np.isfinite(scales))
        or np.any(scales <= 0.0)
    ):
        raise ValueError("packet-resolution field scales are invalid")
    return scales


@dataclass(frozen=True)
class CausalFiveFieldLocalSymbolStencil:
    """One exact frozen block-row stencil in fixed physical coordinates."""

    radius: float
    cell_index: int
    offsets: np.ndarray
    descriptor_blocks: np.ndarray
    evolving_blocks: np.ndarray
    principal_blocks: np.ndarray
    field_scales: np.ndarray
    maximum_descriptor_omitted_fraction: float
    maximum_evolving_omitted_fraction: float
    maximum_principal_omitted_fraction: float
    maximum_row_symbol_parity_defect: float
    touches_boundary: bool

    def matrices(
        self,
        theta: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return descriptor, complete evolving, and principal symbols."""

        phase = np.exp(1.0j * float(theta) * self.offsets)
        descriptor = np.einsum(
            "o,oij->ij",
            phase,
            self.descriptor_blocks,
        )
        evolving = np.einsum(
            "o,oij->ij",
            phase,
            self.evolving_blocks,
        )
        principal = np.einsum(
            "o,oij->ij",
            phase,
            self.principal_blocks,
        )
        return descriptor, evolving, principal

    def generators(
        self,
        theta: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return complete and principal-only generators."""

        descriptor, evolving, principal = self.matrices(theta)
        complete = -np.linalg.solve(descriptor, evolving)
        principal_only = -np.linalg.solve(descriptor, principal)
        return complete, principal_only


def _physical_normalized_row_blocks(
    tangent: CausalFiveFieldMonolithicFrozenTangent,
    matrix: np.ndarray,
    cell_index: int,
    field_scales: np.ndarray,
) -> np.ndarray:
    """Convert one scaled residual row to fixed physical input coordinates."""

    base = np.asarray(tangent.base_primitives, dtype=float)
    n_cells = int(base.shape[0])
    dimensions = _N_FIELDS * n_cells
    values = np.asarray(matrix, dtype=float)
    if values.shape != (dimensions, dimensions):
        raise ValueError("local-symbol matrix has the wrong shape")
    columns = np.asarray(
        tangent.primitive_column_scales,
        dtype=float,
    ).reshape(n_cells, _N_FIELDS)
    rows = np.asarray(
        tangent.conservation_row_scales,
        dtype=float,
    ).reshape(n_cells, _N_FIELDS)
    row = int(cell_index)
    row_slice = slice(_N_FIELDS * row, _N_FIELDS * (row + 1))
    result = np.empty((n_cells, _N_FIELDS, _N_FIELDS), dtype=float)
    input_scale = np.diag(field_scales)
    for column in range(n_cells):
        column_slice = slice(
            _N_FIELDS * column,
            _N_FIELDS * (column + 1),
        )
        scaled_block = values[row_slice, column_slice]
        physical_block = (
            rows[row, :, None]
            * scaled_block
            / columns[column, None, :]
        )
        result[column] = physical_block @ input_scale
    return result


def _retained_offsets(
    matrices: tuple[np.ndarray, ...],
    cell_index: int,
    relative_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    block_norms = np.maximum.reduce(
        [
            np.linalg.norm(matrix, axis=(1, 2))
            for matrix in matrices
        ]
    )
    scale = max(float(np.max(block_norms)), np.finfo(float).tiny)
    retained = block_norms > float(relative_tolerance) * scale
    retained[int(cell_index)] = True
    indices = np.flatnonzero(retained)
    return indices - int(cell_index), indices


def _omitted_fraction(matrix: np.ndarray, retained: np.ndarray) -> float:
    omitted = np.array(matrix, copy=True)
    omitted[retained] = 0.0
    return _relative_norm(omitted, matrix)


def causal_five_field_local_symbol_stencil(
    tangent: CausalFiveFieldMonolithicFrozenTangent,
    cell_index: int,
    field_scales: np.ndarray,
    *,
    relative_block_tolerance: float = 1.0e-13,
    parity_thetas: tuple[float, ...] = (0.07, 0.31, 0.73),
) -> CausalFiveFieldLocalSymbolStencil:
    """Freeze one exact block row of the self-consistent monolithic tangent."""

    scales = _positive_scales(field_scales)
    base = np.asarray(tangent.base_primitives, dtype=float)
    n_cells = int(base.shape[0])
    row = int(cell_index)
    tolerance = float(relative_block_tolerance)
    if (
        row < 0
        or row >= n_cells
        or not np.isfinite(tolerance)
        or tolerance <= 0.0
    ):
        raise ValueError("local-symbol row selection is invalid")

    descriptor_all = _physical_normalized_row_blocks(
        tangent,
        tangent.descriptor_scaled_matrix,
        row,
        scales,
    )
    evolving_all = _physical_normalized_row_blocks(
        tangent,
        tangent.evolving_scaled_jacobian,
        row,
        scales,
    )
    principal_scaled = sum(
        (
            np.asarray(
                tangent.spatial_tangent.block_scaled_jacobians[name],
                dtype=float,
            )
            for name in _PRINCIPAL_BLOCKS
        ),
        start=np.zeros_like(tangent.evolving_scaled_jacobian),
    )
    principal_all = _physical_normalized_row_blocks(
        tangent,
        principal_scaled,
        row,
        scales,
    )
    offsets, retained = _retained_offsets(
        (descriptor_all, evolving_all, principal_all),
        row,
        tolerance,
    )
    descriptor = descriptor_all[retained]
    evolving = evolving_all[retained]
    principal = principal_all[retained]
    maximum_offset = int(np.max(np.abs(offsets)))
    touches_boundary = bool(
        row - maximum_offset < 0
        or row + maximum_offset >= n_cells
    )

    maximum_parity = 0.0
    for theta in parity_thetas:
        phase_all = np.exp(
            1.0j * float(theta) * (np.arange(n_cells) - row)
        )
        for full, blocks in (
            (descriptor_all, descriptor),
            (evolving_all, evolving),
            (principal_all, principal),
        ):
            direct = np.einsum("o,oij->ij", phase_all, full)
            selected = np.einsum(
                "o,oij->ij",
                np.exp(1.0j * float(theta) * offsets),
                blocks,
            )
            maximum_parity = max(
                maximum_parity,
                _relative_norm(selected - direct, selected, direct),
            )

    face_radii = np.asarray(
        tangent.spatial_tangent.characteristic_face_radii,
        dtype=float,
    )
    cell_radius = float(
        np.sqrt(face_radii[row] * face_radii[row + 1])
    )
    return CausalFiveFieldLocalSymbolStencil(
        radius=cell_radius,
        cell_index=row,
        offsets=np.asarray(offsets, dtype=int),
        descriptor_blocks=np.asarray(descriptor, dtype=float),
        evolving_blocks=np.asarray(evolving, dtype=float),
        principal_blocks=np.asarray(principal, dtype=float),
        field_scales=np.array(scales, copy=True),
        maximum_descriptor_omitted_fraction=_omitted_fraction(
            descriptor_all,
            retained,
        ),
        maximum_evolving_omitted_fraction=_omitted_fraction(
            evolving_all,
            retained,
        ),
        maximum_principal_omitted_fraction=_omitted_fraction(
            principal_all,
            retained,
        ),
        maximum_row_symbol_parity_defect=float(maximum_parity),
        touches_boundary=touches_boundary,
    )


@dataclass(frozen=True)
class CausalFiveFieldContinuumLocalSymbol:
    """Complete and principal continuum generators at one frozen radius."""

    radius: float
    theta: float
    log_spacing: float
    complete_generator_per_s: np.ndarray
    principal_generator_per_s: np.ndarray
    temporal_condition_number: float


def _spline_value(
    log_radii: np.ndarray,
    values: np.ndarray,
    log_radius: float,
) -> np.ndarray:
    return np.asarray(
        make_interp_spline(
            log_radii,
            np.asarray(values),
            k=5,
            axis=0,
        )(float(log_radius)),
        dtype=float,
    )


def _spline_derivative(
    log_radii: np.ndarray,
    values: np.ndarray,
    log_radius: float,
) -> np.ndarray:
    return np.asarray(
        make_interp_spline(
            log_radii,
            np.asarray(values),
            k=5,
            axis=0,
        ).derivative()(float(log_radius)),
        dtype=float,
    )


def causal_five_field_continuum_local_symbol(
    background: CausalFiveFieldContinuumBackground,
    radius: float,
    theta: float,
    log_spacing: float,
    field_scales: np.ndarray,
) -> CausalFiveFieldContinuumLocalSymbol:
    """Assemble the smooth local continuum symbol in fixed physical units."""

    scales = _positive_scales(field_scales)
    radial = float(radius)
    spacing = float(log_spacing)
    angular = float(theta)
    if (
        not np.isfinite(radial)
        or radial <= 0.0
        or not np.isfinite(spacing)
        or spacing <= 0.0
        or not np.isfinite(angular)
    ):
        raise ValueError("continuum local-symbol coordinates are invalid")
    log_radius = float(np.log(radial))
    lower = float(background.log_radii[0])
    upper = float(background.log_radii[-1])
    if log_radius <= lower or log_radius >= upper:
        raise ValueError("continuum local symbol must be strictly interior")

    logs = np.asarray(background.log_radii, dtype=float)
    measure = float(
        _spline_value(logs, background.face_measures, log_radius)
    )
    temporal = _spline_value(
        logs,
        background.temporal_storage_matrices,
        log_radius,
    )
    flux = _spline_value(
        logs,
        background.physical_flux_jacobians,
        log_radius,
    )
    shear = _spline_value(
        logs,
        background.shear_principal_matrices,
        log_radius,
    )
    height = _spline_value(
        logs,
        background.height_principal_matrices,
        log_radius,
    )
    base_gradient = _spline_value(
        logs,
        background.primitive_radius_derivative,
        log_radius,
    )
    base_rate = _spline_value(
        logs,
        background.base_rate_per_s,
        log_radius,
    )
    shear_derivative = _spline_value(
        logs,
        background.shear_principal_derivatives,
        log_radius,
    )
    height_derivative = _spline_value(
        logs,
        background.height_principal_derivatives,
        log_radius,
    )
    mapped_storage_derivative = _spline_value(
        logs,
        background.mapped_conserved_hessians,
        log_radius,
    )
    vertical_storage_derivative = _spline_value(
        logs,
        background.vertical_storage_derivatives,
        log_radius,
    )
    flux_measure_log_derivative = _spline_derivative(
        logs,
        (
            background.face_measures[:, None, None]
            * background.physical_flux_jacobians
        ),
        log_radius,
    )

    shear_state = np.einsum(
        "ijk,j->ik",
        shear_derivative,
        base_gradient,
    )
    height_state = np.einsum(
        "ijk,j->ik",
        height_derivative,
        base_gradient,
    )
    mapped_storage_rate = np.einsum(
        "ijk,j->ik",
        mapped_storage_derivative,
        base_rate,
    )
    vertical_storage_rate = np.einsum(
        "ijk,j->ik",
        vertical_storage_derivative,
        base_rate,
    )
    lower_jacobian = sum(
        (
            _spline_value(logs, values, log_radius)
            for values in background.lower_source_jacobians.values()
        ),
        start=np.zeros((_N_FIELDS, _N_FIELDS), dtype=float),
    )

    wavenumber = angular / spacing
    derivative_matrix = (
        measure / radial * (flux - shear - height)
    )
    principal_zero_order = (
        flux_measure_log_derivative / radial
        - measure * (shear_state + height_state)
    )
    zero_order = (
        principal_zero_order
        - measure * lower_jacobian
        + measure / C
        * (mapped_storage_rate + vertical_storage_rate)
    )
    complete_operator = (
        zero_order + 1.0j * wavenumber * derivative_matrix
    )
    principal_operator = (
        principal_zero_order
        + 1.0j * wavenumber * derivative_matrix
    )
    physical_complete = (
        -C / measure * np.linalg.solve(temporal, complete_operator)
    )
    physical_principal = (
        -C / measure * np.linalg.solve(temporal, principal_operator)
    )
    scale = np.diag(scales)
    inverse_scale = np.diag(1.0 / scales)
    return CausalFiveFieldContinuumLocalSymbol(
        radius=radial,
        theta=angular,
        log_spacing=spacing,
        complete_generator_per_s=(
            inverse_scale @ physical_complete @ scale
        ),
        principal_generator_per_s=(
            inverse_scale @ physical_principal @ scale
        ),
        temporal_condition_number=float(np.linalg.cond(temporal)),
    )


def _matched_eigensystem(
    numerical: np.ndarray,
    continuum: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    continuum_values, continuum_vectors = np.linalg.eig(continuum)
    numerical_values, numerical_vectors = np.linalg.eig(numerical)
    frequency_scale = max(
        float(np.max(np.abs(continuum_values))),
        float(np.max(np.abs(numerical_values))),
        1.0,
    )
    overlaps = np.abs(
        np.linalg.solve(continuum_vectors, numerical_vectors)
    )
    cost = (
        np.abs(
            continuum_values[:, None] - numerical_values[None, :]
        )
        / frequency_scale
        + 0.05 * (1.0 - overlaps / np.maximum(
            np.max(overlaps, axis=0, keepdims=True),
            np.finfo(float).tiny,
        ))
    )
    rows, columns = linear_sum_assignment(cost)
    order = columns[np.argsort(rows)]
    return (
        continuum_values,
        numerical_values[order],
        continuum_vectors,
    )


@dataclass(frozen=True)
class CausalFiveFieldSymbolError:
    """Finite-time and modal error of one local discrete symbol."""

    maximum_complete_semigroup_relative_error: float
    maximum_principal_semigroup_relative_error: float
    maximum_principal_phase_error_radians: float
    maximum_principal_log_amplitude_error: float
    maximum_principal_family_leakage: float
    continuum_principal_basis_condition_number: float
    numerical_principal_basis_condition_number: float


def causal_five_field_matched_principal_eigenvalues(
    numerical_principal: np.ndarray,
    continuum_principal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return continuum eigenvalues and numerical values in matched order."""

    continuum_values, numerical_values, _vectors = _matched_eigensystem(
        np.asarray(numerical_principal, dtype=complex),
        np.asarray(continuum_principal, dtype=complex),
    )
    return continuum_values, numerical_values


def causal_five_field_symbol_error(
    numerical_complete: np.ndarray,
    numerical_principal: np.ndarray,
    continuum_complete: np.ndarray,
    continuum_principal: np.ndarray,
    *,
    times: tuple[float, ...],
) -> CausalFiveFieldSymbolError:
    """Compare local symbols using finite-time propagators and projectors."""

    time_values = tuple(float(value) for value in times)
    if (
        not time_values
        or any(
            not np.isfinite(value) or value <= 0.0
            for value in time_values
        )
    ):
        raise ValueError("symbol-error times are invalid")
    nc = np.asarray(numerical_complete, dtype=complex)
    np_ = np.asarray(numerical_principal, dtype=complex)
    cc = np.asarray(continuum_complete, dtype=complex)
    cp = np.asarray(continuum_principal, dtype=complex)
    for matrix in (nc, np_, cc, cp):
        if matrix.shape != (_N_FIELDS, _N_FIELDS):
            raise ValueError("symbol-error matrix has the wrong shape")

    complete_error = 0.0
    principal_error = 0.0
    leakage = 0.0
    continuum_values, numerical_values, continuum_vectors = (
        _matched_eigensystem(np_, cp)
    )
    numerical_basis = np.linalg.eig(np_)[1]
    left = np.linalg.inv(continuum_vectors)
    identity = np.eye(_N_FIELDS, dtype=complex)
    for time in time_values:
        numerical_complete_step = expm(time * nc)
        continuum_complete_step = expm(time * cc)
        numerical_principal_step = expm(time * np_)
        continuum_principal_step = expm(time * cp)
        complete_error = max(
            complete_error,
            _relative_norm(
                numerical_complete_step - continuum_complete_step,
                numerical_complete_step,
                continuum_complete_step,
            ),
        )
        principal_error = max(
            principal_error,
            _relative_norm(
                numerical_principal_step - continuum_principal_step,
                numerical_principal_step,
                continuum_principal_step,
            ),
        )
        for family in range(_N_FIELDS):
            projector = np.outer(
                continuum_vectors[:, family],
                left[family, :],
            )
            propagated = numerical_principal_step @ projector
            leakage = max(
                leakage,
                _relative_norm(
                    (identity - projector) @ propagated,
                    propagated,
                ),
            )

    horizon = max(time_values)
    difference = numerical_values - continuum_values
    return CausalFiveFieldSymbolError(
        maximum_complete_semigroup_relative_error=float(complete_error),
        maximum_principal_semigroup_relative_error=float(principal_error),
        maximum_principal_phase_error_radians=float(
            horizon * np.max(np.abs(np.imag(difference)))
        ),
        maximum_principal_log_amplitude_error=float(
            horizon * np.max(np.abs(np.real(difference)))
        ),
        maximum_principal_family_leakage=float(leakage),
        continuum_principal_basis_condition_number=float(
            np.linalg.cond(continuum_vectors)
        ),
        numerical_principal_basis_condition_number=float(
            np.linalg.cond(numerical_basis)
        ),
    )


@dataclass(frozen=True)
class CausalPacketSpectrum:
    """Physically normalized spectral-energy summary of one packet."""

    angular_wavenumbers: np.ndarray
    spectral_energy: np.ndarray
    cumulative_energy_fraction: np.ndarray
    quantile: float
    quantile_angular_wavenumber: float
    nyquist_alias_fraction: float
    log_spacing: float


def causal_packet_spectrum(
    normalized_cell_averages: np.ndarray,
    log_spacing: float,
    *,
    quantile: float = 0.99,
    zero_padding_factor: int = 16,
    reference_oversampling: int = 8,
) -> CausalPacketSpectrum:
    """Return the spectrum of compact, negligible-endpoint cell averages.

    ``normalized_cell_averages`` must already be divided by fixed positive
    physical field scales.  The packet is zero padded before transformation;
    callers are responsible for supplying a sufficiently wide compact window
    so the endpoint values are negligible.
    """

    values = np.asarray(normalized_cell_averages, dtype=float)
    spacing = float(log_spacing)
    fraction = float(quantile)
    padding = int(zero_padding_factor)
    oversampling = int(reference_oversampling)
    if (
        values.ndim != 2
        or values.shape[1] != _N_FIELDS
        or values.shape[0] < 8
        or np.any(~np.isfinite(values))
        or not np.isfinite(spacing)
        or spacing <= 0.0
        or not 0.5 < fraction < 1.0
        or padding < 2
        or oversampling < 2
    ):
        raise ValueError("packet-spectrum inputs are invalid")

    transform_size = 1
    target = padding * values.shape[0]
    while transform_size < target:
        transform_size *= 2
    transform = np.fft.rfft(values, n=transform_size, axis=0)
    energy = np.sum(np.abs(transform) ** 2, axis=1)
    if energy.size > 2:
        energy[1:-1] *= 2.0
    total = max(float(np.sum(energy)), np.finfo(float).tiny)
    cumulative = np.cumsum(energy) / total
    wavenumbers = (
        2.0
        * np.pi
        * np.fft.rfftfreq(transform_size, d=spacing)
    )
    index = int(np.searchsorted(cumulative, fraction, side="left"))
    index = min(index, wavenumbers.size - 1)

    # A high-resolution piecewise-constant reconstruction supplies a
    # conservative estimate of energy above the coarse Nyquist frequency.
    refined = np.repeat(values, oversampling, axis=0)
    refined_spacing = spacing / oversampling
    refined_size = 1
    target_refined = padding * refined.shape[0]
    while refined_size < target_refined:
        refined_size *= 2
    refined_transform = np.fft.rfft(
        refined,
        n=refined_size,
        axis=0,
    )
    refined_energy = np.sum(np.abs(refined_transform) ** 2, axis=1)
    if refined_energy.size > 2:
        refined_energy[1:-1] *= 2.0
    refined_wavenumbers = (
        2.0
        * np.pi
        * np.fft.rfftfreq(refined_size, d=refined_spacing)
    )
    nyquist = np.pi / spacing
    alias = float(
        np.sum(refined_energy[refined_wavenumbers > nyquist])
        / max(float(np.sum(refined_energy)), np.finfo(float).tiny)
    )
    return CausalPacketSpectrum(
        angular_wavenumbers=np.asarray(wavenumbers, dtype=float),
        spectral_energy=np.asarray(energy, dtype=float),
        cumulative_energy_fraction=np.asarray(cumulative, dtype=float),
        quantile=fraction,
        quantile_angular_wavenumber=float(wavenumbers[index]),
        nyquist_alias_fraction=alias,
        log_spacing=spacing,
    )
