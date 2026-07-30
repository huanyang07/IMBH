"""Exact attribution tools for the complete causal-inner DAE symbol.

This module is diagnostic only.  It preserves the failed WP10c9d6c6a
packet-resolution contract and answers a narrower question: which part of
the complete frozen DAE makes its finite-time symbol error larger than the
principal-symbol error?

Every local DAE is first left-normalized by its zero-wavenumber temporal
descriptor.  In those coordinates,

    D(theta) q_t + sum_g A_g(theta) q = 0,

where the operator groups are principal transport, mapped storage-rate
work, responsive-height storage-rate work, physical stress relaxation, and
the remaining lower sources.  The continuum descriptor is the identity.
The numerical descriptor is retained as a separate player.

The propagator difference is allocated with an exact Shapley average over
all discrete/continuum hybrid DAEs.  This treats noncommuting cross terms
symmetrically and makes the matrix-valued contributions sum to the complete
propagator difference to roundoff.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math

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
from .causal_inner_packet_resolution import (
    _physical_normalized_row_blocks,
    _positive_scales,
)


_N_FIELDS = 5
_OPERATOR_NAMES = (
    "principal",
    "mapped_storage_rate",
    "height_storage_rate",
    "stress_relaxation",
    "lower_sources",
)
_PLAYER_NAMES = ("descriptor", *_OPERATOR_NAMES)
_PRINCIPAL_BLOCKS = (
    "candidate_conservative_transport",
    "candidate_shear_principal",
    "candidate_height_principal",
)
_LOWER_BLOCKS = (
    "candidate_geometry",
    "candidate_cooling",
    "candidate_stream",
    "candidate_lower_height_work",
)


def _relative_norm(difference: np.ndarray, *references: np.ndarray) -> float:
    scale = max(
        *(float(np.linalg.norm(reference)) for reference in references),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(difference) / scale)


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


@dataclass(frozen=True)
class CausalNormalizedLocalDAESymbol:
    """One left-normalized local DAE symbol in fixed field coordinates."""

    descriptor: np.ndarray
    operators: dict[str, np.ndarray]

    @property
    def generator_per_s(self) -> np.ndarray:
        operator = sum(
            self.operators.values(),
            start=np.zeros((_N_FIELDS, _N_FIELDS), dtype=complex),
        )
        return -np.linalg.solve(self.descriptor, operator)

    def operator(self, name: str) -> np.ndarray:
        if name not in self.operators:
            raise KeyError(name)
        return np.asarray(self.operators[name], dtype=complex)


@dataclass(frozen=True)
class CausalLocalDAEComponentStencil:
    """Exact numerical block-row components for one local frozen symbol."""

    radius: float
    cell_index: int
    offsets: np.ndarray
    mapped_descriptor_blocks: np.ndarray
    height_descriptor_blocks: np.ndarray
    operator_blocks: dict[str, np.ndarray]
    field_scales: np.ndarray
    maximum_component_closure_defect: float
    maximum_omitted_fraction: float
    touches_boundary: bool

    def symbol(self, theta: float) -> CausalNormalizedLocalDAESymbol:
        """Evaluate and left-normalize the numerical DAE symbol."""

        angular = float(theta)
        phase = np.exp(1.0j * angular * self.offsets)

        def evaluate(blocks: np.ndarray) -> np.ndarray:
            return np.einsum("o,oij->ij", phase, blocks)

        mapped = evaluate(self.mapped_descriptor_blocks)
        height = evaluate(self.height_descriptor_blocks)
        mapped_zero = np.sum(self.mapped_descriptor_blocks, axis=0)
        height_zero = np.sum(self.height_descriptor_blocks, axis=0)
        normalizer = mapped_zero + height_zero
        descriptor = np.linalg.solve(normalizer, mapped + height)
        operators = {
            name: np.linalg.solve(normalizer, evaluate(blocks))
            for name, blocks in self.operator_blocks.items()
        }
        return CausalNormalizedLocalDAESymbol(
            descriptor=np.asarray(descriptor, dtype=complex),
            operators={
                name: np.asarray(values, dtype=complex)
                for name, values in operators.items()
            },
        )


def _retained_indices(
    matrices: tuple[np.ndarray, ...],
    cell_index: int,
    relative_tolerance: float,
) -> np.ndarray:
    norms = np.maximum.reduce(
        [
            np.linalg.norm(matrix, axis=(1, 2))
            for matrix in matrices
        ]
    )
    scale = max(float(np.max(norms)), np.finfo(float).tiny)
    retained = norms > float(relative_tolerance) * scale
    retained[int(cell_index)] = True
    return np.flatnonzero(retained)


def _omitted_fraction(matrix: np.ndarray, retained: np.ndarray) -> float:
    omitted = np.array(matrix, copy=True)
    omitted[retained] = 0.0
    return _relative_norm(omitted, matrix)


def causal_local_dae_component_stencil(
    tangent: CausalFiveFieldMonolithicFrozenTangent,
    cell_index: int,
    field_scales: np.ndarray,
    *,
    relative_block_tolerance: float = 1.0e-13,
) -> CausalLocalDAEComponentStencil:
    """Extract the exact component stencils of one monolithic block row."""

    scales = _positive_scales(field_scales)
    row = int(cell_index)
    n_cells = int(np.asarray(tangent.base_primitives).shape[0])
    tolerance = float(relative_block_tolerance)
    if (
        row < 0
        or row >= n_cells
        or not np.isfinite(tolerance)
        or tolerance <= 0.0
    ):
        raise ValueError("local DAE component row is invalid")

    def blocks(matrix: np.ndarray) -> np.ndarray:
        return _physical_normalized_row_blocks(
            tangent,
            matrix,
            row,
            scales,
        )

    mapped_descriptor = blocks(
        tangent.mapped_descriptor_scaled_matrix
    )
    height_descriptor = blocks(
        tangent.responsive_height_descriptor_scaled_matrix
    )
    spatial = tangent.spatial_tangent.block_scaled_jacobians
    principal_matrix = sum(
        (
            np.asarray(spatial[name], dtype=float)
            for name in _PRINCIPAL_BLOCKS
        ),
        start=np.zeros_like(tangent.evolving_scaled_jacobian),
    )
    lower_matrix = sum(
        (
            np.asarray(spatial[name], dtype=float)
            for name in _LOWER_BLOCKS
        ),
        start=np.zeros_like(tangent.evolving_scaled_jacobian),
    )
    component_matrices = {
        "principal": blocks(principal_matrix),
        "mapped_storage_rate": blocks(
            tangent.mapped_storage_rate_derivative_scaled_matrix
        ),
        "height_storage_rate": blocks(
            tangent.responsive_height_storage_rate_derivative_scaled_matrix
        ),
        "stress_relaxation": blocks(
            np.asarray(
                spatial["candidate_local_stress_relaxation"],
                dtype=float,
            )
        ),
        "lower_sources": blocks(lower_matrix),
    }
    all_matrices = (
        mapped_descriptor,
        height_descriptor,
        *component_matrices.values(),
    )
    retained = _retained_indices(all_matrices, row, tolerance)
    offsets = retained - row
    descriptor_total = blocks(tangent.descriptor_scaled_matrix)
    evolving_total = blocks(tangent.evolving_scaled_jacobian)
    component_closure = max(
        _relative_norm(
            mapped_descriptor + height_descriptor - descriptor_total,
            descriptor_total,
        ),
        _relative_norm(
            sum(
                component_matrices.values(),
                start=np.zeros_like(evolving_total),
            )
            - evolving_total,
            evolving_total,
        ),
    )
    maximum_omitted = max(
        _omitted_fraction(matrix, retained)
        for matrix in (
            descriptor_total,
            evolving_total,
            *all_matrices,
        )
    )
    touches_boundary = bool(
        row + int(np.min(offsets)) < 0
        or row + int(np.max(offsets)) >= n_cells
    )
    return CausalLocalDAEComponentStencil(
        radius=float(
            np.sqrt(
                tangent.spatial_tangent.characteristic_face_radii[row]
                * tangent.spatial_tangent.characteristic_face_radii[row + 1]
            )
        ),
        cell_index=row,
        offsets=np.asarray(offsets, dtype=int),
        mapped_descriptor_blocks=np.asarray(
            mapped_descriptor[retained],
            dtype=float,
        ),
        height_descriptor_blocks=np.asarray(
            height_descriptor[retained],
            dtype=float,
        ),
        operator_blocks={
            name: np.asarray(values[retained], dtype=float)
            for name, values in component_matrices.items()
        },
        field_scales=np.array(scales, copy=True),
        maximum_component_closure_defect=float(component_closure),
        maximum_omitted_fraction=float(maximum_omitted),
        touches_boundary=touches_boundary,
    )


def causal_continuum_normalized_local_dae(
    background: CausalFiveFieldContinuumBackground,
    radius: float,
    theta: float,
    log_spacing: float,
    field_scales: np.ndarray,
) -> CausalNormalizedLocalDAESymbol:
    """Assemble the continuum DAE groups used by the c6a local symbol."""

    scales = _positive_scales(field_scales)
    radial = float(radius)
    angular = float(theta)
    spacing = float(log_spacing)
    if (
        not np.isfinite(radial)
        or radial <= 0.0
        or not np.isfinite(angular)
        or not np.isfinite(spacing)
        or spacing <= 0.0
    ):
        raise ValueError("continuum DAE symbol coordinates are invalid")
    log_radius = float(np.log(radial))
    logs = np.asarray(background.log_radii, dtype=float)
    if log_radius <= float(logs[0]) or log_radius >= float(logs[-1]):
        raise ValueError("continuum DAE symbol must be strictly interior")

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
    height_storage_derivative = _spline_value(
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
    height_storage_rate = np.einsum(
        "ijk,j->ik",
        height_storage_derivative,
        base_rate,
    )
    lower = {
        name: _spline_value(logs, values, log_radius)
        for name, values in background.lower_source_jacobians.items()
    }
    derivative = (
        measure / radial * (flux - shear - height)
    )
    principal_zero = (
        flux_measure_log_derivative / radial
        - measure * (shear_state + height_state)
    )
    wavenumber = angular / spacing
    dae_operators = {
        "principal": (
            principal_zero + 1.0j * wavenumber * derivative
        ),
        "mapped_storage_rate": (
            measure / C * mapped_storage_rate
        ),
        "height_storage_rate": (
            measure / C * height_storage_rate
        ),
        "stress_relaxation": (
            -measure * lower["stress_relaxation"]
        ),
        "lower_sources": (
            -measure
            * sum(
                (
                    values
                    for name, values in lower.items()
                    if name != "stress_relaxation"
                ),
                start=np.zeros((_N_FIELDS, _N_FIELDS), dtype=float),
            )
        ),
    }
    state_scale = np.diag(scales)
    inverse_state_scale = np.diag(1.0 / scales)
    operators = {
        name: (
            inverse_state_scale
            @ (
                C
                / measure
                * np.linalg.solve(temporal, values)
            )
            @ state_scale
        )
        for name, values in dae_operators.items()
    }
    return CausalNormalizedLocalDAESymbol(
        descriptor=np.eye(_N_FIELDS, dtype=complex),
        operators={
            name: np.asarray(values, dtype=complex)
            for name, values in operators.items()
        },
    )


@dataclass(frozen=True)
class CausalSymbolShapleyAttribution:
    """Exact matrix-valued allocation of one propagator difference."""

    player_names: tuple[str, ...]
    continuum_propagator: np.ndarray
    numerical_propagator: np.ndarray
    total_difference: np.ndarray
    contributions: np.ndarray
    contribution_norms_relative_to_total: np.ndarray
    contribution_norms_relative_to_propagator: np.ndarray
    contribution_cosines_with_total: np.ndarray
    gram_matrix_relative_to_total: np.ndarray
    only_player_errors_relative_to_propagator: np.ndarray
    leave_one_out_errors_relative_to_propagator: np.ndarray
    maximum_closure_defect: float


@dataclass(frozen=True)
class CausalTrackedSymbolBranches:
    """Overlap-continuous eigenbranches along one ordered spatial path."""

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    consecutive_overlaps: np.ndarray
    minimum_consecutive_overlap: float


def _normalized_eigensystem(
    generator: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eig(np.asarray(generator, dtype=complex))
    norms = np.linalg.norm(vectors, axis=0)
    if np.any(norms <= np.finfo(float).tiny):
        raise RuntimeError("symbol eigenvector has zero norm")
    return values, vectors / norms[None, :]


def _branch_assignment(
    reference_values: np.ndarray,
    reference_vectors: np.ndarray,
    candidate_values: np.ndarray,
    candidate_vectors: np.ndarray,
) -> np.ndarray:
    overlaps = np.abs(reference_vectors.conj().T @ candidate_vectors)
    overlap_scale = np.maximum(
        np.max(overlaps, axis=1, keepdims=True),
        np.finfo(float).tiny,
    )
    frequency_scale = max(
        float(np.max(np.abs(reference_values))),
        float(np.max(np.abs(candidate_values))),
        1.0,
    )
    cost = (
        1.0
        - overlaps / overlap_scale
        + 0.02
        * np.abs(
            reference_values[:, None] - candidate_values[None, :]
        )
        / frequency_scale
    )
    rows, columns = linear_sum_assignment(cost)
    return columns[np.argsort(rows)]


def causal_track_symbol_eigenbranches(
    generators_per_s: np.ndarray,
) -> CausalTrackedSymbolBranches:
    """Track separated right-eigenvector branches by adjacent overlap."""

    generators = np.asarray(generators_per_s, dtype=complex)
    if (
        generators.ndim != 3
        or generators.shape[1:] != (_N_FIELDS, _N_FIELDS)
        or generators.shape[0] < 2
        or np.any(~np.isfinite(generators))
    ):
        raise ValueError("branch-tracking generators are invalid")
    count = int(generators.shape[0])
    values = np.empty((count, _N_FIELDS), dtype=complex)
    vectors = np.empty(
        (count, _N_FIELDS, _N_FIELDS),
        dtype=complex,
    )
    overlaps = np.empty((count - 1, _N_FIELDS), dtype=float)
    first_values, first_vectors = _normalized_eigensystem(generators[0])
    first_order = np.argsort(np.imag(first_values))
    values[0] = first_values[first_order]
    vectors[0] = first_vectors[:, first_order]
    for index in range(1, count):
        candidate_values, candidate_vectors = _normalized_eigensystem(
            generators[index]
        )
        order = _branch_assignment(
            values[index - 1],
            vectors[index - 1],
            candidate_values,
            candidate_vectors,
        )
        current_values = candidate_values[order]
        current_vectors = candidate_vectors[:, order]
        for family in range(_N_FIELDS):
            product = np.vdot(
                vectors[index - 1, :, family],
                current_vectors[:, family],
            )
            if abs(product) > np.finfo(float).tiny:
                current_vectors[:, family] *= np.exp(
                    -1.0j * np.angle(product)
                )
            overlaps[index - 1, family] = abs(
                np.vdot(
                    vectors[index - 1, :, family],
                    current_vectors[:, family],
                )
            )
        values[index] = current_values
        vectors[index] = current_vectors
    return CausalTrackedSymbolBranches(
        eigenvalues=values,
        eigenvectors=vectors,
        consecutive_overlaps=overlaps,
        minimum_consecutive_overlap=float(np.min(overlaps)),
    )


def causal_match_symbol_eigenvalues_to_tracked_branches(
    generators_per_s: np.ndarray,
    tracked: CausalTrackedSymbolBranches,
) -> np.ndarray:
    """Match one perturbed eigensystem at each point to tracked branches."""

    generators = np.asarray(generators_per_s, dtype=complex)
    if generators.shape != (
        tracked.eigenvalues.shape[0],
        _N_FIELDS,
        _N_FIELDS,
    ):
        raise ValueError("tracked-branch matching shape is invalid")
    result = np.empty_like(tracked.eigenvalues)
    for index, generator in enumerate(generators):
        values, vectors = _normalized_eigensystem(generator)
        order = _branch_assignment(
            tracked.eigenvalues[index],
            tracked.eigenvectors[index],
            values,
            vectors,
        )
        result[index] = values[order]
    return result


def _hybrid_propagators(
    numerical: CausalNormalizedLocalDAESymbol,
    continuum: CausalNormalizedLocalDAESymbol,
    time: float,
) -> dict[int, np.ndarray]:
    names = tuple(numerical.operators)
    if names != _OPERATOR_NAMES or tuple(continuum.operators) != names:
        raise ValueError("local DAE operator groups are not canonical")
    interval = float(time)
    if not np.isfinite(interval) or interval <= 0.0:
        raise ValueError("Shapley propagation time is invalid")
    count = 1 + len(names)
    result = {}
    for mask in range(1 << count):
        descriptor = (
            numerical.descriptor
            if mask & 1
            else continuum.descriptor
        )
        operator = np.zeros((_N_FIELDS, _N_FIELDS), dtype=complex)
        for index, name in enumerate(names, start=1):
            operator += (
                numerical.operators[name]
                if mask & (1 << index)
                else continuum.operators[name]
            )
        generator = -np.linalg.solve(descriptor, operator)
        result[mask] = expm(interval * generator)
    return result


def causal_symbol_shapley_attribution(
    numerical: CausalNormalizedLocalDAESymbol,
    continuum: CausalNormalizedLocalDAESymbol,
    time: float,
) -> CausalSymbolShapleyAttribution:
    """Allocate a complete propagator difference over all DAE groups."""

    propagators = _hybrid_propagators(numerical, continuum, time)
    names = _PLAYER_NAMES
    count = len(names)
    full_mask = (1 << count) - 1
    continuum_step = propagators[0]
    numerical_step = propagators[full_mask]
    total = numerical_step - continuum_step
    contributions = np.zeros(
        (count, _N_FIELDS, _N_FIELDS),
        dtype=complex,
    )
    denominator = math.factorial(count)
    for player in range(count):
        remaining = tuple(
            index for index in range(count) if index != player
        )
        for size in range(count):
            weight = (
                math.factorial(size)
                * math.factorial(count - size - 1)
                / denominator
            )
            for subset in combinations(remaining, size):
                mask = sum(1 << index for index in subset)
                contributions[player] += weight * (
                    propagators[mask | (1 << player)]
                    - propagators[mask]
                )

    closure = _relative_norm(
        np.sum(contributions, axis=0) - total,
        total,
        numerical_step,
        continuum_step,
    )
    total_norm = max(float(np.linalg.norm(total)), np.finfo(float).tiny)
    propagator_scale = max(
        float(np.linalg.norm(numerical_step)),
        float(np.linalg.norm(continuum_step)),
        np.finfo(float).tiny,
    )
    contribution_norms = np.linalg.norm(
        contributions.reshape(count, -1),
        axis=1,
    )
    cosines = np.zeros(count, dtype=float)
    total_flat = total.ravel()
    for index in range(count):
        scale = max(
            float(contribution_norms[index]) * total_norm,
            np.finfo(float).tiny,
        )
        cosines[index] = float(
            np.real(np.vdot(contributions[index].ravel(), total_flat))
            / scale
        )
    gram = np.empty((count, count), dtype=float)
    for first in range(count):
        for second in range(count):
            gram[first, second] = float(
                np.real(
                    np.vdot(
                        contributions[first].ravel(),
                        contributions[second].ravel(),
                    )
                )
                / (total_norm * total_norm)
            )
    only = np.empty(count, dtype=float)
    leave = np.empty(count, dtype=float)
    for player in range(count):
        only[player] = float(
            np.linalg.norm(
                propagators[1 << player] - continuum_step
            )
            / propagator_scale
        )
        leave[player] = float(
            np.linalg.norm(
                numerical_step
                - propagators[full_mask ^ (1 << player)]
            )
            / propagator_scale
        )
    return CausalSymbolShapleyAttribution(
        player_names=names,
        continuum_propagator=np.asarray(continuum_step),
        numerical_propagator=np.asarray(numerical_step),
        total_difference=np.asarray(total),
        contributions=np.asarray(contributions),
        contribution_norms_relative_to_total=(
            np.asarray(contribution_norms / total_norm, dtype=float)
        ),
        contribution_norms_relative_to_propagator=(
            np.asarray(contribution_norms / propagator_scale, dtype=float)
        ),
        contribution_cosines_with_total=cosines,
        gram_matrix_relative_to_total=gram,
        only_player_errors_relative_to_propagator=only,
        leave_one_out_errors_relative_to_propagator=leave,
        maximum_closure_defect=float(closure),
    )


def causal_time_ordered_symbol_propagator(
    generators_per_s: np.ndarray,
    time_steps_s: np.ndarray,
) -> np.ndarray:
    """Return the left time-ordered product of local matrix exponentials."""

    generators = np.asarray(generators_per_s, dtype=complex)
    steps = np.asarray(time_steps_s, dtype=float).ravel()
    if (
        generators.ndim != 3
        or generators.shape[1:] != (_N_FIELDS, _N_FIELDS)
        or generators.shape[0] != steps.size
        or np.any(~np.isfinite(generators))
        or np.any(~np.isfinite(steps))
        or np.any(steps <= 0.0)
    ):
        raise ValueError("time-ordered symbol inputs are invalid")
    result = np.eye(_N_FIELDS, dtype=complex)
    for generator, interval in zip(generators, steps, strict=True):
        result = expm(float(interval) * generator) @ result
    return result


__all__ = [
    "CausalLocalDAEComponentStencil",
    "CausalNormalizedLocalDAESymbol",
    "CausalSymbolShapleyAttribution",
    "CausalTrackedSymbolBranches",
    "causal_match_symbol_eigenvalues_to_tracked_branches",
    "causal_continuum_normalized_local_dae",
    "causal_local_dae_component_stencil",
    "causal_symbol_shapley_attribution",
    "causal_track_symbol_eigenbranches",
    "causal_time_ordered_symbol_propagator",
]
