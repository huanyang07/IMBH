"""Audit-only shear root-cause tools for WP10c9c0.

This module does not define a production numerical flux.  It exposes the
sign-explicit principal matrices, the complete-coordinate shear invariant
subspace, a physical local-rest shear energy, and frozen constant-coefficient
symbols for comparing the current split discretization with a monolithic
complete-principal reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imri_qpe.constants import C

from .causal_inner_characteristic_dissipation import (
    DEFAULT_COORDINATE_PRINCIPAL_RELATIVE_STEP,
    CausalFiveFieldCoordinatePrincipalBasis,
    CausalFiveFieldCoordinatePrincipalComponents,
    causal_five_field_coordinate_principal_basis,
    causal_five_field_coordinate_principal_components,
)
from .causal_inner_dae import audit_causal_five_field_principal
from .causal_inner_dae_system import CausalFiveFieldDAEContext, _cell_state
from .causal_inner_stress import causal_stress_relaxation_source


_FIVE_POINT_MULTIPLIERS = np.asarray(
    [-2.0, -1.0, 1.0, 2.0],
    dtype=float,
)
_FIVE_POINT_WEIGHTS = np.asarray(
    [1.0, -8.0, 8.0, -1.0],
    dtype=float,
) / 12.0
_SHEAR_FAMILY_INDICES = np.asarray([1, 3], dtype=int)
_SHEAR_PRIMITIVE_INDICES = np.asarray([2, 4], dtype=int)


@dataclass(frozen=True)
class CausalShearInvariantSubspace:
    """Two-family shear subspace and its physical energy."""

    coordinate_basis: CausalFiveFieldCoordinatePrincipalBasis
    primitive_right_eigenvectors: np.ndarray
    primitive_left_eigenvectors: np.ndarray
    primitive_projector: np.ndarray
    coordinate_speeds_over_c: np.ndarray
    local_rest_evolution_matrix: np.ndarray
    analytic_local_rest_speeds_over_c: np.ndarray
    analytic_local_rest_projectors: np.ndarray
    local_rest_symmetrizer: np.ndarray
    coordinate_energy_gram: np.ndarray
    maximum_projector_idempotence_defect: float
    maximum_projector_complement_defect: float
    maximum_local_rest_symmetry_defect: float
    maximum_analytic_local_projector_defect: float
    maximum_analytic_local_eigenpair_defect: float
    minimum_local_rest_energy_eigenvalue: float
    minimum_coordinate_energy_eigenvalue: float
    coordinate_energy_condition_number: float


@dataclass(frozen=True)
class CausalShearFourierSymbols:
    """Continuum, current-split, and monolithic Fourier generators."""

    theta: float
    spacing: float
    wavenumber: float
    continuum_principal_per_s: np.ndarray
    continuum_relaxing_per_s: np.ndarray
    current_split_principal_per_s: np.ndarray
    current_split_relaxing_per_s: np.ndarray
    monolithic_principal_per_s: np.ndarray
    monolithic_relaxing_per_s: np.ndarray
    monolithic_centered_principal_per_s: np.ndarray
    monolithic_centered_relaxing_per_s: np.ndarray
    physical_flux_only_principal_per_s: np.ndarray
    principal_source_only_per_s: np.ndarray


@dataclass(frozen=True)
class CausalPrincipalManufacturedWaveAudit:
    """Variable-coefficient principal residual for one independent wave."""

    exact_complete_principal: np.ndarray
    current_split_principal: np.ndarray
    monolithic_principal: np.ndarray
    current_split_relative_l2_error: float
    monolithic_relative_l2_error: float


def _relative_matrix_defect(first: np.ndarray, second: np.ndarray) -> float:
    numerator = float(np.max(np.abs(np.asarray(first) - np.asarray(second))))
    denominator = max(
        float(np.max(np.abs(first))),
        float(np.max(np.abs(second))),
        np.finfo(float).tiny,
    )
    return numerator / denominator


def causal_five_field_lower_stress_relaxation_matrix(
    context: CausalFiveFieldDAEContext,
    radius: float,
    primitive_chart: np.ndarray,
    *,
    relative_step: float = DEFAULT_COORDINATE_PRINCIPAL_RELATIVE_STEP,
) -> np.ndarray:
    """Differentiate the local stress relaxation with all gradients zero."""

    context = context.validated()
    radius = float(radius)
    chart = np.asarray(primitive_chart, dtype=float)
    relative_step = float(relative_step)
    components = causal_five_field_coordinate_principal_components(
        context,
        radius,
        chart,
        relative_step=relative_step,
    )
    steps = relative_step * components.primitive_column_scales
    result = np.zeros((5, 5), dtype=float)
    for column, step in enumerate(steps):
        samples = []
        for multiplier in _FIVE_POINT_MULTIPLIERS:
            candidate = np.array(chart, copy=True)
            candidate[column] += multiplier * step
            state = _cell_state(context, radius, candidate)
            samples.append(
                causal_stress_relaxation_source(
                    state.geometry,
                    state.stress,
                    state.closure,
                    positive_shear_rate=0.0,
                )
            )
        result[4, column] = float(
            _FIVE_POINT_WEIGHTS @ np.asarray(samples, dtype=float) / step
        )
    return result


def causal_five_field_straight_principal_path_jump(
    context: CausalFiveFieldDAEContext,
    radius: float,
    left_chart: np.ndarray,
    right_chart: np.ndarray,
    *,
    quadrature_order: int = 8,
    relative_step: float = DEFAULT_COORDINATE_PRINCIPAL_RELATIVE_STEP,
    face_measure: float = 1.0,
) -> np.ndarray:
    """Return ``Delta F - integral C_pr(Psi) Psi_s ds`` on a straight path."""

    context = context.validated()
    radius = float(radius)
    left = np.asarray(left_chart, dtype=float)
    right = np.asarray(right_chart, dtype=float)
    measure = float(face_measure)
    order = int(quadrature_order)
    if (
        left.shape != (5,)
        or right.shape != (5,)
        or np.any(~np.isfinite(left))
        or np.any(~np.isfinite(right))
        or not np.isfinite(measure)
        or measure <= 0.0
        or order < 2
    ):
        raise ValueError("principal path-jump inputs are invalid")
    delta = right - left
    nodes, weights = np.polynomial.legendre.leggauss(order)
    integrated = np.zeros(5, dtype=float)
    for node, weight in zip(nodes, weights, strict=True):
        fraction = 0.5 * (float(node) + 1.0)
        chart = left + fraction * delta
        components = causal_five_field_coordinate_principal_components(
            context,
            radius,
            chart,
            relative_step=relative_step,
        )
        integrated += (
            0.5
            * float(weight)
            * (components.principal_source_matrix @ delta)
        )
    left_flux = _cell_state(context, radius, left).flux_over_c
    right_flux = _cell_state(context, radius, right).flux_over_c
    return measure * (
        np.asarray(right_flux - left_flux, dtype=float) - integrated
    )


def causal_five_field_shear_invariant_subspace(
    context: CausalFiveFieldDAEContext,
    radius: float,
    primitive_chart: np.ndarray,
    *,
    relative_step: float = DEFAULT_COORDINATE_PRINCIPAL_RELATIVE_STEP,
) -> CausalShearInvariantSubspace:
    """Return the complete-coordinate shear projector and physical energy."""

    context = context.validated()
    radius = float(radius)
    chart = np.asarray(primitive_chart, dtype=float)
    basis = causal_five_field_coordinate_principal_basis(
        context,
        radius,
        chart,
        relative_step=relative_step,
    )
    if basis.maximum_imaginary_part > 1.0e-10:
        raise RuntimeError("complete coordinate shear pencil is not real")
    full_right = np.asarray(
        basis.primitive_right_eigenvectors,
        dtype=float,
    )
    full_left = np.linalg.inv(full_right)
    shear_right = full_right[:, _SHEAR_FAMILY_INDICES]
    shear_left = full_left[_SHEAR_FAMILY_INDICES]
    projector = shear_right @ shear_left

    state = _cell_state(context, radius, chart)
    local = audit_causal_five_field_principal(
        state.geometry,
        context.vertical_frequency.eos(radius),
        state.closure,
        surface_density=state.primitive.surface_density,
        radial_velocity_over_c=state.primitive.radial_velocity_over_c,
        azimuthal_velocity_over_c=state.primitive.azimuthal_velocity_over_c,
        temperature=state.thermodynamics.temperature,
    )
    local_mass = np.asarray(local.local_rest_mass_matrix, dtype=float)[
        np.ix_(_SHEAR_PRIMITIVE_INDICES, _SHEAR_PRIMITIVE_INDICES)
    ]
    local_flux = np.asarray(local.local_rest_flux_matrix, dtype=float)[
        np.ix_(_SHEAR_PRIMITIVE_INDICES, _SHEAR_PRIMITIVE_INDICES)
    ]
    local_evolution = np.linalg.solve(local_mass, local_flux)
    enthalpy = float(local_mass[0, 0])
    viscous_speed_squared = float(local_flux[1, 0] / enthalpy)
    if enthalpy <= 0.0 or viscous_speed_squared <= 0.0:
        raise RuntimeError("local-rest shear energy is not positive")
    symmetrizer = np.diag(
        [enthalpy, 1.0 / (enthalpy * viscous_speed_squared)]
    )
    viscous_speed = float(np.sqrt(viscous_speed_squared))
    analytic_projectors = np.asarray(
        [
            0.5 * (np.eye(2) - local_evolution / viscous_speed),
            0.5 * (np.eye(2) + local_evolution / viscous_speed),
        ],
        dtype=float,
    )
    analytic_speeds = np.asarray(
        [-viscous_speed, viscous_speed],
        dtype=float,
    )
    analytic_projector_defect = max(
        float(
            np.max(
                np.abs(projector_i @ projector_i - projector_i)
            )
        )
        for projector_i in analytic_projectors
    )
    analytic_projector_defect = max(
        analytic_projector_defect,
        float(
            np.max(
                np.abs(
                    analytic_projectors[0] @ analytic_projectors[1]
                )
            )
        ),
        float(
            np.max(
                np.abs(
                    np.sum(analytic_projectors, axis=0) - np.eye(2)
                )
            )
        ),
    )
    analytic_eigenpair_defect = max(
        float(
            np.max(
                np.abs(
                    local_evolution @ analytic_projectors[index]
                    - speed * analytic_projectors[index]
                )
            )
        )
        for index, speed in enumerate(analytic_speeds)
    )
    trace_map = shear_right[_SHEAR_PRIMITIVE_INDICES]
    energy_gram = trace_map.T @ symmetrizer @ trace_map
    local_eigenvalues = np.linalg.eigvalsh(symmetrizer)
    coordinate_eigenvalues = np.linalg.eigvalsh(energy_gram)
    complement = np.eye(5) - projector
    return CausalShearInvariantSubspace(
        coordinate_basis=basis,
        primitive_right_eigenvectors=np.asarray(shear_right, dtype=float),
        primitive_left_eigenvectors=np.asarray(shear_left, dtype=float),
        primitive_projector=np.asarray(projector, dtype=float),
        coordinate_speeds_over_c=np.asarray(
            basis.numerical_speeds_over_c[_SHEAR_FAMILY_INDICES],
            dtype=float,
        ),
        local_rest_evolution_matrix=np.asarray(
            local_evolution,
            dtype=float,
        ),
        analytic_local_rest_speeds_over_c=analytic_speeds,
        analytic_local_rest_projectors=analytic_projectors,
        local_rest_symmetrizer=np.asarray(symmetrizer, dtype=float),
        coordinate_energy_gram=np.asarray(energy_gram, dtype=float),
        maximum_projector_idempotence_defect=float(
            np.max(np.abs(projector @ projector - projector))
        ),
        maximum_projector_complement_defect=float(
            max(
                np.max(np.abs(projector @ complement)),
                np.max(np.abs(complement @ projector)),
            )
        ),
        maximum_local_rest_symmetry_defect=float(
            np.max(
                np.abs(
                    symmetrizer @ local_evolution
                    - (symmetrizer @ local_evolution).T
                )
            )
        ),
        maximum_analytic_local_projector_defect=(
            analytic_projector_defect
        ),
        maximum_analytic_local_eigenpair_defect=(
            analytic_eigenpair_defect
        ),
        minimum_local_rest_energy_eigenvalue=float(
            np.min(local_eigenvalues)
        ),
        minimum_coordinate_energy_eigenvalue=float(
            np.min(coordinate_eigenvalues)
        ),
        coordinate_energy_condition_number=float(
            np.linalg.cond(energy_gram)
        ),
    )


def _quadratic_reconstruction_symbols(
    theta: float,
    spacing: float,
) -> tuple[complex, complex, complex]:
    """Return central-face, jump, and arithmetic derivative symbols."""

    theta = float(theta)
    spacing = float(spacing)
    if (
        not np.isfinite(theta)
        or not np.isfinite(spacing)
        or spacing <= 0.0
    ):
        raise ValueError("Fourier stencil inputs are invalid")
    mode = np.exp(1.0j * theta)
    inverse = 1.0 / mode
    left_face = -0.125 * inverse + 0.75 + 0.375 * mode
    right_face = 0.375 + 0.75 * mode - 0.125 * mode**2
    central_face = 0.5 * (left_face + right_face)
    face_factor = (1.0 - inverse) / spacing
    central_derivative = central_face * face_factor
    jump_derivative = (right_face - left_face) * face_factor
    arithmetic_derivative = 0.5 * (mode - inverse) / spacing
    return central_derivative, jump_derivative, arithmetic_derivative


def causal_quadratic_reconstruction_matrices(
    centers: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the unlimited three-cell left/right face maps.

    The map is the linearization of the production quadratic reconstruction
    before its nonlinear admissibility limiter.  Boundary traces are copied
    from the adjacent cell, as in the bounded packet audit.
    """

    centers = np.asarray(centers, dtype=float)
    edges = np.asarray(edges, dtype=float)
    n_cells = int(centers.size)
    if (
        n_cells < 3
        or edges.shape != (n_cells + 1,)
        or np.any(~np.isfinite(centers))
        or np.any(~np.isfinite(edges))
        or np.any(np.diff(centers) <= 0.0)
        or np.any(np.diff(edges) <= 0.0)
    ):
        raise ValueError("quadratic reconstruction grid is invalid")

    def weights(nodes: np.ndarray, target: float) -> np.ndarray:
        result = np.ones(3, dtype=float)
        for first in range(3):
            for second in range(3):
                if first != second:
                    result[first] *= (
                        target - nodes[second]
                    ) / (nodes[first] - nodes[second])
        return result

    left = np.zeros((n_cells + 1, n_cells), dtype=float)
    right = np.zeros_like(left)
    left[0, 0] = 1.0
    right[0, 0] = 1.0
    left[-1, -1] = 1.0
    right[-1, -1] = 1.0
    for cell in range(n_cells):
        start = min(max(cell - 1, 0), n_cells - 3)
        indices = np.arange(start, start + 3)
        nodes = centers[indices]
        left_candidate = weights(nodes, float(edges[cell]))
        right_candidate = weights(nodes, float(edges[cell + 1]))
        if cell > 0:
            right[cell, indices] = left_candidate
        if cell < n_cells - 1:
            left[cell + 1, indices] = right_candidate
    return left, right


def _principal_derivative_matrices(
    context: CausalFiveFieldDAEContext,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log_centers = np.log(np.asarray(context.grid.centers, dtype=float))
    log_edges = np.log(np.asarray(context.grid.edges, dtype=float))
    left, right = causal_quadratic_reconstruction_matrices(
        log_centers,
        log_edges,
    )
    widths = np.diff(np.asarray(context.grid.edges, dtype=float))
    central_faces = 0.5 * (left + right)
    quadratic = (
        central_faces[1:] - central_faces[:-1]
    ) / widths[:, None]
    arithmetic_faces = np.zeros_like(central_faces)
    arithmetic_faces[0, 0] = 1.0
    arithmetic_faces[-1, -1] = 1.0
    for face in range(1, context.grid.centers.size):
        arithmetic_faces[face, face - 1 : face + 1] = 0.5
    arithmetic = (
        arithmetic_faces[1:] - arithmetic_faces[:-1]
    ) / widths[:, None]
    return quadratic, arithmetic, right - left


def causal_five_field_frozen_principal_generator(
    context: CausalFiveFieldDAEContext,
    primitive_charts: np.ndarray,
    primitive_amplitudes: np.ndarray,
    *,
    operator: str,
    include_physical_relaxation: bool = True,
    include_characteristic_dissipation: bool = True,
    relative_step: float = DEFAULT_COORDINATE_PRINCIPAL_RELATIVE_STEP,
) -> np.ndarray:
    """Build an audit-only split or monolithic primitive generator.

    ``current_split`` applies the quadratic face derivative to ``F_p`` and
    the arithmetic cell derivative to ``C_pr``. ``monolithic`` applies the
    same quadratic derivative to the complete ``B = F_p - C_pr`` matrix.
    Both variants use the same complete-pencil characteristic penalty when
    requested, so their difference isolates the principal split.
    """

    context = context.validated()
    charts = np.asarray(primitive_charts, dtype=float)
    amplitudes = np.asarray(primitive_amplitudes, dtype=float)
    n_cells = int(context.grid.centers.size)
    if (
        operator not in ("current_split", "monolithic")
        or charts.shape != (n_cells, 5)
        or amplitudes.shape != charts.shape
        or np.any(~np.isfinite(charts))
        or np.any(~np.isfinite(amplitudes))
        or np.any(amplitudes <= 0.0)
    ):
        raise ValueError("frozen principal generator inputs are invalid")
    quadratic, arithmetic, face_jump = _principal_derivative_matrices(
        context
    )
    components = tuple(
        causal_five_field_coordinate_principal_components(
            context,
            float(radius),
            chart,
            relative_step=relative_step,
        )
        for radius, chart in zip(
            context.grid.centers,
            charts,
            strict=True,
        )
    )
    lower = tuple(
        causal_five_field_lower_stress_relaxation_matrix(
            context,
            float(radius),
            chart,
            relative_step=relative_step,
        )
        if include_physical_relaxation
        else np.zeros((5, 5), dtype=float)
        for radius, chart in zip(
            context.grid.centers,
            charts,
            strict=True,
        )
    )
    physical = np.zeros((5 * n_cells, 5 * n_cells), dtype=float)
    for cell in range(n_cells):
        temporal = components[cell].temporal_storage_matrix
        for neighbor in range(n_cells):
            if operator == "current_split":
                spatial = (
                    quadratic[cell, neighbor]
                    * components[cell].physical_flux_matrix
                    - arithmetic[cell, neighbor]
                    * components[cell].principal_source_matrix
                )
            else:
                spatial = (
                    quadratic[cell, neighbor]
                    * components[cell].spatial_principal_matrix
                )
            if cell == neighbor:
                spatial = spatial - lower[cell]
            block = C * np.linalg.solve(temporal, -spatial)
            rows = slice(5 * cell, 5 * (cell + 1))
            columns = slice(5 * neighbor, 5 * (neighbor + 1))
            physical[rows, columns] += block

    if include_characteristic_dissipation:
        reconstruction_left, reconstruction_right = (
            causal_quadratic_reconstruction_matrices(
                np.log(np.asarray(context.grid.centers, dtype=float)),
                np.log(np.asarray(context.grid.edges, dtype=float)),
            )
        )
        midpoint_charts = 0.5 * (
            reconstruction_left @ charts
            + reconstruction_right @ charts
        )
        penalty_faces = np.zeros(
            (n_cells + 1, 5, 5 * n_cells),
            dtype=float,
        )
        for face in range(1, n_cells):
            basis = causal_five_field_coordinate_principal_basis(
                context,
                float(context.grid.edges[face]),
                midpoint_charts[face],
                relative_step=relative_step,
            )
            right = np.asarray(
                basis.primitive_right_eigenvectors,
                dtype=float,
            )
            absolute = (
                basis.temporal_storage_matrix
                @ right
                @ np.diag(np.abs(basis.numerical_speeds_over_c))
                @ np.linalg.inv(right)
            )
            jump_map = face_jump[face]
            for neighbor, coefficient in enumerate(jump_map):
                columns = slice(5 * neighbor, 5 * (neighbor + 1))
                penalty_faces[face, :, columns] = (
                    -0.5 * float(coefficient) * absolute
                )
        widths = np.diff(np.asarray(context.grid.edges, dtype=float))
        for cell in range(n_cells):
            divergence = (
                penalty_faces[cell + 1] - penalty_faces[cell]
            ) / widths[cell]
            rows = slice(5 * cell, 5 * (cell + 1))
            physical[rows] += (
                -C
                * np.linalg.solve(
                    components[cell].temporal_storage_matrix,
                    divergence,
                )
            )

    scale = amplitudes.ravel()
    return physical * scale[None, :] / scale[:, None]


def causal_five_field_manufactured_principal_wave(
    context: CausalFiveFieldDAEContext,
    primitive_charts: np.ndarray,
    direction: np.ndarray,
    *,
    support_inner_radius: float,
    support_outer_radius: float,
    cycles: float = 1.5,
    relative_step: float = DEFAULT_COORDINATE_PRINCIPAL_RELATIVE_STEP,
) -> CausalPrincipalManufacturedWaveAudit:
    """Audit a smooth variable-coefficient wave against an analytic gradient."""

    context = context.validated()
    charts = np.asarray(primitive_charts, dtype=float)
    direction = np.asarray(direction, dtype=float)
    n_cells = int(context.grid.centers.size)
    inner = float(support_inner_radius)
    outer = float(support_outer_radius)
    if (
        charts.shape != (n_cells, 5)
        or direction.shape != (5,)
        or np.any(~np.isfinite(charts))
        or np.any(~np.isfinite(direction))
        or not 0.0 < inner < outer
        or not np.isfinite(cycles)
        or cycles <= 0.0
    ):
        raise ValueError("manufactured principal-wave inputs are invalid")
    log_centers = np.log(np.asarray(context.grid.centers, dtype=float))
    log_inner = np.log(inner)
    log_outer = np.log(outer)
    wavenumber = 2.0 * np.pi * float(cycles) / (log_outer - log_inner)
    phase = wavenumber * (log_centers - log_inner) + 0.37
    wave = np.sin(phase)[:, None] * direction[None, :]
    radial_derivative = (
        wavenumber
        * np.cos(phase)[:, None]
        * direction[None, :]
        / context.grid.centers[:, None]
    )
    quadratic, arithmetic, _jump = _principal_derivative_matrices(context)
    quadratic_derivative = quadratic @ wave
    arithmetic_derivative = arithmetic @ wave
    exact = np.empty_like(wave)
    split = np.empty_like(wave)
    monolithic = np.empty_like(wave)
    for cell, (radius, chart) in enumerate(
        zip(context.grid.centers, charts, strict=True)
    ):
        components = causal_five_field_coordinate_principal_components(
            context,
            float(radius),
            chart,
            relative_step=relative_step,
        )
        exact[cell] = (
            components.spatial_principal_matrix
            @ radial_derivative[cell]
        )
        split[cell] = (
            components.physical_flux_matrix
            @ quadratic_derivative[cell]
            - components.principal_source_matrix
            @ arithmetic_derivative[cell]
        )
        monolithic[cell] = (
            components.spatial_principal_matrix
            @ quadratic_derivative[cell]
        )
    active = (
        context.grid.centers >= inner
    ) & (
        context.grid.centers <= outer
    )
    if np.count_nonzero(active) < 4:
        raise ValueError("manufactured wave support has too few cells")
    weights = np.asarray(context.grid.cell_measures[active], dtype=float)
    exact_active = exact[active]
    scale = max(
        float(np.sqrt(np.sum(weights[:, None] * exact_active**2))),
        np.finfo(float).tiny,
    )

    def error(candidate: np.ndarray) -> float:
        difference = candidate[active] - exact_active
        return float(
            np.sqrt(np.sum(weights[:, None] * difference**2)) / scale
        )

    return CausalPrincipalManufacturedWaveAudit(
        exact_complete_principal=np.asarray(exact, dtype=float),
        current_split_principal=np.asarray(split, dtype=float),
        monolithic_principal=np.asarray(monolithic, dtype=float),
        current_split_relative_l2_error=error(split),
        monolithic_relative_l2_error=error(monolithic),
    )


def causal_five_field_shear_fourier_symbols(
    components: CausalFiveFieldCoordinatePrincipalComponents,
    basis: CausalFiveFieldCoordinatePrincipalBasis,
    lower_relaxation_matrix: np.ndarray,
    *,
    theta: float,
    spacing: float,
) -> CausalShearFourierSymbols:
    """Return frozen symbols for the current split and monolithic references."""

    temporal = np.asarray(components.temporal_storage_matrix, dtype=float)
    mapped = np.asarray(components.mapped_storage_matrix, dtype=float)
    physical_flux = np.asarray(components.physical_flux_matrix, dtype=float)
    principal_source = np.asarray(
        components.principal_source_matrix,
        dtype=float,
    )
    spatial = np.asarray(components.spatial_principal_matrix, dtype=float)
    lower = np.asarray(lower_relaxation_matrix, dtype=float)
    if lower.shape != (5, 5):
        raise ValueError("lower relaxation matrix must be five by five")
    central, jump, arithmetic = _quadratic_reconstruction_symbols(
        theta,
        spacing,
    )
    wavenumber = float(theta) / float(spacing)
    right = np.asarray(basis.primitive_right_eigenvectors, dtype=float)
    left = np.linalg.inv(right)
    absolute_matrix = (
        temporal
        @ right
        @ np.diag(np.abs(basis.numerical_speeds_over_c))
        @ left
    )
    maximum_speed = float(np.max(np.abs(basis.numerical_speeds_over_c)))
    current_spatial = (
        central * physical_flux
        - arithmetic * principal_source
        - 0.5 * maximum_speed * jump * mapped
    )
    monolithic_spatial = (
        central * spatial - 0.5 * jump * absolute_matrix
    )
    centered_spatial = arithmetic * spatial
    continuum_spatial = 1.0j * wavenumber * spatial

    def generator(operator: np.ndarray, source: np.ndarray) -> np.ndarray:
        return C * np.linalg.solve(temporal, -operator + source)

    zero = np.zeros((5, 5), dtype=complex)
    return CausalShearFourierSymbols(
        theta=float(theta),
        spacing=float(spacing),
        wavenumber=wavenumber,
        continuum_principal_per_s=generator(continuum_spatial, zero),
        continuum_relaxing_per_s=generator(continuum_spatial, lower),
        current_split_principal_per_s=generator(current_spatial, zero),
        current_split_relaxing_per_s=generator(current_spatial, lower),
        monolithic_principal_per_s=generator(monolithic_spatial, zero),
        monolithic_relaxing_per_s=generator(monolithic_spatial, lower),
        monolithic_centered_principal_per_s=generator(
            centered_spatial,
            zero,
        ),
        monolithic_centered_relaxing_per_s=generator(
            centered_spatial,
            lower,
        ),
        physical_flux_only_principal_per_s=generator(
            central * physical_flux,
            zero,
        ),
        principal_source_only_per_s=generator(
            -arithmetic * principal_source,
            zero,
        ),
    )


def causal_five_field_principal_step_defects(
    components: tuple[CausalFiveFieldCoordinatePrincipalComponents, ...],
) -> dict[str, float]:
    """Return maximum adjacent-step defects for each principal matrix."""

    if len(components) < 2:
        raise ValueError("at least two principal component sets are required")
    names = (
        "mapped_storage_matrix",
        "vertical_storage_matrix",
        "temporal_storage_matrix",
        "physical_flux_matrix",
        "shear_principal_source_matrix",
        "vertical_principal_source_matrix",
        "principal_source_matrix",
        "spatial_principal_matrix",
    )
    return {
        name: max(
            _relative_matrix_defect(
                getattr(first, name),
                getattr(second, name),
            )
            for first, second in zip(
                components[:-1],
                components[1:],
                strict=True,
            )
        )
        for name in names
    }
