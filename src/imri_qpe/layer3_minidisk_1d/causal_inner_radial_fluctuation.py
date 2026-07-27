"""Production-neutral radial complete-fluctuation candidate.

WP10c9d4a certifies the reconstructed complete-fluctuation operator at fixed
geometry.  This module extends that audit contract to the actual nonuniform
Kerr--Schild radial grid without changing the production DAE.

The candidate keeps one shared conservative face flux.  At an interior face
the conservative part of the complete characteristic fluctuation defines that
shared flux, while the shear and responsive-height principal-source pieces
remain explicit nonconservative fluctuations.  Within each cell the
conservative contribution uses the two actual endpoint measures,

``A_R F(p_R) - A_L F(p_L)``,

and the derivative-dependent sources are integrated along the declared
``(log R, p)`` path.  Geometry, cooling, stream injection, lower-order height
work, and local Maxwell--Cattaneo relaxation are integrated exactly once and
reported in separate blocks.

The straight reconstructed path and midpoint eigensplit remain audit devices.
This module is not a promoted nonlinear Riemann solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imri_qpe.constants import C

from .causal_inner_characteristic_dissipation import (
    DEFAULT_COORDINATE_PRINCIPAL_RELATIVE_STEP,
    CausalFiveFieldCoordinatePrincipalBasis,
    causal_five_field_coordinate_principal_basis,
    causal_five_field_coordinate_principal_components,
)
from .causal_inner_dae_system import (
    CausalFiveFieldDAEContext,
    CausalFiveFieldFaceReconstruction,
    _cell_state,
    _gauss_legendre_cell_nodes_and_measures,
    _inner_face_flux,
    _interior_rusanov_flux,
    _local_cell_source_density,
    _outer_face_flux,
    causal_five_field_reconstruct_face_charts,
)
from .causal_inner_full_fluctuation import (
    CausalFiveFieldCompletePathJump,
    causal_five_field_complete_principal_path_jump,
)
from .causal_inner_stress import causal_rest_frame_shear_rate
from .causal_inner_thermal import kerr_schild_column_four_velocity


_N_FIELDS = 5
_EXPLICIT_GEOMETRY_LOG_RADIUS_STEP = 2.0e-5


@dataclass(frozen=True)
class CausalFiveFieldRadialPathJump:
    """Complete within-cell radial path with actual endpoint measures."""

    lower_radius: float
    upper_radius: float
    left_chart: np.ndarray
    right_chart: np.ndarray
    conservative_endpoint_jump_over_c: np.ndarray
    shear_source_path_integral_over_c: np.ndarray
    vertical_source_path_integral_over_c: np.ndarray
    principal_source_path_integral_over_c: np.ndarray
    total_principal_jump_over_c: np.ndarray
    source_partition_defect: float
    principal_closure_defect: float


@dataclass(frozen=True)
class CausalFiveFieldRadialInterfaceLedger:
    """Shared conservative flux and signed source fluctuations at faces."""

    production_shared_face_fluxes_over_c: np.ndarray
    candidate_shared_face_fluxes_over_c: np.ndarray
    conservative_face_adjustments_over_c: np.ndarray
    shear_left_cell_fluctuations_over_c: np.ndarray
    shear_right_cell_fluctuations_over_c: np.ndarray
    height_left_cell_fluctuations_over_c: np.ndarray
    height_right_cell_fluctuations_over_c: np.ndarray
    complete_path_jumps_over_c: np.ndarray
    shared_conservative_face_defect: float
    maximum_split_closure_defect: float
    incoming_excision_characteristics: int
    outer_boundary_choked: bool
    outer_incoming_characteristics: int


@dataclass(frozen=True)
class CausalFiveFieldRadialCandidateLedger:
    """Block-complete stationary residual of the radial audit candidate."""

    reconstruction: CausalFiveFieldFaceReconstruction
    interfaces: CausalFiveFieldRadialInterfaceLedger
    within_cell_paths: tuple[CausalFiveFieldRadialPathJump, ...]
    conservative_transport_rows: np.ndarray
    shear_principal_rows: np.ndarray
    height_principal_rows: np.ndarray
    local_stress_relaxation_rows: np.ndarray
    geometry_rows: np.ndarray
    cooling_rows: np.ndarray
    stream_rows: np.ndarray
    lower_height_work_rows: np.ndarray
    residual_rows: np.ndarray
    integrated_lower_source_components_per_ct: dict[str, np.ndarray]
    local_block_ledger_defect: float
    source_double_count_defect: float


def _validate_chart(chart: np.ndarray) -> np.ndarray:
    values = np.asarray(chart, dtype=float)
    if values.shape != (_N_FIELDS,) or np.any(~np.isfinite(values)):
        raise ValueError("primitive chart must contain five finite values")
    return values


def _weighted_physical_flux(
    context: CausalFiveFieldDAEContext,
    radius: float,
    chart: np.ndarray,
) -> np.ndarray:
    state = _cell_state(context, float(radius), _validate_chart(chart))
    return np.asarray(
        state.geometry.face_measure * state.flux_over_c,
        dtype=float,
    )


def _project_signed_halves(
    basis: CausalFiveFieldCoordinatePrincipalBasis,
    vector: np.ndarray,
    *,
    stationary_speed_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return left-cell and right-cell signed halves of one row vector."""

    values = np.asarray(vector, dtype=float)
    if values.shape != (_N_FIELDS,) or np.any(~np.isfinite(values)):
        raise ValueError("signed fluctuation vector is invalid")
    tolerance = float(stationary_speed_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("stationary speed tolerance must be nonnegative")
    coefficients = basis.descriptor_left_eigenvectors @ (
        values / basis.descriptor_row_scales
    )
    speeds = np.asarray(basis.numerical_speeds_over_c, dtype=float)

    def reconstruct(mask: np.ndarray) -> np.ndarray:
        selected = coefficients * np.asarray(mask, dtype=float)
        return basis.descriptor_row_scales * (
            basis.descriptor_right_eigenvectors @ selected
        )

    stationary = np.abs(speeds) <= tolerance
    left_cell = reconstruct(
        (speeds < -tolerance).astype(float)
        + 0.5 * stationary.astype(float)
    )
    right_cell = reconstruct(
        (speeds > tolerance).astype(float)
        + 0.5 * stationary.astype(float)
    )
    return (
        np.asarray(left_cell, dtype=float),
        np.asarray(right_cell, dtype=float),
    )


def causal_five_field_radial_extended_path_jump(
    context: CausalFiveFieldDAEContext,
    lower_radius: float,
    upper_radius: float,
    left_chart: np.ndarray,
    right_chart: np.ndarray,
    *,
    quadrature_order: int = 8,
    relative_step: float = DEFAULT_COORDINATE_PRINCIPAL_RELATIVE_STEP,
) -> CausalFiveFieldRadialPathJump:
    """Integrate the complete principal path across one physical radial cell.

    The declared path is linear in ``log R`` and in the primitive chart.  The
    conservative endpoints carry their own exact face measures.  Because
    ``p_R dR = dp``, the principal-source path uses the local face measure at
    each quadrature point without an additional radial Jacobian.
    """

    context = context.validated()
    lower = float(lower_radius)
    upper = float(upper_radius)
    left = _validate_chart(left_chart)
    right = _validate_chart(right_chart)
    order = int(quadrature_order)
    step = float(relative_step)
    if (
        not np.isfinite(lower)
        or not np.isfinite(upper)
        or lower <= 0.0
        or upper <= lower
        or order < 2
        or not np.isfinite(step)
        or step <= 0.0
    ):
        raise ValueError("radial complete-path inputs are invalid")

    delta = right - left
    nodes, weights = np.polynomial.legendre.leggauss(order)
    log_lower = float(np.log(lower))
    log_upper = float(np.log(upper))
    shear = np.zeros(_N_FIELDS, dtype=float)
    vertical = np.zeros(_N_FIELDS, dtype=float)
    principal = np.zeros(_N_FIELDS, dtype=float)
    for node, weight in zip(nodes, weights, strict=True):
        fraction = 0.5 * (float(node) + 1.0)
        radius = float(
            np.exp(log_lower + fraction * (log_upper - log_lower))
        )
        chart = left + fraction * delta
        components = causal_five_field_coordinate_principal_components(
            context,
            radius,
            chart,
            relative_step=step,
        )
        measure = _cell_state(context, radius, chart).geometry.face_measure
        quadrature_weight = 0.5 * float(weight) * measure
        shear += quadrature_weight * (
            components.shear_principal_source_matrix @ delta
        )
        vertical += quadrature_weight * (
            components.vertical_principal_source_matrix @ delta
        )
        principal += quadrature_weight * (
            components.principal_source_matrix @ delta
        )

    conservative = (
        _weighted_physical_flux(context, upper, right)
        - _weighted_physical_flux(context, lower, left)
    )
    total = conservative - principal
    scale = max(
        float(np.max(np.abs(conservative))),
        float(np.max(np.abs(principal))),
        float(np.max(np.abs(total))),
        np.finfo(float).tiny,
    )
    return CausalFiveFieldRadialPathJump(
        lower_radius=lower,
        upper_radius=upper,
        left_chart=np.array(left, copy=True),
        right_chart=np.array(right, copy=True),
        conservative_endpoint_jump_over_c=np.asarray(
            conservative,
            dtype=float,
        ),
        shear_source_path_integral_over_c=np.asarray(shear, dtype=float),
        vertical_source_path_integral_over_c=np.asarray(
            vertical,
            dtype=float,
        ),
        principal_source_path_integral_over_c=np.asarray(
            principal,
            dtype=float,
        ),
        total_principal_jump_over_c=np.asarray(total, dtype=float),
        source_partition_defect=float(
            np.max(np.abs(principal - shear - vertical)) / scale
        ),
        principal_closure_defect=float(
            np.max(np.abs(total - conservative + principal)) / scale
        ),
    )


def _explicit_geometry_rates(
    context: CausalFiveFieldDAEContext,
    radius: float,
    chart: np.ndarray,
) -> tuple[float, float]:
    """Return shear/height rates from explicit radial geometry at fixed p."""

    radius = float(radius)
    chart = _validate_chart(chart)
    step = _EXPLICIT_GEOMETRY_LOG_RADIUS_STEP
    minus_radius = radius * np.exp(-step)
    plus_radius = radius * np.exp(step)
    center = _cell_state(context, radius, chart)
    minus = _cell_state(context, minus_radius, chart)
    plus = _cell_state(context, plus_radius, chart)
    radial_width = plus_radius - minus_radius
    lower_minus = (
        minus.geometry.spacetime_metric
        @ kerr_schild_column_four_velocity(
            minus.geometry,
            minus.primitive,
        )
    )
    lower_plus = (
        plus.geometry.spacetime_metric
        @ kerr_schild_column_four_velocity(
            plus.geometry,
            plus.primitive,
        )
    )
    shear_rate = causal_rest_frame_shear_rate(
        center.geometry,
        center.primitive,
        radial_lower_four_velocity_derivative=(
            (lower_plus - lower_minus) / radial_width
        ),
    )
    log_height_derivative = (
        np.log(plus.thermodynamics.proper_half_thickness)
        - np.log(minus.thermodynamics.proper_half_thickness)
    ) / radial_width
    four_velocity = kerr_schild_column_four_velocity(
        center.geometry,
        center.primitive,
    )
    height_rate = C * four_velocity[1] * log_height_derivative
    return float(shear_rate), float(height_rate)


def _integrated_lower_sources(
    context: CausalFiveFieldDAEContext,
    cell: int,
    left_chart: np.ndarray,
    right_chart: np.ndarray,
) -> dict[str, np.ndarray]:
    """Integrate all non-principal sources exactly once in one cell."""

    names = (
        "perfect_fluid_geometry",
        "stress_geometry",
        "radiative_cooling",
        "vertical_work",
        "stress_relaxation",
        "stream",
    )
    result = {
        name: np.zeros(_N_FIELDS, dtype=float)
        for name in names
    }
    lower = float(np.log(context.grid.edges[cell]))
    upper = float(np.log(context.grid.edges[cell + 1]))
    left = _validate_chart(left_chart)
    right = _validate_chart(right_chart)
    radii, weights = _gauss_legendre_cell_nodes_and_measures(context, cell)
    for radius, weight in zip(radii, weights, strict=True):
        fraction = (float(np.log(radius)) - lower) / (upper - lower)
        chart = left + fraction * (right - left)
        state = _cell_state(context, float(radius), chart)
        shear_rate, height_rate = _explicit_geometry_rates(
            context,
            float(radius),
            chart,
        )
        _total, _optical_depth, components = _local_cell_source_density(
            context,
            state,
            shear_rate=shear_rate,
            height_rate=height_rate,
        )
        for name in names[:-1]:
            result[name] += float(weight) * np.asarray(
                components[name],
                dtype=float,
            )
    if context.stream_sources is not None:
        result["stream"][:4] = np.asarray(
            context.stream_sources.weighted_killing_source_per_ct[cell],
            dtype=float,
        )
    return result


def _interface_components(
    context: CausalFiveFieldDAEContext,
    face: int,
    left_chart: np.ndarray,
    right_chart: np.ndarray,
    production_flux: np.ndarray,
    *,
    quadrature_order: int,
    relative_step: float,
    stationary_speed_tolerance: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    CausalFiveFieldCompletePathJump,
    float,
    float,
]:
    """Return candidate flux and source fluctuations on one radial face."""

    radius = float(context.grid.edges[face])
    measure = float(context.grid.face_measures[face])
    left = _validate_chart(left_chart)
    right = _validate_chart(right_chart)
    path = causal_five_field_complete_principal_path_jump(
        context,
        radius,
        left,
        right,
        quadrature_order=quadrature_order,
        relative_step=relative_step,
        face_measure=measure,
    )
    midpoint = 0.5 * (left + right)
    basis = causal_five_field_coordinate_principal_basis(
        context,
        radius,
        midpoint,
        relative_step=relative_step,
    )
    conservative_left, conservative_right = _project_signed_halves(
        basis,
        path.conservative_flux_jump_over_c,
        stationary_speed_tolerance=stationary_speed_tolerance,
    )
    shear_left, shear_right = _project_signed_halves(
        basis,
        -path.shear_source_path_integral_over_c,
        stationary_speed_tolerance=stationary_speed_tolerance,
    )
    height_left, height_right = _project_signed_halves(
        basis,
        -path.vertical_source_path_integral_over_c,
        stationary_speed_tolerance=stationary_speed_tolerance,
    )
    left_flux = _weighted_physical_flux(context, radius, left)
    right_flux = _weighted_physical_flux(context, radius, right)
    candidate_from_left = left_flux + conservative_left
    candidate_from_right = right_flux - conservative_right
    scale = max(
        float(np.max(np.abs(candidate_from_left))),
        float(np.max(np.abs(candidate_from_right))),
        float(np.max(np.abs(right_flux - left_flux))),
        np.finfo(float).tiny,
    )
    shared_defect = float(
        np.max(np.abs(candidate_from_left - candidate_from_right)) / scale
    )
    complete_left = conservative_left + shear_left + height_left
    complete_right = conservative_right + shear_right + height_right
    split_scale = max(
        float(np.max(np.abs(path.total_principal_jump_over_c))),
        np.finfo(float).tiny,
    )
    split_defect = float(
        np.max(
            np.abs(
                complete_left
                + complete_right
                - path.total_principal_jump_over_c
            )
        )
        / split_scale
    )
    return (
        np.asarray(candidate_from_left, dtype=float),
        np.asarray(candidate_from_left - production_flux, dtype=float),
        np.asarray(shear_left, dtype=float),
        np.asarray(shear_right, dtype=float),
        np.asarray(height_left, dtype=float),
        np.asarray(height_right, dtype=float),
        path,
        shared_defect,
        split_defect,
    )


def causal_five_field_radial_candidate_ledger(
    context: CausalFiveFieldDAEContext,
    primitive_charts: np.ndarray,
    *,
    quadrature_order: int = 6,
    relative_step: float = DEFAULT_COORDINATE_PRINCIPAL_RELATIVE_STEP,
    stationary_speed_tolerance: float = 1.0e-12,
) -> CausalFiveFieldRadialCandidateLedger:
    """Assemble the production-neutral radial complete-fluctuation residual."""

    context = context.validated()
    charts = np.asarray(primitive_charts, dtype=float)
    n_cells = int(context.grid.centers.size)
    if (
        charts.shape != (n_cells, _N_FIELDS)
        or np.any(~np.isfinite(charts))
    ):
        raise ValueError("radial candidate primitive charts are invalid")
    reconstruction = causal_five_field_reconstruct_face_charts(
        context,
        charts,
        purpose="flux",
    )
    left_faces = np.asarray(reconstruction.left_face_charts, dtype=float)
    right_faces = np.asarray(reconstruction.right_face_charts, dtype=float)

    production_fluxes = np.zeros((n_cells + 1, _N_FIELDS), dtype=float)
    candidate_fluxes = np.zeros_like(production_fluxes)
    adjustments = np.zeros_like(production_fluxes)
    shear_left = np.zeros_like(production_fluxes)
    shear_right = np.zeros_like(production_fluxes)
    height_left = np.zeros_like(production_fluxes)
    height_right = np.zeros_like(production_fluxes)
    complete_jumps = np.zeros_like(production_fluxes)
    shared_defects = []
    split_defects = []

    production_fluxes[0] = _inner_face_flux(context, right_faces[0])
    candidate_fluxes[0] = production_fluxes[0]
    inner_basis = causal_five_field_coordinate_principal_basis(
        context,
        float(context.grid.edges[0]),
        right_faces[0],
        relative_step=relative_step,
    )
    incoming_excision = int(inner_basis.incoming_inner_characteristics)

    for face in range(1, n_cells):
        production_fluxes[face] = _interior_rusanov_flux(
            context,
            face,
            left_faces[face],
            right_faces[face],
        )
        (
            candidate_fluxes[face],
            adjustments[face],
            shear_left[face],
            shear_right[face],
            height_left[face],
            height_right[face],
            path,
            shared_defect,
            split_defect,
        ) = _interface_components(
            context,
            face,
            left_faces[face],
            right_faces[face],
            production_fluxes[face],
            quadrature_order=quadrature_order,
            relative_step=relative_step,
            stationary_speed_tolerance=stationary_speed_tolerance,
        )
        complete_jumps[face] = path.total_principal_jump_over_c
        shared_defects.append(shared_defect)
        split_defects.append(split_defect)

    production_outer, choked, outer_incoming = _outer_face_flux(
        context,
        left_faces[-1],
    )
    production_fluxes[-1] = production_outer
    candidate_fluxes[-1] = production_outer
    if context.outer_boundary_flux_mode == "frozen_exterior_rusanov":
        exterior = np.asarray(
            context.outer_boundary_frozen_exterior_chart,
            dtype=float,
        )
        (
            candidate_fluxes[-1],
            adjustments[-1],
            shear_left[-1],
            _unused_shear_right,
            height_left[-1],
            _unused_height_right,
            path,
            shared_defect,
            split_defect,
        ) = _interface_components(
            context,
            n_cells,
            left_faces[-1],
            exterior,
            production_outer,
            quadrature_order=quadrature_order,
            relative_step=relative_step,
            stationary_speed_tolerance=stationary_speed_tolerance,
        )
        complete_jumps[-1] = path.total_principal_jump_over_c
        shared_defects.append(shared_defect)
        split_defects.append(split_defect)

    interface_ledger = CausalFiveFieldRadialInterfaceLedger(
        production_shared_face_fluxes_over_c=production_fluxes,
        candidate_shared_face_fluxes_over_c=candidate_fluxes,
        conservative_face_adjustments_over_c=adjustments,
        shear_left_cell_fluctuations_over_c=shear_left,
        shear_right_cell_fluctuations_over_c=shear_right,
        height_left_cell_fluctuations_over_c=height_left,
        height_right_cell_fluctuations_over_c=height_right,
        complete_path_jumps_over_c=complete_jumps,
        shared_conservative_face_defect=(
            max(shared_defects) if shared_defects else 0.0
        ),
        maximum_split_closure_defect=(
            max(split_defects) if split_defects else 0.0
        ),
        incoming_excision_characteristics=incoming_excision,
        outer_boundary_choked=bool(choked),
        outer_incoming_characteristics=int(outer_incoming),
    )

    conservative = candidate_fluxes[1:] - candidate_fluxes[:-1]
    shear = np.zeros_like(conservative)
    height = np.zeros_like(conservative)
    geometry = np.zeros_like(conservative)
    cooling = np.zeros_like(conservative)
    stream = np.zeros_like(conservative)
    lower_height = np.zeros_like(conservative)
    local_stress = np.zeros_like(conservative)
    lower_components = {
        name: np.zeros_like(conservative)
        for name in (
            "perfect_fluid_geometry",
            "stress_geometry",
            "radiative_cooling",
            "vertical_work",
            "stress_relaxation",
            "stream",
        )
    }
    paths = []
    for cell in range(n_cells):
        path = causal_five_field_radial_extended_path_jump(
            context,
            float(context.grid.edges[cell]),
            float(context.grid.edges[cell + 1]),
            right_faces[cell],
            left_faces[cell + 1],
            quadrature_order=quadrature_order,
            relative_step=relative_step,
        )
        paths.append(path)
        shear[cell] = (
            -path.shear_source_path_integral_over_c
            + shear_right[cell]
            + shear_left[cell + 1]
        )
        height[cell] = (
            -path.vertical_source_path_integral_over_c
            + height_right[cell]
            + height_left[cell + 1]
        )
        components = _integrated_lower_sources(
            context,
            cell,
            right_faces[cell],
            left_faces[cell + 1],
        )
        for name, values in components.items():
            lower_components[name][cell] = values
        geometry[cell] = -(
            components["perfect_fluid_geometry"]
            + components["stress_geometry"]
        )
        cooling[cell] = -components["radiative_cooling"]
        stream[cell] = -components["stream"]
        lower_height[cell] = -components["vertical_work"]
        local_stress[cell] = -components["stress_relaxation"]

    blocks = (
        conservative,
        shear,
        height,
        local_stress,
        geometry,
        cooling,
        stream,
        lower_height,
    )
    residual = np.sum(np.asarray(blocks), axis=0)
    reconstructed = (
        conservative
        + shear
        + height
        + local_stress
        + geometry
        + cooling
        + stream
        + lower_height
    )
    scale = max(
        float(np.max(np.abs(residual))),
        max(float(np.max(np.abs(block))) for block in blocks),
        np.finfo(float).tiny,
    )
    ledger_defect = float(
        np.max(np.abs(residual - reconstructed)) / scale
    )
    return CausalFiveFieldRadialCandidateLedger(
        reconstruction=reconstruction,
        interfaces=interface_ledger,
        within_cell_paths=tuple(paths),
        conservative_transport_rows=np.asarray(conservative, dtype=float),
        shear_principal_rows=np.asarray(shear, dtype=float),
        height_principal_rows=np.asarray(height, dtype=float),
        local_stress_relaxation_rows=np.asarray(
            local_stress,
            dtype=float,
        ),
        geometry_rows=np.asarray(geometry, dtype=float),
        cooling_rows=np.asarray(cooling, dtype=float),
        stream_rows=np.asarray(stream, dtype=float),
        lower_height_work_rows=np.asarray(lower_height, dtype=float),
        residual_rows=np.asarray(residual, dtype=float),
        integrated_lower_source_components_per_ct=lower_components,
        local_block_ledger_defect=ledger_defect,
        source_double_count_defect=0.0,
    )
