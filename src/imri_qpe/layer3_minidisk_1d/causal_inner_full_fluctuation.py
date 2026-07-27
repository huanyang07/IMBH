"""Production-neutral complete-principal path and fluctuation contracts.

WP10c9d1 shows that the failed conservative inner-micro exports are not
controlled by one characteristic family.  This module therefore treats the
complete five-field principal jump as one object.  For the implemented smooth
equation

``A(p) p_ct + [F_p(p) - C_pr(p)] p_R = lower-order terms``,

the straight-path jump is

``Delta F - integral C_pr(Psi) Psi_s ds``.

The conservative flux jump remains explicit, while the derivative-source
integrals and signed characteristic fluctuations are reported separately.
Nothing in this module changes the production residual or numerical flux.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .causal_inner_characteristic_dissipation import (
    DEFAULT_COORDINATE_PRINCIPAL_RELATIVE_STEP,
    CausalFiveFieldCoordinatePrincipalBasis,
    causal_five_field_coordinate_principal_basis,
    causal_five_field_coordinate_principal_components,
)
from .causal_inner_dae_system import CausalFiveFieldDAEContext, _cell_state


@dataclass(frozen=True)
class CausalFiveFieldCompletePathJump:
    """Sign-explicit complete-principal jump on one primitive-space path."""

    left_chart: np.ndarray
    right_chart: np.ndarray
    face_measure: float
    conservative_flux_jump_over_c: np.ndarray
    shear_source_path_integral_over_c: np.ndarray
    vertical_source_path_integral_over_c: np.ndarray
    principal_source_path_integral_over_c: np.ndarray
    total_principal_jump_over_c: np.ndarray
    source_partition_defect: float
    principal_closure_defect: float


@dataclass(frozen=True)
class CausalFiveFieldSignedFluctuations:
    """Midpoint characteristic split of one complete-principal path jump."""

    path_jump: CausalFiveFieldCompletePathJump
    midpoint_basis: CausalFiveFieldCoordinatePrincipalBasis
    characteristic_jump_coefficients: np.ndarray
    negative_fluctuation_over_c: np.ndarray
    stationary_fluctuation_over_c: np.ndarray
    positive_fluctuation_over_c: np.ndarray
    split_closure_defect: float
    minimum_speed_gap_over_c: float


@dataclass(frozen=True)
class CausalFiveFieldCellFluctuationLedger:
    """Periodic fixed-geometry interface/within-cell fluctuation ledger."""

    interface_total_jumps_over_c: np.ndarray
    interface_negative_fluctuations_over_c: np.ndarray
    interface_stationary_fluctuations_over_c: np.ndarray
    interface_positive_fluctuations_over_c: np.ndarray
    within_cell_total_jumps_over_c: np.ndarray
    within_cell_conservative_jumps_over_c: np.ndarray
    within_cell_principal_source_integrals_over_c: np.ndarray
    cell_principal_residuals_over_c: np.ndarray
    global_conservative_cycle_defect: float
    global_fluctuation_assembly_defect: float
    maximum_interface_split_defect: float


def _validated_path_inputs(
    context: CausalFiveFieldDAEContext,
    radius: float,
    left_chart: np.ndarray,
    right_chart: np.ndarray,
    quadrature_order: int,
    face_measure: float,
) -> tuple[
    CausalFiveFieldDAEContext,
    float,
    np.ndarray,
    np.ndarray,
    int,
    float,
]:
    context = context.validated()
    radius = float(radius)
    left = np.asarray(left_chart, dtype=float)
    right = np.asarray(right_chart, dtype=float)
    order = int(quadrature_order)
    measure = float(face_measure)
    if (
        not np.isfinite(radius)
        or radius <= 0.0
        or left.shape != (5,)
        or right.shape != (5,)
        or np.any(~np.isfinite(left))
        or np.any(~np.isfinite(right))
        or order < 2
        or not np.isfinite(measure)
        or measure <= 0.0
    ):
        raise ValueError("complete-principal path inputs are invalid")
    return context, radius, left, right, order, measure


def causal_five_field_complete_principal_path_jump(
    context: CausalFiveFieldDAEContext,
    radius: float,
    left_chart: np.ndarray,
    right_chart: np.ndarray,
    *,
    quadrature_order: int = 8,
    relative_step: float = DEFAULT_COORDINATE_PRINCIPAL_RELATIVE_STEP,
    face_measure: float = 1.0,
) -> CausalFiveFieldCompletePathJump:
    """Return ``Delta F - integral C_pr(Psi) Psi_s ds`` and its parts.

    ``C_pr`` is the coefficient of the derivative-dependent source on the
    right-hand side of the implemented residual.  The minus sign is therefore
    part of this public contract.  A straight line in the physical primitive
    chart is used only as an audit path; no finite-amplitude production path is
    selected here.
    """

    (
        context,
        radius,
        left,
        right,
        order,
        measure,
    ) = _validated_path_inputs(
        context,
        radius,
        left_chart,
        right_chart,
        quadrature_order,
        face_measure,
    )
    relative_step = float(relative_step)
    if not np.isfinite(relative_step) or relative_step <= 0.0:
        raise ValueError("relative_step must be finite and positive")
    delta = right - left
    nodes, weights = np.polynomial.legendre.leggauss(order)
    shear = np.zeros(5, dtype=float)
    vertical = np.zeros(5, dtype=float)
    principal = np.zeros(5, dtype=float)
    for node, weight in zip(nodes, weights, strict=True):
        fraction = 0.5 * (float(node) + 1.0)
        chart = left + fraction * delta
        components = causal_five_field_coordinate_principal_components(
            context,
            radius,
            chart,
            relative_step=relative_step,
        )
        quadrature_weight = 0.5 * float(weight)
        shear += (
            quadrature_weight
            * (components.shear_principal_source_matrix @ delta)
        )
        vertical += (
            quadrature_weight
            * (components.vertical_principal_source_matrix @ delta)
        )
        principal += (
            quadrature_weight
            * (components.principal_source_matrix @ delta)
        )

    left_flux = np.asarray(
        _cell_state(context, radius, left).flux_over_c,
        dtype=float,
    )
    right_flux = np.asarray(
        _cell_state(context, radius, right).flux_over_c,
        dtype=float,
    )
    conservative = measure * (right_flux - left_flux)
    shear *= measure
    vertical *= measure
    principal *= measure
    total = conservative - principal
    scale = max(
        float(np.max(np.abs(conservative))),
        float(np.max(np.abs(principal))),
        float(np.max(np.abs(total))),
        np.finfo(float).tiny,
    )
    source_partition_defect = float(
        np.max(np.abs(principal - shear - vertical)) / scale
    )
    principal_closure_defect = float(
        np.max(np.abs(total - conservative + principal)) / scale
    )
    return CausalFiveFieldCompletePathJump(
        left_chart=np.array(left, copy=True),
        right_chart=np.array(right, copy=True),
        face_measure=measure,
        conservative_flux_jump_over_c=np.asarray(conservative, dtype=float),
        shear_source_path_integral_over_c=np.asarray(shear, dtype=float),
        vertical_source_path_integral_over_c=np.asarray(vertical, dtype=float),
        principal_source_path_integral_over_c=np.asarray(
            principal,
            dtype=float,
        ),
        total_principal_jump_over_c=np.asarray(total, dtype=float),
        source_partition_defect=source_partition_defect,
        principal_closure_defect=principal_closure_defect,
    )


def causal_five_field_signed_principal_fluctuations(
    context: CausalFiveFieldDAEContext,
    radius: float,
    left_chart: np.ndarray,
    right_chart: np.ndarray,
    *,
    quadrature_order: int = 8,
    relative_step: float = DEFAULT_COORDINATE_PRINCIPAL_RELATIVE_STEP,
    face_measure: float = 1.0,
    stationary_speed_tolerance: float = 1.0e-12,
) -> CausalFiveFieldSignedFluctuations:
    """Split the complete path jump into negative/zero/positive fluctuations.

    The midpoint complete coordinate pencil supplies a production-neutral
    spectral split.  This is a method contract, not a promoted nonlinear
    Riemann solver: a finite-amplitude equilibrium-preserving path and a
    well-balanced cell assembly remain separate gates.
    """

    path = causal_five_field_complete_principal_path_jump(
        context,
        radius,
        left_chart,
        right_chart,
        quadrature_order=quadrature_order,
        relative_step=relative_step,
        face_measure=face_measure,
    )
    midpoint = 0.5 * (path.left_chart + path.right_chart)
    basis = causal_five_field_coordinate_principal_basis(
        context,
        radius,
        midpoint,
        relative_step=relative_step,
    )
    tolerance = float(stationary_speed_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("stationary_speed_tolerance must be nonnegative")
    scaled_jump = (
        path.total_principal_jump_over_c
        / basis.descriptor_row_scales
    )
    coefficients = basis.descriptor_left_eigenvectors @ scaled_jump
    speeds = np.asarray(basis.numerical_speeds_over_c, dtype=float)
    negative_coefficients = np.where(
        speeds < -tolerance,
        coefficients,
        0.0,
    )
    stationary_coefficients = np.where(
        np.abs(speeds) <= tolerance,
        coefficients,
        0.0,
    )
    positive_coefficients = np.where(
        speeds > tolerance,
        coefficients,
        0.0,
    )

    def reconstruct(values: np.ndarray) -> np.ndarray:
        return basis.descriptor_row_scales * (
            basis.descriptor_right_eigenvectors @ values
        )

    negative = reconstruct(negative_coefficients)
    stationary = reconstruct(stationary_coefficients)
    positive = reconstruct(positive_coefficients)
    reconstructed = negative + stationary + positive
    scale = max(
        float(np.max(np.abs(path.total_principal_jump_over_c))),
        np.finfo(float).tiny,
    )
    split_defect = float(
        np.max(
            np.abs(reconstructed - path.total_principal_jump_over_c)
        )
        / scale
    )
    speed_gaps = np.diff(np.sort(speeds))
    minimum_gap = (
        float(np.min(np.abs(speed_gaps)))
        if speed_gaps.size
        else float("inf")
    )
    return CausalFiveFieldSignedFluctuations(
        path_jump=path,
        midpoint_basis=basis,
        characteristic_jump_coefficients=np.asarray(
            coefficients,
            dtype=float,
        ),
        negative_fluctuation_over_c=np.asarray(negative, dtype=float),
        stationary_fluctuation_over_c=np.asarray(stationary, dtype=float),
        positive_fluctuation_over_c=np.asarray(positive, dtype=float),
        split_closure_defect=split_defect,
        minimum_speed_gap_over_c=minimum_gap,
    )


def causal_five_field_periodic_cell_fluctuation_ledger(
    context: CausalFiveFieldDAEContext,
    radius: float,
    cell_left_charts: np.ndarray,
    cell_right_charts: np.ndarray,
    *,
    quadrature_order: int = 4,
    relative_step: float = DEFAULT_COORDINATE_PRINCIPAL_RELATIVE_STEP,
    path_measure: float = 1.0,
) -> CausalFiveFieldCellFluctuationLedger:
    """Assemble periodic interface and within-cell fluctuations.

    Interface ``i`` is the left face of cell ``i``.  Its left state is the
    right trace of cell ``i-1`` and its right state is the left trace of cell
    ``i``.  Cell ``i`` receives the positive fluctuation from interface ``i``,
    its complete within-cell path jump, and the negative fluctuation from
    interface ``i+1``.

    Geometry and face measure are frozen.  Consequently this function is a
    constant-coefficient/manufactured-wave assembly contract, not yet the
    geometry-aware well-balanced radial production operator.
    """

    context = context.validated()
    left = np.asarray(cell_left_charts, dtype=float)
    right = np.asarray(cell_right_charts, dtype=float)
    if (
        left.ndim != 2
        or left.shape[1:] != (5,)
        or right.shape != left.shape
        or left.shape[0] < 2
        or np.any(~np.isfinite(left))
        or np.any(~np.isfinite(right))
    ):
        raise ValueError("cell trace charts must have shape (N, 5), N >= 2")
    n_cells = left.shape[0]
    interface_total = np.zeros((n_cells, 5), dtype=float)
    interface_negative = np.zeros_like(interface_total)
    interface_stationary = np.zeros_like(interface_total)
    interface_positive = np.zeros_like(interface_total)
    interface_split_defects = np.zeros(n_cells, dtype=float)
    interface_conservative = np.zeros_like(interface_total)
    interface_source = np.zeros_like(interface_total)
    for interface in range(n_cells):
        left_state = right[(interface - 1) % n_cells]
        right_state = left[interface]
        if np.array_equal(left_state, right_state):
            continue
        split = causal_five_field_signed_principal_fluctuations(
            context,
            radius,
            left_state,
            right_state,
            quadrature_order=quadrature_order,
            relative_step=relative_step,
            face_measure=path_measure,
        )
        interface_total[interface] = (
            split.path_jump.total_principal_jump_over_c
        )
        interface_negative[interface] = (
            split.negative_fluctuation_over_c
        )
        interface_stationary[interface] = (
            split.stationary_fluctuation_over_c
        )
        interface_positive[interface] = (
            split.positive_fluctuation_over_c
        )
        interface_conservative[interface] = (
            split.path_jump.conservative_flux_jump_over_c
        )
        interface_source[interface] = (
            split.path_jump.principal_source_path_integral_over_c
        )
        interface_split_defects[interface] = split.split_closure_defect

    within_total = np.zeros((n_cells, 5), dtype=float)
    within_conservative = np.zeros_like(within_total)
    within_source = np.zeros_like(within_total)
    for cell in range(n_cells):
        path = causal_five_field_complete_principal_path_jump(
            context,
            radius,
            left[cell],
            right[cell],
            quadrature_order=quadrature_order,
            relative_step=relative_step,
            face_measure=path_measure,
        )
        within_total[cell] = path.total_principal_jump_over_c
        within_conservative[cell] = (
            path.conservative_flux_jump_over_c
        )
        within_source[cell] = (
            path.principal_source_path_integral_over_c
        )

    residuals = np.zeros_like(within_total)
    for cell in range(n_cells):
        residuals[cell] = (
            interface_positive[cell]
            + interface_stationary[cell] * 0.5
            + within_total[cell]
            + interface_negative[(cell + 1) % n_cells]
            + interface_stationary[(cell + 1) % n_cells] * 0.5
        )
    conservative_cycle = np.sum(
        interface_conservative + within_conservative,
        axis=0,
    )
    conservative_scale = max(
        float(np.max(np.abs(interface_conservative))),
        float(np.max(np.abs(within_conservative))),
        np.finfo(float).tiny,
    )
    conservative_defect = float(
        np.max(np.abs(conservative_cycle))
        / (n_cells * conservative_scale)
    )
    assembled_global = np.sum(residuals, axis=0)
    path_global = np.sum(interface_total + within_total, axis=0)
    assembly_scale = max(
        float(np.max(np.abs(assembled_global))),
        float(np.max(np.abs(path_global))),
        float(np.max(np.abs(residuals))),
        np.finfo(float).tiny,
    )
    assembly_defect = float(
        np.max(np.abs(assembled_global - path_global)) / assembly_scale
    )
    return CausalFiveFieldCellFluctuationLedger(
        interface_total_jumps_over_c=interface_total,
        interface_negative_fluctuations_over_c=interface_negative,
        interface_stationary_fluctuations_over_c=interface_stationary,
        interface_positive_fluctuations_over_c=interface_positive,
        within_cell_total_jumps_over_c=within_total,
        within_cell_conservative_jumps_over_c=within_conservative,
        within_cell_principal_source_integrals_over_c=within_source,
        cell_principal_residuals_over_c=residuals,
        global_conservative_cycle_defect=conservative_defect,
        global_fluctuation_assembly_defect=assembly_defect,
        maximum_interface_split_defect=float(
            np.max(interface_split_defects)
        ),
    )
