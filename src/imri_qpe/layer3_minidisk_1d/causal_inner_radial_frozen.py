"""Frozen-linear discrimination helpers for the radial fluctuation candidate.

The WP10c9d5 comparison is deliberately narrower than a nonlinear candidate.
It keeps the certified temporal descriptor and the production-anchor
storage-rate derivative fixed, replaces only the stationary radial residual,
and asks whether that spatial change improves refinement of the exported
physical observables.

All matrices act on the same fixed scaled primitive coordinates as the
production evolving generator.  This module does not change production
defaults and does not define a finite-amplitude time integrator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse.linalg import splu

from .causal_inner_dae_system import (
    CausalFiveFieldDAEContext,
    causal_five_field_colored_central_jacobian,
    causal_five_field_dae_jacobian_color_groups,
    causal_five_field_dae_scaling,
    causal_five_field_reduced_stationary_residual,
    causal_five_field_state_from_primitives,
    evaluate_causal_five_field_dae,
    pack_causal_five_field_state,
)
from .causal_inner_radial_fluctuation import (
    causal_five_field_radial_candidate_ledger,
)
from .causal_inner_spatial_audit import (
    causal_five_field_reduced_storage_matrices,
)


_N_FIELDS = 5


@dataclass(frozen=True)
class CausalFiveFieldRadialFrozenCandidate:
    """One production/candidate frozen-generator comparison."""

    production_scaled_generator_per_s: np.ndarray
    candidate_scaled_generator_per_s: np.ndarray
    descriptor_reduced_scaled_matrix: np.ndarray
    stationary_delta_scaled_jacobian: np.ndarray
    descriptor_solve_scaled_correction: np.ndarray
    primitive_column_scales: np.ndarray
    conservation_row_scales: np.ndarray
    finite_difference_step: float
    path_quadrature_order: int
    color_count: int
    maximum_descriptor_solve_relative_defect: float
    maximum_mass_off_pattern_relative_entry: float
    same_temporal_descriptor: bool
    same_base_rate_storage_derivative: bool


def causal_five_field_radial_reduced_jacobian_pattern(
    n_cells: int,
    *,
    cell_half_bandwidth: int = 3,
) -> csc_matrix:
    """Return a conservative primitive-to-conservation candidate pattern.

    The production quadratic reconstruction uses at most four cells at a
    boundary face.  A three-cell half-bandwidth therefore safely contains the
    two adjacent face stencils, the within-cell path, and all local sources.
    The slight over-declaration is intentional: it preserves correctness at
    the cost of a few additional color groups.
    """

    cells = int(n_cells)
    half = int(cell_half_bandwidth)
    if cells < 3 or half < 1:
        raise ValueError("radial reduced pattern dimensions are invalid")
    size = _N_FIELDS * cells
    pattern = lil_matrix((size, size), dtype=np.int8)
    for row_cell in range(cells):
        lower = max(0, row_cell - half)
        upper = min(cells, row_cell + half + 1)
        row_slice = slice(
            _N_FIELDS * row_cell,
            _N_FIELDS * (row_cell + 1),
        )
        for column_cell in range(lower, upper):
            column_slice = slice(
                _N_FIELDS * column_cell,
                _N_FIELDS * (column_cell + 1),
            )
            pattern[row_slice, column_slice] = 1
    return pattern.tocsc()


def _candidate_stationary_delta(
    context: CausalFiveFieldDAEContext,
    primitives: np.ndarray,
    *,
    path_quadrature_order: int,
) -> np.ndarray:
    charts = np.asarray(primitives, dtype=float).reshape(-1, _N_FIELDS)
    candidate = causal_five_field_radial_candidate_ledger(
        context,
        charts,
        quadrature_order=path_quadrature_order,
    )
    production = causal_five_field_reduced_stationary_residual(
        charts.ravel(),
        context,
    )
    return (
        np.asarray(candidate.residual_rows, dtype=float).ravel()
        - np.asarray(production, dtype=float).ravel()
    )


def causal_five_field_radial_frozen_candidate(
    context: CausalFiveFieldDAEContext,
    base_primitives: np.ndarray,
    production_scaled_generator_per_s: np.ndarray,
    *,
    primitive_column_scales: np.ndarray | None = None,
    conservation_row_scales: np.ndarray | None = None,
    finite_difference_step: float = 4.0e-5,
    storage_difference_step: float = 2.0e-6,
    storage_quadrature_order: int = 4,
    storage_directional_step: float = 1.0e-3,
    path_quadrature_order: int = 6,
) -> CausalFiveFieldRadialFrozenCandidate:
    """Replace only the stationary radial block of a frozen generator.

    If ``G_prod = -M^{-1}(J_prod + DM[p_dot_prod])``, this helper constructs

    ``G_cand = G_prod - M^{-1}(J_cand - J_prod)``.

    Thus both operators use exactly the same descriptor and frozen
    production-anchor ``DM[p_dot]`` term.  This is the binding WP10c9d5 A/B
    discrimination; it is not a claim about a nonlinear candidate trajectory.
    """

    context = context.validated()
    charts = np.asarray(base_primitives, dtype=float)
    n_cells = int(context.grid.centers.size)
    n_reduced = _N_FIELDS * n_cells
    production = np.asarray(
        production_scaled_generator_per_s,
        dtype=float,
    )
    step = float(finite_difference_step)
    if (
        charts.shape != (n_cells, _N_FIELDS)
        or production.shape != (n_reduced, n_reduced)
        or np.any(~np.isfinite(charts))
        or np.any(~np.isfinite(production))
        or not np.isfinite(step)
        or step <= 0.0
    ):
        raise ValueError("radial frozen-candidate inputs are invalid")

    state = causal_five_field_state_from_primitives(context, charts)
    vector = pack_causal_five_field_state(state)
    evaluation = evaluate_causal_five_field_dae(vector, context)
    scaling = causal_five_field_dae_scaling(state, evaluation)
    calculated_columns = np.asarray(
        scaling.column_scales[n_reduced : 2 * n_reduced],
        dtype=float,
    )
    calculated_rows = np.asarray(
        scaling.row_scales[:n_reduced],
        dtype=float,
    )
    column_scales = (
        calculated_columns
        if primitive_column_scales is None
        else np.asarray(primitive_column_scales, dtype=float).ravel()
    )
    row_scales = (
        calculated_rows
        if conservation_row_scales is None
        else np.asarray(conservation_row_scales, dtype=float).ravel()
    )
    if (
        column_scales.shape != (n_reduced,)
        or row_scales.shape != (n_reduced,)
        or np.any(~np.isfinite(column_scales))
        or np.any(~np.isfinite(row_scales))
        or np.any(column_scales <= 0.0)
        or np.any(row_scales <= 0.0)
        or not np.allclose(
            column_scales,
            calculated_columns,
            rtol=2.0e-14,
            atol=0.0,
        )
        or not np.allclose(
            row_scales,
            calculated_rows,
            rtol=2.0e-14,
            atol=0.0,
        )
    ):
        raise ValueError("radial frozen-candidate scaling changed")

    pattern = causal_five_field_radial_reduced_jacobian_pattern(n_cells)
    zero = np.zeros(n_reduced, dtype=float)

    def scaled_delta(scaled_increment: np.ndarray) -> np.ndarray:
        perturbed = charts.ravel() + column_scales * np.asarray(
            scaled_increment,
            dtype=float,
        )
        return _candidate_stationary_delta(
            context,
            perturbed,
            path_quadrature_order=path_quadrature_order,
        ) / row_scales

    delta = causal_five_field_colored_central_jacobian(
        scaled_delta,
        zero,
        pattern,
        finite_difference_step=step,
    ).toarray()
    storage = causal_five_field_reduced_storage_matrices(
        context,
        charts.ravel(),
        primitive_column_scales=column_scales,
        conservation_row_scales=row_scales,
        finite_difference_step=storage_difference_step,
        storage_quadrature_order=storage_quadrature_order,
        storage_directional_step=storage_directional_step,
    )
    mass = np.asarray(
        storage["descriptor_reduced_scaled_matrix"],
        dtype=float,
    )
    sparse_mass = csc_matrix(mass)
    factor = splu(sparse_mass, permc_spec="COLAMD")
    correction = np.asarray(factor.solve(delta), dtype=float)
    residual = mass @ correction - delta
    solve_scale = max(
        float(np.max(np.abs(delta))),
        np.finfo(float).tiny,
    )

    declared_mass_pattern = causal_five_field_radial_reduced_jacobian_pattern(
        n_cells,
        cell_half_bandwidth=1,
    ).toarray().astype(bool)
    off_pattern = np.array(mass, copy=True)
    off_pattern[declared_mass_pattern] = 0.0
    mass_scale = max(
        float(np.max(np.abs(mass))),
        np.finfo(float).tiny,
    )
    candidate = production - correction
    return CausalFiveFieldRadialFrozenCandidate(
        production_scaled_generator_per_s=np.array(production, copy=True),
        candidate_scaled_generator_per_s=np.asarray(candidate, dtype=float),
        descriptor_reduced_scaled_matrix=mass,
        stationary_delta_scaled_jacobian=delta,
        descriptor_solve_scaled_correction=correction,
        primitive_column_scales=column_scales,
        conservation_row_scales=row_scales,
        finite_difference_step=step,
        path_quadrature_order=int(path_quadrature_order),
        color_count=len(
            causal_five_field_dae_jacobian_color_groups(pattern)
        ),
        maximum_descriptor_solve_relative_defect=float(
            np.max(np.abs(residual)) / solve_scale
        ),
        maximum_mass_off_pattern_relative_entry=float(
            np.max(np.abs(off_pattern)) / mass_scale
        ),
        same_temporal_descriptor=True,
        same_base_rate_storage_derivative=True,
    )
