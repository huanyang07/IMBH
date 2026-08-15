"""Ledger-constrained fixed-Q helpers for the monolithic inner DAE.

This module is production neutral.  It defines the exact exterior-domain
``Q3=(M,J,E)`` endpoint map, the ledger-derived reaction coordinate, and an
augmented BDF residual.  It does not advance a trajectory or change any
production integration default.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from imri_qpe.constants import C

from .causal_inner_bdf import (
    causal_bdf_coefficients,
    causal_bdf_weighted_increment,
)
from .causal_inner_dae_system import CausalFiveFieldDAEContext, _cell_state
from .causal_inner_monolithic_bdf import (
    CausalFiveFieldMonolithicBDFEvaluation,
    CausalFiveFieldMonolithicBDFHistory,
    causal_five_field_monolithic_bdf_history,
    evaluate_causal_five_field_monolithic_bdf,
)
from .causal_inner_monolithic_discrete_tangent import (
    causal_five_field_monolithic_discrete_step_matrix,
)
from .causal_inner_monolithic_dae import (
    _integrated_mapped_storage,
    _spatial_nodes,
)
from .causal_inner_monolithic_tangent import (
    _descriptor_matrices,
    _node_reconstruction_weights,
)
from .causal_inner_radial_linear_tangent import (
    causal_five_field_analytic_local_maps,
)


_N_FIELDS = 5
_CONSERVATIVE_FIELDS = (0, 2, 3)


@dataclass(frozen=True)
class CausalFiveFieldFixedQReaction:
    """One state-local ledger reaction normalized to the exact Q3 map."""

    q3_value: np.ndarray
    q3_physical_derivative: np.ndarray
    q3_scaled_derivative: np.ndarray
    q3_derivative_norms: np.ndarray
    descriptor_scaled_matrix: np.ndarray
    reaction_scaled_rows: np.ndarray
    reaction_lift: np.ndarray
    reaction_physical_rows: np.ndarray
    reaction_physical_ledger: np.ndarray
    raw_reaction_scaled_rows: np.ndarray
    raw_reaction_lift: np.ndarray
    raw_reaction_physical_rows: np.ndarray
    raw_reaction_physical_ledger: np.ndarray
    raw_schur_matrix: np.ndarray
    raw_schur_inverse: np.ndarray
    raw_schur_singular_values: np.ndarray
    raw_schur_numerical_rank: int
    raw_schur_condition_number: float
    maximum_raw_schur_solve_relative_defect: float
    support_cell_indices: np.ndarray
    support_envelope: np.ndarray
    minimum_q3_reconstruction_factor: float
    maximum_q3_reconstruction_factor: float
    maximum_descriptor_reconstruction_defect: float
    maximum_descriptor_partition_defect: float
    maximum_identity_defect: float
    maximum_reaction_ledger_relative_defect: float
    maximum_reaction_support_relative_defect: float


@dataclass(frozen=True)
class CausalFiveFieldFixedQBDFEvaluation:
    """Complete state-dependent augmented BDF residual evaluation."""

    monolithic_evaluation: CausalFiveFieldMonolithicBDFEvaluation
    reaction: CausalFiveFieldFixedQReaction
    multipliers: np.ndarray
    q3_target: np.ndarray
    constraint_row_scales: np.ndarray
    reaction_channel_basis: str
    reaction_channel_transform: np.ndarray
    scaled_monolithic_residual: np.ndarray
    scaled_reaction_residual: np.ndarray
    scaled_constraint_residual: np.ndarray
    augmented_scaled_residual: np.ndarray
    maximum_zero_multiplier_reduction_defect: float
    maximum_constraint_relative_defect: float


@dataclass(frozen=True)
class CausalFiveFieldFixedQReactionJVP:
    """Analytic directional derivatives of the normalized fixed-Q reaction."""

    scaled_state_directions: np.ndarray
    q3_scaled_row_derivatives: np.ndarray
    raw_reaction_scaled_row_derivatives: np.ndarray
    reaction_scaled_row_derivatives: np.ndarray
    reaction_lift_derivatives: np.ndarray
    reaction_physical_ledger_derivatives: np.ndarray
    raw_reaction_physical_ledger_derivatives: np.ndarray
    maximum_identity_directional_defect: float
    maximum_reaction_ledger_directional_relative_defect: float


@dataclass(frozen=True)
class CausalFiveFieldFixedQRawReactionJacobian:
    """Exact sparse state Jacobian of the three raw reaction channels."""

    scaled_jacobian: np.ndarray
    physical_ledger_jacobian: np.ndarray
    maximum_ledger_relative_defect: float


@dataclass(frozen=True)
class CausalFiveFieldFixedQAugmentedStepMatrix:
    """Complete bordered Jacobian for raw or frozen-normalized channels."""

    scaled_matrix: np.ndarray
    monolithic_scaled_matrix: np.ndarray
    reaction_state_scaled_matrix: np.ndarray
    reaction_multiplier_scaled_matrix: np.ndarray
    constraint_scaled_matrix: np.ndarray
    reaction_channel_basis: str
    reaction_channel_transform: np.ndarray
    maximum_block_closure_defect: float
    maximum_reaction_ledger_relative_defect: float


@dataclass(frozen=True)
class CausalFiveFieldFixedQBackwardEulerResult:
    """One exact constrained BDF solve used by the consistency audit."""

    primitive_charts: np.ndarray
    primitive_increment: np.ndarray
    scaled_rate_per_s: np.ndarray
    scaled_interval_rate_per_s: np.ndarray
    multipliers: np.ndarray
    evaluation: CausalFiveFieldFixedQBDFEvaluation
    direct_rate_evaluation: CausalFiveFieldFixedQBDFEvaluation
    order: int
    accepted: bool
    acceptance: CausalFiveFieldFixedQStepAcceptance
    iterations: int
    function_evaluations: int
    maximum_scaled_residual: float
    maximum_linear_residual: float
    maximum_scaled_primitive_change: float
    minimum_path_reconstruction_factor: float
    maximum_path_reconstruction_factor: float
    maximum_direct_rate_increment_parity_defect: float
    maximum_multiplier_weighted_action_ledger_relative_defect: float
    scaled_reaction_rate_action_per_s: np.ndarray
    maximum_scaled_q3_rate_tangency_defect: float
    maximum_h_over_r: float | None
    minimum_scattering_optical_depth: float | None
    exact_jacobian_assemblies: int
    broyden_updates: int
    linear_solves: int
    message: str


@dataclass(frozen=True)
class CausalFiveFieldFixedQStepAcceptance:
    """One authoritative fail-closed fixed-Q step decision."""

    nonlinear_root_passed: bool
    complete_residual_passed: bool
    q3_passed: bool
    incoming_excision_passed: bool
    storage_parity_passed: bool
    reconstruction_passed: bool
    reaction_ledger_passed: bool
    constraint_action_ledger_passed: bool
    primitive_change_passed: bool
    reaction_conditioning_passed: bool
    physical_height_passed: bool
    physical_optical_depth_passed: bool
    accepted: bool
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True)
class CausalFiveFieldFixedQBDFRestart:
    """Lossless constrained BDF1 history used for one BDF2 replay."""

    primitive_charts: np.ndarray
    previous_primitive_charts: np.ndarray
    history: CausalFiveFieldMonolithicBDFHistory
    q3_target: np.ndarray
    constraint_row_scales: np.ndarray
    multiplier_predictor: np.ndarray
    reaction_channel_basis: str
    reaction_channel_transform: np.ndarray
    previous_minimum_path_reconstruction_factor: float
    elapsed_time_seconds: float
    completed_steps: int
    next_order: int
    provenance: dict
    schema_version: int = 1


def _validated_scales(
    state: np.ndarray,
    primitive_column_scales: np.ndarray,
    conservation_row_scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    columns = np.asarray(primitive_column_scales, dtype=float).reshape(
        state.shape
    )
    rows = np.asarray(conservation_row_scales, dtype=float).reshape(state.shape)
    if (
        np.any(~np.isfinite(columns))
        or np.any(~np.isfinite(rows))
        or np.any(columns <= 0.0)
        or np.any(rows <= 0.0)
    ):
        raise ValueError("fixed-Q primitive and conservation scales are invalid")
    return columns, rows


def _q3_physical_selectors(
    n_cells: int,
    exterior_face: int,
    conservation_row_scales: np.ndarray,
) -> np.ndarray:
    """Return physical exterior M/J/E selectors in scaled row coordinates."""

    rows = np.asarray(conservation_row_scales, dtype=float).reshape(
        int(n_cells),
        _N_FIELDS,
    )
    selectors = np.zeros((len(_CONSERVATIVE_FIELDS), rows.size), dtype=float)
    for output, component in enumerate(_CONSERVATIVE_FIELDS):
        for cell in range(int(exterior_face), int(n_cells)):
            selectors[output, _N_FIELDS * cell + component] = (
                C * rows[cell, component]
            )
    return selectors


def _raw_reaction_physical_rows(
    context: CausalFiveFieldDAEContext,
    state: np.ndarray,
    support: np.ndarray,
    envelope: np.ndarray,
) -> np.ndarray:
    """Return the three unnormalized physical reaction channels."""

    physical_raw = np.zeros((state.shape[0], _N_FIELDS, 3), dtype=float)
    for weight, cell in zip(envelope, support, strict=True):
        local = _cell_state(
            context,
            float(context.grid.centers[cell]),
            state[cell],
        )
        rest_mass = float(local.conserved[0])
        specific_j = float(local.conserved[2] / rest_mass)
        specific_e = float(local.conserved[3] / rest_mass)
        omega = float(local.stress.coordinate_angular_velocity)
        physical_raw[cell, 0, 0] = weight
        physical_raw[cell, 2, 0] = weight * specific_j
        physical_raw[cell, 3, 0] = weight * specific_e
        physical_raw[cell, 2, 1] = weight
        physical_raw[cell, 3, 1] = weight * omega
        physical_raw[cell, 3, 2] = weight
    return physical_raw


def _descriptor_directional_derivative(
    context: CausalFiveFieldDAEContext,
    state: np.ndarray,
    physical_direction: np.ndarray,
    primitive_column_scales: np.ndarray,
    conservation_row_scales: np.ndarray,
    node_weights: np.ndarray,
    node_cells: np.ndarray,
    node_radii: np.ndarray,
    node_measures: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate the mapped and height descriptor matrices in one direction."""

    n_cells = int(state.shape[0])
    dimensions = int(state.size)
    columns = np.asarray(primitive_column_scales, dtype=float).reshape(
        state.shape
    )
    rows = np.asarray(conservation_row_scales, dtype=float).reshape(state.shape)
    mapped = np.zeros((dimensions, dimensions), dtype=float)
    height = np.zeros_like(mapped)
    for weights, cell, radius, measure in zip(
        node_weights,
        node_cells,
        node_radii,
        node_measures,
        strict=True,
    ):
        local = causal_five_field_analytic_local_maps(
            context,
            float(radius),
            np.asarray(weights @ state, dtype=float),
        )
        node_direction = np.asarray(weights @ physical_direction, dtype=float)
        mapped_local = np.tensordot(
            local.mapped_conserved_hessian,
            node_direction,
            axes=(2, 0),
        )
        height_local = np.tensordot(
            local.vertical_storage_derivative,
            node_direction,
            axes=(2, 0),
        )
        row_slice = slice(
            _N_FIELDS * int(cell),
            _N_FIELDS * (int(cell) + 1),
        )
        row_factor = float(measure) / C / rows[int(cell)]
        for source_cell in np.flatnonzero(weights):
            coefficient = float(weights[source_cell])
            for component in range(_N_FIELDS):
                column = _N_FIELDS * int(source_cell) + component
                column_factor = coefficient * columns[source_cell, component]
                mapped[row_slice, column] += (
                    row_factor * mapped_local[:, component] * column_factor
                )
                height[row_slice, column] += (
                    row_factor * height_local[:, component] * column_factor
                )
    if mapped.shape != (dimensions, dimensions) or n_cells <= 0:
        raise RuntimeError("fixed-Q descriptor derivative shape is invalid")
    return mapped, height


def _raw_reaction_physical_direction(
    context: CausalFiveFieldDAEContext,
    state: np.ndarray,
    physical_direction: np.ndarray,
    support: np.ndarray,
    envelope: np.ndarray,
) -> np.ndarray:
    """Differentiate the unnormalized physical reaction channels."""

    derivative = np.zeros((state.shape[0], _N_FIELDS, 3), dtype=float)
    for weight, cell in zip(envelope, support, strict=True):
        local = causal_five_field_analytic_local_maps(
            context,
            float(context.grid.centers[cell]),
            state[cell],
        )
        direction = physical_direction[cell]
        conserved = np.asarray(local.mapped_conserved, dtype=float)
        conserved_direction = (
            np.asarray(local.mapped_conserved_jacobian, dtype=float) @ direction
        )
        rest_mass = float(conserved[0])
        mass_direction = float(conserved_direction[0])
        specific_j_direction = (
            float(conserved_direction[2]) * rest_mass
            - float(conserved[2]) * mass_direction
        ) / rest_mass**2
        specific_e_direction = (
            float(conserved_direction[3]) * rest_mass
            - float(conserved[3]) * mass_direction
        ) / rest_mass**2
        omega_direction = float(
            np.asarray(
                local.coordinate_angular_velocity_jacobian,
                dtype=float,
            )
            @ direction
        )
        derivative[cell, 2, 0] = weight * specific_j_direction
        derivative[cell, 3, 0] = weight * specific_e_direction
        derivative[cell, 3, 1] = weight * omega_direction
    return derivative


def causal_five_field_exterior_q3(
    context: CausalFiveFieldDAEContext,
    primitive_charts: np.ndarray,
    *,
    exterior_face_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact reconstructed exterior-domain mapped M/J/E state."""

    context = context.validated()
    state = np.asarray(primitive_charts, dtype=float)
    n_cells = int(context.grid.centers.size)
    if state.shape != (n_cells, _N_FIELDS) or np.any(~np.isfinite(state)):
        raise ValueError("fixed-Q primitive state is invalid")
    face = int(exterior_face_index)
    if face != exterior_face_index or not 0 <= face < n_cells:
        raise ValueError("fixed-Q exterior face index is invalid")
    mapped, factors, _nodes = _integrated_mapped_storage(
        context,
        state,
        _spatial_nodes(context),
    )
    q3 = np.sum(mapped[face:, np.asarray(_CONSERVATIVE_FIELDS)], axis=0)
    return np.asarray(q3, dtype=float), np.asarray(factors, dtype=float)


def _stable_fixed_q_schur_inverse(
    raw_schur_matrix: np.ndarray,
    *,
    maximum_condition_number: float,
) -> tuple[np.ndarray, np.ndarray, int, float, float]:
    """Invert one three-channel Schur map with explicit rank diagnostics."""

    matrix = np.asarray(raw_schur_matrix, dtype=float)
    maximum_condition = float(maximum_condition_number)
    if (
        matrix.shape != (3, 3)
        or np.any(~np.isfinite(matrix))
        or not np.isfinite(maximum_condition)
        or maximum_condition <= 1.0
    ):
        raise ValueError("fixed-Q Schur matrix or condition gate is invalid")
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    threshold = (
        np.finfo(float).eps
        * max(matrix.shape)
        * float(singular_values[0])
    )
    rank = int(np.count_nonzero(singular_values > threshold))
    condition = float(
        np.inf
        if singular_values[-1] <= 0.0
        else singular_values[0] / singular_values[-1]
    )
    if rank != 3:
        raise np.linalg.LinAlgError("fixed-Q reaction Schur matrix is rank deficient")
    if not np.isfinite(condition) or condition > maximum_condition:
        raise np.linalg.LinAlgError(
            "fixed-Q reaction Schur matrix exceeds its condition gate"
        )
    identity = np.eye(3)
    row_scale = np.max(np.abs(matrix), axis=1)
    if np.any(~np.isfinite(row_scale)) or np.any(row_scale <= 0.0):
        raise np.linalg.LinAlgError(
            "fixed-Q reaction Schur row equilibration is singular"
        )
    row_scaled = matrix / row_scale[:, None]
    column_scale = np.max(np.abs(row_scaled), axis=0)
    if np.any(~np.isfinite(column_scale)) or np.any(column_scale <= 0.0):
        raise np.linalg.LinAlgError(
            "fixed-Q reaction Schur column equilibration is singular"
        )
    equilibrated = row_scaled / column_scale[None, :]
    inverse = (
        np.diag(1.0 / column_scale)
        @ np.linalg.solve(equilibrated, np.diag(1.0 / row_scale))
    )

    # One deterministic residual-refinement correction is the method selected
    # prospectively by WP10c9d6c7c3b5c4f24e7 at both audited physical states.
    # Accumulate the physical-matrix residual in extended precision, then solve
    # its correction through a globally scaled double-precision system.
    global_scale = float(np.max(np.abs(matrix)))
    normalized = matrix / global_scale
    extended_residual = (
        identity.astype(np.longdouble)
        - matrix.astype(np.longdouble) @ inverse.astype(np.longdouble)
    )
    correction = np.linalg.solve(
        normalized,
        np.asarray(extended_residual, dtype=float),
    ) / global_scale
    inverse = inverse + correction

    defect_scale = max(float(np.linalg.norm(identity)), np.finfo(float).tiny)
    solve_defect = float(
        np.linalg.norm(matrix @ inverse - identity) / defect_scale
    )
    return (
        np.asarray(inverse, dtype=float),
        np.asarray(singular_values, dtype=float),
        rank,
        condition,
        solve_defect,
    )


def causal_five_field_fixed_q_reaction(
    context: CausalFiveFieldDAEContext,
    primitive_charts: np.ndarray,
    *,
    primitive_column_scales: np.ndarray,
    conservation_row_scales: np.ndarray,
    parent_cell_indices: np.ndarray,
    refinement_ratio: int,
    exterior_parent_face: int = 36,
    guard_end_parent_face: int = 48,
    parent_cell_count: int = 64,
    maximum_schur_condition_number: float = 1.0e8,
) -> CausalFiveFieldFixedQReaction:
    """Build the physical ledger reaction and normalize ``DQ M^-1 BQ=I``."""

    context = context.validated()
    state = np.asarray(primitive_charts, dtype=float)
    n_cells = int(context.grid.centers.size)
    if state.shape != (n_cells, _N_FIELDS) or np.any(~np.isfinite(state)):
        raise ValueError("fixed-Q reaction state is invalid")
    columns, rows = _validated_scales(
        state,
        primitive_column_scales,
        conservation_row_scales,
    )
    parents = np.asarray(parent_cell_indices, dtype=int)
    ratio = int(refinement_ratio)
    if (
        parents.shape != (n_cells,)
        or ratio != refinement_ratio
        or ratio <= 0
        or np.any(parents < 0)
    ):
        raise ValueError("fixed-Q parent layout is invalid")

    (
        node_weights,
        node_cells,
        node_radii,
        node_measures,
        reconstruction_defect,
        partition_defect,
    ) = _node_reconstruction_weights(context, state)
    mapped, height = _descriptor_matrices(
        context,
        state,
        columns,
        rows,
        node_weights,
        node_cells,
        node_radii,
        node_measures,
    )
    descriptor = np.asarray(mapped + height, dtype=float)

    exterior_face = int(exterior_parent_face) * ratio
    q_selectors = _q3_physical_selectors(n_cells, exterior_face, rows)
    q_physical = np.asarray(q_selectors @ mapped, dtype=float)
    q_norms = np.linalg.norm(q_physical, axis=1)
    if np.any(q_norms <= np.finfo(float).tiny):
        raise ValueError("fixed-Q derivative is rank deficient")
    q_scaled = q_physical / q_norms[:, None]

    support = np.flatnonzero(
        (parents >= int(guard_end_parent_face))
        & (parents < int(parent_cell_count))
    )
    if support.size == 0:
        raise ValueError("fixed-Q reaction support is empty")
    phase = (
        parents[support] - int(guard_end_parent_face) + 0.5
    ) / (int(parent_cell_count) - int(guard_end_parent_face))
    envelope = np.sin(np.pi * phase) ** 2
    envelope *= np.asarray(context.grid.cell_measures, dtype=float)[support]
    envelope /= np.sum(envelope)

    physical_raw = _raw_reaction_physical_rows(
        context,
        state,
        support,
        envelope,
    )

    raw = (physical_raw / C / rows[:, :, None]).reshape(state.size, 3)
    factor = splu(csc_matrix(descriptor))
    raw_lift = factor.solve(raw)
    raw_schur = q_scaled @ raw_lift
    (
        raw_schur_inverse,
        raw_schur_singular_values,
        raw_schur_numerical_rank,
        raw_schur_condition_number,
        raw_schur_solve_defect,
    ) = _stable_fixed_q_schur_inverse(
        raw_schur,
        maximum_condition_number=maximum_schur_condition_number,
    )
    reaction = raw @ raw_schur_inverse
    reaction_lift = factor.solve(reaction)
    identity = q_scaled @ reaction_lift

    physical = (
        reaction.reshape(n_cells, _N_FIELDS, 3)
        * C
        * rows[:, :, None]
    )
    ledger = np.sum(physical, axis=0)[np.asarray(_CONSERVATIVE_FIELDS)]
    raw_ledger = np.sum(physical_raw, axis=0)[
        np.asarray(_CONSERVATIVE_FIELDS)
    ]
    channel_ledger = raw_ledger @ raw_schur_inverse
    ledger_scale = max(
        float(np.linalg.norm(ledger)),
        float(np.linalg.norm(channel_ledger)),
        np.finfo(float).tiny,
    )
    outside = np.ones(n_cells, dtype=bool)
    outside[support] = False
    support_scale = max(
        float(np.max(np.abs(physical))),
        np.finfo(float).tiny,
    )
    q3, factors = causal_five_field_exterior_q3(
        context,
        state,
        exterior_face_index=exterior_face,
    )
    minimum_q3_factor = float(np.min(factors))
    maximum_q3_factor = float(np.max(factors))
    if maximum_q3_factor > 1.0 + 1.0e-12:
        raise ValueError("fixed-Q Q3 reconstruction factor exceeds one")
    return CausalFiveFieldFixedQReaction(
        q3_value=q3,
        q3_physical_derivative=q_physical,
        q3_scaled_derivative=q_scaled,
        q3_derivative_norms=q_norms,
        descriptor_scaled_matrix=descriptor,
        reaction_scaled_rows=np.asarray(reaction, dtype=float),
        reaction_lift=np.asarray(reaction_lift, dtype=float),
        reaction_physical_rows=np.asarray(physical, dtype=float),
        reaction_physical_ledger=np.asarray(ledger, dtype=float),
        raw_reaction_scaled_rows=np.asarray(raw, dtype=float),
        raw_reaction_lift=np.asarray(raw_lift, dtype=float),
        raw_reaction_physical_rows=np.asarray(physical_raw, dtype=float),
        raw_reaction_physical_ledger=np.asarray(raw_ledger, dtype=float),
        raw_schur_matrix=np.asarray(raw_schur, dtype=float),
        raw_schur_inverse=np.asarray(raw_schur_inverse, dtype=float),
        raw_schur_singular_values=raw_schur_singular_values,
        raw_schur_numerical_rank=raw_schur_numerical_rank,
        raw_schur_condition_number=raw_schur_condition_number,
        maximum_raw_schur_solve_relative_defect=raw_schur_solve_defect,
        support_cell_indices=np.asarray(support, dtype=int),
        support_envelope=np.asarray(envelope, dtype=float),
        minimum_q3_reconstruction_factor=minimum_q3_factor,
        maximum_q3_reconstruction_factor=maximum_q3_factor,
        maximum_descriptor_reconstruction_defect=float(reconstruction_defect),
        maximum_descriptor_partition_defect=float(partition_defect),
        maximum_identity_defect=float(np.linalg.norm(identity - np.eye(3))),
        maximum_reaction_ledger_relative_defect=float(
            np.linalg.norm(ledger - channel_ledger) / ledger_scale
        ),
        maximum_reaction_support_relative_defect=float(
            np.max(np.abs(physical[outside])) / support_scale
        ),
    )


def causal_five_field_fixed_q_reaction_jvp(
    context: CausalFiveFieldDAEContext,
    primitive_charts: np.ndarray,
    scaled_state_directions: np.ndarray,
    *,
    primitive_column_scales: np.ndarray,
    conservation_row_scales: np.ndarray,
    parent_cell_indices: np.ndarray,
    refinement_ratio: int,
    exterior_parent_face: int = 36,
    guard_end_parent_face: int = 48,
    parent_cell_count: int = 64,
    reaction: CausalFiveFieldFixedQReaction | None = None,
) -> CausalFiveFieldFixedQReactionJVP:
    """Apply the exact smooth-state derivative of the normalized reaction.

    Reconstruction weights and admissibility branches are frozen at the base
    state, consistently with the analytic monolithic tangent.  The supplied
    directions use the repository's scaled primitive coordinates.
    """

    context = context.validated()
    state = np.asarray(primitive_charts, dtype=float)
    n_cells = int(context.grid.centers.size)
    if state.shape != (n_cells, _N_FIELDS) or np.any(~np.isfinite(state)):
        raise ValueError("fixed-Q reaction-JVP state is invalid")
    columns, rows = _validated_scales(
        state,
        primitive_column_scales,
        conservation_row_scales,
    )
    directions = np.asarray(scaled_state_directions, dtype=float)
    if directions.ndim == 1:
        directions = directions[None, :]
    elif directions.ndim == 3:
        directions = directions.reshape(directions.shape[0], -1)
    if (
        directions.ndim != 2
        or directions.shape[1] != state.size
        or np.any(~np.isfinite(directions))
    ):
        raise ValueError("fixed-Q reaction-JVP directions are invalid")
    base = reaction
    if base is None:
        base = causal_five_field_fixed_q_reaction(
            context,
            state,
            primitive_column_scales=columns,
            conservation_row_scales=rows,
            parent_cell_indices=parent_cell_indices,
            refinement_ratio=refinement_ratio,
            exterior_parent_face=exterior_parent_face,
            guard_end_parent_face=guard_end_parent_face,
            parent_cell_count=parent_cell_count,
        )
    if (
        base.reaction_scaled_rows.shape != (state.size, 3)
        or base.descriptor_scaled_matrix.shape != (state.size, state.size)
    ):
        raise ValueError("fixed-Q reaction-JVP base reaction is invalid")

    (
        node_weights,
        node_cells,
        node_radii,
        node_measures,
        _reconstruction_defect,
        _partition_defect,
    ) = _node_reconstruction_weights(context, state)
    factor = splu(csc_matrix(base.descriptor_scaled_matrix))
    exterior_face = int(exterior_parent_face) * int(refinement_ratio)
    q_selectors = _q3_physical_selectors(n_cells, exterior_face, rows)
    raw_physical = (
        base.raw_reaction_scaled_rows.reshape(n_cells, _N_FIELDS, 3)
        * C
        * rows[:, :, None]
    )
    raw_ledger = np.sum(raw_physical, axis=0)[
        np.asarray(_CONSERVATIVE_FIELDS)
    ]

    q_row_derivatives = []
    raw_reaction_derivatives = []
    reaction_derivatives = []
    lift_derivatives = []
    ledger_derivatives = []
    raw_ledger_derivatives = []
    identity_defects = []
    ledger_defects = []
    for scaled_direction in directions:
        physical_direction = (
            columns.ravel() * scaled_direction
        ).reshape(state.shape)
        mapped_direction, height_direction = (
            _descriptor_directional_derivative(
                context,
                state,
                physical_direction,
                columns,
                rows,
                node_weights,
                node_cells,
                node_radii,
                node_measures,
            )
        )
        descriptor_direction = mapped_direction + height_direction
        q_physical_direction = q_selectors @ mapped_direction
        q_norm_direction = np.sum(
            base.q3_physical_derivative * q_physical_direction,
            axis=1,
        ) / base.q3_derivative_norms
        q_scaled_direction = (
            q_physical_direction / base.q3_derivative_norms[:, None]
            - base.q3_physical_derivative
            * q_norm_direction[:, None]
            / base.q3_derivative_norms[:, None] ** 2
        )
        raw_physical_direction = _raw_reaction_physical_direction(
            context,
            state,
            physical_direction,
            base.support_cell_indices,
            base.support_envelope,
        )
        raw_direction = (
            raw_physical_direction / C / rows[:, :, None]
        ).reshape(state.size, 3)
        raw_lift_direction = factor.solve(
            raw_direction
            - descriptor_direction @ base.raw_reaction_lift
        )
        schur_direction = (
            q_scaled_direction @ base.raw_reaction_lift
            + base.q3_scaled_derivative @ raw_lift_direction
        )
        reaction_direction = (
            raw_direction @ base.raw_schur_inverse
            - base.reaction_scaled_rows
            @ schur_direction
            @ base.raw_schur_inverse
        )
        lift_direction = factor.solve(
            reaction_direction
            - descriptor_direction @ base.reaction_lift
        )
        identity_direction = (
            q_scaled_direction @ base.reaction_lift
            + base.q3_scaled_derivative @ lift_direction
        )
        reaction_physical_direction = (
            reaction_direction.reshape(n_cells, _N_FIELDS, 3)
            * C
            * rows[:, :, None]
        )
        ledger_direction = np.sum(reaction_physical_direction, axis=0)[
            np.asarray(_CONSERVATIVE_FIELDS)
        ]
        raw_ledger_direction = np.sum(raw_physical_direction, axis=0)[
            np.asarray(_CONSERVATIVE_FIELDS)
        ]
        channel_ledger_direction = (
            raw_ledger_direction @ base.raw_schur_inverse
            - raw_ledger
            @ base.raw_schur_inverse
            @ schur_direction
            @ base.raw_schur_inverse
        )
        ledger_scale = max(
            float(np.linalg.norm(ledger_direction)),
            float(np.linalg.norm(channel_ledger_direction)),
            1.0,
        )
        q_row_derivatives.append(q_scaled_direction)
        raw_reaction_derivatives.append(raw_direction)
        reaction_derivatives.append(reaction_direction)
        lift_derivatives.append(lift_direction)
        ledger_derivatives.append(ledger_direction)
        raw_ledger_derivatives.append(raw_ledger_direction)
        identity_defects.append(float(np.linalg.norm(identity_direction)))
        ledger_defects.append(
            float(
                np.linalg.norm(
                    ledger_direction - channel_ledger_direction
                )
                / ledger_scale
            )
        )
    return CausalFiveFieldFixedQReactionJVP(
        scaled_state_directions=np.asarray(directions, dtype=float),
        q3_scaled_row_derivatives=np.asarray(q_row_derivatives, dtype=float),
        raw_reaction_scaled_row_derivatives=np.asarray(
            raw_reaction_derivatives,
            dtype=float,
        ),
        reaction_scaled_row_derivatives=np.asarray(
            reaction_derivatives,
            dtype=float,
        ),
        reaction_lift_derivatives=np.asarray(lift_derivatives, dtype=float),
        reaction_physical_ledger_derivatives=np.asarray(
            ledger_derivatives,
            dtype=float,
        ),
        raw_reaction_physical_ledger_derivatives=np.asarray(
            raw_ledger_derivatives,
            dtype=float,
        ),
        maximum_identity_directional_defect=max(identity_defects, default=0.0),
        maximum_reaction_ledger_directional_relative_defect=max(
            ledger_defects,
            default=0.0,
        ),
    )


def causal_five_field_fixed_q_raw_reaction_jacobian(
    context: CausalFiveFieldDAEContext,
    primitive_charts: np.ndarray,
    *,
    primitive_column_scales: np.ndarray,
    conservation_row_scales: np.ndarray,
    parent_cell_indices: np.ndarray,
    refinement_ratio: int,
    exterior_parent_face: int = 36,
    guard_end_parent_face: int = 48,
    parent_cell_count: int = 64,
    reaction: CausalFiveFieldFixedQReaction | None = None,
) -> CausalFiveFieldFixedQRawReactionJacobian:
    """Assemble the local raw-channel derivative in scaled coordinates."""

    context = context.validated()
    state = np.asarray(primitive_charts, dtype=float)
    n_cells = int(context.grid.centers.size)
    if state.shape != (n_cells, _N_FIELDS) or np.any(~np.isfinite(state)):
        raise ValueError("fixed-Q raw reaction-Jacobian state is invalid")
    columns, rows = _validated_scales(
        state,
        primitive_column_scales,
        conservation_row_scales,
    )
    base = reaction
    if base is None:
        base = causal_five_field_fixed_q_reaction(
            context,
            state,
            primitive_column_scales=columns,
            conservation_row_scales=rows,
            parent_cell_indices=parent_cell_indices,
            refinement_ratio=refinement_ratio,
            exterior_parent_face=exterior_parent_face,
            guard_end_parent_face=guard_end_parent_face,
            parent_cell_count=parent_cell_count,
        )
    dimensions = int(state.size)
    physical = np.zeros((dimensions, 3, dimensions), dtype=float)
    for weight, cell in zip(
        base.support_envelope,
        base.support_cell_indices,
        strict=True,
    ):
        local = causal_five_field_analytic_local_maps(
            context,
            float(context.grid.centers[cell]),
            state[cell],
        )
        conserved = np.asarray(local.mapped_conserved, dtype=float)
        jacobian = np.asarray(local.mapped_conserved_jacobian, dtype=float)
        rest_mass = float(conserved[0])
        for component in range(_N_FIELDS):
            column = _N_FIELDS * int(cell) + component
            conserved_direction = (
                jacobian[:, component] * columns[cell, component]
            )
            mass_direction = float(conserved_direction[0])
            specific_j_direction = (
                float(conserved_direction[2]) * rest_mass
                - float(conserved[2]) * mass_direction
            ) / rest_mass**2
            specific_e_direction = (
                float(conserved_direction[3]) * rest_mass
                - float(conserved[3]) * mass_direction
            ) / rest_mass**2
            omega_direction = float(
                local.coordinate_angular_velocity_jacobian[component]
                * columns[cell, component]
            )
            physical[_N_FIELDS * int(cell) + 2, 0, column] = (
                weight * specific_j_direction
            )
            physical[_N_FIELDS * int(cell) + 3, 0, column] = (
                weight * specific_e_direction
            )
            physical[_N_FIELDS * int(cell) + 3, 1, column] = (
                weight * omega_direction
            )
    scaled = np.array(physical, copy=True)
    for row in range(dimensions):
        cell, component = divmod(row, _N_FIELDS)
        scaled[row] /= C * rows[cell, component]
    physical_ledger = np.zeros((3, 3, dimensions), dtype=float)
    for output, component in enumerate(_CONSERVATIVE_FIELDS):
        physical_ledger[output] = np.sum(
            physical[component:: _N_FIELDS],
            axis=0,
        )
    reconstructed = np.zeros_like(physical_ledger)
    for output, component in enumerate(_CONSERVATIVE_FIELDS):
        reconstructed[output] = np.sum(
            scaled[component:: _N_FIELDS]
            * C
            * rows[:, component, None, None],
            axis=0,
        )
    scale = max(
        float(np.linalg.norm(physical_ledger)),
        float(np.linalg.norm(reconstructed)),
        1.0,
    )
    return CausalFiveFieldFixedQRawReactionJacobian(
        scaled_jacobian=np.asarray(scaled, dtype=float),
        physical_ledger_jacobian=np.asarray(physical_ledger, dtype=float),
        maximum_ledger_relative_defect=float(
            np.linalg.norm(physical_ledger - reconstructed) / scale
        ),
    )


def causal_five_field_fixed_q_augmented_step_matrix(
    context: CausalFiveFieldDAEContext,
    old_primitive_charts: np.ndarray,
    new_primitive_charts: np.ndarray,
    multipliers: np.ndarray,
    timestep_seconds: float,
    previous_timestep_seconds: float | None,
    *,
    order: int,
    primitive_column_scales: np.ndarray,
    conservation_row_scales: np.ndarray,
    parent_cell_indices: np.ndarray,
    refinement_ratio: int,
    constraint_row_scales: np.ndarray,
    reaction_channel_basis: str = "frozen_normalized",
    reaction_channel_transform: np.ndarray | None = None,
    exterior_parent_face: int = 36,
    guard_end_parent_face: int = 48,
    parent_cell_count: int = 64,
    reaction: CausalFiveFieldFixedQReaction | None = None,
) -> CausalFiveFieldFixedQAugmentedStepMatrix:
    """Assemble the exact complete Jacobian of one augmented BDF endpoint.

    The raw-channel formulation and a normalization frozen at the beginning
    of the step are supported.  They retain the same constrained state root
    while avoiding the ill-conditioned derivative of the state-local Schur
    normalization.  The fully state-normalized channel remains available as
    an audit map through :func:`causal_five_field_fixed_q_reaction_jvp`, but
    is deliberately not used as a nonlinear residual kernel here.
    """

    context = context.validated()
    old = np.asarray(old_primitive_charts, dtype=float)
    new = np.asarray(new_primitive_charts, dtype=float)
    columns, rows = _validated_scales(
        new,
        primitive_column_scales,
        conservation_row_scales,
    )
    if old.shape != new.shape or np.any(~np.isfinite(old)):
        raise ValueError("fixed-Q augmented step states are invalid")
    multiplier = np.asarray(multipliers, dtype=float)
    scales = np.asarray(constraint_row_scales, dtype=float)
    if (
        multiplier.shape != (3,)
        or scales.shape != (3,)
        or np.any(~np.isfinite(multiplier))
        or np.any(~np.isfinite(scales))
        or np.any(scales <= 0.0)
    ):
        raise ValueError("fixed-Q augmented step scaling is invalid")
    base = reaction
    if base is None:
        base = causal_five_field_fixed_q_reaction(
            context,
            new,
            primitive_column_scales=columns,
            conservation_row_scales=rows,
            parent_cell_indices=parent_cell_indices,
            refinement_ratio=refinement_ratio,
            exterior_parent_face=exterior_parent_face,
            guard_end_parent_face=guard_end_parent_face,
            parent_cell_count=parent_cell_count,
        )
    basis = str(reaction_channel_basis)
    if basis == "raw":
        transform = np.eye(3)
    elif basis == "frozen_normalized":
        transform = np.asarray(reaction_channel_transform, dtype=float)
        if (
            transform.shape != (3, 3)
            or np.any(~np.isfinite(transform))
        ):
            raise ValueError("fixed-Q frozen reaction transform is invalid")
    else:
        raise ValueError(
            "complete augmented matrix requires raw or frozen-normalized "
            "reaction channels"
        )
    monolithic = causal_five_field_monolithic_discrete_step_matrix(
        context,
        old,
        new,
        timestep_seconds,
        previous_timestep_seconds,
        primitive_column_scales=columns,
        conservation_row_scales=rows,
        order=order,
    )
    raw_jacobian = causal_five_field_fixed_q_raw_reaction_jacobian(
        context,
        new,
        primitive_column_scales=columns,
        conservation_row_scales=rows,
        parent_cell_indices=parent_cell_indices,
        refinement_ratio=refinement_ratio,
        exterior_parent_face=exterior_parent_face,
        guard_end_parent_face=guard_end_parent_face,
        parent_cell_count=parent_cell_count,
        reaction=base,
    )
    reaction_state = np.tensordot(
        raw_jacobian.scaled_jacobian,
        transform @ multiplier,
        axes=(1, 0),
    )
    reaction_multiplier = base.raw_reaction_scaled_rows @ transform
    constraint = base.q3_physical_derivative / scales[:, None]
    top_left = monolithic.scaled_matrix - reaction_state
    augmented = np.block(
        [
            [top_left, -reaction_multiplier],
            [constraint, np.zeros((3, 3))],
        ]
    )
    reconstructed = np.block(
        [
            [
                monolithic.scaled_matrix - reaction_state,
                -reaction_multiplier,
            ],
            [constraint, np.zeros((3, 3))],
        ]
    )
    scale = max(float(np.linalg.norm(augmented)), np.finfo(float).tiny)
    return CausalFiveFieldFixedQAugmentedStepMatrix(
        scaled_matrix=np.asarray(augmented, dtype=float),
        monolithic_scaled_matrix=np.asarray(
            monolithic.scaled_matrix,
            dtype=float,
        ),
        reaction_state_scaled_matrix=np.asarray(reaction_state, dtype=float),
        reaction_multiplier_scaled_matrix=np.asarray(
            reaction_multiplier,
            dtype=float,
        ),
        constraint_scaled_matrix=np.asarray(constraint, dtype=float),
        reaction_channel_basis=basis,
        reaction_channel_transform=np.asarray(transform, dtype=float),
        maximum_block_closure_defect=float(
            np.linalg.norm(augmented - reconstructed) / scale
        ),
        maximum_reaction_ledger_relative_defect=float(
            raw_jacobian.maximum_ledger_relative_defect
        ),
    )


def evaluate_causal_five_field_fixed_q_bdf(
    old_primitive_charts: np.ndarray,
    new_primitive_charts: np.ndarray,
    multipliers: np.ndarray,
    q3_target: np.ndarray,
    timestep_seconds: float,
    context: CausalFiveFieldDAEContext,
    *,
    order: int,
    primitive_column_scales: np.ndarray,
    conservation_row_scales: np.ndarray,
    parent_cell_indices: np.ndarray,
    refinement_ratio: int,
    history: CausalFiveFieldMonolithicBDFHistory | None = None,
    exterior_parent_face: int = 36,
    guard_end_parent_face: int = 48,
    parent_cell_count: int = 64,
    constraint_row_scales: np.ndarray | None = None,
    reaction_channel_basis: str = "normalized",
    reaction_channel_transform: np.ndarray | None = None,
    scaled_primitive_increment: np.ndarray | None = None,
    scaled_rate_per_s: np.ndarray | None = None,
    maximum_schur_condition_number: float = 1.0e8,
) -> CausalFiveFieldFixedQBDFEvaluation:
    """Evaluate the complete augmented BDF residual at one endpoint."""

    new = np.asarray(new_primitive_charts, dtype=float)
    _columns, rows = _validated_scales(
        new,
        primitive_column_scales,
        conservation_row_scales,
    )
    multiplier = np.asarray(multipliers, dtype=float)
    target = np.asarray(q3_target, dtype=float)
    if (
        multiplier.shape != (3,)
        or target.shape != (3,)
        or np.any(~np.isfinite(multiplier))
        or np.any(~np.isfinite(target))
    ):
        raise ValueError("fixed-Q multiplier or target is invalid")
    monolithic = evaluate_causal_five_field_monolithic_bdf(
        old_primitive_charts,
        new,
        timestep_seconds,
        context,
        order=order,
        history=history,
        current_primitive_increment=(
            None
            if scaled_primitive_increment is None
            else _columns
            * np.asarray(
                scaled_primitive_increment,
                dtype=float,
            ).reshape(new.shape)
        ),
        current_primitive_rate_per_s=(
            None
            if scaled_rate_per_s is None
            else _columns
            * np.asarray(scaled_rate_per_s, dtype=float).reshape(new.shape)
        ),
    )
    reaction = causal_five_field_fixed_q_reaction(
        context,
        new,
        primitive_column_scales=primitive_column_scales,
        conservation_row_scales=rows,
        parent_cell_indices=parent_cell_indices,
        refinement_ratio=refinement_ratio,
        exterior_parent_face=exterior_parent_face,
        guard_end_parent_face=guard_end_parent_face,
        parent_cell_count=parent_cell_count,
        maximum_schur_condition_number=maximum_schur_condition_number,
    )
    constraint_scales = (
        reaction.q3_derivative_norms
        if constraint_row_scales is None
        else np.asarray(constraint_row_scales, dtype=float)
    )
    if (
        constraint_scales.shape != (3,)
        or np.any(~np.isfinite(constraint_scales))
        or np.any(constraint_scales <= 0.0)
    ):
        raise ValueError("fixed-Q constraint row scales are invalid")
    monolithic_scaled = monolithic.residual_rows.ravel() / rows.ravel()
    basis = str(reaction_channel_basis)
    if basis == "normalized":
        reaction_rows = reaction.reaction_scaled_rows
        channel_transform = reaction.raw_schur_inverse
    elif basis == "raw":
        reaction_rows = reaction.raw_reaction_scaled_rows
        channel_transform = np.eye(3)
    elif basis == "frozen_normalized":
        channel_transform = np.asarray(
            reaction_channel_transform,
            dtype=float,
        )
        if (
            channel_transform.shape != (3, 3)
            or np.any(~np.isfinite(channel_transform))
        ):
            raise ValueError("fixed-Q frozen reaction transform is invalid")
        reaction_rows = reaction.raw_reaction_scaled_rows @ channel_transform
    else:
        raise ValueError("fixed-Q reaction channel basis is invalid")
    reaction_scaled = reaction_rows @ multiplier
    constraint = (reaction.q3_value - target) / constraint_scales
    top = monolithic_scaled - reaction_scaled
    augmented = np.concatenate((top, constraint))
    zero_scale = max(
        float(np.linalg.norm(monolithic_scaled)),
        np.finfo(float).tiny,
    )
    q_scale = np.maximum(
        np.maximum(np.abs(reaction.q3_value), np.abs(target)),
        np.finfo(float).tiny,
    )
    return CausalFiveFieldFixedQBDFEvaluation(
        monolithic_evaluation=monolithic,
        reaction=reaction,
        multipliers=np.array(multiplier, copy=True),
        q3_target=np.array(target, copy=True),
        constraint_row_scales=np.array(constraint_scales, copy=True),
        reaction_channel_basis=basis,
        reaction_channel_transform=np.array(channel_transform, copy=True),
        scaled_monolithic_residual=np.asarray(monolithic_scaled, dtype=float),
        scaled_reaction_residual=np.asarray(reaction_scaled, dtype=float),
        scaled_constraint_residual=np.asarray(constraint, dtype=float),
        augmented_scaled_residual=np.asarray(augmented, dtype=float),
        maximum_zero_multiplier_reduction_defect=float(
            np.linalg.norm(top - monolithic_scaled) / zero_scale
            if not np.any(multiplier)
            else 0.0
        ),
        maximum_constraint_relative_defect=float(
            np.max(np.abs(reaction.q3_value - target) / q_scale)
        ),
    )


def _equilibrated_dense_solve(
    matrix: np.ndarray,
    right_hand_side: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Solve one dense bordered system with max-norm row/column scaling."""

    values = np.asarray(matrix, dtype=float)
    right = np.asarray(right_hand_side, dtype=float)
    if (
        values.ndim != 2
        or values.shape[0] != values.shape[1]
        or right.shape != (values.shape[0],)
        or np.any(~np.isfinite(values))
        or np.any(~np.isfinite(right))
    ):
        raise ValueError("fixed-Q dense linear system is invalid")
    tiny = np.finfo(float).tiny
    row_maximum = np.max(np.abs(values), axis=1)
    if np.any(row_maximum <= tiny):
        raise np.linalg.LinAlgError("fixed-Q dense matrix has a zero row")
    row_scale = 1.0 / row_maximum
    row_scaled = row_scale[:, None] * values
    column_maximum = np.max(np.abs(row_scaled), axis=0)
    if np.any(column_maximum <= tiny):
        raise np.linalg.LinAlgError("fixed-Q dense matrix has a zero column")
    column_scale = 1.0 / column_maximum
    balanced = row_scaled * column_scale[None, :]
    balanced_solution = np.linalg.solve(balanced, row_scale * right)
    solution = column_scale * balanced_solution
    scale = max(float(np.linalg.norm(right)), tiny)
    residual = float(np.linalg.norm(values @ solution - right) / scale)
    return np.asarray(solution, dtype=float), residual


def _relative_array_defect(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    scale = max(
        float(np.linalg.norm(left)),
        float(np.linalg.norm(right)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(left - right) / scale)


def _fixed_q_storage_parity_defect(
    increment_primary: CausalFiveFieldFixedQBDFEvaluation,
    direct_rate: CausalFiveFieldFixedQBDFEvaluation,
) -> float:
    increment_temporal = (
        increment_primary.monolithic_evaluation.mapped_temporal_storage_rows
        + increment_primary.monolithic_evaluation
        .responsive_height_temporal_storage_rows
    )
    direct_temporal = (
        direct_rate.monolithic_evaluation.mapped_temporal_storage_rows
        + direct_rate.monolithic_evaluation
        .responsive_height_temporal_storage_rows
    )
    return max(
        _relative_array_defect(increment_temporal, direct_temporal),
        _relative_array_defect(
            increment_primary.scaled_monolithic_residual,
            direct_rate.scaled_monolithic_residual,
        ),
    )


def _multiplier_weighted_action_ledger_defect(
    reaction: CausalFiveFieldFixedQReaction,
    multipliers: np.ndarray,
    channel_transform: np.ndarray,
) -> float:
    multiplier = np.asarray(multipliers, dtype=float)
    transform = np.asarray(channel_transform, dtype=float)
    if multiplier.shape != (3,) or transform.shape != (3, 3):
        raise ValueError("fixed-Q action-ledger inputs are invalid")
    raw_multiplier = transform @ multiplier
    physical_action = reaction.raw_reaction_physical_rows @ raw_multiplier
    summed = np.sum(physical_action, axis=0)[
        np.asarray(_CONSERVATIVE_FIELDS)
    ]
    channel = reaction.raw_reaction_physical_ledger @ raw_multiplier
    scale = max(
        float(np.linalg.norm(summed)),
        float(np.linalg.norm(channel)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(summed - channel) / scale)


def solve_causal_five_field_fixed_q_bdf(
    context: CausalFiveFieldDAEContext,
    old_primitive_charts: np.ndarray,
    timestep_seconds: float,
    initial_scaled_rate_per_s: np.ndarray,
    initial_multipliers: np.ndarray,
    top_left_scaled_matrix: np.ndarray,
    *,
    order: int = 1,
    history: CausalFiveFieldMonolithicBDFHistory | None = None,
    primitive_column_scales: np.ndarray,
    conservation_row_scales: np.ndarray,
    parent_cell_indices: np.ndarray,
    refinement_ratio: int,
    q3_target: np.ndarray | None = None,
    constraint_row_scales: np.ndarray | None = None,
    reaction_channel_basis: Literal["raw", "frozen_normalized"] = (
        "frozen_normalized"
    ),
    reaction_channel_transform: np.ndarray | None = None,
    exterior_parent_face: int = 36,
    guard_end_parent_face: int = 48,
    parent_cell_count: int = 64,
    residual_tolerance: float = 1.0e-10,
    constraint_tolerance: float = 1.0e-12,
    ledger_tolerance: float = 1.0e-12,
    storage_parity_tolerance: float = 1.0e-9,
    minimum_reconstruction_factor: float = 1.0 - 1.0e-12,
    maximum_schur_condition_number: float = 1.0e8,
    maximum_scaled_primitive_change: float = 5.0e-3,
    maximum_newton_iterations: int = 8,
    maximum_line_search_iterations: int = 12,
    refresh_exact_jacobian: bool = False,
    maximum_exact_jacobian_refreshes: int | None = None,
    exact_jacobian_refresh_policy: Literal[
        "per_iteration", "on_line_search_failure"
    ] = "per_iteration",
    progress_callback: Callable[[dict], None] | None = None,
    initial_scaled_increment: np.ndarray | None = None,
    checkpoint_callback: Callable[
        [int, np.ndarray, np.ndarray, CausalFiveFieldFixedQBDFEvaluation],
        None,
    ]
    | None = None,
    base_reaction: CausalFiveFieldFixedQReaction | None = None,
    physical_state_audit: Callable[
        [CausalFiveFieldDAEContext, np.ndarray], dict
    ]
    | None = None,
    require_physical_state_audit: bool = False,
    maximum_h_over_r: float = 0.12,
    minimum_scattering_optical_depth: float = 1.0,
) -> CausalFiveFieldFixedQBackwardEulerResult:
    """Solve one exact finite constrained BDF step for a consistency audit.

    The supplied top-left matrix is a preconditioner/Jacobian seed.  Secant
    updates act on the complete state-dependent augmented residual, so the
    returned accepted state is a root of the declared nonlinear equations,
    not an Euler predictor.
    """

    context = context.validated()
    old = np.asarray(old_primitive_charts, dtype=float)
    n_cells = int(context.grid.centers.size)
    if old.shape != (n_cells, _N_FIELDS) or np.any(~np.isfinite(old)):
        raise ValueError("fixed-Q BE old state is invalid")
    columns, rows = _validated_scales(
        old,
        primitive_column_scales,
        conservation_row_scales,
    )
    dimensions = int(old.size)
    timestep = float(timestep_seconds)
    selected_order = int(order)
    if selected_order != order or selected_order not in (1, 2):
        raise ValueError("fixed-Q BDF order must be one or two")
    validated_history = (
        None
        if history is None
        else history.validated(n_cells=n_cells)
    )
    if (selected_order == 1) != (validated_history is None):
        raise ValueError("fixed-Q BDF history is inconsistent with order")
    coefficients = causal_bdf_coefficients(
        selected_order,
        timestep,
        (
            None
            if validated_history is None
            else validated_history.previous_timestep_seconds
        ),
    )
    rate = np.asarray(initial_scaled_rate_per_s, dtype=float).reshape(-1)
    multiplier = np.asarray(initial_multipliers, dtype=float)
    top_left = np.asarray(top_left_scaled_matrix, dtype=float)
    if (
        not np.isfinite(timestep)
        or timestep <= 0.0
        or rate.shape != (dimensions,)
        or multiplier.shape != (3,)
        or top_left.shape != (dimensions, dimensions)
        or np.any(~np.isfinite(rate))
        or np.any(~np.isfinite(multiplier))
        or np.any(~np.isfinite(top_left))
    ):
        raise ValueError("fixed-Q BE predictor or matrix is invalid")
    if (
        maximum_exact_jacobian_refreshes is not None
        and (
            int(maximum_exact_jacobian_refreshes)
            != maximum_exact_jacobian_refreshes
            or int(maximum_exact_jacobian_refreshes) < 0
        )
    ):
        raise ValueError("fixed-Q exact-Jacobian refresh count is invalid")
    if exact_jacobian_refresh_policy not in {
        "per_iteration",
        "on_line_search_failure",
    }:
        raise ValueError("fixed-Q exact-Jacobian refresh policy is invalid")
    if reaction_channel_basis not in {"raw", "frozen_normalized"}:
        raise ValueError(
            "fixed-Q nonlinear solve requires raw or frozen-normalized "
            "reaction channels"
        )
    base_reaction = (
        causal_five_field_fixed_q_reaction(
            context,
            old,
            primitive_column_scales=columns,
            conservation_row_scales=rows,
            parent_cell_indices=parent_cell_indices,
            refinement_ratio=refinement_ratio,
            exterior_parent_face=exterior_parent_face,
            guard_end_parent_face=guard_end_parent_face,
            parent_cell_count=parent_cell_count,
            maximum_schur_condition_number=maximum_schur_condition_number,
        )
        if base_reaction is None
        else base_reaction
    )
    if base_reaction.q3_value.shape != (3,):
        raise ValueError("fixed-Q BE base reaction is invalid")
    target = (
        base_reaction.q3_value
        if q3_target is None
        else np.asarray(q3_target, dtype=float)
    )
    constraint_scales = (
        base_reaction.q3_derivative_norms
        if constraint_row_scales is None
        else np.asarray(constraint_row_scales, dtype=float)
    )
    if target.shape != (3,) or constraint_scales.shape != (3,):
        raise ValueError("fixed-Q BE target or constraint scales are invalid")
    if reaction_channel_basis == "raw":
        base_reaction_rows = base_reaction.raw_reaction_scaled_rows
        channel_transform = np.eye(3)
    elif reaction_channel_basis == "frozen_normalized":
        channel_transform = (
            base_reaction.raw_schur_inverse
            if reaction_channel_transform is None
            else np.asarray(reaction_channel_transform, dtype=float)
        )
        if (
            channel_transform.shape != (3, 3)
            or np.any(~np.isfinite(channel_transform))
        ):
            raise ValueError("fixed-Q BE frozen reaction transform is invalid")
        base_reaction_rows = (
            base_reaction.raw_reaction_scaled_rows @ channel_transform
        )
    else:
        raise ValueError("fixed-Q BE reaction channel basis is invalid")
    initial_top_left = np.array(top_left, copy=True)
    raw_jacobian = causal_five_field_fixed_q_raw_reaction_jacobian(
        context,
        old,
        primitive_column_scales=columns,
        conservation_row_scales=rows,
        parent_cell_indices=parent_cell_indices,
        refinement_ratio=refinement_ratio,
        exterior_parent_face=exterior_parent_face,
        guard_end_parent_face=guard_end_parent_face,
        parent_cell_count=parent_cell_count,
        reaction=base_reaction,
    )
    initial_top_left -= np.tensordot(
        raw_jacobian.scaled_jacobian,
        channel_transform @ multiplier,
        axes=(1, 0),
    )
    matrix = np.block(
        [
            [initial_top_left, -base_reaction_rows],
            [base_reaction.q3_scaled_derivative, np.zeros((3, 3))],
        ]
    )
    previous_scaled_increment = (
        None
        if validated_history is None
        else (
            validated_history.previous_primitive_increment
            / columns
        ).ravel()
    )
    if initial_scaled_increment is None:
        scaled_increment = timestep * rate
        if previous_scaled_increment is not None:
            scaled_increment = (
                scaled_increment
                - coefficients.previous_increment_coefficient
                * previous_scaled_increment
            ) / coefficients.current_increment_coefficient
    else:
        scaled_increment = np.asarray(
            initial_scaled_increment,
            dtype=float,
        ).reshape(-1)
    if (
        scaled_increment.shape != (dimensions,)
        or np.any(~np.isfinite(scaled_increment))
        or np.max(np.abs(scaled_increment)) > maximum_scaled_primitive_change
    ):
        raise ValueError("fixed-Q BDF initial increment is invalid")
    unknown = np.concatenate((scaled_increment, multiplier))
    function_evaluations = 0

    def residual(values: np.ndarray):
        nonlocal function_evaluations
        function_evaluations += 1
        scaled_increment = np.asarray(values[:dimensions], dtype=float)
        candidate = old + (
            columns.ravel() * scaled_increment
        ).reshape(old.shape)
        evaluation = evaluate_causal_five_field_fixed_q_bdf(
            old,
            candidate,
            np.asarray(values[dimensions:], dtype=float),
            target,
            timestep,
            context,
            order=selected_order,
            history=validated_history,
            primitive_column_scales=columns,
            conservation_row_scales=rows,
            parent_cell_indices=parent_cell_indices,
            refinement_ratio=refinement_ratio,
            exterior_parent_face=exterior_parent_face,
            guard_end_parent_face=guard_end_parent_face,
            parent_cell_count=parent_cell_count,
            constraint_row_scales=constraint_scales,
            reaction_channel_basis=reaction_channel_basis,
            reaction_channel_transform=channel_transform,
            scaled_primitive_increment=scaled_increment,
            maximum_schur_condition_number=maximum_schur_condition_number,
        )
        return evaluation.augmented_scaled_residual, evaluation

    values, evaluation = residual(unknown)
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "initial_residual",
                "maximum_scaled_residual": float(np.max(np.abs(values))),
                "function_evaluations": function_evaluations,
            }
        )
    linear_residuals = []
    exact_jacobian_refreshes = 0
    broyden_updates = 0
    linear_solves = 0
    message = "maximum Newton iterations reached"
    success = False
    iterations = 0
    bound = float(maximum_scaled_primitive_change)

    def assemble_exact_matrix(reason: str) -> np.ndarray:
        nonlocal exact_jacobian_refreshes
        candidate_state = old + (
            columns.ravel() * unknown[:dimensions]
        ).reshape(old.shape)
        exact_augmented = causal_five_field_fixed_q_augmented_step_matrix(
            context,
            old,
            candidate_state,
            unknown[dimensions:],
            timestep,
            (
                None
                if validated_history is None
                else validated_history.previous_timestep_seconds
            ),
            order=selected_order,
            primitive_column_scales=columns,
            conservation_row_scales=rows,
            parent_cell_indices=parent_cell_indices,
            refinement_ratio=refinement_ratio,
            constraint_row_scales=constraint_scales,
            reaction_channel_basis=reaction_channel_basis,
            reaction_channel_transform=channel_transform,
            exterior_parent_face=exterior_parent_face,
            guard_end_parent_face=guard_end_parent_face,
            parent_cell_count=parent_cell_count,
            reaction=evaluation.reaction,
        )
        exact_jacobian_refreshes += 1
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "exact_jacobian_refresh",
                    "reason": reason,
                    "iteration": iterations,
                    "exact_jacobian_assemblies": exact_jacobian_refreshes,
                    "maximum_scaled_residual": float(
                        np.max(np.abs(values))
                    ),
                    "function_evaluations": function_evaluations,
                }
            )
        return exact_augmented.scaled_matrix

    for iteration in range(int(maximum_newton_iterations) + 1):
        iterations = iteration
        maximum = float(np.max(np.abs(values)))
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "newton_iteration",
                    "iteration": iteration,
                    "maximum_scaled_residual": maximum,
                    "function_evaluations": function_evaluations,
                }
            )
        if maximum <= residual_tolerance:
            success = True
            message = "residual gate passed"
            break
        if iteration == maximum_newton_iterations:
            break
        if (
            refresh_exact_jacobian
            and reaction_channel_basis in {"raw", "frozen_normalized"}
            and (
                exact_jacobian_refresh_policy == "per_iteration"
                or exact_jacobian_refreshes == 0
            )
            and (
                maximum_exact_jacobian_refreshes is None
                or exact_jacobian_refreshes
                < int(maximum_exact_jacobian_refreshes)
            )
        ):
            matrix = assemble_exact_matrix(
                "initial"
                if exact_jacobian_refreshes == 0
                else "per_iteration"
            )
        try:
            correction, linear_residual = _equilibrated_dense_solve(
                matrix,
                -values,
            )
            linear_solves += 1
        except np.linalg.LinAlgError:
            message = "fixed-Q bordered Newton matrix is singular"
            break
        linear_residuals.append(linear_residual)
        alpha = 1.0
        state = unknown[:dimensions]
        state_correction = correction[:dimensions]
        positive = state_correction > 0.0
        negative = state_correction < 0.0
        if np.any(positive):
            alpha = min(
                alpha,
                float(
                    np.min(
                        (bound - state[positive])
                        / state_correction[positive]
                    )
                ),
            )
        if np.any(negative):
            alpha = min(
                alpha,
                float(
                    np.min(
                        (-bound - state[negative])
                        / state_correction[negative]
                    )
                ),
            )
        alpha = (
            1.0
            if alpha >= 1.0
            else min(1.0, max(0.0, 0.99 * alpha))
        )
        accepted_correction = False
        merit = float(np.linalg.norm(values))
        for _line_search in range(int(maximum_line_search_iterations)):
            candidate_unknown = unknown + alpha * correction
            candidate_values, candidate_evaluation = residual(candidate_unknown)
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "line_search",
                        "iteration": iteration,
                        "line_search_iteration": _line_search,
                        "alpha": alpha,
                        "maximum_scaled_residual": float(
                            np.max(np.abs(candidate_values))
                        ),
                        "function_evaluations": function_evaluations,
                    }
                )
            candidate_merit = float(np.linalg.norm(candidate_values))
            if (
                candidate_merit < merit
                or np.max(np.abs(candidate_values)) <= residual_tolerance
            ):
                secant_step = candidate_unknown - unknown
                secant_change = candidate_values - values
                denominator = float(secant_step @ secant_step)
                if denominator > np.finfo(float).tiny:
                    matrix = matrix + np.outer(
                        secant_change - matrix @ secant_step,
                        secant_step,
                    ) / denominator
                    broyden_updates += 1
                unknown = candidate_unknown
                values = candidate_values
                evaluation = candidate_evaluation
                accepted_correction = True
                if checkpoint_callback is not None:
                    checkpoint_callback(
                        iteration + 1,
                        np.array(unknown[:dimensions], copy=True),
                        np.array(unknown[dimensions:], copy=True),
                        evaluation,
                    )
                break
            alpha *= 0.5
        if not accepted_correction:
            if float(np.max(np.abs(values))) <= residual_tolerance:
                success = True
                message = "residual gate passed at line-search floor"
            elif (
                exact_jacobian_refresh_policy == "on_line_search_failure"
                and refresh_exact_jacobian
                and reaction_channel_basis in {"raw", "frozen_normalized"}
                and (
                    maximum_exact_jacobian_refreshes is None
                    or exact_jacobian_refreshes
                    < int(maximum_exact_jacobian_refreshes)
                )
                and iteration + 2 <= int(maximum_newton_iterations)
            ):
                matrix = assemble_exact_matrix("line_search_failure")
                message = "fresh exact Jacobian after line-search failure"
                continue
            else:
                message = "fixed-Q bound-aware line search failed"
            if success or message == "fixed-Q bound-aware line search failed":
                break
    scaled_increment = unknown[:dimensions]
    new = old + (columns.ravel() * scaled_increment).reshape(old.shape)
    accepted_scaled_increment = np.asarray(
        unknown[:dimensions],
        dtype=float,
    )
    maximum_residual = float(np.max(np.abs(values)))
    scaled_interval_rate = accepted_scaled_increment / timestep
    scaled_bdf_rate = causal_bdf_weighted_increment(
        accepted_scaled_increment,
        previous_scaled_increment,
        coefficients,
    ) / timestep
    direct_rate_evaluation = evaluate_causal_five_field_fixed_q_bdf(
        old,
        new,
        unknown[dimensions:],
        target,
        timestep,
        context,
        order=selected_order,
        history=validated_history,
        primitive_column_scales=columns,
        conservation_row_scales=rows,
        parent_cell_indices=parent_cell_indices,
        refinement_ratio=refinement_ratio,
        exterior_parent_face=exterior_parent_face,
        guard_end_parent_face=guard_end_parent_face,
        parent_cell_count=parent_cell_count,
        constraint_row_scales=constraint_scales,
        reaction_channel_basis=reaction_channel_basis,
        reaction_channel_transform=channel_transform,
        scaled_primitive_increment=accepted_scaled_increment,
        scaled_rate_per_s=scaled_interval_rate,
        maximum_schur_condition_number=maximum_schur_condition_number,
    )
    storage_parity_defect = _fixed_q_storage_parity_defect(
        evaluation,
        direct_rate_evaluation,
    )
    action_ledger_defect = _multiplier_weighted_action_ledger_defect(
        evaluation.reaction,
        unknown[dimensions:],
        channel_transform,
    )
    maximum_change = float(np.max(np.abs(accepted_scaled_increment)))
    minimum_factor = min(
        float(
            evaluation.monolithic_evaluation.current_storage_increment
            .minimum_path_reconstruction_factor
        ),
        evaluation.reaction.minimum_q3_reconstruction_factor,
    )
    maximum_factor = max(
        float(
            evaluation.monolithic_evaluation.current_storage_increment
            .maximum_path_reconstruction_factor
        ),
        evaluation.reaction.maximum_q3_reconstruction_factor,
    )
    scaled_reaction_action = (
        evaluation.reaction.raw_reaction_lift
        @ channel_transform
        @ unknown[dimensions:]
    )
    q3_rate_tangency_defect = float(
        np.linalg.norm(
            evaluation.reaction.q3_scaled_derivative @ scaled_bdf_rate
        )
        / max(float(np.linalg.norm(scaled_bdf_rate)), np.finfo(float).tiny)
    )
    physical_audit = (
        None
        if physical_state_audit is None
        else physical_state_audit(context, new)
    )
    if physical_audit is not None and not isinstance(physical_audit, dict):
        raise ValueError("fixed-Q physical state audit is invalid")
    physical_height_passed = bool(
        not require_physical_state_audit
        or (
            physical_audit is not None
            and float(physical_audit["maximum_h_over_r"])
            <= float(maximum_h_over_r)
        )
    )
    physical_optical_passed = bool(
        not require_physical_state_audit
        or (
            physical_audit is not None
            and float(physical_audit["minimum_scattering_optical_depth"])
            >= float(minimum_scattering_optical_depth)
        )
    )
    checks = {
        "nonlinear_root": success,
        "complete_residual": maximum_residual <= residual_tolerance,
        "Q3": evaluation.maximum_constraint_relative_defect
        <= constraint_tolerance,
        "incoming_excision": (
            evaluation.monolithic_evaluation.incoming_excision_characteristics
            == 0
        ),
        "storage_parity": storage_parity_defect <= storage_parity_tolerance,
        "reconstruction": (
            minimum_factor >= minimum_reconstruction_factor
            and maximum_factor <= 1.0 + 1.0e-12
        ),
        "reaction_ledger": (
            evaluation.reaction.maximum_reaction_ledger_relative_defect
            <= ledger_tolerance
        ),
        "constraint_action_ledger": action_ledger_defect <= ledger_tolerance,
        "primitive_change": maximum_change <= maximum_scaled_primitive_change,
        "reaction_conditioning": (
            evaluation.reaction.raw_schur_numerical_rank == 3
            and evaluation.reaction.raw_schur_condition_number
            <= maximum_schur_condition_number
            and evaluation.reaction.maximum_raw_schur_solve_relative_defect
            <= ledger_tolerance
        ),
        "physical_height": physical_height_passed,
        "physical_optical_depth": physical_optical_passed,
    }
    failure_reasons = tuple(name for name, passed in checks.items() if not passed)
    accepted = not failure_reasons
    acceptance = CausalFiveFieldFixedQStepAcceptance(
        nonlinear_root_passed=checks["nonlinear_root"],
        complete_residual_passed=checks["complete_residual"],
        q3_passed=checks["Q3"],
        incoming_excision_passed=checks["incoming_excision"],
        storage_parity_passed=checks["storage_parity"],
        reconstruction_passed=checks["reconstruction"],
        reaction_ledger_passed=checks["reaction_ledger"],
        constraint_action_ledger_passed=checks["constraint_action_ledger"],
        primitive_change_passed=checks["primitive_change"],
        reaction_conditioning_passed=checks["reaction_conditioning"],
        physical_height_passed=checks["physical_height"],
        physical_optical_depth_passed=checks["physical_optical_depth"],
        accepted=accepted,
        failure_reasons=failure_reasons,
    )
    if success and not accepted:
        message = "fixed-Q root exceeds gates: " + ", ".join(failure_reasons)
    return CausalFiveFieldFixedQBackwardEulerResult(
        primitive_charts=np.array(new, copy=True),
        primitive_increment=np.asarray(
            columns.ravel() * accepted_scaled_increment,
            dtype=float,
        ).reshape(old.shape),
        scaled_rate_per_s=np.asarray(scaled_bdf_rate, dtype=float),
        scaled_interval_rate_per_s=np.asarray(
            scaled_interval_rate,
            dtype=float,
        ),
        multipliers=np.asarray(
            unknown[dimensions:],
            dtype=float,
        ),
        evaluation=evaluation,
        direct_rate_evaluation=direct_rate_evaluation,
        order=selected_order,
        accepted=accepted,
        acceptance=acceptance,
        iterations=iterations,
        function_evaluations=function_evaluations,
        maximum_scaled_residual=maximum_residual,
        maximum_linear_residual=max(linear_residuals, default=0.0),
        maximum_scaled_primitive_change=maximum_change,
        minimum_path_reconstruction_factor=minimum_factor,
        maximum_path_reconstruction_factor=maximum_factor,
        maximum_direct_rate_increment_parity_defect=storage_parity_defect,
        maximum_multiplier_weighted_action_ledger_relative_defect=(
            action_ledger_defect
        ),
        scaled_reaction_rate_action_per_s=np.asarray(
            scaled_reaction_action,
            dtype=float,
        ),
        maximum_scaled_q3_rate_tangency_defect=q3_rate_tangency_defect,
        maximum_h_over_r=(
            None
            if physical_audit is None
            else float(physical_audit["maximum_h_over_r"])
        ),
        minimum_scattering_optical_depth=(
            None
            if physical_audit is None
            else float(physical_audit["minimum_scattering_optical_depth"])
        ),
        exact_jacobian_assemblies=exact_jacobian_refreshes,
        broyden_updates=broyden_updates,
        linear_solves=linear_solves,
        message=message,
    )


def causal_five_field_fixed_q_accepted_history(
    result: CausalFiveFieldFixedQBackwardEulerResult,
) -> CausalFiveFieldMonolithicBDFHistory:
    """Freeze history only from one fully accepted constrained BDF step."""

    if not result.accepted or not result.acceptance.accepted:
        raise ValueError("rejected fixed-Q step cannot define BDF history")
    return causal_five_field_monolithic_bdf_history(
        result.primitive_increment,
        result.evaluation.monolithic_evaluation.current_storage_increment,
        result.evaluation.monolithic_evaluation.coefficients
        .current_timestep_seconds,
    )


def causal_five_field_fixed_q_bdf_restart(
    result: CausalFiveFieldFixedQBackwardEulerResult,
    context: CausalFiveFieldDAEContext,
    previous_primitive_charts: np.ndarray,
    *,
    primitive_column_scales: np.ndarray,
    conservation_row_scales: np.ndarray,
    parent_cell_indices: np.ndarray,
    refinement_ratio: int,
    exterior_parent_face: int = 36,
    guard_end_parent_face: int = 48,
    parent_cell_count: int = 64,
    maximum_schur_condition_number: float = 1.0e8,
    q3_target: np.ndarray,
    constraint_row_scales: np.ndarray,
    reaction_channel_basis: Literal["raw", "frozen_normalized"],
    elapsed_time_seconds: float,
    completed_steps: int,
    provenance: dict,
) -> CausalFiveFieldFixedQBDFRestart:
    """Construct one replay payload from an accepted constrained BDF1 step."""

    if result.order != 1:
        raise ValueError("fixed-Q replay restart must follow BDF1 startup")
    history = causal_five_field_fixed_q_accepted_history(result)
    previous = np.asarray(previous_primitive_charts, dtype=float)
    target = np.asarray(q3_target, dtype=float)
    scales = np.asarray(constraint_row_scales, dtype=float)
    endpoint_reaction = causal_five_field_fixed_q_reaction(
        context,
        result.primitive_charts,
        primitive_column_scales=primitive_column_scales,
        conservation_row_scales=conservation_row_scales,
        parent_cell_indices=parent_cell_indices,
        refinement_ratio=refinement_ratio,
        exterior_parent_face=exterior_parent_face,
        guard_end_parent_face=guard_end_parent_face,
        parent_cell_count=parent_cell_count,
        maximum_schur_condition_number=maximum_schur_condition_number,
    )
    transform = (
        np.eye(3)
        if reaction_channel_basis == "raw"
        else endpoint_reaction.raw_schur_inverse
    )
    if (
        previous.shape != result.primitive_charts.shape
        or np.any(~np.isfinite(previous))
        or target.shape != (3,)
        or scales.shape != (3,)
        or transform.shape != (3, 3)
        or np.any(~np.isfinite(target))
        or np.any(~np.isfinite(scales))
        or np.any(scales <= 0.0)
        or np.any(~np.isfinite(transform))
        or reaction_channel_basis not in {"raw", "frozen_normalized"}
    ):
        raise ValueError("fixed-Q restart inputs are invalid")
    return CausalFiveFieldFixedQBDFRestart(
        primitive_charts=np.array(result.primitive_charts, copy=True),
        previous_primitive_charts=np.array(previous, copy=True),
        history=history,
        q3_target=np.array(target, copy=True),
        constraint_row_scales=np.array(scales, copy=True),
        multiplier_predictor=np.array(result.multipliers, copy=True),
        reaction_channel_basis=str(reaction_channel_basis),
        reaction_channel_transform=np.array(transform, copy=True),
        previous_minimum_path_reconstruction_factor=float(
            result.minimum_path_reconstruction_factor
        ),
        elapsed_time_seconds=float(elapsed_time_seconds),
        completed_steps=int(completed_steps),
        next_order=2,
        provenance=dict(provenance),
    )


def _validated_fixed_q_restart(
    context: CausalFiveFieldDAEContext,
    restart: CausalFiveFieldFixedQBDFRestart,
) -> CausalFiveFieldFixedQBDFRestart:
    context = context.validated()
    shape = (int(context.grid.centers.size), _N_FIELDS)
    current = np.asarray(restart.primitive_charts, dtype=float)
    previous = np.asarray(restart.previous_primitive_charts, dtype=float)
    target = np.asarray(restart.q3_target, dtype=float)
    scales = np.asarray(restart.constraint_row_scales, dtype=float)
    multiplier = np.asarray(restart.multiplier_predictor, dtype=float)
    transform = np.asarray(restart.reaction_channel_transform, dtype=float)
    history = restart.history.validated(n_cells=shape[0])
    elapsed = float(restart.elapsed_time_seconds)
    factor = float(restart.previous_minimum_path_reconstruction_factor)
    if (
        current.shape != shape
        or previous.shape != shape
        or np.any(~np.isfinite(current))
        or np.any(~np.isfinite(previous))
        or target.shape != (3,)
        or scales.shape != (3,)
        or multiplier.shape != (3,)
        or transform.shape != (3, 3)
        or np.any(~np.isfinite(target))
        or np.any(~np.isfinite(scales))
        or np.any(scales <= 0.0)
        or np.any(~np.isfinite(multiplier))
        or np.any(~np.isfinite(transform))
        or restart.reaction_channel_basis not in {"raw", "frozen_normalized"}
        or not np.isfinite(factor)
        or factor < 1.0 - 1.0e-12
        or not np.isfinite(elapsed)
        or elapsed < 0.0
        or int(restart.completed_steps) != restart.completed_steps
        or restart.completed_steps < 1
        or int(restart.next_order) != restart.next_order
        or restart.next_order != 2
        or restart.schema_version != 1
        or not isinstance(restart.provenance, dict)
    ):
        raise ValueError("fixed-Q BDF restart is invalid")
    if not np.array_equal(
        current,
        previous + history.previous_primitive_increment,
    ):
        raise ValueError("fixed-Q restart primitive history is inconsistent")
    return CausalFiveFieldFixedQBDFRestart(
        primitive_charts=np.array(current, copy=True),
        previous_primitive_charts=np.array(previous, copy=True),
        history=history,
        q3_target=np.array(target, copy=True),
        constraint_row_scales=np.array(scales, copy=True),
        multiplier_predictor=np.array(multiplier, copy=True),
        reaction_channel_basis=str(restart.reaction_channel_basis),
        reaction_channel_transform=np.array(transform, copy=True),
        previous_minimum_path_reconstruction_factor=factor,
        elapsed_time_seconds=elapsed,
        completed_steps=int(restart.completed_steps),
        next_order=2,
        provenance=dict(restart.provenance),
        schema_version=1,
    )


def causal_five_field_fixed_q_bdf_restarts_equal(
    left: CausalFiveFieldFixedQBDFRestart,
    right: CausalFiveFieldFixedQBDFRestart,
) -> bool:
    """Return whether two constrained restart payloads are bitwise equal."""

    arrays = (
        (left.primitive_charts, right.primitive_charts),
        (left.previous_primitive_charts, right.previous_primitive_charts),
        (
            left.history.previous_primitive_increment,
            right.history.previous_primitive_increment,
        ),
        (
            left.history.previous_mapped_storage_increment,
            right.history.previous_mapped_storage_increment,
        ),
        (
            left.history.previous_responsive_height_storage_increment,
            right.history.previous_responsive_height_storage_increment,
        ),
        (left.q3_target, right.q3_target),
        (left.constraint_row_scales, right.constraint_row_scales),
        (left.multiplier_predictor, right.multiplier_predictor),
        (left.reaction_channel_transform, right.reaction_channel_transform),
    )
    return bool(
        all(np.array_equal(first, second) for first, second in arrays)
        and left.history.previous_timestep_seconds
        == right.history.previous_timestep_seconds
        and left.history.temporal_path_scheme
        == right.history.temporal_path_scheme
        and left.reaction_channel_basis == right.reaction_channel_basis
        and left.previous_minimum_path_reconstruction_factor
        == right.previous_minimum_path_reconstruction_factor
        and left.elapsed_time_seconds == right.elapsed_time_seconds
        and left.completed_steps == right.completed_steps
        and left.next_order == right.next_order
        and left.provenance == right.provenance
        and left.schema_version == right.schema_version
    )


def save_causal_five_field_fixed_q_bdf_restart(
    path: str | Path,
    context: CausalFiveFieldDAEContext,
    restart: CausalFiveFieldFixedQBDFRestart,
) -> None:
    """Atomically persist one complete constrained BDF restart."""

    validated = _validated_fixed_q_restart(context, restart)
    destination = Path(path)
    if destination.suffix != ".npz":
        raise ValueError("fixed-Q BDF restart path must end in .npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            primitive_charts=validated.primitive_charts,
            previous_primitive_charts=validated.previous_primitive_charts,
            previous_primitive_increment=(
                validated.history.previous_primitive_increment
            ),
            previous_mapped_storage_increment=(
                validated.history.previous_mapped_storage_increment
            ),
            previous_responsive_height_storage_increment=(
                validated.history.previous_responsive_height_storage_increment
            ),
            previous_timestep_seconds=np.asarray(
                validated.history.previous_timestep_seconds,
                dtype="<f8",
            ),
            temporal_path_scheme=np.asarray(
                validated.history.temporal_path_scheme
            ),
            q3_target=validated.q3_target,
            constraint_row_scales=validated.constraint_row_scales,
            multiplier_predictor=validated.multiplier_predictor,
            reaction_channel_basis=np.asarray(
                validated.reaction_channel_basis
            ),
            reaction_channel_transform=validated.reaction_channel_transform,
            previous_minimum_path_reconstruction_factor=np.asarray(
                validated.previous_minimum_path_reconstruction_factor,
                dtype="<f8",
            ),
            elapsed_time_seconds=np.asarray(
                validated.elapsed_time_seconds,
                dtype="<f8",
            ),
            completed_steps=np.asarray(validated.completed_steps, dtype="<i8"),
            next_order=np.asarray(validated.next_order, dtype="<i8"),
            provenance_json=np.asarray(
                json.dumps(
                    validated.provenance,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            ),
            schema_version=np.asarray(validated.schema_version, dtype="<i8"),
        )
    temporary.replace(destination)


def load_causal_five_field_fixed_q_bdf_restart(
    path: str | Path,
    context: CausalFiveFieldDAEContext,
    *,
    expected_provenance: dict | None = None,
) -> CausalFiveFieldFixedQBDFRestart:
    """Load and validate one complete constrained BDF restart."""

    with np.load(Path(path), allow_pickle=False) as source:
        restart = CausalFiveFieldFixedQBDFRestart(
            primitive_charts=np.asarray(source["primitive_charts"], dtype=float),
            previous_primitive_charts=np.asarray(
                source["previous_primitive_charts"], dtype=float
            ),
            history=CausalFiveFieldMonolithicBDFHistory(
                previous_primitive_increment=np.asarray(
                    source["previous_primitive_increment"], dtype=float
                ),
                previous_mapped_storage_increment=np.asarray(
                    source["previous_mapped_storage_increment"], dtype=float
                ),
                previous_responsive_height_storage_increment=np.asarray(
                    source["previous_responsive_height_storage_increment"],
                    dtype=float,
                ),
                previous_timestep_seconds=float(
                    source["previous_timestep_seconds"]
                ),
                temporal_path_scheme=str(
                    source["temporal_path_scheme"].item()
                ),
            ),
            q3_target=np.asarray(source["q3_target"], dtype=float),
            constraint_row_scales=np.asarray(
                source["constraint_row_scales"], dtype=float
            ),
            multiplier_predictor=np.asarray(
                source["multiplier_predictor"], dtype=float
            ),
            reaction_channel_basis=str(
                source["reaction_channel_basis"].item()
            ),
            reaction_channel_transform=np.asarray(
                source["reaction_channel_transform"], dtype=float
            ),
            previous_minimum_path_reconstruction_factor=float(
                source["previous_minimum_path_reconstruction_factor"]
            ),
            elapsed_time_seconds=float(source["elapsed_time_seconds"]),
            completed_steps=int(source["completed_steps"]),
            next_order=int(source["next_order"]),
            provenance=json.loads(str(source["provenance_json"].item())),
            schema_version=int(source["schema_version"]),
        )
    validated = _validated_fixed_q_restart(context, restart)
    if (
        expected_provenance is not None
        and validated.provenance != expected_provenance
    ):
        raise ValueError("fixed-Q BDF restart provenance differs")
    return validated


def solve_causal_five_field_fixed_q_backward_euler(
    *args,
    **kwargs,
) -> CausalFiveFieldFixedQBackwardEulerResult:
    """Backward-compatible wrapper for one exact constrained BDF1 step."""

    if "order" in kwargs or "history" in kwargs:
        raise TypeError(
            "backward-Euler wrapper does not accept order or history"
        )
    return solve_causal_five_field_fixed_q_bdf(
        *args,
        order=1,
        history=None,
        **kwargs,
    )
