"""Ledger-constrained fixed-Q helpers for the monolithic inner DAE.

This module is production neutral.  It defines the exact exterior-domain
``Q3=(M,J,E)`` endpoint map, the ledger-derived reaction coordinate, and an
augmented BDF residual.  It does not advance a trajectory or change any
production integration default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from imri_qpe.constants import C

from .causal_inner_dae_system import CausalFiveFieldDAEContext, _cell_state
from .causal_inner_monolithic_bdf import (
    CausalFiveFieldMonolithicBDFEvaluation,
    CausalFiveFieldMonolithicBDFHistory,
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
    raw_schur_inverse: np.ndarray
    support_cell_indices: np.ndarray
    support_envelope: np.ndarray
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
    """One exact constrained backward-Euler solve used by the limit audit."""

    primitive_charts: np.ndarray
    scaled_rate_per_s: np.ndarray
    multipliers: np.ndarray
    evaluation: CausalFiveFieldFixedQBDFEvaluation
    accepted: bool
    iterations: int
    function_evaluations: int
    maximum_scaled_residual: float
    maximum_linear_residual: float
    message: str


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
    raw_schur_inverse = np.linalg.inv(raw_schur)
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
    if float(np.max(factors)) > 1.0:
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
        raw_schur_inverse=np.asarray(raw_schur_inverse, dtype=float),
        support_cell_indices=np.asarray(support, dtype=int),
        support_envelope=np.asarray(envelope, dtype=float),
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
    scaled_rate_per_s: np.ndarray | None = None,
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


def solve_causal_five_field_fixed_q_backward_euler(
    context: CausalFiveFieldDAEContext,
    old_primitive_charts: np.ndarray,
    timestep_seconds: float,
    initial_scaled_rate_per_s: np.ndarray,
    initial_multipliers: np.ndarray,
    top_left_scaled_matrix: np.ndarray,
    *,
    primitive_column_scales: np.ndarray,
    conservation_row_scales: np.ndarray,
    parent_cell_indices: np.ndarray,
    refinement_ratio: int,
    q3_target: np.ndarray | None = None,
    constraint_row_scales: np.ndarray | None = None,
    reaction_channel_basis: str = "frozen_normalized",
    reaction_channel_transform: np.ndarray | None = None,
    exterior_parent_face: int = 36,
    guard_end_parent_face: int = 48,
    parent_cell_count: int = 64,
    residual_tolerance: float = 1.0e-10,
    constraint_tolerance: float = 1.0e-12,
    maximum_scaled_primitive_change: float = 5.0e-3,
    maximum_newton_iterations: int = 8,
    maximum_line_search_iterations: int = 12,
    refresh_exact_jacobian: bool = False,
    maximum_exact_jacobian_refreshes: int | None = None,
    progress_callback: Callable[[dict], None] | None = None,
    initial_scaled_increment: np.ndarray | None = None,
    checkpoint_callback: Callable[
        [int, np.ndarray, np.ndarray, CausalFiveFieldFixedQBDFEvaluation],
        None,
    ]
    | None = None,
    base_reaction: CausalFiveFieldFixedQReaction | None = None,
) -> CausalFiveFieldFixedQBackwardEulerResult:
    """Solve an exact finite constrained BE step for the KKT-limit audit.

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
    if reaction_channel_basis == "normalized":
        base_reaction_rows = base_reaction.reaction_scaled_rows
        channel_transform = base_reaction.raw_schur_inverse
    elif reaction_channel_basis == "raw":
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
    if reaction_channel_basis in {"raw", "frozen_normalized"}:
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
    scaled_increment = (
        timestep * rate
        if initial_scaled_increment is None
        else np.asarray(initial_scaled_increment, dtype=float).reshape(-1)
    )
    if (
        scaled_increment.shape != (dimensions,)
        or np.any(~np.isfinite(scaled_increment))
        or np.max(np.abs(scaled_increment)) > maximum_scaled_primitive_change
    ):
        raise ValueError("fixed-Q BE initial increment is invalid")
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
            order=1,
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
            scaled_rate_per_s=scaled_increment / timestep,
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
    message = "maximum Newton iterations reached"
    success = False
    iterations = 0
    bound = float(maximum_scaled_primitive_change)
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
                maximum_exact_jacobian_refreshes is None
                or exact_jacobian_refreshes
                < int(maximum_exact_jacobian_refreshes)
            )
        ):
            candidate_state = old + (
                columns.ravel() * unknown[:dimensions]
            ).reshape(old.shape)
            exact_augmented = causal_five_field_fixed_q_augmented_step_matrix(
                context,
                old,
                candidate_state,
                unknown[dimensions:],
                timestep,
                None,
                order=1,
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
            matrix = exact_augmented.scaled_matrix
            exact_jacobian_refreshes += 1
        try:
            correction, linear_residual = _equilibrated_dense_solve(
                matrix,
                -values,
            )
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
            if np.max(np.abs(candidate_values)) < maximum:
                secant_step = candidate_unknown - unknown
                secant_change = candidate_values - values
                denominator = float(secant_step @ secant_step)
                if denominator > np.finfo(float).tiny:
                    matrix = matrix + np.outer(
                        secant_change - matrix @ secant_step,
                        secant_step,
                    ) / denominator
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
            else:
                message = "fixed-Q bound-aware line search failed"
            break
    scaled_increment = unknown[:dimensions]
    new = old + (columns.ravel() * scaled_increment).reshape(old.shape)
    maximum_residual = float(np.max(np.abs(values)))
    accepted = bool(
        success
        and maximum_residual <= residual_tolerance
        and evaluation.maximum_constraint_relative_defect
        <= constraint_tolerance
        and evaluation.monolithic_evaluation.maximum_block_ledger_defect
        <= 1.0e-12
        and evaluation.reaction.maximum_reaction_ledger_relative_defect
        <= 1.0e-12
        and evaluation.monolithic_evaluation.incoming_excision_characteristics
        == 0
    )
    if success and not accepted:
        message = "fixed-Q root exceeds one or more acceptance gates"
    return CausalFiveFieldFixedQBackwardEulerResult(
        primitive_charts=np.array(new, copy=True),
        scaled_rate_per_s=np.asarray(
            scaled_increment / timestep,
            dtype=float,
        ),
        multipliers=np.asarray(
            unknown[dimensions:],
            dtype=float,
        ),
        evaluation=evaluation,
        accepted=accepted,
        iterations=iterations,
        function_evaluations=function_evaluations,
        maximum_scaled_residual=maximum_residual,
        maximum_linear_residual=max(linear_residuals, default=0.0),
        message=message,
    )
