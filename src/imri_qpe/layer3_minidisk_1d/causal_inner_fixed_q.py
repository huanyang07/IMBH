"""Ledger-constrained fixed-Q helpers for the monolithic inner DAE.

This module is production neutral.  It defines the exact exterior-domain
``Q3=(M,J,E)`` endpoint map, the ledger-derived reaction coordinate, and an
augmented BDF residual.  It does not advance a trajectory or change any
production integration default.
"""

from __future__ import annotations

from dataclasses import dataclass

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
from .causal_inner_monolithic_dae import (
    _integrated_mapped_storage,
    _spatial_nodes,
)
from .causal_inner_monolithic_tangent import (
    _descriptor_matrices,
    _node_reconstruction_weights,
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
    scaled_monolithic_residual: np.ndarray
    scaled_reaction_residual: np.ndarray
    scaled_constraint_residual: np.ndarray
    augmented_scaled_residual: np.ndarray
    maximum_zero_multiplier_reduction_defect: float
    maximum_constraint_relative_defect: float


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
    q_physical = []
    for component in _CONSERVATIVE_FIELDS:
        selector = np.zeros(state.size, dtype=float)
        for cell in range(exterior_face, n_cells):
            selector[_N_FIELDS * cell + component] = C * rows[cell, component]
        q_physical.append(selector @ mapped)
    q_physical = np.asarray(q_physical, dtype=float)
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

    physical_raw = np.zeros((n_cells, _N_FIELDS, 3), dtype=float)
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

    raw = (physical_raw / C / rows[:, :, None]).reshape(state.size, 3)
    factor = splu(csc_matrix(descriptor))
    raw_lift = factor.solve(raw)
    raw_schur = q_scaled @ raw_lift
    reaction = raw @ np.linalg.inv(raw_schur)
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
    channel_ledger = raw_ledger @ np.linalg.inv(raw_schur)
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
    monolithic_scaled = monolithic.residual_rows.ravel() / rows.ravel()
    reaction_scaled = reaction.reaction_scaled_rows @ multiplier
    constraint = (reaction.q3_value - target) / reaction.q3_derivative_norms
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
