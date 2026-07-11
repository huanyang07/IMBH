"""Augmented coupled solve with an emergent inner accretion rate."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from .coupled_inner_outer import (
    CoupledInnerOuterContext,
    CoupledResidualEvaluation,
    colored_coupled_jacobian,
    coupled_row_slices,
    coupled_state_bounds,
    evaluate_coupled_inner_outer_residual,
    unpack_coupled_state,
)
from .transonic_collocation import sonic_residual_jacobian


@dataclass(frozen=True)
class CoupledOpenOverflowContext:
    """Fixed data for the mass-wall to open-edge continuation."""

    base: CoupledInnerOuterContext
    boundary_fraction: float
    mass_flux_scale: float
    torque_scale: float
    mdot_ratio_bounds: tuple[float, float] = (1.0e-3, 1.0)

    def __post_init__(self) -> None:
        if not 0.0 <= self.boundary_fraction <= 1.0:
            raise ValueError("boundary_fraction must lie in [0,1]")
        if self.mass_flux_scale <= 0.0 or self.torque_scale <= 0.0:
            raise ValueError("open-overflow scales must be positive")
        lower, upper = self.mdot_ratio_bounds
        if not 0.0 < lower < upper:
            raise ValueError("mdot_ratio_bounds must be positive and ordered")
        if self.base.coupling_fraction != 1.0:
            raise ValueError("open overflow requires the fully coupled base")
        if self.base.wall_pattern_power_fraction != 0.0:
            raise ValueError("open-overflow control must not include wall power")


@dataclass(frozen=True)
class CoupledOpenOverflowEvaluation:
    """Named residual and physical edge data for one augmented state."""

    residual: np.ndarray
    base: CoupledResidualEvaluation
    edge_boundary: float
    mdot_inner: float
    mdot_outer: float
    outer_torque: float
    trial_context: CoupledInnerOuterContext


@dataclass(frozen=True)
class CoupledOpenOverflowResult:
    """Nonlinear result for one fixed boundary continuation fraction."""

    state: np.ndarray
    evaluation: CoupledOpenOverflowEvaluation
    accepted: bool
    nfev: int
    maximum_residual: float
    message: str


@dataclass(frozen=True)
class CoupledOpenRankAudit:
    """Full-system and interface/sonic rank diagnostics."""

    jacobian_shape: tuple[int, int]
    singular_values: np.ndarray
    ranks_by_relative_threshold: dict[str, int]
    condition_estimate: float
    preboundary_nullity: int
    interface_response_singular_values: np.ndarray
    interface_response_rank: int
    sonic_singular_values: np.ndarray
    sonic_rank: int


def coupled_open_state_size(context: CoupledOpenOverflowContext) -> int:
    """Return one more than the square base coupled state size."""

    return context.base.inner_params.n_nodes * 2 + (
        context.base.outer_grid.centers.size * 3
    ) + 5


def pack_coupled_open_state(
    base_state,
    mdot_inner: float,
    context: CoupledOpenOverflowContext,
) -> np.ndarray:
    """Append ``log(Mdot_inner/mass_flux_scale)`` to a base state."""

    base_state = np.asarray(base_state, dtype=float)
    if base_state.shape != (coupled_open_state_size(context) - 1,):
        raise ValueError("base coupled state has the wrong size")
    if not np.isfinite(mdot_inner) or mdot_inner <= 0.0:
        raise ValueError("mdot_inner must be positive and finite")
    return np.concatenate(
        (
            base_state,
            [np.log(float(mdot_inner) / context.mass_flux_scale)],
        )
    )


def unpack_coupled_open_state(
    state,
    context: CoupledOpenOverflowContext,
) -> tuple[np.ndarray, float]:
    """Return the base state and physical positive inner accretion rate."""

    state = np.asarray(state, dtype=float)
    if state.shape != (coupled_open_state_size(context),):
        raise ValueError("open-overflow state has the wrong size")
    return (
        np.asarray(state[:-1], dtype=float),
        float(context.mass_flux_scale * np.exp(state[-1])),
    )


def _trial_context(
    base_state,
    mdot_inner: float,
    context: CoupledOpenOverflowContext,
) -> CoupledInnerOuterContext:
    """Shift all mass faces while preserving exact source divergence."""

    template = context.base.outer_template
    delta = float(mdot_inner) - float(template.mdot_faces[0])
    shifted_template = replace(
        template,
        mdot_faces=np.asarray(template.mdot_faces + delta, dtype=float),
    )
    params = replace(context.base.inner_params, Mdot_g_s=float(mdot_inner))
    return replace(
        context.base,
        inner_params=params,
        outer_template=shifted_template,
    )


def evaluate_coupled_open_overflow_residual(
    state,
    context: CoupledOpenOverflowContext,
    *,
    include_inner_profile: bool = True,
) -> CoupledOpenOverflowEvaluation:
    """Evaluate the base equations plus one wall-to-open edge row."""

    base_state, mdot_inner = unpack_coupled_open_state(state, context)
    trial_context = _trial_context(base_state, mdot_inner, context)
    base = evaluate_coupled_inner_outer_residual(
        base_state,
        trial_context,
        include_inner_profile=include_inner_profile,
    )
    mdot_outer = float(base.outer_transport.mdot_faces[-1])
    outer_torque = float(base.outer_transport.viscous_torque_faces[-1])
    fraction = float(context.boundary_fraction)
    edge = (
        (1.0 - fraction) * mdot_outer / context.mass_flux_scale
        + fraction * outer_torque / context.torque_scale
    )
    residual = np.concatenate((base.residual, [edge]))
    if residual.shape != (coupled_open_state_size(context),):
        raise RuntimeError("open-overflow residual is not square")
    return CoupledOpenOverflowEvaluation(
        residual=np.asarray(residual, dtype=float),
        base=base,
        edge_boundary=float(edge),
        mdot_inner=mdot_inner,
        mdot_outer=mdot_outer,
        outer_torque=outer_torque,
        trial_context=trial_context,
    )


def coupled_open_state_bounds(
    context: CoupledOpenOverflowContext,
    seed_state,
) -> tuple[np.ndarray, np.ndarray]:
    """Return base physical bounds plus the declared Mdot-ratio bounds."""

    base_state, mdot_inner = unpack_coupled_open_state(seed_state, context)
    trial_context = _trial_context(base_state, mdot_inner, context)
    lower, upper = coupled_state_bounds(trial_context, base_state)
    ratio_lower, ratio_upper = context.mdot_ratio_bounds
    return (
        np.concatenate((lower, [np.log(ratio_lower)])),
        np.concatenate((upper, [np.log(ratio_upper)])),
    )


def _bounded_difference_step(value: float, lower: float, upper: float) -> float:
    step = 1.0e-6 * max(1.0, abs(value))
    if value + step <= upper:
        return step
    if value - step >= lower:
        return -step
    return 0.5 * (upper - lower)


def coupled_open_jacobian(
    state,
    context: CoupledOpenOverflowContext,
):
    """Return the base colored Jacobian with one dense Mdot column."""

    state = np.asarray(state, dtype=float)
    base_state, mdot_inner = unpack_coupled_open_state(state, context)
    trial_context = _trial_context(base_state, mdot_inner, context)
    base_jacobian = colored_coupled_jacobian(base_state, trial_context)
    size = state.size
    jacobian = lil_matrix((size, size), dtype=float)
    jacobian[:-1, :-1] = base_jacobian
    lower, upper = coupled_open_state_bounds(context, state)
    base_evaluation = evaluate_coupled_open_overflow_residual(
        state,
        context,
        include_inner_profile=False,
    )

    mdot_step = _bounded_difference_step(state[-1], lower[-1], upper[-1])
    mdot_trial = np.array(state, copy=True)
    mdot_trial[-1] += mdot_step
    mdot_residual = evaluate_coupled_open_overflow_residual(
        mdot_trial,
        context,
        include_inner_profile=False,
    ).residual
    jacobian[:, -1] = (
        (mdot_residual - base_evaluation.residual) / mdot_step
    )[:, None]

    ni = context.base.inner_params.n_nodes
    no = context.base.outer_grid.centers.size
    inner_size = 2 * ni + 2
    omega_start = inner_size + 2 * no
    edge_columns = [size - 3]
    edge_columns.extend(
        range(omega_start + max(0, no - 2), omega_start + no)
    )
    for column in edge_columns:
        step = _bounded_difference_step(state[column], lower[column], upper[column])
        trial = np.array(state, copy=True)
        trial[column] += step
        changed = evaluate_coupled_open_overflow_residual(
            trial,
            context,
            include_inner_profile=False,
        ).edge_boundary
        jacobian[-1, column] = (
            changed - base_evaluation.edge_boundary
        ) / step
    return jacobian.tocsr()


def solve_coupled_open_overflow_steady(
    initial_state,
    context: CoupledOpenOverflowContext,
    *,
    tolerance: float = 1.0e-7,
    max_nfev: int = 1000,
) -> CoupledOpenOverflowResult:
    """Solve one fixed wall-to-open continuation stage."""

    state = np.asarray(initial_state, dtype=float)
    lower, upper = coupled_open_state_bounds(context, state)
    state = np.clip(state, lower + 1.0e-12, upper - 1.0e-12)
    nfev = 0
    message = "maximum Newton iterations reached"
    for _iteration in range(int(max_nfev)):
        evaluation = evaluate_coupled_open_overflow_residual(
            state,
            context,
            include_inner_profile=False,
        )
        nfev += 1
        residual = evaluation.residual
        maximum = float(np.max(np.abs(residual)))
        if maximum <= tolerance:
            message = "residual tolerance reached"
            break
        jacobian = coupled_open_jacobian(state, context)
        direction = np.asarray(spsolve(jacobian, -residual), dtype=float)
        if np.any(~np.isfinite(direction)):
            message = "non-finite Newton direction"
            break
        norm = float(np.linalg.norm(residual))
        accepted_step = False
        for backtrack in range(20):
            step = 0.5**backtrack
            candidate = np.clip(
                state + step * direction,
                lower + 1.0e-12,
                upper - 1.0e-12,
            )
            candidate_residual = evaluate_coupled_open_overflow_residual(
                candidate,
                context,
                include_inner_profile=False,
            ).residual
            nfev += 1
            if np.linalg.norm(candidate_residual) < norm * (
                1.0 - 1.0e-4 * step
            ):
                state = candidate
                accepted_step = True
                break
        if not accepted_step:
            message = "Newton line search failed"
            break
    evaluation = evaluate_coupled_open_overflow_residual(state, context)
    maximum = float(np.max(np.abs(evaluation.residual)))
    return CoupledOpenOverflowResult(
        state=np.asarray(state, dtype=float),
        evaluation=evaluation,
        accepted=bool(maximum <= tolerance),
        nfev=nfev,
        maximum_residual=maximum,
        message=message,
    )


def audit_coupled_open_rank(
    state,
    context: CoupledOpenOverflowContext,
) -> CoupledOpenRankAudit:
    """Audit full rank and the retained interface/sonic responses."""

    state = np.asarray(state, dtype=float)
    jacobian = coupled_open_jacobian(state, context).toarray()
    singular = np.linalg.svd(jacobian, compute_uv=False)
    largest = max(float(singular[0]), 1.0e-300)
    ranks = {
        f"{threshold:.0e}": int(np.sum(singular / largest > threshold))
        for threshold in (1.0e-8, 1.0e-10, 1.0e-12)
    }
    rows = coupled_row_slices(context.base)
    boundary = rows["interface_boundary"]
    removed = np.arange(boundary.start, boundary.stop)
    preboundary = np.delete(jacobian, removed, axis=0)
    _left, pre_singular, pre_right = np.linalg.svd(
        preboundary,
        full_matrices=True,
    )
    pre_largest = max(float(pre_singular[0]), 1.0e-300)
    pre_rank = int(np.sum(pre_singular / pre_largest > 1.0e-10))
    nullity = int(jacobian.shape[1] - pre_rank)
    null_basis = pre_right[-2:, :].T
    response = jacobian[boundary, :] @ null_basis
    response_singular = np.linalg.svd(response, compute_uv=False)
    response_rank = int(
        np.sum(
            response_singular
            / max(float(response_singular[0]), 1.0e-300)
            > 1.0e-10
        )
    )

    base_state, mdot_inner = unpack_coupled_open_state(state, context)
    trial_context = _trial_context(base_state, mdot_inner, context)
    inner_state, *_rest = unpack_coupled_state(base_state, trial_context)
    sonic_components = (
        "D",
        (
            "K"
            if trial_context.sonic_pivot in {"K", "svd"}
            else trial_context.sonic_pivot
        ),
    )
    sonic_jacobian = sonic_residual_jacobian(
        inner_state,
        trial_context.inner_params,
        components=sonic_components,
        rel_step=1.0e-6,
    )
    sonic_singular = np.linalg.svd(sonic_jacobian, compute_uv=False)
    sonic_rank = int(
        np.sum(
            sonic_singular
            / max(float(sonic_singular[0]), 1.0e-300)
            > 1.0e-10
        )
    )
    return CoupledOpenRankAudit(
        jacobian_shape=jacobian.shape,
        singular_values=np.asarray(singular, dtype=float),
        ranks_by_relative_threshold=ranks,
        condition_estimate=float(singular[0] / singular[-1]),
        preboundary_nullity=nullity,
        interface_response_singular_values=np.asarray(
            response_singular,
            dtype=float,
        ),
        interface_response_rank=response_rank,
        sonic_singular_values=np.asarray(sonic_singular, dtype=float),
        sonic_rank=sonic_rank,
    )
