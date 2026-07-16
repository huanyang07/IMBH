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
    pack_coupled_state,
    unpack_coupled_state,
)
from .interface_flux import ConservedInterfaceFlux
from .signed_flux_common_stress import build_nonkeplerian_residual_scales
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


@dataclass(frozen=True)
class CoupledSupplyContinuationStage:
    """One fixed supply stage on the open-overflow branch."""

    supply_fraction: float
    accepted: bool
    maximum_residual: float
    nfev: int
    message: str
    mdot_inner_over_initial_supply: float
    mdot_inner_over_stage_supply: float
    mdot_outer_over_initial_supply: float
    mdot_outer_over_stage_supply: float


@dataclass(frozen=True)
class CoupledSupplyContinuationResult:
    """Final context, state, and immutable records for a supply continuation."""

    context: CoupledOpenOverflowContext
    state: np.ndarray
    stages: tuple[CoupledSupplyContinuationStage, ...]
    accepted: bool


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


def rescale_coupled_open_supply(
    state,
    context: CoupledOpenOverflowContext,
    supply_factor: float,
) -> tuple[CoupledOpenOverflowContext, np.ndarray]:
    """Scale one no-tide open root's complete stream ledger for continuation."""

    factor = float(supply_factor)
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError("supply_factor must be positive and finite")
    template = context.base.outer_template
    if np.any(np.asarray(template.external_angular_rate_cells) != 0.0):
        raise ValueError("supply rescaling requires zero external torque")
    base_state, mdot_inner = unpack_coupled_open_state(state, context)
    (
        inner_state,
        sigma,
        temperature,
        omega,
        interface_angular,
        interface_energy,
    ) = unpack_coupled_state(base_state, context.base)

    scaled_template = replace(
        template,
        viscous_torque_centers=np.asarray(
            factor * template.viscous_torque_centers, dtype=float
        ),
        viscous_torque_faces=np.asarray(
            factor * template.viscous_torque_faces, dtype=float
        ),
        mdot_faces=np.asarray(factor * template.mdot_faces, dtype=float),
        angular_flux_faces=np.asarray(
            factor * template.angular_flux_faces, dtype=float
        ),
        source_mass_rate_cells=np.asarray(
            factor * template.source_mass_rate_cells, dtype=float
        ),
        source_angular_rate_cells=np.asarray(
            factor * template.source_angular_rate_cells, dtype=float
        ),
        source_total_energy_rate_cells=np.asarray(
            factor * template.source_total_energy_rate_cells, dtype=float
        ),
        mass_rate_cells=np.asarray(
            factor * template.mass_rate_cells, dtype=float
        ),
        mass_budget_rate=float(factor * template.mass_budget_rate),
        angular_momentum_rate_from_state=float(
            factor * template.angular_momentum_rate_from_state
        ),
        angular_momentum_budget_rate=float(
            factor * template.angular_momentum_budget_rate
        ),
        angular_momentum_budget_defect=float(
            factor * template.angular_momentum_budget_defect
        ),
    )
    scaled_mdot = factor * mdot_inner
    scaled_angular = factor * interface_angular
    scaled_energy = factor * interface_energy
    scaled_interface = ConservedInterfaceFlux(
        mdot=scaled_mdot,
        angular_momentum=scaled_angular,
        total_energy=scaled_energy,
    )
    shifted_template = replace(
        scaled_template,
        mdot_faces=np.asarray(
            scaled_template.mdot_faces
            + (scaled_mdot - float(scaled_template.mdot_faces[0])),
            dtype=float,
        ),
    )
    scaled_base = replace(
        context.base,
        inner_params=replace(
            context.base.inner_params,
            Mdot_g_s=scaled_mdot,
        ),
        outer_template=scaled_template,
        angular_flux_scale=factor * context.base.angular_flux_scale,
        energy_flux_scale=factor * context.base.energy_flux_scale,
    )
    scaled_scales = build_nonkeplerian_residual_scales(
        scaled_base.outer_grid,
        shifted_template,
        sigma,
        temperature,
        omega,
        scaled_base.inner_params.M2_g,
        closure=scaled_base.outer_closure,
        prescribed_inner_flux=scaled_interface,
    )
    scaled_base = replace(scaled_base, outer_scales=scaled_scales)
    scaled_context = replace(
        context,
        base=scaled_base,
        mass_flux_scale=factor * context.mass_flux_scale,
        torque_scale=factor * context.torque_scale,
    )
    scaled_base_state = pack_coupled_state(
        inner_state,
        sigma,
        temperature,
        omega,
        scaled_angular,
        scaled_energy,
        scaled_base,
    )
    scaled_state = pack_coupled_open_state(
        scaled_base_state,
        scaled_mdot,
        scaled_context,
    )
    return scaled_context, scaled_state


def continue_coupled_open_supply(
    state,
    context: CoupledOpenOverflowContext,
    supply_fractions,
    *,
    tolerance: float = 1.0e-7,
    max_nfev: int = 200,
) -> CoupledSupplyContinuationResult:
    """Continue an accepted open root through fixed decreasing supply stages."""

    fractions = tuple(float(value) for value in supply_fractions)
    if not fractions:
        raise ValueError("supply continuation needs at least one target")
    if any(not np.isfinite(value) or value <= 0.0 for value in fractions):
        raise ValueError("supply fractions must be positive and finite")
    if any(right >= left for left, right in zip((1.0,) + fractions, fractions)):
        raise ValueError("supply fractions must decrease strictly from one")
    initial_scale = float(context.mass_flux_scale)
    current_fraction = 1.0
    current_context = context
    current_state = np.asarray(state, dtype=float)
    stages = []
    for target_fraction in fractions:
        stage_factor = target_fraction / current_fraction
        trial_context, trial_state = rescale_coupled_open_supply(
            current_state,
            current_context,
            stage_factor,
        )
        result = solve_coupled_open_overflow_steady(
            trial_state,
            trial_context,
            tolerance=tolerance,
            max_nfev=max_nfev,
        )
        evaluation = result.evaluation
        stages.append(
            CoupledSupplyContinuationStage(
                supply_fraction=target_fraction,
                accepted=result.accepted,
                maximum_residual=result.maximum_residual,
                nfev=result.nfev,
                message=result.message,
                mdot_inner_over_initial_supply=(
                    evaluation.mdot_inner / initial_scale
                ),
                mdot_inner_over_stage_supply=(
                    evaluation.mdot_inner / trial_context.mass_flux_scale
                ),
                mdot_outer_over_initial_supply=(
                    evaluation.mdot_outer / initial_scale
                ),
                mdot_outer_over_stage_supply=(
                    evaluation.mdot_outer / trial_context.mass_flux_scale
                ),
            )
        )
        current_context = trial_context
        current_state = result.state
        current_fraction = target_fraction
        if not result.accepted:
            break
    return CoupledSupplyContinuationResult(
        context=current_context,
        state=current_state,
        stages=tuple(stages),
        accepted=bool(
            len(stages) == len(fractions)
            and stages[-1].accepted
            and current_fraction == fractions[-1]
        ),
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
