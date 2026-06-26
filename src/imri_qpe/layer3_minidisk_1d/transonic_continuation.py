"""Continuation helpers for the isolated transonic slim-disk solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .transonic_collocation import (
    TransonicSlimParams,
    TransonicResidualAudit,
    TransonicSlimProfile,
    TransonicSolveResult,
    TransonicSolveStatus,
    _status_from_profile,
    collocation_jacobian,
    collocation_residual,
    computational_grid,
    jac_sparsity_pattern,
    pack_state,
    profile_from_state_vector,
    replace_mdot,
    residual_audit_from_state_vector,
    select_sonic_compatibility_pivot,
    solve_transonic_outer_branch,
    square_collocation_jacobian,
    square_collocation_residual,
    square_jac_sparsity_pattern,
    state_bounds,
)


@dataclass(frozen=True)
class TransonicContinuationResult:
    """Sequence of transonic solves along accretion-rate continuation."""

    results: tuple[TransonicSolveResult, ...]
    mdot_values: np.ndarray


@dataclass(frozen=True)
class ContinuationMetric:
    """Blockwise pseudo-arclength metric scales."""

    n_nodes: int
    logu_scale: float
    logT_scale: float
    logR_son_scale: float
    lambda0_scale: float
    mu_scale: float

    def scale_vector(self) -> np.ndarray:
        n = int(self.n_nodes)
        return np.concatenate(
            [
                np.full(n, np.sqrt(float(n)) * self.logu_scale),
                np.full(n, np.sqrt(float(n)) * self.logT_scale),
                np.array([self.logR_son_scale, self.lambda0_scale, self.mu_scale]),
            ]
        )


@dataclass(frozen=True)
class ContinuationTangentAudit:
    """Composition of the scaled continuation tangent."""

    method: str
    logu_fraction: float
    logT_fraction: float
    logR_son_fraction: float
    lambda0_fraction: float
    mu_fraction: float
    dmu_ds: float


@dataclass(frozen=True)
class TransonicPseudoArclengthResult:
    """One pseudo-arclength corrector step for the transonic branch."""

    z: np.ndarray
    mdot_ratio: float
    predicted_mdot_ratio: float
    params: TransonicSlimParams
    profile: TransonicSlimProfile
    residual_audit: TransonicResidualAudit
    status: TransonicSolveStatus
    max_residual: float
    arclength_residual: float
    arclength_step: float
    tangent: np.ndarray
    metric: ContinuationMetric
    tangent_audit: ContinuationTangentAudit
    cost: float
    nfev: int
    optimizer_success: bool
    message: str
    corrector_method: str = "least_squares"
    corrector_iterations: int = 0
    jacobian_evaluations: int = 0
    line_search_reductions: int = 0
    condition_estimate: float = np.nan
    initial_max_residual: float = np.nan
    predictor_correction_norm: float = np.nan


@dataclass(frozen=True)
class _BorderedNewtonCorrectorResult:
    """Internal result from a scaled bordered Newton pseudo-arclength corrector."""

    w: np.ndarray
    success: bool
    cost: float
    optimality: float
    nfev: int
    njev: int
    iterations: int
    line_search_reductions: int
    condition_estimate: float
    message: str


def _state_from_profile(profile) -> np.ndarray:
    return pack_state(np.log(profile.u), np.log(profile.T), np.log(profile.sonic_radius), profile.lambda0)


def legacy_continuation_metric(params: TransonicSlimParams) -> ContinuationMetric:
    """Return the old uniform state scaling as a metric object."""

    state_scale = np.sqrt(float(2 * params.n_nodes + 2))
    block_scale = float(state_scale / np.sqrt(float(params.n_nodes)))
    return ContinuationMetric(
        n_nodes=params.n_nodes,
        logu_scale=block_scale,
        logT_scale=block_scale,
        logR_son_scale=float(state_scale),
        lambda0_scale=float(state_scale),
        mu_scale=1.0,
    )


def blockwise_continuation_metric(
    z_previous,
    previous_mdot_ratio: float,
    z_current,
    current_mdot_ratio: float,
    params: TransonicSlimParams,
    *,
    logu_floor: float = 2.0e-2,
    logT_floor: float = 1.0e-2,
    logR_son_floor: float = 1.0e-2,
    lambda0_floor: float = 1.0e-2,
    mu_floor: float = 2.0e-2,
) -> ContinuationMetric:
    """Build the blockwise metric from the latest accepted secant."""

    if previous_mdot_ratio <= 0.0 or current_mdot_ratio <= 0.0:
        raise ValueError("mdot ratios must be positive")
    z_previous = np.asarray(z_previous, dtype=float)
    z_current = np.asarray(z_current, dtype=float)
    if z_previous.shape != z_current.shape:
        raise ValueError("state vectors must have the same shape")
    n = params.n_nodes
    dlogu = z_current[:n] - z_previous[:n]
    dlogT = z_current[n : 2 * n] - z_previous[n : 2 * n]
    dlogR_son = float(z_current[2 * n] - z_previous[2 * n])
    dlambda0 = float(z_current[2 * n + 1] - z_previous[2 * n + 1])
    dmu = float(np.log(current_mdot_ratio / previous_mdot_ratio))
    return ContinuationMetric(
        n_nodes=n,
        logu_scale=float(max(np.sqrt(np.mean(dlogu**2)), logu_floor)),
        logT_scale=float(max(np.sqrt(np.mean(dlogT**2)), logT_floor)),
        logR_son_scale=float(max(abs(dlogR_son), logR_son_floor)),
        lambda0_scale=float(max(abs(dlambda0), lambda0_floor)),
        mu_scale=float(max(abs(dmu), mu_floor)),
    )


def _scaled_secant_tangent(w_previous: np.ndarray, w_current: np.ndarray, scales: np.ndarray) -> tuple[np.ndarray, float]:
    delta_scaled = (w_current - w_previous) / scales
    tangent_norm = float(np.linalg.norm(delta_scaled))
    if not np.isfinite(tangent_norm) or tangent_norm <= 0.0:
        raise ValueError("previous and current branch points must be distinct")
    return delta_scaled / tangent_norm, tangent_norm


def tangent_audit_from_scaled_tangent(
    tangent: np.ndarray,
    metric: ContinuationMetric,
    *,
    method: str,
) -> ContinuationTangentAudit:
    """Return block fractions for a scaled metric-normalized tangent."""

    tangent = np.asarray(tangent, dtype=float)
    n = metric.n_nodes
    norm2 = float(np.dot(tangent, tangent))
    if norm2 <= 0.0 or not np.isfinite(norm2):
        norm2 = np.nan

    def frac(values) -> float:
        return float(np.dot(values, values) / norm2) if np.isfinite(norm2) else np.nan

    return ContinuationTangentAudit(
        method=method,
        logu_fraction=frac(tangent[:n]),
        logT_fraction=frac(tangent[n : 2 * n]),
        logR_son_fraction=frac(tangent[2 * n : 2 * n + 1]),
        lambda0_fraction=frac(tangent[2 * n + 1 : 2 * n + 2]),
        mu_fraction=frac(tangent[2 * n + 2 : 2 * n + 3]),
        dmu_ds=float(metric.mu_scale * tangent[2 * n + 2]),
    )


def remap_profile_to_new_sonic_grid(profile, new_params: TransonicSlimParams, temperature_mdot_power: float = 0.25) -> np.ndarray:
    """Map a converged profile onto the grid and accretion rate of ``new_params``."""

    old_mdot = float(np.median(2.0 * np.pi * profile.R * profile.Sigma * profile.u))
    if old_mdot <= 0.0 or not np.isfinite(old_mdot):
        raise ValueError("profile must have a positive accretion rate")
    mdot_factor = new_params.Mdot_g_s / old_mdot
    logR_son = float(np.log(profile.sonic_radius))
    logR_new = computational_grid(new_params, logR_son)
    logR_old = np.log(profile.R)
    logu = np.interp(logR_new, logR_old, np.log(profile.u), left=np.log(profile.u[0]), right=np.log(profile.u[-1]))
    logT = np.interp(logR_new, logR_old, np.log(profile.T), left=np.log(profile.T[0]), right=np.log(profile.T[-1]))
    logT = logT + temperature_mdot_power * np.log(mdot_factor)
    logu = np.clip(logu, new_params.logu_bounds[0], new_params.logu_bounds[1])
    logT = np.clip(logT, new_params.logT_bounds[0], new_params.logT_bounds[1])
    return pack_state(logu, logT, logR_son, profile.lambda0)


def _resolve_continuation_pivot(z, params: TransonicSlimParams, pivot: str) -> str:
    if pivot == "auto":
        return select_sonic_compatibility_pivot(z, params)
    if pivot == "svd":
        return "K"
    if pivot not in {"C1", "C2", "K"}:
        raise ValueError("sonic_pivot must be 'auto', 'svd', 'K', 'C1', or 'C2'")
    return pivot


def _continuation_equation_residual(
    z,
    params: TransonicSlimParams,
    *,
    residual_mode: str,
    sonic_pivot: str,
) -> np.ndarray:
    if residual_mode == "full":
        return collocation_residual(z, params)
    if residual_mode == "square":
        return square_collocation_residual(z, params, pivot=sonic_pivot)
    raise ValueError("residual_mode must be 'square' or 'full'")


def _pseudo_arclength_sparsity_pattern(params: TransonicSlimParams, residual_mode: str = "square"):
    try:
        from scipy.sparse import lil_matrix
    except Exception:
        return None

    if residual_mode == "square":
        base = square_jac_sparsity_pattern(params)
    elif residual_mode == "full":
        base = jac_sparsity_pattern(params)
    else:
        raise ValueError("residual_mode must be 'square' or 'full'")
    if base is None:
        return None
    n_residual, n_state = base.shape
    pattern = lil_matrix((n_residual + 1, n_state + 1), dtype=int)
    pattern[:n_residual, :n_state] = base
    pattern[:n_residual, n_state] = 1
    pattern[n_residual, :] = 1
    return pattern.tocsr()


def _continuation_top_jacobian(
    w,
    base_params: TransonicSlimParams,
    mdot_unit_g_s: float,
    residual_mode: str,
    sonic_pivot: str,
    rel_step: float = 1.0e-5,
):
    try:
        from scipy.sparse import csr_matrix, hstack
    except Exception as exc:
        raise RuntimeError("scipy is required for continuation jacobian") from exc

    z = np.asarray(w[:-1], dtype=float)
    mu = float(w[-1])
    params = replace_mdot(base_params, float(np.exp(mu)) * mdot_unit_g_s)
    if residual_mode == "square":
        state_jac = square_collocation_jacobian(z, params, pivot=sonic_pivot)
        base = square_collocation_residual(z, params, pivot=sonic_pivot)
    elif residual_mode == "full":
        state_jac = collocation_jacobian(z, params)
        base = collocation_residual(z, params)
    else:
        raise ValueError("residual_mode must be 'square' or 'full'")
    step = rel_step * max(1.0, abs(mu))
    if step <= 0.0:
        mdot_column = np.zeros_like(base)
    else:
        plus_params = replace_mdot(base_params, float(np.exp(mu + step)) * mdot_unit_g_s)
        minus_params = replace_mdot(base_params, float(np.exp(mu - step)) * mdot_unit_g_s)
        plus = _continuation_equation_residual(
            z,
            plus_params,
            residual_mode=residual_mode,
            sonic_pivot=sonic_pivot,
        )
        minus = _continuation_equation_residual(
            z,
            minus_params,
            residual_mode=residual_mode,
            sonic_pivot=sonic_pivot,
        )
        mdot_column = (plus - minus) / (2.0 * step)
    return hstack([state_jac, csr_matrix(mdot_column[:, None])], format="csr")


def _jacobian_scaled_tangent(
    w_current: np.ndarray,
    base_params: TransonicSlimParams,
    mdot_unit_g_s: float,
    scales: np.ndarray,
    reference_tangent: np.ndarray,
    residual_mode: str,
    sonic_pivot: str,
    rel_step: float = 1.0e-5,
) -> np.ndarray:
    """Solve the bordered tangent equation in scaled coordinates."""

    try:
        from scipy.sparse import csr_matrix, diags, vstack
        from scipy.sparse.linalg import lsmr, spsolve
    except Exception as exc:
        raise RuntimeError("scipy is required for continuation tangent solve") from exc

    top = _continuation_top_jacobian(
        w_current,
        base_params,
        mdot_unit_g_s,
        residual_mode,
        sonic_pivot,
        rel_step=rel_step,
    )
    scaled_top = top @ diags(scales)
    reference = np.asarray(reference_tangent, dtype=float)
    border = csr_matrix(reference[None, :])
    matrix = vstack([scaled_top, border], format="csc")
    rhs = np.zeros(matrix.shape[0], dtype=float)
    rhs[-1] = 1.0
    try:
        tangent = np.asarray(spsolve(matrix, rhs), dtype=float)
    except Exception:
        tangent = np.linalg.lstsq(matrix.toarray(), rhs, rcond=None)[0]
    norm = float(np.linalg.norm(tangent))
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("Jacobian tangent solve returned a non-finite tangent")
    tangent = tangent / norm
    if float(np.dot(tangent, reference)) < 0.0:
        tangent = -tangent
    if abs(float(tangent[-1])) < max(1.0e-4, 1.0e-2 * abs(float(reference[-1]))):
        method = "jacobian_mu"
        state_top = scaled_top[:, :-1]
        mu_column = np.asarray(scaled_top[:, -1].todense()).ravel()
        mu_sign = 1.0 if float(reference[-1]) >= 0.0 else -1.0
        rhs_mu = -mu_sign * mu_column
        for damping in (1.0e-2, 1.0e-3, 1.0e-1, 1.0):
            state_tangent = lsmr(
                state_top,
                rhs_mu,
                damp=damping,
                atol=1.0e-10,
                btol=1.0e-10,
                maxiter=max(20, 5 * state_top.shape[1]),
            )[0]
            candidate = np.concatenate([np.asarray(state_tangent, dtype=float), np.array([mu_sign])])
            candidate_norm = float(np.linalg.norm(candidate))
            if np.isfinite(candidate_norm) and candidate_norm > 0.0:
                candidate = candidate / candidate_norm
                if float(np.dot(candidate, reference)) < 0.0:
                    candidate = -candidate
                if abs(float(candidate[-1])) > abs(float(tangent[-1])):
                    tangent = candidate
                    break
        else:
            method = "jacobian"
    else:
        method = "jacobian"
    return tangent, method


def _pseudo_arclength_jacobian(
    w,
    base_params: TransonicSlimParams,
    mdot_unit_g_s: float,
    scales: np.ndarray,
    tangent: np.ndarray,
    arclength_weight: float,
    residual_mode: str,
    sonic_pivot: str,
    rel_step: float = 1.0e-5,
):
    try:
        from scipy.sparse import csr_matrix, vstack
    except Exception as exc:
        raise RuntimeError("scipy is required for pseudo-arclength jacobian") from exc

    top = _continuation_top_jacobian(
        w,
        base_params,
        mdot_unit_g_s,
        residual_mode,
        sonic_pivot,
        rel_step=rel_step,
    )
    arc_row = csr_matrix((arclength_weight * tangent / scales)[None, :])
    return vstack([top, arc_row], format="csr")


def _continuation_residual_vector(
    w,
    w_predicted: np.ndarray,
    base_params: TransonicSlimParams,
    mdot_unit_g_s: float,
    scales: np.ndarray,
    tangent: np.ndarray,
    arclength_weight: float,
    residual_mode: str,
    sonic_pivot: str,
) -> np.ndarray:
    """Return the bordered pseudo-arclength residual ``[F, g]``."""

    ratio = float(np.exp(float(w[-1])))
    params = replace_mdot(base_params, ratio * mdot_unit_g_s)
    equation_residual = _continuation_equation_residual(
        w[:-1],
        params,
        residual_mode=residual_mode,
        sonic_pivot=sonic_pivot,
    )
    arclength_residual = float(np.dot((np.asarray(w, dtype=float) - w_predicted) / scales, tangent))
    return np.concatenate([equation_residual, np.array([arclength_weight * arclength_residual])])


def _continuation_merit(residual: np.ndarray) -> float:
    """Return the least-squares merit for a bordered residual vector."""

    residual = np.asarray(residual, dtype=float)
    return 0.5 * float(np.dot(residual, residual))


def _max_alpha_inside_bounds(w: np.ndarray, step: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Return the largest step fraction that remains inside simple bounds."""

    alpha = 1.0
    positive = step > 0.0
    if np.any(positive):
        alpha = min(alpha, float(np.min((upper[positive] - w[positive]) / step[positive])))
    negative = step < 0.0
    if np.any(negative):
        alpha = min(alpha, float(np.min((lower[negative] - w[negative]) / step[negative])))
    if not np.isfinite(alpha):
        return 0.0
    return max(0.0, min(1.0, 0.999 * alpha))


def _condition_estimate(matrix, max_dense_size: int = 260) -> float:
    """Return a dense condition estimate for moderate bordered matrices."""

    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] > max_dense_size:
        return np.nan
    try:
        return float(np.linalg.cond(matrix.toarray()))
    except Exception:
        return np.nan


def _solve_bordered_scaled_step(matrix, residual: np.ndarray, condition_estimate: float) -> np.ndarray:
    """Solve the scaled bordered Newton system."""

    try:
        from scipy.sparse.linalg import lsmr, splu
    except Exception as exc:
        raise RuntimeError("scipy is required for bordered Newton") from exc

    rhs = -np.asarray(residual, dtype=float)
    use_direct = bool(np.isfinite(condition_estimate) and condition_estimate < 1.0e8)
    if use_direct:
        try:
            return np.asarray(splu(matrix.tocsc()).solve(rhs), dtype=float)
        except Exception:
            pass
    step = lsmr(
        matrix,
        rhs,
        damp=1.0e-8,
        atol=1.0e-10,
        btol=1.0e-10,
        maxiter=max(40, 5 * matrix.shape[1]),
    )[0]
    return np.asarray(step, dtype=float)


def _bordered_newton_corrector(
    w0: np.ndarray,
    w_predicted: np.ndarray,
    base_params: TransonicSlimParams,
    mdot_unit_g_s: float,
    scales: np.ndarray,
    tangent: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    arclength_weight: float,
    residual_mode: str,
    sonic_pivot: str,
    residual_tol: float,
    max_iter: int,
    rel_step: float,
    max_scaled_step_norm: float,
    line_search_min_alpha: float,
    line_search_max_reductions: int,
) -> _BorderedNewtonCorrectorResult:
    """Correct a pseudo-arclength predictor using sparse bordered Newton."""

    try:
        from scipy.sparse import csr_matrix, diags, vstack
    except Exception as exc:
        raise RuntimeError("scipy is required for bordered Newton") from exc

    if residual_mode != "square":
        raise ValueError("bordered Newton requires residual_mode='square'")
    w = np.clip(np.asarray(w0, dtype=float), lower + 1.0e-12, upper - 1.0e-12)
    residual = _continuation_residual_vector(
        w,
        w_predicted,
        base_params,
        mdot_unit_g_s,
        scales,
        tangent,
        arclength_weight,
        residual_mode,
        sonic_pivot,
    )
    nfev = 1
    njev = 0
    reductions_total = 0
    condition = np.nan
    best_w = np.array(w, copy=True)
    best_residual = np.array(residual, copy=True)
    best_merit = _continuation_merit(residual)
    message = "maximum bordered Newton iterations reached"

    for iteration in range(max_iter + 1):
        equation_max = float(np.max(np.abs(residual[:-1])))
        arc_abs = float(abs(residual[-1]))
        if equation_max <= residual_tol and arc_abs <= max(residual_tol, 1.0e-8):
            return _BorderedNewtonCorrectorResult(
                w=w,
                success=True,
                cost=_continuation_merit(residual),
                optimality=max(equation_max, arc_abs),
                nfev=nfev,
                njev=njev,
                iterations=iteration,
                line_search_reductions=reductions_total,
                condition_estimate=condition,
                message="bordered Newton converged",
            )
        if iteration == max_iter:
            break

        top = _continuation_top_jacobian(
            w,
            base_params,
            mdot_unit_g_s,
            residual_mode,
            sonic_pivot,
            rel_step=rel_step,
        )
        scaled_top = top @ diags(scales)
        arc_row = csr_matrix((arclength_weight * tangent)[None, :])
        matrix = vstack([scaled_top, arc_row], format="csr")
        condition = _condition_estimate(matrix)
        njev += 1

        scaled_step = _solve_bordered_scaled_step(matrix, residual, condition)
        step_norm = float(np.linalg.norm(scaled_step, ord=np.inf))
        if not np.isfinite(step_norm) or step_norm <= 0.0:
            message = "bordered Newton linear solve returned a non-finite step"
            break
        if step_norm > max_scaled_step_norm:
            scaled_step = scaled_step * (max_scaled_step_norm / step_norm)
        step = scales * scaled_step
        alpha = _max_alpha_inside_bounds(w, step, lower, upper)
        if alpha < line_search_min_alpha:
            message = "bordered Newton step is blocked by bounds"
            break

        merit = _continuation_merit(residual)
        accepted = False
        reductions = 0
        while alpha >= line_search_min_alpha and reductions <= line_search_max_reductions:
            trial = w + alpha * step
            trial_residual = _continuation_residual_vector(
                trial,
                w_predicted,
                base_params,
                mdot_unit_g_s,
                scales,
                tangent,
                arclength_weight,
                residual_mode,
                sonic_pivot,
            )
            nfev += 1
            trial_merit = _continuation_merit(trial_residual)
            if trial_merit < merit:
                w = trial
                residual = trial_residual
                accepted = True
                if trial_merit < best_merit:
                    best_w = np.array(w, copy=True)
                    best_residual = np.array(residual, copy=True)
                    best_merit = trial_merit
                break
            alpha *= 0.5
            reductions += 1
        reductions_total += reductions
        if not accepted:
            message = "bordered Newton line search failed to reduce the merit"
            break

    final_residual = best_residual
    equation_max = float(np.max(np.abs(final_residual[:-1])))
    arc_abs = float(abs(final_residual[-1]))
    return _BorderedNewtonCorrectorResult(
        w=best_w,
        success=False,
        cost=best_merit,
        optimality=max(equation_max, arc_abs),
        nfev=nfev,
        njev=njev,
        iterations=max_iter,
        line_search_reductions=reductions_total,
        condition_estimate=condition,
        message=message,
    )


def pseudo_arclength_step(
    base_params: TransonicSlimParams,
    mdot_unit_g_s: float,
    previous_profile,
    previous_mdot_ratio: float,
    current_profile,
    current_mdot_ratio: float,
    *,
    step_multiplier: float = 1.0,
    arclength_step: float | None = None,
    mdot_ratio_bounds: tuple[float, float] = (1.0e-6, 10.0),
    max_nfev: int = 1500,
    residual_tol: float | None = None,
    arclength_weight: float = 1.0,
    residual_mode: str = "square",
    sonic_pivot: str = "auto",
    metric_mode: str = "blockwise",
    metric: ContinuationMetric | None = None,
    tangent_mode: str = "jacobian",
    tangent_rel_step: float = 1.0e-5,
    corrector_method: str = "hybrid",
    bordered_max_iter: int | None = None,
    bordered_max_scaled_step_norm: float = 5.0,
    bordered_line_search_min_alpha: float = 1.0e-6,
    bordered_line_search_max_reductions: int = 12,
    verbose: int = 0,
) -> TransonicPseudoArclengthResult:
    """Advance one branch point using a pseudo-arclength corrector.

    The continuation unknown is ``[state_vector, log(Mdot / mdot_unit_g_s)]``.
    The arclength equation is formed in RMS-scaled state coordinates plus the
    unscaled logarithmic accretion-rate coordinate.
    """

    try:
        from scipy.optimize import least_squares
    except Exception as exc:
        raise RuntimeError("scipy is required for pseudo_arclength_step") from exc

    if mdot_unit_g_s <= 0.0:
        raise ValueError("mdot_unit_g_s must be positive")
    if previous_mdot_ratio <= 0.0 or current_mdot_ratio <= 0.0:
        raise ValueError("mdot ratios must be positive")
    if step_multiplier <= 0.0:
        raise ValueError("step_multiplier must be positive")
    if arclength_weight <= 0.0:
        raise ValueError("arclength_weight must be positive")
    if residual_mode not in {"square", "full"}:
        raise ValueError("residual_mode must be 'square' or 'full'")
    if metric_mode not in {"blockwise", "legacy"}:
        raise ValueError("metric_mode must be 'blockwise' or 'legacy'")
    if tangent_mode not in {"jacobian", "secant"}:
        raise ValueError("tangent_mode must be 'jacobian' or 'secant'")
    if corrector_method not in {"hybrid", "bordered_newton", "least_squares"}:
        raise ValueError("corrector_method must be 'hybrid', 'bordered_newton', or 'least_squares'")
    if corrector_method in {"hybrid", "bordered_newton"} and residual_mode != "square":
        raise ValueError("bordered Newton corrector requires residual_mode='square'")
    if bordered_max_scaled_step_norm <= 0.0:
        raise ValueError("bordered_max_scaled_step_norm must be positive")
    if bordered_line_search_min_alpha <= 0.0:
        raise ValueError("bordered_line_search_min_alpha must be positive")
    if bordered_line_search_max_reductions < 0:
        raise ValueError("bordered_line_search_max_reductions must be non-negative")
    if mdot_ratio_bounds[0] <= 0.0 or mdot_ratio_bounds[1] <= mdot_ratio_bounds[0]:
        raise ValueError("mdot_ratio_bounds must be positive and increasing")

    previous_params = replace_mdot(base_params, previous_mdot_ratio * mdot_unit_g_s)
    current_params = replace_mdot(base_params, current_mdot_ratio * mdot_unit_g_s)
    z_previous = remap_profile_to_new_sonic_grid(previous_profile, previous_params)
    z_current = remap_profile_to_new_sonic_grid(current_profile, current_params)
    w_previous = np.concatenate([z_previous, np.array([np.log(previous_mdot_ratio)])])
    w_current = np.concatenate([z_current, np.array([np.log(current_mdot_ratio)])])

    if metric is None:
        metric = (
            blockwise_continuation_metric(
                z_previous,
                previous_mdot_ratio,
                z_current,
                current_mdot_ratio,
                current_params,
            )
            if metric_mode == "blockwise"
            else legacy_continuation_metric(current_params)
        )
    scales = metric.scale_vector()
    secant_tangent, tangent_norm = _scaled_secant_tangent(w_previous, w_current, scales)
    resolved_pivot = _resolve_continuation_pivot(z_current, current_params, sonic_pivot)
    tangent = secant_tangent
    tangent_method = "secant"
    if tangent_mode == "jacobian":
        try:
            tangent, tangent_method = _jacobian_scaled_tangent(
                w_current,
                base_params,
                mdot_unit_g_s,
                scales,
                secant_tangent,
                residual_mode,
                resolved_pivot,
                rel_step=tangent_rel_step,
            )
        except Exception:
            tangent = secant_tangent
            tangent_method = "secant_fallback"

    step = float(tangent_norm * step_multiplier if arclength_step is None else arclength_step)
    if step <= 0.0:
        raise ValueError("arclength step must be positive")
    w_predicted = w_current + step * scales * tangent

    lower_state, upper_state = state_bounds(base_params)
    lower = np.concatenate([lower_state, np.array([np.log(mdot_ratio_bounds[0])])])
    upper = np.concatenate([upper_state, np.array([np.log(mdot_ratio_bounds[1])])])
    w0 = np.clip(w_predicted, lower + 1.0e-12, upper - 1.0e-12)
    residual_tol_value = float(base_params.residual_tol if residual_tol is None else residual_tol)
    initial_residual = _continuation_residual_vector(
        w0,
        w_predicted,
        base_params,
        mdot_unit_g_s,
        scales,
        tangent,
        arclength_weight,
        residual_mode,
        resolved_pivot,
    )
    initial_max_residual = float(np.max(np.abs(initial_residual[:-1])))

    def residual(w):
        return _continuation_residual_vector(
            w,
            w_predicted,
            base_params,
            mdot_unit_g_s,
            scales,
            tangent,
            arclength_weight,
            residual_mode,
            resolved_pivot,
        )

    bordered_result: _BorderedNewtonCorrectorResult | None = None
    w_final = np.array(w0, copy=True)
    corrector_used = "least_squares"
    corrector_iterations = 0
    jacobian_evaluations = 0
    line_search_reductions = 0
    condition_estimate = np.nan
    if corrector_method in {"hybrid", "bordered_newton"}:
        max_bordered_iter = int(min(max_nfev, 30) if bordered_max_iter is None else bordered_max_iter)
        try:
            bordered_result = _bordered_newton_corrector(
                w0,
                w_predicted,
                base_params,
                mdot_unit_g_s,
                scales,
                tangent,
                lower,
                upper,
                arclength_weight=arclength_weight,
                residual_mode=residual_mode,
                sonic_pivot=resolved_pivot,
                residual_tol=residual_tol_value,
                max_iter=max_bordered_iter,
                rel_step=tangent_rel_step,
                max_scaled_step_norm=bordered_max_scaled_step_norm,
                line_search_min_alpha=bordered_line_search_min_alpha,
                line_search_max_reductions=bordered_line_search_max_reductions,
            )
        except Exception as exc:
            if corrector_method == "bordered_newton":
                raise
            bordered_result = _BorderedNewtonCorrectorResult(
                w=w0,
                success=False,
                cost=_continuation_merit(initial_residual),
                optimality=float(np.max(np.abs(initial_residual))),
                nfev=1,
                njev=0,
                iterations=0,
                line_search_reductions=0,
                condition_estimate=np.nan,
                message=f"bordered Newton exception: {exc}",
            )
        if bordered_result.success or corrector_method == "bordered_newton":
            w_final = np.asarray(bordered_result.w, dtype=float)
            cost = float(bordered_result.cost)
            nfev = int(bordered_result.nfev)
            optimizer_success = bool(bordered_result.success)
            message = bordered_result.message
            corrector_used = "bordered_newton"
            corrector_iterations = int(bordered_result.iterations)
            jacobian_evaluations = int(bordered_result.njev)
            line_search_reductions = int(bordered_result.line_search_reductions)
            condition_estimate = float(bordered_result.condition_estimate)
        else:
            w0 = np.clip(np.asarray(bordered_result.w, dtype=float), lower + 1.0e-12, upper - 1.0e-12)

    if corrector_method == "least_squares" or (corrector_method == "hybrid" and (bordered_result is None or not bordered_result.success)):
        result = least_squares(
            residual,
            w0,
            bounds=(lower, upper),
            jac=lambda w: _pseudo_arclength_jacobian(
                w,
                base_params,
                mdot_unit_g_s,
                scales,
                tangent,
                arclength_weight,
                residual_mode,
                resolved_pivot,
            ),
            x_scale="jac",
            ftol=1.0e-10,
            xtol=1.0e-10,
            gtol=1.0e-10 if residual_tol is None else min(1.0e-6, max(1.0e-10, 1.0e-2 * residual_tol)),
            max_nfev=max_nfev,
            verbose=verbose,
        )
        w_final = np.asarray(result.x, dtype=float)
        cost = float(result.cost)
        nfev = int(result.nfev)
        if bordered_result is not None:
            nfev += int(bordered_result.nfev)
        optimizer_success = bool(result.success)
        lsq_message = str(result.message)
        if bordered_result is not None:
            message = f"{bordered_result.message}; least_squares fallback: {lsq_message}"
            corrector_used = "hybrid_least_squares"
            corrector_iterations = int(bordered_result.iterations)
            jacobian_evaluations = int(bordered_result.njev + (-1 if result.njev is None else int(result.njev)))
            line_search_reductions = int(bordered_result.line_search_reductions)
            condition_estimate = float(bordered_result.condition_estimate)
        else:
            message = lsq_message
            corrector_used = "least_squares"
            corrector_iterations = int(result.nfev)
            jacobian_evaluations = -1 if result.njev is None else int(result.njev)

    z = np.asarray(w_final[:-1], dtype=float)
    mdot_ratio = float(np.exp(w_final[-1]))
    params = replace_mdot(base_params, mdot_ratio * mdot_unit_g_s)
    profile = profile_from_state_vector(z, params)
    audit = residual_audit_from_state_vector(z, params)
    equation_residual = _continuation_equation_residual(
        z,
        params,
        residual_mode=residual_mode,
        sonic_pivot=resolved_pivot,
    )
    max_residual = float(np.max(np.abs(equation_residual)))
    arc = float(np.dot((w_final - w_predicted) / scales, tangent))
    status = _status_from_profile(profile, audit, params, bool(optimizer_success), max_residual)
    tangent_audit = tangent_audit_from_scaled_tangent(tangent, metric, method=tangent_method)
    return TransonicPseudoArclengthResult(
        z=z,
        mdot_ratio=mdot_ratio,
        predicted_mdot_ratio=float(np.exp(w_predicted[-1])),
        params=params,
        profile=profile,
        residual_audit=audit,
        status=status,
        max_residual=max_residual,
        arclength_residual=arc,
        arclength_step=step,
        tangent=np.asarray(tangent, dtype=float),
        metric=metric,
        tangent_audit=tangent_audit,
        cost=float(cost),
        nfev=int(nfev),
        optimizer_success=bool(optimizer_success),
        message=str(message),
        corrector_method=corrector_used,
        corrector_iterations=int(corrector_iterations),
        jacobian_evaluations=int(jacobian_evaluations),
        line_search_reductions=int(line_search_reductions),
        condition_estimate=float(condition_estimate),
        initial_max_residual=initial_max_residual,
        predictor_correction_norm=float(np.linalg.norm((w_final - w_predicted) / scales)),
    )


def continue_in_mdot(
    base_params: TransonicSlimParams,
    mdot_values,
    initial_guess=None,
    reset_after_failure: bool = True,
    keep_last_accepted_on_failure: bool = False,
) -> TransonicContinuationResult:
    """Continue the transonic outer branch through a sequence of ``Mdot`` values."""

    mdot_values = np.asarray(mdot_values, dtype=float)
    if mdot_values.ndim != 1 or len(mdot_values) == 0:
        raise ValueError("mdot_values must be a non-empty one-dimensional array")
    if np.any(mdot_values <= 0.0):
        raise ValueError("mdot_values must be positive")

    results: list[TransonicSolveResult] = []
    previous_profile = None
    for idx, Mdot in enumerate(mdot_values):
        params = replace_mdot(base_params, float(Mdot))
        if idx == 0 and initial_guess is not None:
            guess = initial_guess
        elif previous_profile is not None:
            guess = remap_profile_to_new_sonic_grid(previous_profile, params)
        else:
            guess = None
        result = solve_transonic_outer_branch(params, initial_guess=guess)
        results.append(result)
        if result.converged and result.profile is not None:
            previous_profile = result.profile
        elif reset_after_failure:
            previous_profile = None
        elif not keep_last_accepted_on_failure:
            previous_profile = result.profile
    return TransonicContinuationResult(results=tuple(results), mdot_values=mdot_values)
