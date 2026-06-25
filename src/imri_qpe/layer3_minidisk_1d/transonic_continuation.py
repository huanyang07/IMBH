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
    solve_transonic_outer_branch,
    state_bounds,
)


@dataclass(frozen=True)
class TransonicContinuationResult:
    """Sequence of transonic solves along accretion-rate continuation."""

    results: tuple[TransonicSolveResult, ...]
    mdot_values: np.ndarray


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
    cost: float
    nfev: int
    optimizer_success: bool
    message: str


def _state_from_profile(profile) -> np.ndarray:
    return pack_state(np.log(profile.u), np.log(profile.T), np.log(profile.sonic_radius), profile.lambda0)


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


def _pseudo_arclength_sparsity_pattern(params: TransonicSlimParams):
    try:
        from scipy.sparse import lil_matrix
    except Exception:
        return None

    base = jac_sparsity_pattern(params)
    if base is None:
        return None
    n_residual, n_state = base.shape
    pattern = lil_matrix((n_residual + 1, n_state + 1), dtype=int)
    pattern[:n_residual, :n_state] = base
    pattern[:n_residual, n_state] = 1
    pattern[n_residual, :] = 1
    return pattern.tocsr()


def _pseudo_arclength_jacobian(
    w,
    base_params: TransonicSlimParams,
    mdot_unit_g_s: float,
    w_current: np.ndarray,
    scales: np.ndarray,
    tangent: np.ndarray,
    arclength_weight: float,
    rel_step: float = 1.0e-5,
):
    try:
        from scipy.sparse import csr_matrix, hstack, vstack
    except Exception as exc:
        raise RuntimeError("scipy is required for pseudo-arclength jacobian") from exc

    z = np.asarray(w[:-1], dtype=float)
    mu = float(w[-1])
    params = replace_mdot(base_params, float(np.exp(mu)) * mdot_unit_g_s)
    state_jac = collocation_jacobian(z, params)
    base = collocation_residual(z, params)
    step = rel_step * max(1.0, abs(mu))
    if step <= 0.0:
        mdot_column = np.zeros_like(base)
    else:
        plus = collocation_residual(z, replace_mdot(base_params, float(np.exp(mu + step)) * mdot_unit_g_s))
        minus = collocation_residual(z, replace_mdot(base_params, float(np.exp(mu - step)) * mdot_unit_g_s))
        mdot_column = (plus - minus) / (2.0 * step)
    top = hstack([state_jac, csr_matrix(mdot_column[:, None])], format="csr")
    arc_row = csr_matrix((arclength_weight * tangent / scales)[None, :])
    return vstack([top, arc_row], format="csr")


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
    if mdot_ratio_bounds[0] <= 0.0 or mdot_ratio_bounds[1] <= mdot_ratio_bounds[0]:
        raise ValueError("mdot_ratio_bounds must be positive and increasing")

    previous_params = replace_mdot(base_params, previous_mdot_ratio * mdot_unit_g_s)
    current_params = replace_mdot(base_params, current_mdot_ratio * mdot_unit_g_s)
    z_previous = remap_profile_to_new_sonic_grid(previous_profile, previous_params)
    z_current = remap_profile_to_new_sonic_grid(current_profile, current_params)
    w_previous = np.concatenate([z_previous, np.array([np.log(previous_mdot_ratio)])])
    w_current = np.concatenate([z_current, np.array([np.log(current_mdot_ratio)])])

    state_scale = np.sqrt(float(z_current.size))
    scales = np.concatenate([np.full(z_current.size, state_scale), np.ones(1)])
    delta_scaled = (w_current - w_previous) / scales
    tangent_norm = float(np.linalg.norm(delta_scaled))
    if not np.isfinite(tangent_norm) or tangent_norm <= 0.0:
        raise ValueError("previous and current branch points must be distinct")
    tangent = delta_scaled / tangent_norm
    step = float(tangent_norm * step_multiplier if arclength_step is None else arclength_step)
    if step <= 0.0:
        raise ValueError("arclength step must be positive")
    w_predicted = w_current + step * scales * tangent

    lower_state, upper_state = state_bounds(base_params)
    lower = np.concatenate([lower_state, np.array([np.log(mdot_ratio_bounds[0])])])
    upper = np.concatenate([upper_state, np.array([np.log(mdot_ratio_bounds[1])])])
    w0 = np.clip(w_predicted, lower + 1.0e-12, upper - 1.0e-12)

    def residual(w):
        ratio = float(np.exp(w[-1]))
        params = replace_mdot(base_params, ratio * mdot_unit_g_s)
        equation_residual = collocation_residual(w[:-1], params)
        arclength_residual = float(np.dot((w - w_current) / scales, tangent) - step)
        return np.concatenate([equation_residual, np.array([arclength_weight * arclength_residual])])

    result = least_squares(
        residual,
        w0,
        bounds=(lower, upper),
        jac=lambda w: _pseudo_arclength_jacobian(
            w,
            base_params,
            mdot_unit_g_s,
            w_current,
            scales,
            tangent,
            arclength_weight,
        ),
        x_scale="jac",
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10 if residual_tol is None else min(1.0e-6, max(1.0e-10, 1.0e-2 * residual_tol)),
        max_nfev=max_nfev,
        verbose=verbose,
    )
    z = np.asarray(result.x[:-1], dtype=float)
    mdot_ratio = float(np.exp(result.x[-1]))
    params = replace_mdot(base_params, mdot_ratio * mdot_unit_g_s)
    profile = profile_from_state_vector(z, params)
    audit = residual_audit_from_state_vector(z, params)
    equation_residual = collocation_residual(z, params)
    max_residual = float(np.max(np.abs(equation_residual)))
    arc = float(np.dot((result.x - w_current) / scales, tangent) - step)
    status = _status_from_profile(profile, audit, params, bool(result.success), max_residual)
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
        cost=float(result.cost),
        nfev=int(result.nfev),
        optimizer_success=bool(result.success),
        message=str(result.message),
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
