"""Continuation helpers for the isolated transonic slim-disk solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .transonic_collocation import (
    TransonicSlimParams,
    TransonicSolveResult,
    computational_grid,
    pack_state,
    replace_mdot,
    solve_transonic_outer_branch,
)


@dataclass(frozen=True)
class TransonicContinuationResult:
    """Sequence of transonic solves along accretion-rate continuation."""

    results: tuple[TransonicSolveResult, ...]
    mdot_values: np.ndarray


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


def continue_in_mdot(
    base_params: TransonicSlimParams,
    mdot_values,
    initial_guess=None,
    reset_after_failure: bool = True,
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
        else:
            previous_profile = result.profile
    return TransonicContinuationResult(results=tuple(results), mdot_values=mdot_values)
