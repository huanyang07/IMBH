"""Supersonic inward continuation of an accepted transonic slim profile."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .entropy_advection import gas_radiation_adiabatic_sound_speed_squared
from .transonic_collocation import TransonicSlimParams, TransonicSlimProfile
from .transonic_local import (
    algebraic_state,
    local_ode_rhs,
    local_scaled_residual,
    sonic_derivative_branches,
)


@dataclass(frozen=True)
class TransonicPlungeProfile:
    """Stationary inward branch and time-dependent characteristic audit."""

    R: np.ndarray
    u: np.ndarray
    T: np.ndarray
    Sigma: np.ndarray
    H: np.ndarray
    rho: np.ndarray
    P: np.ndarray
    Pi: np.ndarray
    e: np.ndarray
    tau: np.ndarray
    Omega: np.ndarray
    Omega_K: np.ndarray
    l: np.ndarray
    W: np.ndarray
    effective_sound_speed: np.ndarray
    radial_mach_number: np.ndarray
    incoming_characteristics: np.ndarray
    selected_sonic_gradient: np.ndarray
    resolved_outer_gradient: np.ndarray
    sonic_gradient_mismatch: float
    maximum_scaled_differential_residual: float
    sonic_offset: float

    @property
    def inner_is_causally_outgoing(self) -> bool:
        return bool(self.incoming_characteristics[0] == 0)


def _resolved_outer_gradient(profile: TransonicSlimProfile) -> np.ndarray:
    if np.asarray(profile.R).size < 2:
        raise ValueError("transonic profile needs two nodes to select a branch")
    dx = float(np.log(profile.R[1] / profile.R[0]))
    if not np.isfinite(dx) or dx <= 0.0:
        raise ValueError("transonic profile radii must increase")
    return np.asarray(
        [
            np.log(profile.u[1] / profile.u[0]) / dx,
            np.log(profile.T[1] / profile.T[0]) / dx,
        ],
        dtype=float,
    )


def continue_transonic_supersonic_plunge(
    profile: TransonicSlimProfile,
    params: TransonicSlimParams,
    inner_radius: float,
    *,
    n_nodes: int = 64,
    sonic_offset: float = 1.0e-6,
    rtol: float = 1.0e-9,
    atol: float = 1.0e-11,
    maximum_log_step: float = 5.0e-3,
) -> TransonicPlungeProfile:
    """Continue the accepted regular branch inward from its sonic node.

    The sonic derivative is selected by proximity to the resolved first
    outer interval. Integration starts one small logarithmic offset inside the
    critical point and uses the production local transonic equations without a
    ballistic, adiabatic, or free-fall replacement.
    """

    sonic_radius = float(profile.sonic_radius)
    inner_radius = float(inner_radius)
    if not np.isfinite(inner_radius) or not params.potential.r_pw < inner_radius < sonic_radius:
        raise ValueError("inner plunge radius must lie between r_pw and R_son")
    if int(n_nodes) != n_nodes or n_nodes < 3:
        raise ValueError("plunge continuation requires at least three nodes")
    if not np.isfinite(sonic_offset) or not 0.0 < sonic_offset < 1.0e-2:
        raise ValueError("sonic_offset must be positive and small")
    if not np.isfinite(maximum_log_step) or maximum_log_step <= 0.0:
        raise ValueError("maximum_log_step must be positive")

    log_sonic = float(np.log(sonic_radius))
    sonic_state = np.log(
        np.asarray([profile.u[0], profile.T[0]], dtype=float)
    )
    outer_gradient = _resolved_outer_gradient(profile)
    branches = sonic_derivative_branches(
        log_sonic,
        sonic_state,
        profile.lambda0,
        params,
        half_width=100.0,
        scan_points=801,
    )
    if not branches:
        raise RuntimeError("no regular sonic derivative branch was found")
    branch = min(
        branches,
        key=lambda candidate: float(
            np.linalg.norm(candidate.gradient - outer_gradient)
        ),
    )
    gradient = np.asarray(branch.gradient, dtype=float)
    start_x = log_sonic - float(sonic_offset)
    start_y = sonic_state - float(sonic_offset) * gradient
    inner_x = float(np.log(inner_radius))

    solution = solve_ivp(
        lambda x, y: local_ode_rhs(
            x, y, profile.lambda0, params
        ),
        (start_x, inner_x),
        start_y,
        rtol=float(rtol),
        atol=float(atol),
        max_step=float(maximum_log_step),
        dense_output=True,
    )
    if not solution.success or solution.sol is None:
        raise RuntimeError(
            f"inward transonic continuation failed: {solution.message}"
        )

    log_radius = np.linspace(inner_x, log_sonic, int(n_nodes))
    log_state = np.empty((2, int(n_nodes)), dtype=float)
    log_state[:, :-1] = solution.sol(log_radius[:-1])
    log_state[:, -1] = sonic_state
    if np.any(~np.isfinite(log_state)):
        raise RuntimeError("inward transonic continuation is not finite")

    states = [
        algebraic_state(
            float(x),
            float(y[0]),
            float(y[1]),
            profile.lambda0,
            params,
        )
        for x, y in zip(log_radius, log_state.T)
    ]
    density = np.asarray([state.rho for state in states], dtype=float)
    temperature = np.asarray([state.T for state in states], dtype=float)
    sound_speed = np.sqrt(
        np.asarray(
            gas_radiation_adiabatic_sound_speed_squared(
                density,
                temperature,
                mu_mol=params.mu_mol,
                gamma_gas=params.gamma_gas,
            ),
            dtype=float,
        )
    )
    radial_velocity = -np.asarray([state.u for state in states], dtype=float)
    incoming = np.asarray(
        [
            sum(
                value > 0.0
                for value in (
                    velocity - sound,
                    velocity,
                    velocity,
                    velocity + sound,
                )
            )
            for velocity, sound in zip(radial_velocity, sound_speed)
        ],
        dtype=int,
    )

    residuals = []
    for x, y in zip(log_radius[:-1], log_state.T[:-1]):
        local_gradient = local_ode_rhs(
            float(x), y, profile.lambda0, params
        )
        residuals.append(
            np.max(
                np.abs(
                    local_scaled_residual(
                        float(x),
                        y,
                        local_gradient,
                        profile.lambda0,
                        params,
                    )
                )
            )
        )

    def values(name: str) -> np.ndarray:
        return np.asarray([getattr(state, name) for state in states], dtype=float)

    return TransonicPlungeProfile(
        R=values("R"),
        u=values("u"),
        T=temperature,
        Sigma=values("Sigma"),
        H=values("H"),
        rho=density,
        P=values("P"),
        Pi=values("Pi"),
        e=values("e"),
        tau=values("tau"),
        Omega=values("Omega"),
        Omega_K=values("Omega_K"),
        l=values("l"),
        W=values("W"),
        effective_sound_speed=sound_speed,
        radial_mach_number=radial_velocity / sound_speed,
        incoming_characteristics=incoming,
        selected_sonic_gradient=gradient,
        resolved_outer_gradient=outer_gradient,
        sonic_gradient_mismatch=float(
            np.linalg.norm(gradient - outer_gradient)
        ),
        maximum_scaled_differential_residual=float(max(residuals)),
        sonic_offset=float(sonic_offset),
    )
