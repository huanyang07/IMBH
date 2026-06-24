"""Nearly Keplerian global slim/wind minidisk diagnostics.

This module implements the near-term global upgrade described in
``Note/CODEX_SLIM_WIND_UPGRADE.md``. It is not a transonic GR slim-disk
solver; it is a radially consistent Keplerian diagnostic that computes
``v_R`` from angular-momentum transport and ``Q_adv`` from radial entropy
gradients.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imri_qpe.constants import DEFAULT_KAPPA_ES, DEFAULT_MU_MOL, G, SIGMA_SB
from imri_qpe.scales import gas_constant_per_gram

from .entropy_advection import (
    entropy_temperature_gradient,
    mdot_from_vr,
    q_advective,
    specific_internal_energy,
    total_pressure,
    xi_eff,
)
from .grid import RadialGrid
from .winds import energy_limited_wind, q_available, q_edd_vertical, wind_energy_per_mass


@dataclass(frozen=True)
class GlobalSlimParams:
    """Parameters for the nearly Keplerian global slim-disk evaluator."""

    M2_g: float
    alpha: float = 0.01
    mu_mol: float = DEFAULT_MU_MOL
    kappa: float = DEFAULT_KAPPA_ES
    epsilon_wind: float = 0.0
    v_inf_factor: float = 0.0
    gamma_gas: float = 5.0 / 3.0
    zero_inner_torque: bool = True


@dataclass(frozen=True)
class GlobalSlimProfile:
    """Radial diagnostic profile for a candidate slim/wind minidisk state."""

    R: np.ndarray
    area: np.ndarray
    Sigma: np.ndarray
    T: np.ndarray
    Omega_K: np.ndarray
    H: np.ndarray
    rho: np.ndarray
    P: np.ndarray
    e: np.ndarray
    nu: np.ndarray
    W: np.ndarray
    v_R: np.ndarray
    Mdot: np.ndarray
    TdsdR: np.ndarray
    xi_eff: np.ndarray
    Q_visc: np.ndarray
    Q_rad_diffusion: np.ndarray
    Q_adv: np.ndarray
    Q_edd: np.ndarray
    Q_wind: np.ndarray
    Q_rad_limited: np.ndarray
    dotSigma_w: np.ndarray
    energy_residual: np.ndarray
    advective_fraction: np.ndarray
    H_over_R: np.ndarray
    tau: np.ndarray


@dataclass(frozen=True)
class GlobalSlimRelaxationResult:
    """Result of a fixed-Sigma global energy-balance relaxation."""

    profile: GlobalSlimProfile
    converged: bool
    iterations: int
    max_normalized_residual: float
    history: np.ndarray


def _require_positive(name: str, value) -> None:
    if np.any(np.asarray(value) <= 0.0):
        raise ValueError(f"{name} must be positive")


def keplerian_omega(M2_g: float, R):
    """Return Keplerian angular frequency."""

    _require_positive("M2_g", M2_g)
    _require_positive("R", R)
    return np.sqrt(G * M2_g / np.asarray(R, dtype=float) ** 3)


def keplerian_specific_angular_momentum(M2_g: float, R):
    """Return l = sqrt(G M R)."""

    _require_positive("M2_g", M2_g)
    _require_positive("R", R)
    return np.sqrt(G * M2_g * np.asarray(R, dtype=float))


def vertical_structure_arrays(
    Sigma,
    T,
    M2_g: float,
    R,
    mu_mol: float = DEFAULT_MU_MOL,
    kappa: float = DEFAULT_KAPPA_ES,
    gamma_gas: float = 5.0 / 3.0,
):
    """Return vectorized one-zone vertical structure arrays."""

    _require_positive("Sigma", Sigma)
    _require_positive("T", T)
    R = np.asarray(R, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    T = np.asarray(T, dtype=float)
    if R.shape != Sigma.shape or R.shape != T.shape:
        raise ValueError("R, Sigma, and T must have the same shape")

    Omega = keplerian_omega(M2_g, R)
    R_gas = gas_constant_per_gram(mu_mol)
    # A_RAD = 4 sigma/c, but importing through radiation pressure keeps this
    # module tied to the same constants as Layer 2.
    from imri_qpe.constants import A_RAD

    radiation_term = 2.0 * A_RAD * T**4 / (3.0 * Sigma)
    H = (radiation_term + np.sqrt(radiation_term**2 + 4.0 * Omega**2 * R_gas * T)) / (2.0 * Omega**2)
    rho = Sigma / (2.0 * H)
    P = total_pressure(rho, T, mu_mol=mu_mol)
    e = specific_internal_energy(rho, T, mu_mol=mu_mol, gamma_gas=gamma_gas)
    tau = 0.5 * kappa * Sigma
    return Omega, H, rho, np.asarray(P), np.asarray(e), tau


def alpha_viscosity(alpha: float, H, Omega_K):
    """Return nu = alpha H^2 Omega_K."""

    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    return alpha * np.asarray(H, dtype=float) ** 2 * np.asarray(Omega_K, dtype=float)


def keplerian_stress(Sigma, nu, Omega_K):
    """Return positive outward stress magnitude W = 1.5 nu Sigma Omega_K."""

    return 1.5 * np.asarray(nu, dtype=float) * np.asarray(Sigma, dtype=float) * np.asarray(Omega_K, dtype=float)


def keplerian_viscous_heating(nu, Sigma, Omega_K):
    """Return two-face Keplerian viscous heating ``Q+ = 9/4 nu Sigma Omega^2``."""

    return 9.0 / 4.0 * np.asarray(nu, dtype=float) * np.asarray(Sigma, dtype=float) * np.asarray(Omega_K, dtype=float) ** 2


def radial_velocity_from_angular_momentum(
    R,
    Sigma,
    W,
    M2_g: float,
    source=None,
    l_in=None,
    lambda_tide=None,
    wind_loss=None,
    l_w=None,
    zero_inner_torque: bool = True,
):
    """Compute v_R from the reduced angular-momentum equation."""

    R = np.asarray(R, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    W = np.asarray(W, dtype=float).copy()
    if R.shape != Sigma.shape or R.shape != W.shape:
        raise ValueError("R, Sigma, and W must have the same shape")
    _require_positive("R", R)
    _require_positive("Sigma", Sigma)

    l = keplerian_specific_angular_momentum(M2_g, R)
    dl_dR = 0.5 * l / R
    if zero_inner_torque and len(W) > 0:
        W[0] = 0.0

    dR2W_dR = np.gradient(R**2 * W, R, edge_order=1)
    source = np.zeros_like(R) if source is None else np.asarray(source, dtype=float)
    lambda_tide = np.zeros_like(R) if lambda_tide is None else np.asarray(lambda_tide, dtype=float)
    wind_loss = np.zeros_like(R) if wind_loss is None else np.asarray(wind_loss, dtype=float)
    l_w = l if l_w is None else np.asarray(l_w, dtype=float)
    if l_in is None:
        l_in_array = l
    else:
        l_in_array = np.asarray(l_in, dtype=float) + np.zeros_like(R)

    numerator = (
        -dR2W_dR / R
        + Sigma * lambda_tide
        + source * (l_in_array - l)
        - wind_loss * (l_w - l)
    )
    return numerator / (Sigma * dl_dR)


def evaluate_global_slim_profile(
    grid: RadialGrid,
    Sigma,
    T,
    params: GlobalSlimParams,
    source=None,
    l_in=None,
    lambda_tide=None,
    wind_loss_for_angular_momentum=None,
    l_w=None,
) -> GlobalSlimProfile:
    """Evaluate a radially consistent nearly Keplerian slim/wind profile."""

    R = grid.centers
    Sigma = np.asarray(Sigma, dtype=float)
    T = np.asarray(T, dtype=float)
    if Sigma.shape != R.shape or T.shape != R.shape:
        raise ValueError("Sigma and T must match grid centers")

    Omega, H, rho, P, e, tau = vertical_structure_arrays(
        Sigma,
        T,
        params.M2_g,
        R,
        params.mu_mol,
        params.kappa,
        params.gamma_gas,
    )
    nu = alpha_viscosity(params.alpha, H, Omega)
    W = keplerian_stress(Sigma, nu, Omega)
    v_R = radial_velocity_from_angular_momentum(
        R,
        Sigma,
        W,
        params.M2_g,
        source=source,
        l_in=l_in,
        lambda_tide=lambda_tide,
        wind_loss=wind_loss_for_angular_momentum,
        l_w=l_w,
        zero_inner_torque=params.zero_inner_torque,
    )
    Mdot = mdot_from_vr(R, Sigma, v_R)
    TdsdR = entropy_temperature_gradient(R, rho, T, P=P, e=e)
    xi = xi_eff(R, rho, P, TdsdR)
    Q_adv = np.asarray(q_advective(Sigma, v_R, TdsdR), dtype=float)
    Q_visc = keplerian_viscous_heating(nu, Sigma, Omega)
    Q_rad_diffusion = 16.0 * SIGMA_SB * T**4 / (3.0 * params.kappa * Sigma)
    Q_edd = q_edd_vertical(Omega, H, params.kappa)
    if params.epsilon_wind == 0.0:
        Q_wind = np.zeros_like(Q_visc)
        Q_rad_limited = Q_rad_diffusion
        dotSigma_w = np.zeros_like(Q_visc)
    else:
        v_esc = np.sqrt(2.0 * G * params.M2_g / R)
        E_w = wind_energy_per_mass(params.M2_g, R, v_inf=params.v_inf_factor * v_esc)
        Q_avail = q_available(Q_visc, Q_adv=Q_adv)
        Q_wind, Q_rad_limited, dotSigma_w = energy_limited_wind(Q_avail, Q_edd, E_w, params.epsilon_wind)
    energy_residual = Q_visc - Q_rad_limited - Q_adv - Q_wind
    advective_fraction = np.divide(Q_adv, Q_visc, out=np.full_like(Q_adv, np.nan), where=Q_visc != 0.0)

    return GlobalSlimProfile(
        R=R,
        area=grid.area,
        Sigma=Sigma,
        T=T,
        Omega_K=Omega,
        H=H,
        rho=rho,
        P=P,
        e=e,
        nu=nu,
        W=W,
        v_R=v_R,
        Mdot=np.asarray(Mdot, dtype=float),
        TdsdR=TdsdR,
        xi_eff=np.asarray(xi, dtype=float),
        Q_visc=Q_visc,
        Q_rad_diffusion=Q_rad_diffusion,
        Q_adv=Q_adv,
        Q_edd=np.asarray(Q_edd, dtype=float),
        Q_wind=np.asarray(Q_wind, dtype=float),
        Q_rad_limited=np.asarray(Q_rad_limited, dtype=float),
        dotSigma_w=np.asarray(dotSigma_w, dtype=float),
        energy_residual=energy_residual,
        advective_fraction=advective_fraction,
        H_over_R=H / R,
        tau=tau,
    )


def integrated_energy_residual(profile: GlobalSlimProfile) -> float:
    """Return a dimensionless global energy residual using cell areas."""

    heat = np.sum(profile.Q_visc * profile.area)
    residual = np.sum(profile.energy_residual * profile.area)
    if heat == 0.0:
        return np.nan
    return float(residual / heat)


def relax_temperature_energy_balance(
    grid: RadialGrid,
    Sigma,
    T_initial,
    params: GlobalSlimParams,
    max_iter: int = 200,
    tol: float = 1.0e-3,
    damping: float = 0.25,
    max_log_step: float = 0.2,
    T_bounds: tuple[float, float] = (1.0e3, 1.0e10),
    source=None,
    l_in=None,
    lambda_tide=None,
    wind_loss_for_angular_momentum=None,
    l_w=None,
) -> GlobalSlimRelaxationResult:
    """Relax temperature at fixed Sigma against the global energy residual.

    This is a diagnostic fixed-profile relaxation, not a transonic slim-disk
    boundary-value solver. It asks whether a supplied surface-density profile
    can find temperatures for which the actual radial entropy advection,
    radiation, and energy-limited wind close the vertically integrated energy
    equation.
    """

    if max_iter < 0:
        raise ValueError("max_iter must be non-negative")
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    if damping <= 0.0:
        raise ValueError("damping must be positive")
    if max_log_step <= 0.0:
        raise ValueError("max_log_step must be positive")
    if T_bounds[0] <= 0.0 or T_bounds[1] <= T_bounds[0]:
        raise ValueError("T_bounds must be positive and increasing")

    T_initial = np.asarray(T_initial, dtype=float)
    _require_positive("T_initial", T_initial)
    logT = np.log(np.clip(T_initial, T_bounds[0], T_bounds[1]))
    logT_min = np.log(T_bounds[0])
    logT_max = np.log(T_bounds[1])
    history: list[float] = []

    profile = evaluate_global_slim_profile(
        grid,
        Sigma,
        np.exp(logT),
        params,
        source=source,
        l_in=l_in,
        lambda_tide=lambda_tide,
        wind_loss_for_angular_momentum=wind_loss_for_angular_momentum,
        l_w=l_w,
    )
    converged = False
    iterations = 0
    max_residual = np.inf

    for iteration in range(max_iter + 1):
        profile = evaluate_global_slim_profile(
            grid,
            Sigma,
            np.exp(logT),
            params,
            source=source,
            l_in=l_in,
            lambda_tide=lambda_tide,
            wind_loss_for_angular_momentum=wind_loss_for_angular_momentum,
            l_w=l_w,
        )
        scale = (
            np.abs(profile.Q_visc)
            + np.abs(profile.Q_rad_limited)
            + np.abs(profile.Q_adv)
            + np.abs(profile.Q_wind)
            + 1.0e-300
        )
        normalized = profile.energy_residual / scale
        max_residual = float(np.max(np.abs(normalized)))
        history.append(max_residual)
        iterations = iteration
        if max_residual <= tol:
            converged = True
            break
        if iteration == max_iter:
            break
        logT += damping * np.clip(normalized, -max_log_step, max_log_step)
        logT = np.clip(logT, logT_min, logT_max)

    return GlobalSlimRelaxationResult(
        profile=profile,
        converged=converged,
        iterations=iterations,
        max_normalized_residual=max_residual,
        history=np.asarray(history, dtype=float),
    )
