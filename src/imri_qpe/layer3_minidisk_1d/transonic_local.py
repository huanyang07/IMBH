"""Local differential system for an isolated transonic slim disk."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imri_qpe.constants import A_RAD, C
from imri_qpe.scales import gas_constant_per_gram

from .transonic_potential import PaczynskiWiitaPotential
from .transonic_thermo import integrated_stress, radiative_cooling, surface_density, vertical_state


@dataclass(frozen=True)
class AlgebraicTransonicState:
    """Algebraic state at one radius for the transonic equations."""

    R: float
    u: float
    T: float
    Sigma: float
    H: float
    rho: float
    P_gas: float
    P_rad: float
    P: float
    Pi: float
    e: float
    tau: float
    W: float
    l0: float
    l: float
    Omega: float
    Omega_K: float
    l_K: float
    Q_rad: float
    H_over_R: float


@dataclass(frozen=True)
class LocalPartials:
    """Explicit and state partials of local quantities in log variables."""

    x: dict[str, float]
    y: dict[str, np.ndarray]


@dataclass(frozen=True)
class SonicDiagnostics:
    """Matrix criticality and compatibility diagnostics."""

    D: float
    N: float
    smin_over_smax: float
    singular_values: np.ndarray
    left_null: np.ndarray
    right_null: np.ndarray


def algebraic_state(logR: float, logu: float, logT: float, lambda0: float, params) -> AlgebraicTransonicState:
    """Return local algebraic state from ``x=ln R`` and ``y=[ln u, ln T]``."""

    potential = PaczynskiWiitaPotential(params.M2_g)
    R = float(np.exp(logR))
    u = float(np.exp(logu))
    T = float(np.exp(logT))
    Sigma = float(surface_density(params.Mdot_g_s, R, u))
    vertical = vertical_state(
        Sigma,
        T,
        R,
        potential,
        mu_mol=params.mu_mol,
        kappa=params.kappa,
        gamma_gas=params.gamma_gas,
    )
    W = float(integrated_stress(vertical, params.alpha, mu_stress=params.mu_stress, stress_factor=params.stress_factor))
    l0 = float(lambda0 * potential.r_g * C)
    l = float(l0 + 2.0 * np.pi * R**2 * W / params.Mdot_g_s)
    Omega = float(l / R**2)
    Omega_K = float(potential.omega_k(R))
    Q_rad = float(radiative_cooling(vertical, kappa=params.kappa))
    return AlgebraicTransonicState(
        R=R,
        u=u,
        T=T,
        Sigma=Sigma,
        H=float(vertical.H),
        rho=float(vertical.rho),
        P_gas=float(vertical.P_gas),
        P_rad=float(vertical.P_rad),
        P=float(vertical.P_tot),
        Pi=float(vertical.Pi),
        e=float(vertical.e),
        tau=float(vertical.tau),
        W=W,
        l0=l0,
        l=l,
        Omega=Omega,
        Omega_K=Omega_K,
        l_K=float(potential.l_k(R)),
        Q_rad=Q_rad,
        H_over_R=float(vertical.H) / R,
    )


def _quantity_vector(logR: float, y, lambda0: float, params) -> dict[str, float]:
    state = algebraic_state(logR, float(y[0]), float(y[1]), lambda0, params)
    return {
        "Pi": state.Pi,
        "rho": state.rho,
        "e": state.e,
        "Omega": state.Omega,
    }


def finite_difference_state_partials(
    logR: float,
    y,
    lambda0: float,
    params,
    eps_x: float = 1.0e-5,
    eps_y: float = 1.0e-5,
) -> LocalPartials:
    """Return finite-difference partial derivatives in ``x=ln R`` and ``y``."""

    y = np.asarray(y, dtype=float)
    keys = tuple(_quantity_vector(logR, y, lambda0, params).keys())
    plus_x = _quantity_vector(logR + eps_x, y, lambda0, params)
    minus_x = _quantity_vector(logR - eps_x, y, lambda0, params)
    x_partials = {key: (plus_x[key] - minus_x[key]) / (2.0 * eps_x) for key in keys}

    y_partials: dict[str, np.ndarray] = {}
    for key in keys:
        columns = []
        for column in range(2):
            delta = np.zeros(2)
            delta[column] = eps_y
            plus_y = _quantity_vector(logR, y + delta, lambda0, params)
            minus_y = _quantity_vector(logR, y - delta, lambda0, params)
            columns.append((plus_y[key] - minus_y[key]) / (2.0 * eps_y))
        y_partials[key] = np.asarray(columns, dtype=float)

    return LocalPartials(x=x_partials, y=y_partials)


def _directional_derivatives(state: AlgebraicTransonicState, params, dln_sigma: float, dln_T: float, dln_omega_k: float, dln_R: float) -> dict[str, float]:
    """Return analytic local derivatives for one log-variable direction."""

    R_gas = gas_constant_per_gram(params.mu_mol)
    omega2 = state.Omega_K**2
    radiation_term = 2.0 * A_RAD * state.T**4 / (3.0 * state.Sigma)
    dA = R_gas * state.T * dln_T
    dB = radiation_term * (4.0 * dln_T - dln_sigma)
    dOmega2 = 2.0 * omega2 * dln_omega_k
    denominator = 2.0 * omega2 * state.H - radiation_term
    dH = (state.H * dB + dA - state.H**2 * dOmega2) / denominator
    dln_H = dH / state.H
    dln_rho = dln_sigma - dln_H
    drho = state.rho * dln_rho

    dP_gas = state.P_gas * (dln_rho + dln_T)
    dP_rad = state.P_rad * 4.0 * dln_T
    dP = dP_gas + dP_rad
    dPi = state.Pi * (dln_H + dP / state.P)

    e_gas = R_gas * state.T / (params.gamma_gas - 1.0)
    e_rad = A_RAD * state.T**4 / state.rho
    de = e_gas * dln_T + e_rad * (4.0 * dln_T - dln_rho)

    dln_P_gas = dP_gas / state.P_gas
    dln_P = dP / state.P
    dW = state.W * (dln_H + params.mu_stress * dln_P_gas + (1.0 - params.mu_stress) * dln_P)
    dOmega = -2.0 * (state.l0 / state.R**2) * dln_R + (2.0 * np.pi / params.Mdot_g_s) * dW

    return {
        "Pi": float(dPi),
        "rho": float(drho),
        "e": float(de),
        "Omega": float(dOmega),
    }


def analytic_state_partials(logR: float, y, lambda0: float, params) -> LocalPartials:
    """Return analytic partial derivatives in ``x=ln R`` and ``y``."""

    y = np.asarray(y, dtype=float)
    state = algebraic_state(logR, float(y[0]), float(y[1]), lambda0, params)
    potential = PaczynskiWiitaPotential(params.M2_g)
    dln_omega_k_dlnR = float(potential.dln_omega_k_dlnR(state.R))

    x_partials = _directional_derivatives(
        state,
        params,
        dln_sigma=-1.0,
        dln_T=0.0,
        dln_omega_k=dln_omega_k_dlnR,
        dln_R=1.0,
    )
    logu_partials = _directional_derivatives(
        state,
        params,
        dln_sigma=-1.0,
        dln_T=0.0,
        dln_omega_k=0.0,
        dln_R=0.0,
    )
    logT_partials = _directional_derivatives(
        state,
        params,
        dln_sigma=0.0,
        dln_T=1.0,
        dln_omega_k=0.0,
        dln_R=0.0,
    )
    y_partials = {key: np.asarray([logu_partials[key], logT_partials[key]], dtype=float) for key in x_partials}
    return LocalPartials(x=x_partials, y=y_partials)


def state_partials(logR: float, y, lambda0: float, params, eps_x: float = 1.0e-5, eps_y: float = 1.0e-5) -> LocalPartials:
    """Return analytic partial derivatives in ``x=ln R`` and ``y``."""

    _ = eps_x, eps_y
    return analytic_state_partials(logR, y, lambda0, params)


def differential_residual(logR: float, y, g, lambda0: float, params) -> np.ndarray:
    """Return radial-momentum and energy residuals for a local gradient."""

    y = np.asarray(y, dtype=float)
    g = np.asarray(g, dtype=float)
    state = algebraic_state(logR, y[0], y[1], lambda0, params)
    partials = state_partials(logR, y, lambda0, params, eps_x=params.partial_eps, eps_y=params.partial_eps)

    dPi_dx = partials.x["Pi"] + float(np.dot(partials.y["Pi"], g))
    drho_dx = partials.x["rho"] + float(np.dot(partials.y["rho"], g))
    de_dx = partials.x["e"] + float(np.dot(partials.y["e"], g))
    dOmega_dx = partials.x["Omega"] + float(np.dot(partials.y["Omega"], g))

    radial = state.u**2 * g[0] - state.R**2 * (state.Omega**2 - state.Omega_K**2) + dPi_dx / state.Sigma
    Tdsdx = de_dx - state.P / state.rho**2 * drho_dx
    Q_visc = -state.W * dOmega_dx
    Q_adv = -(state.Sigma * state.u / state.R) * Tdsdx
    energy = Q_visc - state.Q_rad - Q_adv
    return np.asarray([radial, energy], dtype=float)


def differential_matrix(logR: float, y, lambda0: float, params) -> tuple[np.ndarray, np.ndarray]:
    """Return ``A, c`` such that ``F(g) = A @ g + c``."""

    y = np.asarray(y, dtype=float)
    c = differential_residual(logR, y, np.zeros(2), lambda0, params)
    col0 = differential_residual(logR, y, np.array([1.0, 0.0]), lambda0, params) - c
    col1 = differential_residual(logR, y, np.array([0.0, 1.0]), lambda0, params) - c
    return np.column_stack([col0, col1]), c


def local_gradient(logR: float, y, lambda0: float, params) -> np.ndarray:
    """Return the nonsingular local gradient solving ``A g + c = 0``."""

    A, c = differential_matrix(logR, y, lambda0, params)
    return np.linalg.solve(A, -c)


def entropy_gradient_log(logR: float, y, g, lambda0: float, params) -> float:
    """Return ``T ds/dlnR`` along a local gradient."""

    y = np.asarray(y, dtype=float)
    g = np.asarray(g, dtype=float)
    state = algebraic_state(logR, y[0], y[1], lambda0, params)
    partials = state_partials(logR, y, lambda0, params, eps_x=params.partial_eps, eps_y=params.partial_eps)
    drho_dx = partials.x["rho"] + float(np.dot(partials.y["rho"], g))
    de_dx = partials.x["e"] + float(np.dot(partials.y["e"], g))
    return de_dx - state.P / state.rho**2 * drho_dx


def xi_eff_from_gradient(logR: float, y, g, lambda0: float, params) -> float:
    """Return local ``xi_eff`` using ``d/dlnR`` form."""

    state = algebraic_state(logR, float(y[0]), float(y[1]), lambda0, params)
    return -state.rho / state.P * entropy_gradient_log(logR, y, g, lambda0, params)


def sonic_diagnostics(logR: float, y, lambda0: float, params, floor: float = 1.0e-300) -> SonicDiagnostics:
    """Return critical determinant and compatibility diagnostics."""

    A, c = differential_matrix(logR, y, lambda0, params)
    U, singular_values, Vt = np.linalg.svd(A)
    det_norm = np.linalg.norm(A[:, 0]) * np.linalg.norm(A[:, 1]) + floor
    D = float(np.linalg.det(A) / det_norm)
    left_null = U[:, -1]
    right_null = Vt[-1, :]
    N = float(np.dot(left_null, c) / (np.linalg.norm(c) + floor))
    smax = float(np.max(singular_values))
    smin = float(np.min(singular_values))
    return SonicDiagnostics(
        D=D,
        N=N,
        smin_over_smax=smin / (smax + floor),
        singular_values=singular_values,
        left_null=left_null,
        right_null=right_null,
    )
