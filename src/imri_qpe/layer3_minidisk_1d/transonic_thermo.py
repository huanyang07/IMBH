"""Local thermodynamic closures for the isolated transonic slim disk."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imri_qpe.constants import A_RAD, DEFAULT_KAPPA_ES, DEFAULT_MU_MOL, SIGMA_SB
from imri_qpe.scales import gas_constant_per_gram


def _as_float_or_array(value):
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    return array


def _require_positive(name: str, value) -> None:
    if np.any(np.asarray(value) <= 0.0):
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class TransonicVerticalState:
    """One-zone vertical state for the transonic solver."""

    Sigma: np.ndarray | float
    T: np.ndarray | float
    R: np.ndarray | float
    Omega_K: np.ndarray | float
    H: np.ndarray | float
    rho: np.ndarray | float
    P_gas: np.ndarray | float
    P_rad: np.ndarray | float
    P_tot: np.ndarray | float
    Pi: np.ndarray | float
    e: np.ndarray | float
    tau: np.ndarray | float


def surface_density(Mdot_g_s, R, u):
    """Return ``Sigma = Mdot/(2 pi R u)`` for inward speed ``u > 0``."""

    _require_positive("Mdot_g_s", Mdot_g_s)
    _require_positive("R", R)
    _require_positive("u", u)
    return _as_float_or_array(np.asarray(Mdot_g_s, dtype=float) / (2.0 * np.pi * np.asarray(R, dtype=float) * np.asarray(u, dtype=float)))


def vertical_state(
    Sigma,
    T,
    R,
    potential,
    mu_mol: float = DEFAULT_MU_MOL,
    kappa: float = DEFAULT_KAPPA_ES,
    gamma_gas: float = 5.0 / 3.0,
) -> TransonicVerticalState:
    """Return the pseudo-Newtonian one-zone vertical state."""

    _require_positive("Sigma", Sigma)
    _require_positive("T", T)
    _require_positive("R", R)
    if gamma_gas <= 1.0:
        raise ValueError("gamma_gas must exceed one")

    Sigma = np.asarray(Sigma, dtype=float)
    T = np.asarray(T, dtype=float)
    R = np.asarray(R, dtype=float)
    Omega_K = potential.omega_k(R)
    R_gas = gas_constant_per_gram(mu_mol)

    radiation_term = 2.0 * A_RAD * T**4 / (3.0 * Sigma)
    H = (radiation_term + np.sqrt(radiation_term**2 + 4.0 * Omega_K**2 * R_gas * T)) / (2.0 * Omega_K**2)
    rho = Sigma / (2.0 * H)
    P_gas = rho * R_gas * T
    P_rad = A_RAD * T**4 / 3.0
    P_tot = P_gas + P_rad
    Pi = 2.0 * H * P_tot
    e = R_gas * T / (gamma_gas - 1.0) + A_RAD * T**4 / rho
    tau = 0.5 * kappa * Sigma

    return TransonicVerticalState(
        Sigma=_as_float_or_array(Sigma),
        T=_as_float_or_array(T),
        R=_as_float_or_array(R),
        Omega_K=_as_float_or_array(Omega_K),
        H=_as_float_or_array(H),
        rho=_as_float_or_array(rho),
        P_gas=_as_float_or_array(P_gas),
        P_rad=_as_float_or_array(P_rad),
        P_tot=_as_float_or_array(P_tot),
        Pi=_as_float_or_array(Pi),
        e=_as_float_or_array(e),
        tau=_as_float_or_array(tau),
    )


def integrated_stress(state: TransonicVerticalState, alpha: float, mu_stress: float = 0.0, stress_factor: float = 1.0):
    """Return positive vertically integrated alpha stress magnitude."""

    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    if not 0.0 <= mu_stress <= 1.0:
        raise ValueError("mu_stress must be between zero and one")
    _require_positive("stress_factor", stress_factor)
    P_mix = np.asarray(state.P_gas) ** mu_stress * np.asarray(state.P_tot) ** (1.0 - mu_stress)
    return _as_float_or_array(stress_factor * alpha * 2.0 * np.asarray(state.H) * P_mix)


def radiative_cooling(state: TransonicVerticalState, kappa: float = DEFAULT_KAPPA_ES):
    """Return two-face electron-scattering diffusion cooling."""

    _require_positive("kappa", kappa)
    return _as_float_or_array(16.0 * SIGMA_SB * np.asarray(state.T) ** 4 / (3.0 * kappa * np.asarray(state.Sigma)))
