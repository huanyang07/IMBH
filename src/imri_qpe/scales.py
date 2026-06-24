"""Analytic scale estimates used to anchor the numerical model."""

from __future__ import annotations

import math

from .constants import (
    A_RAD,
    C,
    DEFAULT_KAPPA_ES,
    DEFAULT_MU_MOL,
    G,
    K_BOLTZMANN,
    M_PROTON,
    SIGMA_SB,
)


def gas_constant_per_gram(mu_mol: float = DEFAULT_MU_MOL) -> float:
    """Return k_B / (mu_mol m_p)."""

    if mu_mol <= 0.0:
        raise ValueError("mu_mol must be positive")
    return K_BOLTZMANN / (mu_mol * M_PROTON)


def omega_k(M_g: float, R_cm: float) -> float:
    """Return Keplerian angular frequency around mass ``M_g``."""

    if M_g <= 0.0:
        raise ValueError("M_g must be positive")
    if R_cm <= 0.0:
        raise ValueError("R_cm must be positive")
    return math.sqrt(G * M_g / R_cm**3)


def transition_temperature(
    Omega_K: float,
    alpha_cool: float,
    kappa: float = DEFAULT_KAPPA_ES,
    a_rad: float = A_RAD,
) -> float:
    """Return the analytic ``P_rad ~= P_gas`` transition temperature."""

    if Omega_K <= 0.0:
        raise ValueError("Omega_K must be positive")
    if alpha_cool <= 0.0:
        raise ValueError("alpha_cool must be positive")
    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    return (32.0 * SIGMA_SB * Omega_K / (3.0 * alpha_cool * kappa * a_rad**2)) ** 0.25


def transition_surface_density(
    T_tr: float,
    Omega_K: float,
    alpha_cool: float,
    kappa: float = DEFAULT_KAPPA_ES,
    mu_mol: float = DEFAULT_MU_MOL,
) -> float:
    """Return the analytic transition surface density from the note."""

    if T_tr <= 0.0:
        raise ValueError("T_tr must be positive")
    if Omega_K <= 0.0:
        raise ValueError("Omega_K must be positive")
    if alpha_cool <= 0.0:
        raise ValueError("alpha_cool must be positive")
    if kappa <= 0.0:
        raise ValueError("kappa must be positive")

    R_gas = gas_constant_per_gram(mu_mol)
    prefactor = 128.0 * SIGMA_SB / (27.0 * alpha_cool * kappa * R_gas * Omega_K)
    return math.sqrt(prefactor) * T_tr**1.5


def critical_participating_mass(R_u_cm: float, Sigma_max: float, f_area: float = 1.0) -> float:
    """Return Mcrit ~= f_area pi R_u^2 Sigma_max in grams."""

    if R_u_cm <= 0.0:
        raise ValueError("R_u_cm must be positive")
    if Sigma_max < 0.0:
        raise ValueError("Sigma_max must be non-negative")
    if f_area < 0.0:
        raise ValueError("f_area must be non-negative")
    return f_area * math.pi * R_u_cm**2 * Sigma_max


def eddington_luminosity(M_g: float, kappa: float = DEFAULT_KAPPA_ES) -> float:
    """Return L_Edd = 4 pi G M c / kappa."""

    if M_g <= 0.0:
        raise ValueError("M_g must be positive")
    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    return 4.0 * math.pi * G * M_g * C / kappa


def eddington_mdot(M_g: float, eta: float = 0.1, kappa: float = DEFAULT_KAPPA_ES) -> float:
    """Return Mdot_Edd = L_Edd / (eta c^2) in g/s."""

    if eta <= 0.0:
        raise ValueError("eta must be positive")
    return eddington_luminosity(M_g, kappa=kappa) / (eta * C**2)

