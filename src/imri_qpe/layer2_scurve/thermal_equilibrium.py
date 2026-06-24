"""Local heating/cooling balance for S-curve calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imri_qpe.constants import DEFAULT_KAPPA_ES, DEFAULT_MU_MOL, G, SIGMA_SB
from imri_qpe.scales import eddington_mdot, omega_k

from .stress_laws import stress_pressure
from .vertical_structure import vertical_state


@dataclass(frozen=True)
class ThermalEquilibriumParams:
    """Parameters for the local thermal-equilibrium residual."""

    alpha: float = 0.01
    mu_stress: float = 0.0
    kappa: float = DEFAULT_KAPPA_ES
    mu_mol: float = DEFAULT_MU_MOL
    q_stream: float = 0.0
    advective_cooling_fraction: float = 0.0
    advective_entropy_gradient: float = 0.0
    advective_temperature: float | None = None
    advective_power: float = 6.0
    wind_cooling_fraction: float = 0.0
    wind_mass_loss_strength: float = 0.0
    wind_mass_loss_power: float = 1.0
    wind_energy_factor: float = 1.0
    wind_mdot_edd_factor: float = 1.0
    eta: float = 0.1

    def __post_init__(self) -> None:
        if self.alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        if not 0.0 <= self.mu_stress <= 1.0:
            raise ValueError("mu_stress must be between 0 and 1")
        if self.kappa <= 0.0:
            raise ValueError("kappa must be positive")
        if self.mu_mol <= 0.0:
            raise ValueError("mu_mol must be positive")
        if self.advective_cooling_fraction < 0.0:
            raise ValueError("advective_cooling_fraction must be non-negative")
        if self.advective_entropy_gradient < 0.0:
            raise ValueError("advective_entropy_gradient must be non-negative")
        if self.advective_temperature is not None and self.advective_temperature <= 0.0:
            raise ValueError("advective_temperature must be positive")
        if self.advective_power <= 0.0:
            raise ValueError("advective_power must be positive")
        if self.wind_cooling_fraction < 0.0:
            raise ValueError("wind_cooling_fraction must be non-negative")
        if self.wind_mass_loss_strength < 0.0:
            raise ValueError("wind_mass_loss_strength must be non-negative")
        if self.wind_mass_loss_power <= 0.0:
            raise ValueError("wind_mass_loss_power must be positive")
        if self.wind_energy_factor < 0.0:
            raise ValueError("wind_energy_factor must be non-negative")
        if self.wind_mdot_edd_factor <= 0.0:
            raise ValueError("wind_mdot_edd_factor must be positive")
        if self.eta <= 0.0:
            raise ValueError("eta must be positive")


def _as_float_or_array(value):
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    return array


def q_rad_minus(T, Sigma, kappa: float = DEFAULT_KAPPA_ES):
    """Return the note's two-sided optically thick radiative cooling rate."""

    if np.any(np.asarray(T) <= 0.0):
        raise ValueError("T must be positive")
    if np.any(np.asarray(Sigma) <= 0.0):
        raise ValueError("Sigma must be positive")
    if kappa <= 0.0:
        raise ValueError("kappa must be positive")

    T = np.asarray(T, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    return _as_float_or_array(16.0 * SIGMA_SB * T**4 / (3.0 * kappa * Sigma))


def q_plus_alpha(
    Sigma,
    T,
    R_cm: float,
    M2_g: float,
    alpha: float,
    mu_stress: float,
    mu_mol: float = DEFAULT_MU_MOL,
):
    """Return viscous heating for the generalized alpha stress law."""

    state = vertical_state(Sigma, T, M2_g, R_cm, mu_mol=mu_mol)
    P_stress = stress_pressure(state.P_gas, state.P_tot, alpha=alpha, mu_stress=mu_stress)
    Q_plus = 9.0 / 4.0 * np.asarray(state.Omega_K) * np.asarray(state.H) * np.asarray(P_stress)
    return _as_float_or_array(Q_plus)


def temperature_activated_cooling_fraction(T, T0: float | None, power: float):
    """Return a smooth high-temperature cooling fraction.

    This is a deliberately simple slim-disk/wind stand-in. It is disabled when
    ``T0`` is ``None`` and tends to unity as ``T >> T0``.
    """

    if T0 is None:
        return 0.0
    if T0 <= 0.0:
        raise ValueError("T0 must be positive")
    if power <= 0.0:
        raise ValueError("power must be positive")
    x = (np.asarray(T, dtype=float) / T0) ** power
    return _as_float_or_array(x / (1.0 + x))


def accretion_rate_from_heating(Q_plus, Omega_K):
    """Convert local two-sided heating to the steady thin-disk Mdot estimate."""

    if np.any(np.asarray(Q_plus) < 0.0):
        raise ValueError("Q_plus must be non-negative")
    if np.any(np.asarray(Omega_K) <= 0.0):
        raise ValueError("Omega_K must be positive")

    return _as_float_or_array(8.0 * np.pi * np.asarray(Q_plus, dtype=float) / (3.0 * np.asarray(Omega_K) ** 2))


def q_adv_minus(Sigma, T, R_cm: float, M2_g: float, Q_plus, xi: float, mu_mol: float = DEFAULT_MU_MOL):
    """Return a local slim-disk advective cooling estimate.

    ``xi`` is the dimensionless radial entropy-gradient coefficient in the
    common one-zone estimate ``Q_adv ~ xi Mdot P / (2 pi R^2 rho)``.
    """

    if xi < 0.0:
        raise ValueError("xi must be non-negative")
    if xi == 0.0:
        return 0.0
    state = vertical_state(Sigma, T, M2_g, R_cm, mu_mol=mu_mol)
    Mdot = accretion_rate_from_heating(Q_plus, state.Omega_K)
    enthalpy_scale = np.asarray(state.P_tot, dtype=float) / np.asarray(state.rho_c, dtype=float)
    Q_adv = xi * np.asarray(Mdot, dtype=float) * enthalpy_scale / (2.0 * np.pi * R_cm**2)
    return _as_float_or_array(Q_adv)


def q_wind_minus(
    Mdot,
    R_cm: float,
    M2_g: float,
    strength: float,
    eta: float = 0.1,
    kappa: float = DEFAULT_KAPPA_ES,
    mdot_edd_factor: float = 1.0,
    power: float = 1.0,
    energy_factor: float = 1.0,
):
    """Return a local super-Eddington wind energy sink.

    The sink is activated above ``mdot_edd_factor * Mdot_Edd`` and removes a
    smooth fraction of the local mass flux with specific energy
    ``energy_factor * GM / (2R)``.
    """

    if strength < 0.0:
        raise ValueError("strength must be non-negative")
    if strength == 0.0:
        return 0.0
    if R_cm <= 0.0:
        raise ValueError("R_cm must be positive")
    if M2_g <= 0.0:
        raise ValueError("M2_g must be positive")
    if mdot_edd_factor <= 0.0:
        raise ValueError("mdot_edd_factor must be positive")
    if power <= 0.0:
        raise ValueError("power must be positive")
    if energy_factor < 0.0:
        raise ValueError("energy_factor must be non-negative")

    Mdot = np.asarray(Mdot, dtype=float)
    Mdot_crit = mdot_edd_factor * eddington_mdot(M2_g, eta=eta, kappa=kappa)
    excess = np.maximum(Mdot / Mdot_crit - 1.0, 0.0)
    loading_fraction = strength * (excess / (1.0 + excess)) ** power
    wind_mdot_per_area = loading_fraction * Mdot / (2.0 * np.pi * R_cm**2)
    specific_energy = energy_factor * G * M2_g / (2.0 * R_cm)
    return _as_float_or_array(wind_mdot_per_area * specific_energy)


def equilibrium_residual(T, Sigma, R_cm: float, M2_g: float, params: ThermalEquilibriumParams):
    """Return Qplus + Qstream - Qrad - Qadv - Qwind."""

    Q_plus = q_plus_alpha(
        Sigma,
        T,
        R_cm,
        M2_g,
        params.alpha,
        params.mu_stress,
        mu_mol=params.mu_mol,
    )
    Q_rad = q_rad_minus(T, Sigma, kappa=params.kappa)
    Mdot = accretion_rate_from_heating(Q_plus, omega_k(M2_g, R_cm))
    Q_adv = q_adv_minus(
        Sigma,
        T,
        R_cm,
        M2_g,
        Q_plus,
        params.advective_entropy_gradient,
        mu_mol=params.mu_mol,
    )
    Q_wind = q_wind_minus(
        Mdot,
        R_cm,
        M2_g,
        params.wind_mass_loss_strength,
        eta=params.eta,
        kappa=params.kappa,
        mdot_edd_factor=params.wind_mdot_edd_factor,
        power=params.wind_mass_loss_power,
        energy_factor=params.wind_energy_factor,
    )
    extra_fraction = (
        params.advective_cooling_fraction
        + params.wind_cooling_fraction
        + np.asarray(
            temperature_activated_cooling_fraction(T, params.advective_temperature, params.advective_power)
        )
    )
    residual = (
        np.asarray(Q_plus) * (1.0 - extra_fraction)
        + params.q_stream
        - np.asarray(Q_rad)
        - np.asarray(Q_adv)
        - np.asarray(Q_wind)
    )
    return _as_float_or_array(residual)


def thermal_equilibrium_quantities(T, Sigma, R_cm: float, M2_g: float, params: ThermalEquilibriumParams):
    """Return common local quantities at a thermal-equilibrium candidate."""

    state = vertical_state(Sigma, T, M2_g, R_cm, mu_mol=params.mu_mol, kappa=params.kappa)
    Q_plus = q_plus_alpha(Sigma, T, R_cm, M2_g, params.alpha, params.mu_stress, params.mu_mol)
    Q_rad = q_rad_minus(T, Sigma, params.kappa)
    Mdot = accretion_rate_from_heating(Q_plus, omega_k(M2_g, R_cm))
    Q_adv = q_adv_minus(Sigma, T, R_cm, M2_g, Q_plus, params.advective_entropy_gradient, params.mu_mol)
    Q_wind = q_wind_minus(
        Mdot,
        R_cm,
        M2_g,
        params.wind_mass_loss_strength,
        eta=params.eta,
        kappa=params.kappa,
        mdot_edd_factor=params.wind_mdot_edd_factor,
        power=params.wind_mass_loss_power,
        energy_factor=params.wind_energy_factor,
    )
    return state, Q_plus, Q_rad, Q_adv, Q_wind, Mdot
