"""One-zone vertical structure for a circumsecondary disk annulus."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imri_qpe.constants import A_RAD, DEFAULT_KAPPA_ES, DEFAULT_MU_MOL, SIGMA_SB
from imri_qpe.scales import gas_constant_per_gram, omega_k


@dataclass(frozen=True)
class VerticalState:
    """Midplane and column quantities for one annulus."""

    Sigma: np.ndarray | float
    T_c: np.ndarray | float
    Omega_K: np.ndarray | float
    H: np.ndarray | float
    rho_c: np.ndarray | float
    P_gas: np.ndarray | float
    P_rad: np.ndarray | float
    P_tot: np.ndarray | float
    tau: np.ndarray | float


def _as_float_or_array(value):
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    return array


def _require_positive(name: str, value) -> None:
    if np.any(np.asarray(value) <= 0.0):
        raise ValueError(f"{name} must be positive")


def gas_pressure(rho, T, mu_mol: float = DEFAULT_MU_MOL):
    """Return P_gas = rho k_B T / (mu_mol m_p)."""

    _require_positive("mu_mol", mu_mol)
    rho = np.asarray(rho, dtype=float)
    T = np.asarray(T, dtype=float)
    return _as_float_or_array(rho * gas_constant_per_gram(mu_mol) * T)


def radiation_pressure(T):
    """Return P_rad = a_rad T^4 / 3."""

    T = np.asarray(T, dtype=float)
    return _as_float_or_array(A_RAD * T**4 / 3.0)


def solve_scale_height(Sigma, T, M2_g: float, R_cm: float, mu_mol: float = DEFAULT_MU_MOL):
    """Solve vertical hydrostatic balance for the positive scale height.

    The equation is

    ``0.5 Sigma Omega_K^2 H = Sigma R_gas T / (2 H) + a_rad T^4 / 3``.
    """

    _require_positive("Sigma", Sigma)
    _require_positive("T", T)
    _require_positive("M2_g", M2_g)
    _require_positive("R_cm", R_cm)

    Sigma = np.asarray(Sigma, dtype=float)
    T = np.asarray(T, dtype=float)
    Omega = omega_k(M2_g, R_cm)
    R_gas = gas_constant_per_gram(mu_mol)

    radiation_term = 2.0 * A_RAD * T**4 / (3.0 * Sigma)
    discriminant = radiation_term**2 + 4.0 * Omega**2 * R_gas * T
    H = (radiation_term + np.sqrt(discriminant)) / (2.0 * Omega**2)
    return _as_float_or_array(H)


def midplane_density(Sigma, H):
    """Return rho_c = Sigma / (2 H)."""

    _require_positive("Sigma", Sigma)
    _require_positive("H", H)
    return _as_float_or_array(np.asarray(Sigma, dtype=float) / (2.0 * np.asarray(H, dtype=float)))


def optical_depth(Sigma, kappa: float = DEFAULT_KAPPA_ES):
    """Return the midplane-to-surface optical depth tau = kappa Sigma / 2."""

    _require_positive("Sigma", Sigma)
    _require_positive("kappa", kappa)
    return _as_float_or_array(0.5 * kappa * np.asarray(Sigma, dtype=float))


def effective_temperature(Q_minus):
    """Return T_eff for a two-sided cooling rate Q_minus = 2 sigma_SB T_eff^4."""

    _require_positive("Q_minus", Q_minus)
    return _as_float_or_array((np.asarray(Q_minus, dtype=float) / (2.0 * SIGMA_SB)) ** 0.25)


def vertical_state(
    Sigma,
    T,
    M2_g: float,
    R_cm: float,
    mu_mol: float = DEFAULT_MU_MOL,
    kappa: float = DEFAULT_KAPPA_ES,
) -> VerticalState:
    """Return the one-zone vertical state at ``Sigma`` and ``T``."""

    H = solve_scale_height(Sigma, T, M2_g, R_cm, mu_mol=mu_mol)
    rho_c = midplane_density(Sigma, H)
    P_gas = gas_pressure(rho_c, T, mu_mol=mu_mol)
    P_rad = radiation_pressure(T)
    P_tot = np.asarray(P_gas) + np.asarray(P_rad)
    return VerticalState(
        Sigma=_as_float_or_array(Sigma),
        T_c=_as_float_or_array(T),
        Omega_K=omega_k(M2_g, R_cm),
        H=H,
        rho_c=rho_c,
        P_gas=P_gas,
        P_rad=P_rad,
        P_tot=_as_float_or_array(P_tot),
        tau=optical_depth(Sigma, kappa=kappa),
    )

