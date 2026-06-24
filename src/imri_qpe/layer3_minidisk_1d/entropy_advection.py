"""Entropy-gradient advection diagnostics for radial minidisk models."""

from __future__ import annotations

import numpy as np

from imri_qpe.constants import A_RAD, DEFAULT_MU_MOL
from imri_qpe.layer2_scurve.vertical_structure import gas_pressure, radiation_pressure
from imri_qpe.scales import gas_constant_per_gram


def _as_float_or_array(value):
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    return array


def _require_positive(name: str, value) -> None:
    if np.any(np.asarray(value) <= 0.0):
        raise ValueError(f"{name} must be positive")


def total_pressure(rho, T, mu_mol: float = DEFAULT_MU_MOL):
    """Return P_gas + P_rad."""

    return _as_float_or_array(np.asarray(gas_pressure(rho, T, mu_mol)) + np.asarray(radiation_pressure(T)))


def specific_internal_energy(rho, T, mu_mol: float = DEFAULT_MU_MOL, gamma_gas: float = 5.0 / 3.0):
    """Return gas+radiation specific internal energy.

    ``e = P_gas / ((gamma_gas - 1) rho) + a_rad T^4 / rho``.
    """

    _require_positive("rho", rho)
    _require_positive("T", T)
    if gamma_gas <= 1.0:
        raise ValueError("gamma_gas must exceed 1")

    rho = np.asarray(rho, dtype=float)
    T = np.asarray(T, dtype=float)
    gas_e = gas_constant_per_gram(mu_mol) * T / (gamma_gas - 1.0)
    rad_e = A_RAD * T**4 / rho
    return _as_float_or_array(gas_e + rad_e)


def slope_limited_gradient(y, R) -> np.ndarray:
    """Return a minmod-limited radial gradient on a 1D grid."""

    y = np.asarray(y, dtype=float)
    R = np.asarray(R, dtype=float)
    if y.shape != R.shape:
        raise ValueError("y and R must have the same shape")
    if y.ndim != 1:
        raise ValueError("inputs must be one-dimensional")
    if len(y) < 2:
        raise ValueError("at least two points are required")
    if np.any(np.diff(R) <= 0.0):
        raise ValueError("R must be strictly increasing")

    gradient = np.empty_like(y)
    gradient[0] = (y[1] - y[0]) / (R[1] - R[0])
    gradient[-1] = (y[-1] - y[-2]) / (R[-1] - R[-2])
    if len(y) == 2:
        return gradient

    left = (y[1:-1] - y[:-2]) / (R[1:-1] - R[:-2])
    right = (y[2:] - y[1:-1]) / (R[2:] - R[1:-1])
    same_sign = np.sign(left) == np.sign(right)
    limited = np.where(same_sign, np.sign(left) * np.minimum(np.abs(left), np.abs(right)), 0.0)
    gradient[1:-1] = limited
    return gradient


def centered_gradient(y, R) -> np.ndarray:
    """Return a second-order centered radial gradient with one-sided edges."""

    y = np.asarray(y, dtype=float)
    R = np.asarray(R, dtype=float)
    if y.shape != R.shape:
        raise ValueError("y and R must have the same shape")
    if y.ndim != 1:
        raise ValueError("inputs must be one-dimensional")
    if len(y) < 2:
        raise ValueError("at least two points are required")
    if np.any(np.diff(R) <= 0.0):
        raise ValueError("R must be strictly increasing")
    return np.gradient(y, R, edge_order=1)


def radial_gradient(y, R, method: str = "limited") -> np.ndarray:
    """Return a radial gradient using the requested stencil."""

    if method == "limited":
        return slope_limited_gradient(y, R)
    if method == "centered":
        return centered_gradient(y, R)
    raise ValueError("method must be 'limited' or 'centered'")


def entropy_temperature_gradient(R, rho, T, P=None, e=None, gradient_method: str = "limited") -> np.ndarray:
    """Return ``T ds/dR = de/dR - P/rho^2 d rho/dR``."""

    R = np.asarray(R, dtype=float)
    rho = np.asarray(rho, dtype=float)
    T = np.asarray(T, dtype=float)
    if R.shape != rho.shape or R.shape != T.shape:
        raise ValueError("R, rho, and T must have the same shape")
    _require_positive("rho", rho)
    _require_positive("T", T)

    P = total_pressure(rho, T) if P is None else np.asarray(P, dtype=float)
    e = specific_internal_energy(rho, T) if e is None else np.asarray(e, dtype=float)
    if P.shape != R.shape or e.shape != R.shape:
        raise ValueError("P and e must match R shape")

    de_dR = radial_gradient(e, R, method=gradient_method)
    drho_dR = radial_gradient(rho, R, method=gradient_method)
    return de_dR - P / rho**2 * drho_dR


def entropy_gradient_log_formula(
    R,
    rho,
    T,
    mu_mol: float = DEFAULT_MU_MOL,
    gamma_gas: float = 5.0 / 3.0,
    gradient_method: str = "limited",
) -> np.ndarray:
    """Return ``T ds/dR`` from gas+radiation log-gradient formula."""

    R = np.asarray(R, dtype=float)
    rho = np.asarray(rho, dtype=float)
    T = np.asarray(T, dtype=float)
    if R.shape != rho.shape or R.shape != T.shape:
        raise ValueError("R, rho, and T must have the same shape")
    _require_positive("rho", rho)
    _require_positive("T", T)
    if gamma_gas <= 1.0:
        raise ValueError("gamma_gas must exceed 1")

    R_gas = gas_constant_per_gram(mu_mol)
    dlnT_dR = radial_gradient(np.log(T), R, method=gradient_method)
    dlnrho_dR = radial_gradient(np.log(rho), R, method=gradient_method)

    gas = R_gas * T * ((1.0 / (gamma_gas - 1.0)) * dlnT_dR - dlnrho_dR)
    rad = (4.0 * A_RAD * T**4 / rho) * (dlnT_dR - (1.0 / 3.0) * dlnrho_dR)
    return gas + rad


def entropy_gradient_consistency_error(TdsdR_first_law, TdsdR_log, floor: float = 1.0e-300) -> np.ndarray:
    """Return bounded relative disagreement between entropy-gradient formulae."""

    if floor <= 0.0:
        raise ValueError("floor must be positive")
    first = np.asarray(TdsdR_first_law, dtype=float)
    log = np.asarray(TdsdR_log, dtype=float)
    if first.shape != log.shape:
        raise ValueError("entropy-gradient arrays must have the same shape")
    denom = np.abs(first) + np.abs(log) + floor
    return np.abs(first - log) / denom


def q_advective(Sigma, v_R, TdsdR):
    """Return ``Q_adv = Sigma v_R T ds/dR``.

    With the project sign convention, inward flow has ``v_R < 0``. If entropy
    increases inward, ``TdsdR < 0`` and this term is positive cooling.
    """

    Sigma = np.asarray(Sigma, dtype=float)
    v_R = np.asarray(v_R, dtype=float)
    TdsdR = np.asarray(TdsdR, dtype=float)
    if np.any(Sigma < 0.0):
        raise ValueError("Sigma must be non-negative")
    return _as_float_or_array(Sigma * v_R * TdsdR)


def xi_eff(R, rho, P, TdsdR):
    """Return ``xi_eff = - R rho/P * TdsdR``."""

    R = np.asarray(R, dtype=float)
    rho = np.asarray(rho, dtype=float)
    P = np.asarray(P, dtype=float)
    TdsdR = np.asarray(TdsdR, dtype=float)
    _require_positive("R", R)
    _require_positive("rho", rho)
    _require_positive("P", P)
    return _as_float_or_array(-R * rho / P * TdsdR)


def mdot_from_vr(R, Sigma, v_R):
    """Return inward-positive ``Mdot = -2 pi R Sigma v_R``."""

    R = np.asarray(R, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    v_R = np.asarray(v_R, dtype=float)
    _require_positive("R", R)
    if np.any(Sigma < 0.0):
        raise ValueError("Sigma must be non-negative")
    return _as_float_or_array(-2.0 * np.pi * R * Sigma * v_R)
