"""S-curve construction and turning-point utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imri_qpe.layer2_scurve.thermal_equilibrium import (
    ThermalEquilibriumParams,
    accretion_rate_from_heating,
    equilibrium_residual,
    q_plus_alpha,
    q_rad_minus,
)
from imri_qpe.layer2_scurve.vertical_structure import effective_temperature, vertical_state
from imri_qpe.scales import omega_k


@dataclass(frozen=True)
class ScurveResult:
    """Flattened equilibrium points from a local S-curve scan."""

    Sigma: np.ndarray
    T_c: np.ndarray
    H: np.ndarray
    tau: np.ndarray
    T_eff: np.ndarray
    mdot: np.ndarray
    stable: np.ndarray
    branch_index: np.ndarray


def _bisect_log_temperature(
    Sigma: float,
    R_cm: float,
    M2_g: float,
    params: ThermalEquilibriumParams,
    logT_left: float,
    logT_right: float,
    max_iter: int,
    rtol: float,
) -> float:
    left = logT_left
    right = logT_right
    f_left = equilibrium_residual(np.exp(left), Sigma, R_cm, M2_g, params)
    f_right = equilibrium_residual(np.exp(right), Sigma, R_cm, M2_g, params)

    if f_left == 0.0:
        return float(np.exp(left))
    if f_right == 0.0:
        return float(np.exp(right))
    if np.sign(f_left) == np.sign(f_right):
        raise ValueError("temperature bracket does not straddle a root")

    mid = 0.5 * (left + right)
    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        f_mid = equilibrium_residual(np.exp(mid), Sigma, R_cm, M2_g, params)
        if abs(right - left) <= rtol:
            break
        if f_mid == 0.0:
            break
        if np.sign(f_left) == np.sign(f_mid):
            left = mid
            f_left = f_mid
        else:
            right = mid
    return float(np.exp(mid))


def _roots_for_sigma(
    Sigma: float,
    R_cm: float,
    M2_g: float,
    params: ThermalEquilibriumParams,
    T_bounds: tuple[float, float],
    n_T_brackets: int,
    max_iter: int,
    rtol: float,
) -> list[float]:
    T_grid = np.geomspace(T_bounds[0], T_bounds[1], n_T_brackets)
    logT_grid = np.log(T_grid)
    residuals = np.array([equilibrium_residual(T, Sigma, R_cm, M2_g, params) for T in T_grid], dtype=float)
    roots: list[float] = []

    for idx in range(len(T_grid) - 1):
        f_left = residuals[idx]
        f_right = residuals[idx + 1]
        if not np.isfinite(f_left) or not np.isfinite(f_right):
            continue
        if f_left == 0.0:
            roots.append(float(T_grid[idx]))
            continue
        if np.sign(f_left) == np.sign(f_right):
            continue
        root = _bisect_log_temperature(
            Sigma,
            R_cm,
            M2_g,
            params,
            logT_grid[idx],
            logT_grid[idx + 1],
            max_iter=max_iter,
            rtol=rtol,
        )
        if not roots or abs(np.log(root / roots[-1])) > 10.0 * rtol:
            roots.append(root)

    if residuals[-1] == 0.0:
        roots.append(float(T_grid[-1]))
    return roots


def _thermal_stability(T: float, Sigma: float, R_cm: float, M2_g: float, params: ThermalEquilibriumParams) -> bool:
    dlogT = 1.0e-4
    T_low = T * np.exp(-dlogT)
    T_high = T * np.exp(dlogT)
    derivative = (
        equilibrium_residual(T_high, Sigma, R_cm, M2_g, params)
        - equilibrium_residual(T_low, Sigma, R_cm, M2_g, params)
    ) / (T_high - T_low)
    return bool(derivative < 0.0)


def compute_scurve(
    R_cm: float,
    Sigma_grid,
    M2_g: float,
    params: ThermalEquilibriumParams,
    T_bounds: tuple[float, float] = (1.0e3, 1.0e9),
    n_T_brackets: int = 400,
    max_iter: int = 100,
    rtol: float = 1.0e-10,
) -> ScurveResult:
    """Return all local thermal-equilibrium roots over a Sigma grid."""

    if R_cm <= 0.0:
        raise ValueError("R_cm must be positive")
    if M2_g <= 0.0:
        raise ValueError("M2_g must be positive")
    if T_bounds[0] <= 0.0 or T_bounds[1] <= T_bounds[0]:
        raise ValueError("T_bounds must be positive and increasing")
    if n_T_brackets < 3:
        raise ValueError("n_T_brackets must be at least 3")

    Sigma_values = np.asarray(Sigma_grid, dtype=float)
    if np.any(Sigma_values <= 0.0):
        raise ValueError("Sigma_grid values must be positive")

    rows = []
    Omega = omega_k(M2_g, R_cm)
    for Sigma in Sigma_values:
        roots = _roots_for_sigma(
            float(Sigma),
            R_cm,
            M2_g,
            params,
            T_bounds,
            n_T_brackets,
            max_iter,
            rtol,
        )
        for branch_index, T in enumerate(roots):
            state = vertical_state(Sigma, T, M2_g, R_cm, mu_mol=params.mu_mol, kappa=params.kappa)
            Q_plus = q_plus_alpha(Sigma, T, R_cm, M2_g, params.alpha, params.mu_stress, params.mu_mol)
            Q_rad = q_rad_minus(T, Sigma, params.kappa)
            rows.append(
                (
                    float(Sigma),
                    T,
                    float(state.H),
                    float(state.tau),
                    float(effective_temperature(Q_rad)),
                    float(accretion_rate_from_heating(Q_plus, Omega)),
                    _thermal_stability(T, float(Sigma), R_cm, M2_g, params),
                    branch_index,
                )
            )

    if not rows:
        empty = np.array([], dtype=float)
        return ScurveResult(empty, empty, empty, empty, empty, empty, np.array([], dtype=bool), np.array([], dtype=int))

    data = np.array(rows, dtype=object)
    return ScurveResult(
        Sigma=data[:, 0].astype(float),
        T_c=data[:, 1].astype(float),
        H=data[:, 2].astype(float),
        tau=data[:, 3].astype(float),
        T_eff=data[:, 4].astype(float),
        mdot=data[:, 5].astype(float),
        stable=data[:, 6].astype(bool),
        branch_index=data[:, 7].astype(int),
    )


def find_turning_points(Sigma_grid, Mdot_grid) -> dict[str, np.ndarray | float | None]:
    """Locate local extrema of Mdot(Sigma) from slope sign changes."""

    Sigma = np.asarray(Sigma_grid, dtype=float)
    Mdot = np.asarray(Mdot_grid, dtype=float)
    if Sigma.shape != Mdot.shape:
        raise ValueError("Sigma_grid and Mdot_grid must have the same shape")
    if Sigma.ndim != 1:
        raise ValueError("inputs must be one-dimensional")
    if len(Sigma) < 3:
        return {"maxima": np.array([], dtype=int), "minima": np.array([], dtype=int), "Sigma_max": None, "Sigma_min": None}
    if np.any(Sigma <= 0.0) or np.any(Mdot <= 0.0):
        raise ValueError("Sigma_grid and Mdot_grid must be positive")

    order = np.argsort(Sigma)
    Sigma_sorted = Sigma[order]
    Mdot_sorted = Mdot[order]
    slopes = np.diff(np.log(Mdot_sorted)) / np.diff(np.log(Sigma_sorted))

    maxima = []
    minima = []
    for idx in range(len(slopes) - 1):
        left = slopes[idx]
        right = slopes[idx + 1]
        if left > 0.0 and right < 0.0:
            maxima.append(idx + 1)
        elif left < 0.0 and right > 0.0:
            minima.append(idx + 1)

    maxima = np.array(maxima, dtype=int)
    minima = np.array(minima, dtype=int)
    return {
        "maxima": order[maxima],
        "minima": order[minima],
        "Sigma_max": float(Sigma_sorted[maxima[-1]]) if len(maxima) else None,
        "Sigma_min": float(Sigma_sorted[minima[0]]) if len(minima) else None,
    }

