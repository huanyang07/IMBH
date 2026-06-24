"""Audit metrics for global slim/wind minidisk profiles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .entropy_advection import radial_gradient


@dataclass(frozen=True)
class EnergyResidualMetrics:
    """Integrated and pointwise residual diagnostics."""

    signed: float
    L1: float
    L2: float
    max_abs: float
    max_abs_index: int
    max_abs_R: float | None
    max_abs_interior: float
    max_abs_interior_index: int
    max_abs_interior_R: float | None
    max_abs_is_boundary: bool


def _as_array(name: str, value) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _require_same_shape(*arrays: np.ndarray) -> None:
    shape = arrays[0].shape
    for array in arrays[1:]:
        if array.shape != shape:
            raise ValueError("all arrays must have the same shape")


def pointwise_energy_residual(Qplus, Qrad, Qadv, Qwind, floor: float = 1.0e-300) -> np.ndarray:
    """Return pointwise normalized energy residual."""

    if floor <= 0.0:
        raise ValueError("floor must be positive")
    Qplus = np.asarray(Qplus, dtype=float)
    Qrad = np.asarray(Qrad, dtype=float)
    Qadv = np.asarray(Qadv, dtype=float)
    Qwind = np.asarray(Qwind, dtype=float)
    if Qplus.shape != Qrad.shape or Qplus.shape != Qadv.shape or Qplus.shape != Qwind.shape:
        raise ValueError("energy arrays must have the same shape")

    raw = Qplus - Qrad - Qadv - Qwind
    denom = np.abs(Qplus) + np.abs(Qrad) + np.abs(Qadv) + np.abs(Qwind) + floor
    return raw / denom


def energy_residual_metrics(
    area,
    Qplus,
    Qrad,
    Qadv,
    Qwind,
    R=None,
    boundary_exclude: int = 0,
    floor: float = 1.0e-300,
) -> EnergyResidualMetrics:
    """Return signed, L1, L2, and max residual diagnostics."""

    if boundary_exclude < 0:
        raise ValueError("boundary_exclude must be non-negative")
    if floor <= 0.0:
        raise ValueError("floor must be positive")

    area = _as_array("area", area)
    Qplus = _as_array("Qplus", Qplus)
    Qrad = _as_array("Qrad", Qrad)
    Qadv = _as_array("Qadv", Qadv)
    Qwind = _as_array("Qwind", Qwind)
    _require_same_shape(area, Qplus, Qrad, Qadv, Qwind)
    if np.any(area <= 0.0):
        raise ValueError("area must be positive")

    if R is None:
        R_array = None
    else:
        R_array = _as_array("R", R)
        _require_same_shape(area, R_array)

    raw = Qplus - Qrad - Qadv - Qwind
    heat_norm = float(np.sum(np.abs(Qplus) * area) + floor)
    signed = float(np.sum(raw * area) / heat_norm)
    L1 = float(np.sum(np.abs(raw) * area) / heat_norm)
    L2_den = float(np.sqrt(np.sum(Qplus**2 * area)) + floor)
    L2 = float(np.sqrt(np.sum(raw**2 * area)) / L2_den)

    pointwise = np.abs(pointwise_energy_residual(Qplus, Qrad, Qadv, Qwind, floor=floor))
    max_abs_index = int(np.argmax(pointwise))
    max_abs = float(pointwise[max_abs_index])
    max_abs_R = None if R_array is None else float(R_array[max_abs_index])

    if boundary_exclude == 0 or 2 * boundary_exclude >= len(pointwise):
        interior = pointwise
        offset = 0
    else:
        interior = pointwise[boundary_exclude:-boundary_exclude]
        offset = boundary_exclude
    interior_local = int(np.argmax(interior))
    max_abs_interior_index = offset + interior_local
    max_abs_interior = float(pointwise[max_abs_interior_index])
    max_abs_interior_R = None if R_array is None else float(R_array[max_abs_interior_index])
    max_abs_is_boundary = max_abs_index != max_abs_interior_index

    return EnergyResidualMetrics(
        signed=signed,
        L1=L1,
        L2=L2,
        max_abs=max_abs,
        max_abs_index=max_abs_index,
        max_abs_R=max_abs_R,
        max_abs_interior=max_abs_interior,
        max_abs_interior_index=max_abs_interior_index,
        max_abs_interior_R=max_abs_interior_R,
        max_abs_is_boundary=max_abs_is_boundary,
    )


def energy_residual_metrics_from_profile(profile, boundary_exclude: int = 0, floor: float = 1.0e-300):
    """Return residual diagnostics from a ``GlobalSlimProfile``-like object."""

    return energy_residual_metrics(
        profile.area,
        profile.Q_visc,
        profile.Q_rad_limited,
        profile.Q_adv,
        profile.Q_wind,
        R=profile.R,
        boundary_exclude=boundary_exclude,
        floor=floor,
    )


def mdot_continuity_residual(R, Mdot, S_Sigma=0.0, dotSigma_w=0.0, gradient_method: str = "limited") -> np.ndarray:
    """Return residual of ``dMdot/dR = 2 pi R (dotSigma_w - S_Sigma)``."""

    R = _as_array("R", R)
    Mdot = _as_array("Mdot", Mdot)
    _require_same_shape(R, Mdot)
    if np.any(R <= 0.0):
        raise ValueError("R must be positive")
    S_Sigma = np.zeros_like(R) + np.asarray(S_Sigma, dtype=float)
    dotSigma_w = np.zeros_like(R) + np.asarray(dotSigma_w, dtype=float)
    _require_same_shape(R, S_Sigma, dotSigma_w)

    dMdot_dR = radial_gradient(Mdot, R, method=gradient_method)
    rhs = 2.0 * np.pi * R * (dotSigma_w - S_Sigma)
    return dMdot_dR - rhs


def normalized_mdot_continuity_residual(
    R,
    Mdot,
    S_Sigma=0.0,
    dotSigma_w=0.0,
    gradient_method: str = "limited",
    floor: float = 1.0e-300,
) -> np.ndarray:
    """Return a bounded normalized mass-continuity residual."""

    if floor <= 0.0:
        raise ValueError("floor must be positive")
    R = _as_array("R", R)
    Mdot = _as_array("Mdot", Mdot)
    S_Sigma = np.zeros_like(R) + np.asarray(S_Sigma, dtype=float)
    dotSigma_w = np.zeros_like(R) + np.asarray(dotSigma_w, dtype=float)
    _require_same_shape(R, Mdot, S_Sigma, dotSigma_w)
    dMdot_dR = radial_gradient(Mdot, R, method=gradient_method)
    rhs = 2.0 * np.pi * R * (dotSigma_w - S_Sigma)
    return (dMdot_dR - rhs) / (np.abs(dMdot_dR) + np.abs(rhs) + floor)
