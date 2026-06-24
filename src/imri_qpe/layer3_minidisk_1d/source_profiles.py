"""Mass-source profiles for stream deposition into the minidisk."""

from __future__ import annotations

import numpy as np

from .grid import RadialGrid


def _as_float_or_array(value):
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    return array


def normalize_surface_source(R, profile, Mdot_total: float, area_weights=None):
    """Normalize a non-negative surface source to a total mass rate.

    ``area_weights`` should be annular cell areas for finite-volume grids. If
    omitted, the function uses a trapezoidal approximation to
    ``int 2 pi R S(R) dR``.
    """

    if Mdot_total < 0.0:
        raise ValueError("Mdot_total must be non-negative")
    R = np.asarray(R, dtype=float)
    profile = np.asarray(profile, dtype=float)
    if R.shape != profile.shape:
        raise ValueError("R and profile must have the same shape")
    if np.any(R <= 0.0):
        raise ValueError("R values must be positive")
    if np.any(profile < 0.0):
        raise ValueError("profile must be non-negative")

    if area_weights is None:
        if R.size < 2:
            raise ValueError("at least two R values are required without area_weights")
        norm = np.trapezoid(2.0 * np.pi * R * profile, R)
    else:
        area_weights = np.asarray(area_weights, dtype=float)
        if area_weights.shape != profile.shape:
            raise ValueError("area_weights and profile must have the same shape")
        if np.any(area_weights <= 0.0):
            raise ValueError("area_weights must be positive")
        norm = np.sum(profile * area_weights)

    if norm <= 0.0:
        raise ValueError("profile integral must be positive")
    return _as_float_or_array(profile * Mdot_total / norm)


def gaussian_source(R, R_c: float, width: float, Mdot_cap: float):
    """Return S_m(R) normalized to total ``Mdot_cap``."""

    if R_c <= 0.0:
        raise ValueError("R_c must be positive")
    if width <= 0.0:
        raise ValueError("width must be positive")
    R = np.asarray(R, dtype=float)
    profile = np.exp(-0.5 * ((R - R_c) / width) ** 2)
    return normalize_surface_source(R, profile, Mdot_cap)


def gaussian_source_on_grid(grid: RadialGrid, R_c: float, width: float, Mdot_cap: float):
    """Return a Gaussian source normalized with exact annular cell areas."""

    if R_c <= 0.0:
        raise ValueError("R_c must be positive")
    if width <= 0.0:
        raise ValueError("width must be positive")
    profile = np.exp(-0.5 * ((grid.centers - R_c) / width) ** 2)
    return normalize_surface_source(grid.centers, profile, Mdot_cap, area_weights=grid.area)

