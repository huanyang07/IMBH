"""Prototype explicit viscous diffusion tools for a 1D minidisk."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import RadialGrid


@dataclass(frozen=True)
class DiffusionStepResult:
    """Output from one prescribed-S-curve diffusion step."""

    Sigma: np.ndarray
    is_hot: np.ndarray
    nu: np.ndarray
    rhs: np.ndarray


def _as_array_like_grid(name: str, value, grid: RadialGrid) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full_like(grid.centers, float(array), dtype=float)
    if array.shape != grid.centers.shape:
        raise ValueError(f"{name} must be scalar or match grid shape")
    return array


def mass_from_surface_density(grid: RadialGrid, Sigma) -> float:
    """Return disk mass from surface density on a finite-volume grid."""

    Sigma = _as_array_like_grid("Sigma", Sigma, grid)
    if np.any(Sigma < 0.0):
        raise ValueError("Sigma must be non-negative")
    return float(np.sum(Sigma * grid.area))


def viscous_diffusion_rhs(grid: RadialGrid, Sigma, nu, source=None, wind_loss=None) -> np.ndarray:
    """Return dSigma/dt for the thin-disk diffusion equation.

    The prototype uses zero-gradient flux boundaries:

    ``dSigma/dt = (3/R) d/dR [R^1/2 d(nu Sigma R^1/2)/dR] + source - wind``.
    """

    Sigma = _as_array_like_grid("Sigma", Sigma, grid)
    nu = _as_array_like_grid("nu", nu, grid)
    if np.any(Sigma < 0.0):
        raise ValueError("Sigma must be non-negative")
    if np.any(nu < 0.0):
        raise ValueError("nu must be non-negative")

    source_array = np.zeros_like(Sigma) if source is None else _as_array_like_grid("source", source, grid)
    wind_array = np.zeros_like(Sigma) if wind_loss is None else _as_array_like_grid("wind_loss", wind_loss, grid)
    if np.any(source_array < 0.0):
        raise ValueError("source must be non-negative")
    if np.any(wind_array < 0.0):
        raise ValueError("wind_loss must be non-negative")

    y = nu * Sigma * np.sqrt(grid.centers)
    flux_term = np.zeros(len(grid.edges), dtype=float)
    if len(grid.centers) > 1:
        dy = np.diff(y)
        dR = np.diff(grid.centers)
        flux_term[1:-1] = np.sqrt(grid.edges[1:-1]) * dy / dR

    divergence = 3.0 / grid.centers * np.diff(flux_term) / grid.widths
    return divergence + source_array - wind_array


def explicit_diffusion_step(grid: RadialGrid, Sigma, nu, dt: float, source=None, wind_loss=None, floor: float = 0.0):
    """Advance surface density by one explicit Euler diffusion step."""

    if dt < 0.0:
        raise ValueError("dt must be non-negative")
    if floor < 0.0:
        raise ValueError("floor must be non-negative")
    rhs = viscous_diffusion_rhs(grid, Sigma, nu, source=source, wind_loss=wind_loss)
    Sigma_new = np.asarray(Sigma, dtype=float) + dt * rhs
    return np.maximum(Sigma_new, floor)


def update_branch_state(Sigma, Sigma_min, Sigma_max, is_hot=None) -> np.ndarray:
    """Update hot/cold branch state with S-curve hysteresis.

    Cold cells switch hot at ``Sigma >= Sigma_max``. Hot cells remain hot until
    ``Sigma <= Sigma_min``.
    """

    Sigma = np.asarray(Sigma, dtype=float)
    Sigma_min = np.asarray(Sigma_min, dtype=float)
    Sigma_max = np.asarray(Sigma_max, dtype=float)
    if np.any(Sigma < 0.0):
        raise ValueError("Sigma must be non-negative")
    if np.any(Sigma_min < 0.0) or np.any(Sigma_max <= 0.0):
        raise ValueError("thresholds must be positive")
    if np.any(Sigma_min >= Sigma_max):
        raise ValueError("Sigma_min must be smaller than Sigma_max")

    if is_hot is None:
        return Sigma >= Sigma_max

    is_hot = np.asarray(is_hot, dtype=bool)
    if is_hot.shape != Sigma.shape:
        raise ValueError("is_hot must match Sigma shape")
    return np.where(is_hot, Sigma > Sigma_min, Sigma >= Sigma_max)


def prescribed_scurve_step(
    grid: RadialGrid,
    Sigma,
    is_hot,
    dt: float,
    nu_cold,
    nu_hot,
    Sigma_min,
    Sigma_max,
    source=None,
    wind_loss=None,
    floor: float = 0.0,
) -> DiffusionStepResult:
    """Advance one explicit step using a prescribed hot/cold viscosity switch."""

    branch_before = update_branch_state(Sigma, Sigma_min, Sigma_max, is_hot=is_hot)
    nu_cold = _as_array_like_grid("nu_cold", nu_cold, grid)
    nu_hot = _as_array_like_grid("nu_hot", nu_hot, grid)
    if np.any(nu_cold < 0.0) or np.any(nu_hot < 0.0):
        raise ValueError("viscosities must be non-negative")

    nu = np.where(branch_before, nu_hot, nu_cold)
    rhs = viscous_diffusion_rhs(grid, Sigma, nu, source=source, wind_loss=wind_loss)
    Sigma_new = np.maximum(np.asarray(Sigma, dtype=float) + dt * rhs, floor)
    branch_after = update_branch_state(Sigma_new, Sigma_min, Sigma_max, is_hot=branch_before)
    return DiffusionStepResult(Sigma=Sigma_new, is_hot=branch_after, nu=nu, rhs=rhs)

