"""Common physical-validity gates for transonic/reservoir overlap searches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .transonic_potential import PaczynskiWiitaPotential


@dataclass(frozen=True)
class OverlapGateConfig:
    max_radial_pressure_fraction: float = 0.05
    min_dln_l_k_dln_R: float = 0.2
    max_H_over_R: float = 0.35
    max_radial_mach: float = 0.1
    min_tau_scattering: float = 10.0
    min_tau_effective: float = 1.0
    min_gradient_length_over_H: float = 3.0
    max_source_fraction: float = 1.0e-8


@dataclass(frozen=True)
class OverlapDiagnostics:
    radius: np.ndarray
    radial_pressure_fraction: np.ndarray
    dln_l_k_dln_R: np.ndarray
    H_over_R: np.ndarray
    radial_mach: np.ndarray
    tau_scattering: np.ndarray
    tau_absorption_low: np.ndarray
    tau_absorption_high: np.ndarray
    tau_effective_low: np.ndarray
    tau_effective_high: np.ndarray
    gradient_length_over_H: np.ndarray
    source_fraction: np.ndarray
    passes: np.ndarray


def kramers_absorption_opacity(
    density,
    temperature,
    *,
    coefficient: float = 6.4e22,
) -> np.ndarray:
    """Return diagnostic Kramers opacity ``coefficient*rho*T^-7/2``."""

    rho = np.asarray(density, dtype=float)
    temperature = np.asarray(temperature, dtype=float)
    if coefficient <= 0.0 or not np.isfinite(coefficient):
        raise ValueError("absorption coefficient must be positive and finite")
    if np.any(rho <= 0.0) or np.any(temperature <= 0.0):
        raise ValueError("density and temperature must be positive")
    return np.asarray(float(coefficient) * rho * temperature**-3.5, dtype=float)


def effective_optical_depth(tau_absorption, tau_scattering) -> np.ndarray:
    """Return ``sqrt(tau_abs*(tau_abs+tau_es))``."""

    tau_abs = np.asarray(tau_absorption, dtype=float)
    tau_es = np.asarray(tau_scattering, dtype=float)
    if np.any(tau_abs < 0.0) or np.any(tau_es < 0.0):
        raise ValueError("optical depths must be non-negative")
    return np.sqrt(tau_abs * (tau_abs + tau_es))


def overlap_diagnostics(
    radius,
    surface_density,
    temperature,
    H,
    density,
    radial_velocity,
    radial_pressure_fraction,
    M_g: float,
    *,
    tau_scattering,
    source_fraction=0.0,
    config: OverlapGateConfig | None = None,
) -> OverlapDiagnostics:
    """Evaluate common overlap metrics on one monotonically increasing grid."""

    config = OverlapGateConfig() if config is None else config
    radius = np.asarray(radius, dtype=float)
    arrays = [
        np.asarray(value, dtype=float) if np.asarray(value).ndim else np.full_like(radius, float(value))
        for value in (
            surface_density,
            temperature,
            H,
            density,
            radial_velocity,
            radial_pressure_fraction,
            tau_scattering,
            source_fraction,
        )
    ]
    sigma, temperature, H, rho, velocity, pressure_fraction, tau_es, source = arrays
    if any(value.shape != radius.shape for value in arrays):
        raise ValueError("overlap arrays must match the radius grid")
    if np.any(np.diff(radius) <= 0.0) or np.any(radius <= 0.0):
        raise ValueError("radius must be positive and increasing")
    potential = PaczynskiWiitaPotential(float(M_g))
    omega_k = np.asarray(potential.omega_k(radius), dtype=float)
    H_over_R = H / radius
    mach = np.abs(velocity) / (omega_k * H)
    dln_l = 2.0 + np.asarray(potential.dln_omega_k_dlnR(radius), dtype=float)
    logR = np.log(radius)
    maximum_slope = np.maximum.reduce(
        [
            np.abs(np.gradient(np.log(value), logR, edge_order=2))
            for value in (sigma, temperature, H)
        ]
    )
    gradient_length = 1.0 / np.maximum(H_over_R * maximum_slope, 1.0e-300)
    tau_abs_low = 0.5 * sigma * kramers_absorption_opacity(rho, temperature)
    tau_abs_high = 0.5 * sigma * kramers_absorption_opacity(rho, temperature, coefficient=5.0e24)
    tau_eff_low = effective_optical_depth(tau_abs_low, tau_es)
    tau_eff_high = effective_optical_depth(tau_abs_high, tau_es)
    passes = (
        (pressure_fraction <= config.max_radial_pressure_fraction)
        & (dln_l >= config.min_dln_l_k_dln_R)
        & (H_over_R <= config.max_H_over_R)
        & (mach <= config.max_radial_mach)
        & (tau_es >= config.min_tau_scattering)
        & (tau_eff_low >= config.min_tau_effective)
        & (gradient_length >= config.min_gradient_length_over_H)
        & (source <= config.max_source_fraction)
    )
    return OverlapDiagnostics(
        radius, pressure_fraction, dln_l, H_over_R, mach, tau_es,
        tau_abs_low, tau_abs_high, tau_eff_low, tau_eff_high,
        gradient_length, source, np.asarray(passes, dtype=bool),
    )


def contiguous_passing_bands(diagnostics: OverlapDiagnostics) -> list[tuple[float, float]]:
    """Return inclusive physical-radius bands whose cells pass every gate."""

    indices = np.flatnonzero(diagnostics.passes)
    if indices.size == 0:
        return []
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    return [(float(diagnostics.radius[g[0]]), float(diagnostics.radius[g[-1]])) for g in groups]


def intersect_bands(
    first: list[tuple[float, float]],
    second: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Return intersections between two ordered sets of closed bands."""

    intersections: list[tuple[float, float]] = []
    for first_left, first_right in first:
        for second_left, second_right in second:
            left = max(first_left, second_left)
            right = min(first_right, second_right)
            if left <= right:
                intersections.append((float(left), float(right)))
    return intersections
