"""Exact physical-entropy kernel for the dynamic-height Legendre test.

The candidate conserved rest-frame chart is

``U_H = (Sigma, K, Z_H, P_H)``

where ``K`` excludes only the linear rest-mass energy, ``Z_H=Sigma*H`` and
``P_H=Sigma*w_H``.  The exact gas+radiation entropy is recovered after
subtracting vertical kinetic and gravitational energy.  A strictly convex
``-Sigma*s`` is necessary for a one-piece Godunov/Legendre completion.

This module is diagnostic only.  It does not advance a trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from imri_qpe.constants import A_RAD, C, DEFAULT_MU_MOL
from imri_qpe.scales import gas_constant_per_gram

from .entropy_advection import gas_radiation_specific_entropy


@dataclass(frozen=True)
class DynamicHeightEntropyState:
    """Recovered physical state and entropy in the height chart."""

    surface_mass: float
    thermal_plus_vertical_energy: float
    height_content: float
    vertical_momentum: float
    proper_half_thickness: float
    density: float
    temperature: float
    specific_internal_energy: float
    specific_entropy: float
    integrated_pressure: float
    vertical_energy: float
    mathematical_entropy: float


@dataclass(frozen=True)
class DynamicHeightEntropyHessian:
    """Scaled and diagonally equilibrated entropy Hessian."""

    coordinate_scales: np.ndarray
    centered_step_factor: float
    scaled_hessian: np.ndarray
    equilibrated_hessian: np.ndarray
    equilibrated_eigenvalues: np.ndarray
    equilibrated_eigenvectors: np.ndarray
    symmetry_defect: float


def _temperature_from_internal_energy(
    density: float,
    specific_internal_energy: float,
    temperature_seed: float,
    *,
    mu_mol: float,
    gamma_gas: float,
) -> float:
    rho = float(density)
    energy = float(specific_internal_energy)
    seed = float(temperature_seed)
    if rho <= 0.0 or energy <= 0.0 or seed <= 0.0:
        raise ValueError("height entropy recovery requires a physical state")
    gas_constant = gas_constant_per_gram(mu_mol)

    def residual(log_temperature: float) -> float:
        temperature = np.exp(log_temperature)
        return float(
            gas_constant * temperature / (gamma_gas - 1.0)
            + A_RAD * temperature**4 / rho
            - energy
        )

    center = np.log(seed)
    lower = center - 4.0
    upper = center + 4.0
    while residual(lower) > 0.0:
        lower -= 4.0
    while residual(upper) < 0.0:
        upper += 4.0
    temperature = np.exp(
        brentq(residual, lower, upper, xtol=1.0e-13, rtol=4.0e-15)
    )
    return float(temperature)


def dynamic_height_entropy_state(
    conserved,
    *,
    proper_vertical_frequency: float,
    temperature_seed: float,
    mu_mol: float = DEFAULT_MU_MOL,
    gamma_gas: float = 5.0 / 3.0,
) -> DynamicHeightEntropyState:
    """Recover exact gas+radiation entropy from ``(Sigma,K,Z_H,P_H)``."""

    values = np.asarray(conserved, dtype=float)
    if values.shape != (4,) or np.any(~np.isfinite(values)):
        raise ValueError("dynamic-height conserved chart must be finite length four")
    sigma, energy, height_content, vertical_momentum = map(float, values)
    omega = float(proper_vertical_frequency)
    if sigma <= 0.0 or height_content <= 0.0 or omega <= 0.0:
        raise ValueError("surface mass, height content and frequency must be positive")
    height = height_content / sigma
    density = sigma**2 / (2.0 * height_content)
    vertical_energy = (
        vertical_momentum**2 / (2.0 * sigma)
        + 0.5 * omega**2 * height_content**2 / sigma
    )
    specific_internal_energy = (energy - vertical_energy) / sigma
    temperature = _temperature_from_internal_energy(
        density,
        specific_internal_energy,
        temperature_seed,
        mu_mol=mu_mol,
        gamma_gas=gamma_gas,
    )
    gas_constant = gas_constant_per_gram(mu_mol)
    pressure = density * gas_constant * temperature + A_RAD * temperature**4 / 3.0
    integrated_pressure = 2.0 * height * pressure
    specific_entropy = gas_radiation_specific_entropy(
        density,
        temperature,
        mu_mol=mu_mol,
        gamma_gas=gamma_gas,
    )
    return DynamicHeightEntropyState(
        surface_mass=sigma,
        thermal_plus_vertical_energy=energy,
        height_content=height_content,
        vertical_momentum=vertical_momentum,
        proper_half_thickness=float(height),
        density=float(density),
        temperature=temperature,
        specific_internal_energy=float(specific_internal_energy),
        specific_entropy=float(specific_entropy),
        integrated_pressure=float(integrated_pressure),
        vertical_energy=float(vertical_energy),
        mathematical_entropy=float(-sigma * specific_entropy),
    )


def equilibrium_dynamic_height_conserved(
    *,
    surface_mass: float,
    temperature: float,
    proper_half_thickness: float,
    proper_vertical_frequency: float,
    mu_mol: float = DEFAULT_MU_MOL,
    gamma_gas: float = 5.0 / 3.0,
) -> np.ndarray:
    """Return the exact hydrostatic ``P_H=0`` candidate chart."""

    sigma = float(surface_mass)
    temp = float(temperature)
    height = float(proper_half_thickness)
    omega = float(proper_vertical_frequency)
    if min(sigma, temp, height, omega) <= 0.0:
        raise ValueError("equilibrium height inputs must be positive")
    density = sigma / (2.0 * height)
    gas_constant = gas_constant_per_gram(mu_mol)
    internal = (
        gas_constant * temp / (gamma_gas - 1.0)
        + A_RAD * temp**4 / density
    )
    height_content = sigma * height
    vertical = 0.5 * omega**2 * height_content**2 / sigma
    return np.asarray((sigma, sigma * internal + vertical, height_content, 0.0))


def default_dynamic_height_coordinate_scales(
    state: DynamicHeightEntropyState,
) -> np.ndarray:
    """Return the prospectively frozen linear coordinate scales."""

    thermal_energy = state.surface_mass * state.specific_internal_energy
    return np.asarray(
        (
            state.surface_mass,
            thermal_energy,
            state.height_content,
            state.surface_mass * C,
        ),
        dtype=float,
    )


def centered_dynamic_height_entropy_hessian(
    conserved,
    *,
    proper_vertical_frequency: float,
    temperature_seed: float,
    coordinate_scales=None,
    step_factor: float = 1.0e-3,
) -> DynamicHeightEntropyHessian:
    """Return a centered Hessian in a fixed linear scaled chart."""

    center = np.asarray(conserved, dtype=float)
    base = dynamic_height_entropy_state(
        center,
        proper_vertical_frequency=proper_vertical_frequency,
        temperature_seed=temperature_seed,
    )
    scales = (
        default_dynamic_height_coordinate_scales(base)
        if coordinate_scales is None
        else np.asarray(coordinate_scales, dtype=float)
    )
    if scales.shape != (4,) or np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("height Hessian scales must be positive finite length four")
    step = float(step_factor)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("height Hessian step factor must be positive")

    def entropy(offset: np.ndarray) -> float:
        return dynamic_height_entropy_state(
            center + scales * offset,
            proper_vertical_frequency=proper_vertical_frequency,
            temperature_seed=temperature_seed,
        ).mathematical_entropy

    zero = np.zeros(4)
    base_entropy = entropy(zero)
    hessian = np.empty((4, 4), dtype=float)
    for first in range(4):
        first_step = np.zeros(4)
        first_step[first] = step
        hessian[first, first] = (
            entropy(first_step) - 2.0 * base_entropy + entropy(-first_step)
        ) / step**2
        for second in range(first):
            second_step = np.zeros(4)
            second_step[second] = step
            value = (
                entropy(first_step + second_step)
                - entropy(first_step - second_step)
                - entropy(-first_step + second_step)
                + entropy(-first_step - second_step)
            ) / (4.0 * step**2)
            hessian[first, second] = value
            hessian[second, first] = value
    diagonal = np.diag(hessian)
    if np.any(diagonal <= 0.0):
        equilibrated = np.array(hessian, copy=True)
    else:
        root = np.sqrt(diagonal)
        equilibrated = hessian / np.outer(root, root)
    eigenvalues, eigenvectors = np.linalg.eigh(
        0.5 * (equilibrated + equilibrated.T)
    )
    symmetry_scale = max(float(np.max(np.abs(hessian))), np.finfo(float).tiny)
    symmetry = float(np.max(np.abs(hessian - hessian.T)) / symmetry_scale)
    return DynamicHeightEntropyHessian(
        coordinate_scales=np.array(scales, copy=True),
        centered_step_factor=step,
        scaled_hessian=hessian,
        equilibrated_hessian=equilibrated,
        equilibrated_eigenvalues=eigenvalues,
        equilibrated_eigenvectors=eigenvectors,
        symmetry_defect=symmetry,
    )


def height_force_identity_defect(
    state: DynamicHeightEntropyState,
    *,
    proper_vertical_frequency: float,
) -> float:
    """Return the hydrostatic pressure/gravity force residual."""

    pressure_force = state.integrated_pressure / state.proper_half_thickness
    gravity_force = (
        state.surface_mass
        * proper_vertical_frequency**2
        * state.proper_half_thickness
    )
    return float(
        abs(pressure_force - gravity_force)
        / max(abs(pressure_force), abs(gravity_force), np.finfo(float).tiny)
    )
