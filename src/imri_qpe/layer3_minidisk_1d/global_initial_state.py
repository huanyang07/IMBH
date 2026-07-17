"""Conservative finite-volume mapping for global initial-value profiles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from imri_qpe.constants import DEFAULT_KAPPA_ES

from .global_signed_evolution import (
    GlobalConservativeState,
    evaluate_global_rusanov_profile,
    global_conservative_rhs,
    state_from_thermodynamic_primitives,
)
from .grid import make_log_grid
from .transonic_potential import PaczynskiWiitaPotential
from .transonic_thermo import vertical_state


@dataclass(frozen=True)
class GlobalConstantPressureStartupAudit:
    """Construction diagnostics for one fresh radial-force equilibrium."""

    integrated_pressure: float
    total_mass: float
    minimum_scattering_optical_depth: float
    maximum_scattering_optical_depth: float
    minimum_temperature: float
    maximum_temperature: float
    maximum_relative_pressure_defect: float
    maximum_relative_aspect_ratio_defect: float
    maximum_radial_speed_over_orbital_speed: float
    maximum_relative_rotation_correction: float


def construct_global_constant_pressure_startup(
    M_g: float,
    n_cells: int,
    *,
    inner_radius: float,
    outer_radius: float,
    aspect_ratio: float = 0.05,
    minimum_scattering_optical_depth: float = 10.0,
    viscous_drift_alpha: float = 0.0,
    temperature_bounds: tuple[float, float] = (1.0e3, 1.0e12),
    kappa: float = DEFAULT_KAPPA_ES,
):
    """Construct a fresh constant-integrated-pressure startup state.

    Constant ``Pi`` and ``Omega=Omega_K`` cancel the finite-volume pressure
    flux, cylindrical pressure source, centrifugal force, and gravity source
    when ``viscous_drift_alpha`` is zero. With drift enabled, a tiny rotation
    correction restores the same discrete radial balance. The state is an
    initial-value datum, not a claimed viscous or thermal steady solution.
    """

    if int(n_cells) != n_cells or n_cells < 2:
        raise ValueError("startup construction requires at least two cells")
    if not np.isfinite(inner_radius) or inner_radius <= 0.0:
        raise ValueError("inner radius must be positive and finite")
    if not np.isfinite(outer_radius) or outer_radius <= inner_radius:
        raise ValueError("outer radius must exceed the inner radius")
    if not np.isfinite(aspect_ratio) or not 0.0 < aspect_ratio < 1.0:
        raise ValueError("aspect ratio must lie strictly between zero and one")
    if (
        not np.isfinite(minimum_scattering_optical_depth)
        or minimum_scattering_optical_depth <= 0.0
    ):
        raise ValueError("minimum optical depth must be positive and finite")
    if not np.isfinite(kappa) or kappa <= 0.0:
        raise ValueError("opacity must be positive and finite")
    if (
        not np.isfinite(viscous_drift_alpha)
        or not 0.0 <= viscous_drift_alpha <= 1.0
    ):
        raise ValueError("viscous drift alpha must lie between zero and one")
    lower_temperature, upper_temperature = map(float, temperature_bounds)
    if not 0.0 < lower_temperature < upper_temperature:
        raise ValueError("temperature bounds must be positive and ordered")

    grid = make_log_grid(inner_radius, outer_radius, int(n_cells))
    potential = PaczynskiWiitaPotential(float(M_g))
    omega = np.asarray(potential.omega_k(grid.centers), dtype=float)
    orbital_speed_squared = (grid.centers * omega) ** 2
    sigma_per_pressure = 1.0 / (
        aspect_ratio**2 * orbital_speed_squared
    )
    target_minimum_sigma = (
        2.0 * minimum_scattering_optical_depth / kappa
    )
    inner_orbital_speed_squared = float(
        (inner_radius * potential.omega_k(inner_radius)) ** 2
    )
    integrated_pressure = float(
        target_minimum_sigma
        * aspect_ratio**2
        * inner_orbital_speed_squared
    )
    sigma = integrated_pressure * sigma_per_pressure

    log_lower = float(np.log(lower_temperature))
    log_upper = float(np.log(upper_temperature))
    temperature = np.empty(grid.centers.size, dtype=float)
    for index, radius in enumerate(grid.centers):

        def pressure_residual(log_temperature: float) -> float:
            column = vertical_state(
                float(sigma[index]),
                float(np.exp(log_temperature)),
                float(radius),
                potential,
                kappa=kappa,
            )
            return float(column.Pi) - integrated_pressure

        lower_residual = pressure_residual(log_lower)
        upper_residual = pressure_residual(log_upper)
        if lower_residual > 0.0 or upper_residual < 0.0:
            raise ValueError(
                f"cell {index} pressure target lies outside temperature bounds"
            )
        temperature[index] = np.exp(
            brentq(
                pressure_residual,
                log_lower,
                log_upper,
                xtol=1.0e-12,
                rtol=1.0e-12,
            )
        )

    vertical = vertical_state(
        sigma,
        temperature,
        grid.centers,
        potential,
        kappa=kappa,
    )
    keplerian_omega = np.array(omega, copy=True)
    velocity = (
        -viscous_drift_alpha
        * aspect_ratio**2
        * grid.centers
        * keplerian_omega
    )
    correction = np.zeros(grid.centers.size, dtype=float)
    provisional = state_from_thermodynamic_primitives(
        grid,
        sigma,
        velocity,
        omega,
        temperature,
        M_g,
        kappa=kappa,
        specific_mechanical_energy_correction=correction,
    )
    if viscous_drift_alpha > 0.0:
        profile = evaluate_global_rusanov_profile(
            grid,
            provisional,
            M_g,
            reference_state=provisional,
            kappa=kappa,
            specific_mechanical_energy_correction=correction,
        )
        rhs = global_conservative_rhs(
            profile.face_fluxes, profile.cell_sources
        )
        omega_squared = keplerian_omega**2 - (
            rhs.radial_momentum / (grid.area * sigma * grid.centers)
        )
        if np.any(~np.isfinite(omega_squared)) or np.any(
            omega_squared <= 0.0
        ):
            raise ValueError("viscous drift has no positive radial balance")
        omega = np.sqrt(omega_squared)
    state = state_from_thermodynamic_primitives(
        grid,
        sigma,
        velocity,
        omega,
        temperature,
        M_g,
        kappa=kappa,
        specific_mechanical_energy_correction=correction,
    )
    optical_depth = 0.5 * kappa * sigma
    pressure_defect = np.abs(
        np.asarray(vertical.Pi, dtype=float) / integrated_pressure - 1.0
    )
    aspect_defect = np.abs(
        np.asarray(vertical.H, dtype=float)
        / (aspect_ratio * grid.centers)
        - 1.0
    )
    audit = GlobalConstantPressureStartupAudit(
        integrated_pressure=integrated_pressure,
        total_mass=float(np.sum(state.mass)),
        minimum_scattering_optical_depth=float(np.min(optical_depth)),
        maximum_scattering_optical_depth=float(np.max(optical_depth)),
        minimum_temperature=float(np.min(temperature)),
        maximum_temperature=float(np.max(temperature)),
        maximum_relative_pressure_defect=float(np.max(pressure_defect)),
        maximum_relative_aspect_ratio_defect=float(np.max(aspect_defect)),
        maximum_radial_speed_over_orbital_speed=float(
            np.max(
                np.abs(velocity)
                / (grid.centers * keplerian_omega)
            )
        ),
        maximum_relative_rotation_correction=float(
            np.max(np.abs(omega / keplerian_omega - 1.0))
        ),
    )
    return grid, state, correction, audit


def _interpolate_profile(query, nodes, values, *, positive: bool) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=float)
    values = np.asarray(values, dtype=float)
    query = np.asarray(query, dtype=float)
    work = np.log(values) if positive else values
    result = np.interp(query, nodes, work)
    left = query < nodes[0]
    right = query > nodes[-1]
    if np.any(left):
        slope = (work[1] - work[0]) / (nodes[1] - nodes[0])
        result[left] = work[0] + slope * (query[left] - nodes[0])
    if np.any(right):
        slope = (work[-1] - work[-2]) / (nodes[-1] - nodes[-2])
        result[right] = work[-1] + slope * (query[right] - nodes[-1])
    return np.exp(result) if positive else result


def conservatively_map_global_profile(
    radius,
    surface_density,
    radial_velocity,
    omega,
    temperature,
    M_g: float,
    n_cells: int,
    *,
    inner_radius: float | None = None,
    outer_radius: float | None = None,
    quadrature_order: int = 32,
):
    """Map one continuous primitive profile into global cell integrals."""

    radius = np.asarray(radius, dtype=float)
    sigma = np.asarray(surface_density, dtype=float)
    velocity = np.asarray(radial_velocity, dtype=float)
    omega = np.asarray(omega, dtype=float)
    temperature = np.asarray(temperature, dtype=float)
    if radius.ndim != 1 or radius.size < 2 or np.any(np.diff(radius) <= 0.0):
        raise ValueError("profile radius must be a strictly increasing vector")
    if any(value.shape != radius.shape for value in (
        sigma,
        velocity,
        omega,
        temperature,
    )):
        raise ValueError("all primitive profiles must match radius")
    if (
        np.any(~np.isfinite(radius))
        or np.any(~np.isfinite(velocity))
        or np.any(sigma <= 0.0)
        or np.any(omega <= 0.0)
        or np.any(temperature <= 0.0)
    ):
        raise ValueError("profile primitives must be finite and physical")
    if int(n_cells) < 2 or int(quadrature_order) < 2:
        raise ValueError("mapping requires at least two cells and quadrature nodes")
    inner = float(radius[0] if inner_radius is None else inner_radius)
    outer = float(radius[-1] if outer_radius is None else outer_radius)
    if inner < radius[0] or outer > radius[-1] or outer <= inner:
        raise ValueError("mapping bounds must lie inside the supplied profile")

    grid = make_log_grid(inner, outer, int(n_cells))
    potential = PaczynskiWiitaPotential(float(M_g))
    log_radius = np.log(radius)
    nodes, weights = np.polynomial.legendre.leggauss(int(quadrature_order))
    components = [np.empty(grid.centers.size, dtype=float) for _ in range(4)]
    mechanical_energy = np.empty(grid.centers.size, dtype=float)
    internal_energy = np.empty(grid.centers.size, dtype=float)
    for index, (left, right) in enumerate(
        zip(np.log(grid.edges[:-1]), np.log(grid.edges[1:]))
    ):
        x = 0.5 * (left + right) + 0.5 * (right - left) * nodes
        r = np.exp(x)
        sigma_q = _interpolate_profile(x, log_radius, sigma, positive=True)
        velocity_q = _interpolate_profile(
            x, log_radius, velocity, positive=False
        )
        omega_q = _interpolate_profile(x, log_radius, omega, positive=True)
        temperature_q = _interpolate_profile(
            x, log_radius, temperature, positive=True
        )
        vertical = vertical_state(sigma_q, temperature_q, r, potential)
        specific_mechanical = (
            np.asarray(potential.phi(r), dtype=float)
            + 0.5 * velocity_q**2
            + 0.5 * (r * omega_q) ** 2
        )
        measure = 2.0 * np.pi * r**2 * sigma_q
        specific_total = specific_mechanical + np.asarray(
            vertical.e, dtype=float
        )
        integrands = (
            measure,
            measure * velocity_q,
            measure * r**2 * omega_q,
            measure * specific_total,
        )
        width = 0.5 * (right - left)
        for component, integrand in zip(components, integrands):
            component[index] = width * np.sum(weights * integrand)
        mechanical_energy[index] = width * np.sum(
            weights * measure * specific_mechanical
        )
        internal_energy[index] = width * np.sum(
            weights * measure * np.asarray(vertical.e, dtype=float)
        )

    state = GlobalConservativeState(*components).validated()
    cell_velocity = state.radial_momentum / state.mass
    cell_omega = state.angular_momentum / (state.mass * grid.centers**2)
    center_mechanical = (
        np.asarray(potential.phi(grid.centers), dtype=float)
        + 0.5 * cell_velocity**2
        + 0.5 * (grid.centers * cell_omega) ** 2
    )
    correction = mechanical_energy / state.mass - center_mechanical
    if (
        np.any(~np.isfinite(correction))
        or np.any(internal_energy / state.mass <= 0.0)
    ):
        raise ValueError("mapped finite-volume energy is not physical")
    return grid, state, np.asarray(correction, dtype=float)
