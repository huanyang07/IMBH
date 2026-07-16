"""Conservative finite-volume mapping for global initial-value profiles."""

from __future__ import annotations

import numpy as np

from .global_signed_evolution import GlobalConservativeState
from .grid import make_log_grid
from .transonic_potential import PaczynskiWiitaPotential
from .transonic_thermo import vertical_state


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
