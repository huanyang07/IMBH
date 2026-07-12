"""Outer-boundary operators for the conservative time-dependent DAE.

The time-dependent model needs an explicit distinction between a torque
reconstructed from the disk state and a torque assembled from transported
mass and angular-momentum fluxes.  This module keeps those contracts separate
and provides the zero-torque remap used by the open-edge audit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .grid import RadialGrid
from .signed_flux_thermal import SignedThermalClosure
from .transonic_potential import PaczynskiWiitaPotential
from .transonic_thermo import integrated_stress, vertical_state


@dataclass(frozen=True)
class ZeroTorqueRemapResult:
    """Thermodynamic state remapped through regular zero-torque coordinates."""

    surface_density: np.ndarray
    temperature: np.ndarray
    torque_centers: np.ndarray
    gas_pressure_fraction: np.ndarray
    maximum_inversion_residual: float


@dataclass(frozen=True)
class EliminatedBoundaryState:
    """Full primitives recovered from a zero-torque boundary chart."""

    surface_density: np.ndarray
    temperature: np.ndarray
    omega: np.ndarray
    torque_centers: np.ndarray
    gas_pressure_fraction_last: float
    inversion_residual: float


def _grid_values(name: str, values, grid: RadialGrid, *, positive: bool = False):
    values = np.asarray(values, dtype=float)
    if values.shape != grid.centers.shape or np.any(~np.isfinite(values)):
        raise ValueError(f"{name} must be finite and match the grid")
    if positive and np.any(values <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    return values


def linear_torque_faces(grid: RadialGrid, torque_centers) -> np.ndarray:
    """Linearly reconstruct torque to faces, including endpoint extrapolation.

    Linear reconstruction in physical radius exactly preserves the regular
    open-edge asymptotic ``G = amplitude * (R_out - R)``.  Logarithmic
    reconstruction cannot represent the exactly zero face value.
    """

    torque = _grid_values("torque_centers", torque_centers, grid)
    if torque.size < 2:
        raise ValueError("at least two torque centers are required")
    faces = np.interp(grid.edges, grid.centers, torque)
    left_slope = (torque[1] - torque[0]) / (
        grid.centers[1] - grid.centers[0]
    )
    right_slope = (torque[-1] - torque[-2]) / (
        grid.centers[-1] - grid.centers[-2]
    )
    faces[0] = torque[0] + left_slope * (grid.edges[0] - grid.centers[0])
    faces[-1] = torque[-1] + right_slope * (
        grid.edges[-1] - grid.centers[-1]
    )
    return np.asarray(faces, dtype=float)


def common_stress_torque_centers(
    grid: RadialGrid,
    surface_density,
    temperature,
    M_g: float,
    *,
    alpha: float,
    closure: SignedThermalClosure,
    mu_stress: float = 0.0,
    stress_factor: float = 1.0,
) -> np.ndarray:
    """Return the constitutive common-alpha torque at cell centers."""

    sigma = _grid_values(
        "surface_density", surface_density, grid, positive=True
    )
    temperature = _grid_values(
        "temperature", temperature, grid, positive=True
    )
    potential = PaczynskiWiitaPotential(float(M_g))
    state = vertical_state(
        sigma,
        temperature,
        grid.centers,
        potential,
        mu_mol=closure.mu_mol,
        kappa=closure.kappa,
        gamma_gas=closure.gamma_gas,
    )
    stress = integrated_stress(
        state,
        alpha,
        mu_stress=mu_stress,
        stress_factor=stress_factor,
    )
    return np.asarray(2.0 * np.pi * grid.centers**2 * stress, dtype=float)


def constitutive_outer_torque(
    grid: RadialGrid,
    surface_density,
    temperature,
    M_g: float,
    *,
    alpha: float,
    closure: SignedThermalClosure,
    mu_stress: float = 0.0,
    stress_factor: float = 1.0,
) -> float:
    """Return the state-dependent outer-face torque before imposing its BC."""

    centers = common_stress_torque_centers(
        grid,
        surface_density,
        temperature,
        M_g,
        alpha=alpha,
        closure=closure,
        mu_stress=mu_stress,
        stress_factor=stress_factor,
    )
    return float(linear_torque_faces(grid, centers)[-1])


def transport_outer_torque(
    mdot_out: float,
    specific_angular_momentum_out: float,
    angular_flux_out: float,
) -> float:
    """Return ``G_out = Mdot_out*l_out - J_out`` from transported fluxes."""

    values = np.asarray(
        [mdot_out, specific_angular_momentum_out, angular_flux_out],
        dtype=float,
    )
    if np.any(~np.isfinite(values)):
        raise ValueError("outer transport values must be finite")
    return float(mdot_out * specific_angular_momentum_out - angular_flux_out)


def regularized_zero_torque_remap(
    source_grid: RadialGrid,
    target_grid: RadialGrid,
    torque_centers,
) -> np.ndarray:
    """Remap ``G/(R_out-R)`` and restore the vanishing-stress profile."""

    if not np.isclose(
        source_grid.edges[-1],
        target_grid.edges[-1],
        rtol=1.0e-13,
        atol=0.0,
    ):
        raise ValueError("source and target grids must share the outer edge")
    torque = _grid_values("torque_centers", torque_centers, source_grid)
    source_distance = source_grid.edges[-1] - source_grid.centers
    target_distance = target_grid.edges[-1] - target_grid.centers
    if np.any(source_distance <= 0.0) or np.any(target_distance <= 0.0):
        raise ValueError("cell centers must lie inside the outer edge")
    regularized = torque / source_distance
    target_regularized = np.interp(
        np.log(target_grid.centers),
        np.log(source_grid.centers),
        regularized,
    )
    return np.asarray(target_regularized * target_distance, dtype=float)


def gas_pressure_fraction(state) -> np.ndarray:
    """Return ``P_gas/P_tot`` for a vertical-state object."""

    fraction = np.asarray(state.P_gas, dtype=float) / np.asarray(
        state.P_tot, dtype=float
    )
    if np.any(~np.isfinite(fraction)) or np.any(
        (fraction <= 0.0) | (fraction >= 1.0)
    ):
        raise ValueError("gas-pressure fraction must lie strictly in (0,1)")
    return fraction


def _logit(value):
    value = np.asarray(value, dtype=float)
    return np.log(value) - np.log1p(-value)


def pack_eliminated_boundary_coordinates(
    grid: RadialGrid,
    surface_density,
    temperature,
    omega,
    M_g: float,
    *,
    closure: SignedThermalClosure,
) -> np.ndarray:
    """Pack ``3N-1`` coordinates with ``beta`` replacing the last Sigma/T pair."""

    sigma = _grid_values(
        "surface_density", surface_density, grid, positive=True
    )
    temperature = _grid_values(
        "temperature", temperature, grid, positive=True
    )
    omega = _grid_values("omega", omega, grid, positive=True)
    potential = PaczynskiWiitaPotential(float(M_g))
    last_state = vertical_state(
        sigma[-1],
        temperature[-1],
        grid.centers[-1],
        potential,
        mu_mol=closure.mu_mol,
        kappa=closure.kappa,
        gamma_gas=closure.gamma_gas,
    )
    beta_last = float(last_state.P_gas / last_state.P_tot)
    return np.concatenate(
        (
            np.log(sigma[:-1]),
            np.log(temperature[:-1]),
            np.log(omega),
            [float(_logit(beta_last))],
        )
    )


def unpack_eliminated_boundary_coordinates(
    coordinates,
    grid: RadialGrid,
    M_g: float,
    *,
    alpha: float,
    closure: SignedThermalClosure,
    stress_factor: float = 1.0,
) -> EliminatedBoundaryState:
    """Recover full primitives while satisfying linear ``G_out=0`` exactly."""

    if alpha <= 0.0 or stress_factor <= 0.0:
        raise ValueError("alpha and stress_factor must be positive")
    coordinates = np.asarray(coordinates, dtype=float)
    n = grid.centers.size
    if coordinates.shape != (3 * n - 1,) or np.any(~np.isfinite(coordinates)):
        raise ValueError("eliminated coordinates have the wrong shape")
    sigma = np.empty(n, dtype=float)
    temperature = np.empty(n, dtype=float)
    sigma[:-1] = np.exp(coordinates[: n - 1])
    temperature[:-1] = np.exp(coordinates[n - 1 : 2 * (n - 1)])
    omega = np.exp(coordinates[2 * (n - 1) : 3 * n - 2])
    beta_last = float(1.0 / (1.0 + np.exp(-coordinates[-1])))
    prefix_grid = RadialGrid(
        centers=grid.centers[:-1],
        edges=grid.edges[:-1],
        widths=grid.widths[:-1],
        area=grid.area[:-1],
    )
    prefix_torque = common_stress_torque_centers(
        prefix_grid,
        sigma[:-1],
        temperature[:-1],
        M_g,
        alpha=alpha,
        closure=closure,
        mu_stress=0.0,
        stress_factor=stress_factor,
    )
    outer_distance = grid.edges[-1] - grid.centers[-1]
    previous_distance = grid.edges[-1] - grid.centers[-2]
    target_torque = float(prefix_torque[-1] * outer_distance / previous_distance)
    target_pi = target_torque / (
        2.0 * np.pi * alpha * stress_factor * grid.centers[-1] ** 2
    )
    recovered_sigma, recovered_temperature, residual = (
        recover_thermodynamics_from_pi_beta(
            [grid.centers[-1]],
            [target_pi],
            [beta_last],
            M_g,
            closure=closure,
            surface_density_seed=[sigma[-2]],
            temperature_seed=[temperature[-2]],
        )
    )
    sigma[-1] = recovered_sigma[0]
    temperature[-1] = recovered_temperature[0]
    torque = np.concatenate((prefix_torque, [target_torque]))
    return EliminatedBoundaryState(
        surface_density=sigma,
        temperature=temperature,
        omega=omega,
        torque_centers=torque,
        gas_pressure_fraction_last=beta_last,
        inversion_residual=residual,
    )


def recover_thermodynamics_from_pi_beta(
    radius,
    target_pi,
    target_beta,
    M_g: float,
    *,
    closure: SignedThermalClosure,
    surface_density_seed,
    temperature_seed,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Recover ``(Sigma,T)`` from integrated pressure and gas fraction."""

    radius = np.atleast_1d(np.asarray(radius, dtype=float))
    target_pi = np.atleast_1d(np.asarray(target_pi, dtype=float))
    target_beta = np.atleast_1d(np.asarray(target_beta, dtype=float))
    sigma_seed = np.atleast_1d(np.asarray(surface_density_seed, dtype=float))
    temperature_seed = np.atleast_1d(np.asarray(temperature_seed, dtype=float))
    shape = radius.shape
    if any(
        values.shape != shape
        for values in (target_pi, target_beta, sigma_seed, temperature_seed)
    ):
        raise ValueError("thermodynamic inversion arrays must share one shape")
    if (
        np.any(radius <= 0.0)
        or np.any(target_pi <= 0.0)
        or np.any((target_beta <= 0.0) | (target_beta >= 1.0))
        or np.any(sigma_seed <= 0.0)
        or np.any(temperature_seed <= 0.0)
    ):
        raise ValueError("thermodynamic inversion inputs are outside their domain")

    potential = PaczynskiWiitaPotential(float(M_g))
    sigma = np.empty(shape, dtype=float)
    temperature = np.empty(shape, dtype=float)
    maximum_residual = 0.0
    for index in np.ndindex(shape):
        local_radius = float(radius[index])
        local_pi = float(target_pi[index])
        local_beta = float(target_beta[index])

        def residual(log_values):
            local_sigma, local_temperature = np.exp(log_values)
            state = vertical_state(
                local_sigma,
                local_temperature,
                local_radius,
                potential,
                mu_mol=closure.mu_mol,
                kappa=closure.kappa,
                gamma_gas=closure.gamma_gas,
            )
            beta = float(state.P_gas / state.P_tot)
            return np.asarray(
                [
                    np.log(float(state.Pi) / local_pi),
                    _logit(beta) - _logit(local_beta),
                ],
                dtype=float,
            )

        result = least_squares(
            residual,
            np.log(
                [
                    float(sigma_seed[index]),
                    float(temperature_seed[index]),
                ]
            ),
            xtol=1.0e-12,
            ftol=1.0e-12,
            gtol=1.0e-12,
            max_nfev=100,
        )
        local_residual = float(np.max(np.abs(result.fun)))
        if not result.success or local_residual > 1.0e-9:
            raise RuntimeError(
                "pressure-beta thermodynamic inversion did not converge"
            )
        sigma[index], temperature[index] = np.exp(result.x)
        maximum_residual = max(maximum_residual, local_residual)
    return sigma, temperature, maximum_residual


def remap_zero_torque_thermodynamics(
    source_grid: RadialGrid,
    target_grid: RadialGrid,
    surface_density,
    temperature,
    M_g: float,
    *,
    alpha: float,
    closure: SignedThermalClosure,
    stress_factor: float = 1.0,
) -> ZeroTorqueRemapResult:
    """Remap a total-pressure-alpha state through ``(G/(Rout-R), beta)``."""

    if alpha <= 0.0 or stress_factor <= 0.0:
        raise ValueError("alpha and stress_factor must be positive")
    sigma = _grid_values(
        "surface_density", surface_density, source_grid, positive=True
    )
    temperature = _grid_values(
        "temperature", temperature, source_grid, positive=True
    )
    potential = PaczynskiWiitaPotential(float(M_g))
    state = vertical_state(
        sigma,
        temperature,
        source_grid.centers,
        potential,
        mu_mol=closure.mu_mol,
        kappa=closure.kappa,
        gamma_gas=closure.gamma_gas,
    )
    torque = common_stress_torque_centers(
        source_grid,
        sigma,
        temperature,
        M_g,
        alpha=alpha,
        closure=closure,
        mu_stress=0.0,
        stress_factor=stress_factor,
    )
    target_torque = regularized_zero_torque_remap(
        source_grid, target_grid, torque
    )
    target_pi = target_torque / (
        2.0 * np.pi * alpha * stress_factor * target_grid.centers**2
    )
    beta = gas_pressure_fraction(state)
    target_logit_beta = np.interp(
        np.log(target_grid.centers),
        np.log(source_grid.centers),
        _logit(beta),
    )
    target_beta = 1.0 / (1.0 + np.exp(-target_logit_beta))
    sigma_seed = np.exp(
        np.interp(
            np.log(target_grid.centers),
            np.log(source_grid.centers),
            np.log(sigma),
        )
    )
    temperature_seed = np.exp(
        np.interp(
            np.log(target_grid.centers),
            np.log(source_grid.centers),
            np.log(temperature),
        )
    )
    target_sigma, target_temperature, residual = (
        recover_thermodynamics_from_pi_beta(
            target_grid.centers,
            target_pi,
            target_beta,
            M_g,
            closure=closure,
            surface_density_seed=sigma_seed,
            temperature_seed=temperature_seed,
        )
    )
    return ZeroTorqueRemapResult(
        surface_density=target_sigma,
        temperature=target_temperature,
        torque_centers=target_torque,
        gas_pressure_fraction=target_beta,
        maximum_inversion_residual=residual,
    )
