"""Small physical outer-domain prototype for time-dependent DAE rank audits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .grid import RadialGrid
from .signed_flux_thermal import SignedThermalClosure
from .time_dae_boundary import (
    common_stress_torque_centers,
    linear_torque_faces,
    unpack_eliminated_boundary_coordinates,
)
from .transonic_potential import PaczynskiWiitaPotential
from .transonic_thermo import radiative_cooling, vertical_state


@dataclass(frozen=True)
class OuterDAEProfile:
    """Storage, fluxes, and quasi-static residual for one outer DAE state."""

    surface_density: np.ndarray
    temperature: np.ndarray
    omega: np.ndarray
    radial_velocity: np.ndarray
    mass_cells: np.ndarray
    angular_momentum_cells: np.ndarray
    energy_cells: np.ndarray
    H: np.ndarray
    enthalpy: np.ndarray
    torque_centers: np.ndarray
    torque_faces: np.ndarray
    mdot_faces: np.ndarray
    angular_flux_faces: np.ndarray
    energy_flux_faces: np.ndarray
    mass_rhs: np.ndarray
    angular_rhs: np.ndarray
    energy_rhs: np.ndarray
    radial_residual: np.ndarray
    source_mass_rate_cells: np.ndarray
    source_angular_rate_cells: np.ndarray
    source_energy_rate_cells: np.ndarray
    radiative_loss_rate_cells: np.ndarray


@dataclass(frozen=True)
class OuterDAEInstantaneousResult:
    """Algebraic face flux compatible with one eliminated boundary state."""

    mdot_faces: np.ndarray
    profile: OuterDAEProfile
    accepted: bool
    maximum_residual: float
    nfev: int
    message: str


@dataclass(frozen=True)
class OuterDAEStepResult:
    """One fully implicit eliminated-boundary backward-Euler step."""

    coordinates: np.ndarray
    mdot_faces: np.ndarray
    profile: OuterDAEProfile
    accepted: bool
    maximum_residual: float
    nfev: int
    message: str


@dataclass(frozen=True)
class OuterDAELedgerAudit:
    """Independent global backward-Euler conservation defects."""

    mass_defect: float
    angular_momentum_defect: float
    energy_defect: float
    relative_mass_defect: float
    relative_angular_momentum_defect: float
    relative_energy_defect: float


@dataclass(frozen=True)
class FluxPrimaryOuterDAEProfile:
    """Outer profile whose torque is reconstructed from Mdot and angular flux."""

    profile: OuterDAEProfile
    common_torque_centers: np.ndarray
    stress_residual: np.ndarray


@dataclass(frozen=True)
class FluxPrimaryOuterDAEStepResult:
    """One fully implicit flux-primary backward-Euler outer step."""

    state: np.ndarray
    mdot_faces: np.ndarray
    angular_flux_faces: np.ndarray
    evaluation: FluxPrimaryOuterDAEProfile
    ledger: OuterDAELedgerAudit
    accepted: bool
    maximum_residual: float
    nfev: int
    message: str


@dataclass(frozen=True)
class OuterRadialBoundaryState:
    """Inner-edge state for cross-interface outer gradient stencils."""

    radius: float
    integrated_pressure: float
    radial_velocity: float
    blend_fraction: float = 1.0

    def __post_init__(self) -> None:
        values = np.asarray(
            [self.radius, self.integrated_pressure, self.radial_velocity],
            dtype=float,
        )
        if np.any(~np.isfinite(values)) or np.any(values[:2] <= 0.0):
            raise ValueError("outer radial boundary state is not physical")
        if not 0.0 <= self.blend_fraction <= 1.0:
            raise ValueError("boundary blend_fraction must lie in [0,1]")


def pack_outer_primitives(surface_density, temperature, omega) -> np.ndarray:
    """Pack full logarithmic outer primitives."""

    sigma = np.asarray(surface_density, dtype=float)
    temperature = np.asarray(temperature, dtype=float)
    omega = np.asarray(omega, dtype=float)
    if (
        sigma.ndim != 1
        or temperature.shape != sigma.shape
        or omega.shape != sigma.shape
        or np.any(sigma <= 0.0)
        or np.any(temperature <= 0.0)
        or np.any(omega <= 0.0)
    ):
        raise ValueError("outer primitives must be positive one-dimensional arrays")
    return np.concatenate((np.log(sigma), np.log(temperature), np.log(omega)))


def unpack_outer_primitives(state, grid: RadialGrid):
    """Unpack full logarithmic outer primitives."""

    state = np.asarray(state, dtype=float)
    n = grid.centers.size
    if state.shape != (3 * n,) or np.any(~np.isfinite(state)):
        raise ValueError("outer primitive state has the wrong shape")
    return (
        np.exp(state[:n]),
        np.exp(state[n : 2 * n]),
        np.exp(state[2 * n :]),
    )


def _positive_faces(grid: RadialGrid, values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    log_values = np.log(values)
    faces = np.interp(np.log(grid.edges), np.log(grid.centers), log_values)
    if values.size > 1:
        left = (log_values[1] - log_values[0]) / (
            np.log(grid.centers[1]) - np.log(grid.centers[0])
        )
        right = (log_values[-1] - log_values[-2]) / (
            np.log(grid.centers[-1]) - np.log(grid.centers[-2])
        )
        faces[0] = log_values[0] + left * (
            np.log(grid.edges[0]) - np.log(grid.centers[0])
        )
        faces[-1] = log_values[-1] + right * (
            np.log(grid.edges[-1]) - np.log(grid.centers[-1])
        )
    return np.exp(faces)


def _donor_faces(values, mdot_faces) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mdot_faces = np.asarray(mdot_faces, dtype=float)
    n = values.size
    donor = np.empty(n + 1, dtype=float)
    donor[0] = values[0]
    donor[-1] = values[-1]
    for face in range(1, n):
        donor[face] = values[face] if mdot_faces[face] >= 0.0 else values[face - 1]
    return donor


def _cell_source(values, size: int, name: str) -> np.ndarray:
    """Return one finite cell-integrated source array or zeros."""

    if values is None:
        return np.zeros(int(size), dtype=float)
    source = np.asarray(values, dtype=float)
    if source.shape != (int(size),) or np.any(~np.isfinite(source)):
        raise ValueError(f"{name} must be finite and match the cell grid")
    return np.asarray(source, dtype=float)


def _gradient_with_inner_boundary(
    values,
    log_radius,
    boundary_radius: float,
    boundary_value: float,
) -> np.ndarray:
    """Differentiate at outer centers using the physical interface as a ghost."""

    values = np.asarray(values, dtype=float)
    log_radius = np.asarray(log_radius, dtype=float)
    extended_radius = np.concatenate(([np.log(boundary_radius)], log_radius))
    extended_values = np.concatenate(([boundary_value], values))
    return np.asarray(
        np.gradient(extended_values, extended_radius, edge_order=2)[1:],
        dtype=float,
    )


def angular_fluxes_for_common_stress_open_edge(
    grid: RadialGrid,
    mdot_faces,
    omega,
    common_torque_centers,
) -> np.ndarray:
    """Construct face angular fluxes closing center stress and ``G_out=0``."""

    mdot_faces = np.asarray(mdot_faces, dtype=float)
    omega = np.asarray(omega, dtype=float)
    common_torque = np.asarray(common_torque_centers, dtype=float)
    n = grid.centers.size
    if (
        mdot_faces.shape != (n + 1,)
        or omega.shape != (n,)
        or common_torque.shape != (n,)
    ):
        raise ValueError("open-edge angular initialization has wrong shapes")
    specific_l = grid.centers**2 * omega
    omega_faces = _positive_faces(grid, omega)
    specific_l_out = grid.edges[-1] ** 2 * omega_faces[-1]
    mdot_centers = 0.5 * (mdot_faces[:-1] + mdot_faces[1:])

    def recurse(first):
        angular = np.empty(n + 1, dtype=float)
        angular[0] = first
        for cell in range(n):
            angular[cell + 1] = (
                2.0 * mdot_centers[cell] * specific_l[cell]
                - 2.0 * common_torque[cell]
                - angular[cell]
            )
        return angular

    zero = recurse(0.0)
    target_outer = mdot_faces[-1] * specific_l_out
    response = 1.0 if n % 2 == 0 else -1.0
    first = (target_outer - zero[-1]) / response
    return recurse(first)


def evaluate_outer_dae_profile(
    grid: RadialGrid,
    surface_density,
    temperature,
    omega,
    mdot_faces,
    M_g: float,
    *,
    alpha: float,
    closure: SignedThermalClosure,
    stress_factor: float = 1.0,
    source_mass_rate_cells=None,
    source_angular_rate_cells=None,
    source_energy_rate_cells=None,
    include_radiative_cooling: bool = False,
) -> OuterDAEProfile:
    """Evaluate the wind-free, source-free low-Mach outer DAE prototype."""

    sigma = np.asarray(surface_density, dtype=float)
    temperature = np.asarray(temperature, dtype=float)
    omega = np.asarray(omega, dtype=float)
    mdot_faces = np.asarray(mdot_faces, dtype=float)
    n = grid.centers.size
    if (
        sigma.shape != (n,)
        or temperature.shape != (n,)
        or omega.shape != (n,)
        or mdot_faces.shape != (n + 1,)
        or np.any(sigma <= 0.0)
        or np.any(temperature <= 0.0)
        or np.any(omega <= 0.0)
        or np.any(~np.isfinite(mdot_faces))
    ):
        raise ValueError("outer DAE profile inputs have incompatible shapes")
    potential = PaczynskiWiitaPotential(float(M_g))
    vertical = vertical_state(
        sigma,
        temperature,
        grid.centers,
        potential,
        mu_mol=closure.mu_mol,
        kappa=closure.kappa,
        gamma_gas=closure.gamma_gas,
    )
    torque_centers = common_stress_torque_centers(
        grid,
        sigma,
        temperature,
        M_g,
        alpha=alpha,
        closure=closure,
        mu_stress=0.0,
        stress_factor=stress_factor,
    )
    torque_faces = linear_torque_faces(grid, torque_centers)
    specific_l = grid.centers**2 * omega
    specific_energy = (
        potential.phi(grid.centers)
        + 0.5 * (grid.centers * omega) ** 2
        + np.asarray(vertical.e, dtype=float)
    )
    enthalpy = np.asarray(vertical.Pi, dtype=float) / sigma
    bernoulli = specific_energy + enthalpy
    donor_l = _donor_faces(specific_l, mdot_faces)
    donor_B = _donor_faces(bernoulli, mdot_faces)
    omega_faces = _positive_faces(grid, omega)
    angular_flux = mdot_faces * donor_l - torque_faces
    energy_flux = mdot_faces * donor_B - omega_faces * torque_faces
    mass = sigma * grid.area
    angular = mass * specific_l
    energy = mass * specific_energy
    source_mass = _cell_source(source_mass_rate_cells, n, "source mass")
    source_angular = _cell_source(
        source_angular_rate_cells, n, "source angular momentum"
    )
    source_energy = _cell_source(source_energy_rate_cells, n, "source energy")
    radiative = (
        np.asarray(radiative_cooling(vertical, kappa=closure.kappa), dtype=float)
        * grid.area
        if include_radiative_cooling
        else np.zeros(n, dtype=float)
    )
    mass_rhs = mdot_faces[1:] - mdot_faces[:-1] + source_mass
    angular_rhs = angular_flux[1:] - angular_flux[:-1] + source_angular
    log_radius = np.log(grid.centers)
    dlnH = np.gradient(np.log(vertical.H), log_radius, edge_order=2)
    mdot_centers = 0.5 * (mdot_faces[:-1] + mdot_faces[1:])
    radial_work = mdot_centers * enthalpy * dlnH
    energy_rhs = (
        energy_flux[1:]
        - energy_flux[:-1]
        + radial_work
        + source_energy
        - radiative
    )
    radial_velocity = -mdot_centers / (
        2.0 * np.pi * grid.centers * sigma
    )
    inertia = 0.5 * np.gradient(
        radial_velocity**2, log_radius, edge_order=2
    )
    pressure_gradient = np.gradient(
        vertical.Pi, log_radius, edge_order=2
    )
    omega_k = potential.omega_k(grid.centers)
    radial = (
        inertia
        - grid.centers**2 * (omega**2 - omega_k**2)
        + pressure_gradient / sigma
    ) / (grid.centers**2 * omega_k**2)
    return OuterDAEProfile(
        surface_density=sigma,
        temperature=temperature,
        omega=omega,
        radial_velocity=np.asarray(radial_velocity, dtype=float),
        mass_cells=np.asarray(mass, dtype=float),
        angular_momentum_cells=np.asarray(angular, dtype=float),
        energy_cells=np.asarray(energy, dtype=float),
        H=np.asarray(vertical.H, dtype=float),
        enthalpy=np.asarray(enthalpy, dtype=float),
        torque_centers=np.asarray(torque_centers, dtype=float),
        torque_faces=np.asarray(torque_faces, dtype=float),
        mdot_faces=np.asarray(mdot_faces, dtype=float),
        angular_flux_faces=np.asarray(angular_flux, dtype=float),
        energy_flux_faces=np.asarray(energy_flux, dtype=float),
        mass_rhs=np.asarray(mass_rhs, dtype=float),
        angular_rhs=np.asarray(angular_rhs, dtype=float),
        energy_rhs=np.asarray(energy_rhs, dtype=float),
        radial_residual=np.asarray(radial, dtype=float),
        source_mass_rate_cells=source_mass,
        source_angular_rate_cells=source_angular,
        source_energy_rate_cells=source_energy,
        radiative_loss_rate_cells=radiative,
    )


def evaluate_flux_primary_outer_dae_profile(
    grid: RadialGrid,
    surface_density,
    temperature,
    omega,
    mdot_faces,
    angular_flux_faces,
    M_g: float,
    *,
    alpha: float,
    closure: SignedThermalClosure,
    stress_factor: float = 1.0,
    source_mass_rate_cells=None,
    source_angular_rate_cells=None,
    source_energy_rate_cells=None,
    include_radiative_cooling: bool = False,
    inner_energy_flux: float | None = None,
    inner_radial_boundary: OuterRadialBoundaryState | None = None,
) -> FluxPrimaryOuterDAEProfile:
    """Evaluate the repository-compatible mixed flux/common-stress operator."""

    sigma = np.asarray(surface_density, dtype=float)
    temperature = np.asarray(temperature, dtype=float)
    omega = np.asarray(omega, dtype=float)
    mdot_faces = np.asarray(mdot_faces, dtype=float)
    angular_flux_faces = np.asarray(angular_flux_faces, dtype=float)
    n = grid.centers.size
    if (
        sigma.shape != (n,)
        or temperature.shape != (n,)
        or omega.shape != (n,)
        or mdot_faces.shape != (n + 1,)
        or angular_flux_faces.shape != (n + 1,)
        or np.any(sigma <= 0.0)
        or np.any(temperature <= 0.0)
        or np.any(omega <= 0.0)
        or np.any(~np.isfinite(mdot_faces))
        or np.any(~np.isfinite(angular_flux_faces))
    ):
        raise ValueError("flux-primary DAE inputs have incompatible shapes")
    potential = PaczynskiWiitaPotential(float(M_g))
    vertical = vertical_state(
        sigma,
        temperature,
        grid.centers,
        potential,
        mu_mol=closure.mu_mol,
        kappa=closure.kappa,
        gamma_gas=closure.gamma_gas,
    )
    common_torque = common_stress_torque_centers(
        grid,
        sigma,
        temperature,
        M_g,
        alpha=alpha,
        closure=closure,
        mu_stress=0.0,
        stress_factor=stress_factor,
    )
    specific_l = grid.centers**2 * omega
    omega_faces = _positive_faces(grid, omega)
    specific_l_faces = grid.edges**2 * omega_faces
    mdot_centers = 0.5 * (mdot_faces[:-1] + mdot_faces[1:])
    angular_centers = 0.5 * (
        angular_flux_faces[:-1] + angular_flux_faces[1:]
    )
    torque_centers = mdot_centers * specific_l - angular_centers
    torque_faces = mdot_faces * specific_l_faces - angular_flux_faces
    torque_scale = np.maximum(np.abs(common_torque), 1.0)
    stress_residual = (torque_centers - common_torque) / torque_scale
    specific_energy = (
        potential.phi(grid.centers)
        + 0.5 * (grid.centers * omega) ** 2
        + np.asarray(vertical.e, dtype=float)
    )
    enthalpy = np.asarray(vertical.Pi, dtype=float) / sigma
    bernoulli = specific_energy + enthalpy
    donor_B = _donor_faces(bernoulli, mdot_faces)
    energy_flux = mdot_faces * donor_B - omega_faces * torque_faces
    if inner_energy_flux is not None:
        if not np.isfinite(inner_energy_flux):
            raise ValueError("inner_energy_flux must be finite")
        energy_flux = np.array(energy_flux, copy=True)
        energy_flux[0] = float(inner_energy_flux)
    mass = sigma * grid.area
    angular = mass * specific_l
    energy = mass * specific_energy
    source_mass = _cell_source(source_mass_rate_cells, n, "source mass")
    source_angular = _cell_source(
        source_angular_rate_cells, n, "source angular momentum"
    )
    source_energy = _cell_source(source_energy_rate_cells, n, "source energy")
    radiative = (
        np.asarray(radiative_cooling(vertical, kappa=closure.kappa), dtype=float)
        * grid.area
        if include_radiative_cooling
        else np.zeros(n, dtype=float)
    )
    mass_rhs = mdot_faces[1:] - mdot_faces[:-1] + source_mass
    angular_rhs = (
        angular_flux_faces[1:] - angular_flux_faces[:-1] + source_angular
    )
    log_radius = np.log(grid.centers)
    dlnH = np.gradient(np.log(vertical.H), log_radius, edge_order=2)
    radial_work = mdot_centers * enthalpy * dlnH
    energy_rhs = (
        energy_flux[1:]
        - energy_flux[:-1]
        + radial_work
        + source_energy
        - radiative
    )
    radial_velocity = -mdot_centers / (
        2.0 * np.pi * grid.centers * sigma
    )
    if inner_radial_boundary is None:
        inertia = 0.5 * np.gradient(
            radial_velocity**2, log_radius, edge_order=2
        )
        pressure_gradient = np.gradient(
            vertical.Pi, log_radius, edge_order=2
        )
    else:
        outer_inertia = 0.5 * np.gradient(
            radial_velocity**2, log_radius, edge_order=2
        )
        interface_inertia = 0.5 * _gradient_with_inner_boundary(
            radial_velocity**2,
            log_radius,
            inner_radial_boundary.radius,
            inner_radial_boundary.radial_velocity**2,
        )
        outer_pressure_gradient = np.gradient(
            vertical.Pi, log_radius, edge_order=2
        )
        interface_pressure_gradient = _gradient_with_inner_boundary(
            vertical.Pi,
            log_radius,
            inner_radial_boundary.radius,
            inner_radial_boundary.integrated_pressure,
        )
        fraction = inner_radial_boundary.blend_fraction
        inertia = (
            (1.0 - fraction) * outer_inertia + fraction * interface_inertia
        )
        pressure_gradient = (
            (1.0 - fraction) * outer_pressure_gradient
            + fraction * interface_pressure_gradient
        )
    omega_k = potential.omega_k(grid.centers)
    radial = (
        inertia
        - grid.centers**2 * (omega**2 - omega_k**2)
        + pressure_gradient / sigma
    ) / (grid.centers**2 * omega_k**2)
    profile = OuterDAEProfile(
        surface_density=sigma,
        temperature=temperature,
        omega=omega,
        radial_velocity=np.asarray(radial_velocity, dtype=float),
        mass_cells=np.asarray(mass, dtype=float),
        angular_momentum_cells=np.asarray(angular, dtype=float),
        energy_cells=np.asarray(energy, dtype=float),
        H=np.asarray(vertical.H, dtype=float),
        enthalpy=np.asarray(enthalpy, dtype=float),
        torque_centers=np.asarray(torque_centers, dtype=float),
        torque_faces=np.asarray(torque_faces, dtype=float),
        mdot_faces=np.asarray(mdot_faces, dtype=float),
        angular_flux_faces=np.asarray(angular_flux_faces, dtype=float),
        energy_flux_faces=np.asarray(energy_flux, dtype=float),
        mass_rhs=np.asarray(mass_rhs, dtype=float),
        angular_rhs=np.asarray(angular_rhs, dtype=float),
        energy_rhs=np.asarray(energy_rhs, dtype=float),
        radial_residual=np.asarray(radial, dtype=float),
        source_mass_rate_cells=source_mass,
        source_angular_rate_cells=source_angular,
        source_energy_rate_cells=source_energy,
        radiative_loss_rate_cells=radiative,
    )
    return FluxPrimaryOuterDAEProfile(
        profile=profile,
        common_torque_centers=np.asarray(common_torque, dtype=float),
        stress_residual=np.asarray(stress_residual, dtype=float),
    )


def outer_storage_matrix(
    state,
    grid: RadialGrid,
    M_g: float,
    *,
    closure: SignedThermalClosure,
    relative_step: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the low-Mach storage matrix and natural row scales."""

    state = np.asarray(state, dtype=float)
    sigma, temperature, omega = unpack_outer_primitives(state, grid)
    n = grid.centers.size
    zero_flux = np.zeros(n + 1)
    base = evaluate_outer_dae_profile(
        grid,
        sigma,
        temperature,
        omega,
        zero_flux,
        M_g,
        alpha=1.0,
        closure=closure,
    )
    storage = np.zeros((3 * n, 3 * n), dtype=float)
    for column in range(3 * n):
        step = relative_step * max(1.0, abs(state[column]))
        plus = state.copy()
        minus = state.copy()
        plus[column] += step
        minus[column] -= step
        plus_profile = evaluate_outer_dae_profile(
            grid,
            *unpack_outer_primitives(plus, grid),
            zero_flux,
            M_g,
            alpha=1.0,
            closure=closure,
        )
        minus_profile = evaluate_outer_dae_profile(
            grid,
            *unpack_outer_primitives(minus, grid),
            zero_flux,
            M_g,
            alpha=1.0,
            closure=closure,
        )
        storage[:n, column] = (
            plus_profile.mass_cells - minus_profile.mass_cells
        ) / (2.0 * step)
        storage[n : 2 * n, column] = (
            plus_profile.angular_momentum_cells
            - minus_profile.angular_momentum_cells
        ) / (2.0 * step)
        energy_derivative = (
            plus_profile.energy_cells - minus_profile.energy_cells
        ) / (2.0 * step)
        dlnH = (
            np.log(plus_profile.H) - np.log(minus_profile.H)
        ) / (2.0 * step)
        storage[2 * n :, column] = (
            energy_derivative + base.mass_cells * base.enthalpy * dlnH
        )
    scales = np.concatenate(
        (
            np.maximum(np.abs(base.mass_cells), 1.0),
            np.maximum(np.abs(base.angular_momentum_cells), 1.0),
            np.maximum(
                np.abs(base.energy_cells)
                + np.abs(base.mass_cells * base.enthalpy),
                1.0,
            ),
        )
    )
    return storage, scales


def eliminated_boundary_tangent(
    coordinates,
    grid: RadialGrid,
    M_g: float,
    *,
    alpha: float,
    closure: SignedThermalClosure,
    stress_factor: float = 1.0,
    relative_step: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return full primitives and ``dq/dqhat`` for the boundary chart."""

    coordinates = np.asarray(coordinates, dtype=float)

    def full_state(values):
        recovered = unpack_eliminated_boundary_coordinates(
            values,
            grid,
            M_g,
            alpha=alpha,
            closure=closure,
            stress_factor=stress_factor,
        )
        return pack_outer_primitives(
            recovered.surface_density,
            recovered.temperature,
            recovered.omega,
        )

    state = full_state(coordinates)
    tangent = np.empty((state.size, coordinates.size), dtype=float)
    for column in range(coordinates.size):
        step = relative_step * max(1.0, abs(coordinates[column]))
        plus = coordinates.copy()
        minus = coordinates.copy()
        plus[column] += step
        minus[column] -= step
        tangent[:, column] = (full_state(plus) - full_state(minus)) / (
            2.0 * step
        )
    return state, tangent


def _eliminated_state_profile(
    coordinates,
    mdot_faces,
    grid: RadialGrid,
    M_g: float,
    *,
    alpha: float,
    closure: SignedThermalClosure,
    stress_factor: float,
):
    recovered = unpack_eliminated_boundary_coordinates(
        coordinates,
        grid,
        M_g,
        alpha=alpha,
        closure=closure,
        stress_factor=stress_factor,
    )
    profile = evaluate_outer_dae_profile(
        grid,
        recovered.surface_density,
        recovered.temperature,
        recovered.omega,
        mdot_faces,
        M_g,
        alpha=alpha,
        closure=closure,
        stress_factor=stress_factor,
    )
    return recovered, profile


def solve_eliminated_instantaneous_flux(
    coordinates,
    mdot_seed,
    grid: RadialGrid,
    M_g: float,
    *,
    alpha: float,
    closure: SignedThermalClosure,
    stress_factor: float = 1.0,
    tolerance: float = 1.0e-9,
    max_nfev: int = 200,
) -> OuterDAEInstantaneousResult:
    """Solve radial balance plus normal conservation for all face fluxes."""

    coordinates = np.asarray(coordinates, dtype=float)
    mdot_seed = np.asarray(mdot_seed, dtype=float)
    if mdot_seed.shape != grid.edges.shape:
        raise ValueError("mdot_seed must match the grid faces")
    mdot_scale = max(float(np.max(np.abs(mdot_seed))), 1.0)
    state, tangent = eliminated_boundary_tangent(
        coordinates,
        grid,
        M_g,
        alpha=alpha,
        closure=closure,
        stress_factor=stress_factor,
    )
    storage, row_scales = outer_storage_matrix(
        state,
        grid,
        M_g,
        closure=closure,
    )
    storage_tangent = storage / row_scales[:, None] @ tangent
    left_vectors = np.linalg.svd(storage_tangent, full_matrices=True)[0]
    normal = left_vectors[:, -1]
    reference, _profile = _eliminated_state_profile(
        coordinates,
        mdot_seed,
        grid,
        M_g,
        alpha=alpha,
        closure=closure,
        stress_factor=stress_factor,
    )
    reference_mass = float(np.sum(reference.surface_density * grid.area))
    time_scale = reference_mass / mdot_scale

    def residual(scaled_flux):
        _recovered, profile = _eliminated_state_profile(
            coordinates,
            mdot_scale * scaled_flux,
            grid,
            M_g,
            alpha=alpha,
            closure=closure,
            stress_factor=stress_factor,
        )
        rhs = np.concatenate(
            (profile.mass_rhs, profile.angular_rhs, profile.energy_rhs)
        )
        normal_residual = float(normal @ (time_scale * rhs / row_scales))
        return np.concatenate((profile.radial_residual, [normal_residual]))

    result = least_squares(
        residual,
        mdot_seed / mdot_scale,
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=max_nfev,
    )
    mdot_faces = mdot_scale * result.x
    _recovered, profile = _eliminated_state_profile(
        coordinates,
        mdot_faces,
        grid,
        M_g,
        alpha=alpha,
        closure=closure,
        stress_factor=stress_factor,
    )
    maximum = float(np.max(np.abs(residual(result.x))))
    accepted = bool(
        result.success
        and maximum <= tolerance
        and mdot_faces[0] >= 0.0
        and mdot_faces[-1] <= 0.0
    )
    return OuterDAEInstantaneousResult(
        mdot_faces=np.asarray(mdot_faces, dtype=float),
        profile=profile,
        accepted=accepted,
        maximum_residual=maximum,
        nfev=int(result.nfev),
        message=str(result.message),
    )


def advance_eliminated_outer_dae_backward_euler(
    coordinates,
    mdot_seed,
    grid: RadialGrid,
    M_g: float,
    dt: float,
    *,
    alpha: float,
    closure: SignedThermalClosure,
    stress_factor: float = 1.0,
    tolerance: float = 1.0e-8,
    max_nfev: int = 300,
) -> OuterDAEStepResult:
    """Advance the source-free low-Mach outer DAE by one implicit step."""

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    coordinates = np.asarray(coordinates, dtype=float)
    mdot_seed = np.asarray(mdot_seed, dtype=float)
    old_recovered, old_profile = _eliminated_state_profile(
        coordinates,
        mdot_seed,
        grid,
        M_g,
        alpha=alpha,
        closure=closure,
        stress_factor=stress_factor,
    )
    mdot_scale = max(float(np.max(np.abs(mdot_seed))), 1.0)
    mass_scale = np.maximum(
        dt
        * (
            np.abs(old_profile.mdot_faces[:-1])
            + np.abs(old_profile.mdot_faces[1:])
        ),
        1.0e-12 * np.abs(old_profile.mass_cells),
    )
    angular_scale = np.maximum(
        dt
        * (
            np.abs(old_profile.angular_flux_faces[:-1])
            + np.abs(old_profile.angular_flux_faces[1:])
        ),
        1.0e-12 * np.abs(old_profile.angular_momentum_cells),
    )
    old_flux_divergence = (
        old_profile.energy_flux_faces[1:]
        - old_profile.energy_flux_faces[:-1]
    )
    old_radial_work = old_profile.energy_rhs - old_flux_divergence
    energy_scale = np.maximum(
        dt
        * (
            np.abs(old_profile.energy_flux_faces[:-1])
            + np.abs(old_profile.energy_flux_faces[1:])
            + np.abs(old_radial_work)
        ),
        1.0e-12
        * (
            np.abs(old_profile.energy_cells)
            + np.abs(old_profile.mass_cells * old_profile.enthalpy)
        ),
    )
    initial = np.concatenate((coordinates, mdot_seed / mdot_scale))
    coordinate_size = coordinates.size

    def residual(trial):
        new_coordinates = trial[:coordinate_size]
        new_mdot = mdot_scale * trial[coordinate_size:]
        _new_recovered, new_profile = _eliminated_state_profile(
            new_coordinates,
            new_mdot,
            grid,
            M_g,
            alpha=alpha,
            closure=closure,
            stress_factor=stress_factor,
        )
        mass = (
            new_profile.mass_cells
            - old_profile.mass_cells
            - dt * new_profile.mass_rhs
        ) / mass_scale
        angular = (
            new_profile.angular_momentum_cells
            - old_profile.angular_momentum_cells
            - dt * new_profile.angular_rhs
        ) / angular_scale
        temporal_vertical_work = 0.5 * (
            new_profile.mass_cells * new_profile.enthalpy
            + old_profile.mass_cells * old_profile.enthalpy
        ) * np.log(new_profile.H / old_profile.H)
        energy = (
            new_profile.energy_cells
            - old_profile.energy_cells
            + temporal_vertical_work
            - dt * new_profile.energy_rhs
        ) / energy_scale
        return np.concatenate(
            (mass, angular, energy, new_profile.radial_residual)
        )

    result = least_squares(
        residual,
        initial,
        xtol=1.0e-11,
        ftol=1.0e-11,
        gtol=1.0e-11,
        max_nfev=max_nfev,
    )
    new_coordinates = np.asarray(result.x[:coordinate_size], dtype=float)
    new_mdot = np.asarray(
        mdot_scale * result.x[coordinate_size:], dtype=float
    )
    _new_recovered, new_profile = _eliminated_state_profile(
        new_coordinates,
        new_mdot,
        grid,
        M_g,
        alpha=alpha,
        closure=closure,
        stress_factor=stress_factor,
    )
    maximum = float(np.max(np.abs(residual(result.x))))
    accepted = bool(
        result.success
        and maximum <= tolerance
        and new_mdot[0] >= 0.0
        and new_mdot[-1] <= 0.0
    )
    return OuterDAEStepResult(
        coordinates=new_coordinates,
        mdot_faces=new_mdot,
        profile=new_profile,
        accepted=accepted,
        maximum_residual=maximum,
        nfev=int(result.nfev),
        message=str(result.message),
    )


def audit_outer_dae_backward_euler_ledgers(
    old_profile: OuterDAEProfile,
    new_profile: OuterDAEProfile,
    dt: float,
) -> OuterDAELedgerAudit:
    """Audit telescoped source-free mass, angular, and column-energy steps."""

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    if old_profile.mass_cells.shape != new_profile.mass_cells.shape:
        raise ValueError("old and new profiles must share one grid")
    mass_change = float(
        np.sum(new_profile.mass_cells - old_profile.mass_cells)
    )
    mass_boundary = float(
        dt
        * (
            new_profile.mdot_faces[-1]
            - new_profile.mdot_faces[0]
            + np.sum(new_profile.source_mass_rate_cells)
        )
    )
    angular_change = float(
        np.sum(
            new_profile.angular_momentum_cells
            - old_profile.angular_momentum_cells
        )
    )
    angular_boundary = float(
        dt
        * (
            new_profile.angular_flux_faces[-1]
            - new_profile.angular_flux_faces[0]
            + np.sum(new_profile.source_angular_rate_cells)
        )
    )
    temporal_vertical_work = float(
        np.sum(
            0.5
            * (
                new_profile.mass_cells * new_profile.enthalpy
                + old_profile.mass_cells * old_profile.enthalpy
            )
            * np.log(new_profile.H / old_profile.H)
        )
    )
    energy_change = float(
        np.sum(new_profile.energy_cells - old_profile.energy_cells)
        + temporal_vertical_work
    )
    flux_divergence = (
        new_profile.energy_flux_faces[1:]
        - new_profile.energy_flux_faces[:-1]
    )
    radial_work = (
        new_profile.energy_rhs
        - flux_divergence
        - new_profile.source_energy_rate_cells
        + new_profile.radiative_loss_rate_cells
    )
    energy_boundary = float(
        dt
        * (
            new_profile.energy_flux_faces[-1]
            - new_profile.energy_flux_faces[0]
            + np.sum(radial_work)
            + np.sum(new_profile.source_energy_rate_cells)
            - np.sum(new_profile.radiative_loss_rate_cells)
        )
    )
    mass_defect = mass_change - mass_boundary
    angular_defect = angular_change - angular_boundary
    energy_defect = energy_change - energy_boundary
    mass_scale = max(
        float(np.sum(np.abs(new_profile.mass_cells - old_profile.mass_cells))),
        float(
            dt
            * (
                abs(new_profile.mdot_faces[-1])
                + abs(new_profile.mdot_faces[0])
                + np.sum(np.abs(new_profile.source_mass_rate_cells))
            )
        ),
        1.0,
    )
    angular_scale = max(
        float(
            np.sum(
                np.abs(
                    new_profile.angular_momentum_cells
                    - old_profile.angular_momentum_cells
                )
            )
        ),
        float(
            dt
            * (
                abs(new_profile.angular_flux_faces[-1])
                + abs(new_profile.angular_flux_faces[0])
                + np.sum(np.abs(new_profile.source_angular_rate_cells))
            )
        ),
        1.0,
    )
    energy_scale = max(
        float(
            np.sum(
                np.abs(new_profile.energy_cells - old_profile.energy_cells)
            )
            + abs(temporal_vertical_work)
        ),
        float(
            dt
            * (
                abs(new_profile.energy_flux_faces[-1])
                + abs(new_profile.energy_flux_faces[0])
                + np.sum(np.abs(radial_work))
                + np.sum(np.abs(new_profile.source_energy_rate_cells))
                + np.sum(new_profile.radiative_loss_rate_cells)
            )
        ),
        1.0,
    )
    return OuterDAELedgerAudit(
        mass_defect=mass_defect,
        angular_momentum_defect=angular_defect,
        energy_defect=energy_defect,
        relative_mass_defect=abs(mass_defect) / mass_scale,
        relative_angular_momentum_defect=abs(angular_defect) / angular_scale,
        relative_energy_defect=abs(energy_defect) / energy_scale,
    )


def advance_flux_primary_outer_dae_backward_euler(
    state,
    mdot_faces,
    angular_flux_faces,
    grid: RadialGrid,
    M_g: float,
    dt: float,
    *,
    alpha: float,
    closure: SignedThermalClosure,
    stress_factor: float = 1.0,
    tolerance: float = 1.0e-8,
    mass_ledger_tolerance: float = 1.0e-9,
    energy_ledger_tolerance: float = 1.0e-8,
    max_nfev: int = 300,
) -> FluxPrimaryOuterDAEStepResult:
    """Advance the selected flux-primary, low-Mach outer DAE one step."""

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    state = np.asarray(state, dtype=float)
    mdot_faces = np.asarray(mdot_faces, dtype=float)
    angular_flux_faces = np.asarray(angular_flux_faces, dtype=float)
    n = grid.centers.size
    if (
        state.shape != (3 * n,)
        or mdot_faces.shape != (n + 1,)
        or angular_flux_faces.shape != (n + 1,)
    ):
        raise ValueError("flux-primary step inputs have incompatible shapes")
    sigma, temperature, omega = unpack_outer_primitives(state, grid)
    old = evaluate_flux_primary_outer_dae_profile(
        grid,
        sigma,
        temperature,
        omega,
        mdot_faces,
        angular_flux_faces,
        M_g,
        alpha=alpha,
        closure=closure,
        stress_factor=stress_factor,
    )
    old_profile = old.profile
    mdot_scale = max(float(np.max(np.abs(mdot_faces))), 1.0)
    angular_scale = max(float(np.max(np.abs(angular_flux_faces))), 1.0)
    torque_scale = max(float(np.max(np.abs(old.common_torque_centers))), 1.0)
    inner_angular_target = float(angular_flux_faces[0])
    mass_scale = np.maximum(
        dt
        * (
            np.abs(mdot_faces[:-1])
            + np.abs(mdot_faces[1:])
        ),
        1.0e-12 * np.abs(old_profile.mass_cells),
    )
    angular_row_scale = np.maximum(
        dt
        * (
            np.abs(angular_flux_faces[:-1])
            + np.abs(angular_flux_faces[1:])
        ),
        1.0e-12 * np.abs(old_profile.angular_momentum_cells),
    )
    old_flux_divergence = (
        old_profile.energy_flux_faces[1:]
        - old_profile.energy_flux_faces[:-1]
    )
    old_radial_work = old_profile.energy_rhs - old_flux_divergence
    energy_scale = np.maximum(
        dt
        * (
            np.abs(old_profile.energy_flux_faces[:-1])
            + np.abs(old_profile.energy_flux_faces[1:])
            + np.abs(old_radial_work)
        ),
        1.0e-12
        * (
            np.abs(old_profile.energy_cells)
            + np.abs(old_profile.mass_cells * old_profile.enthalpy)
        ),
    )
    initial = np.concatenate(
        (
            state,
            mdot_faces / mdot_scale,
            angular_flux_faces / angular_scale,
        )
    )

    def evaluate_trial(trial):
        local_state = trial[: 3 * n]
        local_mdot = mdot_scale * trial[3 * n : 4 * n + 1]
        local_angular = angular_scale * trial[4 * n + 1 :]
        local_sigma, local_temperature, local_omega = unpack_outer_primitives(
            local_state, grid
        )
        local = evaluate_flux_primary_outer_dae_profile(
            grid,
            local_sigma,
            local_temperature,
            local_omega,
            local_mdot,
            local_angular,
            M_g,
            alpha=alpha,
            closure=closure,
            stress_factor=stress_factor,
        )
        return local_state, local_mdot, local_angular, local

    def residual(trial):
        _local_state, _local_mdot, local_angular, local = evaluate_trial(trial)
        profile = local.profile
        mass = (
            profile.mass_cells
            - old_profile.mass_cells
            - dt * profile.mass_rhs
        ) / mass_scale
        angular = (
            profile.angular_momentum_cells
            - old_profile.angular_momentum_cells
            - dt * profile.angular_rhs
        ) / angular_row_scale
        temporal_vertical_work = 0.5 * (
            profile.mass_cells * profile.enthalpy
            + old_profile.mass_cells * old_profile.enthalpy
        ) * np.log(profile.H / old_profile.H)
        energy = (
            profile.energy_cells
            - old_profile.energy_cells
            + temporal_vertical_work
            - dt * profile.energy_rhs
        ) / energy_scale
        inner_angular = (
            local_angular[0] - inner_angular_target
        ) / angular_scale
        edge = profile.torque_faces[-1] / torque_scale
        return np.concatenate(
            (
                mass,
                angular,
                energy,
                local.stress_residual,
                profile.radial_residual,
                [inner_angular, edge],
            )
        )

    result = least_squares(
        residual,
        initial,
        xtol=1.0e-11,
        ftol=1.0e-11,
        gtol=1.0e-11,
        max_nfev=max_nfev,
    )
    new_state, new_mdot, new_angular, evaluation = evaluate_trial(result.x)
    maximum = float(np.max(np.abs(residual(result.x))))
    ledger = audit_outer_dae_backward_euler_ledgers(
        old_profile,
        evaluation.profile,
        dt,
    )
    accepted = bool(
        result.success
        and maximum <= tolerance
        and new_mdot[0] >= 0.0
        and new_mdot[-1] <= 0.0
        and ledger.relative_mass_defect <= mass_ledger_tolerance
        and ledger.relative_angular_momentum_defect <= mass_ledger_tolerance
        and ledger.relative_energy_defect <= energy_ledger_tolerance
    )
    return FluxPrimaryOuterDAEStepResult(
        state=np.asarray(new_state, dtype=float),
        mdot_faces=np.asarray(new_mdot, dtype=float),
        angular_flux_faces=np.asarray(new_angular, dtype=float),
        evaluation=evaluation,
        ledger=ledger,
        accepted=accepted,
        maximum_residual=maximum,
        nfev=int(result.nfev),
        message=str(result.message),
    )
