"""Simultaneous surface-density/temperature solve with shared alpha stress."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .grid import RadialGrid
from .interface_flux import ConservedInterfaceFlux
from .signed_flux_disk import SignedFluxTransport
from .signed_flux_thermal import SignedThermalClosure
from .signed_flux_total_energy import (
    SignedTotalEnergyProfile,
    signed_total_energy_profile,
)
from .transonic_potential import PaczynskiWiitaPotential
from .transonic_thermo import integrated_stress, vertical_state


@dataclass(frozen=True)
class CommonStressSteadyResult:
    """Root of the common-stress constitutive and total-energy equations."""

    surface_density: np.ndarray
    temperature: np.ndarray
    transport: SignedFluxTransport
    energy_profile: SignedTotalEnergyProfile
    stress_fraction: float
    accepted: bool
    nfev: int
    maximum_stress_residual: float
    maximum_energy_residual: float
    message: str


@dataclass(frozen=True)
class NonKeplerianCommonStressResult:
    """Simultaneous common-stress, radial-momentum, and energy root."""

    surface_density: np.ndarray
    temperature: np.ndarray
    omega: np.ndarray
    transport: SignedFluxTransport
    energy_profile: SignedTotalEnergyProfile
    radial_support_fraction: float
    accepted: bool
    nfev: int
    maximum_stress_residual: float
    maximum_radial_residual: float
    maximum_energy_residual: float
    minimum_dln_l_dln_R: float
    maximum_dln_omega_dln_R: float
    message: str


def common_alpha_stress_torque(
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
    """Return ``2*pi*R^2*W_alpha`` using the transonic stress routine."""

    potential = PaczynskiWiitaPotential(float(M_g))
    state = vertical_state(
        surface_density,
        temperature,
        grid.centers,
        potential,
        mu_mol=closure.mu_mol,
        kappa=closure.kappa,
        gamma_gas=closure.gamma_gas,
    )
    stress = np.asarray(
        integrated_stress(
            state,
            alpha,
            mu_stress=mu_stress,
            stress_factor=stress_factor,
        ),
        dtype=float,
    )
    return np.asarray(2.0 * np.pi * grid.centers**2 * stress, dtype=float)


def diffusive_alpha_torque(
    grid: RadialGrid,
    surface_density,
    temperature,
    M_g: float,
    *,
    alpha: float,
    closure: SignedThermalClosure,
) -> np.ndarray:
    """Return the legacy ``nu=alpha*H^2*Omega_K`` torque."""

    potential = PaczynskiWiitaPotential(float(M_g))
    state = vertical_state(
        surface_density,
        temperature,
        grid.centers,
        potential,
        mu_mol=closure.mu_mol,
        kappa=closure.kappa,
        gamma_gas=closure.gamma_gas,
    )
    omega = np.asarray(potential.omega_k(grid.centers), dtype=float)
    shear = np.asarray(
        omega * potential.dln_omega_k_dlnR(grid.centers) / grid.centers,
        dtype=float,
    )
    viscosity = alpha * np.asarray(state.H, dtype=float) ** 2 * omega
    return np.asarray(
        -2.0
        * np.pi
        * grid.centers**3
        * viscosity
        * np.asarray(surface_density, dtype=float)
        * shear,
        dtype=float,
    )


def _jacobian_sparsity(n: int):
    pattern = lil_matrix((2 * n, 2 * n), dtype=int)
    for cell in range(n):
        pattern[cell, cell] = 1
        pattern[cell, n + cell] = 1
        for neighbor in range(max(0, cell - 1), min(n, cell + 2)):
            pattern[n + cell, neighbor] = 1
            pattern[n + cell, n + neighbor] = 1
    return pattern.tocsr()


def _nonkeplerian_jacobian_sparsity(n: int):
    pattern = lil_matrix((3 * n, 3 * n), dtype=int)
    for equation_block in range(3):
        for cell in range(n):
            row = equation_block * n + cell
            radius = 0 if equation_block == 0 else 2
            for neighbor in range(max(0, cell - radius), min(n, cell + radius + 1)):
                pattern[row, neighbor] = 1
                pattern[row, n + neighbor] = 1
                pattern[row, 2 * n + neighbor] = 1
    return pattern.tocsr()


def _log_edge_values(grid: RadialGrid, values: np.ndarray) -> np.ndarray:
    log_centers = np.log(grid.centers)
    log_edges = np.log(grid.edges)
    edge = np.interp(log_edges, log_centers, values)
    if values.size > 1:
        left_slope = (values[1] - values[0]) / (
            log_centers[1] - log_centers[0]
        )
        right_slope = (values[-1] - values[-2]) / (
            log_centers[-1] - log_centers[-2]
        )
        edge[0] = values[0] + left_slope * (log_edges[0] - log_centers[0])
        edge[-1] = values[-1] + right_slope * (
            log_edges[-1] - log_centers[-1]
        )
    return edge


def _transport_with_rotation(
    grid: RadialGrid,
    template: SignedFluxTransport,
    surface_density: np.ndarray,
    omega: np.ndarray,
) -> SignedFluxTransport:
    log_omega_faces = _log_edge_values(grid, np.log(omega))
    omega_faces = np.exp(log_omega_faces)
    specific_l = grid.centers**2 * omega
    specific_l_faces = grid.edges**2 * omega_faces
    mdot_centers = 0.5 * (template.mdot_faces[:-1] + template.mdot_faces[1:])
    angular_centers = 0.5 * (
        template.angular_flux_faces[:-1] + template.angular_flux_faces[1:]
    )
    torque_centers = mdot_centers * specific_l - angular_centers
    torque_faces = (
        template.mdot_faces * specific_l_faces - template.angular_flux_faces
    )
    state_angular_rate = float(np.sum(template.mass_rate_cells * specific_l))
    return replace(
        template,
        surface_density=np.asarray(surface_density, dtype=float),
        # Viscosity is not a constitutive variable in this solve. The common
        # stress residual below supplies the physical torque closure.
        viscosity=template.viscosity,
        specific_angular_momentum=np.asarray(specific_l, dtype=float),
        omega=np.asarray(omega, dtype=float),
        omega_faces=np.asarray(omega_faces, dtype=float),
        viscous_torque_centers=np.asarray(torque_centers, dtype=float),
        viscous_torque_faces=np.asarray(torque_faces, dtype=float),
        angular_momentum_rate_from_state=state_angular_rate,
        angular_momentum_budget_defect=float(
            state_angular_rate - template.angular_momentum_budget_rate
        ),
    )


def solve_common_stress_total_energy_steady(
    grid: RadialGrid,
    template_transport: SignedFluxTransport,
    surface_density_seed,
    temperature_seed,
    M_g: float,
    *,
    alpha: float,
    closure: SignedThermalClosure,
    prescribed_inner_flux: ConservedInterfaceFlux | None = None,
    stress_fraction: float = 1.0,
    mu_stress: float = 0.0,
    stress_factor: float = 1.0,
    tolerance: float = 1.0e-7,
    max_nfev: int = 1000,
) -> CommonStressSteadyResult:
    """Solve common-stress constitutive and corrected total-energy rows."""

    if not 0.0 <= stress_fraction <= 1.0:
        raise ValueError("stress_fraction must lie in [0,1]")
    if alpha <= 0.0 or not np.isfinite(alpha):
        raise ValueError("alpha must be positive and finite")
    sigma_seed = np.asarray(surface_density_seed, dtype=float)
    temperature_seed = np.asarray(temperature_seed, dtype=float)
    if (
        sigma_seed.shape != grid.centers.shape
        or temperature_seed.shape != grid.centers.shape
        or np.any(sigma_seed <= 0.0)
        or np.any(temperature_seed <= 0.0)
    ):
        raise ValueError("positive seeds must match the radial grid")
    required_torque = np.asarray(
        template_transport.viscous_torque_centers,
        dtype=float,
    )
    torque_floor = 1.0e-8 * max(float(np.max(np.abs(required_torque))), 1.0)
    torque_scale = np.maximum(np.abs(required_torque), torque_floor)

    reference = signed_total_energy_profile(
        grid,
        replace(template_transport, surface_density=sigma_seed),
        temperature_seed,
        M_g,
        closure=closure,
        prescribed_inner_flux=prescribed_inner_flux,
    )
    reference_flux_difference = (
        reference.total_energy_flux_faces[1:]
        - reference.total_energy_flux_faces[:-1]
    )
    global_floor = max(
        float(np.sum(np.abs(reference.stream_energy_rate_cells))) / sigma_seed.size,
        float(np.sum(reference.radiative_loss_rate_cells)) / sigma_seed.size,
        1.0,
    )
    energy_scale = np.maximum(
        np.abs(reference_flux_difference)
        + np.abs(reference.vertical_work_rate_cells)
        + reference.radiative_loss_rate_cells
        + np.abs(reference.stream_energy_rate_cells)
        + np.abs(reference.external_power_rate_cells),
        1.0e-6 * global_floor,
    )

    def evaluate(state_vector):
        sigma = np.exp(state_vector[: sigma_seed.size])
        temperature = np.exp(state_vector[sigma_seed.size :])
        diffusive = diffusive_alpha_torque(
            grid,
            sigma,
            temperature,
            M_g,
            alpha=alpha,
            closure=closure,
        )
        common = common_alpha_stress_torque(
            grid,
            sigma,
            temperature,
            M_g,
            alpha=alpha,
            closure=closure,
            mu_stress=mu_stress,
            stress_factor=stress_factor,
        )
        modeled_torque = (
            (1.0 - float(stress_fraction)) * diffusive
            + float(stress_fraction) * common
        )
        effective_viscosity = template_transport.viscosity * (
            modeled_torque / np.maximum(diffusive, torque_floor)
        )
        transport = replace(
            template_transport,
            surface_density=np.asarray(sigma, dtype=float),
            viscosity=np.asarray(effective_viscosity, dtype=float),
        )
        energy = signed_total_energy_profile(
            grid,
            transport,
            temperature,
            M_g,
            closure=closure,
            prescribed_inner_flux=prescribed_inner_flux,
        )
        stress_residual = (modeled_torque - required_torque) / torque_scale
        energy_residual = energy.net_energy_rate_cells / energy_scale
        return stress_residual, energy_residual, transport, energy

    def residual(state_vector):
        stress, energy, _transport, _profile = evaluate(state_vector)
        return np.concatenate((stress, energy))

    lower_temperature, upper_temperature = closure.temperature_bounds
    lower = np.concatenate(
        (
            np.log(sigma_seed) - np.log(1.0e4),
            np.full(sigma_seed.size, np.log(lower_temperature)),
        )
    )
    upper = np.concatenate(
        (
            np.log(sigma_seed) + np.log(1.0e4),
            np.full(sigma_seed.size, np.log(upper_temperature)),
        )
    )
    initial = np.concatenate((np.log(sigma_seed), np.log(temperature_seed)))
    solved = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        jac_sparsity=_jacobian_sparsity(sigma_seed.size),
        x_scale="jac",
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-10,
        max_nfev=int(max_nfev),
    )
    stress_residual, energy_residual, transport, energy = evaluate(solved.x)
    maximum_stress = float(np.max(np.abs(stress_residual)))
    maximum_energy = float(np.max(np.abs(energy_residual)))
    return CommonStressSteadyResult(
        surface_density=np.exp(solved.x[: sigma_seed.size]),
        temperature=np.exp(solved.x[sigma_seed.size :]),
        transport=transport,
        energy_profile=energy,
        stress_fraction=float(stress_fraction),
        accepted=bool(
            maximum_stress <= tolerance and maximum_energy <= tolerance
        ),
        nfev=int(solved.nfev),
        maximum_stress_residual=maximum_stress,
        maximum_energy_residual=maximum_energy,
        message=str(solved.message),
    )


def solve_nonkeplerian_common_stress_steady(
    grid: RadialGrid,
    template_transport: SignedFluxTransport,
    surface_density_seed,
    temperature_seed,
    omega_seed,
    M_g: float,
    *,
    alpha: float,
    closure: SignedThermalClosure,
    prescribed_inner_flux: ConservedInterfaceFlux | None = None,
    radial_support_fraction: float = 1.0,
    mu_stress: float = 0.0,
    stress_factor: float = 1.0,
    tolerance: float = 1.0e-7,
    max_nfev: int = 2000,
) -> NonKeplerianCommonStressResult:
    """Solve one simultaneous non-Keplerian reservoir residual.

    ``radial_support_fraction`` provides the only continuation: zero enforces
    Keplerian rotation and one enforces the full radial-momentum equation.
    The mass and angular fluxes remain the exact conservative fluxes carried
    by ``template_transport``.
    """

    if not 0.0 <= radial_support_fraction <= 1.0:
        raise ValueError("radial_support_fraction must lie in [0,1]")
    if alpha <= 0.0 or not np.isfinite(alpha):
        raise ValueError("alpha must be positive and finite")
    n = grid.centers.size
    sigma_seed = np.asarray(surface_density_seed, dtype=float)
    temperature_seed = np.asarray(temperature_seed, dtype=float)
    omega_seed = np.asarray(omega_seed, dtype=float)
    if any(value.shape != grid.centers.shape for value in (
        sigma_seed,
        temperature_seed,
        omega_seed,
    )):
        raise ValueError("all seeds must match the radial grid")
    if any(np.any(value <= 0.0) for value in (
        sigma_seed,
        temperature_seed,
        omega_seed,
    )):
        raise ValueError("all seeds must be strictly positive")

    potential = PaczynskiWiitaPotential(float(M_g))
    omega_k = np.asarray(potential.omega_k(grid.centers), dtype=float)
    log_radius = np.log(grid.centers)
    edge_order = 2 if n > 2 else 1

    reference_transport = _transport_with_rotation(
        grid, template_transport, sigma_seed, omega_seed
    )
    reference_energy = signed_total_energy_profile(
        grid,
        reference_transport,
        temperature_seed,
        M_g,
        closure=closure,
        prescribed_inner_flux=prescribed_inner_flux,
    )
    reference_flux_difference = (
        reference_energy.total_energy_flux_faces[1:]
        - reference_energy.total_energy_flux_faces[:-1]
    )
    global_floor = max(
        float(np.sum(np.abs(reference_energy.stream_energy_rate_cells))) / n,
        float(np.sum(reference_energy.radiative_loss_rate_cells)) / n,
        1.0,
    )
    energy_scale = np.maximum(
        np.abs(reference_flux_difference)
        + np.abs(reference_energy.vertical_work_rate_cells)
        + reference_energy.radiative_loss_rate_cells
        + np.abs(reference_energy.stream_energy_rate_cells)
        + np.abs(reference_energy.external_power_rate_cells),
        1.0e-6 * global_floor,
    )
    reference_torque = np.asarray(
        reference_transport.viscous_torque_centers, dtype=float
    )
    torque_floor = 1.0e-8 * max(float(np.max(np.abs(reference_torque))), 1.0)
    torque_scale = np.maximum(np.abs(reference_torque), torque_floor)

    def evaluate(state_vector):
        sigma = np.exp(state_vector[:n])
        temperature = np.exp(state_vector[n : 2 * n])
        omega = np.exp(state_vector[2 * n :])
        transport = _transport_with_rotation(
            grid, template_transport, sigma, omega
        )
        common_torque = common_alpha_stress_torque(
            grid,
            sigma,
            temperature,
            M_g,
            alpha=alpha,
            closure=closure,
            mu_stress=mu_stress,
            stress_factor=stress_factor,
        )
        stress_residual = (
            common_torque - transport.viscous_torque_centers
        ) / torque_scale

        energy = signed_total_energy_profile(
            grid,
            transport,
            temperature,
            M_g,
            closure=closure,
            prescribed_inner_flux=prescribed_inner_flux,
        )
        radial_velocity = np.asarray(energy.radial_velocity, dtype=float)
        inertia = 0.5 * np.gradient(
            radial_velocity**2, log_radius, edge_order=edge_order
        )
        pressure_gradient = np.gradient(
            energy.vertically_integrated_pressure,
            log_radius,
            edge_order=edge_order,
        )
        radial_full = (
            inertia
            - grid.centers**2 * (omega**2 - omega_k**2)
            + pressure_gradient / sigma
        ) / (grid.centers**2 * omega_k**2)
        radial_residual = (
            (1.0 - float(radial_support_fraction)) * np.log(omega / omega_k)
            + float(radial_support_fraction) * radial_full
        )
        energy_residual = energy.net_energy_rate_cells / energy_scale
        return (
            stress_residual,
            radial_residual,
            energy_residual,
            transport,
            energy,
        )

    def residual(state_vector):
        stress, radial, energy, _transport, _profile = evaluate(state_vector)
        return np.concatenate((stress, radial, energy))

    lower_temperature, upper_temperature = closure.temperature_bounds
    lower = np.concatenate(
        (
            np.log(sigma_seed) - np.log(1.0e4),
            np.full(n, np.log(lower_temperature)),
            np.log(0.5 * omega_k),
        )
    )
    upper = np.concatenate(
        (
            np.log(sigma_seed) + np.log(1.0e4),
            np.full(n, np.log(upper_temperature)),
            np.log(1.2 * omega_k),
        )
    )
    initial = np.concatenate(
        (np.log(sigma_seed), np.log(temperature_seed), np.log(omega_seed))
    )
    jacobian_sparsity = (
        None if n <= 64 else _nonkeplerian_jacobian_sparsity(n)
    )
    solver_options = {}
    if jacobian_sparsity is not None:
        solver_options["tr_options"] = {
            "atol": 1.0e-12,
            "btol": 1.0e-12,
            "maxiter": max(100, 3 * n),
        }
    solved = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        jac_sparsity=jacobian_sparsity,
        x_scale="jac",
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-10,
        max_nfev=int(max_nfev),
        **solver_options,
    )
    stress, radial, energy_residual, transport, energy = evaluate(solved.x)
    sigma = np.exp(solved.x[:n])
    temperature = np.exp(solved.x[n : 2 * n])
    omega = np.exp(solved.x[2 * n :])
    dln_l = np.gradient(
        np.log(grid.centers**2 * omega), log_radius, edge_order=edge_order
    )
    dln_omega = np.gradient(np.log(omega), log_radius, edge_order=edge_order)
    maximum_stress = float(np.max(np.abs(stress)))
    maximum_radial = float(np.max(np.abs(radial)))
    maximum_energy = float(np.max(np.abs(energy_residual)))
    physical_slopes = bool(np.min(dln_l) > 0.0 and np.max(dln_omega) < 0.0)
    return NonKeplerianCommonStressResult(
        surface_density=sigma,
        temperature=temperature,
        omega=omega,
        transport=transport,
        energy_profile=energy,
        radial_support_fraction=float(radial_support_fraction),
        accepted=bool(
            maximum_stress <= tolerance
            and maximum_radial <= tolerance
            and maximum_energy <= tolerance
            and physical_slopes
        ),
        nfev=int(solved.nfev),
        maximum_stress_residual=maximum_stress,
        maximum_radial_residual=maximum_radial,
        maximum_energy_residual=maximum_energy,
        minimum_dln_l_dln_R=float(np.min(dln_l)),
        maximum_dln_omega_dln_R=float(np.max(dln_omega)),
        message=str(solved.message),
    )
