"""Column total-energy closure for the angularly closed signed-flux disk."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .energy_identity import enthalpy_vertical_work
from .grid import RadialGrid
from .interface_flux import ConservedInterfaceFlux
from .signed_flux_disk import (
    SignedFluxBoundary,
    SignedFluxTransport,
    StreamInjectionState,
    solve_signed_flux_steady,
)
from .signed_flux_thermal import SignedThermalClosure
from .transonic_potential import PaczynskiWiitaPotential
from .transonic_thermo import radiative_cooling, vertical_state


@dataclass(frozen=True)
class SignedTotalEnergyProfile:
    """Total-energy flux and source decomposition for one column state."""

    temperature: np.ndarray
    radial_velocity: np.ndarray
    specific_enthalpy: np.ndarray
    column_bernoulli: np.ndarray
    column_total_energy_cells: np.ndarray
    advective_energy_flux_faces: np.ndarray
    torque_work_flux_faces: np.ndarray
    total_energy_flux_faces: np.ndarray
    vertical_work_rate_cells: np.ndarray
    radiative_loss_rate_cells: np.ndarray
    stream_energy_rate_cells: np.ndarray
    external_power_rate_cells: np.ndarray
    net_energy_rate_cells: np.ndarray
    H: np.ndarray
    rho: np.ndarray
    tau: np.ndarray
    vertically_integrated_pressure: np.ndarray
    radial_pressure_force_fraction: np.ndarray
    dln_l_k_dln_R: np.ndarray
    total_energy_equation_rate: float
    total_energy_telescoping_defect: float

    @property
    def total_energy_ledger_rate(self) -> float:
        """Backward-compatible alias for the global equation rate."""

        return self.total_energy_equation_rate

    @property
    def total_energy_ledger_defect(self) -> float:
        """Backward-compatible alias for the bookkeeping-only defect."""

        return self.total_energy_telescoping_defect


@dataclass(frozen=True)
class SignedTotalEnergySteadyResult:
    """Fixed-transport total-energy root."""

    temperature: np.ndarray
    profile: SignedTotalEnergyProfile
    accepted: bool
    nfev: int
    maximum_normalized_residual: float
    message: str


@dataclass(frozen=True)
class SignedTotalEnergyThermoviscousResult:
    """Alpha-viscosity fixed point using the total-energy compatibility row."""

    transport: SignedFluxTransport
    energy: SignedTotalEnergySteadyResult
    viscosity: np.ndarray
    converged: bool
    iterations: int
    maximum_log_viscosity_change: float
    history: np.ndarray


def _grid_array(name: str, values, grid: RadialGrid) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = np.full_like(grid.centers, float(array), dtype=float)
    if array.shape != grid.centers.shape or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be finite and match the radial grid")
    return array


def _positive_temperature(temperature, grid: RadialGrid) -> np.ndarray:
    array = _grid_array("temperature", temperature, grid)
    if np.any(array <= 0.0):
        raise ValueError("temperature must be positive")
    return array


def _edge_values(log_centers: np.ndarray, values: np.ndarray, log_edges: np.ndarray) -> np.ndarray:
    edge = np.interp(log_edges, log_centers, values)
    if values.size > 1:
        left_slope = (values[1] - values[0]) / (log_centers[1] - log_centers[0])
        right_slope = (values[-1] - values[-2]) / (
            log_centers[-1] - log_centers[-2]
        )
        edge[0] = values[0] + left_slope * (log_edges[0] - log_centers[0])
        edge[-1] = values[-1] + right_slope * (
            log_edges[-1] - log_centers[-1]
        )
    return edge


def _upwind_flux(mdot_faces: np.ndarray, specific_quantity: np.ndarray) -> np.ndarray:
    n = specific_quantity.size
    if mdot_faces[0] < 0.0:
        raise ValueError("inner-boundary outward inflow needs an energy state")
    if mdot_faces[-1] > 0.0:
        raise ValueError("outer-boundary inward flow needs an energy state")
    flux = np.empty(n + 1, dtype=float)
    flux[0] = mdot_faces[0] * specific_quantity[0]
    flux[-1] = mdot_faces[-1] * specific_quantity[-1]
    for face in range(1, n):
        donor = face if mdot_faces[face] >= 0.0 else face - 1
        flux[face] = mdot_faces[face] * specific_quantity[donor]
    return flux


def signed_vertical_work_rate_cells(
    grid: RadialGrid,
    mdot_centers,
    surface_density,
    pressure,
    integrated_pressure,
    density,
) -> np.ndarray:
    """Integrate the one-zone vertical-column work across each radial cell."""

    mdot = _grid_array("mdot_centers", mdot_centers, grid)
    sigma = _grid_array("surface_density", surface_density, grid)
    gas_pressure = _grid_array("pressure", pressure, grid)
    pi = _grid_array("integrated_pressure", integrated_pressure, grid)
    rho = _grid_array("density", density, grid)
    if np.any(sigma <= 0.0) or np.any(rho <= 0.0):
        raise ValueError("surface density and density must be positive")
    log_centers = np.log(grid.centers)
    log_edges = np.log(grid.edges)
    sigma_edges = np.exp(_edge_values(log_centers, np.log(sigma), log_edges))
    rho_edges = np.exp(_edge_values(log_centers, np.log(rho), log_edges))
    return np.asarray(
        enthalpy_vertical_work(
            mdot,
            sigma,
            pi,
            sigma_edges[1:] - sigma_edges[:-1],
            gas_pressure,
            rho,
            rho_edges[1:] - rho_edges[:-1],
        ),
        dtype=float,
    )


def signed_total_energy_profile(
    grid: RadialGrid,
    transport: SignedFluxTransport,
    temperature,
    M_g: float,
    *,
    closure: SignedThermalClosure | None = None,
    external_power_rate_cells=None,
    prescribed_inner_flux: ConservedInterfaceFlux | None = None,
) -> SignedTotalEnergyProfile:
    """Evaluate the total-energy compatibility ledger without viscous double-counting."""

    closure = SignedThermalClosure() if closure is None else closure
    temperature = _positive_temperature(temperature, grid)
    external_power = (
        np.zeros_like(grid.centers)
        if external_power_rate_cells is None
        else _grid_array("external_power_rate_cells", external_power_rate_cells, grid)
    )
    potential = PaczynskiWiitaPotential(float(M_g))
    state = vertical_state(
        transport.surface_density,
        temperature,
        grid.centers,
        potential,
        mu_mol=closure.mu_mol,
        kappa=closure.kappa,
        gamma_gas=closure.gamma_gas,
    )
    sigma = transport.surface_density
    mdot_centers = 0.5 * (transport.mdot_faces[:-1] + transport.mdot_faces[1:])
    radial_velocity = -mdot_centers / (2.0 * np.pi * grid.centers * sigma)
    integrated_pressure = np.asarray(state.Pi, dtype=float)
    enthalpy = np.asarray(state.e, dtype=float) + integrated_pressure / sigma
    specific_l = transport.specific_angular_momentum
    orbital = np.asarray(
        potential.phi(grid.centers) + 0.5 * (specific_l / grid.centers) ** 2,
        dtype=float,
    )
    bernoulli = orbital + enthalpy + 0.5 * radial_velocity**2
    total_energy_cells = sigma * (
        orbital + np.asarray(state.e, dtype=float) + 0.5 * radial_velocity**2
    ) * grid.area

    advective_flux = _upwind_flux(transport.mdot_faces, bernoulli)
    omega_edges = np.asarray(potential.omega_k(grid.edges), dtype=float)
    torque_work_flux = -omega_edges * transport.viscous_torque_faces
    if prescribed_inner_flux is not None:
        mass_scale = max(abs(prescribed_inner_flux.mdot), 1.0)
        angular_scale = max(abs(prescribed_inner_flux.angular_momentum), 1.0)
        if (
            abs(transport.mdot_faces[0] - prescribed_inner_flux.mdot)
            > 1.0e-12 * mass_scale
        ):
            raise ValueError("transport inner mass flux does not match prescribed flux")
        if (
            abs(
                transport.angular_flux_faces[0]
                - prescribed_inner_flux.angular_momentum
            )
            > 1.0e-12 * angular_scale
        ):
            raise ValueError("transport inner angular flux does not match prescribed flux")
        advective_flux[0] = prescribed_inner_flux.total_energy - torque_work_flux[0]
    total_flux = advective_flux + torque_work_flux

    rho = np.asarray(state.rho, dtype=float)
    vertical_work = signed_vertical_work_rate_cells(
        grid,
        mdot_centers,
        sigma,
        state.P_tot,
        integrated_pressure,
        rho,
    )
    radiative = np.asarray(
        radiative_cooling(state, kappa=closure.kappa), dtype=float
    ) * grid.area
    stream_energy = transport.source_total_energy_rate_cells
    net = (
        total_flux[1:]
        - total_flux[:-1]
        + vertical_work
        - radiative
        + stream_energy
        + external_power
    )

    edge_order = 2 if grid.centers.size > 2 else 1
    pressure_gradient = np.gradient(
        integrated_pressure, grid.centers, edge_order=edge_order
    )
    radial_pressure_fraction = np.abs(pressure_gradient / sigma) / (
        grid.centers * transport.omega**2
    )
    dln_l = 2.0 + np.asarray(
        potential.dln_omega_k_dlnR(grid.centers), dtype=float
    )
    ledger_rate = float(
        total_flux[-1]
        - total_flux[0]
        + np.sum(vertical_work - radiative + stream_energy + external_power)
    )
    return SignedTotalEnergyProfile(
        temperature=temperature,
        radial_velocity=np.asarray(radial_velocity, dtype=float),
        specific_enthalpy=np.asarray(enthalpy, dtype=float),
        column_bernoulli=np.asarray(bernoulli, dtype=float),
        column_total_energy_cells=np.asarray(total_energy_cells, dtype=float),
        advective_energy_flux_faces=np.asarray(advective_flux, dtype=float),
        torque_work_flux_faces=np.asarray(torque_work_flux, dtype=float),
        total_energy_flux_faces=np.asarray(total_flux, dtype=float),
        vertical_work_rate_cells=np.asarray(vertical_work, dtype=float),
        radiative_loss_rate_cells=np.asarray(radiative, dtype=float),
        stream_energy_rate_cells=np.asarray(stream_energy, dtype=float),
        external_power_rate_cells=np.asarray(external_power, dtype=float),
        net_energy_rate_cells=np.asarray(net, dtype=float),
        H=np.asarray(state.H, dtype=float),
        rho=rho,
        tau=np.asarray(state.tau, dtype=float),
        vertically_integrated_pressure=integrated_pressure,
        radial_pressure_force_fraction=np.asarray(
            radial_pressure_fraction, dtype=float
        ),
        dln_l_k_dln_R=np.asarray(dln_l, dtype=float),
        total_energy_equation_rate=ledger_rate,
        total_energy_telescoping_defect=float(np.sum(net) - ledger_rate),
    )


def _jacobian_sparsity(n: int):
    pattern = lil_matrix((n, n), dtype=int)
    for row in range(n):
        for column in range(max(0, row - 1), min(n, row + 2)):
            pattern[row, column] = 1
    return pattern.tocsr()


def solve_signed_total_energy_steady(
    grid: RadialGrid,
    transport: SignedFluxTransport,
    temperature_seed,
    M_g: float,
    *,
    closure: SignedThermalClosure | None = None,
    external_power_rate_cells=None,
    prescribed_inner_flux: ConservedInterfaceFlux | None = None,
    tolerance: float = 1.0e-6,
    max_nfev: int = 500,
) -> SignedTotalEnergySteadyResult:
    """Solve the fixed-transport total-energy compatibility equation."""

    closure = SignedThermalClosure() if closure is None else closure
    seed = _positive_temperature(temperature_seed, grid)
    lower, upper = closure.temperature_bounds
    reference = signed_total_energy_profile(
        grid,
        transport,
        seed,
        M_g,
        closure=closure,
        external_power_rate_cells=external_power_rate_cells,
        prescribed_inner_flux=prescribed_inner_flux,
    )
    reference_flux_difference = (
        reference.total_energy_flux_faces[1:]
        - reference.total_energy_flux_faces[:-1]
    )
    global_floor = max(
        float(np.sum(np.abs(reference.stream_energy_rate_cells))) / seed.size,
        float(np.sum(reference.radiative_loss_rate_cells)) / seed.size,
        1.0,
    )
    residual_scale = np.maximum(
        np.abs(reference_flux_difference)
        + np.abs(reference.vertical_work_rate_cells)
        + reference.radiative_loss_rate_cells
        + np.abs(reference.stream_energy_rate_cells)
        + np.abs(reference.external_power_rate_cells),
        1.0e-6 * global_floor,
    )

    def residual(log_temperature):
        profile = signed_total_energy_profile(
            grid,
            transport,
            np.exp(log_temperature),
            M_g,
            closure=closure,
            external_power_rate_cells=external_power_rate_cells,
            prescribed_inner_flux=prescribed_inner_flux,
        )
        return profile.net_energy_rate_cells / residual_scale

    result = least_squares(
        residual,
        np.log(seed),
        bounds=(np.log(lower), np.log(upper)),
        jac_sparsity=_jacobian_sparsity(seed.size),
        x_scale="jac",
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-10,
        max_nfev=int(max_nfev),
    )
    temperature = np.exp(result.x)
    profile = signed_total_energy_profile(
        grid,
        transport,
        temperature,
        M_g,
        closure=closure,
        external_power_rate_cells=external_power_rate_cells,
        prescribed_inner_flux=prescribed_inner_flux,
    )
    maximum = float(np.max(np.abs(residual(result.x))))
    return SignedTotalEnergySteadyResult(
        temperature=temperature,
        profile=profile,
        accepted=bool(maximum <= tolerance),
        nfev=int(result.nfev),
        maximum_normalized_residual=maximum,
        message=str(result.message),
    )


def solve_signed_total_energy_thermoviscous_steady(
    grid: RadialGrid,
    M_g: float,
    *,
    alpha: float,
    boundary: SignedFluxBoundary,
    stream_state: StreamInjectionState,
    closure: SignedThermalClosure,
    temperature_seed,
    external_angular_rate_cells=None,
    external_power_rate_cells=None,
    prescribed_inner_flux: ConservedInterfaceFlux | None = None,
    initial_H_over_R: float = 0.1,
    damping: float = 0.3,
    tolerance: float = 1.0e-3,
    max_iterations: int = 50,
    energy_tolerance: float = 1.0e-6,
    energy_max_nfev: int = 500,
) -> SignedTotalEnergyThermoviscousResult:
    """Iterate alpha viscosity against the total-energy steady root."""

    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be positive and finite")
    if not np.isfinite(damping) or not 0.0 < damping <= 1.0:
        raise ValueError("damping must lie in (0,1]")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")
    if external_angular_rate_cells is not None and external_power_rate_cells is None:
        external_angular = _grid_array(
            "external_angular_rate_cells", external_angular_rate_cells, grid
        )
        if np.any(external_angular != 0.0):
            raise ValueError("external angular torque requires named external power")
    potential = PaczynskiWiitaPotential(float(M_g))
    omega = np.asarray(potential.omega_k(grid.centers), dtype=float)
    viscosity = np.asarray(
        alpha * initial_H_over_R**2 * grid.centers**2 * omega, dtype=float
    )
    temperature = _positive_temperature(temperature_seed, grid)
    history = []
    transport = None
    energy = None
    maximum_change = float("inf")
    converged = False

    def transport_for(trial_viscosity):
        return solve_signed_flux_steady(
            grid,
            trial_viscosity,
            M_g,
            boundary=boundary,
            stream_state=stream_state,
            external_angular_rate_cells=external_angular_rate_cells,
            prescribed_inner_flux=prescribed_inner_flux,
        )

    for iteration in range(1, int(max_iterations) + 1):
        transport = transport_for(viscosity)
        energy = solve_signed_total_energy_steady(
            grid,
            transport,
            temperature,
            M_g,
            closure=closure,
            external_power_rate_cells=external_power_rate_cells,
            prescribed_inner_flux=prescribed_inner_flux,
            tolerance=energy_tolerance,
            max_nfev=energy_max_nfev,
        )
        temperature = energy.temperature
        target_viscosity = np.asarray(alpha * energy.profile.H**2 * omega, dtype=float)
        log_ratio = np.log(target_viscosity / viscosity)
        maximum_change = float(np.max(np.abs(log_ratio)))
        history.append(
            [
                float(iteration),
                maximum_change,
                energy.maximum_normalized_residual,
                float(np.max(energy.profile.H / grid.centers)),
            ]
        )
        if energy.accepted and maximum_change <= tolerance:
            viscosity = target_viscosity
            transport = transport_for(viscosity)
            energy = solve_signed_total_energy_steady(
                grid,
                transport,
                temperature,
                M_g,
                closure=closure,
                external_power_rate_cells=external_power_rate_cells,
                prescribed_inner_flux=prescribed_inner_flux,
                tolerance=energy_tolerance,
                max_nfev=energy_max_nfev,
            )
            temperature = energy.temperature
            final_target = np.asarray(
                alpha * energy.profile.H**2 * omega, dtype=float
            )
            final_log_ratio = np.log(final_target / viscosity)
            maximum_change = float(np.max(np.abs(final_log_ratio)))
            history[-1] = [
                float(iteration),
                maximum_change,
                energy.maximum_normalized_residual,
                float(np.max(energy.profile.H / grid.centers)),
            ]
            if energy.accepted and maximum_change <= tolerance:
                converged = True
                break
            viscosity = np.exp(
                np.log(viscosity) + float(damping) * final_log_ratio
            )
            continue
        viscosity = np.exp(np.log(viscosity) + float(damping) * log_ratio)

    assert transport is not None and energy is not None
    if not converged:
        transport = transport_for(viscosity)
        energy = solve_signed_total_energy_steady(
            grid,
            transport,
            temperature,
            M_g,
            closure=closure,
            external_power_rate_cells=external_power_rate_cells,
            prescribed_inner_flux=prescribed_inner_flux,
            tolerance=energy_tolerance,
            max_nfev=energy_max_nfev,
        )
        final_target = np.asarray(alpha * energy.profile.H**2 * omega, dtype=float)
        maximum_change = float(
            np.max(np.abs(np.log(final_target / viscosity)))
        )
    return SignedTotalEnergyThermoviscousResult(
        transport=transport,
        energy=energy,
        viscosity=np.asarray(viscosity, dtype=float),
        converged=converged,
        iterations=len(history),
        maximum_log_viscosity_change=maximum_change,
        history=np.asarray(history, dtype=float),
    )
