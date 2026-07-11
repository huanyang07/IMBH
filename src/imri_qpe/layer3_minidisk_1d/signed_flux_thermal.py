"""Thermal-energy ledger for the independent-Sigma signed-flux disk."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, least_squares

from .grid import RadialGrid
from .signed_flux_disk import (
    SignedFluxBoundary,
    SignedFluxTransport,
    solve_signed_flux_steady,
)
from .transonic_potential import PaczynskiWiitaPotential
from .transonic_thermo import radiative_cooling, vertical_state


@dataclass(frozen=True)
class SignedThermalClosure:
    """One-zone thermal and stream-energy parameters."""

    mu_mol: float = 0.62
    kappa: float = 0.34
    gamma_gas: float = 5.0 / 3.0
    stream_specific_angular_momentum: float | None = None
    stream_specific_total_energy: float | None = None
    temperature_bounds: tuple[float, float] = (1.0e3, 1.0e10)

    def __post_init__(self) -> None:
        if self.mu_mol <= 0.0 or self.kappa <= 0.0 or self.gamma_gas <= 1.0:
            raise ValueError("thermal material parameters must be positive")
        if self.temperature_bounds[0] <= 0.0 or self.temperature_bounds[1] <= self.temperature_bounds[0]:
            raise ValueError("temperature_bounds must be positive and increasing")
        for name, value in {
            "stream_specific_angular_momentum": self.stream_specific_angular_momentum,
            "stream_specific_total_energy": self.stream_specific_total_energy,
        }.items():
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} must be finite when supplied")


@dataclass(frozen=True)
class SignedThermalProfile:
    """Thermal state and conservative energy-rate decomposition."""

    temperature: np.ndarray
    specific_internal_energy: np.ndarray
    thermal_energy_cells: np.ndarray
    thermal_energy_flux_faces: np.ndarray
    advective_rate_cells: np.ndarray
    viscous_heating_rate_cells: np.ndarray
    radiative_cooling_rate_cells: np.ndarray
    stream_heating_rate_cells: np.ndarray
    net_energy_rate_cells: np.ndarray
    H: np.ndarray
    rho: np.ndarray
    tau: np.ndarray
    impact_specific_energy: np.ndarray
    energy_budget_rate: float
    energy_budget_defect: float


@dataclass(frozen=True)
class SignedThermalStepResult:
    temperature: np.ndarray
    profile: SignedThermalProfile
    dt: float
    maximum_energy_residual: float


@dataclass(frozen=True)
class SignedThermalSteadyResult:
    temperature: np.ndarray
    profile: SignedThermalProfile
    accepted: bool
    nfev: int
    maximum_normalized_residual: float
    message: str


@dataclass(frozen=True)
class SignedThermoviscousSteadyResult:
    """Coupled fixed point of mass transport, vertical state, and alpha viscosity."""

    transport: SignedFluxTransport
    thermal: SignedThermalSteadyResult
    viscosity: np.ndarray
    converged: bool
    iterations: int
    maximum_log_viscosity_change: float
    history: np.ndarray


def _temperature_array(temperature, grid: RadialGrid) -> np.ndarray:
    array = np.asarray(temperature, dtype=float)
    if array.ndim == 0:
        array = np.full_like(grid.centers, float(array), dtype=float)
    if array.shape != grid.centers.shape or np.any(~np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError("temperature must be positive, finite, and match the grid")
    return array


def _vertical(grid, transport: SignedFluxTransport, temperature, M_g: float, closure):
    potential = PaczynskiWiitaPotential(float(M_g))
    return vertical_state(
        transport.surface_density,
        temperature,
        grid.centers,
        potential,
        mu_mol=closure.mu_mol,
        kappa=closure.kappa,
        gamma_gas=closure.gamma_gas,
    )


def _upwind_energy_flux(mdot_faces: np.ndarray, specific_energy: np.ndarray) -> np.ndarray:
    n = specific_energy.size
    flux = np.zeros(n + 1, dtype=float)
    if mdot_faces[0] < 0.0:
        raise ValueError("inner-boundary outward inflow needs an injected energy state")
    if mdot_faces[-1] > 0.0:
        raise ValueError("outer-boundary inward flow needs a reservoir energy state")
    flux[0] = mdot_faces[0] * specific_energy[0]
    flux[-1] = mdot_faces[-1] * specific_energy[-1]
    for face in range(1, n):
        donor = face if mdot_faces[face] >= 0.0 else face - 1
        flux[face] = mdot_faces[face] * specific_energy[donor]
    return flux


def signed_thermal_profile(
    grid: RadialGrid,
    transport: SignedFluxTransport,
    temperature,
    M_g: float,
    *,
    closure: SignedThermalClosure | None = None,
) -> SignedThermalProfile:
    """Evaluate conservative internal-energy transport and local source terms."""

    closure = SignedThermalClosure() if closure is None else closure
    temperature = _temperature_array(temperature, grid)
    state = _vertical(grid, transport, temperature, M_g, closure)
    specific_e = np.asarray(state.e, dtype=float)
    thermal_cells = transport.surface_density * specific_e * grid.area
    energy_flux = _upwind_energy_flux(transport.mdot_faces, specific_e)
    advective = energy_flux[1:] - energy_flux[:-1]
    potential = PaczynskiWiitaPotential(float(M_g))
    shear_log = np.asarray(
        state.Omega_K * potential.dln_omega_k_dlnR(grid.centers), dtype=float
    )
    q_visc = transport.viscosity * transport.surface_density * shear_log**2
    viscous = q_visc * grid.area
    radiative = np.asarray(radiative_cooling(state, kappa=closure.kappa), dtype=float) * grid.area

    if closure.stream_specific_total_energy is None:
        impact_specific = np.zeros_like(grid.centers)
        stream_heating = np.zeros_like(grid.centers)
    else:
        orbital = np.asarray(
            potential.phi(grid.centers)
            + 0.5 * (potential.l_k(grid.centers) / grid.centers) ** 2,
            dtype=float,
        )
        impact_specific = float(closure.stream_specific_total_energy) - orbital
        stream_heating = transport.source_mass_rate_cells * impact_specific

    net = advective + viscous + stream_heating - radiative
    boundary_and_sources = float(
        energy_flux[-1]
        - energy_flux[0]
        + np.sum(viscous + stream_heating - radiative)
    )
    return SignedThermalProfile(
        temperature=temperature,
        specific_internal_energy=specific_e,
        thermal_energy_cells=thermal_cells,
        thermal_energy_flux_faces=energy_flux,
        advective_rate_cells=advective,
        viscous_heating_rate_cells=viscous,
        radiative_cooling_rate_cells=radiative,
        stream_heating_rate_cells=stream_heating,
        net_energy_rate_cells=net,
        H=np.asarray(state.H, dtype=float),
        rho=np.asarray(state.rho, dtype=float),
        tau=np.asarray(state.tau, dtype=float),
        impact_specific_energy=np.asarray(impact_specific, dtype=float),
        energy_budget_rate=boundary_and_sources,
        energy_budget_defect=float(np.sum(net) - boundary_and_sources),
    )


def advance_signed_thermal_implicit(
    grid: RadialGrid,
    transport: SignedFluxTransport,
    temperature,
    M_g: float,
    dt: float,
    *,
    closure: SignedThermalClosure | None = None,
) -> SignedThermalStepResult:
    """Operator-split thermal step with local cooling treated implicitly."""

    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError("dt must be finite and non-negative")
    closure = SignedThermalClosure() if closure is None else closure
    old = signed_thermal_profile(
        grid, transport, temperature, M_g, closure=closure
    )
    base = old.thermal_energy_cells + float(dt) * (
        old.advective_rate_cells
        + old.viscous_heating_rate_cells
        + old.stream_heating_rate_cells
    )
    if np.any(base <= 0.0):
        raise ValueError("thermal step has non-positive energy before cooling")
    lower, upper = closure.temperature_bounds
    updated = np.empty_like(old.temperature)
    residual = np.empty_like(old.temperature)
    for idx in range(updated.size):
        def equation(log_temperature: float) -> float:
            trial_temperature = float(np.exp(log_temperature))
            state = vertical_state(
                transport.surface_density[idx],
                trial_temperature,
                grid.centers[idx],
                PaczynskiWiitaPotential(float(M_g)),
                mu_mol=closure.mu_mol,
                kappa=closure.kappa,
                gamma_gas=closure.gamma_gas,
            )
            energy = transport.surface_density[idx] * float(state.e) * grid.area[idx]
            cooling = float(radiative_cooling(state, kappa=closure.kappa)) * grid.area[idx]
            return energy + float(dt) * cooling - base[idx]

        low_log = float(np.log(lower))
        high_log = float(np.log(upper))
        if equation(low_log) > 0.0 or equation(high_log) < 0.0:
            raise ValueError("implicit thermal root lies outside temperature bounds")
        updated[idx] = float(np.exp(brentq(equation, low_log, high_log, xtol=1.0e-12)))
        residual[idx] = equation(float(np.log(updated[idx])))
    profile = signed_thermal_profile(
        grid, transport, updated, M_g, closure=closure
    )
    scale = np.maximum(np.abs(base), 1.0)
    return SignedThermalStepResult(
        temperature=updated,
        profile=profile,
        dt=float(dt),
        maximum_energy_residual=float(np.max(np.abs(residual) / scale)),
    )


def solve_signed_thermal_steady(
    grid: RadialGrid,
    transport: SignedFluxTransport,
    temperature_seed,
    M_g: float,
    *,
    closure: SignedThermalClosure | None = None,
    tolerance: float = 1.0e-6,
    max_nfev: int = 500,
) -> SignedThermalSteadyResult:
    """Solve the fixed-Sigma steady conservative thermal ledger."""

    closure = SignedThermalClosure() if closure is None else closure
    seed = _temperature_array(temperature_seed, grid)
    lower, upper = closure.temperature_bounds

    def residual(log_temperature):
        profile = signed_thermal_profile(
            grid,
            transport,
            np.exp(log_temperature),
            M_g,
            closure=closure,
        )
        scale = np.maximum(
            np.abs(profile.viscous_heating_rate_cells)
            + np.abs(profile.stream_heating_rate_cells)
            + np.abs(profile.radiative_cooling_rate_cells),
            1.0,
        )
        return profile.net_energy_rate_cells / scale

    result = least_squares(
        residual,
        np.log(seed),
        bounds=(np.log(lower), np.log(upper)),
        jac_sparsity=_thermal_jacobian_sparsity(seed.size),
        x_scale="jac",
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-10,
        max_nfev=int(max_nfev),
    )
    temperature = np.exp(result.x)
    profile = signed_thermal_profile(
        grid, transport, temperature, M_g, closure=closure
    )
    maximum = float(np.max(np.abs(residual(result.x))))
    return SignedThermalSteadyResult(
        temperature=temperature,
        profile=profile,
        accepted=bool(maximum <= tolerance),
        nfev=int(result.nfev),
        maximum_normalized_residual=maximum,
        message=str(result.message),
    )


def _thermal_jacobian_sparsity(n: int):
    from scipy.sparse import lil_matrix

    pattern = lil_matrix((n, n), dtype=int)
    for row in range(n):
        pattern[row, row] = 1
        if row > 0:
            pattern[row, row - 1] = 1
        if row + 1 < n:
            pattern[row, row + 1] = 1
    return pattern.tocsr()


def solve_signed_thermoviscous_steady(
    grid: RadialGrid,
    M_g: float,
    *,
    alpha: float,
    boundary: SignedFluxBoundary,
    source_mass_rate_cells,
    source_specific_angular_momentum,
    thermal_closure: SignedThermalClosure,
    temperature_seed,
    initial_H_over_R: float = 0.1,
    damping: float = 0.3,
    tolerance: float = 1.0e-3,
    max_iterations: int = 50,
    thermal_tolerance: float = 1.0e-6,
    thermal_max_nfev: int = 500,
) -> SignedThermoviscousSteadyResult:
    """Iterate ``nu=alpha H^2 Omega_K`` with mass and thermal steady solves."""

    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be positive and finite")
    if not np.isfinite(initial_H_over_R) or initial_H_over_R <= 0.0:
        raise ValueError("initial_H_over_R must be positive and finite")
    if not np.isfinite(damping) or not 0.0 < damping <= 1.0:
        raise ValueError("damping must lie in (0,1]")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")
    potential = PaczynskiWiitaPotential(float(M_g))
    omega = np.asarray(potential.omega_k(grid.centers), dtype=float)
    viscosity = np.asarray(
        alpha * initial_H_over_R**2 * grid.centers**2 * omega,
        dtype=float,
    )
    temperature = _temperature_array(temperature_seed, grid)
    history = []
    transport = None
    thermal = None
    maximum_change = float("inf")
    for iteration in range(1, int(max_iterations) + 1):
        transport = solve_signed_flux_steady(
            grid,
            viscosity,
            M_g,
            boundary=boundary,
            source_mass_rate_cells=source_mass_rate_cells,
            source_specific_angular_momentum=source_specific_angular_momentum,
        )
        thermal = solve_signed_thermal_steady(
            grid,
            transport,
            temperature,
            M_g,
            closure=thermal_closure,
            tolerance=thermal_tolerance,
            max_nfev=thermal_max_nfev,
        )
        temperature = thermal.temperature
        target_viscosity = np.asarray(alpha * thermal.profile.H**2 * omega, dtype=float)
        log_ratio = np.log(target_viscosity / viscosity)
        maximum_change = float(np.max(np.abs(log_ratio)))
        history.append(
            [
                float(iteration),
                maximum_change,
                thermal.maximum_normalized_residual,
                float(np.max(thermal.profile.H / grid.centers)),
            ]
        )
        if thermal.accepted and maximum_change <= tolerance:
            viscosity = target_viscosity
            break
        viscosity = np.exp(np.log(viscosity) + float(damping) * log_ratio)

    assert transport is not None and thermal is not None
    converged = bool(thermal.accepted and maximum_change <= tolerance)
    if converged:
        transport = solve_signed_flux_steady(
            grid,
            viscosity,
            M_g,
            boundary=boundary,
            source_mass_rate_cells=source_mass_rate_cells,
            source_specific_angular_momentum=source_specific_angular_momentum,
        )
        thermal = solve_signed_thermal_steady(
            grid,
            transport,
            temperature,
            M_g,
            closure=thermal_closure,
            tolerance=thermal_tolerance,
            max_nfev=thermal_max_nfev,
        )
    return SignedThermoviscousSteadyResult(
        transport=transport,
        thermal=thermal,
        viscosity=np.asarray(viscosity, dtype=float),
        converged=converged,
        iterations=len(history),
        maximum_log_viscosity_change=maximum_change,
        history=np.asarray(history, dtype=float),
    )
