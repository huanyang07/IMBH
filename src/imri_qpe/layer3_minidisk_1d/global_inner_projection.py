"""Local steady projection for the causally outgoing global plunge cells."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from imri_qpe.constants import C, DEFAULT_KAPPA_ES, DEFAULT_MU_MOL

from .global_signed_evolution import (
    GlobalCellSources,
    GlobalConservativeState,
    GlobalInviscidProfile,
    evaluate_global_rusanov_profile,
    global_conservative_rhs,
    global_effective_sound_speed,
    recover_global_primitives,
    state_from_thermodynamic_primitives,
)
from .grid import RadialGrid
from .hill_roche_nozzle import OverflowBoundaryProvider


_COMPONENTS = ("mass", "radial_momentum", "angular_momentum", "total_energy")


@dataclass(frozen=True)
class GlobalInnerProjectionAudit:
    """Rank, residual, interface, and primitive-change gates for a projection."""

    accepted: bool
    projected_cells: int
    projected_outer_edge: float
    maximum_scaled_residual: float
    jacobian_rank: int
    jacobian_size: int
    jacobian_condition_estimate: float
    solver_nfev: int
    solver_status: int
    solver_message: str
    interface_flux_relative_changes: dict[str, float]
    first_unprojected_residual_before: dict[str, float]
    first_unprojected_residual_after: dict[str, float]
    maximum_log_surface_density_change: float
    maximum_log_temperature_change: float
    maximum_log_omega_change: float
    maximum_relative_radial_velocity_change: float
    projected_cells_remain_supersonic: bool
    first_unprojected_cell_remains_subsonic: bool


@dataclass(frozen=True)
class GlobalInnerProjectionResult:
    """Projected state and its production-operator profile."""

    state: GlobalConservativeState
    profile: GlobalInviscidProfile
    audit: GlobalInnerProjectionAudit


def global_supersonic_prefix_cell_count(radial_mach_number) -> int:
    """Count the contiguous inner cells with inward Mach number below -1."""

    mach = np.asarray(radial_mach_number, dtype=float)
    if mach.ndim != 1 or mach.size == 0 or np.any(~np.isfinite(mach)):
        raise ValueError("radial Mach number must be a finite nonempty vector")
    count = 0
    for value in mach:
        if value >= -1.0:
            break
        count += 1
    if count == 0:
        raise ValueError("global state has no causally outgoing inner cell")
    if count == mach.size:
        raise ValueError("global state has no subsonic cell outside the plunge")
    return count


def _exchange_scales(
    profile: GlobalInviscidProfile, count: int
) -> dict[str, np.ndarray]:
    scales = {}
    for name in _COMPONENTS:
        flux = np.asarray(getattr(profile.face_fluxes, name), dtype=float)
        source = np.asarray(getattr(profile.cell_sources, name), dtype=float)
        floor = max(
            float(np.max(np.abs(flux[: count + 1]))),
            float(np.max(np.abs(source[:count]))),
            1.0,
        ) * 1.0e-12
        scales[name] = np.maximum.reduce(
            (
                np.abs(flux[:count]),
                np.abs(flux[1 : count + 1]),
                np.abs(source[:count]),
                np.full(count, floor, dtype=float),
            )
        )
    return scales


def _normalized_cell_residual(
    profile: GlobalInviscidProfile, index: int
) -> dict[str, float]:
    rhs = global_conservative_rhs(profile.face_fluxes, profile.cell_sources)
    result = {}
    for name in _COMPONENTS:
        flux = np.asarray(getattr(profile.face_fluxes, name), dtype=float)
        source = np.asarray(getattr(profile.cell_sources, name), dtype=float)
        scale = max(
            abs(float(flux[index])),
            abs(float(flux[index + 1])),
            abs(float(source[index])),
            1.0,
        )
        result[name] = abs(float(getattr(rhs, name)[index])) / scale
    return result


def solve_global_inner_steady_projection(
    grid: RadialGrid,
    state: GlobalConservativeState,
    M_g: float,
    *,
    alpha: float,
    reference_state: GlobalConservativeState,
    external_sources: GlobalCellSources,
    outer_overflow_provider: OverflowBoundaryProvider,
    specific_mechanical_energy_correction,
    boundary_mode: str = "roche_outer",
    stress_boundary_mode: str = "outer_zero_torque",
    include_radiative_cooling: bool = True,
    include_vertical_column_work: bool = True,
    temperature_bounds: tuple[float, float] = (1.0e3, 1.0e12),
    mu_mol: float = DEFAULT_MU_MOL,
    kappa: float = DEFAULT_KAPPA_ES,
    gamma_gas: float = 5.0 / 3.0,
    maximum_nfev: int = 100,
    residual_tolerance: float = 1.0e-8,
    interface_flux_tolerance: float = 2.0e-2,
    primitive_log_change_tolerance: float = 1.0e-1,
    radial_velocity_change_tolerance: float = 1.5e-1,
) -> GlobalInnerProjectionResult:
    """Project only the causally outgoing plunge onto the steady FV operator."""

    state = state.validated()
    reference_state = reference_state.validated()
    if (
        state.n_cells != grid.centers.size
        or reference_state.n_cells != state.n_cells
    ):
        raise ValueError("projection states must match the supplied grid")
    external_sources = external_sources.validated_for(state.n_cells)
    old = recover_global_primitives(
        grid,
        state,
        M_g,
        temperature_bounds=temperature_bounds,
        mu_mol=mu_mol,
        kappa=kappa,
        gamma_gas=gamma_gas,
        specific_mechanical_energy_correction=(
            specific_mechanical_energy_correction
        ),
    )
    sound = global_effective_sound_speed(old, gamma_gas=gamma_gas)
    projected_cells = global_supersonic_prefix_cell_count(
        old.radial_velocity / sound
    )
    for name in _COMPONENTS:
        source = np.asarray(getattr(external_sources, name))
        if np.any(source[:projected_cells] != 0.0):
            raise ValueError("projected plunge cells must contain no external source")

    def evaluate(trial: GlobalConservativeState) -> GlobalInviscidProfile:
        return evaluate_global_rusanov_profile(
            grid,
            trial,
            M_g,
            reference_state=reference_state,
            boundary_mode=boundary_mode,
            temperature_bounds=temperature_bounds,
            mu_mol=mu_mol,
            kappa=kappa,
            gamma_gas=gamma_gas,
            alpha=alpha,
            stress_boundary_mode=stress_boundary_mode,
            include_radiative_cooling=include_radiative_cooling,
            include_vertical_column_work=include_vertical_column_work,
            external_sources=external_sources,
            outer_overflow_provider=outer_overflow_provider,
            specific_mechanical_energy_correction=(
                specific_mechanical_energy_correction
            ),
        )

    initial_profile = evaluate(state)
    scales = _exchange_scales(initial_profile, projected_cells)
    initial = np.concatenate(
        (
            np.log(old.surface_density[:projected_cells]),
            old.radial_velocity[:projected_cells] / C,
            np.log(old.omega[:projected_cells]),
            np.log(old.temperature[:projected_cells]),
        )
    )

    def reconstruct(values: np.ndarray) -> GlobalConservativeState:
        count = projected_cells
        sigma = np.array(old.surface_density, copy=True)
        velocity = np.array(old.radial_velocity, copy=True)
        omega = np.array(old.omega, copy=True)
        temperature = np.array(old.temperature, copy=True)
        sigma[:count] = np.exp(values[:count])
        velocity[:count] = C * values[count : 2 * count]
        omega[:count] = np.exp(values[2 * count : 3 * count])
        temperature[:count] = np.exp(values[3 * count :])
        return state_from_thermodynamic_primitives(
            grid,
            sigma,
            velocity,
            omega,
            temperature,
            M_g,
            mu_mol=mu_mol,
            kappa=kappa,
            gamma_gas=gamma_gas,
            specific_mechanical_energy_correction=(
                specific_mechanical_energy_correction
            ),
        )

    def residual(values: np.ndarray) -> np.ndarray:
        profile = evaluate(reconstruct(values))
        rhs = global_conservative_rhs(profile.face_fluxes, profile.cell_sources)
        return np.concatenate(
            tuple(
                np.asarray(getattr(rhs, name))[:projected_cells]
                / scales[name]
                for name in _COMPONENTS
            )
        )

    lower_temperature, upper_temperature = map(float, temperature_bounds)
    lower = np.concatenate(
        (
            np.full(3 * projected_cells, -np.inf),
            np.full(projected_cells, np.log(lower_temperature)),
        )
    )
    upper = np.concatenate(
        (
            np.full(3 * projected_cells, np.inf),
            np.full(projected_cells, np.log(upper_temperature)),
        )
    )
    solve = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        x_scale="jac",
        ftol=1.0e-13,
        xtol=1.0e-13,
        gtol=1.0e-13,
        max_nfev=int(maximum_nfev),
        diff_step=1.0e-6,
    )
    projected = reconstruct(solve.x)
    final_profile = evaluate(projected)
    final = recover_global_primitives(
        grid,
        projected,
        M_g,
        temperature_bounds=temperature_bounds,
        mu_mol=mu_mol,
        kappa=kappa,
        gamma_gas=gamma_gas,
        specific_mechanical_energy_correction=(
            specific_mechanical_energy_correction
        ),
    )
    maximum_residual = float(np.max(np.abs(residual(solve.x))))
    singular_values = np.linalg.svd(np.asarray(solve.jac), compute_uv=False)
    rank_tolerance = (
        max(np.asarray(solve.jac).shape)
        * np.finfo(float).eps
        * singular_values[0]
    )
    rank = int(np.count_nonzero(singular_values > rank_tolerance))
    condition = float(
        singular_values[0] / singular_values[-1]
        if singular_values[-1] > 0.0
        else np.inf
    )
    interface_changes = {}
    for name in _COMPONENTS:
        initial_flux = float(
            getattr(initial_profile.face_fluxes, name)[projected_cells]
        )
        final_flux = float(
            getattr(final_profile.face_fluxes, name)[projected_cells]
        )
        interface_changes[name] = abs(final_flux - initial_flux) / max(
            abs(initial_flux), 1.0
        )
    outside_before = _normalized_cell_residual(
        initial_profile, projected_cells
    )
    outside_after = _normalized_cell_residual(final_profile, projected_cells)
    count = projected_cells
    max_log_sigma = float(
        np.max(
            np.abs(
                np.log(
                    final.surface_density[:count]
                    / old.surface_density[:count]
                )
            )
        )
    )
    max_log_temperature = float(
        np.max(
            np.abs(
                np.log(final.temperature[:count] / old.temperature[:count])
            )
        )
    )
    max_log_omega = float(
        np.max(np.abs(np.log(final.omega[:count] / old.omega[:count])))
    )
    max_velocity_change = float(
        np.max(
            np.abs(
                final.radial_velocity[:count]
                / old.radial_velocity[:count]
                - 1.0
            )
        )
    )
    final_mach = final.radial_velocity / global_effective_sound_speed(
        final, gamma_gas=gamma_gas
    )
    projected_supersonic = bool(np.all(final_mach[:count] < -1.0))
    outside_subsonic = bool(final_mach[count] >= -1.0)
    interface_pass = max(interface_changes.values()) <= interface_flux_tolerance
    outside_pass = max(outside_after.values()) <= max(outside_before.values())
    primitive_pass = (
        max(max_log_sigma, max_log_temperature, max_log_omega)
        <= primitive_log_change_tolerance
        and max_velocity_change <= radial_velocity_change_tolerance
    )
    size = 4 * projected_cells
    accepted = bool(
        solve.success
        and maximum_residual <= residual_tolerance
        and rank == size
        and interface_pass
        and outside_pass
        and primitive_pass
        and projected_supersonic
        and outside_subsonic
    )
    audit = GlobalInnerProjectionAudit(
        accepted=accepted,
        projected_cells=projected_cells,
        projected_outer_edge=float(grid.edges[projected_cells]),
        maximum_scaled_residual=maximum_residual,
        jacobian_rank=rank,
        jacobian_size=size,
        jacobian_condition_estimate=condition,
        solver_nfev=int(solve.nfev),
        solver_status=int(solve.status),
        solver_message=str(solve.message),
        interface_flux_relative_changes=interface_changes,
        first_unprojected_residual_before=outside_before,
        first_unprojected_residual_after=outside_after,
        maximum_log_surface_density_change=max_log_sigma,
        maximum_log_temperature_change=max_log_temperature,
        maximum_log_omega_change=max_log_omega,
        maximum_relative_radial_velocity_change=max_velocity_change,
        projected_cells_remain_supersonic=projected_supersonic,
        first_unprojected_cell_remains_subsonic=outside_subsonic,
    )
    return GlobalInnerProjectionResult(
        state=projected,
        profile=final_profile,
        audit=audit,
    )
