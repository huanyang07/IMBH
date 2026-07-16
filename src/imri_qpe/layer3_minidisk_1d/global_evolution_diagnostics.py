"""Mesh-comparable diagnostics for global conservative evolution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .global_signed_evolution import (
    GlobalFaceFluxes,
    GlobalPrimitiveState,
    GlobalRocheBoundaryAudit,
    global_effective_sound_speed,
)
from .grid import RadialGrid
from .hill_roche_nozzle import (
    OverflowBoundaryProvider,
    hill_roche_disk_matched_potential,
)
from .transonic_potential import PaczynskiWiitaPotential


@dataclass(frozen=True)
class GlobalFixedRadiusDiagnostic:
    """Primitive, characteristic, and conservative flux data at one radius."""

    radius: float
    surface_density: float
    radial_velocity: float
    temperature: float
    omega_over_omega_k: float
    H_over_R: float
    radial_mach_number: float
    characteristic_speeds: tuple[float, float, float, float]
    mass_flux: float
    angular_momentum_flux: float
    total_energy_flux: float


@dataclass(frozen=True)
class GlobalSonicResolutionDiagnostic:
    """Location and radial resolution of the innermost sonic transition."""

    sonic_radius: float | None
    cells_between_inner_face_and_sonic_radius: int | None
    minimum_velocity_gradient_length_over_cell_width: float
    minimum_over_cell_width_index: int
    minimum_over_cell_width_radius: float
    minimum_velocity_gradient_length_over_H: float
    minimum_over_H_index: int
    minimum_over_H_radius: float
    gradient_audit_outer_radius: float


@dataclass(frozen=True)
class GlobalRocheClosureDiagnostic:
    """Dimensionless energetic margin and active-set residual at the edge."""

    barrier_specific_energy: float
    available_specific_energy: float
    normalized_available_specific_energy: float
    channel_state: str
    active_set_residual: float
    nozzle_residual: float | None


def _interpolate(radius: float, nodes, values, *, positive: bool) -> float:
    nodes = np.log(np.asarray(nodes, dtype=float))
    values = np.asarray(values, dtype=float)
    work = np.log(values) if positive else values
    query = float(np.log(radius))
    result = float(np.interp(query, nodes, work))
    if nodes.size >= 2 and query < nodes[0]:
        result = float(
            work[0]
            + (work[1] - work[0]) / (nodes[1] - nodes[0])
            * (query - nodes[0])
        )
    elif nodes.size >= 2 and query > nodes[-1]:
        result = float(
            work[-1]
            + (work[-1] - work[-2]) / (nodes[-1] - nodes[-2])
            * (query - nodes[-1])
        )
    return float(np.exp(result)) if positive else result


def _interpolate_positive(radius: float, nodes, values) -> float:
    return _interpolate(radius, nodes, values, positive=True)


def _interpolate_signed(radius: float, nodes, values) -> float:
    return _interpolate(radius, nodes, values, positive=False)


def global_fixed_radius_diagnostics(
    grid: RadialGrid,
    primitives: GlobalPrimitiveState,
    face_fluxes: GlobalFaceFluxes,
    M_g: float,
    radii,
) -> tuple[GlobalFixedRadiusDiagnostic, ...]:
    """Evaluate mesh-comparable state and flux diagnostics at fixed radii."""

    face_fluxes = face_fluxes.validated_for(grid.centers.size)
    radii = np.asarray(radii, dtype=float)
    if radii.ndim != 1 or np.any(~np.isfinite(radii)):
        raise ValueError("diagnostic radii must be a finite vector")
    if np.any(radii < grid.edges[0]) or np.any(radii > grid.edges[-1]):
        raise ValueError("diagnostic radii must lie inside the radial domain")
    sound_speed = global_effective_sound_speed(primitives)
    potential = PaczynskiWiitaPotential(float(M_g))
    results = []
    for radius in radii:
        radius = float(radius)
        velocity = _interpolate_signed(
            radius, grid.centers, primitives.radial_velocity
        )
        sound = _interpolate_positive(radius, grid.centers, sound_speed)
        omega = _interpolate_positive(radius, grid.centers, primitives.omega)
        H = _interpolate_positive(radius, grid.centers, primitives.vertical.H)
        characteristic_speeds = (
            velocity - sound,
            velocity,
            velocity,
            velocity + sound,
        )
        results.append(
            GlobalFixedRadiusDiagnostic(
                radius=radius,
                surface_density=_interpolate_positive(
                    radius, grid.centers, primitives.surface_density
                ),
                radial_velocity=velocity,
                temperature=_interpolate_positive(
                    radius, grid.centers, primitives.temperature
                ),
                omega_over_omega_k=float(
                    omega / float(potential.omega_k(radius))
                ),
                H_over_R=H / radius,
                radial_mach_number=velocity / sound,
                characteristic_speeds=characteristic_speeds,
                mass_flux=_interpolate_signed(
                    radius, grid.edges, face_fluxes.mass
                ),
                angular_momentum_flux=_interpolate_signed(
                    radius, grid.edges, face_fluxes.angular_momentum
                ),
                total_energy_flux=_interpolate_signed(
                    radius, grid.edges, face_fluxes.total_energy
                ),
            )
        )
    return tuple(results)


def global_sonic_resolution_diagnostic(
    grid: RadialGrid,
    primitives: GlobalPrimitiveState,
) -> GlobalSonicResolutionDiagnostic:
    """Locate the innermost ``|Mach|=1`` crossing and velocity-gradient scale."""

    velocity = np.asarray(primitives.radial_velocity, dtype=float)
    sound_speed = global_effective_sound_speed(primitives)
    mach_offset = np.abs(velocity / sound_speed) - 1.0
    crossing_radius = None
    crossing_interval = None
    for index in range(grid.centers.size - 1):
        left = float(mach_offset[index])
        right = float(mach_offset[index + 1])
        if left == 0.0:
            crossing_radius = float(grid.centers[index])
            crossing_interval = index
            break
        if left * right <= 0.0:
            fraction = -left / (right - left)
            log_radius = (
                np.log(grid.centers[index])
                + fraction
                * (
                    np.log(grid.centers[index + 1])
                    - np.log(grid.centers[index])
                )
            )
            crossing_radius = float(np.exp(log_radius))
            crossing_interval = index
            break
    cell_count = (
        None
        if crossing_radius is None
        else int(np.count_nonzero(grid.centers < crossing_radius))
    )
    speed_floor = max(float(np.max(np.abs(velocity))) * 1.0e-14, 1.0e-300)
    log_speed = np.log(np.maximum(np.abs(velocity), speed_floor))
    edge_order = 2 if grid.centers.size >= 3 else 1
    slope = np.gradient(log_speed, grid.centers, edge_order=edge_order)
    gradient_length = np.divide(
        1.0,
        np.abs(slope),
        out=np.full_like(slope, np.inf),
        where=np.abs(slope) > 0.0,
    )
    over_width = gradient_length / grid.widths
    over_H = gradient_length / np.asarray(primitives.vertical.H, dtype=float)
    audit_stop = (
        grid.centers.size
        if crossing_interval is None
        else min(grid.centers.size, crossing_interval + 3)
    )
    audited_width = over_width[:audit_stop]
    audited_H = over_H[:audit_stop]
    width_index = int(np.argmin(audited_width))
    H_index = int(np.argmin(audited_H))
    return GlobalSonicResolutionDiagnostic(
        sonic_radius=crossing_radius,
        cells_between_inner_face_and_sonic_radius=cell_count,
        minimum_velocity_gradient_length_over_cell_width=float(
            audited_width[width_index]
        ),
        minimum_over_cell_width_index=width_index,
        minimum_over_cell_width_radius=float(grid.centers[width_index]),
        minimum_velocity_gradient_length_over_H=float(audited_H[H_index]),
        minimum_over_H_index=H_index,
        minimum_over_H_radius=float(grid.centers[H_index]),
        gradient_audit_outer_radius=float(grid.edges[audit_stop]),
    )


def global_roche_closure_diagnostic(
    boundary: GlobalRocheBoundaryAudit,
    provider: OverflowBoundaryProvider,
    *,
    mass_flux_scale: float,
) -> GlobalRocheClosureDiagnostic:
    """Normalize the Roche energy margin by the edge-to-saddle barrier."""

    if not np.isfinite(mass_flux_scale) or mass_flux_scale <= 0.0:
        raise ValueError("mass_flux_scale must be positive and finite")
    geometry = provider.geometry.validated()
    edge_radius = float(boundary.edge_state.radius)
    edge_potential = float(
        hill_roche_disk_matched_potential(
            edge_radius,
            geometry.secondary_mass,
            geometry.pattern_omega,
            edge_radius,
        )
    )
    saddle_potential = float(
        hill_roche_disk_matched_potential(
            geometry.saddle_radius,
            geometry.secondary_mass,
            geometry.pattern_omega,
            edge_radius,
        )
    )
    barrier = saddle_potential - edge_potential
    if not np.isfinite(barrier) or barrier <= 0.0:
        raise ValueError("Roche edge-to-saddle barrier must be positive")
    available = float(boundary.gate.available_specific_energy)
    normalized = available / barrier
    if boundary.gate.choked:
        solution = boundary.gate.solution
        if solution is None:
            raise ValueError("choked Roche gate lacks a nozzle solution")
        nozzle_residual = float(solution.sonic_residual)
        active_set_residual = max(0.0, -normalized) + abs(
            nozzle_residual / barrier
        )
        state = "choked"
    else:
        nozzle_residual = None
        active_set_residual = max(0.0, normalized) + abs(
            boundary.applied_mass_flux / mass_flux_scale
        )
        state = "closed"
    return GlobalRocheClosureDiagnostic(
        barrier_specific_energy=barrier,
        available_specific_energy=available,
        normalized_available_specific_energy=normalized,
        channel_state=state,
        active_set_residual=float(active_set_residual),
        nozzle_residual=nozzle_residual,
    )
