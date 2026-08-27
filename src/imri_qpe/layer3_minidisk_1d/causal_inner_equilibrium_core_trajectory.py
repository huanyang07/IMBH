"""Accepted-state-only trajectory and restart for the equilibrium core."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np

from .causal_inner_conservative_entropy_projection_microstep import (
    ConservativeEntropyProjectionResult,
    EquilibriumPrimitiveSeed,
    conservative_entropy_projected_midpoint_microstep,
)
from .causal_inner_geometry import (
    KerrSchildColumnGeometry,
    kerr_schild_column_geometry,
)
from .causal_inner_nonlinear_port_atlas import (
    EquilibriumEntropyPoint,
    equilibrium_entropy_point_from_primitive,
    equilibrium_mathematical_entropy_decimal,
    equilibrium_temporal_conserved,
)


@dataclass(frozen=True)
class EquilibriumCoreTrajectoryState:
    geometry: KerrSchildColumnGeometry
    proper_half_thickness: float
    points: tuple[EquilibriumEntropyPoint, ...]
    seeds: tuple[EquilibriumPrimitiveSeed, ...]
    accepted_steps: int
    accumulated_courant_time: float
    initial_conserved_total: np.ndarray
    initial_entropy_decimal: str


@dataclass(frozen=True)
class EquilibriumCoreTrajectoryDiagnostics:
    cumulative_conservation_relative_defect: float
    cumulative_entropy_relative_defect: float


@dataclass(frozen=True)
class EquilibriumCoreTrajectoryAdvance:
    state: EquilibriumCoreTrajectoryState
    microstep: ConservativeEntropyProjectionResult
    accepted: bool


def _total_entropy(points: tuple[EquilibriumEntropyPoint, ...]) -> Decimal:
    with localcontext() as context:
        context.prec = 50
        return sum(
            (equilibrium_mathematical_entropy_decimal(point) for point in points),
            Decimal(0),
        )


def initialize_equilibrium_core_trajectory(
    *,
    geometry: KerrSchildColumnGeometry,
    proper_half_thickness: float,
    points,
    seeds,
) -> EquilibriumCoreTrajectoryState:
    points = tuple(points)
    seeds = tuple(seeds)
    if len(points) < 3 or len(points) != len(seeds):
        raise ValueError("trajectory needs matching periodic points and seeds")
    total = np.sum(
        np.asarray(
            [equilibrium_temporal_conserved(point) for point in points], dtype=float
        ),
        axis=0,
    )
    return EquilibriumCoreTrajectoryState(
        geometry,
        float(proper_half_thickness),
        points,
        seeds,
        0,
        0.0,
        total,
        str(_total_entropy(points)),
    )


def advance_equilibrium_core_trajectory(
    state: EquilibriumCoreTrajectoryState,
    *,
    courant_factor: float,
) -> EquilibriumCoreTrajectoryAdvance:
    if not isinstance(state, EquilibriumCoreTrajectoryState):
        raise TypeError("state must be EquilibriumCoreTrajectoryState")
    result = conservative_entropy_projected_midpoint_microstep(
        geometry=state.geometry,
        proper_half_thickness=state.proper_half_thickness,
        points=state.points,
        seeds=state.seeds,
        courant_factor=courant_factor,
    )
    if not result.passed:
        return EquilibriumCoreTrajectoryAdvance(state, result, False)
    advanced = EquilibriumCoreTrajectoryState(
        state.geometry,
        state.proper_half_thickness,
        result.points,
        result.seeds,
        state.accepted_steps + 1,
        state.accumulated_courant_time + float(courant_factor),
        state.initial_conserved_total,
        state.initial_entropy_decimal,
    )
    return EquilibriumCoreTrajectoryAdvance(advanced, result, True)


def audit_equilibrium_core_trajectory(
    state: EquilibriumCoreTrajectoryState,
) -> EquilibriumCoreTrajectoryDiagnostics:
    current_total = np.sum(
        np.asarray(
            [equilibrium_temporal_conserved(point) for point in state.points],
            dtype=float,
        ),
        axis=0,
    )
    conserved_scale = max(
        float(np.max(np.abs(state.initial_conserved_total))), 1.0
    )
    conservation = float(
        np.linalg.norm(current_total - state.initial_conserved_total, ord=np.inf)
        / conserved_scale
    )
    initial_entropy = Decimal(state.initial_entropy_decimal)
    entropy_scale = max(abs(initial_entropy), Decimal(1))
    entropy = float(abs(_total_entropy(state.points) - initial_entropy) / entropy_scale)
    return EquilibriumCoreTrajectoryDiagnostics(conservation, entropy)


def trajectory_primitive_array(state: EquilibriumCoreTrajectoryState) -> np.ndarray:
    return np.asarray(
        [
            (
                seed.density,
                seed.temperature,
                seed.radial_velocity_over_c,
                seed.azimuthal_velocity_over_c,
            )
            for seed in state.seeds
        ],
        dtype=float,
    )


def save_equilibrium_core_trajectory_checkpoint(
    path: str | Path,
    state: EquilibriumCoreTrajectoryState,
) -> None:
    destination = Path(path)
    np.savez_compressed(
        destination,
        schema_version=np.asarray(1, dtype=np.int64),
        radius=np.asarray(state.geometry.radius, dtype=float),
        gravitational_radius=np.asarray(
            state.geometry.gravitational_radius, dtype=float
        ),
        proper_half_thickness=np.asarray(state.proper_half_thickness, dtype=float),
        primitive=trajectory_primitive_array(state),
        accepted_steps=np.asarray(state.accepted_steps, dtype=np.int64),
        accumulated_courant_time=np.asarray(
            state.accumulated_courant_time, dtype=float
        ),
        initial_conserved_total=np.asarray(
            state.initial_conserved_total, dtype=float
        ),
        initial_entropy_decimal=np.asarray(state.initial_entropy_decimal),
    )


def load_equilibrium_core_trajectory_checkpoint(
    path: str | Path,
) -> EquilibriumCoreTrajectoryState:
    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        if int(payload["schema_version"]) != 1:
            raise ValueError("unsupported equilibrium-core checkpoint schema")
        radius = float(payload["radius"])
        gravitational_radius = float(payload["gravitational_radius"])
        height = float(payload["proper_half_thickness"])
        primitive = np.asarray(payload["primitive"], dtype=float)
        accepted_steps = int(payload["accepted_steps"])
        accumulated = float(payload["accumulated_courant_time"])
        initial_total = np.asarray(payload["initial_conserved_total"], dtype=float)
        initial_entropy = str(payload["initial_entropy_decimal"])
    if primitive.ndim != 2 or primitive.shape[1] != 4 or primitive.shape[0] < 3:
        raise ValueError("checkpoint primitive payload has the wrong shape")
    geometry = kerr_schild_column_geometry(radius, gravitational_radius)
    seeds = tuple(
        EquilibriumPrimitiveSeed(*map(float, row)) for row in primitive
    )
    points = tuple(
        equilibrium_entropy_point_from_primitive(
            geometry,
            density=seed.density,
            temperature=seed.temperature,
            proper_half_thickness=height,
            radial_velocity_over_c=seed.radial_velocity_over_c,
            azimuthal_velocity_over_c=seed.azimuthal_velocity_over_c,
        )
        for seed in seeds
    )
    return EquilibriumCoreTrajectoryState(
        geometry,
        height,
        points,
        seeds,
        accepted_steps,
        accumulated,
        initial_total,
        initial_entropy,
    )


__all__ = [
    "EquilibriumCoreTrajectoryAdvance",
    "EquilibriumCoreTrajectoryDiagnostics",
    "EquilibriumCoreTrajectoryState",
    "advance_equilibrium_core_trajectory",
    "audit_equilibrium_core_trajectory",
    "initialize_equilibrium_core_trajectory",
    "load_equilibrium_core_trajectory_checkpoint",
    "save_equilibrium_core_trajectory_checkpoint",
    "trajectory_primitive_array",
]
