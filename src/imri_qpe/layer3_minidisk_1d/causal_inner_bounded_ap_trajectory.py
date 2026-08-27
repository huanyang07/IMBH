"""Bounded coarse trajectories for the eleven-field entropy-port AP model.

The online object contains only hash-lockable matrices.  Physical witness
construction belongs to the offline atlas builder and is deliberately absent
from the stepping path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.linalg import expm


ComplexVectorFunction = Callable[[float], np.ndarray]
ScalarFunction = Callable[[float], float]


@dataclass(frozen=True)
class APAtlasPath:
    radial_start: np.ndarray
    source_start: np.ndarray
    radial_end: np.ndarray
    source_end: np.ndarray

    def __post_init__(self):
        matrices = tuple(
            np.asarray(value, dtype=float)
            for value in (
                self.radial_start,
                self.source_start,
                self.radial_end,
                self.source_end,
            )
        )
        if any(matrix.shape != (11, 11) for matrix in matrices):
            raise ValueError("the AP atlas path requires four 11x11 matrices")
        if any(np.linalg.norm(matrix - matrix.T) > 2.0e-11 for matrix in matrices):
            raise ValueError("the AP atlas matrices must be symmetric")
        for source in (matrices[1], matrices[3]):
            if np.max(np.linalg.eigvalsh(source)) > 2.0e-11:
                raise ValueError("the AP source must be entropy dissipative")
        for name, matrix in zip(
            ("radial_start", "source_start", "radial_end", "source_end"),
            matrices,
        ):
            object.__setattr__(self, name, matrix)

    @staticmethod
    def interpolation_weight(time: float, horizon: float) -> float:
        coordinate = float(np.clip(time / horizon, 0.0, 1.0))
        return coordinate * coordinate * (3.0 - 2.0 * coordinate)

    def matrices(self, time: float, horizon: float) -> tuple[np.ndarray, np.ndarray]:
        weight = self.interpolation_weight(time, horizon)
        return (
            (1.0 - weight) * self.radial_start + weight * self.radial_end,
            (1.0 - weight) * self.source_start + weight * self.source_end,
        )


@dataclass(frozen=True)
class APTrajectoryResult:
    final_state: np.ndarray
    step_count: int
    final_time: float
    maximum_state_norm: float
    maximum_homogeneous_step_expansivity: float
    wall_seconds: float


@dataclass(frozen=True)
class APTrajectoryCheckpoint:
    path: APAtlasPath
    state: np.ndarray
    time: float
    atlas_horizon: float
    stiffness: float
    completed_steps: int


def deterministic_slow_forcing(time: float, horizon: float) -> np.ndarray:
    """A smooth nonzero drive used only by the architecture certificate."""
    phase = 2.0 * np.pi * float(time) / float(horizon)
    indices = np.arange(1.0, 12.0)
    amplitude = np.where(indices <= 4.0, 8.0e-3, 1.5e-3)
    return amplitude * (
        np.cos(phase * indices / 11.0)
        + 1j * np.sin(phase * (12.0 - indices) / 13.0)
    )


def deterministic_initial_state() -> np.ndarray:
    indices = np.arange(1.0, 12.0)
    return 2.5e-2 * (
        np.cos(0.37 * indices) + 1j * np.sin(0.23 * indices)
    )


def exponential_midpoint_ap_step(
    path: APAtlasPath,
    state: np.ndarray,
    *,
    time: float,
    timestep: float,
    horizon: float,
    stiffness: float,
    wave_number: ScalarFunction,
    forcing: ComplexVectorFunction,
) -> tuple[np.ndarray, float]:
    midpoint = float(time) + 0.5 * float(timestep)
    radial, source = path.matrices(midpoint, horizon)
    generator = -1j * float(wave_number(midpoint)) * radial + float(stiffness) * source
    drive = np.asarray(forcing(midpoint), dtype=complex)
    value = np.asarray(state, dtype=complex)
    if value.shape != (11,) or drive.shape != (11,):
        raise ValueError("AP state and forcing must have eleven entries")
    augmented = np.zeros((12, 12), dtype=complex)
    augmented[:11, :11] = float(timestep) * generator
    augmented[:11, 11] = float(timestep) * drive
    propagator = expm(augmented)
    result = propagator @ np.concatenate((value, np.ones(1, dtype=complex)))
    homogeneous = propagator[:11, :11]
    expansivity = max(float(np.linalg.svd(homogeneous, compute_uv=False)[0] - 1.0), 0.0)
    return result[:11], expansivity


def integrate_ap_trajectory(
    path: APAtlasPath,
    initial_state: np.ndarray,
    *,
    start_time: float,
    end_time: float,
    atlas_horizon: float,
    step_count: int,
    stiffness: float,
    wave_number: ScalarFunction,
    forcing: ComplexVectorFunction,
) -> APTrajectoryResult:
    import time as clock

    began = clock.perf_counter()
    state = np.asarray(initial_state, dtype=complex).copy()
    timestep = (float(end_time) - float(start_time)) / int(step_count)
    maximum_norm = float(np.linalg.norm(state))
    maximum_expansivity = 0.0
    current = float(start_time)
    for _ in range(int(step_count)):
        state, expansivity = exponential_midpoint_ap_step(
            path,
            state,
            time=current,
            timestep=timestep,
            horizon=atlas_horizon,
            stiffness=stiffness,
            wave_number=wave_number,
            forcing=forcing,
        )
        current += timestep
        maximum_norm = max(maximum_norm, float(np.linalg.norm(state)))
        maximum_expansivity = max(maximum_expansivity, expansivity)
    return APTrajectoryResult(
        state,
        int(step_count),
        current,
        maximum_norm,
        maximum_expansivity,
        clock.perf_counter() - began,
    )


def source_nullity(source: np.ndarray, tolerance: float = 1.0e-11) -> int:
    return int(np.count_nonzero(np.abs(np.linalg.eigvalsh(source)) <= tolerance))


def save_ap_checkpoint(checkpoint: APTrajectoryCheckpoint, path) -> None:
    np.savez(
        path,
        radial_start=checkpoint.path.radial_start,
        source_start=checkpoint.path.source_start,
        radial_end=checkpoint.path.radial_end,
        source_end=checkpoint.path.source_end,
        state=np.asarray(checkpoint.state, dtype=complex),
        time=np.asarray(checkpoint.time),
        atlas_horizon=np.asarray(checkpoint.atlas_horizon),
        stiffness=np.asarray(checkpoint.stiffness),
        completed_steps=np.asarray(checkpoint.completed_steps, dtype=np.int64),
    )


def load_ap_checkpoint(path) -> APTrajectoryCheckpoint:
    with np.load(path, allow_pickle=False) as payload:
        atlas = APAtlasPath(
            payload["radial_start"],
            payload["source_start"],
            payload["radial_end"],
            payload["source_end"],
        )
        return APTrajectoryCheckpoint(
            atlas,
            np.asarray(payload["state"], dtype=complex),
            float(payload["time"]),
            float(payload["atlas_horizon"]),
            float(payload["stiffness"]),
            int(payload["completed_steps"]),
        )


def fast_slaving_defect(
    path: APAtlasPath,
    state: np.ndarray,
    *,
    time: float,
    horizon: float,
    stiffness: float,
    wave_number: ScalarFunction,
    forcing: ComplexVectorFunction,
) -> float:
    radial, source = path.matrices(time, horizon)
    generator = -1j * float(wave_number(time)) * radial + float(stiffness) * source
    drive = np.asarray(forcing(time), dtype=complex)
    slow = np.asarray(state[:4], dtype=complex)
    fast = np.asarray(state[4:], dtype=complex)
    target = -np.linalg.solve(generator[4:, 4:], generator[4:, :4] @ slow + drive[4:])
    return float(np.linalg.norm(fast - target) / max(np.linalg.norm(state), np.finfo(float).tiny))


__all__ = [
    "APAtlasPath",
    "APTrajectoryCheckpoint",
    "APTrajectoryResult",
    "deterministic_initial_state",
    "deterministic_slow_forcing",
    "exponential_midpoint_ap_step",
    "fast_slaving_defect",
    "integrate_ap_trajectory",
    "load_ap_checkpoint",
    "save_ap_checkpoint",
    "source_nullity",
]
