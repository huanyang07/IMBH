"""Production-size periodic Fourier proof kernel for the eleven-field AP port."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.linalg import expm

from .causal_inner_bounded_ap_trajectory import APAtlasPath


@dataclass(frozen=True)
class GlobalFourierAPResult:
    final_state: np.ndarray
    final_time: float
    step_count: int
    maximum_state_norm: float
    maximum_homogeneous_mode_expansivity: float
    maximum_core_total_conservation_defect: float
    wall_seconds: float


def deterministic_global_initial_state(cell_count: int) -> np.ndarray:
    cells = np.arange(int(cell_count), dtype=float)[:, None]
    fields = np.arange(1.0, 12.0)[None, :]
    phase = 2.0 * np.pi * cells / float(cell_count)
    baseline = np.where(fields <= 4.0, 1.5e-2 / fields, 0.0)
    perturbation = 7.5e-3 * (
        np.cos(phase * (1.0 + np.mod(fields, 4.0)))
        + 0.5 * np.sin(phase * (2.0 + np.mod(fields, 3.0)))
    )
    return np.asarray(baseline + perturbation, dtype=complex)


def deterministic_global_forcing(
    time: float,
    horizon: float,
    cell_count: int,
) -> np.ndarray:
    cells = np.arange(int(cell_count), dtype=float)[:, None]
    fields = np.arange(1.0, 12.0)[None, :]
    spatial_phase = 2.0 * np.pi * cells / float(cell_count)
    temporal_phase = 2.0 * np.pi * float(time) / float(horizon)
    amplitude = np.where(fields <= 4.0, 2.0e-3, 5.0e-4)
    # Every field has exactly zero spatial mean.  Thus the forcing tests the
    # affine action without changing the periodic core totals.
    return np.asarray(
        amplitude
        * (
            np.cos((1.0 + np.mod(fields, 3.0)) * spatial_phase + temporal_phase)
            + 0.35j
            * np.sin((2.0 + np.mod(fields, 4.0)) * spatial_phase - temporal_phase)
        ),
        dtype=complex,
    )


def _mode_generator(
    radial: np.ndarray,
    source: np.ndarray,
    *,
    angle: float,
    spacing: float,
    stiffness: float,
) -> np.ndarray:
    wave_number = np.sin(float(angle)) / float(spacing)
    rusanov_speed = float(np.max(np.abs(np.linalg.eigvalsh(radial))))
    dissipation = rusanov_speed * (np.cos(float(angle)) - 1.0) / float(spacing)
    return (
        -1j * wave_number * np.asarray(radial)
        + float(stiffness) * np.asarray(source)
        + dissipation * np.eye(11)
    )


def global_fourier_ap_step(
    path: APAtlasPath,
    state: np.ndarray,
    *,
    time: float,
    timestep: float,
    atlas_horizon: float,
    stiffness: float,
    forcing: Callable[[float], np.ndarray],
    spacing: float = 1.0,
    audit_expansivity: bool = True,
) -> tuple[np.ndarray, float]:
    value = np.asarray(state, dtype=complex)
    if value.ndim != 2 or value.shape[1] != 11:
        raise ValueError("the global AP state must have shape (N_R,11)")
    cell_count = value.shape[0]
    midpoint = float(time) + 0.5 * float(timestep)
    radial, source = path.matrices(midpoint, atlas_horizon)
    drive = np.asarray(forcing(midpoint), dtype=complex)
    if drive.shape != value.shape:
        raise ValueError("the global AP forcing must match the state")
    state_modes = np.fft.fft(value, axis=0, norm="ortho")
    drive_modes = np.fft.fft(drive, axis=0, norm="ortho")
    result_modes = np.empty_like(state_modes)
    maximum_expansivity = 0.0
    for mode in range(cell_count):
        angle = 2.0 * np.pi * mode / float(cell_count)
        generator = _mode_generator(
            radial,
            source,
            angle=angle,
            spacing=spacing,
            stiffness=stiffness,
        )
        augmented = np.zeros((12, 12), dtype=complex)
        augmented[:11, :11] = float(timestep) * generator
        augmented[:11, 11] = float(timestep) * drive_modes[mode]
        propagator = expm(augmented)
        result_modes[mode] = (
            propagator
            @ np.concatenate((state_modes[mode], np.ones(1, dtype=complex)))
        )[:11]
        if audit_expansivity:
            homogeneous = propagator[:11, :11]
            maximum_expansivity = max(
                maximum_expansivity,
                max(
                    float(np.linalg.svd(homogeneous, compute_uv=False)[0] - 1.0),
                    0.0,
                ),
            )
    return np.fft.ifft(result_modes, axis=0, norm="ortho"), maximum_expansivity


def integrate_global_fourier_ap(
    path: APAtlasPath,
    initial_state: np.ndarray,
    *,
    start_time: float,
    end_time: float,
    atlas_horizon: float,
    step_count: int,
    stiffness: float,
    forcing: Callable[[float], np.ndarray],
    spacing: float = 1.0,
    audit_expansivity: bool = True,
) -> GlobalFourierAPResult:
    import time as clock

    began = clock.perf_counter()
    state = np.asarray(initial_state, dtype=complex).copy()
    timestep = (float(end_time) - float(start_time)) / int(step_count)
    current = float(start_time)
    initial_totals = np.sum(state[:, :4], axis=0)
    total_scale = max(float(np.linalg.norm(initial_totals)), 1.0)
    maximum_norm = float(np.linalg.norm(state))
    maximum_expansivity = 0.0
    maximum_conservation = 0.0
    for _ in range(int(step_count)):
        state, expansivity = global_fourier_ap_step(
            path,
            state,
            time=current,
            timestep=timestep,
            atlas_horizon=atlas_horizon,
            stiffness=stiffness,
            forcing=forcing,
            spacing=spacing,
            audit_expansivity=audit_expansivity,
        )
        current += timestep
        maximum_norm = max(maximum_norm, float(np.linalg.norm(state)))
        maximum_expansivity = max(maximum_expansivity, expansivity)
        totals = np.sum(state[:, :4], axis=0)
        maximum_conservation = max(
            maximum_conservation,
            float(np.linalg.norm(totals - initial_totals) / total_scale),
        )
    return GlobalFourierAPResult(
        state,
        current,
        int(step_count),
        maximum_norm,
        maximum_expansivity,
        maximum_conservation,
        clock.perf_counter() - began,
    )


__all__ = [
    "GlobalFourierAPResult",
    "deterministic_global_forcing",
    "deterministic_global_initial_state",
    "global_fourier_ap_step",
    "integrate_global_fourier_ap",
]
