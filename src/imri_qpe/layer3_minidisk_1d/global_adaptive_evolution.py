"""Adaptive accepted-state continuation for global conservative evolution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .global_signed_evolution import (
    GlobalBackwardEulerStepResult,
    GlobalConservativeState,
    GlobalMechanicalEnergyReference,
    advance_global_backward_euler,
    recover_global_primitives,
)
from .grid import RadialGrid


def _state_hash(grid: RadialGrid, state: GlobalConservativeState) -> str:
    state = state.validated()
    digest = hashlib.sha256()
    for values in (
        grid.edges,
        state.mass,
        state.radial_momentum,
        state.angular_momentum,
        state.total_energy,
    ):
        array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class GlobalAdaptiveStepConfig:
    """Deterministic reject/halve/grow policy for one accepted step."""

    minimum_dt: float
    maximum_dt: float
    shrink_factor: float = 0.5
    growth_factor: float = 1.5
    maximum_retries: int = 8
    easy_nfev: int = 20
    maximum_log_surface_density_change: float = 0.05
    maximum_log_temperature_change: float = 0.05
    maximum_relative_thickness_change: float = 0.05

    def validated(self) -> GlobalAdaptiveStepConfig:
        positive = (
            self.minimum_dt,
            self.maximum_dt,
            self.shrink_factor,
            self.growth_factor,
            self.maximum_log_surface_density_change,
            self.maximum_log_temperature_change,
            self.maximum_relative_thickness_change,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("adaptive controller values must be positive and finite")
        if self.minimum_dt > self.maximum_dt:
            raise ValueError("minimum_dt must not exceed maximum_dt")
        if not self.shrink_factor < 1.0:
            raise ValueError("shrink_factor must be below one")
        if not self.growth_factor > 1.0:
            raise ValueError("growth_factor must exceed one")
        if int(self.maximum_retries) != self.maximum_retries or self.maximum_retries < 0:
            raise ValueError("maximum_retries must be a non-negative integer")
        if int(self.easy_nfev) != self.easy_nfev or self.easy_nfev < 1:
            raise ValueError("easy_nfev must be a positive integer")
        return self


@dataclass(frozen=True)
class GlobalAdaptiveAttempt:
    """One accepted or rejected nonlinear attempt."""

    dt: float
    nonlinear_accepted: bool
    physical_change_accepted: bool
    nfev: int
    maximum_scaled_residual: float
    maximum_log_surface_density_change: float
    maximum_log_temperature_change: float
    maximum_relative_thickness_change: float
    message: str


@dataclass(frozen=True)
class GlobalAdaptiveStepResult:
    """One adaptive step, including every rejected retry."""

    state: GlobalConservativeState
    accepted: bool
    dt_used: float
    dt_next: float
    step: GlobalBackwardEulerStepResult
    attempts: tuple[GlobalAdaptiveAttempt, ...]
    message: str


def advance_global_adaptive_backward_euler(
    grid: RadialGrid,
    state: GlobalConservativeState,
    M_g: float,
    dt: float,
    config: GlobalAdaptiveStepConfig,
    *,
    specific_mechanical_energy_correction=None,
    step_options: dict | None = None,
) -> GlobalAdaptiveStepResult:
    """Retry one physical backward-Euler step until all gates pass."""

    config = config.validated()
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("adaptive initial dt must be positive and finite")
    options = {} if step_options is None else dict(step_options)
    if "specific_mechanical_energy_correction" in options:
        raise ValueError("mechanical correction must use the dedicated argument")
    state = state.validated()
    recovery_options = {
        name: options[name]
        for name in (
            "temperature_bounds",
            "mu_mol",
            "kappa",
            "gamma_gas",
        )
        if name in options
    }
    old = recover_global_primitives(
        grid,
        state,
        M_g,
        specific_mechanical_energy_correction=(
            specific_mechanical_energy_correction
        ),
        **recovery_options,
    )
    old_thickness = np.asarray(old.vertical.H, dtype=float) / grid.centers
    trial_dt = float(np.clip(dt, config.minimum_dt, config.maximum_dt))
    attempts: list[GlobalAdaptiveAttempt] = []
    last_step: GlobalBackwardEulerStepResult | None = None
    for _retry in range(config.maximum_retries + 1):
        step = advance_global_backward_euler(
            grid,
            state,
            M_g,
            trial_dt,
            specific_mechanical_energy_correction=(
                specific_mechanical_energy_correction
            ),
            **options,
        )
        last_step = step
        sigma_change = np.inf
        temperature_change = np.inf
        thickness_change = np.inf
        physical_accepted = False
        message = step.message
        if step.accepted:
            new = recover_global_primitives(
                grid,
                step.state,
                M_g,
                specific_mechanical_energy_correction=(
                    specific_mechanical_energy_correction
                ),
                **recovery_options,
            )
            new_thickness = np.asarray(new.vertical.H, dtype=float) / grid.centers
            sigma_change = float(
                np.max(np.abs(np.log(new.surface_density / old.surface_density)))
            )
            temperature_change = float(
                np.max(np.abs(np.log(new.temperature / old.temperature)))
            )
            thickness_change = float(
                np.max(
                    np.abs(new_thickness - old_thickness)
                    / np.maximum(old_thickness, 1.0e-300)
                )
            )
            physical_accepted = bool(
                sigma_change <= config.maximum_log_surface_density_change
                and temperature_change <= config.maximum_log_temperature_change
                and thickness_change <= config.maximum_relative_thickness_change
            )
            if not physical_accepted:
                message = "accepted nonlinear root exceeds adaptive physical-change gate"
        attempts.append(
            GlobalAdaptiveAttempt(
                dt=trial_dt,
                nonlinear_accepted=step.accepted,
                physical_change_accepted=physical_accepted,
                nfev=step.nfev,
                maximum_scaled_residual=step.maximum_scaled_residual,
                maximum_log_surface_density_change=sigma_change,
                maximum_log_temperature_change=temperature_change,
                maximum_relative_thickness_change=thickness_change,
                message=message,
            )
        )
        if step.accepted and physical_accepted:
            easy = bool(
                step.nfev <= config.easy_nfev
                and sigma_change
                <= 0.5 * config.maximum_log_surface_density_change
                and temperature_change
                <= 0.5 * config.maximum_log_temperature_change
                and thickness_change
                <= 0.5 * config.maximum_relative_thickness_change
            )
            dt_next = trial_dt * config.growth_factor if easy else trial_dt
            return GlobalAdaptiveStepResult(
                state=step.state,
                accepted=True,
                dt_used=trial_dt,
                dt_next=float(min(dt_next, config.maximum_dt)),
                step=step,
                attempts=tuple(attempts),
                message="accepted",
            )
        next_dt = trial_dt * config.shrink_factor
        if next_dt < config.minimum_dt:
            break
        trial_dt = next_dt
    assert last_step is not None
    return GlobalAdaptiveStepResult(
        state=state,
        accepted=False,
        dt_used=0.0,
        dt_next=float(max(config.minimum_dt, trial_dt)),
        step=last_step,
        attempts=tuple(attempts),
        message="adaptive retries exhausted without an accepted state",
    )


@dataclass(frozen=True)
class GlobalAdaptiveRestart:
    """Restart-safe state, controller, and mechanical-reference payload."""

    state: GlobalConservativeState
    reference_state: GlobalConservativeState
    mechanical_reference: GlobalMechanicalEnergyReference
    elapsed_time: float
    dt_next: float
    accepted_steps: int
    rejected_attempts: int
    provenance: dict
    schema_version: int = 1


def save_global_adaptive_restart(
    path: str | Path,
    grid: RadialGrid,
    restart: GlobalAdaptiveRestart,
) -> None:
    """Store a complete deterministic adaptive continuation checkpoint."""

    state = restart.state.validated()
    reference = restart.reference_state.validated()
    if state.n_cells != grid.centers.size or reference.n_cells != state.n_cells:
        raise ValueError("adaptive restart states do not match the grid")
    mechanical = restart.mechanical_reference.validated_for(grid)
    if not np.isfinite(restart.elapsed_time) or restart.elapsed_time < 0.0:
        raise ValueError("restart elapsed_time must be finite and non-negative")
    if not np.isfinite(restart.dt_next) or restart.dt_next <= 0.0:
        raise ValueError("restart dt_next must be positive and finite")
    if restart.accepted_steps < 0 or restart.rejected_attempts < 0:
        raise ValueError("restart counters must be non-negative")
    provenance = json.dumps(
        restart.provenance, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    mechanical_provenance = json.dumps(
        mechanical.provenance,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    destination = Path(path)
    if destination.suffix != ".npz":
        raise ValueError("adaptive restart path must end in .npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        schema_version=np.asarray(restart.schema_version, dtype=np.int64),
        grid_edges=grid.edges,
        state_mass=state.mass,
        state_radial_momentum=state.radial_momentum,
        state_angular_momentum=state.angular_momentum,
        state_total_energy=state.total_energy,
        reference_mass=reference.mass,
        reference_radial_momentum=reference.radial_momentum,
        reference_angular_momentum=reference.angular_momentum,
        reference_total_energy=reference.total_energy,
        state_sha256=np.asarray(_state_hash(grid, state)),
        reference_sha256=np.asarray(_state_hash(grid, reference)),
        mechanical_offset=mechanical.specific_offset,
        mechanical_reference_state_sha256=np.asarray(
            mechanical.reference_state_sha256
        ),
        mechanical_offset_sha256=np.asarray(mechanical.offset_sha256),
        mechanical_provenance_json=np.asarray(mechanical_provenance),
        elapsed_time=np.asarray(restart.elapsed_time),
        dt_next=np.asarray(restart.dt_next),
        accepted_steps=np.asarray(restart.accepted_steps, dtype=np.int64),
        rejected_attempts=np.asarray(restart.rejected_attempts, dtype=np.int64),
        provenance_json=np.asarray(provenance),
    )


def load_global_adaptive_restart(
    path: str | Path,
    *,
    grid: RadialGrid | None = None,
) -> tuple[RadialGrid, GlobalAdaptiveRestart]:
    """Load and checksum every field required to resume adaptive evolution."""

    with np.load(Path(path), allow_pickle=False) as data:
        edges = np.asarray(data["grid_edges"], dtype=float)
        loaded_grid = RadialGrid(
            centers=np.sqrt(edges[:-1] * edges[1:]),
            edges=edges,
            widths=np.diff(edges),
            area=np.pi * (edges[1:] ** 2 - edges[:-1] ** 2),
        )
        if grid is not None and not np.array_equal(grid.edges, edges):
            raise ValueError("adaptive restart grid does not match")
        state = GlobalConservativeState(
            np.asarray(data["state_mass"], dtype=float),
            np.asarray(data["state_radial_momentum"], dtype=float),
            np.asarray(data["state_angular_momentum"], dtype=float),
            np.asarray(data["state_total_energy"], dtype=float),
        ).validated()
        reference = GlobalConservativeState(
            np.asarray(data["reference_mass"], dtype=float),
            np.asarray(data["reference_radial_momentum"], dtype=float),
            np.asarray(data["reference_angular_momentum"], dtype=float),
            np.asarray(data["reference_total_energy"], dtype=float),
        ).validated()
        if str(data["state_sha256"].item()) != _state_hash(loaded_grid, state):
            raise ValueError("adaptive restart state checksum mismatch")
        if str(data["reference_sha256"].item()) != _state_hash(
            loaded_grid, reference
        ):
            raise ValueError("adaptive restart reference checksum mismatch")
        mechanical = GlobalMechanicalEnergyReference(
            grid_edges=edges,
            specific_offset=np.asarray(data["mechanical_offset"], dtype=float),
            reference_state_sha256=str(
                data["mechanical_reference_state_sha256"].item()
            ),
            offset_sha256=str(data["mechanical_offset_sha256"].item()),
            provenance=json.loads(
                str(data["mechanical_provenance_json"].item())
            ),
        ).validated_for(loaded_grid)
        restart = GlobalAdaptiveRestart(
            state=state,
            reference_state=reference,
            mechanical_reference=mechanical,
            elapsed_time=float(data["elapsed_time"]),
            dt_next=float(data["dt_next"]),
            accepted_steps=int(data["accepted_steps"]),
            rejected_attempts=int(data["rejected_attempts"]),
            provenance=json.loads(str(data["provenance_json"].item())),
            schema_version=int(data["schema_version"]),
        )
    if restart.schema_version != 1:
        raise ValueError("unsupported adaptive restart schema version")
    return loaded_grid, restart
