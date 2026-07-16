"""Adaptive accepted-state continuation for global conservative evolution."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .global_signed_evolution import (
    GlobalBackwardEulerStepResult,
    GlobalConservativeState,
    GlobalMechanicalEnergyReference,
    GlobalNonlinearSolveAudit,
    advance_global_backward_euler,
    global_effective_sound_speed,
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
class GlobalAdaptiveControllerAudit:
    """Cell and characteristic state controlling one physical-change gate."""

    variable: str
    cell_index: int
    radius: float
    old_value: float
    new_value: float
    change_metric: float
    limit: float
    fraction_of_limit: float
    radial_mach_number: float
    characteristic_speeds: tuple[float, float, float, float]
    causally_disconnected_from_outer_disk: bool


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
    controller: GlobalAdaptiveControllerAudit | None = None
    nonlinear_solve_audit: GlobalNonlinearSolveAudit | None = None


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
        controller = None
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
            sigma_changes = np.abs(
                np.log(new.surface_density / old.surface_density)
            )
            temperature_changes = np.abs(
                np.log(new.temperature / old.temperature)
            )
            thickness_changes = (
                np.abs(new_thickness - old_thickness)
                / np.maximum(old_thickness, 1.0e-300)
            )
            sigma_change = float(np.max(sigma_changes))
            temperature_change = float(np.max(temperature_changes))
            thickness_change = float(np.max(thickness_changes))
            candidates = (
                (
                    "log_surface_density",
                    sigma_changes,
                    old.surface_density,
                    new.surface_density,
                    config.maximum_log_surface_density_change,
                ),
                (
                    "log_temperature",
                    temperature_changes,
                    old.temperature,
                    new.temperature,
                    config.maximum_log_temperature_change,
                ),
                (
                    "relative_thickness",
                    thickness_changes,
                    old_thickness,
                    new_thickness,
                    config.maximum_relative_thickness_change,
                ),
            )
            controlling = max(
                candidates,
                key=lambda item: float(np.max(item[1])) / item[4],
            )
            variable, changes, old_values, new_values, limit = controlling
            cell_index = int(np.argmax(changes))
            sound_speed = global_effective_sound_speed(new)
            velocity = float(new.radial_velocity[cell_index])
            sound = float(sound_speed[cell_index])
            characteristic_speeds = (
                velocity - sound,
                velocity,
                velocity,
                velocity + sound,
            )
            controller = GlobalAdaptiveControllerAudit(
                variable=variable,
                cell_index=cell_index,
                radius=float(grid.centers[cell_index]),
                old_value=float(old_values[cell_index]),
                new_value=float(new_values[cell_index]),
                change_metric=float(changes[cell_index]),
                limit=float(limit),
                fraction_of_limit=float(changes[cell_index] / limit),
                radial_mach_number=velocity / sound,
                characteristic_speeds=characteristic_speeds,
                causally_disconnected_from_outer_disk=bool(
                    max(characteristic_speeds) < 0.0
                ),
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
                controller=controller,
                nonlinear_solve_audit=getattr(
                    step, "nonlinear_solve_audit", None
                ),
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


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_global_adaptive_milestone(
    directory: str | Path,
    case: str,
    grid: RadialGrid,
    restart: GlobalAdaptiveRestart,
    *,
    metadata: dict | None = None,
) -> dict:
    """Write one immutable checkpoint and append its checksums to a manifest."""

    safe_case = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(case)).strip("-.")
    if not safe_case:
        raise ValueError("milestone case must contain a filesystem-safe character")
    state_sha256 = _state_hash(grid, restart.state)
    git_sha = str(
        restart.provenance.get("git", {}).get("full_sha") or "unknown"
    )
    git_tag = re.sub(r"[^A-Za-z0-9]+", "", git_sha)[:12] or "unknown"
    time_tag = f"{float(restart.elapsed_time):.17e}"
    filename = (
        f"{safe_case}_N{grid.centers.size:03d}_tphys_{time_tag}_"
        f"git_{git_tag}_state_{state_sha256[:12]}.npz"
    )
    destination_directory = Path(directory)
    destination_directory.mkdir(parents=True, exist_ok=True)
    destination = destination_directory / filename
    if destination.exists():
        loaded_grid, loaded = load_global_adaptive_restart(
            destination, grid=grid
        )
        if (
            not np.array_equal(loaded_grid.edges, grid.edges)
            or _state_hash(grid, loaded.state) != state_sha256
            or loaded.elapsed_time != restart.elapsed_time
            or loaded.provenance != restart.provenance
        ):
            raise ValueError("existing milestone does not match requested state")
    else:
        temporary = destination.with_name(f".{destination.name}.tmp.npz")
        save_global_adaptive_restart(temporary, grid, restart)
        os.replace(temporary, destination)
    checkpoint_sha256 = _file_sha256(destination)
    entry = {
        "path": destination.name,
        "checkpoint_sha256": checkpoint_sha256,
        "state_sha256": state_sha256,
        "reference_state_sha256": _state_hash(grid, restart.reference_state),
        "mechanical_offset_sha256": (
            restart.mechanical_reference.offset_sha256
        ),
        "n_cells": int(grid.centers.size),
        "elapsed_time_seconds": float(restart.elapsed_time),
        "dt_next_seconds": float(restart.dt_next),
        "accepted_steps": int(restart.accepted_steps),
        "rejected_attempts": int(restart.rejected_attempts),
        "metadata": {} if metadata is None else metadata,
    }
    json.dumps(entry, sort_keys=True, allow_nan=False)
    manifest_path = destination_directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"schema_version": 1, "checkpoints": []}
    if manifest.get("schema_version") != 1 or not isinstance(
        manifest.get("checkpoints"), list
    ):
        raise ValueError("unsupported adaptive milestone manifest")
    matching = [
        item for item in manifest["checkpoints"] if item.get("path") == filename
    ]
    if matching and matching[0] != entry:
        raise ValueError("milestone manifest entry conflicts with checkpoint")
    if not matching:
        manifest["checkpoints"].append(entry)
        manifest["checkpoints"].sort(
            key=lambda item: (
                item["elapsed_time_seconds"], item["n_cells"], item["path"]
            )
        )
        temporary_manifest = manifest_path.with_name(".manifest.json.tmp")
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_manifest, manifest_path)
    return entry


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
