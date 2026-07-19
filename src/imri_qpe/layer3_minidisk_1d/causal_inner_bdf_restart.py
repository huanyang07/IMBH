"""Complete restart payload for increment-primary causal BDF evolution."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .causal_inner_bdf import (
    CausalFiveFieldBDFHistory,
    causal_bdf_coefficients,
)
from .causal_inner_dae import causal_five_field_dae_count
from .causal_inner_dae_system import CausalFiveFieldDAEContext


@dataclass(frozen=True)
class CausalFiveFieldBDFRestart:
    """State, two-step history, controller state, and provenance."""

    state_vector: np.ndarray
    history: CausalFiveFieldBDFHistory
    elapsed_time: float
    dt_next: float
    next_order: int
    accepted_steps: int
    rejected_attempts: int
    provenance: dict
    schema_version: int = 1


@dataclass(frozen=True)
class CausalFiveFieldAdaptiveBDF2Restart:
    """Complete adaptive BDF2 state, estimator history, and horizon ledger."""

    state_vector: np.ndarray
    history: CausalFiveFieldBDFHistory
    older_physical_increment: np.ndarray
    older_timestep_seconds: float
    cumulative_actual_conserved_storage: np.ndarray
    cumulative_actual_vertical_storage: np.ndarray
    cumulative_boundary_transport: np.ndarray
    cumulative_endogenous_source: np.ndarray
    cumulative_stream_source: np.ndarray
    cumulative_closure_defect: np.ndarray
    elapsed_time: float
    dt_next: float
    next_order: int
    accepted_steps: int
    accepted_bdf2_steps: int
    rejected_attempts: int
    audit_count: int
    provenance: dict
    schema_version: int = 1


def _validated_restart(
    context: CausalFiveFieldDAEContext,
    restart: CausalFiveFieldBDFRestart,
) -> CausalFiveFieldBDFRestart:
    context = context.validated()
    n_cells = int(context.grid.centers.size)
    count = causal_five_field_dae_count(n_cells)
    state = np.asarray(restart.state_vector, dtype=float)
    history = restart.history.validated(
        total_unknowns=count.total_unknowns,
        n_cells=n_cells,
    )
    elapsed = float(restart.elapsed_time)
    dt_next = float(restart.dt_next)
    if (
        state.shape != (count.total_unknowns,)
        or np.any(~np.isfinite(state))
        or not np.isfinite(elapsed)
        or elapsed < 0.0
        or not np.isfinite(dt_next)
        or dt_next <= 0.0
        or int(restart.next_order) != restart.next_order
        or restart.next_order not in (1, 2)
        or restart.accepted_steps < 0
        or restart.rejected_attempts < 0
        or restart.schema_version != 1
        or not isinstance(restart.provenance, dict)
    ):
        raise ValueError("causal BDF restart is invalid")
    if restart.next_order == 2:
        causal_bdf_coefficients(
            2,
            dt_next,
            history.previous_timestep_seconds,
        )
    return CausalFiveFieldBDFRestart(
        state_vector=state,
        history=history,
        elapsed_time=elapsed,
        dt_next=dt_next,
        next_order=int(restart.next_order),
        accepted_steps=int(restart.accepted_steps),
        rejected_attempts=int(restart.rejected_attempts),
        provenance=dict(restart.provenance),
        schema_version=1,
    )


def _restart_hash(
    context: CausalFiveFieldDAEContext,
    restart: CausalFiveFieldBDFRestart,
) -> str:
    digest = hashlib.sha256()
    arrays = (
        context.grid.edges,
        restart.state_vector,
        restart.history.previous_physical_increment,
        restart.history.previous_vertical_killing_increment,
        np.asarray(
            (
                restart.elapsed_time,
                restart.dt_next,
                restart.history.previous_timestep_seconds,
            ),
            dtype="<f8",
        ),
        np.asarray(
            (
                restart.next_order,
                restart.accepted_steps,
                restart.rejected_attempts,
                restart.schema_version,
            ),
            dtype="<i8",
        ),
    )
    for values in arrays:
        array = np.ascontiguousarray(values)
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    digest.update(
        restart.history.temporal_height_scheme.encode("ascii")
    )
    return digest.hexdigest()


def _validated_adaptive_bdf2_restart(
    context: CausalFiveFieldDAEContext,
    restart: CausalFiveFieldAdaptiveBDF2Restart,
) -> CausalFiveFieldAdaptiveBDF2Restart:
    context = context.validated()
    n_cells = int(context.grid.centers.size)
    count = causal_five_field_dae_count(n_cells)
    state = np.asarray(restart.state_vector, dtype=float)
    history = restart.history.validated(
        total_unknowns=count.total_unknowns,
        n_cells=n_cells,
    )
    older_increment = np.asarray(
        restart.older_physical_increment,
        dtype=float,
    )
    ledger_arrays = tuple(
        np.asarray(values, dtype=float)
        for values in (
            restart.cumulative_actual_conserved_storage,
            restart.cumulative_actual_vertical_storage,
            restart.cumulative_boundary_transport,
            restart.cumulative_endogenous_source,
            restart.cumulative_stream_source,
            restart.cumulative_closure_defect,
        )
    )
    older_dt = float(restart.older_timestep_seconds)
    elapsed = float(restart.elapsed_time)
    dt_next = float(restart.dt_next)
    integer_values = (
        restart.next_order,
        restart.accepted_steps,
        restart.accepted_bdf2_steps,
        restart.rejected_attempts,
        restart.audit_count,
        restart.schema_version,
    )
    if (
        state.shape != (count.total_unknowns,)
        or older_increment.shape != state.shape
        or np.any(~np.isfinite(state))
        or np.any(~np.isfinite(older_increment))
        or any(
            values.shape != (5,) or np.any(~np.isfinite(values))
            for values in ledger_arrays
        )
        or not np.isfinite(older_dt)
        or older_dt <= 0.0
        or not np.isfinite(elapsed)
        or elapsed < 0.0
        or not np.isfinite(dt_next)
        or dt_next <= 0.0
        or any(int(value) != value for value in integer_values)
        or restart.next_order not in (1, 2)
        or restart.accepted_steps < 0
        or restart.accepted_bdf2_steps < 0
        or restart.accepted_bdf2_steps > restart.accepted_steps
        or restart.rejected_attempts < 0
        or restart.audit_count < 0
        or restart.schema_version != 1
        or not isinstance(restart.provenance, dict)
    ):
        raise ValueError("adaptive causal BDF2 restart is invalid")
    if restart.next_order == 2:
        causal_bdf_coefficients(
            2,
            dt_next,
            history.previous_timestep_seconds,
        )
    return CausalFiveFieldAdaptiveBDF2Restart(
        state_vector=state,
        history=history,
        older_physical_increment=older_increment,
        older_timestep_seconds=older_dt,
        cumulative_actual_conserved_storage=ledger_arrays[0],
        cumulative_actual_vertical_storage=ledger_arrays[1],
        cumulative_boundary_transport=ledger_arrays[2],
        cumulative_endogenous_source=ledger_arrays[3],
        cumulative_stream_source=ledger_arrays[4],
        cumulative_closure_defect=ledger_arrays[5],
        elapsed_time=elapsed,
        dt_next=dt_next,
        next_order=int(restart.next_order),
        accepted_steps=int(restart.accepted_steps),
        accepted_bdf2_steps=int(restart.accepted_bdf2_steps),
        rejected_attempts=int(restart.rejected_attempts),
        audit_count=int(restart.audit_count),
        provenance=dict(restart.provenance),
        schema_version=1,
    )


def _adaptive_bdf2_restart_hash(
    context: CausalFiveFieldDAEContext,
    restart: CausalFiveFieldAdaptiveBDF2Restart,
) -> str:
    digest = hashlib.sha256()
    arrays = (
        context.grid.edges,
        restart.state_vector,
        restart.history.previous_physical_increment,
        restart.history.previous_vertical_killing_increment,
        restart.older_physical_increment,
        restart.cumulative_actual_conserved_storage,
        restart.cumulative_actual_vertical_storage,
        restart.cumulative_boundary_transport,
        restart.cumulative_endogenous_source,
        restart.cumulative_stream_source,
        restart.cumulative_closure_defect,
        np.asarray(
            (
                restart.history.previous_timestep_seconds,
                restart.older_timestep_seconds,
                restart.elapsed_time,
                restart.dt_next,
            ),
            dtype="<f8",
        ),
        np.asarray(
            (
                restart.next_order,
                restart.accepted_steps,
                restart.accepted_bdf2_steps,
                restart.rejected_attempts,
                restart.audit_count,
                restart.schema_version,
            ),
            dtype="<i8",
        ),
    )
    for values in arrays:
        array = np.ascontiguousarray(values)
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    digest.update(restart.history.temporal_height_scheme.encode("ascii"))
    return digest.hexdigest()


def causal_five_field_bdf_restarts_equal(
    left: CausalFiveFieldBDFRestart,
    right: CausalFiveFieldBDFRestart,
) -> bool:
    """Return whether two BDF restart payloads are bitwise identical."""

    return bool(
        np.array_equal(left.state_vector, right.state_vector)
        and np.array_equal(
            left.history.previous_physical_increment,
            right.history.previous_physical_increment,
        )
        and np.array_equal(
            left.history.previous_vertical_killing_increment,
            right.history.previous_vertical_killing_increment,
        )
        and left.history.previous_timestep_seconds
        == right.history.previous_timestep_seconds
        and left.history.temporal_height_scheme
        == right.history.temporal_height_scheme
        and left.elapsed_time == right.elapsed_time
        and left.dt_next == right.dt_next
        and left.next_order == right.next_order
        and left.accepted_steps == right.accepted_steps
        and left.rejected_attempts == right.rejected_attempts
        and left.provenance == right.provenance
        and left.schema_version == right.schema_version
    )


def causal_five_field_adaptive_bdf2_restarts_equal(
    left: CausalFiveFieldAdaptiveBDF2Restart,
    right: CausalFiveFieldAdaptiveBDF2Restart,
) -> bool:
    """Return whether two adaptive BDF2 restarts are bitwise identical."""

    array_pairs = (
        (left.state_vector, right.state_vector),
        (
            left.history.previous_physical_increment,
            right.history.previous_physical_increment,
        ),
        (
            left.history.previous_vertical_killing_increment,
            right.history.previous_vertical_killing_increment,
        ),
        (
            left.older_physical_increment,
            right.older_physical_increment,
        ),
        (
            left.cumulative_actual_conserved_storage,
            right.cumulative_actual_conserved_storage,
        ),
        (
            left.cumulative_actual_vertical_storage,
            right.cumulative_actual_vertical_storage,
        ),
        (
            left.cumulative_boundary_transport,
            right.cumulative_boundary_transport,
        ),
        (
            left.cumulative_endogenous_source,
            right.cumulative_endogenous_source,
        ),
        (
            left.cumulative_stream_source,
            right.cumulative_stream_source,
        ),
        (
            left.cumulative_closure_defect,
            right.cumulative_closure_defect,
        ),
    )
    return bool(
        all(np.array_equal(one, two) for one, two in array_pairs)
        and left.history.previous_timestep_seconds
        == right.history.previous_timestep_seconds
        and left.history.temporal_height_scheme
        == right.history.temporal_height_scheme
        and left.older_timestep_seconds == right.older_timestep_seconds
        and left.elapsed_time == right.elapsed_time
        and left.dt_next == right.dt_next
        and left.next_order == right.next_order
        and left.accepted_steps == right.accepted_steps
        and left.accepted_bdf2_steps == right.accepted_bdf2_steps
        and left.rejected_attempts == right.rejected_attempts
        and left.audit_count == right.audit_count
        and left.provenance == right.provenance
        and left.schema_version == right.schema_version
    )


def save_causal_five_field_bdf_restart(
    path: str | Path,
    context: CausalFiveFieldDAEContext,
    restart: CausalFiveFieldBDFRestart,
) -> None:
    """Persist one complete causal BDF restart."""

    validated = _validated_restart(context, restart)
    destination = Path(path)
    if destination.suffix != ".npz":
        raise ValueError("causal BDF restart path must end in .npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    provenance = json.dumps(
        validated.provenance,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    np.savez_compressed(
        destination,
        schema_version=np.asarray(
            validated.schema_version,
            dtype=np.int64,
        ),
        grid_edges=context.grid.edges,
        spatial_reconstruction=np.asarray(
            context.spatial_reconstruction
        ),
        state_vector=validated.state_vector,
        previous_physical_increment=(
            validated.history.previous_physical_increment
        ),
        previous_vertical_killing_increment=(
            validated.history.previous_vertical_killing_increment
        ),
        previous_timestep_seconds=np.asarray(
            validated.history.previous_timestep_seconds
        ),
        temporal_height_scheme=np.asarray(
            validated.history.temporal_height_scheme
        ),
        elapsed_time=np.asarray(validated.elapsed_time),
        dt_next=np.asarray(validated.dt_next),
        next_order=np.asarray(validated.next_order, dtype=np.int64),
        accepted_steps=np.asarray(
            validated.accepted_steps,
            dtype=np.int64,
        ),
        rejected_attempts=np.asarray(
            validated.rejected_attempts,
            dtype=np.int64,
        ),
        state_history_sha256=np.asarray(
            _restart_hash(context, validated)
        ),
        provenance_json=np.asarray(provenance),
    )


def load_causal_five_field_bdf_restart(
    path: str | Path,
    context: CausalFiveFieldDAEContext,
) -> CausalFiveFieldBDFRestart:
    """Load and checksum one complete causal BDF restart."""

    context = context.validated()
    with np.load(Path(path), allow_pickle=False) as data:
        edges = np.asarray(data["grid_edges"], dtype=float)
        if not np.array_equal(edges, context.grid.edges):
            raise ValueError("causal BDF restart grid does not match")
        provenance = json.loads(str(data["provenance_json"].item()))
        stored_reconstruction = (
            str(data["spatial_reconstruction"].item())
            if "spatial_reconstruction" in data.files
            else str(
                provenance.get(
                    "spatial_reconstruction",
                    "piecewise_constant",
                )
            )
        )
        if stored_reconstruction != context.spatial_reconstruction:
            raise ValueError(
                "causal BDF restart spatial reconstruction differs"
            )
        restart = CausalFiveFieldBDFRestart(
            state_vector=np.asarray(data["state_vector"], dtype=float),
            history=CausalFiveFieldBDFHistory(
                previous_physical_increment=np.asarray(
                    data["previous_physical_increment"],
                    dtype=float,
                ),
                previous_vertical_killing_increment=np.asarray(
                    data["previous_vertical_killing_increment"],
                    dtype=float,
                ),
                previous_timestep_seconds=float(
                    data["previous_timestep_seconds"]
                ),
                temporal_height_scheme=str(
                    data["temporal_height_scheme"].item()
                ),
            ),
            elapsed_time=float(data["elapsed_time"]),
            dt_next=float(data["dt_next"]),
            next_order=int(data["next_order"]),
            accepted_steps=int(data["accepted_steps"]),
            rejected_attempts=int(data["rejected_attempts"]),
            provenance=provenance,
            schema_version=int(data["schema_version"]),
        )
        expected_hash = str(data["state_history_sha256"].item())
    validated = _validated_restart(context, restart)
    if expected_hash != _restart_hash(context, validated):
        raise ValueError("causal BDF restart checksum mismatch")
    return validated


def save_causal_five_field_adaptive_bdf2_restart(
    path: str | Path,
    context: CausalFiveFieldDAEContext,
    restart: CausalFiveFieldAdaptiveBDF2Restart,
) -> None:
    """Persist one complete adaptive causal BDF2 restart."""

    validated = _validated_adaptive_bdf2_restart(context, restart)
    destination = Path(path)
    if destination.suffix != ".npz":
        raise ValueError("adaptive causal BDF2 restart must end in .npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    provenance = json.dumps(
        validated.provenance,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    np.savez_compressed(
        destination,
        schema_version=np.asarray(validated.schema_version, dtype=np.int64),
        grid_edges=context.grid.edges,
        spatial_reconstruction=np.asarray(
            context.spatial_reconstruction
        ),
        state_vector=validated.state_vector,
        previous_physical_increment=(
            validated.history.previous_physical_increment
        ),
        previous_vertical_killing_increment=(
            validated.history.previous_vertical_killing_increment
        ),
        previous_timestep_seconds=np.asarray(
            validated.history.previous_timestep_seconds
        ),
        temporal_height_scheme=np.asarray(
            validated.history.temporal_height_scheme
        ),
        older_physical_increment=validated.older_physical_increment,
        older_timestep_seconds=np.asarray(
            validated.older_timestep_seconds
        ),
        cumulative_actual_conserved_storage=(
            validated.cumulative_actual_conserved_storage
        ),
        cumulative_actual_vertical_storage=(
            validated.cumulative_actual_vertical_storage
        ),
        cumulative_boundary_transport=(
            validated.cumulative_boundary_transport
        ),
        cumulative_endogenous_source=(
            validated.cumulative_endogenous_source
        ),
        cumulative_stream_source=validated.cumulative_stream_source,
        cumulative_closure_defect=validated.cumulative_closure_defect,
        elapsed_time=np.asarray(validated.elapsed_time),
        dt_next=np.asarray(validated.dt_next),
        next_order=np.asarray(validated.next_order, dtype=np.int64),
        accepted_steps=np.asarray(
            validated.accepted_steps,
            dtype=np.int64,
        ),
        accepted_bdf2_steps=np.asarray(
            validated.accepted_bdf2_steps,
            dtype=np.int64,
        ),
        rejected_attempts=np.asarray(
            validated.rejected_attempts,
            dtype=np.int64,
        ),
        audit_count=np.asarray(validated.audit_count, dtype=np.int64),
        state_history_sha256=np.asarray(
            _adaptive_bdf2_restart_hash(context, validated)
        ),
        provenance_json=np.asarray(provenance),
    )


def load_causal_five_field_adaptive_bdf2_restart(
    path: str | Path,
    context: CausalFiveFieldDAEContext,
) -> CausalFiveFieldAdaptiveBDF2Restart:
    """Load and checksum one complete adaptive causal BDF2 restart."""

    context = context.validated()
    with np.load(Path(path), allow_pickle=False) as data:
        if not np.array_equal(
            np.asarray(data["grid_edges"], dtype=float),
            context.grid.edges,
        ):
            raise ValueError("adaptive causal BDF2 restart grid differs")
        provenance = json.loads(str(data["provenance_json"].item()))
        stored_reconstruction = (
            str(data["spatial_reconstruction"].item())
            if "spatial_reconstruction" in data.files
            else str(
                provenance.get(
                    "spatial_reconstruction",
                    "piecewise_constant",
                )
            )
        )
        if stored_reconstruction != context.spatial_reconstruction:
            raise ValueError(
                "adaptive BDF2 restart spatial reconstruction differs"
            )
        restart = CausalFiveFieldAdaptiveBDF2Restart(
            state_vector=np.asarray(data["state_vector"], dtype=float),
            history=CausalFiveFieldBDFHistory(
                previous_physical_increment=np.asarray(
                    data["previous_physical_increment"],
                    dtype=float,
                ),
                previous_vertical_killing_increment=np.asarray(
                    data["previous_vertical_killing_increment"],
                    dtype=float,
                ),
                previous_timestep_seconds=float(
                    data["previous_timestep_seconds"]
                ),
                temporal_height_scheme=str(
                    data["temporal_height_scheme"].item()
                ),
            ),
            older_physical_increment=np.asarray(
                data["older_physical_increment"],
                dtype=float,
            ),
            older_timestep_seconds=float(
                data["older_timestep_seconds"]
            ),
            cumulative_actual_conserved_storage=np.asarray(
                data["cumulative_actual_conserved_storage"],
                dtype=float,
            ),
            cumulative_actual_vertical_storage=np.asarray(
                data["cumulative_actual_vertical_storage"],
                dtype=float,
            ),
            cumulative_boundary_transport=np.asarray(
                data["cumulative_boundary_transport"],
                dtype=float,
            ),
            cumulative_endogenous_source=np.asarray(
                data["cumulative_endogenous_source"],
                dtype=float,
            ),
            cumulative_stream_source=np.asarray(
                data["cumulative_stream_source"],
                dtype=float,
            ),
            cumulative_closure_defect=np.asarray(
                data["cumulative_closure_defect"],
                dtype=float,
            ),
            elapsed_time=float(data["elapsed_time"]),
            dt_next=float(data["dt_next"]),
            next_order=int(data["next_order"]),
            accepted_steps=int(data["accepted_steps"]),
            accepted_bdf2_steps=int(data["accepted_bdf2_steps"]),
            rejected_attempts=int(data["rejected_attempts"]),
            audit_count=int(data["audit_count"]),
            provenance=provenance,
            schema_version=int(data["schema_version"]),
        )
        expected_hash = str(data["state_history_sha256"].item())
    validated = _validated_adaptive_bdf2_restart(context, restart)
    if expected_hash != _adaptive_bdf2_restart_hash(
        context,
        validated,
    ):
        raise ValueError("adaptive causal BDF2 restart checksum mismatch")
    return validated
