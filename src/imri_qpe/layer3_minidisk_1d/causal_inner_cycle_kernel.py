"""Production adapter for the structure-preserving reduced hybrid cycle.

The online state is ``(Q_1, ..., Q_4, phi)``.  Every smooth right-hand-side
query is obtained from the cycle driver, every accepted endpoint is checked
against the structure-preserving branch atlas, and every discrete transition
uses an oriented guard sheet and one atomic duration/phase/ledger reset.

The adapter has no truth-solver fallback.  A production construction fails
closed unless its metadata explicitly records complete physical payloads and
both independent heldout validations.  Synthetic fixtures must opt in with
``require_physical=False``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .causal_inner_cycle_atlas import (
    CycleBranchValue,
    GuardSheetLocation,
    interpolate_cycle_branch,
    interpolate_cycle_driver,
    interpolate_cycle_event,
    locate_guard_sheet,
)
from .causal_inner_reduced_hybrid_cycle import (
    ReducedEventReset,
    ReducedHybridCheckpoint,
    ReducedHybridIntegration,
    ReducedHybridTransition,
    integrate_reduced_hybrid,
)


Array = np.ndarray
TWO_PI = 2.0 * np.pi


def _finite(value, *, ndim: int, name: str, dtype=float) -> Array:
    result = np.asarray(value, dtype=dtype)
    if result.ndim != ndim or (dtype is not int and np.any(~np.isfinite(result))):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return result


def _relative(left, right) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    return float(
        np.linalg.norm(a - b)
        / max(np.linalg.norm(a), np.linalg.norm(b), np.finfo(float).tiny)
    )


def _unwrap_around(values: Array, center: float) -> Array:
    raw = np.asarray(values, dtype=float)
    return float(center) + (raw - float(center) + np.pi) % TWO_PI - np.pi


def require_production_cycle_metadata(metadata: Mapping[str, object]) -> None:
    """Reject incomplete, synthetic, unhashed, or unvalidated production data."""

    required_true = (
        "physical_model_complete",
        "physical_payload_hashes_complete",
        "heldout_physical_validation_complete",
        "independent_spatial_holdout_complete",
        "independent_sequence_or_cycle_holdout_complete",
    )
    missing = [name for name in required_true if metadata.get(name) is not True]
    if missing:
        raise ValueError(f"production cycle metadata is incomplete: {missing}")
    if metadata.get("synthetic_fixture") is not False:
        raise ValueError("synthetic or undeclared cycle data cannot be used in production")
    digest = str(metadata.get("physical_bundle_sha256", ""))
    if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest.lower()):
        raise ValueError("physical cycle bundle needs one canonical SHA-256 digest")


@dataclass(frozen=True)
class CycleKernelTransitionSpec:
    name: str
    transition_class_index: int
    source_mode_index: int
    destination_mode_index: int
    crossing_direction: int

    def __post_init__(self) -> None:
        if not self.name or self.transition_class_index < 0:
            raise ValueError("cycle transition name/class is invalid")
        if self.source_mode_index < 0 or self.destination_mode_index < 0:
            raise ValueError("cycle transition mode is invalid")
        if self.source_mode_index == self.destination_mode_index:
            raise ValueError("a hybrid transition must change mode")
        if self.crossing_direction not in (-1, 1):
            raise ValueError("crossing direction must be -1 or +1")


@dataclass
class CycleKernelCounters:
    driver_queries: int = 0
    branch_queries: int = 0
    guard_queries: int = 0
    reset_queries: int = 0

    def copy(self) -> "CycleKernelCounters":
        return CycleKernelCounters(
            self.driver_queries,
            self.branch_queries,
            self.guard_queries,
            self.reset_queries,
        )

    def difference(self, older: "CycleKernelCounters") -> "CycleKernelCounters":
        return CycleKernelCounters(
            self.driver_queries - older.driver_queries,
            self.branch_queries - older.branch_queries,
            self.guard_queries - older.guard_queries,
            self.reset_queries - older.reset_queries,
        )


@dataclass(frozen=True)
class CycleKernelEndpointAudit:
    time_seconds: float
    mode_index: int
    minimum_barycentric_weight: float
    invariant_relative_defect: float
    maximum_radial_symmetry_defect: float
    maximum_source_entropy_positive_eigenvalue: float
    minimum_source_nullity: int
    fast_spectral_gap_per_second: float
    inner_incoming_count: int
    outer_incoming_count: int


@dataclass(frozen=True)
class CycleKernelIntegration:
    reduced: ReducedHybridIntegration
    endpoint_audits: tuple[CycleKernelEndpointAudit, ...]
    query_counts: CycleKernelCounters
    reduced_ledger_relative_defect: float


@dataclass(frozen=True)
class CycleKernelCheckpoint:
    reduced: ReducedHybridCheckpoint
    physical_bundle_sha256: str
    kernel_contract_sha256: str


@dataclass
class CycleAtlasKernel:
    metadata: Mapping[str, object]
    driver: Mapping[str, object]
    branch: Mapping[str, object]
    events: Mapping[str, object]
    additions: Mapping[str, object]
    conservation_map: Array
    minimum_norm_normal: Array
    transition_specs: Sequence[CycleKernelTransitionSpec]
    require_physical: bool = True
    counters: CycleKernelCounters | None = None

    def __post_init__(self) -> None:
        if self.require_physical:
            require_production_cycle_metadata(self.metadata)
        elif self.metadata.get("synthetic_fixture") is not True:
            raise ValueError("nonproduction construction is reserved for declared fixtures")
        self.conservation_map = _finite(
            self.conservation_map, ndim=2, name="conservation map"
        )
        self.minimum_norm_normal = _finite(
            self.minimum_norm_normal, ndim=2, name="minimum-norm normal"
        )
        if self.conservation_map.shape != (4, 1232):
            raise ValueError("cycle conservation map must have shape (4,1232)")
        if self.minimum_norm_normal.shape != (1232, 4):
            raise ValueError("cycle minimum-norm normal must have shape (1232,4)")
        if _relative(
            self.conservation_map @ self.minimum_norm_normal, np.eye(4)
        ) > 2.0e-12:
            raise ValueError("cycle conservation normal is not a right inverse")
        self.transition_specs = tuple(self.transition_specs)
        if not self.transition_specs:
            raise ValueError("cycle kernel needs at least one transition")
        classes = [value.transition_class_index for value in self.transition_specs]
        if len(set(classes)) != len(classes):
            raise ValueError("cycle transition classes must be unique")
        count = len(np.asarray(self.events.get("pre_invariants4")))
        for name in ("integrated_phase_advance", "destination_guard_margin"):
            values = _finite(self.events.get(name), ndim=1, name=name)
            if values.shape != (count,) or np.any(values <= 0.0):
                raise ValueError(f"event field {name} must be positive per anchor")
        if self.counters is None:
            self.counters = CycleKernelCounters()

    @property
    def physical_bundle_sha256(self) -> str:
        return str(self.metadata.get("physical_bundle_sha256", "synthetic-fixture"))

    def driver_value(self, state5, mode_index: int):
        state = _finite(state5, ndim=1, name="reduced cycle state")
        if state.shape != (5,):
            raise ValueError("reduced cycle state must have dimension five")
        self.counters.driver_queries += 1
        return interpolate_cycle_driver(
            self.driver,
            q_simplices=self.additions["q_simplices"],
            q_scales=self.additions["q_scales"],
            query_invariants=state[:4],
            phase=float(state[4]),
            mode_index=int(mode_index),
            conservation_map=self.conservation_map,
        )

    def rhs(self, _time_seconds: float, state5, mode_index: int) -> Array:
        value = self.driver_value(state5, mode_index)
        qdot = self.conservation_map @ value.slow_forcing_per_second
        return np.concatenate((qdot, [value.phase_rate_per_second]))

    def branch_value(self, state5, mode_index: int) -> CycleBranchValue:
        state = _finite(state5, ndim=1, name="branch reduced state")
        if state.shape != (5,):
            raise ValueError("branch reduced state must have dimension five")
        self.counters.branch_queries += 1
        return interpolate_cycle_branch(
            self.branch,
            branch_simplices=self.additions["branch_simplices"],
            branch_simplex_modes=self.additions["branch_simplex_modes"],
            q_scales=self.additions["q_scales"],
            phase_scale=float(self.additions["phase_scale"]),
            query_invariants=state[:4],
            phase=float(state[4]),
            mode_index=int(mode_index),
            conservation_map=self.conservation_map,
            minimum_norm_normal=self.minimum_norm_normal,
        )

    def endpoint_audit(
        self, time_seconds: float, state5, mode_index: int
    ) -> CycleKernelEndpointAudit:
        value = self.branch_value(state5, mode_index)
        return CycleKernelEndpointAudit(
            float(time_seconds),
            int(mode_index),
            value.location.minimum_weight,
            value.invariant_relative_defect,
            value.maximum_radial_symmetry_defect,
            value.maximum_source_entropy_positive_eigenvalue,
            value.minimum_source_nullity,
            value.fast_spectral_gap_per_second,
            value.inner_incoming_count,
            value.outer_incoming_count,
        )

    def guard_location(self, state5, transition_class: int) -> GuardSheetLocation:
        state = _finite(state5, ndim=1, name="guard reduced state")
        if state.shape != (5,):
            raise ValueError("guard reduced state must have dimension five")
        self.counters.guard_queries += 1
        q_scales = _finite(self.additions["q_scales"], ndim=1, name="q scales")
        phase_scale = float(self.additions["phase_scale"])
        center = float(state[4]) % TWO_PI
        event_q = _finite(
            self.events["pre_invariants4"], ndim=2, name="event invariants"
        )
        event_phase = _finite(self.events["phase"], ndim=1, name="event phase")
        nodes = np.column_stack(
            (event_q / q_scales, _unwrap_around(event_phase, center) / phase_scale)
        )
        query = np.concatenate((state[:4] / q_scales, [center / phase_scale]))
        return locate_guard_sheet(
            nodes,
            self.additions["event_simplices"],
            self.events["reduced_guard_normals5"],
            query,
            simplex_classes=self.additions["event_simplex_classes"],
            transition_class=int(transition_class),
        )

    def guard_value(self, state5, transition_class: int) -> float:
        return self.guard_location(state5, transition_class).signed_guard_distance

    def event_reset(
        self, _entry_time_seconds: float, state5, spec: CycleKernelTransitionSpec
    ) -> ReducedEventReset:
        state = _finite(state5, ndim=1, name="event entry state")
        if state.shape != (5,):
            raise ValueError("event entry state must have dimension five")
        self.counters.reset_queries += 1
        branch = self.branch_value(state, spec.source_mode_index)
        derivative = self.rhs(0.0, state, spec.source_mode_index)
        q_scales = np.asarray(self.additions["q_scales"], dtype=float)
        phase_scale = float(self.additions["phase_scale"])
        flow_scaled = np.concatenate(
            (derivative[:4] / q_scales, [derivative[4] / phase_scale])
        )
        event = interpolate_cycle_event(
            self.events,
            event_simplices=self.additions["event_simplices"],
            event_simplex_classes=self.additions["event_simplex_classes"],
            q_scales=q_scales,
            phase_scale=phase_scale,
            query_invariants=state[:4],
            phase=float(state[4]),
            transition_class=spec.transition_class_index,
            reduced_flow_scaled=flow_scaled,
            pre_state=branch.state,
            conservation_map=self.conservation_map,
            minimum_norm_normal=self.minimum_norm_normal,
        )
        if (
            event.source_mode_index != spec.source_mode_index
            or event.destination_mode_index != spec.destination_mode_index
        ):
            raise ValueError("interpolated event modes disagree with transition spec")
        indices = event.guard.vertex_indices
        weights = event.guard.weights
        phase_advance = float(
            weights @ np.asarray(self.events["integrated_phase_advance"])[indices]
        )
        destination_margin = float(
            weights @ np.asarray(self.events["destination_guard_margin"])[indices]
        )
        if phase_advance <= 0.0 or destination_margin <= 0.0:
            raise ValueError("interpolated finite event reset is not forward admissible")
        return ReducedEventReset(
            np.asarray(event.ledger_impulse, dtype=float),
            event.duration_seconds,
            phase_advance,
            destination_margin,
        )

    def transitions(self) -> tuple[ReducedHybridTransition, ...]:
        result = []
        for spec in self.transition_specs:
            result.append(
                ReducedHybridTransition(
                    spec.name,
                    spec.source_mode_index,
                    spec.destination_mode_index,
                    spec.crossing_direction,
                    lambda state, value=spec: self.guard_value(
                        state, value.transition_class_index
                    ),
                    lambda time_seconds, state, value=spec: self.event_reset(
                        time_seconds, state, value
                    ),
                )
            )
        return tuple(result)


def integrate_cycle_kernel(
    kernel: CycleAtlasKernel,
    checkpoint: ReducedHybridCheckpoint,
    *,
    end_time_seconds: float,
    absolute_tolerance,
    relative_tolerance: float,
    maximum_accepted_steps: int = 100000,
) -> CycleKernelIntegration:
    """Integrate and re-audit every accepted endpoint against the branch atlas."""

    if not isinstance(kernel, CycleAtlasKernel):
        raise TypeError("kernel must be CycleAtlasKernel")
    before = kernel.counters.copy()
    initial = np.asarray(checkpoint.state5, dtype=float)
    reduced = integrate_reduced_hybrid(
        kernel.rhs,
        checkpoint,
        end_time_seconds=end_time_seconds,
        transitions=kernel.transitions(),
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        maximum_accepted_steps=maximum_accepted_steps,
    )
    audits = tuple(
        kernel.endpoint_audit(value.time_seconds, value.state5, value.mode_index)
        for value in reduced.accepted_checkpoints
    )
    if len(audits) != len(reduced.accepted_checkpoints):
        raise RuntimeError("not every accepted cycle endpoint was audited")
    realized = reduced.checkpoint.state5[:4] - initial[:4]
    ledger = (
        reduced.checkpoint.cumulative_smooth_ledger4
        - checkpoint.cumulative_smooth_ledger4
        + reduced.checkpoint.cumulative_event_ledger4
        - checkpoint.cumulative_event_ledger4
    )
    ledger_defect = _relative(realized, ledger)
    if ledger_defect > 2.0e-12:
        raise RuntimeError("reduced cycle ledger does not close")
    return CycleKernelIntegration(
        reduced, audits, kernel.counters.difference(before), ledger_defect
    )


def save_cycle_kernel_checkpoint(
    checkpoint: CycleKernelCheckpoint, path: str | Path
) -> None:
    if not isinstance(checkpoint, CycleKernelCheckpoint):
        raise TypeError("checkpoint must be CycleKernelCheckpoint")
    value = checkpoint.reduced
    np.savez_compressed(
        Path(path),
        state5=np.asarray(value.state5),
        time_seconds=np.asarray(value.time_seconds),
        mode_index=np.asarray(value.mode_index, dtype=np.int64),
        next_timestep_seconds=np.asarray(value.next_timestep_seconds),
        cumulative_smooth_ledger4=np.asarray(value.cumulative_smooth_ledger4),
        cumulative_event_ledger4=np.asarray(value.cumulative_event_ledger4),
        accepted_steps=np.asarray(value.accepted_steps, dtype=np.int64),
        rejected_steps=np.asarray(value.rejected_steps, dtype=np.int64),
        completed_events=np.asarray(value.completed_events, dtype=np.int64),
        physical_bundle_sha256=np.asarray(checkpoint.physical_bundle_sha256),
        kernel_contract_sha256=np.asarray(checkpoint.kernel_contract_sha256),
    )


def load_cycle_kernel_checkpoint(
    path: str | Path,
    *,
    expected_physical_bundle_sha256: str,
    expected_kernel_contract_sha256: str,
) -> CycleKernelCheckpoint:
    with np.load(Path(path), allow_pickle=False) as payload:
        physical = str(payload["physical_bundle_sha256"].item())
        contract = str(payload["kernel_contract_sha256"].item())
        if physical != str(expected_physical_bundle_sha256):
            raise ValueError("cycle checkpoint physical bundle hash changed")
        if contract != str(expected_kernel_contract_sha256):
            raise ValueError("cycle checkpoint kernel contract hash changed")
        reduced = ReducedHybridCheckpoint(
            np.array(payload["state5"], copy=True),
            float(payload["time_seconds"]),
            int(payload["mode_index"]),
            float(payload["next_timestep_seconds"]),
            np.array(payload["cumulative_smooth_ledger4"], copy=True),
            np.array(payload["cumulative_event_ledger4"], copy=True),
            int(payload["accepted_steps"]),
            int(payload["rejected_steps"]),
            int(payload["completed_events"]),
        )
    return CycleKernelCheckpoint(reduced, physical, contract)


__all__ = [
    "CycleAtlasKernel",
    "CycleKernelCheckpoint",
    "CycleKernelCounters",
    "CycleKernelEndpointAudit",
    "CycleKernelIntegration",
    "CycleKernelTransitionSpec",
    "integrate_cycle_kernel",
    "load_cycle_kernel_checkpoint",
    "require_production_cycle_metadata",
    "save_cycle_kernel_checkpoint",
]
