"""Event-aware reduced integration for the slow hybrid cycle architecture."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


Array = np.ndarray
ReducedRHS = Callable[[float, Array, int], Array]
Guard = Callable[[Array], float]
Reset = Callable[[float, Array], "ReducedEventReset"]


def _finite(value, *, ndim: int, name: str) -> Array:
    result = np.asarray(value, dtype=float)
    if result.ndim != ndim or np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return result


def _relative(left, right) -> float:
    a = np.asarray(left, dtype=float); b = np.asarray(right, dtype=float)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), np.linalg.norm(b), np.finfo(float).tiny))


@dataclass(frozen=True)
class DormandPrinceStep:
    state_fifth: Array
    state_fourth: Array
    error: Array
    derivative_start: Array
    derivative_end: Array
    stage_states: Array
    stage_derivatives: Array


@dataclass(frozen=True)
class ReducedEventReset:
    ledger_impulse4: Array
    duration_seconds: float
    phase_advance: float
    destination_guard_margin: float


@dataclass(frozen=True)
class ReducedHybridTransition:
    name: str
    source_mode_index: int
    destination_mode_index: int
    crossing_direction: int
    guard: Guard
    reset: Reset


@dataclass(frozen=True)
class ReducedHybridEventRecord:
    name: str
    entry_time_seconds: float
    exit_time_seconds: float
    entry_state5: Array
    exit_state5: Array
    source_mode_index: int
    destination_mode_index: int
    ledger_impulse4: Array
    phase_advance: float
    localized_guard_value: float


@dataclass(frozen=True)
class ReducedHybridCheckpoint:
    state5: Array
    time_seconds: float
    mode_index: int
    next_timestep_seconds: float
    cumulative_smooth_ledger4: Array
    cumulative_event_ledger4: Array
    accepted_steps: int
    rejected_steps: int
    completed_events: int


@dataclass(frozen=True)
class ReducedHybridIntegration:
    checkpoint: ReducedHybridCheckpoint
    events: tuple[ReducedHybridEventRecord, ...]
    accepted_checkpoints: tuple[ReducedHybridCheckpoint, ...]
    maximum_scaled_error: float
    maximum_smooth_ledger_defect: float


@dataclass(frozen=True)
class HeldoutAtlasSequenceAudit:
    maximum_branch_state_relative_defect: float
    maximum_branch_rate_relative_defect: float
    maximum_port_action_relative_defect: float
    maximum_event_time_relative_defect: float
    maximum_event_post_state_relative_defect: float
    maximum_event_ledger_relative_defect: float
    sequence_endpoint_relative_defect: float
    sequence_ledger_relative_defect: float
    discrete_modes_and_event_order_exact: bool
    all_structure_gates_passed: bool
    restart_suffix_replay_bitwise: bool

    @property
    def passed(self) -> bool:
        return bool(
            self.maximum_branch_state_relative_defect <= 2.0e-2
            and self.maximum_branch_rate_relative_defect <= 5.0e-2
            and self.maximum_port_action_relative_defect <= 5.0e-2
            and self.maximum_event_time_relative_defect <= 2.0e-2
            and self.maximum_event_post_state_relative_defect <= 5.0e-2
            and self.maximum_event_ledger_relative_defect <= 2.0e-2
            and self.sequence_endpoint_relative_defect <= 5.0e-2
            and self.sequence_ledger_relative_defect <= 2.0e-2
            and self.discrete_modes_and_event_order_exact
            and self.all_structure_gates_passed
            and self.restart_suffix_replay_bitwise
        )


def dormand_prince_step(
    rhs: ReducedRHS,
    time_seconds: float,
    state,
    timestep_seconds: float,
    mode_index: int,
) -> DormandPrinceStep:
    y = _finite(state, ndim=1, name="reduced state")
    t = float(time_seconds); h = float(timestep_seconds); mode = int(mode_index)
    if y.shape != (5,) or not np.isfinite(t) or not np.isfinite(h) or h <= 0.0:
        raise ValueError("Dormand-Prince input is invalid")

    def evaluate(stage_time, stage_state):
        value = _finite(rhs(float(stage_time), np.asarray(stage_state), mode), ndim=1, name="reduced derivative")
        if value.shape != (5,) or value[4] <= 0.0:
            raise ValueError("reduced derivative must have dimension five and positive phase rate")
        return value

    k1 = evaluate(t, y)
    y2 = y + h * ((1.0 / 5.0) * k1); k2 = evaluate(t + h * 1.0 / 5.0, y2)
    y3 = y + h * ((3.0 / 40.0) * k1 + (9.0 / 40.0) * k2); k3 = evaluate(t + h * 3.0 / 10.0, y3)
    y4 = y + h * ((44.0 / 45.0) * k1 - (56.0 / 15.0) * k2 + (32.0 / 9.0) * k3); k4 = evaluate(t + h * 4.0 / 5.0, y4)
    y5s = y + h * ((19372.0 / 6561.0) * k1 - (25360.0 / 2187.0) * k2 + (64448.0 / 6561.0) * k3 - (212.0 / 729.0) * k4); k5 = evaluate(t + h * 8.0 / 9.0, y5s)
    y6 = y + h * ((9017.0 / 3168.0) * k1 - (355.0 / 33.0) * k2 + (46732.0 / 5247.0) * k3 + (49.0 / 176.0) * k4 - (5103.0 / 18656.0) * k5); k6 = evaluate(t + h, y6)
    fifth = y + h * ((35.0 / 384.0) * k1 + (500.0 / 1113.0) * k3 + (125.0 / 192.0) * k4 - (2187.0 / 6784.0) * k5 + (11.0 / 84.0) * k6)
    k7 = evaluate(t + h, fifth)
    fourth = y + h * ((5179.0 / 57600.0) * k1 + (7571.0 / 16695.0) * k3 + (393.0 / 640.0) * k4 - (92097.0 / 339200.0) * k5 + (187.0 / 2100.0) * k6 + (1.0 / 40.0) * k7)
    return DormandPrinceStep(fifth, fourth, fifth - fourth, k1, k7, np.asarray((y, y2, y3, y4, y5s, y6, fifth)), np.asarray((k1, k2, k3, k4, k5, k6, k7)))


def hermite_dense_state(old, new, derivative_old, derivative_new, timestep: float, fraction: float) -> Array:
    y0 = _finite(old, ndim=1, name="dense old state"); y1 = _finite(new, ndim=1, name="dense new state"); f0 = _finite(derivative_old, ndim=1, name="dense old derivative"); f1 = _finite(derivative_new, ndim=1, name="dense new derivative")
    h = float(timestep); theta = float(fraction)
    if y0.shape != y1.shape or y0.shape != f0.shape or y0.shape != f1.shape or h <= 0.0 or theta < 0.0 or theta > 1.0: raise ValueError("dense-output inputs disagree")
    h00 = 2.0 * theta**3 - 3.0 * theta**2 + 1.0; h10 = theta**3 - 2.0 * theta**2 + theta; h01 = -2.0 * theta**3 + 3.0 * theta**2; h11 = theta**3 - theta**2
    return h00 * y0 + h10 * h * f0 + h01 * y1 + h11 * h * f1


def _localized_crossing(transition: ReducedHybridTransition, old: Array, step: DormandPrinceStep, timestep: float) -> tuple[float, Array, float]:
    left = 0.0; right = 1.0; g_left = float(transition.guard(old)); g_right = float(transition.guard(step.state_fifth))
    if transition.crossing_direction > 0 and not (g_left < 0.0 <= g_right): raise ValueError("positive crossing is not bracketed")
    if transition.crossing_direction < 0 and not (g_left > 0.0 >= g_right): raise ValueError("negative crossing is not bracketed")
    for _ in range(80):
        middle = 0.5 * (left + right); state = hermite_dense_state(old, step.state_fifth, step.derivative_start, step.derivative_end, timestep, middle); value = float(transition.guard(state))
        if abs(value) <= 1.0e-13 or right - left <= 1.0e-10: return middle, state, value
        same_as_left = (value < 0.0) == (g_left < 0.0)
        if same_as_left: left = middle; g_left = value
        else: right = middle; g_right = value
    middle = 0.5 * (left + right); state = hermite_dense_state(old, step.state_fifth, step.derivative_start, step.derivative_end, timestep, middle); return middle, state, float(transition.guard(state))


def _scaled_error(error: Array, old: Array, new: Array, absolute_tolerance: Array, relative_tolerance: float) -> float:
    scale = absolute_tolerance + float(relative_tolerance) * np.maximum(np.abs(old), np.abs(new))
    return float(np.sqrt(np.mean((error / scale) ** 2)))


def integrate_reduced_hybrid(
    rhs: ReducedRHS,
    checkpoint: ReducedHybridCheckpoint,
    *,
    end_time_seconds: float,
    transitions: Sequence[ReducedHybridTransition],
    absolute_tolerance,
    relative_tolerance: float,
    maximum_accepted_steps: int = 100000,
) -> ReducedHybridIntegration:
    if not isinstance(checkpoint, ReducedHybridCheckpoint): raise TypeError("checkpoint must be ReducedHybridCheckpoint")
    state = _finite(checkpoint.state5, ndim=1, name="checkpoint state").copy(); atol = _finite(absolute_tolerance, ndim=1, name="absolute tolerance")
    if state.shape != (5,) or atol.shape != (5,) or np.any(atol <= 0.0): raise ValueError("reduced checkpoint/tolerances have wrong dimensions")
    time = float(checkpoint.time_seconds); end = float(end_time_seconds); mode = int(checkpoint.mode_index); next_step = float(checkpoint.next_timestep_seconds); rtol = float(relative_tolerance)
    if end <= time or next_step <= 0.0 or rtol <= 0.0: raise ValueError("integration interval and tolerances must be positive")
    smooth = np.asarray(checkpoint.cumulative_smooth_ledger4, dtype=float).copy(); event_ledger = np.asarray(checkpoint.cumulative_event_ledger4, dtype=float).copy(); accepted = int(checkpoint.accepted_steps); rejected = int(checkpoint.rejected_steps); completed_events = int(checkpoint.completed_events)
    records = []; checkpoints = []; maximum_error = 0.0; maximum_ledger_defect = 0.0
    while time < end:
        if accepted - checkpoint.accepted_steps >= int(maximum_accepted_steps): raise RuntimeError("reduced integration exhausted its accepted-step budget")
        h = min(next_step, end - time); old = state.copy(); old_time = time
        trial = dormand_prince_step(rhs, time, old, h, mode); error = _scaled_error(trial.error, old, trial.state_fifth, atol, rtol); maximum_error = max(maximum_error, error)
        if error > 1.0:
            rejected += 1; next_step = h * max(0.2, min(0.9 * error ** (-0.2), 0.8)); continue
        eligible = [transition for transition in transitions if transition.source_mode_index == mode]
        crossed = []
        for transition in eligible:
            g0 = float(transition.guard(old)); g1 = float(transition.guard(trial.state_fifth))
            if (transition.crossing_direction > 0 and g0 < 0.0 <= g1) or (transition.crossing_direction < 0 and g0 > 0.0 >= g1): crossed.append(transition)
        if len(crossed) > 1: raise RuntimeError("one reduced step contains multiple unresolved event classes")
        if crossed:
            transition = crossed[0]; fraction, entry, guard_value = _localized_crossing(transition, old, trial, h); entry_time = old_time + fraction * h; reset = transition.reset(entry_time, entry)
            impulse = _finite(reset.ledger_impulse4, ndim=1, name="event ledger impulse")
            if impulse.shape != (4,) or reset.duration_seconds < 0.0 or reset.phase_advance < 0.0 or reset.destination_guard_margin <= 0.0 or transition.destination_mode_index < 0: raise ValueError("event reset violates the hybrid contract")
            exit_state = entry.copy(); exit_state[:4] += impulse; exit_state[4] += float(reset.phase_advance); exit_time = entry_time + float(reset.duration_seconds)
            if exit_time > end + 1.0e-14: raise RuntimeError("compressed event exits beyond the requested integration interval")
            smooth_increment = entry[:4] - old[:4]; smooth += smooth_increment; event_ledger += impulse
            ledger_defect = _relative(smooth_increment, entry[:4] - old[:4]); maximum_ledger_defect = max(maximum_ledger_defect, ledger_defect)
            records.append(ReducedHybridEventRecord(transition.name, entry_time, exit_time, entry, exit_state, mode, transition.destination_mode_index, impulse, float(reset.phase_advance), guard_value))
            state = exit_state; time = exit_time; mode = transition.destination_mode_index; completed_events += 1; accepted += 1
        else:
            state = trial.state_fifth; time += h; smooth_increment = state[:4] - old[:4]; smooth += smooth_increment; maximum_ledger_defect = max(maximum_ledger_defect, _relative(smooth_increment, state[:4] - old[:4])); accepted += 1
        factor = 2.0 if error == 0.0 else max(0.2, min(2.0, 0.9 * error ** (-0.2))); next_step = max(np.finfo(float).eps, h * factor)
        current = ReducedHybridCheckpoint(state.copy(), time, mode, next_step, smooth.copy(), event_ledger.copy(), accepted, rejected, completed_events); checkpoints.append(current)
    final = ReducedHybridCheckpoint(state.copy(), time, mode, next_step, smooth.copy(), event_ledger.copy(), accepted, rejected, completed_events)
    return ReducedHybridIntegration(final, tuple(records), tuple(checkpoints), maximum_error, maximum_ledger_defect)


def integrate_fixed_dopri5(rhs: ReducedRHS, state, *, start_time: float, end_time: float, step_count: int, mode_index: int) -> Array:
    value = _finite(state, ndim=1, name="fixed-step state").copy(); count = int(step_count); start = float(start_time); end = float(end_time)
    if value.shape != (5,) or count < 1 or end <= start: raise ValueError("fixed-step integration inputs are invalid")
    h = (end - start) / count; time = start
    for _ in range(count): value = dormand_prince_step(rhs, time, value, h, mode_index).state_fifth; time += h
    return value


def validate_heldout_atlas_and_sequence(*, branch_predicted_states, branch_truth_states, branch_predicted_rates, branch_truth_rates, predicted_port_actions, truth_port_actions, predicted_event_times, truth_event_times, predicted_event_post_states, truth_event_post_states, predicted_event_ledgers, truth_event_ledgers, predicted_sequence_endpoint, truth_sequence_endpoint, predicted_sequence_ledger, truth_sequence_ledger, predicted_mode_sequence, truth_mode_sequence, all_structure_gates_passed: bool, restart_suffix_replay_bitwise: bool) -> HeldoutAtlasSequenceAudit:
    def maximum_rows(left, right):
        a = _finite(left, ndim=2, name="heldout prediction"); b = _finite(right, ndim=2, name="heldout truth")
        if a.shape != b.shape: raise ValueError("heldout prediction/truth shapes disagree")
        return max(_relative(x, y) for x, y in zip(a, b, strict=True))
    event_time = _finite(predicted_event_times, ndim=1, name="event times"); truth_time = _finite(truth_event_times, ndim=1, name="truth event times")
    if event_time.shape != truth_time.shape: raise ValueError("event-time shapes disagree")
    event_time_defect = max(abs(float(a - b)) / max(abs(float(b)), np.finfo(float).tiny) for a, b in zip(event_time, truth_time, strict=True))
    return HeldoutAtlasSequenceAudit(maximum_rows(branch_predicted_states, branch_truth_states), maximum_rows(branch_predicted_rates, branch_truth_rates), maximum_rows(predicted_port_actions, truth_port_actions), float(event_time_defect), maximum_rows(predicted_event_post_states, truth_event_post_states), maximum_rows(predicted_event_ledgers, truth_event_ledgers), _relative(predicted_sequence_endpoint, truth_sequence_endpoint), _relative(predicted_sequence_ledger, truth_sequence_ledger), tuple(predicted_mode_sequence) == tuple(truth_mode_sequence), bool(all_structure_gates_passed), bool(restart_suffix_replay_bitwise))


def save_reduced_hybrid_checkpoint(checkpoint: ReducedHybridCheckpoint, path: str | Path) -> None:
    if not isinstance(checkpoint, ReducedHybridCheckpoint): raise TypeError("checkpoint must be ReducedHybridCheckpoint")
    np.savez_compressed(Path(path), state5=np.asarray(checkpoint.state5), time_seconds=np.asarray(checkpoint.time_seconds), mode_index=np.asarray(checkpoint.mode_index, dtype=np.int64), next_timestep_seconds=np.asarray(checkpoint.next_timestep_seconds), cumulative_smooth_ledger4=np.asarray(checkpoint.cumulative_smooth_ledger4), cumulative_event_ledger4=np.asarray(checkpoint.cumulative_event_ledger4), accepted_steps=np.asarray(checkpoint.accepted_steps, dtype=np.int64), rejected_steps=np.asarray(checkpoint.rejected_steps, dtype=np.int64), completed_events=np.asarray(checkpoint.completed_events, dtype=np.int64))


def load_reduced_hybrid_checkpoint(path: str | Path) -> ReducedHybridCheckpoint:
    with np.load(Path(path), allow_pickle=False) as payload: return ReducedHybridCheckpoint(np.array(payload["state5"], copy=True), float(payload["time_seconds"]), int(payload["mode_index"]), float(payload["next_timestep_seconds"]), np.array(payload["cumulative_smooth_ledger4"], copy=True), np.array(payload["cumulative_event_ledger4"], copy=True), int(payload["accepted_steps"]), int(payload["rejected_steps"]), int(payload["completed_events"]))


__all__ = ["DormandPrinceStep", "HeldoutAtlasSequenceAudit", "ReducedEventReset", "ReducedHybridCheckpoint", "ReducedHybridEventRecord", "ReducedHybridIntegration", "ReducedHybridTransition", "dormand_prince_step", "hermite_dense_state", "integrate_fixed_dopri5", "integrate_reduced_hybrid", "load_reduced_hybrid_checkpoint", "save_reduced_hybrid_checkpoint", "validate_heldout_atlas_and_sequence"]
