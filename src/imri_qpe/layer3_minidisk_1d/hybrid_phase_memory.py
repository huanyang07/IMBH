"""Truth-free conservative hybrid phase-memory online integrator.

The full 470-coordinate state is decoded from an exactly retained macro
ledger and a low-rank, mode-local phase table.  This module deliberately has
no dependency on the monolithic truth residual or nonlinear solvers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


def _array(value, *, ndim: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != ndim or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return result.copy()


@dataclass(frozen=True)
class ConservativePhaseMode:
    """One monotone phase interval with an exact conservative macro ledger."""

    name: str
    phase_knots: np.ndarray
    phase_speeds_per_second: np.ndarray
    macro_ledger_knots: np.ndarray
    hidden_coefficient_knots: np.ndarray
    hidden_origin: np.ndarray
    hidden_embedding_basis: np.ndarray
    macro_lift: np.ndarray
    hidden_lift: np.ndarray
    macro_restriction: np.ndarray

    def __post_init__(self) -> None:
        phase = _array(self.phase_knots, ndim=1, name="phase_knots")
        speeds = _array(
            self.phase_speeds_per_second,
            ndim=1,
            name="phase_speeds_per_second",
        )
        ledger = _array(self.macro_ledger_knots, ndim=2, name="macro_ledger_knots")
        coefficients = _array(
            self.hidden_coefficient_knots,
            ndim=2,
            name="hidden_coefficient_knots",
        )
        hidden_origin = _array(self.hidden_origin, ndim=1, name="hidden_origin")
        hidden_basis = _array(
            self.hidden_embedding_basis,
            ndim=2,
            name="hidden_embedding_basis",
        )
        macro_lift = _array(self.macro_lift, ndim=2, name="macro_lift")
        hidden_lift = _array(self.hidden_lift, ndim=2, name="hidden_lift")
        restriction = _array(
            self.macro_restriction, ndim=2, name="macro_restriction"
        )
        count = len(phase)
        if not self.name:
            raise ValueError("mode name must be nonempty")
        if count < 2 or phase[0] != 0.0 or phase[-1] != 1.0:
            raise ValueError("phase knots must span exactly [0, 1]")
        if np.any(np.diff(phase) <= 0.0):
            raise ValueError("phase knots must increase strictly")
        if speeds.shape != (count - 1,) or np.any(speeds <= 0.0):
            raise ValueError("one positive phase speed is required per interval")
        if ledger.shape[0] != count or coefficients.shape[0] != count:
            raise ValueError("phase tables must share the knot count")
        if hidden_basis.shape != (len(hidden_origin), coefficients.shape[1]):
            raise ValueError("hidden basis and coefficient dimensions disagree")
        coordinate_dimension, macro_dimension = macro_lift.shape
        if ledger.shape[1] != macro_dimension:
            raise ValueError("macro ledger and lift dimensions disagree")
        if hidden_lift.shape != (coordinate_dimension, len(hidden_origin)):
            raise ValueError("hidden lift dimensions disagree")
        if restriction.shape != (macro_dimension, coordinate_dimension):
            raise ValueError("macro restriction dimensions disagree")
        identity_defect = np.linalg.norm(
            restriction @ macro_lift - np.eye(macro_dimension), ord=np.inf
        )
        kernel_defect = np.linalg.norm(restriction @ hidden_lift, ord=np.inf)
        if max(identity_defect, kernel_defect) > 5.0e-12:
            raise ValueError("decoder does not preserve the macro ledger")
        for field, value in (
            ("phase_knots", phase),
            ("phase_speeds_per_second", speeds),
            ("macro_ledger_knots", ledger),
            ("hidden_coefficient_knots", coefficients),
            ("hidden_origin", hidden_origin),
            ("hidden_embedding_basis", hidden_basis),
            ("macro_lift", macro_lift),
            ("hidden_lift", hidden_lift),
            ("macro_restriction", restriction),
        ):
            value.setflags(write=False)
            object.__setattr__(self, field, value)

    @property
    def macro_dimension(self) -> int:
        return int(self.macro_lift.shape[1])

    @property
    def coordinate_dimension(self) -> int:
        return int(self.macro_lift.shape[0])

    @property
    def duration_seconds(self) -> float:
        return float(
            np.sum(np.diff(self.phase_knots) / self.phase_speeds_per_second)
        )

    def _interval(self, phase: float) -> tuple[int, float]:
        value = float(phase)
        if not 0.0 <= value <= 1.0:
            raise ValueError("phase must lie in [0, 1]")
        index = min(
            int(np.searchsorted(self.phase_knots, value, side="right") - 1),
            len(self.phase_knots) - 2,
        )
        left = self.phase_knots[index]
        right = self.phase_knots[index + 1]
        weight = float((value - left) / (right - left))
        return index, min(max(weight, 0.0), 1.0)

    def _interpolate(self, table: np.ndarray, phase: float) -> np.ndarray:
        index, weight = self._interval(phase)
        return (1.0 - weight) * table[index] + weight * table[index + 1]

    def ledger(self, phase: float) -> np.ndarray:
        return self._interpolate(self.macro_ledger_knots, phase)

    def hidden_coordinates(self, phase: float) -> np.ndarray:
        coefficients = self._interpolate(self.hidden_coefficient_knots, phase)
        return self.hidden_origin + self.hidden_embedding_basis @ coefficients

    def decode(self, macro_state: np.ndarray, phase: float) -> np.ndarray:
        macro = _array(macro_state, ndim=1, name="macro_state")
        if macro.shape != (self.macro_dimension,):
            raise ValueError("macro state dimension disagrees with mode")
        return self.macro_lift @ macro + self.hidden_lift @ self.hidden_coordinates(phase)

    def phase_speed(self, phase: float) -> float:
        index, _ = self._interval(phase)
        return float(self.phase_speeds_per_second[index])

    def advance_phase(self, phase: float, duration_seconds: float) -> tuple[float, float, bool]:
        """Advance exactly across piecewise-constant phase-speed intervals."""

        current = float(phase)
        remaining = float(duration_seconds)
        if remaining < 0.0:
            raise ValueError("duration must be nonnegative")
        used = 0.0
        tolerance = 8.0 * np.finfo(float).eps
        while remaining > 0.0 and current < 1.0 - tolerance:
            index, _ = self._interval(current)
            speed = float(self.phase_speeds_per_second[index])
            boundary = float(self.phase_knots[index + 1])
            interval_time = max((boundary - current) / speed, 0.0)
            if remaining < interval_time:
                current += speed * remaining
                used += remaining
                remaining = 0.0
            else:
                current = boundary
                used += interval_time
                remaining -= interval_time
        reached = current >= 1.0 - tolerance
        return (1.0 if reached else current), used, reached


@dataclass(frozen=True)
class HybridPhaseState:
    macro_state: np.ndarray
    phase: float
    mode: str
    elapsed_seconds: float = 0.0
    event_count: int = 0

    def __post_init__(self) -> None:
        macro = _array(self.macro_state, ndim=1, name="macro_state")
        macro.setflags(write=False)
        object.__setattr__(self, "macro_state", macro)
        if not 0.0 <= float(self.phase) <= 1.0:
            raise ValueError("phase must lie in [0, 1]")
        if float(self.elapsed_seconds) < 0.0 or int(self.event_count) < 0:
            raise ValueError("elapsed time and event count must be nonnegative")

    def to_payload(self) -> dict:
        return {
            "macro_state": self.macro_state.tolist(),
            "phase": float(self.phase),
            "mode": self.mode,
            "elapsed_seconds": float(self.elapsed_seconds),
            "event_count": int(self.event_count),
        }

    @classmethod
    def from_payload(cls, payload: Mapping) -> "HybridPhaseState":
        return cls(
            macro_state=np.asarray(payload["macro_state"], dtype=float),
            phase=float(payload["phase"]),
            mode=str(payload["mode"]),
            elapsed_seconds=float(payload["elapsed_seconds"]),
            event_count=int(payload["event_count"]),
        )


@dataclass(frozen=True)
class HybridAdvanceResult:
    state: HybridPhaseState
    requested_seconds: float
    advanced_seconds: float
    terminal: bool
    events_crossed: tuple[str, ...]


class ConservativeHybridPhaseEngine:
    """Advance table-based phase modes while updating only ledger increments."""

    def __init__(
        self,
        modes: Mapping[str, ConservativePhaseMode],
        next_modes: Mapping[str, str | None],
        macro_resets: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        self.modes = dict(modes)
        self.next_modes = dict(next_modes)
        self.macro_resets = {
            name: np.asarray(value, dtype=float).copy()
            for name, value in (macro_resets or {}).items()
        }
        if not self.modes or set(self.modes) != set(self.next_modes):
            raise ValueError("every phase mode requires one transition declaration")
        dimensions = {mode.macro_dimension for mode in self.modes.values()}
        if len(dimensions) != 1:
            raise ValueError("all phase modes must use the same macro dimension")
        for source, target in self.next_modes.items():
            if target is not None and target not in self.modes:
                raise ValueError(f"unknown next mode for {source}: {target}")
            reset = self.macro_resets.get(source)
            if reset is not None and reset.shape != (next(iter(dimensions)),):
                raise ValueError("macro reset dimension disagrees with modes")

    def decode(self, state: HybridPhaseState) -> np.ndarray:
        return self.modes[state.mode].decode(state.macro_state, state.phase)

    def advance(self, state: HybridPhaseState, duration_seconds: float) -> HybridAdvanceResult:
        requested = float(duration_seconds)
        if requested < 0.0 or state.mode not in self.modes:
            raise ValueError("invalid duration or mode")
        macro = np.asarray(state.macro_state, dtype=float).copy()
        phase = float(state.phase)
        mode_name = state.mode
        remaining = requested
        advanced = 0.0
        events: list[str] = []
        terminal = False
        while remaining > 0.0:
            mode = self.modes[mode_name]
            old_ledger = mode.ledger(phase)
            new_phase, used, reached = mode.advance_phase(phase, remaining)
            macro += mode.ledger(new_phase) - old_ledger
            phase = new_phase
            remaining -= used
            advanced += used
            if not reached:
                break
            target = self.next_modes[mode_name]
            if target is None:
                terminal = True
                break
            macro += self.macro_resets.get(
                mode_name, np.zeros(mode.macro_dimension, dtype=float)
            )
            events.append(f"{mode_name}->{target}")
            mode_name = target
            phase = 0.0
            if remaining <= 8.0 * np.finfo(float).eps:
                remaining = 0.0
                break
        result_state = HybridPhaseState(
            macro_state=macro,
            phase=phase,
            mode=mode_name,
            elapsed_seconds=state.elapsed_seconds + advanced,
            event_count=state.event_count + len(events),
        )
        return HybridAdvanceResult(
            state=result_state,
            requested_seconds=requested,
            advanced_seconds=advanced,
            terminal=terminal,
            events_crossed=tuple(events),
        )
