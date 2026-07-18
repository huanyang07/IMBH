"""Increment-primary BDF coefficients, history, and ledger primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CAUSAL_BDF2_MAXIMUM_STEP_RATIO = 1.0 + np.sqrt(2.0)


@dataclass(frozen=True)
class CausalBDFCoefficients:
    """One validated BDF1 or variable-step BDF2 formula."""

    order: int
    current_timestep_seconds: float
    previous_timestep_seconds: float | None
    step_ratio: float | None
    current_increment_coefficient: float
    previous_increment_coefficient: float

    def validated(self) -> CausalBDFCoefficients:
        """Reject manually constructed coefficients outside the formula."""

        expected = causal_bdf_coefficients(
            self.order,
            self.current_timestep_seconds,
            self.previous_timestep_seconds,
        )
        if self != expected:
            raise ValueError(
                "causal BDF coefficients do not match the declared formula"
            )
        return expected


@dataclass(frozen=True)
class CausalFiveFieldBDFHistory:
    """Fixed history needed by one increment-primary BDF2 residual."""

    previous_physical_increment: np.ndarray
    previous_vertical_killing_increment: np.ndarray
    previous_timestep_seconds: float
    temporal_height_scheme: str = "path_integrated"

    def validated(
        self,
        *,
        total_unknowns: int,
        n_cells: int,
    ) -> CausalFiveFieldBDFHistory:
        """Return normalized history after validating its complete shape."""

        increment = np.asarray(
            self.previous_physical_increment,
            dtype=float,
        )
        vertical = np.asarray(
            self.previous_vertical_killing_increment,
            dtype=float,
        )
        timestep = float(self.previous_timestep_seconds)
        if (
            increment.shape != (total_unknowns,)
            or vertical.shape != (n_cells, 4)
            or np.any(~np.isfinite(increment))
            or np.any(~np.isfinite(vertical))
            or not np.isfinite(timestep)
            or timestep <= 0.0
            or self.temporal_height_scheme
            not in ("endpoint", "path_integrated")
        ):
            raise ValueError("causal BDF history is invalid")
        return CausalFiveFieldBDFHistory(
            previous_physical_increment=increment,
            previous_vertical_killing_increment=vertical,
            previous_timestep_seconds=timestep,
            temporal_height_scheme=self.temporal_height_scheme,
        )


@dataclass(frozen=True)
class CausalBDFDiscreteLedger:
    """BDF-weighted storage plus the new-endpoint balance rate."""

    weighted_storage_increment: np.ndarray
    endpoint_balance_integral: np.ndarray
    closure_defect: np.ndarray


@dataclass(frozen=True)
class CausalPhysicalIntervalLedger:
    """Actual storage change plus trapezoidal physical transport."""

    actual_storage_increment: np.ndarray
    trapezoidal_balance_integral: np.ndarray
    closure_defect: np.ndarray


def causal_bdf_coefficients(
    order: int,
    current_timestep_seconds: float,
    previous_timestep_seconds: float | None = None,
) -> CausalBDFCoefficients:
    """Return increment-form BDF1 or variable-step BDF2 coefficients."""

    if int(order) != order or order not in (1, 2):
        raise ValueError("causal BDF order must be one or two")
    current = float(current_timestep_seconds)
    if not np.isfinite(current) or current <= 0.0:
        raise ValueError("current BDF timestep must be positive")
    if order == 1:
        if previous_timestep_seconds is not None:
            raise ValueError("BDF1 does not consume a previous timestep")
        return CausalBDFCoefficients(
            order=1,
            current_timestep_seconds=current,
            previous_timestep_seconds=None,
            step_ratio=None,
            current_increment_coefficient=1.0,
            previous_increment_coefficient=0.0,
        )

    if previous_timestep_seconds is None:
        raise ValueError("BDF2 requires a previous timestep")
    previous = float(previous_timestep_seconds)
    if not np.isfinite(previous) or previous <= 0.0:
        raise ValueError("previous BDF timestep must be positive")
    ratio = current / previous
    if ratio > CAUSAL_BDF2_MAXIMUM_STEP_RATIO:
        raise ValueError(
            "BDF2 step ratio exceeds the variable-step stability bound"
        )
    current_coefficient = (1.0 + 2.0 * ratio) / (1.0 + ratio)
    previous_coefficient = -(ratio**2) / (1.0 + ratio)
    return CausalBDFCoefficients(
        order=2,
        current_timestep_seconds=current,
        previous_timestep_seconds=previous,
        step_ratio=ratio,
        current_increment_coefficient=current_coefficient,
        previous_increment_coefficient=previous_coefficient,
    )


def causal_bdf_weighted_increment(
    current_increment: np.ndarray | float,
    previous_increment: np.ndarray | float | None,
    coefficients: CausalBDFCoefficients,
) -> np.ndarray:
    """Return the BDF-weighted current and previous finite increments."""

    if not isinstance(coefficients, CausalBDFCoefficients):
        raise TypeError("coefficients must be CausalBDFCoefficients")
    coefficients = coefficients.validated()
    current = np.asarray(current_increment, dtype=float)
    if np.any(~np.isfinite(current)):
        raise ValueError("current BDF increment must be finite")
    weighted = coefficients.current_increment_coefficient * current
    if coefficients.order == 1:
        if previous_increment is not None:
            raise ValueError("BDF1 does not consume a previous increment")
        return np.asarray(weighted, dtype=float)
    if previous_increment is None:
        raise ValueError("BDF2 requires a previous increment")
    previous = np.asarray(previous_increment, dtype=float)
    if previous.shape != current.shape or np.any(~np.isfinite(previous)):
        raise ValueError("previous BDF increment is incompatible")
    return np.asarray(
        weighted
        + coefficients.previous_increment_coefficient * previous,
        dtype=float,
    )


def causal_bdf_increment_rate(
    current_increment: np.ndarray | float,
    previous_increment: np.ndarray | float | None,
    coefficients: CausalBDFCoefficients,
) -> np.ndarray:
    """Return the BDF derivative represented by finite increments."""

    if not isinstance(coefficients, CausalBDFCoefficients):
        raise TypeError("coefficients must be CausalBDFCoefficients")
    coefficients = coefficients.validated()
    return (
        causal_bdf_weighted_increment(
            current_increment,
            previous_increment,
            coefficients,
        )
        / coefficients.current_timestep_seconds
    )


def causal_bdf_quadratic_history_predictor(
    current_state: np.ndarray | float,
    previous_increment: np.ndarray | float,
    previous_timestep_seconds: float,
    older_increment: np.ndarray | float,
    older_timestep_seconds: float,
    requested_timestep_seconds: float,
) -> np.ndarray:
    """Extrapolate one state from its two most recent finite increments."""

    current = np.asarray(current_state, dtype=float)
    previous = np.asarray(previous_increment, dtype=float)
    older = np.asarray(older_increment, dtype=float)
    previous_dt = float(previous_timestep_seconds)
    older_dt = float(older_timestep_seconds)
    requested_dt = float(requested_timestep_seconds)
    if (
        previous.shape != current.shape
        or older.shape != current.shape
        or np.any(~np.isfinite(current))
        or np.any(~np.isfinite(previous))
        or np.any(~np.isfinite(older))
        or not np.isfinite(previous_dt)
        or previous_dt <= 0.0
        or not np.isfinite(older_dt)
        or older_dt <= 0.0
        or not np.isfinite(requested_dt)
        or requested_dt <= 0.0
    ):
        raise ValueError("causal BDF predictor history is invalid")
    recent_rate = previous / previous_dt
    older_rate = older / older_dt
    second_divided_difference = (
        (recent_rate - older_rate) / (previous_dt + older_dt)
    )
    return np.asarray(
        current
        + requested_dt * recent_rate
        + requested_dt
        * (requested_dt + previous_dt)
        * second_divided_difference,
        dtype=float,
    )


def causal_bdf_discrete_ledger(
    current_storage_increment: np.ndarray | float,
    previous_storage_increment: np.ndarray | float | None,
    endpoint_balance_rate: np.ndarray | float,
    coefficients: CausalBDFCoefficients,
) -> CausalBDFDiscreteLedger:
    """Audit the discrete BDF equation in integrated increment units."""

    if not isinstance(coefficients, CausalBDFCoefficients):
        raise TypeError("coefficients must be CausalBDFCoefficients")
    coefficients = coefficients.validated()
    weighted = causal_bdf_weighted_increment(
        current_storage_increment,
        previous_storage_increment,
        coefficients,
    )
    endpoint = np.asarray(endpoint_balance_rate, dtype=float)
    if endpoint.shape != weighted.shape or np.any(~np.isfinite(endpoint)):
        raise ValueError("BDF endpoint balance rate is incompatible")
    integrated = coefficients.current_timestep_seconds * endpoint
    return CausalBDFDiscreteLedger(
        weighted_storage_increment=weighted,
        endpoint_balance_integral=integrated,
        closure_defect=weighted + integrated,
    )


def causal_trapezoidal_physical_interval_ledger(
    actual_storage_increment: np.ndarray | float,
    old_balance_rate: np.ndarray | float,
    new_balance_rate: np.ndarray | float,
    timestep_seconds: float,
) -> CausalPhysicalIntervalLedger:
    """Audit one physical interval with second-order rate quadrature."""

    storage = np.asarray(actual_storage_increment, dtype=float)
    old_rate = np.asarray(old_balance_rate, dtype=float)
    new_rate = np.asarray(new_balance_rate, dtype=float)
    timestep = float(timestep_seconds)
    if (
        old_rate.shape != storage.shape
        or new_rate.shape != storage.shape
        or np.any(~np.isfinite(storage))
        or np.any(~np.isfinite(old_rate))
        or np.any(~np.isfinite(new_rate))
        or not np.isfinite(timestep)
        or timestep <= 0.0
    ):
        raise ValueError("physical interval ledger inputs are invalid")
    integrated = 0.5 * timestep * (old_rate + new_rate)
    return CausalPhysicalIntervalLedger(
        actual_storage_increment=storage,
        trapezoidal_balance_integral=integrated,
        closure_defect=storage + integrated,
    )
