"""Low-rank phase-domain collocation primitives.

The classes in this module are deliberately independent of the expensive
monolithic residual.  They represent a polynomial phase chart and expose its
physical-time derivative so that an offline driver can compare it with an
independently evaluated vector field.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _finite_array(value, *, ndim: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != ndim or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return result.copy()


@dataclass(frozen=True)
class PolynomialPhaseSegment:
    """One polynomial multiple-shooting segment on normalized phase [0, 1]."""

    start_time_seconds: float
    end_time_seconds: float
    coefficients: np.ndarray
    constraint_condition_number: float

    def __post_init__(self) -> None:
        coefficients = _finite_array(
            self.coefficients, ndim=2, name="coefficients"
        )
        start = float(self.start_time_seconds)
        end = float(self.end_time_seconds)
        condition = float(self.constraint_condition_number)
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            raise ValueError("phase segment requires an increasing finite time interval")
        if coefficients.shape[0] < 2:
            raise ValueError("phase segment requires at least a linear polynomial")
        if not np.isfinite(condition) or condition < 1.0:
            raise ValueError("constraint condition number is invalid")
        coefficients.setflags(write=False)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "start_time_seconds", start)
        object.__setattr__(self, "end_time_seconds", end)
        object.__setattr__(self, "constraint_condition_number", condition)

    @property
    def duration_seconds(self) -> float:
        return self.end_time_seconds - self.start_time_seconds

    @property
    def coordinate_dimension(self) -> int:
        return int(self.coefficients.shape[1])

    @property
    def degree(self) -> int:
        return int(self.coefficients.shape[0] - 1)

    def phase(self, time_seconds: float) -> float:
        value = (float(time_seconds) - self.start_time_seconds) / self.duration_seconds
        tolerance = 16.0 * np.finfo(float).eps
        if value < -tolerance or value > 1.0 + tolerance:
            raise ValueError("time lies outside the phase segment")
        return min(max(value, 0.0), 1.0)

    def value_at_phase(self, phase: float) -> np.ndarray:
        value = float(phase)
        if not 0.0 <= value <= 1.0:
            raise ValueError("phase must lie in [0, 1]")
        powers = value ** np.arange(self.degree + 1)
        return powers @ self.coefficients

    def rate_at_phase(self, phase: float) -> np.ndarray:
        value = float(phase)
        if not 0.0 <= value <= 1.0:
            raise ValueError("phase must lie in [0, 1]")
        orders = np.arange(1, self.degree + 1)
        derivative = (orders * value ** (orders - 1)) @ self.coefficients[1:]
        return derivative / self.duration_seconds

    def value(self, time_seconds: float) -> np.ndarray:
        return self.value_at_phase(self.phase(time_seconds))

    def rate(self, time_seconds: float) -> np.ndarray:
        return self.rate_at_phase(self.phase(time_seconds))

    @classmethod
    def from_constraints(
        cls,
        *,
        start_time_seconds: float,
        end_time_seconds: float,
        value_times_seconds: np.ndarray,
        values: np.ndarray,
        rate_times_seconds: np.ndarray,
        rates_per_second: np.ndarray,
    ) -> "PolynomialPhaseSegment":
        """Interpolate value and physical-time derivative constraints exactly."""

        value_times = _finite_array(
            value_times_seconds, ndim=1, name="value_times_seconds"
        )
        rate_times = _finite_array(
            rate_times_seconds, ndim=1, name="rate_times_seconds"
        )
        values_array = _finite_array(values, ndim=2, name="values")
        rates_array = _finite_array(
            rates_per_second, ndim=2, name="rates_per_second"
        )
        if values_array.shape[0] != len(value_times):
            raise ValueError("value constraints disagree")
        if rates_array.shape[0] != len(rate_times):
            raise ValueError("rate constraints disagree")
        if values_array.shape[1] != rates_array.shape[1]:
            raise ValueError("value and rate coordinate dimensions disagree")
        count = len(value_times) + len(rate_times)
        if count < 2:
            raise ValueError("at least two total constraints are required")
        start = float(start_time_seconds)
        end = float(end_time_seconds)
        duration = end - start
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("invalid phase duration")
        value_phase = (value_times - start) / duration
        rate_phase = (rate_times - start) / duration
        if (
            np.any(value_phase < 0.0)
            or np.any(value_phase > 1.0)
            or np.any(rate_phase < 0.0)
            or np.any(rate_phase > 1.0)
        ):
            raise ValueError("constraints must lie inside the phase interval")
        degree = count - 1
        matrix = []
        right_hand_side = []
        for phase, value in zip(value_phase, values_array, strict=True):
            matrix.append(phase ** np.arange(degree + 1))
            right_hand_side.append(value)
        orders = np.arange(degree + 1)
        for phase, rate in zip(rate_phase, rates_array, strict=True):
            row = np.zeros(degree + 1)
            row[1:] = orders[1:] * phase ** (orders[1:] - 1)
            matrix.append(row)
            right_hand_side.append(duration * rate)
        constraint_matrix = np.asarray(matrix, dtype=float)
        coefficients = np.linalg.solve(
            constraint_matrix, np.asarray(right_hand_side, dtype=float)
        )
        return cls(
            start_time_seconds=start,
            end_time_seconds=end,
            coefficients=coefficients,
            constraint_condition_number=float(np.linalg.cond(constraint_matrix)),
        )


@dataclass(frozen=True)
class PiecewisePhaseCollocation:
    """Ordered continuous collection of polynomial shooting segments."""

    segments: tuple[PolynomialPhaseSegment, ...]

    def __post_init__(self) -> None:
        segments = tuple(self.segments)
        if not segments:
            raise ValueError("at least one phase segment is required")
        dimension = segments[0].coordinate_dimension
        for left, right in zip(segments[:-1], segments[1:], strict=True):
            if left.coordinate_dimension != dimension or right.coordinate_dimension != dimension:
                raise ValueError("phase segment dimensions disagree")
            if left.end_time_seconds != right.start_time_seconds:
                raise ValueError("phase segments must be contiguous")
        object.__setattr__(self, "segments", segments)

    def _segment(self, time_seconds: float) -> PolynomialPhaseSegment:
        value = float(time_seconds)
        for segment in self.segments:
            if value <= segment.end_time_seconds:
                if value >= segment.start_time_seconds:
                    return segment
                break
        raise ValueError("time lies outside the piecewise phase atlas")

    def value(self, time_seconds: float) -> np.ndarray:
        return self._segment(time_seconds).value(time_seconds)

    def rate(self, time_seconds: float) -> np.ndarray:
        return self._segment(time_seconds).rate(time_seconds)

    def interface_value_defects(self) -> np.ndarray:
        return np.asarray(
            [
                np.linalg.norm(
                    left.value_at_phase(1.0) - right.value_at_phase(0.0)
                )
                for left, right in zip(self.segments[:-1], self.segments[1:], strict=True)
            ],
            dtype=float,
        )


def gauss_lobatto_nodes(count: int) -> np.ndarray:
    """Return Legendre--Gauss--Lobatto nodes mapped to [0, 1]."""

    if int(count) != count or count < 2:
        raise ValueError("Gauss-Lobatto node count must be an integer at least two")
    if count == 2:
        return np.asarray((0.0, 1.0))
    legendre = np.polynomial.legendre.Legendre.basis(count - 1)
    interior = np.sort(legendre.deriv().roots())
    return 0.5 * (np.concatenate(([-1.0], interior, [1.0])) + 1.0)


def lagrange_differentiation_matrix(nodes: np.ndarray) -> np.ndarray:
    """Return the nodal derivative matrix for distinct phase nodes.

    If ``values`` contains samples of a polynomial of degree at most
    ``len(nodes) - 1``, then ``D @ values`` is its derivative at the same
    nodes.  The small dense construction is intentionally explicit: the
    phase pilots use at most eight nodes and benefit from an independently
    auditable Vandermonde definition.
    """

    phase = _finite_array(nodes, ndim=1, name="nodes")
    if len(phase) < 2 or len(np.unique(phase)) != len(phase):
        raise ValueError("differentiation nodes must be distinct")
    powers = np.arange(len(phase))
    vandermonde = phase[:, None] ** powers[None, :]
    derivative = np.zeros_like(vandermonde)
    derivative[:, 1:] = (
        powers[None, 1:] * phase[:, None] ** (powers[None, 1:] - 1)
    )
    return derivative @ np.linalg.solve(vandermonde, np.eye(len(phase)))


def lagrange_integration_matrix(nodes: np.ndarray) -> np.ndarray:
    """Return integrals of nodal Lagrange polynomials from zero to each node.

    ``A[i, j]`` equals the integral from zero to ``nodes[i]`` of the
    Lagrange cardinal polynomial associated with ``nodes[j]``. Consequently
    ``A @ values`` integrates the unique nodal interpolating polynomial.
    """

    phase = _finite_array(nodes, ndim=1, name="nodes")
    if len(phase) < 2 or len(np.unique(phase)) != len(phase):
        raise ValueError("integration nodes must be distinct")
    powers = np.arange(len(phase))
    vandermonde = phase[:, None] ** powers[None, :]
    coefficients = np.linalg.solve(vandermonde, np.eye(len(phase)))
    integrated_powers = (
        phase[:, None] ** (powers[None, :] + 1) / (powers[None, :] + 1)
    )
    return integrated_powers @ coefficients


def relative_vector_defect(predicted: np.ndarray, reference: np.ndarray) -> float:
    predicted_array = np.asarray(predicted, dtype=float)
    reference_array = np.asarray(reference, dtype=float)
    if predicted_array.shape != reference_array.shape:
        raise ValueError("vector defect operands must have identical shapes")
    denominator = max(float(np.linalg.norm(reference_array)), np.finfo(float).tiny)
    return float(np.linalg.norm(predicted_array - reference_array) / denominator)


def direction_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    denominator = max(
        float(np.linalg.norm(left_array) * np.linalg.norm(right_array)),
        np.finfo(float).tiny,
    )
    return float(np.clip((left_array @ right_array) / denominator, -1.0, 1.0))
