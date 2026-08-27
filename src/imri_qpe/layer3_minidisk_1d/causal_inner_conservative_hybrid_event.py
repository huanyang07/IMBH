"""Ledger-exact resets and dense guard localization for hybrid AP cycles.

This module supplies numerical structure only.  A physical event model must
still provide the guard, integrated ledger impulse, constitutive jump, event
duration, and destination mode from independently validated truth data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import brentq


Array = np.ndarray
Guard = Callable[[Array, float], float]


def _finite(value, *, ndim: int, name: str) -> Array:
    array = np.asarray(value, dtype=float)
    if array.ndim != ndim or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return array.copy()


@dataclass(frozen=True)
class EntropyLedgerResetGeometry:
    """Weighted minimum-entropy normal and ledger-null projection."""

    conservation_map: Array
    entropy_weights: Array
    minimum_norm_normal: Array
    normal_gram: Array

    @property
    def ledger_dimension(self) -> int:
        return int(self.conservation_map.shape[0])

    @property
    def state_dimension(self) -> int:
        return int(self.conservation_map.shape[1])

    def project_ledger_null(self, value) -> Array:
        vector = _finite(value, ndim=1, name="ledger-null candidate")
        if vector.shape != (self.state_dimension,):
            raise ValueError("ledger-null candidate has the wrong dimension")
        return vector - self.minimum_norm_normal @ (self.conservation_map @ vector)

    def minimum_norm_jump(self, ledger_impulse) -> Array:
        impulse = _finite(ledger_impulse, ndim=1, name="ledger impulse")
        if impulse.shape != (self.ledger_dimension,):
            raise ValueError("ledger impulse has the wrong dimension")
        return self.minimum_norm_normal @ impulse

    def reset_jump(self, ledger_impulse, constitutive_jump) -> Array:
        return self.minimum_norm_jump(ledger_impulse) + self.project_ledger_null(
            constitutive_jump
        )

    def reset_state(self, state, ledger_impulse, constitutive_jump) -> Array:
        value = _finite(state, ndim=1, name="pre-event state")
        if value.shape != (self.state_dimension,):
            raise ValueError("pre-event state has the wrong dimension")
        return value + self.reset_jump(ledger_impulse, constitutive_jump)

    def weighted_norm(self, value) -> float:
        vector = _finite(value, ndim=1, name="weighted-norm value")
        if vector.shape != (self.state_dimension,):
            raise ValueError("weighted-norm value has the wrong dimension")
        return float(np.sqrt(vector @ (self.entropy_weights * vector)))


@dataclass(frozen=True)
class EntropyLedgerResetGeometryAudit:
    ledger_rank: int
    normal_gram_condition_number: float
    normal_identity_defect: float
    null_projector_constraint_defect: float
    weighted_normal_null_orthogonality_defect: float

    @property
    def passed(self) -> bool:
        return bool(
            self.ledger_rank > 0
            and np.isfinite(self.normal_gram_condition_number)
            and self.normal_gram_condition_number <= 1.0e8
            and self.normal_identity_defect <= 2.0e-12
            and self.null_projector_constraint_defect <= 2.0e-12
            and self.weighted_normal_null_orthogonality_defect <= 2.0e-12
        )


@dataclass(frozen=True)
class EntropyLedgerResetAudit:
    ledger_relative_defect: float
    projected_constitutive_ledger_defect: float
    minimum_to_null_weighted_orthogonality_defect: float
    minimum_norm_jump: float
    realized_jump_norm: float

    @property
    def passed(self) -> bool:
        return bool(
            self.ledger_relative_defect <= 2.0e-12
            and self.projected_constitutive_ledger_defect <= 2.0e-12
            and self.minimum_to_null_weighted_orthogonality_defect <= 2.0e-12
            and self.realized_jump_norm + 2.0e-14 >= self.minimum_norm_jump
        )


def build_entropy_ledger_reset_geometry(
    conservation_map,
    entropy_weights,
    *,
    maximum_condition_number: float = 1.0e8,
) -> EntropyLedgerResetGeometry:
    conservation = _finite(conservation_map, ndim=2, name="conservation map")
    weights = _finite(entropy_weights, ndim=1, name="entropy weights")
    ledger_dimension, state_dimension = conservation.shape
    if ledger_dimension < 1 or state_dimension <= ledger_dimension:
        raise ValueError("reset geometry needs a proper ledger subspace")
    if weights.shape != (state_dimension,) or np.any(weights <= 0.0):
        raise ValueError("entropy weights must be positive and match the state")
    if np.linalg.matrix_rank(conservation) != ledger_dimension:
        raise ValueError("conservation map must have full row rank")
    inverse_weights = 1.0 / weights
    inverse_square_root_weights = np.sqrt(inverse_weights)
    weighted_conservation = conservation * inverse_square_root_weights[None, :]
    gram = weighted_conservation @ weighted_conservation.T
    condition = float(np.linalg.cond(gram))
    if not np.isfinite(condition) or condition > float(maximum_condition_number):
        raise ValueError("reset normal Gram matrix is ill-conditioned")
    # If D=C W^-1/2, the weighted minimum-norm right inverse is
    # N=W^-1/2 D^+.  Form D^+ by a thin SVD rather than through the normal
    # equations: the latter square the condition number and materially lose
    # ledger closure for the native four-ledger physical scaling.
    left_vectors, singular_values, right_vectors_transpose = np.linalg.svd(
        weighted_conservation, full_matrices=False
    )
    if np.min(singular_values) <= np.finfo(float).eps * np.max(singular_values):
        raise ValueError("weighted conservation map is numerically rank deficient")
    weighted_pseudoinverse = (
        right_vectors_transpose.T / singular_values[None, :]
    ) @ left_vectors.T
    normal = inverse_square_root_weights[:, None] * weighted_pseudoinverse
    for value in (conservation, weights, normal, gram):
        value.setflags(write=False)
    return EntropyLedgerResetGeometry(conservation, weights, normal, gram)


def audit_entropy_ledger_reset_geometry(
    geometry: EntropyLedgerResetGeometry,
) -> EntropyLedgerResetGeometryAudit:
    if not isinstance(geometry, EntropyLedgerResetGeometry):
        raise TypeError("geometry must be EntropyLedgerResetGeometry")
    conservation = geometry.conservation_map
    normal = geometry.minimum_norm_normal
    identity = np.eye(geometry.ledger_dimension)
    normal_identity = conservation @ normal
    null_constraint = conservation - normal_identity @ conservation
    weighted_normal = normal.T * geometry.entropy_weights[None, :]
    normal_null = weighted_normal - (weighted_normal @ normal) @ conservation
    return EntropyLedgerResetGeometryAudit(
        int(np.linalg.matrix_rank(conservation)),
        float(np.linalg.cond(geometry.normal_gram)),
        float(np.linalg.norm(normal_identity - identity, ord=np.inf)),
        float(np.linalg.norm(null_constraint, ord=np.inf) / max(np.linalg.norm(conservation, ord=np.inf), 1.0)),
        float(np.linalg.norm(normal_null, ord=np.inf) / max(np.linalg.norm(weighted_normal, ord=np.inf), 1.0)),
    )


def audit_entropy_ledger_reset(
    geometry: EntropyLedgerResetGeometry,
    ledger_impulse,
    constitutive_jump,
) -> EntropyLedgerResetAudit:
    impulse = _finite(ledger_impulse, ndim=1, name="ledger impulse")
    candidate = _finite(constitutive_jump, ndim=1, name="constitutive jump")
    minimum = geometry.minimum_norm_jump(impulse)
    projected = geometry.project_ledger_null(candidate)
    realized = minimum + projected
    ledger = geometry.conservation_map @ realized
    ledger_scale = max(float(np.linalg.norm(impulse)), 1.0)
    projected_scale = max(float(np.linalg.norm(projected)), 1.0)
    minimum_scale = max(geometry.weighted_norm(minimum) * geometry.weighted_norm(projected), 1.0)
    return EntropyLedgerResetAudit(
        float(np.linalg.norm(ledger - impulse) / ledger_scale),
        float(np.linalg.norm(geometry.conservation_map @ projected) / projected_scale),
        abs(float(minimum @ (geometry.entropy_weights * projected))) / minimum_scale,
        geometry.weighted_norm(minimum),
        geometry.weighted_norm(realized),
    )


def cubic_hermite_dense_state(
    left_state,
    right_state,
    left_rate,
    right_rate,
    *,
    timestep: float,
    fraction: float,
) -> Array:
    left = _finite(left_state, ndim=1, name="left state")
    right = _finite(right_state, ndim=1, name="right state")
    f_left = _finite(left_rate, ndim=1, name="left rate")
    f_right = _finite(right_rate, ndim=1, name="right rate")
    if not (left.shape == right.shape == f_left.shape == f_right.shape):
        raise ValueError("dense-output state and rate dimensions disagree")
    step = float(timestep)
    theta = float(fraction)
    if not np.isfinite(step) or step <= 0.0 or not 0.0 <= theta <= 1.0:
        raise ValueError("dense-output timestep/fraction is invalid")
    theta2 = theta * theta
    theta3 = theta2 * theta
    return (
        (2.0 * theta3 - 3.0 * theta2 + 1.0) * left
        + (theta3 - 2.0 * theta2 + theta) * step * f_left
        + (-2.0 * theta3 + 3.0 * theta2) * right
        + (theta3 - theta2) * step * f_right
    )


@dataclass(frozen=True)
class GuardLocalization:
    fraction: float
    event_time: float
    event_state: Array
    guard_value: float
    bracket_left_fraction: float
    bracket_right_fraction: float
    iterations: int
    orientation: str


def localize_bracketed_guard(
    guard: Guard,
    left_state,
    right_state,
    left_rate,
    right_rate,
    *,
    start_time: float,
    timestep: float,
    orientation: str = "either",
    scan_subintervals: int = 64,
    absolute_tolerance: float = 1.0e-13,
) -> GuardLocalization:
    if orientation not in ("either", "negative_to_positive", "positive_to_negative"):
        raise ValueError("unsupported guard orientation")
    count = int(scan_subintervals)
    if count < 2:
        raise ValueError("guard scan needs at least two subintervals")
    time0 = float(start_time)
    step = float(timestep)

    def state(theta: float) -> Array:
        return cubic_hermite_dense_state(
            left_state,
            right_state,
            left_rate,
            right_rate,
            timestep=step,
            fraction=theta,
        )

    def value(theta: float) -> float:
        result = float(guard(state(theta), time0 + theta * step))
        if not np.isfinite(result):
            raise ValueError("guard returned a nonfinite value")
        return result

    nodes = np.linspace(0.0, 1.0, count + 1)
    values = np.asarray([value(theta) for theta in nodes])
    brackets = [
        (position, position + 1)
        for position in range(count)
        if values[position] * values[position + 1] < 0.0
    ]
    interior_zeros = np.flatnonzero(np.abs(values[1:-1]) <= float(absolute_tolerance)) + 1
    root_count = len(brackets) + len(interior_zeros)
    if root_count != 1:
        raise ValueError("guard must contain exactly one resolved transverse crossing")
    if len(interior_zeros) == 1:
        root = float(nodes[int(interior_zeros[0])])
        left_index = max(int(interior_zeros[0]) - 1, 0)
        right_index = min(int(interior_zeros[0]) + 1, count)
        iterations = 0
    else:
        left_index, right_index = brackets[0]
        root, result = brentq(
            value,
            float(nodes[left_index]),
            float(nodes[right_index]),
            xtol=float(absolute_tolerance),
            rtol=max(4.0 * np.finfo(float).eps, float(absolute_tolerance)),
            full_output=True,
            disp=True,
        )
        iterations = int(result.iterations)
    left_value = values[left_index]
    right_value = values[right_index]
    crossing = (
        "negative_to_positive" if left_value < 0.0 < right_value else "positive_to_negative"
    )
    if orientation != "either" and crossing != orientation:
        raise ValueError("guard crossing has the wrong orientation")
    event_state = state(root)
    event_value = float(guard(event_state, time0 + root * step))
    return GuardLocalization(
        float(root),
        float(time0 + root * step),
        event_state,
        event_value,
        float(nodes[left_index]),
        float(nodes[right_index]),
        iterations,
        crossing,
    )


__all__ = [
    "EntropyLedgerResetAudit",
    "EntropyLedgerResetGeometry",
    "EntropyLedgerResetGeometryAudit",
    "GuardLocalization",
    "audit_entropy_ledger_reset",
    "audit_entropy_ledger_reset_geometry",
    "build_entropy_ledger_reset_geometry",
    "cubic_hermite_dense_state",
    "localize_bracketed_guard",
]
