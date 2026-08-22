"""Conservative hidden-amplitude reduction of an original physical field.

The 470-coordinate chart is split exactly into retained macro coordinates
and a complementary hidden fiber.  A reduced hidden basis may compress the
fiber, but the macro ledger is never projected or inferred from hidden
coefficients.  This module has no dependency on truth evaluators, fixed-Q
reactions, or nonlinear solvers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


Array = np.ndarray
ReducedRateEvaluator = Callable[[Array, Array, float], Array]


def _finite(value, *, ndim: int, name: str) -> Array:
    result = np.asarray(value, dtype=float)
    if result.ndim != ndim or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return result.copy()


def _relative(left: Array, right: Array) -> float:
    return float(
        np.linalg.norm(np.asarray(left) - np.asarray(right))
        / max(
            float(np.linalg.norm(left)),
            float(np.linalg.norm(right)),
            np.finfo(float).tiny,
        )
    )


@dataclass(frozen=True)
class ConservativeCoordinateSplit:
    """Biorthogonal macro/hidden coordinate decomposition."""

    macro_restriction: Array
    macro_lift: Array
    hidden_dual: Array
    hidden_lift: Array
    tolerance: float = 5.0e-11

    def __post_init__(self) -> None:
        restriction = _finite(
            self.macro_restriction, ndim=2, name="macro_restriction"
        )
        macro_lift = _finite(self.macro_lift, ndim=2, name="macro_lift")
        hidden_dual = _finite(self.hidden_dual, ndim=2, name="hidden_dual")
        hidden_lift = _finite(self.hidden_lift, ndim=2, name="hidden_lift")
        coordinate_dimension, macro_dimension = macro_lift.shape
        if restriction.shape != (macro_dimension, coordinate_dimension):
            raise ValueError("macro restriction/lift dimensions disagree")
        hidden_dimension = hidden_dual.shape[0]
        if hidden_dual.shape[1] != coordinate_dimension:
            raise ValueError("hidden dual has the wrong coordinate dimension")
        if hidden_lift.shape != (coordinate_dimension, hidden_dimension):
            raise ValueError("hidden lift/dual dimensions disagree")
        tolerance = float(self.tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        identities = self._identity_defects(
            restriction, macro_lift, hidden_dual, hidden_lift
        )
        if max(identities.values()) > tolerance:
            raise ValueError(f"coordinate split identities failed: {identities}")
        for name, value in (
            ("macro_restriction", restriction),
            ("macro_lift", macro_lift),
            ("hidden_dual", hidden_dual),
            ("hidden_lift", hidden_lift),
        ):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @staticmethod
    def _identity_defects(
        restriction: Array,
        macro_lift: Array,
        hidden_dual: Array,
        hidden_lift: Array,
    ) -> dict[str, float]:
        coordinate_dimension = macro_lift.shape[0]
        macro_dimension = macro_lift.shape[1]
        hidden_dimension = hidden_dual.shape[0]
        return {
            "macro_identity": float(
                np.linalg.norm(
                    restriction @ macro_lift - np.eye(macro_dimension), ord=np.inf
                )
            ),
            "hidden_identity": float(
                np.linalg.norm(
                    hidden_dual @ hidden_lift - np.eye(hidden_dimension), ord=np.inf
                )
            ),
            "macro_hidden_kernel": float(
                np.linalg.norm(restriction @ hidden_lift, ord=np.inf)
            ),
            "hidden_macro_kernel": float(
                np.linalg.norm(hidden_dual @ macro_lift, ord=np.inf)
            ),
            "partition": float(
                np.linalg.norm(
                    macro_lift @ restriction
                    + hidden_lift @ hidden_dual
                    - np.eye(coordinate_dimension),
                    ord=np.inf,
                )
            ),
        }

    @property
    def coordinate_dimension(self) -> int:
        return int(self.macro_lift.shape[0])

    @property
    def macro_dimension(self) -> int:
        return int(self.macro_lift.shape[1])

    @property
    def hidden_dimension(self) -> int:
        return int(self.hidden_dual.shape[0])

    @property
    def identity_defects(self) -> dict[str, float]:
        return self._identity_defects(
            self.macro_restriction,
            self.macro_lift,
            self.hidden_dual,
            self.hidden_lift,
        )

    def split(self, coordinate: Array) -> tuple[Array, Array]:
        value = _finite(coordinate, ndim=1, name="coordinate")
        if value.shape != (self.coordinate_dimension,):
            raise ValueError("coordinate has the wrong dimension")
        return self.macro_restriction @ value, self.hidden_dual @ value

    def compose(self, macro: Array, hidden: Array) -> Array:
        q = _finite(macro, ndim=1, name="macro")
        h = _finite(hidden, ndim=1, name="hidden")
        if q.shape != (self.macro_dimension,) or h.shape != (
            self.hidden_dimension,
        ):
            raise ValueError("macro or hidden coordinate has the wrong dimension")
        return self.macro_lift @ q + self.hidden_lift @ h

    def split_rate(self, coordinate_rate: Array) -> tuple[Array, Array]:
        return self.split(coordinate_rate)


def canonical_rate_basis(rate_samples: Array, rank: int) -> tuple[Array, Array, Array]:
    """Return a deterministic orthonormal basis for normalized rate rows."""

    values = _finite(rate_samples, ndim=2, name="rate_samples")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= np.finfo(float).tiny):
        raise ValueError("rate samples must be nonzero")
    selected_rank = int(rank)
    if selected_rank < 1 or selected_rank > min(values.shape):
        raise ValueError("rank lies outside the sample matrix dimensions")
    unit = values / norms[:, None]
    _left, singular, right = np.linalg.svd(unit, full_matrices=False)
    basis = np.asarray(right[:selected_rank].T, dtype=float)
    for column in range(basis.shape[1]):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0.0:
            basis[:, column] *= -1.0
    energy = np.cumsum(singular * singular) / np.sum(singular * singular)
    return basis, singular, energy


def relative_projection_defects(rate_samples: Array, basis: Array) -> Array:
    values = _finite(rate_samples, ndim=2, name="rate_samples")
    vectors = _finite(basis, ndim=2, name="basis")
    if vectors.shape[0] != values.shape[1]:
        raise ValueError("basis and rates have inconsistent dimensions")
    orthogonality = float(
        np.linalg.norm(vectors.T @ vectors - np.eye(vectors.shape[1]), ord=np.inf)
    )
    if orthogonality > 1.0e-10:
        raise ValueError("basis must be column orthonormal")
    projected = (values @ vectors) @ vectors.T
    return np.linalg.norm(values - projected, axis=1) / np.maximum(
        np.linalg.norm(values, axis=1), np.finfo(float).tiny
    )


def polynomial_holdout(
    nodes: Array,
    values: Array,
    training_indices: Array,
) -> tuple[Array, Array, Array]:
    """Interpolate vector data through selected nodes and return holdout errors."""

    x = _finite(nodes, ndim=1, name="nodes")
    table = _finite(values, ndim=2, name="values")
    indices = np.asarray(training_indices, dtype=int)
    if table.shape[0] != len(x) or indices.ndim != 1:
        raise ValueError("nodes, values, and training indices disagree")
    if len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(
        indices >= len(x)
    ):
        raise ValueError("training indices are invalid")
    heldout = np.arange(len(x))[~np.isin(np.arange(len(x)), indices)]
    vandermonde = np.vander(x[indices], N=len(indices), increasing=True)
    coefficients = np.linalg.solve(vandermonde, table[indices])
    predictions = np.vander(
        x[heldout], N=len(indices), increasing=True
    ) @ coefficients
    relative = np.asarray(
        [_relative(prediction, table[index]) for prediction, index in zip(
            predictions, heldout, strict=True
        )]
    )
    return heldout, predictions, relative


@dataclass(frozen=True)
class HiddenAmplitudeState:
    macro: Array
    amplitudes: Array
    forcing_phase: float
    mode: str
    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        macro = _finite(self.macro, ndim=1, name="macro")
        amplitudes = _finite(self.amplitudes, ndim=1, name="amplitudes")
        phase = float(self.forcing_phase)
        elapsed = float(self.elapsed_seconds)
        if not np.isfinite(phase) or not np.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("forcing phase and elapsed time must be finite")
        if not self.mode:
            raise ValueError("mode must be nonempty")
        macro.setflags(write=False)
        amplitudes.setflags(write=False)
        object.__setattr__(self, "macro", macro)
        object.__setattr__(self, "amplitudes", amplitudes)


@dataclass(frozen=True)
class ConservativeHiddenAmplitudeModel:
    """Decode and project an original free coordinate rate."""

    split: ConservativeCoordinateSplit
    hidden_origin: Array
    hidden_basis: Array

    def __post_init__(self) -> None:
        origin = _finite(self.hidden_origin, ndim=1, name="hidden_origin")
        basis = _finite(self.hidden_basis, ndim=2, name="hidden_basis")
        if origin.shape != (self.split.hidden_dimension,):
            raise ValueError("hidden origin has the wrong dimension")
        if basis.shape[0] != self.split.hidden_dimension:
            raise ValueError("hidden basis has the wrong dimension")
        defect = float(
            np.linalg.norm(basis.T @ basis - np.eye(basis.shape[1]), ord=np.inf)
        )
        if defect > 1.0e-10:
            raise ValueError("hidden basis must be column orthonormal")
        origin.setflags(write=False)
        basis.setflags(write=False)
        object.__setattr__(self, "hidden_origin", origin)
        object.__setattr__(self, "hidden_basis", basis)

    @property
    def amplitude_dimension(self) -> int:
        return int(self.hidden_basis.shape[1])

    def decode(self, state: HiddenAmplitudeState) -> Array:
        if state.macro.shape != (self.split.macro_dimension,) or state.amplitudes.shape != (
            self.amplitude_dimension,
        ):
            raise ValueError("state dimensions disagree with model")
        hidden = self.hidden_origin + self.hidden_basis @ state.amplitudes
        return self.split.compose(state.macro, hidden)

    def project_rate(self, coordinate_rate: Array) -> tuple[Array, Array, float]:
        macro_rate, hidden_rate = self.split.split_rate(coordinate_rate)
        amplitude_rate = self.hidden_basis.T @ hidden_rate
        unresolved = hidden_rate - self.hidden_basis @ amplitude_rate
        defect = float(
            np.linalg.norm(unresolved)
            / max(float(np.linalg.norm(hidden_rate)), np.finfo(float).tiny)
        )
        return macro_rate, amplitude_rate, defect
