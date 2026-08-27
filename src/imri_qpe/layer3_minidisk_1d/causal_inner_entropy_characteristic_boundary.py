"""Entropy-characteristic boundary operators for the eleven-field AP system.

The radial principal matrix is symmetric in the entropy variables used by the
offline port atlas.  Consequently the incoming subspace at a boundary is an
orthogonal spectral subspace of the outward-normal matrix ``A_n = n A``.
This module keeps that sign convention explicit and supplies the positive
semidefinite incoming penalty ``(-A_n)_+``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OutwardEntropyCharacteristicBoundary:
    """Spectral incoming/outgoing split of a symmetric entropy port."""

    outward_normal: float
    outward_matrix: np.ndarray
    characteristic_speeds: np.ndarray
    characteristic_vectors: np.ndarray
    incoming_mask: np.ndarray
    incoming_projector: np.ndarray
    incoming_penalty: np.ndarray

    @property
    def incoming_count(self) -> int:
        return int(np.count_nonzero(self.incoming_mask))


@dataclass(frozen=True)
class OutwardEntropyCharacteristicBoundaryAudit:
    outward_matrix_symmetry_defect: float
    characteristic_reconstruction_defect: float
    characteristic_orthogonality_defect: float
    projector_symmetry_defect: float
    projector_idempotence_defect: float
    penalty_symmetry_defect: float
    penalty_minimum_eigenvalue: float
    penalty_reconstruction_defect: float
    projector_penalty_commutator_defect: float
    incoming_count: int
    neutral_count: int
    maximum_absolute_speed: float

    @property
    def passed(self) -> bool:
        return bool(
            self.outward_matrix_symmetry_defect <= 2.0e-12
            and self.characteristic_reconstruction_defect <= 2.0e-12
            and self.characteristic_orthogonality_defect <= 2.0e-12
            and self.projector_symmetry_defect <= 2.0e-12
            and self.projector_idempotence_defect <= 2.0e-12
            and self.penalty_symmetry_defect <= 2.0e-12
            and self.penalty_minimum_eigenvalue >= -2.0e-12
            and self.penalty_reconstruction_defect <= 2.0e-12
            and self.projector_penalty_commutator_defect <= 2.0e-12
        )


def build_outward_entropy_characteristic_boundary(
    radial_matrix,
    *,
    outward_normal: float,
    symmetry_tolerance: float = 2.0e-12,
    neutral_tolerance: float = 1.0e-13,
) -> OutwardEntropyCharacteristicBoundary:
    """Build the orthogonal incoming split and ``(-A_n)_+`` penalty.

    Negative eigenvalues of the outward-normal matrix are incoming.  A zero
    speed is kept neutral instead of being assigned by floating-point noise.
    """

    matrix = np.asarray(radial_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("radial_matrix must be square")
    if np.any(~np.isfinite(matrix)):
        raise ValueError("radial_matrix must be finite")
    normal = float(outward_normal)
    if normal not in (-1.0, 1.0):
        raise ValueError("outward_normal must be -1 or +1")
    tolerance = float(symmetry_tolerance)
    neutral = float(neutral_tolerance)
    if tolerance < 0.0 or neutral < 0.0:
        raise ValueError("boundary tolerances must be nonnegative")
    symmetry = float(np.linalg.norm(matrix - matrix.T, ord=np.inf))
    if symmetry > tolerance:
        raise ValueError("radial_matrix is not symmetric in entropy variables")

    symmetric = 0.5 * (matrix + matrix.T)
    outward = normal * symmetric
    speeds, vectors = np.linalg.eigh(outward)
    incoming = speeds < -neutral
    incoming_vectors = vectors[:, incoming]
    projector = incoming_vectors @ incoming_vectors.T
    penalty_weights = np.maximum(-speeds, 0.0)
    penalty = (vectors * penalty_weights) @ vectors.T
    return OutwardEntropyCharacteristicBoundary(
        normal,
        outward,
        speeds,
        vectors,
        incoming,
        0.5 * (projector + projector.T),
        0.5 * (penalty + penalty.T),
    )


def audit_outward_entropy_characteristic_boundary(
    boundary: OutwardEntropyCharacteristicBoundary,
    *,
    neutral_tolerance: float = 1.0e-13,
) -> OutwardEntropyCharacteristicBoundaryAudit:
    if not isinstance(boundary, OutwardEntropyCharacteristicBoundary):
        raise TypeError("boundary must be an OutwardEntropyCharacteristicBoundary")
    matrix = boundary.outward_matrix
    speeds = boundary.characteristic_speeds
    vectors = boundary.characteristic_vectors
    projector = boundary.incoming_projector
    penalty = boundary.incoming_penalty
    reconstructed = (vectors * speeds) @ vectors.T
    expected_penalty = (vectors * np.maximum(-speeds, 0.0)) @ vectors.T
    scale = max(float(np.linalg.norm(matrix)), 1.0)
    penalty_scale = max(float(np.linalg.norm(penalty)), 1.0)
    identity = np.eye(matrix.shape[0])
    neutral_count = int(np.count_nonzero(np.abs(speeds) <= float(neutral_tolerance)))
    return OutwardEntropyCharacteristicBoundaryAudit(
        float(np.linalg.norm(matrix - matrix.T) / scale),
        float(np.linalg.norm(reconstructed - matrix) / scale),
        float(np.linalg.norm(vectors.T @ vectors - identity)),
        float(np.linalg.norm(projector - projector.T)),
        float(np.linalg.norm(projector @ projector - projector)),
        float(np.linalg.norm(penalty - penalty.T) / penalty_scale),
        float(np.min(np.linalg.eigvalsh(penalty))),
        float(np.linalg.norm(expected_penalty - penalty) / penalty_scale),
        float(np.linalg.norm(projector @ penalty - penalty @ projector) / penalty_scale),
        boundary.incoming_count,
        neutral_count,
        float(np.max(np.abs(speeds))),
    )


__all__ = [
    "OutwardEntropyCharacteristicBoundary",
    "OutwardEntropyCharacteristicBoundaryAudit",
    "audit_outward_entropy_characteristic_boundary",
    "build_outward_entropy_characteristic_boundary",
]
