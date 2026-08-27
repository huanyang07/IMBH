"""Sparse nonperiodic SBP/SAT action for the native eleven-field AP system.

The operator is formulated in cell-local entropy coordinates.  Its split
principal part is weighted skew-adjoint in the interior.  Jump viscosity,
the dissipative part of the local source, and maximally dissipative SAT
terms therefore give an exact, inspectable semidiscrete entropy identity.
The inner boundary can be pure excision while the outer incoming
characteristics enter through a separate affine control map.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, spsolve

from .causal_inner_entropy_characteristic_boundary import (
    OutwardEntropyCharacteristicBoundary,
    build_outward_entropy_characteristic_boundary,
)


Array = np.ndarray
Sparse = sparse.csr_matrix


def _finite(value, *, ndim: int, name: str) -> Array:
    result = np.asarray(value, dtype=float)
    if result.ndim != ndim or np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return result


def _sparse_frobenius(matrix: sparse.spmatrix) -> float:
    value = matrix.tocsr()
    return float(np.sqrt(np.dot(value.data, value.data)))


def _absolute_symmetric(matrix: Array) -> Array:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    result = (vectors * np.abs(values)) @ vectors.T
    return 0.5 * (result + result.T)


@dataclass(frozen=True)
class NonperiodicGlobalAPOperator:
    generator: Sparse
    outer_control: Sparse
    entropy_weights: Array
    sbp_q: Sparse
    jump: Sparse
    viscosity: Array
    radial_matrices: Array
    source_matrices: Array
    transport_rate_scale_per_second: float
    inner_boundary: OutwardEntropyCharacteristicBoundary
    outer_boundary: OutwardEntropyCharacteristicBoundary
    outer_incoming_vectors: Array
    weighted_symmetric_part: Sparse
    expected_weighted_symmetric_part: Sparse

    @property
    def cell_count(self) -> int:
        return int(len(self.entropy_weights))

    @property
    def field_count(self) -> int:
        return int(self.radial_matrices.shape[1])

    @property
    def state_dimension(self) -> int:
        return int(self.cell_count * self.field_count)

    @property
    def outer_control_dimension(self) -> int:
        return int(self.outer_control.shape[1])

    def entropy(self, state) -> float:
        value = _finite(state, ndim=1, name="state")
        if value.shape != (self.state_dimension,):
            raise ValueError("state has the wrong dimension")
        cells = value.reshape(self.cell_count, self.field_count)
        return 0.5 * float(np.sum(self.entropy_weights[:, None] * cells * cells))

    def affine_rhs(self, state, outer_incoming_amplitudes) -> Array:
        value = _finite(state, ndim=1, name="state")
        amplitudes = _finite(
            outer_incoming_amplitudes, ndim=1, name="outer incoming amplitudes"
        )
        if value.shape != (self.state_dimension,):
            raise ValueError("state has the wrong dimension")
        if amplitudes.shape != (self.outer_control_dimension,):
            raise ValueError("outer incoming amplitudes have the wrong dimension")
        return np.asarray(self.generator @ value + self.outer_control @ amplitudes)


@dataclass(frozen=True)
class NonperiodicGlobalAPOperatorAudit:
    sbp_adjoint_defect: float
    sbp_constant_defect: float
    maximum_radial_symmetry_defect: float
    maximum_source_entropy_positive_eigenvalue: float
    minimum_source_nullity: int
    minimum_viscosity: float
    energy_identity_relative_defect: float
    maximum_homogeneous_entropy_growth_eigenvalue: float
    inner_incoming_count: int
    outer_incoming_count: int
    inner_penalty_norm: float
    affine_action_relative_defect: float

    @property
    def passed(self) -> bool:
        return bool(
            self.sbp_adjoint_defect <= 5.0e-14
            and self.sbp_constant_defect <= 5.0e-14
            and self.maximum_radial_symmetry_defect <= 2.0e-12
            and self.maximum_source_entropy_positive_eigenvalue <= 2.0e-12
            and self.minimum_source_nullity >= 1
            and self.minimum_viscosity >= 0.0
            and self.energy_identity_relative_defect <= 5.0e-12
            and self.maximum_homogeneous_entropy_growth_eigenvalue <= 5.0e-11
            and self.inner_penalty_norm <= 2.0e-12
            and self.affine_action_relative_defect <= 5.0e-12
        )


@dataclass(frozen=True)
class NonperiodicGlobalAPMidpointStep:
    state: Array
    entropy_before: float
    entropy_after: float
    homogeneous_dissipation: float
    outer_boundary_work: float
    entropy_ledger_relative_defect: float


@dataclass(frozen=True)
class NonperiodicGlobalAPCheckpoint:
    state: Array
    outer_incoming_amplitudes: Array
    elapsed_time_seconds: float
    completed_steps: int


def native_sbp_q(cell_count: int) -> Sparse:
    """Return the cell-centred finite-volume SBP matrix.

    It obeys ``Q+Q.T=diag(-1,0,...,0,+1)`` and ``Q@1=0`` exactly.
    """

    count = int(cell_count)
    if count < 2:
        raise ValueError("nonperiodic SBP grid needs at least two cells")
    q = sparse.lil_matrix((count, count), dtype=float)
    q[0, 0] = -0.5
    q[-1, -1] = 0.5
    for cell in range(count - 1):
        q[cell, cell + 1] = 0.5
        q[cell + 1, cell] = -0.5
    return q.tocsr()


def native_jump_matrix(cell_count: int, field_count: int) -> Sparse:
    count = int(cell_count)
    fields = int(field_count)
    if count < 2 or fields < 1:
        raise ValueError("jump dimensions must be positive")
    row_count = (count - 1) * fields
    rows = np.repeat(np.arange(row_count), 2)
    local = np.arange(row_count)
    face = local // fields
    component = local % fields
    columns = np.column_stack(
        (face * fields + component, (face + 1) * fields + component)
    ).reshape(-1)
    data = np.tile(np.asarray((-1.0, 1.0)), row_count)
    return sparse.csr_matrix((data, (rows, columns)), shape=(row_count, count * fields))


def build_nonperiodic_global_ap_operator(
    radial_matrices,
    source_matrices,
    cell_measures,
    *,
    transport_rate_scale_per_second: float = 1.0,
    symmetry_tolerance: float = 2.0e-12,
) -> NonperiodicGlobalAPOperator:
    radial = _finite(radial_matrices, ndim=3, name="radial matrices")
    source = _finite(source_matrices, ndim=3, name="source matrices")
    measures = _finite(cell_measures, ndim=1, name="cell measures")
    if radial.shape != source.shape or radial.shape[1] != radial.shape[2]:
        raise ValueError("radial and source matrices must have matching square blocks")
    cells, fields, _ = radial.shape
    if measures.shape != (cells,) or np.any(measures <= 0.0):
        raise ValueError("cell measures must be positive and match the radial cells")
    rate = float(transport_rate_scale_per_second)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("transport rate scale must be positive")
    radial_defect = float(np.max(np.linalg.norm(radial - radial.transpose(0, 2, 1), axis=(1, 2))))
    if radial_defect > float(symmetry_tolerance):
        raise ValueError("radial principal matrix is not symmetric in entropy coordinates")
    radial = 0.5 * (radial + radial.transpose(0, 2, 1))

    weights = measures / float(np.mean(measures))
    h_diagonal = np.repeat(weights, fields)
    h = sparse.diags(h_diagonal, format="csr")
    h_inverse = sparse.diags(1.0 / h_diagonal, format="csr")
    q = native_sbp_q(cells)
    q_fields = sparse.kron(q, sparse.eye(fields, format="csr"), format="csr")
    a = sparse.block_diag(
        [sparse.csr_matrix(rate * value) for value in radial], format="csr"
    )
    s = sparse.block_diag(
        [sparse.csr_matrix(value) for value in source], format="csr"
    )
    split = -0.5 * (h_inverse @ (q_fields @ a + a @ q_fields))

    jump = native_jump_matrix(cells, fields)
    viscosity = np.asarray(
        [rate * np.max(np.abs(np.linalg.eigvalsh(0.5 * (radial[i] + radial[i + 1])))) for i in range(cells - 1)]
    )
    r = sparse.diags(np.repeat(viscosity, fields), format="csr")
    viscosity_operator = -(h_inverse @ jump.T @ r @ jump)

    inner = build_outward_entropy_characteristic_boundary(
        rate * radial[0], outward_normal=-1.0
    )
    outer = build_outward_entropy_characteristic_boundary(
        rate * radial[-1], outward_normal=1.0
    )
    sat = sparse.lil_matrix((cells * fields, cells * fields), dtype=float)
    sat[:fields, :fields] = -inner.incoming_penalty / weights[0]
    sat[-fields:, -fields:] = -outer.incoming_penalty / weights[-1]
    sat = sat.tocsr()
    incoming_vectors = outer.characteristic_vectors[:, outer.incoming_mask]
    control = sparse.lil_matrix((cells * fields, incoming_vectors.shape[1]), dtype=float)
    if incoming_vectors.shape[1]:
        control[-fields:, :] = (
            outer.incoming_penalty @ incoming_vectors / weights[-1]
        )
    control = control.tocsr()

    generator = (split + viscosity_operator + s + sat).tocsr()
    weighted = (h @ generator + generator.T @ h).tocsr()
    expected = (-2.0 * jump.T @ r @ jump + h @ (s + s.T)).tolil()
    inner_absolute = _absolute_symmetric(inner.outward_matrix)
    outer_absolute = _absolute_symmetric(outer.outward_matrix)
    expected[:fields, :fields] = expected[:fields, :fields] - inner_absolute
    expected[-fields:, -fields:] = expected[-fields:, -fields:] - outer_absolute
    expected = expected.tocsr()
    return NonperiodicGlobalAPOperator(
        generator,
        control,
        weights,
        q,
        jump,
        viscosity,
        radial,
        source,
        rate,
        inner,
        outer,
        incoming_vectors,
        weighted,
        expected,
    )


def audit_nonperiodic_global_ap_operator(
    operator: NonperiodicGlobalAPOperator,
) -> NonperiodicGlobalAPOperatorAudit:
    if not isinstance(operator, NonperiodicGlobalAPOperator):
        raise TypeError("operator must be a NonperiodicGlobalAPOperator")
    count = operator.cell_count
    boundary = sparse.diags(
        np.concatenate(([-1.0], np.zeros(count - 2), [1.0])), format="csr"
    )
    sbp = operator.sbp_q + operator.sbp_q.T - boundary
    sbp_adjoint = _sparse_frobenius(sbp)
    sbp_constant = float(np.linalg.norm(operator.sbp_q @ np.ones(count)))
    radial_defect = float(
        np.max(
            np.linalg.norm(
                operator.radial_matrices
                - operator.radial_matrices.transpose(0, 2, 1),
                axis=(1, 2),
            )
        )
    )
    source_positive = float(
        max(
            np.max(np.linalg.eigvalsh(0.5 * (value + value.T)))
            for value in operator.source_matrices
        )
    )
    source_nullity = min(
        value.shape[0] - np.linalg.matrix_rank(value, tol=1.0e-10)
        for value in operator.source_matrices
    )
    difference = operator.weighted_symmetric_part - operator.expected_weighted_symmetric_part
    identity_defect = _sparse_frobenius(difference) / max(
        _sparse_frobenius(operator.expected_weighted_symmetric_part), 1.0
    )
    inverse_root = sparse.diags(
        np.repeat(1.0 / np.sqrt(operator.entropy_weights), operator.field_count),
        format="csr",
    )
    normalized = (
        inverse_root @ operator.weighted_symmetric_part @ inverse_root
    ).tocsr()
    normalized = 0.5 * (normalized + normalized.T)
    maximum_growth = float(
        eigsh(normalized, k=1, which="LA", return_eigenvectors=False, tol=1.0e-11)[0]
    )
    rng = np.random.default_rng(2026082703)
    state = rng.normal(size=operator.state_dimension)
    amplitudes = rng.normal(size=operator.outer_control_dimension)
    homogeneous = np.asarray(operator.generator @ state)
    affine = operator.affine_rhs(state, amplitudes)
    expected_affine = np.asarray(operator.outer_control @ amplitudes)
    affine_defect = float(
        np.linalg.norm((affine - homogeneous) - expected_affine)
        / max(np.linalg.norm(expected_affine), 1.0)
    )
    return NonperiodicGlobalAPOperatorAudit(
        sbp_adjoint,
        sbp_constant,
        radial_defect,
        source_positive,
        int(source_nullity),
        float(np.min(operator.viscosity)),
        float(identity_defect),
        maximum_growth,
        operator.inner_boundary.incoming_count,
        operator.outer_boundary.incoming_count,
        float(np.linalg.norm(operator.inner_boundary.incoming_penalty)),
        affine_defect,
    )


def midpoint_affine_step(
    operator: NonperiodicGlobalAPOperator,
    state,
    outer_incoming_amplitudes,
    timestep_seconds: float,
) -> NonperiodicGlobalAPMidpointStep:
    if not isinstance(operator, NonperiodicGlobalAPOperator):
        raise TypeError("operator must be a NonperiodicGlobalAPOperator")
    old = _finite(state, ndim=1, name="state")
    amplitudes = _finite(
        outer_incoming_amplitudes, ndim=1, name="outer incoming amplitudes"
    )
    dt = float(timestep_seconds)
    if old.shape != (operator.state_dimension,):
        raise ValueError("state has the wrong dimension")
    if amplitudes.shape != (operator.outer_control_dimension,):
        raise ValueError("outer incoming amplitudes have the wrong dimension")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("timestep must be positive")
    identity = sparse.eye(operator.state_dimension, format="csr")
    forcing = np.asarray(operator.outer_control @ amplitudes)
    new = np.asarray(
        spsolve(
            identity - 0.5 * dt * operator.generator,
            (identity + 0.5 * dt * operator.generator) @ old + dt * forcing,
        )
    )
    midpoint = 0.5 * (old + new)
    weighted_midpoint = np.repeat(
        operator.entropy_weights, operator.field_count
    ) * midpoint
    dissipation = -0.5 * dt * float(
        midpoint @ operator.weighted_symmetric_part @ midpoint
    )
    boundary_work = dt * float(weighted_midpoint @ forcing)
    before = operator.entropy(old)
    after = operator.entropy(new)
    balance = -dissipation + boundary_work
    defect = abs((after - before) - balance) / max(
        before, after, abs(dissipation), abs(boundary_work), np.finfo(float).tiny
    )
    return NonperiodicGlobalAPMidpointStep(
        new,
        before,
        after,
        float(dissipation),
        float(boundary_work),
        float(defect),
    )


def save_nonperiodic_global_ap_checkpoint(
    checkpoint: NonperiodicGlobalAPCheckpoint, path: str | Path
) -> None:
    if not isinstance(checkpoint, NonperiodicGlobalAPCheckpoint):
        raise TypeError("checkpoint must be a NonperiodicGlobalAPCheckpoint")
    np.savez_compressed(
        Path(path),
        state=np.asarray(checkpoint.state, dtype=float),
        outer_incoming_amplitudes=np.asarray(
            checkpoint.outer_incoming_amplitudes, dtype=float
        ),
        elapsed_time_seconds=np.asarray(checkpoint.elapsed_time_seconds),
        completed_steps=np.asarray(checkpoint.completed_steps, dtype=np.int64),
    )


def load_nonperiodic_global_ap_checkpoint(
    path: str | Path,
) -> NonperiodicGlobalAPCheckpoint:
    with np.load(Path(path), allow_pickle=False) as payload:
        return NonperiodicGlobalAPCheckpoint(
            np.array(payload["state"], copy=True),
            np.array(payload["outer_incoming_amplitudes"], copy=True),
            float(payload["elapsed_time_seconds"]),
            int(payload["completed_steps"]),
        )


__all__ = [
    "NonperiodicGlobalAPCheckpoint",
    "NonperiodicGlobalAPMidpointStep",
    "NonperiodicGlobalAPOperator",
    "NonperiodicGlobalAPOperatorAudit",
    "audit_nonperiodic_global_ap_operator",
    "build_nonperiodic_global_ap_operator",
    "load_nonperiodic_global_ap_checkpoint",
    "midpoint_affine_step",
    "native_jump_matrix",
    "native_sbp_q",
    "save_nonperiodic_global_ap_checkpoint",
]
