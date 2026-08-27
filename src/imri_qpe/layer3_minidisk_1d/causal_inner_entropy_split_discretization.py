"""Entropy-stable periodic proof discretization for a frozen port-atlas anchor.

This module is deliberately local and linear.  It supplies the algebraic
building block used to certify the split method before any nonlinear atlas
walk or physical trajectory is authorized.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .causal_inner_full_port_atlas import FullPortAtlasAnchor


@dataclass(frozen=True)
class FrozenSplitOperators:
    centered_difference: np.ndarray
    jump_laplacian: np.ndarray
    transport_generator: np.ndarray
    source_generator: np.ndarray
    cell_count: int
    field_count: int
    cell_light_crossing_seconds: float
    interface_signal_speed_over_c: float


@dataclass(frozen=True)
class MidpointLedger:
    state: np.ndarray
    energy_before: float
    energy_after: float
    dissipated_energy: float
    ledger_relative_defect: float


@dataclass(frozen=True)
class StrangLedger:
    state: np.ndarray
    energy_before: float
    energy_after: float
    source_heat_deposit: float
    interface_entropy_dissipation: float
    total_ledger_relative_defect: float


@dataclass(frozen=True)
class FrozenSplitOperatorAudit:
    centered_skew_defect: float
    jump_symmetry_defect: float
    jump_positive_part: float
    transport_entropy_positive_part: float
    source_entropy_positive_part: float
    constant_state_transport_defect: float

    @property
    def passed(self) -> bool:
        return (
            self.centered_skew_defect <= 2.0e-13
            and self.jump_symmetry_defect <= 2.0e-13
            and self.jump_positive_part <= 2.0e-13
            and self.transport_entropy_positive_part <= 2.0e-13
            and self.source_entropy_positive_part <= 2.0e-13
            and self.constant_state_transport_defect <= 2.0e-13
        )


def periodic_centered_difference(cell_count: int) -> np.ndarray:
    """Dimensionless centered first difference on a periodic unit stencil."""

    count = int(cell_count)
    if count < 3:
        raise ValueError("periodic proof grid requires at least three cells")
    derivative = np.zeros((count, count), dtype=float)
    for cell in range(count):
        derivative[cell, (cell + 1) % count] += 0.5
        derivative[cell, (cell - 1) % count] -= 0.5
    return derivative


def periodic_jump_laplacian(cell_count: int) -> np.ndarray:
    """Negative-semidefinite periodic jump operator."""

    count = int(cell_count)
    if count < 3:
        raise ValueError("periodic proof grid requires at least three cells")
    laplacian = np.zeros((count, count), dtype=float)
    for cell in range(count):
        laplacian[cell, cell] -= 2.0
        laplacian[cell, (cell + 1) % count] += 1.0
        laplacian[cell, (cell - 1) % count] += 1.0
    return laplacian


def build_frozen_split_operators(
    anchor: FullPortAtlasAnchor,
    *,
    cell_count: int,
    cell_light_crossing_seconds: float,
) -> FrozenSplitOperators:
    if not isinstance(anchor, FullPortAtlasAnchor):
        raise TypeError("anchor must be FullPortAtlasAnchor")
    crossing = float(cell_light_crossing_seconds)
    if not np.isfinite(crossing) or crossing <= 0.0:
        raise ValueError("cell light-crossing time must be positive")
    centered = periodic_centered_difference(cell_count)
    jump = periodic_jump_laplacian(cell_count)
    fields = anchor.temporal_matrix.shape[0]
    identity = np.eye(fields)
    signal = float(np.max(np.abs(np.linalg.eigvalsh(anchor.coordinate_radial_matrix))))
    transport = (
        -np.kron(centered, anchor.coordinate_radial_matrix)
        + 0.5 * signal * np.kron(jump, identity)
    ) / crossing
    source = np.kron(np.eye(cell_count), anchor.source_matrix)
    return FrozenSplitOperators(
        centered,
        jump,
        transport,
        source,
        int(cell_count),
        fields,
        crossing,
        signal,
    )


def _energy(state: np.ndarray) -> float:
    return 0.5 * float(state @ state)


def midpoint_cayley_matrix(generator: np.ndarray, timestep: float) -> np.ndarray:
    matrix = np.asarray(generator, dtype=float)
    dt = float(timestep)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("generator must be square")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("timestep must be positive")
    identity = np.eye(matrix.shape[0])
    return np.linalg.solve(identity - 0.5 * dt * matrix, identity + 0.5 * dt * matrix)


def midpoint_cayley_step(
    generator: np.ndarray, state: np.ndarray, timestep: float
) -> MidpointLedger:
    old = np.asarray(state, dtype=float)
    matrix = np.asarray(generator, dtype=float)
    if old.shape != (matrix.shape[0],):
        raise ValueError("state and generator dimensions do not agree")
    update = midpoint_cayley_matrix(matrix, timestep)
    new = update @ old
    midpoint = 0.5 * (old + new)
    symmetric = 0.5 * (matrix + matrix.T)
    dissipated = -float(timestep) * float(midpoint @ symmetric @ midpoint)
    before = _energy(old)
    after = _energy(new)
    defect = abs((before - after) - dissipated) / max(
        before, after, abs(dissipated), np.finfo(float).tiny
    )
    return MidpointLedger(new, before, after, dissipated, float(defect))


def strang_split_step(
    operators: FrozenSplitOperators, state: np.ndarray, timestep: float
) -> StrangLedger:
    first = midpoint_cayley_step(
        operators.source_generator, state, 0.5 * float(timestep)
    )
    transport = midpoint_cayley_step(
        operators.transport_generator, first.state, float(timestep)
    )
    second = midpoint_cayley_step(
        operators.source_generator, transport.state, 0.5 * float(timestep)
    )
    source_heat = first.dissipated_energy + second.dissipated_energy
    interface_loss = transport.dissipated_energy
    total_loss = source_heat + interface_loss
    defect = abs((first.energy_before - second.energy_after) - total_loss) / max(
        first.energy_before,
        second.energy_after,
        abs(total_loss),
        np.finfo(float).tiny,
    )
    return StrangLedger(
        second.state,
        first.energy_before,
        second.energy_after,
        float(source_heat),
        float(interface_loss),
        float(defect),
    )


def audit_frozen_split_operators(
    operators: FrozenSplitOperators,
) -> FrozenSplitOperatorAudit:
    centered = operators.centered_difference
    jump = operators.jump_laplacian
    transport_symmetric = 0.5 * (
        operators.transport_generator + operators.transport_generator.T
    )
    source_symmetric = 0.5 * (
        operators.source_generator + operators.source_generator.T
    )
    constant = np.tile(
        np.linspace(-0.13, 0.17, operators.field_count), operators.cell_count
    )
    scale = max(float(np.linalg.norm(constant)), 1.0)
    return FrozenSplitOperatorAudit(
        centered_skew_defect=float(np.linalg.norm(centered + centered.T)),
        jump_symmetry_defect=float(np.linalg.norm(jump - jump.T)),
        jump_positive_part=max(float(np.max(np.linalg.eigvalsh(jump))), 0.0),
        transport_entropy_positive_part=max(
            float(np.max(np.linalg.eigvalsh(transport_symmetric))), 0.0
        ),
        source_entropy_positive_part=max(
            float(np.max(np.linalg.eigvalsh(source_symmetric))), 0.0
        ),
        constant_state_transport_defect=float(
            np.linalg.norm(operators.transport_generator @ constant) / scale
        ),
    )


__all__ = [
    "FrozenSplitOperatorAudit",
    "FrozenSplitOperators",
    "MidpointLedger",
    "StrangLedger",
    "audit_frozen_split_operators",
    "build_frozen_split_operators",
    "midpoint_cayley_matrix",
    "midpoint_cayley_step",
    "periodic_centered_difference",
    "periodic_jump_laplacian",
    "strang_split_step",
]
