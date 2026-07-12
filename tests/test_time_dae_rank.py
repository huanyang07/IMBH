from __future__ import annotations

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    constrained_tangency_audit,
    eliminated_descriptor_audit,
    matrix_rank_audit,
    normal_closure_audit,
    shared_flux_time_dae_count,
)


def _regular_manufactured_blocks():
    n = 6
    m = 4
    storage = np.diag(np.linspace(1.0, 2.0, n))
    tangent = np.vstack((np.eye(n - 1), np.zeros((1, n - 1))))
    flux_jacobian = np.zeros((n, m))
    flux_jacobian[-1, -1] = 1.0
    algebraic_jacobian = np.hstack(
        (np.eye(m - 1), np.zeros((m - 1, 1)))
    )
    constraint_gradient = np.zeros(n)
    constraint_gradient[-1] = 1.0
    return (
        storage,
        tangent,
        flux_jacobian,
        algebraic_jacobian,
        constraint_gradient,
    )


def test_shared_flux_time_dae_counts_are_exactly_square() -> None:
    constrained = shared_flux_time_dae_count(
        24, 12, boundary_eliminated=False
    )
    eliminated = shared_flux_time_dae_count(
        24, 12, boundary_eliminated=True
    )

    assert constrained.unknowns == constrained.residuals == 101
    assert constrained.differential_variables == 36
    assert eliminated.unknowns == eliminated.residuals == 100
    assert eliminated.differential_variables == 35
    assert constrained.algebraic_variables == eliminated.algebraic_variables == 65


def test_eliminated_descriptor_and_normal_closure_are_full_rank() -> None:
    storage, tangent, flux, algebraic, _constraint = (
        _regular_manufactured_blocks()
    )

    descriptor = eliminated_descriptor_audit(
        storage, tangent, flux, algebraic
    )
    normal = normal_closure_audit(storage, tangent, flux, algebraic)

    assert descriptor.matrix.shape == (9, 9)
    assert descriptor.rank == 9
    assert normal.matrix.shape == (4, 4)
    assert normal.rank == 4


def test_constrained_hidden_tangency_closes_missing_algebraic_direction() -> None:
    storage, _tangent, flux, algebraic, constraint = (
        _regular_manufactured_blocks()
    )

    audit = constrained_tangency_audit(
        storage, constraint, flux, algebraic
    )

    assert audit.matrix.shape == (4, 4)
    assert audit.rank == 4


def test_rank_audits_detect_missing_normal_flux_response() -> None:
    storage, tangent, flux, algebraic, constraint = (
        _regular_manufactured_blocks()
    )
    flux[-1, -1] = 0.0

    eliminated = eliminated_descriptor_audit(
        storage, tangent, flux, algebraic
    )
    constrained = constrained_tangency_audit(
        storage, constraint, flux, algebraic
    )

    assert eliminated.rank == eliminated.matrix.shape[0] - 1
    assert constrained.rank == constrained.matrix.shape[0] - 1


def test_declared_equilibration_recovers_scale_independent_rank() -> None:
    matrix = np.diag([1.0e-20, 1.0, 1.0e20])

    raw = matrix_rank_audit(matrix, relative_threshold=1.0e-10)
    equilibrated = matrix_rank_audit(
        matrix, relative_threshold=1.0e-10, equilibrate=True
    )

    assert raw.rank == 1
    assert equilibrated.rank == 3
    assert np.isclose(equilibrated.condition_estimate, 1.0)
