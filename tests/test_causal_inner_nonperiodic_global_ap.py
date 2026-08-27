import numpy as np

from imri_qpe.layer3_minidisk_1d.causal_inner_nonperiodic_global_ap import (
    NonperiodicGlobalAPCheckpoint,
    audit_nonperiodic_global_ap_operator,
    build_nonperiodic_global_ap_operator,
    load_nonperiodic_global_ap_checkpoint,
    midpoint_affine_step,
    native_sbp_q,
    save_nonperiodic_global_ap_checkpoint,
)


def _operator():
    cells = 5
    fields = 3
    speeds = np.linspace(-0.8, -0.7, cells)
    radial = np.asarray([np.diag((value, 1.1 * value, 1.2 * value)) for value in speeds])
    source = np.broadcast_to(np.diag((0.0, -0.4, -0.9)), (cells, fields, fields)).copy()
    measures = np.geomspace(0.7, 1.4, cells)
    return build_nonperiodic_global_ap_operator(radial, source, measures)


def test_native_sbp_identities_are_exact():
    q = native_sbp_q(8).toarray()
    boundary = np.diag((-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    assert np.array_equal(q + q.T, boundary)
    assert np.array_equal(q @ np.ones(8), np.zeros(8))


def test_nonperiodic_operator_closes_energy_identity_and_boundary_counts():
    operator = _operator()
    audit = audit_nonperiodic_global_ap_operator(operator)
    assert audit.passed
    assert audit.inner_incoming_count == 0
    assert audit.outer_incoming_count == 3
    assert audit.maximum_homogeneous_entropy_growth_eigenvalue <= 2e-12


def test_affine_midpoint_step_closes_entropy_ledger():
    operator = _operator()
    state = np.linspace(-0.12, 0.17, operator.state_dimension)
    amplitudes = np.asarray((0.03, -0.01, 0.02))
    step = midpoint_affine_step(operator, state, amplitudes, 0.03)
    assert step.homogeneous_dissipation >= -1e-13
    assert step.entropy_ledger_relative_defect <= 2e-13


def test_checkpoint_roundtrip_is_bitwise(tmp_path):
    operator = _operator()
    checkpoint = NonperiodicGlobalAPCheckpoint(
        np.linspace(-0.1, 0.2, operator.state_dimension),
        np.asarray((0.01, 0.02, -0.03)),
        0.125,
        7,
    )
    path = tmp_path / "checkpoint.npz"
    save_nonperiodic_global_ap_checkpoint(checkpoint, path)
    loaded = load_nonperiodic_global_ap_checkpoint(path)
    assert np.array_equal(loaded.state, checkpoint.state)
    assert np.array_equal(
        loaded.outer_incoming_amplitudes, checkpoint.outer_incoming_amplitudes
    )
    assert loaded.elapsed_time_seconds == checkpoint.elapsed_time_seconds
    assert loaded.completed_steps == checkpoint.completed_steps
