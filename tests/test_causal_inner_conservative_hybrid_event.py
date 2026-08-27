import numpy as np
import pytest

from imri_qpe.layer3_minidisk_1d.causal_inner_conservative_hybrid_event import (
    audit_entropy_ledger_reset,
    audit_entropy_ledger_reset_geometry,
    build_entropy_ledger_reset_geometry,
    cubic_hermite_dense_state,
    localize_bracketed_guard,
)


def _geometry():
    rng = np.random.default_rng(20260827)
    conservation = rng.normal(size=(4, 31))
    weights = np.exp(rng.uniform(-1.0, 1.0, size=31))
    return build_entropy_ledger_reset_geometry(conservation, weights)


def test_weighted_minimum_entropy_reset_is_ledger_exact():
    geometry = _geometry()
    assert audit_entropy_ledger_reset_geometry(geometry).passed
    rng = np.random.default_rng(44)
    for _ in range(32):
        impulse = rng.normal(scale=1e-3, size=4)
        constitutive = rng.normal(size=31)
        audit = audit_entropy_ledger_reset(geometry, impulse, constitutive)
        assert audit.passed
        jump = geometry.reset_jump(impulse, constitutive)
        np.testing.assert_allclose(
            geometry.conservation_map @ jump, impulse, atol=2e-14, rtol=0.0
        )


def test_cubic_dense_output_and_guard_localization_are_exact_for_cubic_path():
    coefficient = np.asarray(
        ((0.2, -0.3), (0.7, 0.4), (-0.2, 0.1), (0.05, -0.03))
    )
    start = 1.7
    step = 5.8

    def state(time):
        x = time - start
        return coefficient[0] + coefficient[1] * x + coefficient[2] * x**2 + coefficient[3] * x**3

    def rate(time):
        x = time - start
        return coefficient[1] + 2 * coefficient[2] * x + 3 * coefficient[3] * x**2

    theta = 0.371
    target = state(start + theta * step)[0]
    result = localize_bracketed_guard(
        lambda value, _time: value[0] - target,
        state(start),
        state(start + step),
        rate(start),
        rate(start + step),
        start_time=start,
        timestep=step,
        orientation="negative_to_positive",
    )
    np.testing.assert_allclose(result.event_state, state(result.event_time), atol=2e-13)
    assert abs(result.fraction - theta) <= 2e-13
    assert abs(result.guard_value) <= 2e-13
    midpoint = cubic_hermite_dense_state(
        state(start), state(start + step), rate(start), rate(start + step), timestep=step, fraction=0.5
    )
    np.testing.assert_allclose(midpoint, state(start + 0.5 * step), atol=2e-14)


def test_unbracketed_and_multiple_crossings_fail_closed():
    left = np.asarray((0.0,))
    right = np.asarray((1.0,))
    rate = np.asarray((1.0,))
    with pytest.raises(ValueError, match="exactly one"):
        localize_bracketed_guard(
            lambda value, _time: value[0] + 2.0,
            left,
            right,
            rate,
            rate,
            start_time=0.0,
            timestep=1.0,
        )
    with pytest.raises(ValueError, match="exactly one"):
        localize_bracketed_guard(
            lambda _value, time: np.sin(4.0 * np.pi * time),
            left,
            right,
            rate,
            rate,
            start_time=0.01,
            timestep=0.98,
            scan_subintervals=128,
        )


def test_rank_deficient_reset_map_fails_closed():
    with pytest.raises(ValueError, match="full row rank"):
        build_entropy_ledger_reset_geometry(np.ones((2, 5)), np.ones(5))


def test_weighted_svd_reset_preserves_closure_for_conditioned_ledger_rows():
    rng = np.random.default_rng(9127)
    left, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    right, _ = np.linalg.qr(rng.normal(size=(16, 4)))
    conservation = left @ np.diag((1.0, 0.1, 0.01, 0.001)) @ right.T
    geometry = build_entropy_ledger_reset_geometry(conservation, np.ones(16))
    audit = audit_entropy_ledger_reset_geometry(geometry)
    assert 9.0e5 <= audit.normal_gram_condition_number <= 1.1e6
    assert audit.normal_identity_defect <= 2e-12
    assert audit.weighted_normal_null_orthogonality_defect <= 2e-12
    assert audit.passed
