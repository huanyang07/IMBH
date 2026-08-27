import numpy as np

from imri_qpe.layer3_minidisk_1d.causal_inner_reduced_hybrid_cycle import (
    ReducedEventReset,
    ReducedHybridCheckpoint,
    ReducedHybridTransition,
    dormand_prince_step,
    integrate_reduced_hybrid,
)


def test_dormand_prince_constant_flow_is_exact():
    rate = np.asarray((0.1, -0.2, 0.05, 0.03, 1.0))
    rhs = lambda _time, _state, _mode: rate
    initial = np.zeros(5)
    step = dormand_prince_step(rhs, 0.0, initial, 0.3, 0)
    assert np.allclose(step.state_fifth, 0.3 * rate, atol=2e-16, rtol=0.0)
    assert np.linalg.norm(step.error) <= 2e-16


def test_hybrid_integration_localizes_and_advances_finite_event():
    rates = {0: np.asarray((0.01, 0.0, 0.0, 0.0, 1.0)), 1: np.asarray((0.0, 0.02, 0.0, 0.0, 1.0))}
    rhs = lambda _time, _state, mode: rates[mode]
    first = ReducedHybridTransition("first", 0, 1, 1, lambda state: float(state[4] - 1.0), lambda _time, _state: ReducedEventReset(np.asarray((0.1, 0.0, 0.0, 0.0)), 0.1, 0.1, 0.2))
    second = ReducedHybridTransition("second", 1, 0, 1, lambda state: float(state[4] - 2.0), lambda _time, _state: ReducedEventReset(np.asarray((0.0, 0.2, 0.0, 0.0)), 0.2, 0.2, 0.2))
    checkpoint = ReducedHybridCheckpoint(np.zeros(5), 0.0, 0, 0.3, np.zeros(4), np.zeros(4), 0, 0, 0)
    result = integrate_reduced_hybrid(rhs, checkpoint, end_time_seconds=2.5, transitions=(first, second), absolute_tolerance=np.full(5, 1e-10), relative_tolerance=1e-9)
    assert [event.name for event in result.events] == ["first", "second"]
    assert np.allclose([event.entry_time_seconds for event in result.events], (1.0, 2.0), atol=2e-10, rtol=0.0)
    assert result.checkpoint.mode_index == 0
    assert abs(result.checkpoint.state5[4] - 2.5) <= 2e-12
    expected = 1.3 * rates[0][:4] + 0.9 * rates[1][:4] + np.asarray((0.1, 0.2, 0.0, 0.0))
    assert np.allclose(result.checkpoint.state5[:4], expected, atol=2e-11, rtol=0.0)
