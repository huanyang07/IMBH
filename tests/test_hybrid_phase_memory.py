from __future__ import annotations

import json

import numpy as np

from imri_qpe.layer3_minidisk_1d.hybrid_phase_memory import (
    ConservativeHybridPhaseEngine,
    ConservativePhaseMode,
    HybridPhaseState,
)


def _mode(name: str, ledger_scale: float = 1.0) -> ConservativePhaseMode:
    return ConservativePhaseMode(
        name=name,
        phase_knots=np.asarray([0.0, 0.5, 1.0]),
        phase_speeds_per_second=np.asarray([1.0, 2.0]),
        macro_ledger_knots=ledger_scale * np.asarray([[0.0], [1.0], [3.0]]),
        hidden_coefficient_knots=np.asarray([[0.0], [0.5], [1.0]]),
        hidden_origin=np.asarray([2.0]),
        hidden_embedding_basis=np.asarray([[1.0]]),
        macro_lift=np.asarray([[1.0], [0.0]]),
        hidden_lift=np.asarray([[0.0], [1.0]]),
        macro_restriction=np.asarray([[1.0, 0.0]]),
    )


def test_mode_decoder_preserves_macro_state() -> None:
    mode = _mode("a")
    decoded = mode.decode(np.asarray([7.0]), 0.25)
    assert np.array_equal(mode.macro_restriction @ decoded, np.asarray([7.0]))


def test_engine_crosses_event_and_telescopes_ledgers() -> None:
    first = _mode("a")
    second = _mode("b", ledger_scale=2.0)
    engine = ConservativeHybridPhaseEngine(
        {"a": first, "b": second}, {"a": "b", "b": None}
    )
    start = HybridPhaseState(np.asarray([10.0]), 0.0, "a")
    result = engine.advance(start, first.duration_seconds + 0.25)
    assert result.events_crossed == ("a->b",)
    assert result.state.mode == "b"
    assert result.state.phase == 0.25
    expected = 10.0 + first.ledger(1.0)[0] + second.ledger(0.25)[0]
    assert result.state.macro_state[0] == expected


def test_state_json_roundtrip_is_bitwise() -> None:
    state = HybridPhaseState(np.asarray([1.25, -3.5]), 0.375, "cold", 4.0, 2)
    replay = HybridPhaseState.from_payload(json.loads(json.dumps(state.to_payload())))
    assert np.array_equal(replay.macro_state, state.macro_state)
    assert replay.phase == state.phase
    assert replay.elapsed_seconds == state.elapsed_seconds
    assert replay.event_count == state.event_count
