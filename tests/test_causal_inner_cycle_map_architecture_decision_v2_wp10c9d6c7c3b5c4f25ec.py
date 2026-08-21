from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_cycle_map_architecture_decision_v2_wp10c9d6c7c3b5c4f25ec as decision


def test_corrected_engine_uses_accepted_transition_and_explicit_reset() -> None:
    engine, data, _arrays = decision._build_corrected_engine()
    assert tuple(engine.modes) == (
        "cold_observed",
        "fixed_Q_transition_observed",
        "post_transition_collocation_observed",
    )
    assert engine.modes["fixed_Q_transition_observed"].hidden_embedding_basis.shape == (388, 8)
    assert engine.modes["post_transition_collocation_observed"].hidden_embedding_basis.shape == (388, 4)
    assert engine.macro_resets["cold_observed"].shape == (82,)
    np.testing.assert_allclose(
        data["cold_initial_macro"]
        + engine.modes["cold_observed"].macro_ledger_knots[-1]
        + data["cold_to_transition_macro_reset"],
        data["transition"]["macro"][0],
        atol=1.0e-16,
    )


def test_v2_architecture_is_event_to_event_not_microstep_based() -> None:
    metrics = {
        "observed_mode_names": ("cold", "transition", "post"),
        "gate_values": {
            "cold_to_transition_macro_reset_norm": 1.0,
            "measured_100k_decode_wall_seconds": 2.0,
        }
    }
    architecture = decision._architecture(metrics)
    assert architecture["continuous_online_dimension"] == 83
    assert "event-to-event" in architecture["online_method"]
    assert not architecture["predictive_cycle_authorized"]
