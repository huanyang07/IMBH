from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_cycle_map_architecture_decision_wp10c9d6c7c3b5c4f25ec as decision


def test_extended_engine_has_scalar_phase_and_three_observed_modes() -> None:
    engine, data, _arrays = decision._build_extended_engine()
    assert tuple(engine.modes) == (
        "cold_observed",
        "fixed_Q_transition_observed",
        "post_transition_collocation_observed",
    )
    post = engine.modes["post_transition_collocation_observed"]
    assert post.hidden_embedding_basis.shape == (388, 4)
    assert post.macro_dimension == 82
    assert post.coordinate_dimension == 470
    assert len(data["post_phase"]) == 15
    assert np.all(np.diff(data["post_phase"]) > 0.0)


def test_architecture_specification_is_event_driven_and_truth_free() -> None:
    metrics = {
        "observed_mode_names": ("cold", "transition", "post"),
        "gate_values": {"measured_100k_decode_wall_seconds": 1.0},
    }
    specification = decision._architecture_specification(metrics)
    assert specification["online_state"]["continuous_dimension"] == 83
    assert specification["cycle_map"]["online_truth_calls"] == 0
    assert specification["cycle_map"]["online_nonlinear_fixed_Q_roots"] == 0
    assert specification["cycle_map"]["online_nanosecond_microsteps"] == 0
    assert not specification["current_boundary"]["complete_predictive_cycle_authorized"]
    assert specification["computational_target"]["several_day_cycle_target_is_architecturally_feasible"]
