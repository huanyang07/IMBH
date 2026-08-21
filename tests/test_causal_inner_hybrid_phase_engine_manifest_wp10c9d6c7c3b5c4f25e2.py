from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_hybrid_phase_engine_manifest_wp10c9d6c7c3b5c4f25e2 as manifest


def test_fit_is_train_only_and_holds_out_cold_states() -> None:
    fit = manifest._contract()["fit"]
    assert set(fit["cold_training_indices"]).isdisjoint(fit["cold_held_out_indices"])
    assert fit["event_macro_reset"] == "zero"


def test_engine_is_truth_free_and_event_driven() -> None:
    engine = manifest._contract()["engine"]
    assert engine["online_truth_calls"] == 0
    assert engine["online_470_roots"] == 0
    assert "event" in engine["event_integrator"]


def test_complete_cycle_remains_unavailable() -> None:
    scope = manifest._contract()["scope"]
    assert scope["hot_exit_missing"]
    assert scope["remaining_cycle_modes_missing"]
    assert not scope["predictive_cycle_authorized"]
