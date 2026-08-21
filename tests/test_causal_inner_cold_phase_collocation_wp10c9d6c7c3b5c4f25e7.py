from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_cold_phase_collocation_wp10c9d6c7c3b5c4f25e7 as cold


def test_cold_collocation_uses_independent_exact_rate_holdouts() -> None:
    metrics, arrays = cold._evaluate()
    assert metrics["passed"]
    assert metrics["exact_saved_full_model_rate_witnesses"] == 4
    assert metrics["new_truth_calls"] == 0
    assert arrays["heldout_true_exact_rates470_per_s"].shape == (2, 470)


def test_cold_collocation_scope_remains_local() -> None:
    metrics, _arrays = cold._evaluate()
    assert metrics["transition_collocation_authorized"]
    assert not metrics["post_transition_segment_authorized"]
    assert not metrics["predictive_cycle_authorized"]
