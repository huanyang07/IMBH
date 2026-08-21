from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_hybrid_phase_engine_wp10c9d6c7c3b5c4f25e3 as engine


def test_rank_model_finds_scalar_linear_data() -> None:
    times = np.asarray([0.0, 1.0, 2.0])
    hidden = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    origin, basis, coefficients, rank, capture = engine._rank_model(
        hidden, times, np.asarray([0, 1, 2])
    )
    assert rank == 1
    assert capture == 1.0
    assert origin.shape == (2,)
    assert basis.shape == (2, 1)
    assert coefficients.shape == (3, 1)


def test_pass_classification_is_scoped_to_observed_modes() -> None:
    assert "observed_modes" in engine.PASS_CLASSIFICATION
    assert "complete_cycle_calibration_missing" in engine.PASS_CLASSIFICATION


def test_saved_array_modes_build_without_truth_calls() -> None:
    online, data, arrays = engine._build_engine()
    assert set(online.modes) == {"cold_observed", "fixed_Q_transition_observed"}
    assert data["cold_hidden_embedding_rank"] <= 4
    assert arrays["cold_macro_ledger_table82"].shape[1] == 82
