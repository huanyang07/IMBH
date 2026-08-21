from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_adaptive_hot_exit_phase_atlas_wp10c9d6c7c3b5c4f25ee as runner


def test_rank_selection_is_minimal_and_canonical() -> None:
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(24, 4)) @ rng.normal(size=(4, 30))
    basis, _singular, rank, defect = runner._canonical_basis(raw)
    assert rank == 4
    assert defect < 1.0e-12
    np.testing.assert_allclose(basis.T @ basis, np.eye(rank), atol=1.0e-12)
    for column in range(rank):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        assert basis[pivot, column] >= 0.0


def test_duration_doubles_only_with_margin_and_holds_after_event() -> None:
    assert runner._next_duration(None) == runner.manifest.INITIAL_DURATION_SECONDS
    base = {
        "duration_seconds": 2.0e-7,
        "event_gate_passed": False,
        "growth_margin_passed": True,
    }
    assert runner._next_duration(base) == 4.0e-7
    base["growth_margin_passed"] = False
    assert runner._next_duration(base) == 2.0e-7
    base["growth_margin_passed"] = True
    base["event_gate_passed"] = True
    assert runner._next_duration(base) == 2.0e-7


def test_stage_names_are_restart_stable() -> None:
    assert runner._stage_directory(1).name.endswith("_window_01")
    assert runner._stage_directory(8).name.endswith("_window_08")
    assert runner._scratch_directory(3).name == "window_03"
