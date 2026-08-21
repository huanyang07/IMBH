from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_adaptive_hot_exit_phase_atlas_manifest_wp10c9d6c7c3b5c4f25ed as manifest


def test_manifest_preserves_event_and_forbids_microsteps() -> None:
    contract = manifest._contract()
    assert contract["hot_exit_event"]["unchanged_from_legacy_exact_BDF_event_definition"]
    assert contract["hot_exit_event"]["consecutive_accepted_windows_required"] == 2
    assert contract["phase_atlas"]["failed_window_never_propagates"]
    assert contract["phase_atlas"]["no_sequential_BDF_microsteps"]
    assert contract["truth_budget"]["new_nonlinear_fixed_Q_roots"] == 0
    assert not contract["decision"]["complete_cycle_execution_authorized"]


def test_manifest_freezes_bounded_adaptation() -> None:
    assert manifest.NODE_COUNT == 8
    assert manifest.INITIAL_DURATION_SECONDS == 2.0e-7
    assert manifest.MAXIMUM_DURATION_SECONDS == 6.4e-6
    assert manifest.MAXIMUM_WINDOWS == 8
    assert manifest.RATE_BASIS_RANKS == (4, 6, 8, 12, 16)
    assert manifest.MAXIMUM_UNIQUE_RATE_STATES_PER_WINDOW == 15
