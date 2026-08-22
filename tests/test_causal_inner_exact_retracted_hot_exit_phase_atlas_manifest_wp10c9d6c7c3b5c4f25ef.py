from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_exact_retracted_hot_exit_phase_atlas_manifest_wp10c9d6c7c3b5c4f25ef as target


def test_recovery_contract_is_local_exact_and_fail_closed() -> None:
    assert target.FIRST_WINDOW_INDEX == 3
    assert target.MAXIMUM_WINDOW_INDEX == 8
    assert target.INITIAL_DURATION_SECONDS == 2.0e-7
    assert target.MAXIMUM_DURATION_SECONDS == 4.0e-7
    assert target.COORDINATE_TOLERANCE == 1.0e-10
    assert target.GAUGE_TOLERANCE == 1.0e-10
    assert target.MAXIMUM_AUGMENTED_CONDITION_NUMBER == 1.0e7
    assert target.MAXIMUM_SCALED_ANCHOR_DEPARTURE == 5.0e-2
    assert target.MAXIMUM_UNIQUE_RATE_STATES_PER_WINDOW == 15


def test_parent_failure_is_chart_localized() -> None:
    parent = target._validate_parent(require_clean=False)
    assert parent["accepted_window_02_hashes"]
    assert parent["rejected_window_03_hashes"]
