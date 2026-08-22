from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_exact_retracted_hot_exit_phase_atlas_wp10c9d6c7c3b5c4f25f0 as target


def test_duration_restarts_at_half_failed_window_and_grows_prospectively() -> None:
    assert target._next_duration(target.manifest.FIRST_WINDOW_INDEX, None) == 2.0e-7
    prior = {"duration_seconds": 2.0e-7, "growth_margin_passed": True}
    assert target._next_duration(4, prior) == 4.0e-7
    prior["growth_margin_passed"] = False
    assert target._next_duration(4, prior) == 2.0e-7


def test_anchor_retraction_is_exact_and_physical() -> None:
    metrics = target._anchor_retraction_metrics()
    assert metrics["coordinate_residual_infinity"] == 0.0
    assert metrics["gauge_residual_infinity"] == 0.0
    assert metrics["maximum_scaled_anchor_departure"] == 0.0
    assert metrics["passed"]


def test_recovery_order_uses_only_accepted_window_two() -> None:
    summary = target._validate_order(target.manifest.FIRST_WINDOW_INDEX)
    assert summary["passed"]
    assert summary["window_index"] == 2
