from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "results/canonical/coupled_wall_pattern_power"


def test_pattern_power_gate_selects_open_overflow() -> None:
    summary = json.loads((CASE / "summary.json").read_text())
    config = json.loads((CASE / "config.json").read_text())
    assert config["reservoir_outer_radius_rg"] == 335.0
    assert not summary["pattern_power_gate"]
    assert not summary["reached_full_pattern_power"]
    assert summary["first_tidal_band_validity_failure_fraction"] == 0.25
    assert summary["last_model_valid_fraction"] == 0.0
    assert summary["last_numerically_accepted_fraction"] == 0.75
    assert summary["next_stage"] == "promote_inner_mdot_and_test_open_overflow"
    assert max(
        row["relative_power_identity_mismatch"] for row in summary["rows"]
    ) < 1.0e-14


def test_pattern_power_checkpoint_is_finite_minidisk_last_accepted_state() -> None:
    with np.load(CASE / "last_accepted_state.npz", allow_pickle=False) as data:
        assert float(data["pattern_power_fraction"]) == 0.75
        outer_radius = np.asarray(data["outer_radius"], dtype=float)
        inner_radius = np.asarray(data["inner_radius"], dtype=float)
        assert outer_radius[-1] / inner_radius[-1] < 10.0
        assert np.isclose(np.sum(data["wall_power_weights"]), 1.0)
