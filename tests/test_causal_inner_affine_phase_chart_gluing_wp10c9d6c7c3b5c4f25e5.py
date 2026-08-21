from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_affine_phase_chart_gluing_wp10c9d6c7c3b5c4f25e5 as gluing


def test_affine_engine_entry_is_absolute() -> None:
    engine, data, arrays = gluing._build_affine_engine()
    transition = engine.modes["fixed_Q_transition_observed"]
    assert np.array_equal(
        transition.hidden_origin,
        arrays["transition_entry_absolute_hidden388"]
        + gluing.manifest.rejected._build_engine()[0].modes[
            "fixed_Q_transition_observed"
        ].hidden_origin,
    )
    assert np.array_equal(data["transition_coordinates_absolute"][0], data["coordinates"][-1])


def test_pass_is_scoped_and_cycle_remains_missing() -> None:
    assert "working_on_observed_modes" in gluing.PASS_CLASSIFICATION
    assert "complete_cycle_calibration_missing" in gluing.PASS_CLASSIFICATION
