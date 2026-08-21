from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_affine_phase_chart_gluing_manifest_wp10c9d6c7c3b5c4f25e4 as manifest


def test_affine_transition_decoder_includes_entry_hidden_state() -> None:
    corrected = manifest._contract()["corrected_gluing"]
    assert "h_entry_absolute" in corrected["transition_decoder"]
    assert corrected["event_reset"] == "q_plus=q_minus_and_s_plus=0"
    assert corrected["no_fitted_event_jump"]


def test_repair_changes_no_gate_or_truth_budget() -> None:
    scope = manifest._contract()["scope"]
    assert scope["coordinate_origin_repair_only"]
    assert scope["no_gate_relaxed"]
    assert scope["new_truth_calls"] == 0
