from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for path in (SCRIPTS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_anchor_excision_audit_wp10c8w as wp10c8w


def _generator_arrays(rate_per_s: float) -> dict[str, np.ndarray]:
    return {
        "generator": np.eye(5) * float(rate_per_s),
        "primitive_column_scales": np.ones((1, 5)),
        "physical_input_amplitudes": np.ones((1, 5)),
    }


def test_propagation_safety_rejects_explosive_generator() -> None:
    stable = wp10c8w._generator_propagation_safety(
        _generator_arrays(-2.0)
    )
    explosive = wp10c8w._generator_propagation_safety(
        _generator_arrays(100.0)
    )
    assert stable["passed"] is True
    assert stable["target_growth_exponent"] < 0.0
    assert explosive["passed"] is False
    assert explosive["target_growth_exponent"] > (
        wp10c8w.MAXIMUM_PROPAGATION_GROWTH_EXPONENT
    )


def test_committed_machine_evidence_keeps_architecture_blocked() -> None:
    if not wp10c8w.DEFAULT_OUTPUT.exists():
        return
    payload = json.loads(
        wp10c8w.DEFAULT_OUTPUT.read_text(encoding="utf-8")
    )
    assert payload["work_package"] == "WP10c8w"
    assert payload["classification"] == (
        "independent_anchor_passed_excision_or_phase_unresolved"
    )
    assert payload["decision"] == {
        "anchor_consistency_passed": True,
        "excision_placement_insensitivity_passed": False,
        "fixed_q_averaging_authorized": False,
        "n512_or_embedded_patch_authorized": False,
        "production_boundary_replacement_authorized": False,
        "selected_trace_refinement_passed": False,
    }
    assert payload["anchor_consistency"]["N256"]["projection"]["passed"]
    assert payload["trace_screen"]["selected_trace"] == "linear_outgoing"
    assert not payload["trace_screen"]["cross_mesh_exterior_metrics"][
        "cell_centered"
    ]["available"]
    assert payload["selected_trace_refinement"][
        "state_observed_order"
    ] < wp10c8w.MINIMUM_SPATIAL_ORDER
    assert payload["selected_trace_refinement"][
        "rate_observed_order"
    ] < wp10c8w.MINIMUM_SPATIAL_ORDER
    assert payload["scope"]["formal_fast_average_certified"] is False
    assert payload["scope"]["reduced_architecture_selected"] is False
