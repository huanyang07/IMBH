from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_complete_cycle_asymptotic_path_diagnosis_wp10c9d6c7c3b5c4f25fg as target  # noqa: E402


def test_parent_is_exact_budget_exhaustion_result() -> None:
    locked = target._validate_parent(require_clean=False)
    assert locked["summary"]["classification"] == target.parent.BUDGET_CLASSIFICATION
    assert not locked["summary"]["complete_cycle_observed"]
    assert locked["metrics"]["gate_values"]["completed_patches"] == 64
    assert locked["metrics"]["gate_values"]["exact_free_field_witnesses"] == 192


def test_truth_field_audit_is_autonomous() -> None:
    audit = target._source_audit()
    assert audit["truth_field_autonomous"]
    assert not audit["external_clock_argument_present"]
    assert audit["rom_forcing_angular_frequency_zero"]


def test_cubic_hermite_reproduces_a_cubic_path() -> None:
    time = target.MACRO_STEP_SECONDS * np.arange(9, dtype=float)
    coordinates = np.column_stack((time, time**2, time**3))
    rates = np.column_stack((
        np.ones_like(time),
        2.0 * time,
        3.0 * time**2,
    ))
    metrics, arrays = target._hermite_replay(coordinates, rates, stride=8)
    assert metrics["passed"]
    assert metrics["maximum_coordinate_defect"] < 2.0e-13
    assert metrics["maximum_rate_defect"] < 2.0e-13
    assert len(arrays["coordinate_defects"]) == 7


def test_canonical_trajectory_selects_wide_open_transport() -> None:
    locked = target._validate_parent(require_clean=False)
    metrics, arrays = target._evaluate(locked)
    assert metrics["passed"]
    assert metrics["classification"] == target.CLASSIFICATION
    assert metrics["source_audit"]["truth_field_autonomous"]
    assert metrics["wide_arclength_transport_supported"]
    assert not metrics["cycle_closure_supported"]
    assert not metrics["equilibrium_closure_supported"]
    assert metrics["gate_values"]["maximum_validated_hermite_stride"] == 16
    assert metrics["gate_values"]["new_exact_free_field_calls"] == 0
    assert np.all(np.diff(arrays["section"]) > 0.0)
    assert np.all(arrays["section_velocity_per_second"] > 0.0)


def test_architecture_forbids_invented_phase_and_fixed_q_clock() -> None:
    locked = target._validate_parent(require_clean=False)
    metrics, _arrays = target._evaluate(locked)
    architecture = target._architecture(metrics)
    assert architecture["truth_system"]["autonomous"]
    assert not architecture["truth_system"]["external_forcing_phase_added"]
    assert architecture["path_system"]["maximum_prevalidated_span_seconds"] == 4.0e-3
    forbidden = architecture["forbidden_shortcuts"]
    assert "fixed-Q physical clock" in forbidden
    assert "invented periodic forcing" in forbidden
