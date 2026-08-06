from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_nonlinear_discrete_bdf_tangent_calibration_wp10c9d6c7c3b5c3h1 as runner


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_complete_discrete_tangent_calibration_passes() -> None:
    summary = _read(runner.SUMMARY_PATH)
    assert summary["passed"] is True
    assert summary["classification"] == (
        "complete_discrete_BDF_tangent_calibrated_middle_cost_bounded_"
        "anchor_manifest_authorized"
    )
    assert summary["analytic_complete_discrete_BDF_tangent_certified"] is True
    assert summary["middle_cost_bounded_anchor_manifest_authorized"] is True
    assert summary["middle_cost_bounded_propagation_authorized"] is False


def test_long_tail_state_export_and_jvp_gates_pass() -> None:
    summary = _read(runner.SUMMARY_PATH)
    long = summary["calibration"]["long"]
    gates = summary["gates"]
    assert long["profiles"] == list(runner.PROFILES)
    assert long["steps"] == 7
    assert long["state"]["maximum_scaled_discrepancy"] <= gates[
        "maximum_scaled_state_response_discrepancy"
    ]
    assert long["instantaneous_Tier_I"]["maximum_scaled_discrepancy"] <= gates[
        "maximum_scaled_Tier_I_response_discrepancy"
    ]
    assert long["windowed_cumulative_Tier_I"][
        "maximum_scaled_discrepancy"
    ] <= gates["maximum_scaled_Tier_I_response_discrepancy"]
    assert long["maximum_step_matrix_jvp_relative_defect"] <= gates[
        "maximum_internal_discrete_residual_jvp_relative_defect"
    ]
    assert long["maximum_export_transport_telescoping_defect"] <= gates[
        "maximum_export_transport_telescoping_defect"
    ]
    assert long["maximum_export_active_prefix_ledger_defect"] <= gates[
        "maximum_export_active_prefix_ledger_defect"
    ]


def test_short_layouts_use_corrected_coupling_face_reference() -> None:
    summary = _read(runner.SUMMARY_PATH)
    gates = summary["gates"]
    corrected = runner._load_npz(runner.b4d.DECISIVE_ARRAYS)
    decisive = runner._load_npz(runner.DECISIVE_ARRAYS)
    for layout in runner.LAYOUTS:
        report = summary["calibration"]["short"][layout]
        assert report["export_reference"] == (
            "WP10c9d6c7c3b4d_corrected_active_coupling_face"
        )
        expected = corrected[
            f"{layout}__{runner.GENERIC_PROFILE}__corrected_face_response"
        ][2]
        actual = decisive[f"{layout}__short_actual_Tier_I_response"]
        assert np.array_equal(actual, expected)
        assert report["state"]["maximum_scaled_discrepancy"] <= gates[
            "maximum_scaled_state_response_discrepancy"
        ]
        assert report["instantaneous_Tier_I"][
            "maximum_scaled_discrepancy"
        ] <= gates["maximum_scaled_Tier_I_response_discrepancy"]
        assert report["instantaneous_Tier_I"]["history_cosine"] >= gates[
            "minimum_Tier_I_response_history_cosine"
        ]


def test_cost_reduction_is_measured_without_new_physical_trajectory() -> None:
    summary = _read(runner.SUMMARY_PATH)
    assert summary["new_physical_trajectory_executed"] is False
    assert summary["operator_changed"] is False
    assert summary["production_defaults_changed"] is False
    for report in summary["calibration"]["short"].values():
        assert report["matrix_assembly_wall_seconds"] > 0.0
        assert report["block_step_wall_seconds"] < 1.0


def test_downstream_work_remains_blocked_and_hashes_close() -> None:
    summary = _read(runner.SUMMARY_PATH)
    assert summary["fine_cost_bounded_propagation_authorized"] is False
    assert summary["third_duration_rung_spatial_convergence_certified"] is False
    assert summary["fourth_duration_rung_manifest_authorized"] is False
    assert summary["fixed_q_micro_solver_authorized"] is False
    assert summary["reduced_slow_evolution_authorized"] is False
    for line in (runner.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        payload = (runner.CANONICAL_DIRECTORY / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == digest
