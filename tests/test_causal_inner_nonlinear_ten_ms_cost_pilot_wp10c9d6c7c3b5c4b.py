from __future__ import annotations

import json

import numpy as np

import run_causal_inner_nonlinear_ten_ms_cost_pilot_wp10c9d6c7c3b5c4b as c4b


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_pilot_uses_complete_5ms_restarts_and_one_comparison() -> None:
    summary = _read(c4b.SUMMARY_PATH)
    assert summary["passed"]
    assert set(summary["trajectory_reports"]) == {"base", "perturbed"}
    for report in summary["trajectory_reports"].values():
        assert report["passed"]
        assert report["method"]["accepted_comparisons"] == 1
        assert report["extraction_partition_audit"]["passed"]
        assert report["extraction_partition_audit"][
            "transport_sign_convention_diagnostic"
        ] == 2.0


def test_runtime_projection_is_advisory_below_48_hours() -> None:
    summary = _read(c4b.SUMMARY_PATH)
    runtime = summary["runtime_projection"]
    assert runtime["resource_projection_is_not_a_physical_gate"]
    expected = runtime["projected_wall_hours"] <= 48.0
    assert runtime["full_screen_resource_authorized"] is expected
    assert summary["ten_ms_screen_propagation_authorized"] is expected


def test_pilot_preserves_extraction_semantics_and_reduction_stops() -> None:
    summary = _read(c4b.SUMMARY_PATH)
    assert not summary["pointwise_horizon_flux_convergence_claimed"]
    assert summary["raw_inner_face_rejection_preserved"]
    assert not summary["twenty_ms_propagation_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_decisive_arrays_contain_extraction_partition_histories() -> None:
    with np.load(c4b.DECISIVE_ARRAYS, allow_pickle=False) as arrays:
        for name in ("base", "perturbed"):
            values = arrays[f"{name}__output_extraction_partition"]
            audits = arrays[f"{name}__extraction_partition_audits"]
            assert values.shape == (2, 13)
            assert audits.shape == (2, 7)
