from __future__ import annotations

import json

import numpy as np

import run_causal_inner_recenter_transition_validation_wp10c9d6c7c3b5c4f25cq as f25cq


def test_forecast_manifest_authorizes_exactly_two_roots():
    frozen = f25cq._validate_manifest(require_clean=False)
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authorized_next"] == f25cq.WORK_PACKAGE
    assert frozen["summary"]["prospective_truth_root_budget"] == 2
    assert frozen["forecast_metrics"]["predicted_trigger_root_index"] == 2
    assert all(frozen["forecast_metrics"]["checks"].values())


def test_warm_root_policy_carries_matrix_and_allows_one_refresh():
    policy = f25cq._root_policy()
    assert policy["order"] == 2
    assert policy["timestep_seconds"] == 1.0e-7
    assert policy["initial_exact_jacobian_required"] is False
    assert policy["maximum_exact_jacobian_refreshes"] == 1
    assert policy["exact_jacobian_refresh_policy"] == (
        "on_line_search_failure_or_iteration_reserve"
    )
    assert policy["residual_tolerance"] == 1.0e-10


def test_forecast_arrays_have_two_frozen_endpoints():
    arrays = f25cq._load_npz(f25cq.manifest.CANONICAL_DIRECTORY / "forecast.npz")
    assert arrays["warm4_truth_coordinate"].shape == (470,)
    assert arrays["prospective_refined_coordinates"].shape == (2, 470)
    assert arrays["predicted_transition_center_coordinate"].shape == (470,)
    assert np.array_equal(
        arrays["predicted_transition_center_coordinate"],
        arrays["prospective_refined_coordinates"][-1],
    )


def test_execution_gate_requires_both_roots_and_trigger_bracket():
    gates = f25cq.manifest._contract()["binding_execution_gates"]

    def record(load: float) -> dict:
        return {
            "accepted": True,
            "maximum_scaled_residual": 1.0e-11,
            "maximum_Q3_relative_defect": 1.0e-13,
            "minimum_path_reconstruction_factor": 1.0,
            "maximum_H_over_R": 0.1,
            "minimum_scattering_optical_depth": 10.0,
            "exact_Jacobian_assemblies": 1,
            "checkpoint_bitwise_roundtrip": True,
            "forecast_endpoint_coordinate_relative_errors": {
                "full": 0.01,
                "q162": 0.01,
                "z280": 0.01,
                "a28": 0.01,
            },
            "exact_scaled_state_load": load,
        }

    records = [record(0.01), record(0.013)]
    assert all(f25cq._execution_checks(records, 0.0, gates).values())
    records[1]["exact_scaled_state_load"] = 0.011
    assert not f25cq._execution_checks(records, 0.0, gates)["trigger_bracket"]
    assert not all(f25cq._execution_checks(records[:1], 0.0, gates).values())


def test_translation_roundtrip_uses_floating_point_bound():
    points = np.asarray(((1.0, 2.0), (3.0, 4.0), (-2.0, 5.0)))
    defect = f25cq._translation_defect(points, points[-1])
    assert defect <= 1.0e-14


def test_canonical_result_if_present():
    if not f25cq.CANONICAL_DIRECTORY.exists():
        return
    f25cq._checksums(f25cq.CANONICAL_DIRECTORY)
    summary = json.loads(
        (f25cq.CANONICAL_DIRECTORY / "summary.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (f25cq.CANONICAL_DIRECTORY / "validation_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["classification"] in {
        f25cq.PASS_CLASSIFICATION,
        f25cq.FAIL_CLASSIFICATION,
    }
    assert summary["passed"] == metrics["passed"]
    assert not summary["predicted_center_used_as_center"]
    assert not summary["predictive_cycle_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
