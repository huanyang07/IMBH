from __future__ import annotations

import numpy as np

import run_causal_inner_departure28_validation_geometry_wp10c9d6c7c3b5c4f25bw as f25bw


def test_manifest_authorizes_geometry_only():
    frozen = f25bw._validate_manifest(require_clean=False)
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authorized_next"] == f25bw.WORK_PACKAGE
    assert frozen["summary"]["new_truth_evaluations"] == 0
    assert frozen["contract"]["mathematical_architecture"][
        "departure_rate_input_dimension"
    ] == 28


def test_candidate_order_is_frozen_high_then_radial():
    candidates = f25bw._candidate_specifications()
    assert len(candidates) == 24
    assert [item["split"] for item in candidates[:16]] == ["holdout"] * 16
    assert [item["split"] for item in candidates[16:]] == ["tuning_low"] * 8
    assert [item["component_bound"] for item in candidates[:16]] == [0.01] * 16
    assert [item["component_bound"] for item in candidates[16:]] == [0.005] * 8
    assert all(np.isclose(np.linalg.norm(item["active_direction"]), 1.0) for item in candidates)


def test_geometry_adapter_uses_certified_engine_and_zero_rate_budget():
    engine = f25bw._fresh_engine()
    contract = f25bw._retraction_contract()
    assert engine.WORK_PACKAGE == f25bw.WORK_PACKAGE
    assert engine.SCRATCH_DIRECTORY == f25bw.SCRATCH_DIRECTORY
    engine_candidates = engine._candidate_specifications()
    local_candidates = f25bw._candidate_specifications()
    assert [item["split"] for item in engine_candidates] == [
        item["split"] for item in local_candidates
    ]
    assert np.array_equal(
        np.vstack([item["active_direction"] for item in engine_candidates]),
        np.vstack([item["active_direction"] for item in local_candidates]),
    )
    gates = contract["binding_geometry_gates"]
    assert gates["nonbase_continuous_rate_evaluations_equal"] == 0
    assert gates["new_full_generator_assemblies_equal"] == 0
    assert gates["new_nonlinear_roots_equal"] == 0


def test_geometry_gates_remain_fail_closed():
    gates = f25bw.manifest._contract()["binding_geometry_gates"]
    metrics = {
        "completed_candidate_count": 48,
        "failed_candidate_count": 0,
        "maximum_coordinate_residual_infinity": 9.0e-11,
        "maximum_normalized_Q3_defect": 9.0e-11,
        "maximum_final_scaled_component": 0.01,
        "maximum_component_bound_fraction": 1.0,
        "minimum_reconstruction_factor": 1.0,
        "maximum_reconstruction_factor": 1.0,
        "maximum_coordinate_Jacobian_condition_number": 2.0e3,
        "minimum_departure_direction_alignment_cosine": 0.999,
        "maximum_departure_transverse_fraction": 0.01,
        "maximum_pair_coordinate_odd_symmetry_defect": 0.001,
        "maximum_H_over_R": 0.10,
        "minimum_scattering_optical_depth": 10.0,
        "nonbase_continuous_rate_evaluations": 0,
        "new_full_generator_assemblies": 0,
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
    }
    assert all(f25bw._gate_checks(metrics, gates).values())
    metrics["failed_candidate_count"] = 1
    assert not all(f25bw._gate_checks(metrics, gates).values())
