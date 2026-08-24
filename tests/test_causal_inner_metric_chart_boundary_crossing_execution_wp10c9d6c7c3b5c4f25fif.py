from __future__ import annotations

import run_causal_inner_metric_chart_boundary_crossing_execution_wp10c9d6c7c3b5c4f25fif as target


def test_frozen_manifest_authorizes_exactly_one_crossing() -> None:
    lock = target._validate_manifest(require_clean=False)
    assert lock["summary"]["classification"] == target.manifest.CLASSIFICATION
    assert lock["contract"]["scope"]["new_accepted_segments_maximum"] == 1
    assert lock["contract"]["scope"]["new_exact_free_field_calls"] == 2


def test_metric_physical_gate_does_not_bind_historical_raw_condition() -> None:
    free = {
        "coordinate_jacobian_rank": 470,
        "coordinate_jacobian_condition_number": 3000.0,
        "coordinate_reconstruction_relative_defect": 1.0e-14,
        "minimum_reconstruction_factor": 1.0,
        "maximum_height_ratio": 0.1,
        "minimum_scattering_optical_depth": 19.0,
        "reaction_free_ledger_passed": True,
    }
    retraction = {
        "passed": True,
        "original_coordinate_residual_infinity": 1.0e-13,
        "metric_coordinate_residual_infinity": 1.0e-13,
        "gauge_residual_infinity": 1.0e-14,
        "maximum_metric_augmented_condition_number": 2.0,
    }
    metric = {
        "metric_jacobian_condition_number": 2.0,
        "metric_augmented_condition_number": 2.0,
        "patch_transition_condition_number": 1.1,
        "transform_inverse_closure_defect": 1.0e-14,
        "raw_condition_reproduction_relative_defect": 1.0e-15,
    }
    assert target._field_physical_passed(free, retraction, metric)


def test_metric_physical_gate_keeps_rank_and_original_physics_binding() -> None:
    free = {
        "coordinate_jacobian_rank": 469,
        "coordinate_jacobian_condition_number": 2.0,
        "coordinate_reconstruction_relative_defect": 1.0e-14,
        "minimum_reconstruction_factor": 1.0,
        "maximum_height_ratio": 0.1,
        "minimum_scattering_optical_depth": 19.0,
        "reaction_free_ledger_passed": True,
    }
    retraction = {
        "passed": True,
        "original_coordinate_residual_infinity": 1.0e-13,
        "metric_coordinate_residual_infinity": 1.0e-13,
        "gauge_residual_infinity": 1.0e-14,
        "maximum_metric_augmented_condition_number": 2.0,
    }
    metric = {
        "metric_jacobian_condition_number": 2.0,
        "metric_augmented_condition_number": 2.0,
        "patch_transition_condition_number": 1.1,
        "transform_inverse_closure_defect": 1.0e-14,
        "raw_condition_reproduction_relative_defect": 1.0e-15,
    }
    assert not target._field_physical_passed(free, retraction, metric)
    free["coordinate_jacobian_rank"] = 470
    free["reaction_free_ledger_passed"] = False
    assert not target._field_physical_passed(free, retraction, metric)
