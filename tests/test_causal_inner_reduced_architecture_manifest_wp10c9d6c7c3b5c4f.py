from __future__ import annotations

import hashlib
import json

import run_causal_inner_reduced_architecture_manifest_wp10c9d6c7c3b5c4f as c4f


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_preserves_certificate_and_authorizes_analysis_only():
    manifest = c4f._manifest()
    assert manifest["definitions_only"]
    assert not manifest["propagation_executed"]
    assert manifest["certified_parent_scope"]["state_and_extraction_response_certificate"]
    assert manifest["certified_parent_scope"]["raw_pointwise_horizon_face_flux_rejected"]
    assert manifest["cost_contract"]["new_nonlinear_trajectories"] == 0
    assert "analysis_only" in manifest["authorized_next"]


def test_slow_coordinates_are_single_valued_and_nested():
    candidates = c4f._manifest()["slow_coordinate_candidates"]
    assert candidates["Q3"]["mapped_storage_rows"] == (0, 2, 3)
    assert candidates["Q3"]["responsive_height_one_form_excluded"]
    assert candidates["Q4"]["extends"] == "Q3"
    assert candidates["Q5"]["extends"] == "Q4"
    assert candidates["Q5"]["mapped_storage_row"] == 4
    assert candidates["staged_screen_order"] == ("Q3", "Q4", "Q5")
    assert candidates["first_analysis_package_tests_Q3_only"]
    assert candidates["post_result_coordinate_addition_forbidden"]


def test_augmented_state_retains_every_bdf_memory_component():
    augmented = c4f._manifest()["augmented_discrete_state"]
    assert augmented["required"]
    assert set(augmented["components"]) == {
        "current_primitive_state",
        "previous_primitive_increment",
        "previous_mapped_storage_increment",
        "previous_responsive_height_storage_increment",
        "previous_timestep_seconds",
    }
    assert augmented["primitive_snapshot_only_analysis_forbidden"]
    assert augmented["responsive_height_history_retained_because_one_form_is_nonexact"]


def test_kinematic_screen_is_not_mislabeled_as_fixed_q_dynamics():
    manifest = c4f._manifest()
    screen = manifest["equal_Q_screen"]
    constraint = manifest["future_fixed_Q_constraint"]
    assert screen["may_not_be_called_a_fixed_Q_attractor_test"]
    assert screen["measure_slow_leakage_without_reprojecting_each_step"]
    assert screen["per_step_reprojection_forbidden"]
    assert constraint["not_authorized_in_this_package"]
    assert constraint["B_Q_equals_DQ_transpose_may_not_be_assumed"]
    assert constraint["manual_primitive_freezing_forbidden"]


def test_memory_and_fine_complement_gates_are_prospective():
    manifest = c4f._manifest()
    memory = manifest["observable_memory_analysis"]
    fine = manifest["fine_complement_contract"]
    assert memory["minimum_observable_energy_capture"] == 0.99
    assert memory["compact_retained_mode_limit"] == 3
    assert memory["minimum_cross_resolution_subspace_cosine"] == 0.90
    assert memory["dense_full_propagator_or_dense_Gramian_forbidden"]
    assert fine["maximum_discarded_output_fraction"] == 0.01
    assert fine["maximum_fraction_of_middle_fine_spatial_difference"] == 0.10


def test_canonical_summary_retains_all_hard_stops():
    summary = _read(c4f.SUMMARY_PATH)
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["analysis_only_memory_screen_authorized"]
    assert not summary["new_trajectory_authorized"]
    assert not summary["fifty_ms_manifest_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    assert not summary["physical_failure_detected"]


def test_canonical_hashes_close():
    for line in (c4f.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        assert _sha256(c4f.CANONICAL_DIRECTORY / name) == expected
