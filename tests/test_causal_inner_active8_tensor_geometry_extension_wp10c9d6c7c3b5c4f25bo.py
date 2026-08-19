from __future__ import annotations

import numpy as np

import run_causal_inner_active8_tensor_geometry_extension_wp10c9d6c7c3b5c4f25bo as f25bo


def test_frozen_manifest_authorizes_only_geometry_extension():
    frozen = f25bo._validate_manifest(require_clean=False)
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authorized_next"] == f25bo.WORK_PACKAGE
    assert not frozen["contract"]["claim_boundary"]["trajectory_authorized"]


def test_candidate_specifications_match_the_frozen_extension_split():
    specifications = f25bo._candidate_specifications()
    assert len(specifications) == 96
    counts = {
        split: sum(item["split"] == split for item in specifications)
        for split in ("training", "tuning_high", "holdout", "tuning_low")
    }
    assert counts == {
        "training": 64,
        "tuning_high": 8,
        "holdout": 16,
        "tuning_low": 8,
    }
    assert all(
        np.isclose(np.linalg.norm(item["active_direction"]), 1.0)
        for item in specifications
    )
    tuning_high = [
        item["active_direction"]
        for item in specifications
        if item["split"] == "tuning_high"
    ]
    tuning_low = [
        item["active_direction"]
        for item in specifications
        if item["split"] == "tuning_low"
    ]
    assert np.array_equal(np.asarray(tuning_high), np.asarray(tuning_low))


def test_certified_retraction_policy_is_reused_without_rate_calls():
    contract = f25bo._retraction_contract()
    inherited = contract["exact_geometric_retraction"]
    assert contract["binding_preflight_gates"] is contract["binding_geometry_gates"]
    assert inherited["line_factors"] == [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
    assert inherited["maximum_Newton_iterations"] == 8
    assert inherited["maximum_radius_rescalings"] == 4
    assert not inherited["rate_reaction_lift_used"]


def test_fresh_engine_is_isolated_and_uses_extension_identity():
    engine = f25bo._fresh_engine()
    assert engine.manifest.CANONICAL_DIRECTORY == f25bo.manifest.CANONICAL_DIRECTORY
    assert engine.manifest.ARTIFACT_DIRECTORY == f25bo.manifest.CANONICAL_DIRECTORY
    assert engine.WORK_PACKAGE == f25bo.WORK_PACKAGE
    assert engine.SCRATCH_DIRECTORY == f25bo.SCRATCH_DIRECTORY
    assert engine._candidate_specifications is f25bo._candidate_specifications


def test_synthetic_complete_metrics_pass_all_geometry_gates():
    gates = f25bo.manifest._contract()["binding_geometry_gates"]
    metrics = {
        "completed_candidate_count": 192,
        "failed_candidate_count": 0,
        "maximum_coordinate_residual_infinity": 0.5e-10,
        "maximum_normalized_Q3_defect": 0.5e-10,
        "maximum_final_scaled_component": 1.0e-2,
        "maximum_component_bound_fraction": 1.0,
        "minimum_reconstruction_factor": 1.0,
        "maximum_reconstruction_factor": 1.0,
        "maximum_coordinate_Jacobian_condition_number": 2.0e3,
        "minimum_departure_direction_alignment_cosine": 0.999,
        "maximum_departure_transverse_fraction": 0.02,
        "maximum_pair_coordinate_odd_symmetry_defect": 0.01,
        "maximum_H_over_R": 0.10,
        "minimum_scattering_optical_depth": 10.0,
        "nonbase_continuous_rate_evaluations": 0,
        "new_full_generator_assemblies": 0,
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
    }
    assert all(f25bo._gate_checks(metrics, gates).values())
