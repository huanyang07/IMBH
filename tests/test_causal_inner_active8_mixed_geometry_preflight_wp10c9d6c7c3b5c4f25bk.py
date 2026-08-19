from __future__ import annotations

import numpy as np

import run_causal_inner_active8_mixed_geometry_preflight_wp10c9d6c7c3b5c4f25bk as f25bk


def test_frozen_manifest_authorizes_only_geometry_preflight():
    frozen = f25bk._validate_manifest(require_clean=False)
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authorized_next"] == f25bk.WORK_PACKAGE
    assert frozen["contract"]["exact_geometry"]["propagated_states"] == 0


def test_candidate_specifications_match_the_frozen_split():
    specifications = f25bk._candidate_specifications()
    assert len(specifications) == 64
    counts = {
        split: sum(item["split"] == split for item in specifications)
        for split in ("training", "tuning_high", "holdout", "tuning_low")
    }
    assert counts == {
        "training": 40,
        "tuning_high": 8,
        "holdout": 8,
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


def test_synthetic_complete_metrics_pass_all_geometry_gates():
    gates = f25bk.manifest._contract()["binding_geometry_gates"]
    metrics = {
        "completed_candidate_count": 128,
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
    assert all(f25bk._gate_checks(metrics, gates).values())


def test_inherited_retraction_receives_legacy_gate_alias():
    contract = f25bk._retraction_contract()
    assert contract["binding_preflight_gates"] is contract["binding_geometry_gates"]


def test_progress_roundtrip_is_lossless(tmp_path, monkeypatch):
    monkeypatch.setattr(f25bk, "SCRATCH_DIRECTORY", tmp_path / "scratch")
    monkeypatch.setattr(f25bk, "_progress_identity", lambda: {"identity": "test"})
    progress = f25bk._empty_progress({"identity": "test"})
    f25bk._save_progress(progress)
    loaded = f25bk._load_or_create_progress()
    assert loaded["identity"] == progress["identity"]
    assert loaded["candidates"] == []
    assert loaded["candidate_scaled_deltas"].shape == (0, 560)
