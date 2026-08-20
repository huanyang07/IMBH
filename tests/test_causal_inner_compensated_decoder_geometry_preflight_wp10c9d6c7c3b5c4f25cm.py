from __future__ import annotations

import json

import numpy as np
import pytest

import run_causal_inner_compensated_decoder_geometry_preflight_wp10c9d6c7c3b5c4f25cm as f25cm


@pytest.fixture(scope="module")
def decoder_fixture():
    frozen = f25cm._validate_manifest(require_clean=False)
    model = f25cm.manifest.parent.vector_field.ReducedVectorField()
    old_extension = f25cm._load_npz(f25cm.manifest.OLD_EXTENSION)
    revealed = f25cm._load_npz(f25cm.manifest.FAILED_RATE_ARRAYS)
    return frozen, model, old_extension, revealed


def test_manifest_authorizes_only_independent_geometry(decoder_fixture):
    frozen, _model, _old, _revealed = decoder_fixture
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authorized_next"] == f25cm.WORK_PACKAGE
    assert frozen["directions"].shape == (4, 28)
    assert np.array_equal(frozen["bounds"], np.asarray((0.0125, 0.015)))


def test_compensated_decoder_preserves_rate_on_revealed_state(decoder_fixture):
    frozen, model, old_extension, revealed = decoder_fixture
    metrics, arrays = f25cm._evaluate_decoder(
        model,
        old_extension,
        frozen["repair"],
        revealed["candidate_scaled_deltas"][0],
        revealed["candidate_departure_coordinates"][0],
    )
    assert arrays["repaired_delta"].shape == (560,)
    assert arrays["compensated_full_state_rate"].shape == (560,)
    assert metrics["repaired_decoder_full_state_relative_error"] < 1.0e-6
    assert metrics["compensated_full_rate_invariance_defect"] < 1.0e-12


def test_rung_gate_binds_decoder_and_rate_invariance():
    gates = f25cm.manifest._contract()["future_independent_geometry_gates"]
    metrics = {
        "completed_candidate_count": 4,
        "failed_candidate_count": 0,
        "maximum_coordinate_residual_infinity": 0.0,
        "maximum_normalized_Q3_defect": 0.0,
        "maximum_final_scaled_component": 0.0125,
        "minimum_reconstruction_factor": 1.0,
        "maximum_reconstruction_factor": 1.0,
        "maximum_coordinate_Jacobian_condition_number": 1.0,
        "minimum_departure_direction_alignment_cosine": 1.0,
        "maximum_departure_transverse_fraction": 0.0,
        "maximum_H_over_R": 0.1,
        "minimum_scattering_optical_depth": 10.0,
        "maximum_repaired_decoder_full_state_relative_error": 0.005,
        "maximum_repaired_decoder_coordinate_relative_mismatch": 0.005,
        "maximum_compensated_full_rate_invariance_defect": 1.0e-12,
        "minimum_repaired_reconstruction_factor": 1.0,
        "maximum_repaired_H_over_R": 0.1,
        "minimum_repaired_scattering_optical_depth": 10.0,
    }
    assert all(f25cm._gate_checks(metrics, gates, 0.0125).values())
    metrics["maximum_repaired_decoder_full_state_relative_error"] = 0.005001
    assert not f25cm._gate_checks(metrics, gates, 0.0125)["decoder_error"]


def test_classification_requires_both_rungs():
    full = f25cm._classify({"passing_rung_count": 2})
    partial = f25cm._classify({"passing_rung_count": 1})
    failed = f25cm._classify({"passing_rung_count": 0})
    assert full["passed"] and full["classification"] == f25cm.FULL_CLASSIFICATION
    assert not partial["passed"]
    assert partial["classification"] == f25cm.PARTIAL_CLASSIFICATION
    assert not failed["passed"]
    assert failed["classification"] == f25cm.FAIL_CLASSIFICATION


def test_canonical_geometry_if_present():
    if not f25cm.CANONICAL_DIRECTORY.exists():
        return
    f25cm._checksums(f25cm.CANONICAL_DIRECTORY)
    summary = json.loads(
        (f25cm.CANONICAL_DIRECTORY / "summary.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (f25cm.CANONICAL_DIRECTORY / "geometry_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["classification"] in {
        f25cm.FULL_CLASSIFICATION,
        f25cm.PARTIAL_CLASSIFICATION,
        f25cm.FAIL_CLASSIFICATION,
    }
    assert summary["passed"] == (metrics["passing_rung_count"] == 2)
    assert not summary["geometry_candidate_became_atlas_center"]
    assert not summary["trajectory_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
