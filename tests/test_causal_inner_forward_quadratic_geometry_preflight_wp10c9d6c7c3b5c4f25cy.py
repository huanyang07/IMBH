from __future__ import annotations

import json

import numpy as np

import run_causal_inner_forward_quadratic_geometry_preflight_wp10c9d6c7c3b5c4f25cy as f25cy


_FROZEN = None


def _frozen():
    global _FROZEN
    if _FROZEN is None:
        _FROZEN = f25cy._validate_manifest(require_clean=False)
    return _FROZEN


def test_manifest_authorizes_geometry_without_truth():
    frozen = _frozen()
    gates = frozen["contract"]["geometry_preflight"]
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authorized_next"] == f25cy.WORK_PACKAGE
    assert gates["count"] == 4
    assert gates["new_exact_rate_calls_equal"] == 0
    assert gates["directions_may_not_change"]
    assert gates["coefficients_may_not_change"]


def test_candidate_specs_are_exactly_the_frozen_directions():
    closure = _frozen()["closure"]
    specs = f25cy._candidate_specs(closure)
    assert len(specs) == 4
    assert np.array_equal(
        np.asarray([item["direction"] for item in specs]),
        closure["blind_directions"],
    )
    assert np.array_equal(
        np.asarray([item["component_bound"] for item in specs]),
        closure["blind_component_bounds"],
    )
    assert np.array_equal(
        np.asarray([item["mixing_magnitude"] for item in specs]),
        closure["blind_mixing_magnitudes"],
    )


def test_geometry_checks_fail_closed():
    gates = _frozen()["contract"]["geometry_preflight"]
    metrics = {
        "completed_candidate_count": 4,
        "failed_candidate_count": 0,
        "minimum_partition_weight": 1.0,
        "minimum_forward_active_coordinate": 1.3,
        "maximum_decoder_relative_error": 0.01,
        "maximum_decoder_coordinate_relative_mismatch": 0.01,
        "minimum_reconstruction_factor": 1.0,
        "maximum_H_over_R": 0.1,
        "minimum_scattering_optical_depth": 10.0,
        "minimum_predicted_q162_Jacobian_rank": 162,
        "maximum_predicted_q162_Jacobian_condition_number": 2000.0,
        "new_exact_rate_calls": 0,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_fixed_Q_roots": 0,
        "propagated_states": 0,
    }
    assert all(f25cy._checks(metrics, gates).values())
    metrics["minimum_partition_weight"] = np.nextafter(1.0, 0.0)
    assert not f25cy._checks(metrics, gates)["partition"]


def test_canonical_geometry_if_present():
    if not f25cy.CANONICAL_DIRECTORY.exists():
        return
    f25cy._checksums(f25cy.CANONICAL_DIRECTORY)
    summary = json.loads(
        (f25cy.CANONICAL_DIRECTORY / "summary.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (f25cy.CANONICAL_DIRECTORY / "geometry_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    arrays = f25cy._load_npz(f25cy.CANONICAL_DIRECTORY / "geometry_arrays.npz")
    assert summary["classification"] in (
        f25cy.PASS_CLASSIFICATION,
        f25cy.FAIL_CLASSIFICATION,
    )
    assert summary["new_exact_rate_calls"] == 0
    assert not summary["directions_changed_after_manifest"]
    assert not summary["coefficients_changed_after_manifest"]
    assert arrays["candidate_primitive_states"].shape[0] <= 4
    if summary["passed"]:
        assert summary["classification"] == f25cy.PASS_CLASSIFICATION
        assert summary["authorized_next"] == f25cy.PASS_AUTHORIZED_NEXT
        assert metrics["passed"] and all(metrics["checks"].values())
        assert arrays["candidate_primitive_states"].shape == (4, 112, 5)
