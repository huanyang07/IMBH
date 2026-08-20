from __future__ import annotations

import json

import numpy as np

import run_causal_inner_forward_quadratic_field_blind_validation_wp10c9d6c7c3b5c4f25cz as f25cz


_FROZEN = None
_INPUTS = None


def _frozen():
    global _FROZEN
    if _FROZEN is None:
        _FROZEN = f25cz._validate_manifest(require_clean=False)
    return _FROZEN


def _inputs():
    global _INPUTS
    if _INPUTS is None:
        _INPUTS = f25cz._load_inputs(_frozen())
    return _INPUTS


def test_geometry_pass_authorizes_four_frozen_truth_calls():
    frozen = _frozen()
    gates = frozen["contract"]["blind_rate_validation"]
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authorized_next"] == f25cz.WORK_PACKAGE
    assert gates["count"] == 4
    assert gates["completed_exact_rate_calls_equal"] == 4
    assert frozen["geometry"]["candidate_primitive_states"].shape == (4, 112, 5)


def test_inputs_are_exactly_the_geometry_certificate():
    frozen = _frozen()
    inputs = _inputs()
    assert np.array_equal(
        inputs["states"], frozen["geometry"]["candidate_primitive_states"]
    )
    assert np.array_equal(inputs["partition_weights"], np.ones(4))
    assert np.max(inputs["coordinate_roundtrip_relative_errors"]) <= 1.0e-8


def test_online_prediction_uses_frozen_jacobian_transport():
    predicted, wall = f25cz._online_prediction_without_coordinate_jacobian(
        _inputs(), 0
    )
    assert predicted["full_rate"].shape == (560,)
    assert predicted["coordinate_rate"].shape == (470,)
    assert predicted["q162_Jacobian"].shape == (162, 560)
    assert predicted["decoded_state"].shape == (112, 5)
    assert wall >= 0.0


def test_progress_contract_is_resumable_and_exact():
    assert f25cz._progress_array_shapes() == f25cz.exact_parent._progress_array_shapes()
    identity = f25cz._progress_identity()
    assert identity["work_package"] == f25cz.WORK_PACKAGE
    assert identity["manifest_commit"] == f25cz.MANIFEST_COMMIT
    assert identity["geometry_arrays_sha256"] == f25cz._sha(f25cz.GEOMETRY_ARRAYS)


def test_canonical_validation_if_present():
    if not f25cz.CANONICAL_DIRECTORY.exists():
        return
    f25cz._checksums(f25cz.CANONICAL_DIRECTORY)
    summary = json.loads(
        (f25cz.CANONICAL_DIRECTORY / "summary.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (f25cz.CANONICAL_DIRECTORY / "validation_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    arrays = f25cz._load_npz(
        f25cz.CANONICAL_DIRECTORY / "validation_arrays.npz"
    )
    assert summary["classification"] in (
        f25cz.PASS_CLASSIFICATION,
        f25cz.FAIL_CLASSIFICATION,
    )
    assert summary["completed_exact_rate_calls"] <= 4
    assert not summary["coefficients_refit_after_holdout_truth"]
    assert arrays["holdout_primitive_states"].shape[0] <= 4
    if summary["passed"]:
        assert summary["classification"] == f25cz.PASS_CLASSIFICATION
        assert summary["authorized_next"] == f25cz.PASS_AUTHORIZED_NEXT
        assert metrics["passed"] and all(metrics["checks"].values())
