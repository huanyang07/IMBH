from __future__ import annotations

import json

import numpy as np

import run_causal_inner_certified_0p015_departure_rate_screen_wp10c9d6c7c3b5c4f25cf as f25cf


def test_manifest_authorizes_targeted_exact_rate_screen():
    frozen = f25cf._validate_manifest(require_clean=False)
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authorized_next"] == f25cf.WORK_PACKAGE
    assert tuple(frozen["design"]["parent_candidate_indices"]) == (8, 9, 12, 13, 16, 17, 18, 19)


def test_inputs_reuse_only_certified_geometry_and_frozen_model():
    frozen = f25cf._validate_manifest(require_clean=False)
    inputs = f25cf._load_inputs(frozen)
    assert inputs["states"].shape == (8, 112, 5)
    assert inputs["deltas"].shape == (8, 560)
    assert inputs["coordinates"].shape == (8, 28)
    assert inputs["generator"].shape == (560, 560)
    assert inputs["departure_basis"].shape == (560, 28)
    assert inputs["memory_basis"].shape == (560, 280)
    assert tuple(inputs["signs"]) == (-1, 1, -1, 1, -1, 1, -1, 1)


def test_progress_schema_is_restartable_and_complete():
    progress = f25cf._empty_progress()
    assert progress["evaluations"] == []
    assert progress["failures"] == []
    for name, shape in f25cf._progress_array_shapes().items():
        assert progress[name].shape == (0,) + shape


def test_forward_classification_is_prospective_and_model_aware():
    contract = f25cf.manifest._contract()
    classification, authorized, behavior = f25cf._classify(
        truth_passed=True,
        forward_cosine=0.2,
        old_field_supported=True,
        contract=contract,
    )
    assert classification == f25cf.OUTWARD_CLASSIFICATION
    assert behavior == "outward"
    assert authorized == "definitions_only_authentic_trajectory_recentered_chart_manifest"
    _, unsupported_authorized, _ = f25cf._classify(
        truth_passed=True,
        forward_cosine=0.2,
        old_field_supported=False,
        contract=contract,
    )
    assert unsupported_authorized == "definitions_only_local_rate_extension_and_recentered_chart_manifest"
    assert f25cf._classify(
        truth_passed=True,
        forward_cosine=-0.2,
        old_field_supported=True,
        contract=contract,
    )[0] == f25cf.INWARD_CLASSIFICATION
    assert f25cf._classify(
        truth_passed=True,
        forward_cosine=0.0,
        old_field_supported=True,
        contract=contract,
    )[0] == f25cf.TANGENTIAL_CLASSIFICATION


def test_canonical_screen_if_present():
    if not f25cf.CANONICAL_DIRECTORY.exists():
        return
    f25cf._checksums(f25cf.CANONICAL_DIRECTORY)
    summary = json.loads(
        (f25cf.CANONICAL_DIRECTORY / "summary.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (f25cf.CANONICAL_DIRECTORY / "rate_metrics.json").read_text(encoding="utf-8")
    )
    arrays = np.load(f25cf.CANONICAL_DIRECTORY / "rate_arrays.npz", allow_pickle=False)
    assert summary["passed"]
    assert summary["classification"] in {
        f25cf.OUTWARD_CLASSIFICATION,
        f25cf.INWARD_CLASSIFICATION,
        f25cf.TANGENTIAL_CLASSIFICATION,
    }
    assert all(metrics["truth_checks"].values())
    assert arrays["exact_departure_rates_per_second"].shape == (8, 28)
    assert arrays["online_470_coordinate_rates_per_second"].shape == (8, 470)
    assert not summary["physical_microburst_authorized"]
    assert not summary["predictive_cycle_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
