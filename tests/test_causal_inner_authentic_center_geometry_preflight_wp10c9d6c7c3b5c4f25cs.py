from __future__ import annotations

import json

import numpy as np

import run_causal_inner_authentic_center_geometry_preflight_wp10c9d6c7c3b5c4f25cs as f25cs


def test_local_field_manifest_authorizes_geometry_only_preflight():
    frozen = f25cs._validate_manifest(require_clean=False)
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authorized_next"] == f25cs.WORK_PACKAGE
    assert frozen["contract"]["next_geometry_preflight_budget"] == {
        "candidate_count": 8,
        "new_continuous_rate_calls": 0,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_fixed_Q_roots": 0,
        "propagated_states": 0,
    }


def test_candidate_specs_preserve_training_before_holdout_order():
    design = f25cs._load_npz(f25cs.DESIGN_ARRAYS)
    specs = f25cs._candidate_specs(design)
    assert len(specs) == 8
    assert [spec["role"] for spec in specs] == ["training"] * 4 + [
        "holdout"
    ] * 4
    assert [spec["role_index"] for spec in specs] == [0, 1, 2, 3] * 2
    assert [spec["component_bound"] for spec in specs[:4]] == [0.0125] * 4
    assert [spec["component_bound"] for spec in specs[4:]] == [0.015] * 4


def test_local_coordinate_composes_exact_affine_translation():
    rng = np.random.default_rng(25)
    center = rng.normal(size=470)
    local = rng.normal(size=470) * 1.0e-3
    absolute = center + local
    restored = absolute - center
    assert np.max(np.abs(restored - local)) <= 5.0e-16


def test_classification_is_fail_closed_and_role_ordered():
    base = {
        "new_continuous_rate_calls": 0,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_fixed_Q_roots": 0,
        "propagated_states": 0,
    }
    full = f25cs._classify({"passing_role_count": 2, **base})
    assert full["classification"] == f25cs.FULL_CLASSIFICATION
    assert full["authorized_next"] == f25cs.FULL_AUTHORIZED_NEXT
    partial = f25cs._classify({"passing_role_count": 1, **base})
    assert partial["classification"] == f25cs.TRAINING_ONLY_CLASSIFICATION
    failed = f25cs._classify({"passing_role_count": 0, **base})
    assert not failed["passed"]
    over_budget = f25cs._classify(
        {"passing_role_count": 2, **{**base, "new_continuous_rate_calls": 1}}
    )
    assert not over_budget["passed"]


def test_canonical_geometry_if_present():
    if not f25cs.CANONICAL_DIRECTORY.exists():
        return
    f25cs._checksums(f25cs.CANONICAL_DIRECTORY)
    summary = json.loads(
        (f25cs.CANONICAL_DIRECTORY / "summary.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (f25cs.CANONICAL_DIRECTORY / "geometry_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    arrays = f25cs._load_npz(
        f25cs.CANONICAL_DIRECTORY / "geometry_arrays.npz"
    )
    assert summary["classification"] in {
        f25cs.FULL_CLASSIFICATION,
        f25cs.TRAINING_ONLY_CLASSIFICATION,
        f25cs.FAIL_CLASSIFICATION,
    }
    assert summary["completed_candidate_count"] == metrics[
        "completed_candidate_count"
    ]
    assert arrays["candidate_primitive_states"].shape[0] == summary[
        "completed_candidate_count"
    ]
    assert arrays["candidate_local_coordinates"].shape[1:] == (470,)
    assert not summary["physical_microburst_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
