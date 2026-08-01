from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "scripts/"
    "run_causal_inner_nonlinear_spatial_export_manifest_"
    "wp10c9d6c7c3b2a.py"
)
SUMMARY = (
    ROOT
    / "results/canonical/"
    "causal_inner_nonlinear_spatial_export_manifest_"
    "wp10c9d6c7c3b2a/summary.json"
)
MANIFEST = SUMMARY.parent / "nonlinear_spatial_export_manifest.json"
ARRAYS = SUMMARY.parent / "decisive_arrays.npz"
CHECKSUMS = SUMMARY.parent / "SHA256SUMS.txt"


def _runner():
    spec = importlib.util.spec_from_file_location("c3b2a_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_freezes_complete_four_step_matrix():
    module = _runner()
    assert len(module.LAYOUTS) == 3
    assert len(module.PROFILES) == 4
    assert len(module.VARIANT_MULTIPLIERS) == 4
    assert module.STEP_COUNT == 4
    np.testing.assert_array_equal(
        module.OUTPUT_TIMES_SECONDS,
        np.arange(5) * 1.0e-5,
    )
    assert module.COUPLING_PARENT_FACE == 48
    assert len(module.OBSERVABLE_NAMES) == 13


def test_manifest_preserves_tier_I_gates_and_tier_II_scope():
    module = _runner()
    audit = {
        "passed": True,
        "case_count": 48,
        "saved_time_count": 5,
        "maximum_step_continuity_defect": 0.0,
    }
    manifest = module._manifest(audit)
    gates = manifest["tier_I_binding_contract"]["gates"]
    assert gates["minimum_rms_order"] == 0.75
    assert gates["minimum_maximum_order"] == 0.75
    assert gates["minimum_significant_component_order"] == 0.75
    assert gates["maximum_fine_normalized_difference"] == 0.05
    assert gates["minimum_history_cosine"] == 0.90
    assert gates["minimum_refinement_error_cosine"] == 0.90
    assert manifest["tier_II_contract"]["may_not_rescue_or_fail_tier_I"]
    assert not manifest["interpretation_limits"][
        "temporal_convergence_certified"
    ]
    assert not manifest["interpretation_limits"][
        "nonlinear_physical_ladder_authorized"
    ]


def test_canonical_manifest_evidence_if_present():
    if not SUMMARY.exists():
        pytest.skip("canonical c3b2a evidence has not been generated")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert summary["work_package"] == "WP10c9d6c7c3b2a"
    assert summary["passed"]
    assert not summary["propagation_executed"]
    assert summary["input_audit"]["case_count"] == 48
    assert summary["input_audit"]["saved_time_count"] == 5
    assert summary["input_audit"][
        "all_step_boundaries_bitwise_continuous"
    ]
    assert summary["short_horizon_spatial_export_pilot_authorized"]
    assert not summary["temporal_convergence_certified"]
    assert not summary["long_nonlinear_physical_ladder_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    assert manifest["classification"] == (
        "nonlinear_short_horizon_spatial_export_manifest_frozen_"
        "canonical_response_pilot_authorized"
    )


def test_canonical_decisive_arrays_if_present():
    if not ARRAYS.exists():
        pytest.skip("canonical c3b2a arrays have not been generated")
    with np.load(ARRAYS, allow_pickle=False) as arrays:
        np.testing.assert_array_equal(
            arrays["output_times_seconds"],
            np.arange(5) * 1.0e-5,
        )
        np.testing.assert_array_equal(
            arrays["inner_refinement_ratios"],
            np.asarray((1, 2, 4)),
        )
        assert arrays["field_scales"].shape == (5,)
        assert arrays["fixed_physical_observable_scales"].shape == (13,)
        assert arrays["step_continuity_defects"].shape == (3, 4, 4, 3)
        assert np.max(arrays["step_continuity_defects"]) == 0.0


def test_canonical_checksums_if_present():
    if not CHECKSUMS.exists():
        pytest.skip("canonical c3b2a checksums have not been generated")
    for line in CHECKSUMS.read_text(encoding="utf-8").splitlines():
        digest, filename = line.split("  ", 1)
        assert _sha256(CHECKSUMS.parent / filename) == digest
