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
    "run_causal_inner_nonlinear_temporal_refinement_manifest_"
    "wp10c9d6c7c3b3a.py"
)
SUMMARY = (
    ROOT
    / "results/canonical/"
    "causal_inner_nonlinear_temporal_refinement_manifest_"
    "wp10c9d6c7c3b3a/summary.json"
)
MANIFEST = SUMMARY.parent / "temporal_refinement_manifest.json"
ARRAYS = SUMMARY.parent / "decisive_arrays.npz"
CHECKSUMS = SUMMARY.parent / "SHA256SUMS.txt"


def _runner():
    spec = importlib.util.spec_from_file_location("c3b3a_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_temporal_manifest_freezes_nested_triplet_and_common_outputs():
    module = _runner()
    np.testing.assert_array_equal(
        module.TIMESTEP_LEVELS_SECONDS,
        np.asarray((1.0e-5, 5.0e-6, 2.5e-6)),
    )
    np.testing.assert_array_equal(
        module.STEP_COUNTS,
        np.asarray((4, 8, 16)),
    )
    np.testing.assert_array_equal(
        module.COMMON_OUTPUT_TIMES_SECONDS,
        np.arange(5) * 1.0e-5,
    )
    assert module.HORIZON_SECONDS == 4.0e-5
    assert module.NEW_REFINED_STEP_COUNT == 24


def test_temporal_manifest_freezes_evidence_selected_fail_fast_stages():
    module = _runner()
    parent, _ = module._validate_parent()
    selection = module._case_selection_audit(parent)
    cost = module._cost_audit()
    manifest = module._manifest(selection, cost)
    assert selection["primary_case_id"] == (
        "p3_buffer45__inward_shear__p1"
    )
    assert selection["secondary_case_id"] == (
        "p3_buffer45__outward_shear__p1"
    )
    stages = manifest["fail_fast_stages"]
    assert [stage["layout"] for stage in stages[:3]] == list(
        module.LAYOUTS
    )
    assert stages[0]["binding"]
    assert stages[1]["conditional_on"] == "c3b3b1_pass"
    assert stages[2]["conditional_on"] == "c3b3b2_pass"
    assert cost["staged_to_full_matrix_cost_ratio"] < 0.2


def test_temporal_manifest_preserves_scientific_stops_and_conditioning():
    module = _runner()
    parent, _ = module._validate_parent()
    manifest = module._manifest(
        module._case_selection_audit(parent),
        module._cost_audit(),
    )
    gates = manifest["temporal_binding_contract"]["gates"]
    assert gates["minimum_rms_order"] == 1.5
    assert gates["minimum_maximum_order"] == 1.5
    assert gates["minimum_significant_component_order"] == 1.5
    assert gates["maximum_fine_normalized_temporal_difference"] == 0.05
    assert gates["maximum_selected_step_richardson_error"] == 0.005
    assert gates["observability_factor"] == 5.0
    assert manifest["numerical_uncertainty_contract"][
        "combination_rule"
    ] == "conservative_envelope_not_RSS"
    limits = manifest["interpretation_limits"]
    assert not limits["temporal_convergence_certified"]
    assert not limits["long_nonlinear_physical_ladder_authorized"]
    assert not limits["fixed_q_micro_solver_authorized"]
    assert not limits["reduced_slow_evolution_authorized"]


def test_canonical_temporal_manifest_evidence_if_present():
    if not SUMMARY.exists():
        pytest.skip("canonical c3b3a evidence has not been generated")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert summary["work_package"] == "WP10c9d6c7c3b3a"
    assert summary["passed"]
    assert not summary["propagation_executed"]
    assert summary["coarse_temporal_screen_authorized"]
    assert not summary["temporal_convergence_certified"]
    assert not summary["long_nonlinear_physical_ladder_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    assert manifest["authorized_next"] == (
        "WP10c9d6c7c3b3b1_coarse_inward_outward_temporal_screen"
    )


def test_canonical_temporal_manifest_arrays_if_present():
    if not ARRAYS.exists():
        pytest.skip("canonical c3b3a arrays have not been generated")
    with np.load(ARRAYS, allow_pickle=False) as arrays:
        np.testing.assert_array_equal(
            arrays["timestep_levels_seconds"],
            np.asarray((1.0e-5, 5.0e-6, 2.5e-6)),
        )
        np.testing.assert_array_equal(
            arrays["step_counts"],
            np.asarray((4, 8, 16)),
        )
        assert arrays["median_step_seconds_by_layout"].shape == (3,)
        assert arrays["stage_estimated_cpu_hours"].shape == (4,)
        assert arrays["selection_error_cosines"].shape == (3,)


def test_canonical_temporal_manifest_checksums_if_present():
    if not CHECKSUMS.exists():
        pytest.skip("canonical c3b3a checksums have not been generated")
    for line in CHECKSUMS.read_text(encoding="utf-8").splitlines():
        digest, filename = line.split("  ", 1)
        assert _sha256(CHECKSUMS.parent / filename) == digest
