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
    / "scripts/run_causal_inner_nonlinear_profile_breadth_controller_"
    "manifest_wp10c9d6c7c3b4a.py"
)
SUMMARY = (
    ROOT
    / "results/canonical/"
    "causal_inner_nonlinear_profile_breadth_controller_manifest_"
    "wp10c9d6c7c3b4a/summary.json"
)
MANIFEST = SUMMARY.parent / "profile_breadth_controller_manifest.json"
ARRAYS = SUMMARY.parent / "decisive_arrays.npz"
CHECKSUMS = SUMMARY.parent / "SHA256SUMS.txt"


def _runner():
    spec = importlib.util.spec_from_file_location("c3b4a_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_profile_breadth_is_frozen_before_propagation():
    module = _runner()
    parent, _, _ = module._validate_parent()
    assert parent["passed"]
    assert tuple(module.PROFILE_DEFINITIONS) == (
        "p4__inward_acoustic",
        "p4__outward_acoustic",
        "p3_buffer45__material",
        "p4__inward_shear_acoustic_mix",
        "p3_buffer45__generic_five_field",
    )
    assert module.READINESS_MULTIPLIERS == (1.0, -1.0, 0.5, -0.5)
    assert module.BINDING_PROPAGATION_MULTIPLIER == 1.0


def test_mixed_profiles_cover_declared_family_space():
    module = _runner()
    inward = np.asarray(module.INWARD_MIXED_COEFFICIENTS)
    generic = np.asarray(module.GENERIC_COEFFICIENTS)
    assert inward.shape == (5,)
    assert np.count_nonzero(inward) == 2
    assert np.all(generic != 0.0)
    assert tuple(module.CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES) == (
        "inward_acoustic",
        "inward_shear",
        "material",
        "outward_shear",
        "outward_acoustic",
    )


def test_campaign_controller_is_fail_fast_and_cost_bounded():
    module = _runner()
    controller = module._campaign_controller()
    assert controller["passed"]
    assert controller["kind"] == "fail_fast_checkpoint_safe_campaign_controller"
    assert controller["estimated_staged_cpu_hours"] < controller[
        "estimated_naive_full_matrix_cpu_hours"
    ]
    assert [stage["work_package"] for stage in controller["stages"]] == [
        "WP10c9d6c7c3b4b1",
        "WP10c9d6c7c3b4b2",
        "WP10c9d6c7c3b4b3",
    ]


def test_support_faces_are_parent_face_indices_not_nominal_resolution():
    module = _runner()
    assert module.COUPLING_PARENT_FACE == 48
    assert module.BUFFER_PARENT_FACE == 45
    assert module.PROFILE_DEFINITIONS["p4__inward_acoustic"][
        "support_upper_parent_face"
    ] == 48
    assert module.PROFILE_DEFINITIONS["p3_buffer45__material"][
        "support_upper_parent_face"
    ] == 45


def test_canonical_manifest_if_present():
    if not SUMMARY.exists():
        pytest.skip("canonical c3b4a evidence has not been generated")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert summary["passed"]
    assert summary["profile_count"] == 5
    assert summary["all_profiles_spectrally_eligible"]
    assert summary["all_initial_readiness_variants_passed"]
    assert not summary["propagation_executed"]
    assert not summary["meaningfully_nonlinear_dynamics_certified"]
    assert not summary["long_nonlinear_physical_ladder_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    assert manifest["authorized_next"].endswith("profile_breadth_screen")


def test_canonical_profile_arrays_if_present():
    if not ARRAYS.exists():
        pytest.skip("canonical c3b4a arrays have not been generated")
    module = _runner()
    with np.load(ARRAYS, allow_pickle=False) as arrays:
        for profile in module.PROFILE_NAMES:
            for label in module.LAYOUTS:
                assert f"{profile}__{label}__primary_physical" in arrays
            for label in module.UNIFORM_LABELS:
                assert f"{profile}__{label}__primary_physical" in arrays


def test_canonical_checksums_if_present():
    if not CHECKSUMS.exists():
        pytest.skip("canonical c3b4a checksums have not been generated")
    for line in CHECKSUMS.read_text(encoding="utf-8").splitlines():
        digest, filename = line.split("  ", 1)
        assert _sha256(CHECKSUMS.parent / filename) == digest
