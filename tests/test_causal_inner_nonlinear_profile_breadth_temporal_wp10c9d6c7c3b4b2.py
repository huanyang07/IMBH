from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "scripts/run_causal_inner_nonlinear_profile_breadth_temporal_"
    "wp10c9d6c7c3b4b2.py"
)
SUMMARY = (
    ROOT
    / "results/canonical/"
    "causal_inner_nonlinear_profile_breadth_temporal_"
    "wp10c9d6c7c3b4b2/summary.json"
)
ARRAYS = SUMMARY.parent / "decisive_arrays.npz"
CHECKSUMS = SUMMARY.parent / "SHA256SUMS.txt"


def _runner():
    spec = importlib.util.spec_from_file_location("b4b2_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_temporal_stage_matches_frozen_controller():
    module = _runner()
    _, breadth, temporal = module._validate_parent()
    stage = breadth["campaign_controller"]["stages"][1]
    np.testing.assert_array_equal(
        module.TIMESTEP_LEVELS_SECONDS, np.asarray((1.0e-5, 5.0e-6, 2.5e-6))
    )
    assert tuple(stage["profiles"]) == module.PROFILES
    assert len(module.PROFILES) == 5
    assert temporal["scope"]["each_level_uses_own_BDF1_startup_then_BDF2"]


def test_common_temporal_indices_are_nested():
    module = _runner()
    np.testing.assert_array_equal(module._common_indices(1.0e-5), (0, 1, 2, 3, 4))
    np.testing.assert_array_equal(module._common_indices(5.0e-6), (0, 2, 4, 6, 8))
    np.testing.assert_array_equal(module._common_indices(2.5e-6), (0, 4, 8, 12, 16))


def test_canonical_temporal_breadth_if_present():
    if not SUMMARY.exists():
        return
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["work_package"] == "WP10c9d6c7c3b4b2"
    assert not summary["heldout_spatial_convergence_certified"]
    assert not summary["long_nonlinear_physical_ladder_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    if summary["passed"]:
        assert summary["coarse_heldout_temporal_convergence_certified"]
        assert summary["middle_fine_heldout_spatial_confirmation_authorized"]
        assert len(summary["temporal_screen"]["case_reports"]) == 5
        assert all(
            report["passed"]
            for report in summary["temporal_screen"]["case_reports"].values()
        )


def test_canonical_temporal_breadth_arrays_if_present():
    if not ARRAYS.exists():
        return
    module = _runner()
    with np.load(ARRAYS, allow_pickle=False) as arrays:
        for profile in module.PROFILES:
            for level in ("h", "h2", "h4"):
                assert f"{profile}__{level}__state_response" in arrays
                assert f"{profile}__{level}__instantaneous_export_response" in arrays
                assert f"{profile}__{level}__cumulative_export_response" in arrays


def test_canonical_temporal_breadth_checksums_if_present():
    if not CHECKSUMS.exists():
        return
    for line in CHECKSUMS.read_text(encoding="utf-8").splitlines():
        digest, filename = line.split("  ", 1)
        assert _sha256(CHECKSUMS.parent / filename) == digest
