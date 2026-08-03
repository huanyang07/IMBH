from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "scripts/run_causal_inner_nonlinear_profile_breadth_spatial_"
    "wp10c9d6c7c3b4b3.py"
)
SUMMARY = (
    ROOT
    / "results/canonical/"
    "causal_inner_nonlinear_profile_breadth_spatial_"
    "wp10c9d6c7c3b4b3/summary.json"
)
ARRAYS = SUMMARY.parent / "decisive_arrays.npz"
CHECKSUMS = SUMMARY.parent / "SHA256SUMS.txt"


def _runner():
    spec = importlib.util.spec_from_file_location("b4b3_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_spatial_stage_matches_frozen_controller():
    module = _runner()
    parent, breadth, spatial = module._validate_parent()
    stage = breadth["campaign_controller"]["stages"][2]
    assert module.LAYOUTS == (
        "N128_exterior_N128_inner_c48",
        "N128_exterior_N256_inner_c48",
        "N128_exterior_N512_inner_c48",
    )
    assert tuple(stage["layouts"]) == module.NEW_LAYOUTS
    assert tuple(stage["profiles"]) == module.PROFILES
    assert module.TIMESTEP_SECONDS == 1.0e-5
    assert parent["middle_fine_heldout_spatial_confirmation_authorized"]
    assert spatial["tier_I_binding_contract"]["gates"]["minimum_rms_order"] == 0.75


def test_spatial_task_matrix_is_exact():
    module = _runner()
    tasks = {
        module._task_id(layout, profile)
        for layout in module.NEW_LAYOUTS
        for profile in module.PROFILES
    }
    assert len(tasks) == 10
    assert all(task.endswith("__p1__dt_1e-5") for task in tasks)


def test_canonical_spatial_breadth_if_present():
    if not SUMMARY.exists():
        return
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["work_package"] == "WP10c9d6c7c3b4b3"
    assert not summary["meaningfully_nonlinear_dynamics_certified"]
    assert not summary["long_nonlinear_physical_ladder_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    if summary["passed"]:
        assert summary["heldout_spatial_convergence_certified"]
        assert summary["short_horizon_profile_breadth_certified"]
        assert summary["variable_step_duration_controller_manifest_authorized"]
        assert len(summary["spatial_confirmation"]["case_reports"]) == 5
        assert all(
            result["passed"]
            for result in summary["spatial_confirmation"]["case_reports"].values()
        )


def test_canonical_spatial_breadth_arrays_if_present():
    if not ARRAYS.exists():
        return
    module = _runner()
    with np.load(ARRAYS, allow_pickle=False) as arrays:
        for layout in module.LAYOUTS:
            for profile in module.PROFILES:
                assert f"{layout}__{profile}__state_response" in arrays
                assert (
                    f"{layout}__{profile}__instantaneous_export_response" in arrays
                )
                assert f"{layout}__{profile}__cumulative_export_response" in arrays


def test_canonical_spatial_breadth_checksums_if_present():
    if not CHECKSUMS.exists():
        return
    for line in CHECKSUMS.read_text(encoding="utf-8").splitlines():
        digest, filename = line.split("  ", 1)
        assert _sha256(CHECKSUMS.parent / filename) == digest
