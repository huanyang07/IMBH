from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "scripts/run_causal_inner_nonlinear_profile_breadth_coarse_screen_"
    "wp10c9d6c7c3b4b1.py"
)
SUMMARY = (
    ROOT
    / "results/canonical/"
    "causal_inner_nonlinear_profile_breadth_coarse_screen_"
    "wp10c9d6c7c3b4b1/summary.json"
)
ARRAYS = SUMMARY.parent / "decisive_arrays.npz"
CHECKSUMS = SUMMARY.parent / "SHA256SUMS.txt"


def _runner():
    spec = importlib.util.spec_from_file_location("b4b1_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_coarse_profile_breadth_stage_matches_parent_manifest():
    module = _runner()
    _, manifest = module._validate_parent()
    stage = manifest["campaign_controller"]["stages"][0]
    assert module.LAYOUT == "N128_exterior_N128_inner_c48"
    assert module.TIMESTEP_SECONDS == 1.0e-5
    assert module.HORIZON_SECONDS == 4.0e-5
    assert tuple(stage["profiles"]) == module.PROFILES
    assert len(module.PROFILES) == 5


def test_canonical_coarse_breadth_screen_if_present():
    if not SUMMARY.exists():
        return
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["work_package"] == "WP10c9d6c7c3b4b1"
    assert summary["method_screen"]["profile_count"] == 5
    assert not summary["temporal_convergence_certified"]
    assert not summary["spatial_convergence_certified"]
    assert not summary["long_nonlinear_physical_ladder_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    if summary["passed"]:
        assert summary["method_screen"]["all_four_step_trajectories_completed"]
        assert summary["method_screen"]["all_export_audits_passed"]
        assert summary["method_screen"]["all_checkpoint_roundtrips_bitwise"]
        assert summary["method_screen"]["all_split_restart_replays_bitwise"]
        assert summary["coarse_heldout_temporal_refinement_authorized"]


def test_canonical_coarse_breadth_arrays_if_present():
    if not ARRAYS.exists():
        return
    module = _runner()
    with np.load(ARRAYS, allow_pickle=False) as arrays:
        for profile in module.PROFILES:
            prefix = module._task_id(profile)
            assert arrays[f"{prefix}__states"].shape[0] == 5
            assert arrays[f"{prefix}__direct_exports"].shape == (5, 13)


def test_canonical_coarse_breadth_checksums_if_present():
    if not CHECKSUMS.exists():
        return
    for line in CHECKSUMS.read_text(encoding="utf-8").splitlines():
        digest, filename = line.split("  ", 1)
        assert _sha256(CHECKSUMS.parent / filename) == digest
