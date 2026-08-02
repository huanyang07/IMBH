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
    "run_causal_inner_nonlinear_temporal_symmetry_"
    "wp10c9d6c7c3b3b4.py"
)
SUMMARY = (
    ROOT
    / "results/canonical/"
    "causal_inner_nonlinear_temporal_symmetry_"
    "wp10c9d6c7c3b3b4/summary.json"
)
ARRAYS = SUMMARY.parent / "decisive_arrays.npz"
CHECKSUMS = SUMMARY.parent / "SHA256SUMS.txt"


def _runner():
    spec = importlib.util.spec_from_file_location("c3b3b4_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_symmetry_controls_match_frozen_fourth_stage():
    module = _runner()
    fine, coarse, manifest = module._validate_parent()
    stage = manifest["fail_fast_stages"][3]
    assert fine["coarse_primary_nonlinear_symmetry_controls_authorized"]
    assert coarse["passed"]
    assert module.LAYOUT == stage["layout"]
    assert [module._case_id(token) for token, _ in module.VARIANTS] == stage[
        "trajectories"
    ]
    assert [value for _, value in module.VARIANTS] == [-1.0, 0.5, -0.5]
    np.testing.assert_array_equal(
        module.TIMESTEP_LEVELS_SECONDS,
        np.asarray((1.0e-5, 5.0e-6, 2.5e-6)),
    )


def test_symmetry_controls_reuse_only_frozen_background_and_plus_case():
    module = _runner()
    _, _, manifest = module._validate_parent()
    assert manifest["fail_fast_stages"][3]["reuses"] == [
        "unperturbed_background",
        "p3_buffer45__inward_shear__p1",
    ]
    assert module.REFINED_TIMESTEPS_SECONDS.tolist() == [5.0e-6, 2.5e-6]


def test_normalized_rms_is_scale_invariant():
    module = _runner()
    values = np.asarray(((2.0, 6.0), (4.0, 12.0)))
    scales = np.asarray((2.0, 3.0))[None, :]
    expected = np.sqrt(np.mean(np.asarray(((1.0, 2.0), (2.0, 4.0))) ** 2))
    assert module._normalized_rms(values, scales) == pytest.approx(expected)


def test_explanatory_remainder_below_activity_is_not_metric_eligible():
    module = _runner()
    gates = {"minimum_relative_activity": 1.0e-8}
    fields = np.ones(5)
    observables = np.ones(13)
    invisible_exports = tuple(
        np.full((5, 13), 0.5e-8) for _ in range(3)
    )
    visible_exports = tuple(
        np.full((5, 13), 2.0e-8) for _ in range(3)
    )
    assert not module._history_has_significant_component(
        "instantaneous_exports",
        invisible_exports,
        fields,
        observables,
        gates,
    )
    assert module._history_has_significant_component(
        "instantaneous_exports",
        visible_exports,
        fields,
        observables,
        gates,
    )


def test_canonical_symmetry_controls_if_present():
    if not SUMMARY.exists():
        pytest.skip("canonical c3b3b4 evidence has not been generated")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["work_package"] == "WP10c9d6c7c3b3b4"
    assert summary["temporal_screen"][
        "all_refined_trajectory_methods_passed"
    ]
    assert summary["temporal_screen"][
        "all_checkpoint_roundtrips_bitwise"
    ]
    assert summary["temporal_screen"][
        "all_split_restart_replays_bitwise"
    ]
    assert not summary["temporal_convergence_certified"]
    assert not summary["meaningfully_nonlinear_dynamics_certified"]
    assert not summary["long_nonlinear_physical_ladder_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    if summary["passed"]:
        assert summary[
            "short_horizon_profile_breadth_controller_manifest_authorized"
        ]
        assert summary["authorized_next"].startswith(
            "WP10c9d6c7c3b4a_"
        )


def test_canonical_symmetry_arrays_if_present():
    if not ARRAYS.exists():
        pytest.skip("canonical c3b3b4 arrays have not been generated")
    with np.load(ARRAYS, allow_pickle=False) as arrays:
        for token in ("m1", "p0p5", "m0p5"):
            case = f"p3_buffer45__inward_shear__{token}"
            for level in ("h", "h2", "h4"):
                assert f"{case}__{level}__state_response" in arrays
                assert (
                    f"{case}__{level}__instantaneous_export_response"
                    in arrays
                )
                assert (
                    f"{case}__{level}__cumulative_export_response"
                    in arrays
                )
        assert "symmetry__state__even_response__h4" in arrays
        assert (
            "symmetry__instantaneous_exports__"
            "odd_amplitude_scale_defect__h4" in arrays
        )


def test_canonical_symmetry_checksums_if_present():
    if not CHECKSUMS.exists():
        pytest.skip("canonical c3b3b4 checksums have not been generated")
    for line in CHECKSUMS.read_text(encoding="utf-8").splitlines():
        digest, filename = line.split("  ", 1)
        assert _sha256(CHECKSUMS.parent / filename) == digest
