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
    "run_causal_inner_nonlinear_temporal_fine_primary_"
    "wp10c9d6c7c3b3b3.py"
)
SUMMARY = (
    ROOT
    / "results/canonical/"
    "causal_inner_nonlinear_temporal_fine_primary_"
    "wp10c9d6c7c3b3b3/summary.json"
)
ARRAYS = SUMMARY.parent / "decisive_arrays.npz"
CHECKSUMS = SUMMARY.parent / "SHA256SUMS.txt"


def _runner():
    spec = importlib.util.spec_from_file_location("c3b3b3_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fine_primary_confirmation_matches_frozen_manifest():
    module = _runner()
    _, manifest = module._validate_parent()
    np.testing.assert_array_equal(
        module.TIMESTEP_LEVELS_SECONDS,
        np.asarray((1.0e-5, 5.0e-6, 2.5e-6)),
    )
    assert module.HORIZON_SECONDS == 4.0e-5
    assert module.LAYOUT == manifest["fail_fast_stages"][2]["layout"]
    assert [module._case_id(profile) for profile in module.PROFILES] == [
        "p3_buffer45__inward_shear__p1",
    ]


def test_fine_primary_confirmation_uses_nested_common_outputs():
    module = _runner()
    np.testing.assert_array_equal(
        module._common_indices(1.0e-5),
        np.asarray((0, 1, 2, 3, 4)),
    )
    np.testing.assert_array_equal(
        module._common_indices(5.0e-6),
        np.asarray((0, 2, 4, 6, 8)),
    )
    np.testing.assert_array_equal(
        module._common_indices(2.5e-6),
        np.asarray((0, 4, 8, 12, 16)),
    )


def test_fine_primary_confirmation_restricts_to_common_parent_grid():
    module = _runner()
    parent_grid, layout = module._restriction_geometry()
    history = np.ones((2, layout.n_cells, 5), dtype=float)
    restricted = module._restrict_state_history(history, layout)
    assert restricted.shape == (2, parent_grid.centers.size, 5)
    np.testing.assert_array_equal(restricted, 1.0)


def test_selected_step_error_uses_second_order_richardson_factor():
    module = _runner()
    coarse = np.asarray(((1.0, 2.0), (3.0, 4.0)))
    medium = coarse + np.asarray(((0.3, -0.6), (0.0, 0.15)))
    scales = np.asarray((2.0, 3.0))[None, :]
    expected = (4.0 / 3.0) * 0.2
    assert module._selected_step_error(coarse, medium, scales) == pytest.approx(
        expected
    )


def test_canonical_fine_primary_confirmation_if_present():
    if not SUMMARY.exists():
        pytest.skip("canonical c3b3b3 evidence has not been generated")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["work_package"] == "WP10c9d6c7c3b3b3"
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
    assert not summary["long_nonlinear_physical_ladder_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    if summary["passed"]:
        assert summary[
            "coarse_primary_nonlinear_symmetry_controls_authorized"
        ]
        assert summary["authorized_next"] == (
            "WP10c9d6c7c3b3b4_"
            "coarse_primary_nonlinear_symmetry_controls"
        )


def test_canonical_fine_primary_arrays_if_present():
    if not ARRAYS.exists():
        pytest.skip("canonical c3b3b3 arrays have not been generated")
    with np.load(ARRAYS, allow_pickle=False) as arrays:
        np.testing.assert_array_equal(
            arrays["timestep_levels_seconds"],
            np.asarray((1.0e-5, 5.0e-6, 2.5e-6)),
        )
        for case in ("p3_buffer45__inward_shear__p1",):
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


def test_canonical_fine_primary_checksums_if_present():
    if not CHECKSUMS.exists():
        pytest.skip("canonical c3b3b3 checksums have not been generated")
    for line in CHECKSUMS.read_text(encoding="utf-8").splitlines():
        digest, filename = line.split("  ", 1)
        assert _sha256(CHECKSUMS.parent / filename) == digest
