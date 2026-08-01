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
    "run_causal_inner_nonlinear_spatial_export_pilot_"
    "wp10c9d6c7c3b2b.py"
)
SUMMARY = (
    ROOT
    / "results/canonical/"
    "causal_inner_nonlinear_spatial_export_pilot_"
    "wp10c9d6c7c3b2b/summary.json"
)
ARRAYS = SUMMARY.parent / "decisive_arrays.npz"
CHECKSUMS = SUMMARY.parent / "SHA256SUMS.txt"


def _runner():
    spec = importlib.util.spec_from_file_location("c3b2b_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pilot_preserves_frozen_matrix_and_fail_fast_order():
    module = _runner()
    assert len(module.LAYOUTS) == 3
    assert len(module.PROFILES) == 4
    assert len(module.VARIANT_MULTIPLIERS) == 4
    np.testing.assert_array_equal(module.TIMES, np.arange(5) * 1.0e-5)
    assert len(module.OBSERVABLE_NAMES) == 13


def test_cumulative_integral_is_exact_for_constant_history():
    module = _runner()
    values = np.ones((module.TIMES.size, 13))
    cumulative = module._cumulative(values)
    np.testing.assert_allclose(
        cumulative,
        module.TIMES[:, None] * np.ones((1, 13)),
        rtol=0.0,
        atol=1.0e-20,
    )


def test_canonical_pilot_evidence_if_present():
    if not SUMMARY.exists():
        pytest.skip("canonical c3b2b evidence has not been generated")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["work_package"] == "WP10c9d6c7c3b2b"
    assert summary["new_propagation_executed"] is False
    assert summary["canonical_saved_history_reused"] is True
    assert summary["state_response"]["case_count"] == 16
    if summary["passed"]:
        assert summary["state_response"]["passed"]
        assert summary["tier_I_exports"]["passed"]
        assert summary["authorized_next"] == (
            "WP10c9d6c7c3b3a_nonlinear_temporal_refinement_"
            "pilot_manifest"
        )
    assert not summary["temporal_convergence_certified"]
    assert not summary["long_nonlinear_physical_ladder_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_canonical_pilot_arrays_if_present():
    if not ARRAYS.exists():
        pytest.skip("canonical c3b2b arrays have not been generated")
    with np.load(ARRAYS, allow_pickle=False) as arrays:
        np.testing.assert_array_equal(
            arrays["times_seconds"],
            np.arange(5) * 1.0e-5,
        )
        state_keys = [
            key for key in arrays.files if key.endswith("__state_response")
        ]
        assert len(state_keys) == 48
        if json.loads(SUMMARY.read_text(encoding="utf-8"))[
            "tier_I_exports"
        ] is not None:
            instantaneous = [
                key
                for key in arrays.files
                if key.endswith("__instantaneous_export_response")
            ]
            cumulative = [
                key
                for key in arrays.files
                if key.endswith("__cumulative_export_response")
            ]
            assert len(instantaneous) == 48
            assert len(cumulative) == 48


def test_canonical_checksums_if_present():
    if not CHECKSUMS.exists():
        pytest.skip("canonical c3b2b checksums have not been generated")
    for line in CHECKSUMS.read_text(encoding="utf-8").splitlines():
        digest, filename = line.split("  ", 1)
        assert _sha256(CHECKSUMS.parent / filename) == digest
