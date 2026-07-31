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
    "run_causal_inner_monolithic_bdf_profile_screen_"
    "wp10c9d6c7c3b1b1.py"
)
SUMMARY = (
    ROOT
    / "results/canonical/"
    "causal_inner_monolithic_bdf_profile_screen_"
    "wp10c9d6c7c3b1b1/summary.json"
)
CHECKSUMS = SUMMARY.parent / "SHA256SUMS.txt"


def _runner():
    spec = importlib.util.spec_from_file_location("c3b1b1_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_case_matrix_is_complete_and_ordered():
    module = _runner()
    cases = module._case_sequence()
    assert len(cases) == 16
    assert len(set(cases)) == 16
    assert tuple(profile for profile, _ in cases[::4]) == module.PROFILES
    assert tuple(multiplier for _, multiplier in cases[:4]) == (
        1.0,
        -1.0,
        0.5,
        -0.5,
    )


def test_scaled_linear_predictor_uses_frozen_rate_and_generator():
    module = _runner()
    configuration = {
        "base": np.zeros((1, 5)),
        "columns": np.arange(1.0, 6.0),
    }
    tangent = type(
        "Tangent",
        (),
        {
            "scaled_base_rate_per_s": np.ones(5),
            "scaled_generator_per_s": 2.0 * np.eye(5),
        },
    )()
    state = np.ones((1, 5))
    result = module._scaled_linear_predictor(
        configuration,
        tangent,
        state,
    )
    expected_rate = 1.0 + 2.0 / configuration["columns"]
    expected = (
        module.TIMESTEP_SECONDS
        * configuration["columns"]
        * expected_rate
    )
    assert np.allclose(result.ravel(), expected, rtol=0.0, atol=0.0)


def test_canonical_profile_screen_evidence_if_present():
    if not SUMMARY.exists():
        pytest.skip("canonical c3b1b1 evidence has not been generated")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["work_package"] == "WP10c9d6c7c3b1b1"
    assert summary["expected_case_count"] == 48
    assert summary["completed_case_count"] <= 48
    if summary["passed"]:
        assert summary["completed_case_count"] == 48
        assert summary["bdf2_restart_depth_authorized"]
        assert (
            summary["classification"]
            == "full_profile_variant_bdf1_screen_certified_"
            "bdf2_restart_depth_authorized"
        )
    else:
        assert not summary["bdf2_restart_depth_authorized"]
    assert not summary["long_nonlinear_physical_ladder_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_canonical_profile_screen_checksums_if_present():
    if not CHECKSUMS.exists():
        pytest.skip("canonical c3b1b1 checksums have not been generated")
    for line in CHECKSUMS.read_text(encoding="utf-8").splitlines():
        digest, filename = line.split("  ", 1)
        assert _sha256(CHECKSUMS.parent / filename) == digest
