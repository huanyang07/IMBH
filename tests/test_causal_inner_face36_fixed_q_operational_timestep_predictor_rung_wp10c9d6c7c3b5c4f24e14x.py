from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / (
    "run_causal_inner_face36_fixed_q_operational_timestep_predictor_rung_"
    "wp10c9d6c7c3b5c4f24e14x.py"
)
SPEC = importlib.util.spec_from_file_location("e14x_rung", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_bounded_predictor_reconstructs_previous_increment():
    previous = np.array([[0.004, -0.002], [0.001, 0.003]])
    continuation = SimpleNamespace(
        history=SimpleNamespace(
            previous_primitive_increment=previous,
            previous_timestep_seconds=1.0e-7,
        ),
        next_reaction_channel_transform=np.eye(3),
        raw_multiplier_predictor=np.array([1.0, 2.0, 3.0]),
    )
    columns = np.ones_like(previous)
    rate, multiplier = MODULE._bounded_predictors(continuation, columns)
    coefficients = MODULE.causal_bdf_coefficients(2, 2.0e-7, 1.0e-7)
    reconstructed = (
        2.0e-7 * rate
        - coefficients.previous_increment_coefficient * previous.ravel()
    ) / coefficients.current_increment_coefficient
    assert np.array_equal(reconstructed, previous.ravel())
    assert np.array_equal(multiplier, np.array([1.0, 2.0, 3.0]))


def test_manifest_validation_when_frozen(monkeypatch):
    if not MODULE.MANIFEST_DIRECTORY.exists():
        pytest.skip("predictor-repair manifest is not frozen yet")
    provenance = MODULE.base._read(MODULE.MANIFEST_DIRECTORY / "provenance.json")
    for name, value in provenance["thread_environment"].items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    with MODULE._patched_runtime():
        frozen = MODULE.base._validate_manifest()
    assert frozen["summary"]["operational_timestep_rung_2e7_execution_authorized"]


def test_canonical_result_when_available():
    path = MODULE.CANONICAL_DIRECTORY / "summary.json"
    if not path.exists():
        pytest.skip("predictor-repair rung has not executed")
    summary = MODULE.base._read(path)
    metrics = MODULE.base._read(MODULE.CANONICAL_DIRECTORY / "metrics.json")
    assert summary["passed"] == metrics["scientific_passed"]
    MODULE.base._checksums(MODULE.CANONICAL_DIRECTORY)
