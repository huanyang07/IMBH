from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_bounded_hot_exit_acquisition_wp10c9d6c7c3b5c4f25do as execution


def test_execution_contract_matches_manifest() -> None:
    contract = execution._static_execution_contract()
    assert contract["one_root_per_command"]
    assert contract["maximum_steps"] == 12
    assert contract["timestep_seconds"] == 1.0e-7
    assert contract["hidden_fraction_max"] == 0.25
    assert contract["persistence_steps"] == 2
    assert contract["rejected_root_never_propagates"]
    assert contract["full_y470_dynamics_binding"]
    assert contract["rank16_coordinates_diagnostic_only"]


def test_feature_gate_uses_hidden_not_macro_distance() -> None:
    static = {
        "model": _IdentityCoordinateModel(),
        "macro_restriction": np.asarray([[1.0, 0.0]]),
        "hidden_basis": np.asarray([[0.0], [1.0]]),
        "hidden_dual": np.asarray([[0.0, 1.0]]),
        "rank16_basis": np.asarray([[1.0]]),
        "anchor_coordinate": np.asarray([0.0, 0.0]),
        "seed_coordinate": np.asarray([0.0, 0.0]),
    }
    previous = np.asarray([0.0, 0.04999999])
    current = np.asarray([0.0, 0.05])
    metrics, arrays = execution._exit_features(static, previous, current)
    assert metrics["hidden_secant_fraction"] == pytest.approx(1.0)
    assert metrics["rank16_hidden_amplitude_from_20ms_anchor"] == pytest.approx(0.05)
    assert metrics["macro_drift_from_warm3_seed"] == pytest.approx(0.0)
    assert metrics["rank16_amplitude_gate_passed"]
    assert metrics["macro_drift_gate_passed"]
    assert not metrics["hidden_fraction_gate_passed"]
    assert arrays["current_coordinate470"].shape == (2,)


def test_stage_paths_are_deterministic() -> None:
    assert execution._stage_artifact(1).endswith("step_01")
    assert execution._stage_directory(12).name.endswith("step_12")
    assert execution._input_checkpoint(1) == execution.manifest.SEED_CHECKPOINT


def test_execution_lock_covers_transitive_solver_and_coordinate_sources() -> None:
    payload = execution._execution_lock_payload()
    sources = payload["transitive_execution_source_hashes"]
    assert execution.legacy.e14d.THIS_RUNNER in sources
    assert execution.screen.geometry.field_manifest.THIS_RUNNER in sources
    assert execution.screen.geometry.field_manifest.vector_field.THIS_RUNNER in sources
    assert len(payload["coordinate_field_arrays_sha256"]) == 64


class _IdentityCoordinateModel:
    def coordinate(self, state):
        return np.asarray(state, dtype=float), np.ones_like(state, dtype=float)
