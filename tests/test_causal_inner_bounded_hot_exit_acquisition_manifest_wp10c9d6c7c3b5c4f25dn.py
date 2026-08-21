from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_bounded_hot_exit_acquisition_manifest_wp10c9d6c7c3b5c4f25dn as manifest


def test_hot_exit_manifest_preserves_cold_only_parent() -> None:
    validation = manifest._validate_parent(require_clean=False)
    assert validation["seed_completed_steps"] == 6
    assert validation["seed_elapsed_time_seconds"] == pytest.approx(0.0200006)


def test_hot_exit_manifest_is_bounded_and_stepwise() -> None:
    contract = manifest._contract()
    assert contract["execution_order"]["one_root_per_command"]
    assert contract["execution_order"]["maximum_new_BDF2_roots"] == 12
    assert contract["execution_order"]["fixed_equal_BDF2_timestep_seconds"] == 1.0e-7
    assert contract["execution_order"]["rejected_root_never_propagates"]
    assert contract["hot_exit_gate"]["saved_secant_hidden_fraction_max"] == 0.25
    assert contract["hot_exit_gate"]["consecutive_accepted_secants_required"] == 2
    assert contract["coordinate_diagnostic"]["hot_distance_uses_hidden_not_macro_displacement"]
    assert contract["coordinate_diagnostic"]["full470_fallback_is_binding"]


def test_hot_exit_manifest_keeps_branch_truth_blocked() -> None:
    boundaries = manifest._contract()["authorization_boundaries"]
    assert boundaries["stepwise_hot_exit_execution_in_next_package"]
    assert not boundaries["branch_root_execution_authorized"]
    assert not boundaries["transition_impulse_fit_authorized"]
    assert not boundaries["online_solver_authorized"]
    assert not boundaries["reduced_slow_evolution_authorized"]
