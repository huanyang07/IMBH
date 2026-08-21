from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_hot_exit_half_step_recovery_manifest_wp10c9d6c7c3b5c4f25dp as manifest


def test_half_step_contract_preserves_the_failed_full_step() -> None:
    contract = manifest._contract()
    assert contract["preserved_rejection"]["step"] == 6
    assert not contract["preserved_rejection"]["physical_failure_selected"]
    assert not contract["preserved_rejection"]["rejected_state_propagated"]
    assert contract["solver_contract"]["maximum_scaled_primitive_change"] == 5.0e-3
    assert not contract["solver_contract"]["trust_bound_relaxed"]


def test_variable_step_requires_two_cold_roots_before_reuse() -> None:
    order = manifest._contract()["execution_order"]
    assert order["new_timestep_seconds"] == 5.0e-8
    assert order["roots_1_and_2_use_cold_exact_initial_matrix"]
    assert order["roots_3_onward_may_use_carried_matrix"]
    assert order["one_root_per_command"]
    assert order["rejected_root_never_propagates"]


def test_no_branch_or_reduced_execution_is_authorized() -> None:
    boundaries = manifest._contract()["authorization_boundaries"]
    assert boundaries["half_step_execution_in_next_package"]
    assert not boundaries["branch_root_execution_authorized"]
    assert not boundaries["transition_impulse_fit_authorized"]
    assert not boundaries["reduced_slow_evolution_authorized"]
