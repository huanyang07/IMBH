from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_nonlinear_middle_1ms_continuation_manifest_wp10c9d6c7c3b5c3h2b0 as runner


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_pilot_measurements_and_cost_tier_are_inherited() -> None:
    manifest = _read(runner.MANIFEST_PATH)
    measured = manifest["measured_basis"]
    assert measured["pilot_accepted_steps"] == 6
    assert measured["pilot_rejected_attempts"] == 0
    assert measured["resource_tier"] == "automatic_continuation"
    assert measured["projected_5ms_wall_hours_with_factor_two_safety"] < 24.0
    assert measured["routine_five_profile_block_solve_seconds"] < 0.1


def test_canonical_targets_and_restart_are_frozen() -> None:
    continuation = _read(runner.MANIFEST_PATH)["continuation"]
    assert tuple(continuation["canonical_target_microseconds"]) == (
        200,
        400,
        600,
        800,
        1000,
    )
    assert tuple(continuation["replay_target_microseconds"]) == (800, 1000)
    assert continuation["single_integer_target_source_required"] is True
    assert continuation["restart_roundtrip_bitwise_required"] is True
    assert continuation["no_new_BDF1_startup"] is True


def test_base_keeps_full_controller_and_anchor_sampling_is_prospective() -> None:
    manifest = _read(runner.MANIFEST_PATH)
    base = manifest["base_schedule_contract"]
    anchor = manifest["generic_anchor_contract"]
    assert base["full_step_doubling_on_every_base_accepted_comparison"] is True
    assert base["all_method_physics_and_ledger_gates_unchanged"] is True
    assert anchor["all_other_steps_use_one_full_nonlinear_solve"] is True
    assert len(anchor["step_doubling_audit_locations"]) == 3
    assert anchor["any_sampled_error_failure_stops_continuation"] is True


def test_tangent_breadth_and_surrogate_gates_are_binding() -> None:
    tangent = _read(runner.MANIFEST_PATH)["tangent_contract"]
    gates = tangent["surrogate_gates"]
    assert tangent["all_profiles_propagated_as_one_block"] is True
    assert tangent["generic_anchor_closes_full_0p2_to_1ms_interval"] is True
    assert tangent["non_generic_full_nonlinear_trajectories_forbidden"] is True
    assert gates["maximum_discrepancy_fraction_of_observable_response"] == 0.01
    assert (
        gates["maximum_uncertainty_fraction_of_observable_spatial_difference"]
        == 0.1
    )


def test_only_1ms_propagation_is_authorized() -> None:
    summary = _read(runner.SUMMARY_PATH)
    assert summary["middle_1ms_propagation_authorized"] is True
    assert summary["middle_2ms_propagation_authorized"] is False
    assert summary["middle_5ms_spatial_confirmation_certified"] is False
    assert summary["fine_cost_bounded_propagation_authorized"] is False
    assert summary["fixed_q_micro_solver_authorized"] is False
    assert summary["reduced_slow_evolution_authorized"] is False


def test_canonical_hashes_close() -> None:
    lines = (runner.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text().splitlines()
    assert len(lines) == 4
    for line in lines:
        digest, name = line.split("  ", 1)
        payload = (runner.CANONICAL_DIRECTORY / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == digest
