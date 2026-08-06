from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_nonlinear_middle_cost_bounded_anchor_hardening_manifest_wp10c9d6c7c3b5c3h2a0 as runner


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_archive_scope_is_not_overclaimed() -> None:
    manifest = _read(runner.MANIFEST_PATH)
    scope = manifest["archive_scope"]
    assert scope["generic_and_heldout_output_times_bitwise_equal"] is True
    assert scope["first_stored_long_duration_output_seconds"] == 2.0e-3
    assert scope["five_profile_long_tangent_calibration_start_seconds"] == 2.4e-3
    assert scope["accepted_internal_states_stored_for_40us_to_first_output"] is False
    assert scope["five_profile_40us_to_5ms_tangent_replay_claim_authorized"] is False


def test_cheap_tangent_hardening_is_binding() -> None:
    manifest = _read(runner.MANIFEST_PATH)
    audits = manifest["cheap_prerequisite_audits"]
    assert audits["analytic_initial_BDF_history_direction_required"] is True
    assert tuple(audits["middle_all_profile_short_step_profiles"]) == runner.PROFILES
    assert tuple(audits["variable_BDF_step_ratio_audit_values"]) == (0.5, 1.0, 2.0)
    assert audits["variable_ratio_complete_residual_JVP_required"] is True


def test_surrogate_gates_are_relative_to_response_and_spatial_difference() -> None:
    gates = _read(runner.MANIFEST_PATH)["surrogate_contract"]
    assert gates["maximum_absolute_scaled_state_discrepancy"] == 5.0e-3
    assert gates["maximum_absolute_scaled_Tier_I_discrepancy"] == 5.0e-3
    assert gates["maximum_discrepancy_fraction_of_observable_response"] == 1.0e-2
    assert (
        gates["maximum_uncertainty_fraction_of_observable_spatial_difference"]
        == 1.0e-1
    )
    assert "do_not_report_order" in gates["unobservable_spatial_difference_action"]


def test_resource_limit_is_soft_and_pilot_projection_is_measured() -> None:
    policy = _read(runner.MANIFEST_PATH)["resource_policy"]
    assert policy["projected_wall_hours_at_most_24"] == "continue_automatically"
    assert "continue_after" in policy["projected_wall_hours_24_to_48"]
    assert policy["scientific_rejection_from_cost_projection_alone_forbidden"] is True
    assert policy["pilot_minimum_accepted_steps"] == 5
    assert len(policy["projection_components"]) == 7


def test_only_cheap_audits_and_middle_pilot_are_authorized() -> None:
    summary = _read(runner.SUMMARY_PATH)
    assert summary["cheap_hardening_audits_authorized"] is True
    assert summary["middle_0p2ms_pilot_authorized_after_cheap_audits"] is True
    assert summary["middle_1ms_propagation_authorized"] is False
    assert summary["fine_cost_bounded_propagation_authorized"] is False
    assert summary["third_duration_rung_spatial_convergence_certified"] is False
    assert summary["fixed_q_micro_solver_authorized"] is False
    assert summary["reduced_slow_evolution_authorized"] is False


def test_canonical_hashes_close() -> None:
    for line in (runner.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        payload = (runner.CANONICAL_DIRECTORY / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == digest
