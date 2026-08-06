from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_nonlinear_middle_cost_bounded_anchor_manifest_wp10c9d6c7c3b5c3h2 as runner


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_manifest_freezes_one_base_one_anchor_and_block_breadth() -> None:
    manifest = _read(runner.MANIFEST_PATH)
    anchor = manifest["tangent_anchor_contract"]
    experiment = manifest["middle_experiment"]
    assert manifest["definitions_only"] is True
    assert manifest["propagation_executed"] is False
    assert experiment["layout"] == runner.MIDDLE_LAYOUT
    assert experiment["one_context_and_one_base_schedule_reused"] is True
    assert experiment["all_profile_tangents_solved_as_one_block"] is True
    assert anchor["generic_full_nonlinear_anchor_required"] is True
    assert anchor["other_full_nonlinear_perturbed_trajectories_forbidden"] is True
    assert tuple(anchor["profiles"]) == runner.PROFILES


def test_fail_fast_cost_pilot_precedes_long_middle_execution() -> None:
    manifest = _read(runner.MANIFEST_PATH)
    stages = manifest["fail_fast_stages"]
    assert stages[0]["stage"] == "h2a_cost_pilot"
    assert stages[0]["stop_seconds"] == 2.0e-4
    assert stages[-1]["stop_seconds"] == 5.0e-3
    cost = manifest["cost_contract"]
    assert cost["maximum_projected_total_new_nonlinear_wall_hours"] == 24.0
    assert cost["maximum_single_unattended_stage_wall_hours"] == 12.0
    assert "stop_before_full_middle" in cost["projection_failure_action"]


def test_layout_schedule_temporal_and_surrogate_budgets_are_binding() -> None:
    manifest = _read(runner.MANIFEST_PATH)
    temporal = manifest["temporal_contract"]
    anchor = manifest["tangent_anchor_contract"]
    assert temporal["layout_owns_adaptive_schedule"] is True
    assert temporal["coarse_schedule_reuse_forbidden"] is True
    assert temporal["same_target_replay_complete_payload_bitwise"] is True
    assert temporal["all_common_strict_outputs_are_compared"] is True
    assert anchor["maximum_scaled_state_discrepancy"] == 5.0e-3
    assert anchor["maximum_scaled_Tier_I_discrepancy"] == 5.0e-3
    assert anchor["surrogate_uncertainty_added_to_spatial_and_temporal_budgets"] is True


def test_only_staged_middle_execution_is_authorized() -> None:
    summary = _read(runner.SUMMARY_PATH)
    assert summary["middle_staged_execution_authorized"] is True
    assert summary["fine_cost_bounded_confirmation_manifest_authorized"] is False
    assert summary["fine_cost_bounded_propagation_authorized"] is False
    assert summary["third_duration_rung_spatial_convergence_certified"] is False
    assert summary["fourth_duration_rung_manifest_authorized"] is False
    assert summary["fixed_q_micro_solver_authorized"] is False
    assert summary["reduced_slow_evolution_authorized"] is False


def test_canonical_hashes_close() -> None:
    for line in (runner.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        payload = (runner.CANONICAL_DIRECTORY / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == digest
