from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_nonlinear_third_duration_rung_breadth_manifest_wp10c9d6c7c3b5c3e as runner


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_manifest_freezes_coarse_heldout_fail_fast_stage() -> None:
    manifest = _read(runner.MANIFEST_PATH)
    stage = manifest["coarse_heldout_duration_stage"]
    assert manifest["propagation_executed"] is False
    assert stage["authorized_now"] is True
    assert tuple(stage["profiles"]) == runner.COARSE_HELDOUT_PROFILES
    assert tuple(stage["fail_fast_execution_order"]) == runner.COARSE_EXECUTION_ORDER
    assert stage["certified_c3d_base_main_replay_strict_reused_by_hash"] is True
    assert stage["same_target_replay_states_exports_histories_and_restart_bitwise"] is True
    assert stage["stop_on_first_profile_failure"] is True


def test_target_and_initial_history_contracts_are_exact() -> None:
    manifest = _read(runner.MANIFEST_PATH)
    common = manifest["common_contract"]
    stage = manifest["coarse_heldout_duration_stage"]
    assert common["independent_target_construction_forbidden"] is True
    assert common["main_target_microseconds"] == runner.MAIN_TARGET_MICROSECONDS.tolist()
    assert common["replay_target_microseconds"] == runner.REPLAY_TARGET_MICROSECONDS.tolist()
    assert common["strict_target_microseconds"] == runner.STRICT_TARGET_MICROSECONDS.tolist()
    assert stage["initial_history_seconds"] == runner.INITIAL_HELDOUT_HISTORY_SECONDS.tolist()
    assert stage["initial_previous_timestep_seconds"] == 2.5e-6


def test_spatial_scope_is_frozen_but_not_authorized() -> None:
    manifest = _read(runner.MANIFEST_PATH)
    summary = _read(runner.SUMMARY_PATH)
    spatial = manifest["generic_spatial_confirmation_scope"]
    assert spatial["scope_frozen_but_propagation_not_authorized"] is True
    assert spatial["active_coupling_face_indices"] == {
        runner.SPATIAL_LAYOUTS[0]: 48,
        runner.SPATIAL_LAYOUTS[1]: 96,
        runner.SPATIAL_LAYOUTS[2]: 192,
    }
    assert spatial["fresh_definitions_only_manifest_required_after_coarse_breadth_passes"] is True
    assert summary["coarse_heldout_duration_screen_authorized"] is True
    assert summary["third_duration_rung_spatial_confirmation_manifest_authorized"] is False
    assert summary["third_duration_rung_spatial_confirmation_propagation_authorized"] is False
    assert summary["fourth_duration_rung_manifest_authorized"] is False
    assert summary["fixed_q_micro_solver_authorized"] is False
    assert summary["reduced_slow_evolution_authorized"] is False


def test_canonical_hashes_close() -> None:
    for line in (runner.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256(
            (runner.CANONICAL_DIRECTORY / name).read_bytes()
        ).hexdigest() == digest
