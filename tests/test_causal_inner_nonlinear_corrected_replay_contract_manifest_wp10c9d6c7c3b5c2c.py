from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_nonlinear_corrected_replay_contract_manifest_wp10c9d6c7c3b5c2c as runner


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_manifest_is_definitions_only_and_preserves_failures() -> None:
    summary = _read(runner.SUMMARY_PATH)
    assert summary["passed"] is True
    assert summary["propagation_executed"] is False
    assert summary["parent_classification_preserved"] == (
        "second_rung_replay_boolean_localized_to_one_ulp_time_label_"
        "fresh_process_replay_roundoff_scale_paired_replay_required"
    )
    assert summary["historical_classification_preserved"] == (
        "second_nonlinear_duration_rung_failed_later_duration_work_blocked"
    )


def test_replay_observables_are_separate_and_bitwise() -> None:
    manifest = _read(runner.MANIFEST_PATH)
    gates = manifest["separate_replay_gates"]
    assert gates["canonical_time_labels_bitwise"] is True
    assert gates["primitive_states_bitwise"] is True
    assert gates["direct_Tier_I_exports_bitwise"] is True
    assert gates["complete_BDF_history_bitwise"] is True
    assert gates["no_combined_short_circuit_boolean"] is True


def test_one_tangent_and_paired_restart_are_frozen() -> None:
    paired = _read(runner.MANIFEST_PATH)["paired_base_replay"]
    assert paired["build_one_frozen_tangent_shared_by_both_branches"] is True
    assert paired["direct_branch_starts_from_in_memory_restart"] is True
    assert paired["serialized_branch_starts_from_save_load_of_same_restart"] is True
    assert paired["accepted_step_count_per_branch"] == 4


def test_only_paired_base_replay_is_authorized() -> None:
    summary = _read(runner.SUMMARY_PATH)
    assert summary["authorized_next"] == (
        "WP10c9d6c7c3b5c2c1_paired_base_replay_validation"
    )
    assert summary["paired_base_replay_validation_authorized"] is True
    assert summary["perturbed_second_rung_authorized"] is False
    assert summary["later_duration_rungs_authorized"] is False
    assert summary["fixed_q_micro_solver_authorized"] is False
    assert summary["reduced_slow_evolution_authorized"] is False


def test_positive_branch_runs_only_missing_perturbed_trajectory() -> None:
    positive = _read(runner.MANIFEST_PATH)["positive_branch"]
    assert positive["reuse_committed_base_main_and_strict_arrays_by_hash"] is True
    assert positive["run_only_missing_perturbed_main_replay_and_strict_trajectory"] is True


def test_canonical_hashes_close() -> None:
    for line in (runner.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256(
            (runner.CANONICAL_DIRECTORY / name).read_bytes()
        ).hexdigest() == digest
