from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_causal_inner_nonlinear_first_duration_rung_wp10c9d6c7c3b5c1 as runner


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_parent_manifest_classification_is_preserved() -> None:
    summary = _read_json(runner.SUMMARY_PATH)
    assert summary["parent_classification_preserved"] == (
        "first_nonlinear_duration_rung_manifest_frozen_"
        "two_e_minus_four_second_propagation_authorized"
    )


def test_both_first_rung_trajectories_pass() -> None:
    summary = _read_json(runner.SUMMARY_PATH)
    assert summary["passed"] is True
    assert summary["all_trajectory_methods_passed"] is True
    assert set(summary["trajectory_reports"]) == {"base", "perturbed"}
    for report in summary["trajectory_reports"].values():
        assert report["passed"] is True
        assert report["accepted_BDF1_steps"] == 1
        assert report["accepted_main_BDF2_steps"] == 12
        assert report["accepted_replay_BDF2_steps"] == 5
        assert report["accepted_strict_shadow_BDF2_steps"] == 4
        assert report["main_rejected_attempts"] == 0
        assert report["strict_rejected_attempts"] == 0
        assert report["maximum_main_local_error_estimate"] <= 2.5e-4
        assert report["sum_main_local_error_estimates"] <= 5.0e-3
        assert report["maximum_scaled_residual"] <= 1.0e-10
        assert report["maximum_discrete_ledger_defect"] <= 1.0e-12
        assert report["maximum_mapped_endpoint_path_closure_defect"] <= 1.0e-9
        assert report["minimum_path_reconstruction_factor"] >= 1.0
        assert report["maximum_incoming_excision_characteristics"] == 0
        assert report["checkpoint_roundtrips_bitwise"] is True
        assert report["split_restart_replay_bitwise"] is True


def test_main_controller_timesteps_and_outputs_are_frozen() -> None:
    config = _read_json(runner.CONFIG_PATH)
    with np.load(runner.DECISIVE_ARRAYS, allow_pickle=False) as arrays:
        for trajectory in ("base", "perturbed"):
            times = arrays[f"{trajectory}__times_seconds"]
            timesteps = arrays[f"{trajectory}__main_accepted_timesteps_seconds"]
            assert np.array_equal(times, np.asarray(config["output_times_seconds"]))
            assert timesteps.shape == (12,)
            assert np.min(timesteps) >= 1.25e-6
            assert np.max(timesteps) <= 2.0e-5
            assert np.max(timesteps[1:] / timesteps[:-1]) <= 2.0 + 1.0e-15


def test_strict_shadow_response_contract_passes() -> None:
    summary = _read_json(runner.SUMMARY_PATH)
    comparison = summary["strict_shadow_comparison"]
    assert comparison["passed"] is True
    assert comparison["maximum_scaled_state_response_difference"] <= 5.0e-3
    assert comparison["maximum_scaled_instantaneous_Tier_I_response_difference"] <= 5.0e-3
    assert comparison["maximum_scaled_cumulative_Tier_I_response_difference"] <= 5.0e-3
    assert comparison["state_response_history_cosine"] >= 0.90
    assert comparison["instantaneous_Tier_I_response_history_cosine"] >= 0.90
    assert comparison["cumulative_Tier_I_response_history_cosine"] >= 0.90


def test_only_second_rung_manifest_is_authorized() -> None:
    summary = _read_json(runner.SUMMARY_PATH)
    assert summary["authorized_next"] == (
        "WP10c9d6c7c3b5c2a_second_duration_rung_manifest"
    )
    assert summary["second_duration_rung_manifest_authorized"] is True
    assert summary["second_duration_rung_propagation_authorized"] is False
    assert summary["fixed_q_micro_solver_authorized"] is False
    assert summary["reduced_slow_evolution_authorized"] is False


def test_canonical_hashes_close() -> None:
    expected = {}
    for line in (runner.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        digest, name = line.split("  ", 1)
        expected[name] = digest
    assert expected
    for name, digest in expected.items():
        assert _sha256(runner.CANONICAL_DIRECTORY / name) == digest
