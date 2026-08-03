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

import run_causal_inner_nonlinear_duration_controller_validation_wp10c9d6c7c3b5b as runner


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_parent_manifest_and_historical_classification_are_preserved() -> None:
    summary = _read_json(runner.SUMMARY_PATH)
    assert summary["parent_classification_preserved"] == (
        "variable_step_monolithic_duration_controller_manifest_frozen_"
        "short_horizon_controller_validation_authorized"
    )
    assert summary["operator_changed"] is False
    assert summary["production_defaults_changed"] is False


def test_both_controller_trajectories_pass_without_retries() -> None:
    summary = _read_json(runner.SUMMARY_PATH)
    assert summary["passed"] is True
    assert summary["all_trajectory_methods_passed"] is True
    assert set(summary["trajectory_reports"]) == {"base", "perturbed"}
    for report in summary["trajectory_reports"].values():
        assert report["passed"] is True
        assert report["accepted_BDF1_steps"] == 1
        assert report["accepted_BDF2_steps"] == 5
        assert report["rejected_attempts"] == 0
        assert report["maximum_local_error_estimate"] <= 2.5e-4
        assert report["sum_local_error_estimates"] <= 5.0e-3
        assert report["maximum_scaled_residual"] <= 1.0e-10
        assert report["maximum_discrete_ledger_defect"] <= 1.0e-12
        assert report["maximum_mapped_endpoint_path_closure_defect"] <= 1.0e-9
        assert report["minimum_path_reconstruction_factor"] >= 1.0
        assert report["maximum_incoming_excision_characteristics"] == 0
        assert report["checkpoint_roundtrip_bitwise"] is True
        assert report["split_restart_replay_bitwise"] is True


def test_timestep_sequence_respects_frozen_controller() -> None:
    config = _read_json(runner.CONFIG_PATH)
    contract = config["controller_contract"]
    with np.load(runner.DECISIVE_ARRAYS, allow_pickle=False) as arrays:
        for trajectory in ("base", "perturbed"):
            timesteps = arrays[f"{trajectory}__accepted_timesteps_seconds"]
            retries = arrays[f"{trajectory}__retries"]
            assert timesteps.shape == (5,)
            assert np.all(timesteps >= contract["minimum_timestep_seconds"])
            assert np.all(timesteps <= contract["maximum_timestep_seconds"])
            assert np.all(
                timesteps[1:] / timesteps[:-1]
                <= contract["maximum_BDF2_step_ratio"] + 1.0e-15
            )
            assert np.array_equal(retries, np.zeros(5, dtype=int))


def test_correct_coupling_face_and_reference_comparison_pass() -> None:
    config = _read_json(runner.CONFIG_PATH)
    summary = _read_json(runner.SUMMARY_PATH)
    contract = config["controller_contract"]
    assert contract["coupling_face_contract"][runner.LAYOUT] == 48
    validation = summary["validation"]
    assert validation["passed"] is True
    assert validation["maximum_scaled_state_difference"] <= 5.0e-3
    assert validation["scaled_state_RMS_difference"] <= 5.0e-3
    assert validation["maximum_scaled_instantaneous_Tier_I_difference"] <= 5.0e-3
    assert validation["maximum_scaled_cumulative_Tier_I_difference"] <= 5.0e-3
    assert validation["state_history_cosine"] >= 0.9
    assert validation["instantaneous_Tier_I_history_cosine"] >= 0.9
    assert validation["cumulative_Tier_I_history_cosine"] >= 0.9


def test_decisive_arrays_have_frozen_output_shapes() -> None:
    with np.load(runner.DECISIVE_ARRAYS, allow_pickle=False) as arrays:
        for trajectory in ("base", "perturbed"):
            assert arrays[f"{trajectory}__times_seconds"].shape == (5,)
            assert arrays[f"{trajectory}__states"].shape[0] == 5
            assert arrays[f"{trajectory}__states"].shape[-1] == 5
            assert arrays[f"{trajectory}__direct_exports"].shape == (5, 13)
        assert arrays["controller_state_response"].shape[0] == 5
        assert arrays["controller_instantaneous_export_response"].shape == (5, 13)
        assert arrays["controller_cumulative_export_response"].shape == (5, 13)


def test_only_first_duration_manifest_is_authorized() -> None:
    summary = _read_json(runner.SUMMARY_PATH)
    assert summary["authorized_next"] == (
        "WP10c9d6c7c3b5c1a_first_duration_rung_manifest"
    )
    assert summary["first_duration_rung_manifest_authorized"] is True
    assert summary["first_duration_rung_propagation_authorized"] is False
    assert summary["long_nonlinear_physical_ladder_authorized"] is False
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
