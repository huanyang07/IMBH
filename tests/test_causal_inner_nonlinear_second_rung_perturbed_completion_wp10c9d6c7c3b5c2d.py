from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_nonlinear_second_rung_perturbed_completion_wp10c9d6c7c3b5c2d as runner


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_perturbed_method_records_binding_replay_failure() -> None:
    report = _read(runner.SUMMARY_PATH)["perturbed_trajectory_report"]
    assert report["passed"] is False
    replay = report["separate_replay_report"]
    assert replay["canonical_time_labels_bitwise"] is True
    assert replay["accumulated_time_labels_bitwise"] is False
    assert replay["maximum_accumulated_time_spacing_units"] == 1.0
    assert replay["primitive_states_bitwise"] is False
    assert replay["direct_Tier_I_exports_bitwise"] is False
    assert replay["primitive_history_bitwise"] is False
    assert replay["mapped_history_bitwise"] is False
    assert replay["height_history_bitwise"] is False
    assert replay["previous_timesteps_bitwise"] is False


def test_strict_shadow_response_passes() -> None:
    shadow = _read(runner.SUMMARY_PATH)["strict_shadow_comparison"]
    assert shadow["passed"] is True
    assert shadow["maximum_scaled_state_response_difference"] <= 5.0e-3
    assert shadow["maximum_scaled_instantaneous_Tier_I_response_difference"] <= 5.0e-3
    assert shadow["maximum_scaled_cumulative_Tier_I_response_difference"] <= 5.0e-3


def test_later_duration_and_reduction_remain_blocked() -> None:
    summary = _read(runner.SUMMARY_PATH)
    assert summary["passed"] is False
    assert summary["classification"] == (
        "second_rung_perturbed_completion_failed_later_duration_blocked"
    )
    assert summary["authorized_next"] == "none"
    assert summary["historical_c2_classification_preserved"] == (
        "second_nonlinear_duration_rung_failed_later_duration_work_blocked"
    )
    assert summary["third_duration_rung_manifest_authorized"] is False
    assert summary["third_duration_rung_propagation_authorized"] is False
    assert summary["fixed_q_micro_solver_authorized"] is False
    assert summary["reduced_slow_evolution_authorized"] is False


def test_canonical_hashes_close() -> None:
    for line in (runner.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256(
            (runner.CANONICAL_DIRECTORY / name).read_bytes()
        ).hexdigest() == digest
