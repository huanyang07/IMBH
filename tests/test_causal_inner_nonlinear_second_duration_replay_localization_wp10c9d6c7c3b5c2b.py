from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_causal_inner_nonlinear_second_duration_replay_localization_wp10c9d6c7c3b5c2b as runner


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_historical_failure_is_preserved() -> None:
    summary = _read(runner.SUMMARY_PATH)
    assert summary["historical_replay_failure_preserved"] is True
    assert summary["parent_classification_preserved"] == (
        "second_nonlinear_duration_rung_failed_later_duration_work_blocked"
    )


def test_time_label_is_one_ulp_only() -> None:
    report = _read(runner.SUMMARY_PATH)["time_label_report"]
    assert report["bitwise_equal"] is False
    assert report["mismatch_indices"] == [9]
    assert report["maximum_spacing_units"] <= 1.0


def test_fresh_process_replay_is_roundoff_scale_not_bitwise() -> None:
    summary = _read(runner.SUMMARY_PATH)
    report = summary["direct_replay_report"]
    assert report["all_steps_passed"] is True
    assert report["restart_roundtrip_bitwise"] is True
    assert report["state_bitwise"] is False
    assert report["export_bitwise"] is False
    assert report["maximum_scaled_state_difference"] <= 1.0e-12
    assert report["maximum_scaled_export_difference"] <= 1.0e-12
    assert summary["fresh_process_replay_within_roundoff_envelope"] is True


def test_historical_combined_boolean_was_not_diagnostic() -> None:
    summary = _read(runner.SUMMARY_PATH)
    assert summary["historical_combined_boolean_short_circuited_by_time_label"] is True
    assert summary["historical_state_export_bitwise_status"] == "not_recorded_separately"


def test_only_corrected_replay_manifest_is_authorized() -> None:
    summary = _read(runner.SUMMARY_PATH)
    assert summary["passed"] is True
    assert summary["authorized_next"] == (
        "WP10c9d6c7c3b5c2c_corrected_replay_contract_manifest"
    )
    assert summary["later_duration_rungs_authorized"] is False
    assert summary["fixed_q_micro_solver_authorized"] is False
    assert summary["reduced_slow_evolution_authorized"] is False


def test_canonical_hashes_close() -> None:
    expected = {}
    for line in (runner.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        expected[name] = digest
    assert expected
    for name, digest in expected.items():
        assert hashlib.sha256(
            (runner.CANONICAL_DIRECTORY / name).read_bytes()
        ).hexdigest() == digest
