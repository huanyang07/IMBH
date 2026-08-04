from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_causal_inner_nonlinear_second_duration_rung_manifest_wp10c9d6c7c3b5c2a as runner


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_parent_duration_certificate_is_preserved() -> None:
    summary = _read_json(runner.SUMMARY_PATH)
    assert summary["parent_classification_preserved"] == (
        "first_nonlinear_duration_rung_certified_"
        "second_rung_manifest_authorized"
    )


def test_manifest_is_definitions_only() -> None:
    summary = _read_json(runner.SUMMARY_PATH)
    assert summary["passed"] is True
    assert summary["propagation_executed"] is False
    assert summary["operator_changed"] is False
    assert summary["production_defaults_changed"] is False


def test_second_rung_continuation_is_frozen() -> None:
    manifest = _read_json(runner.MANIFEST_PATH)
    continuation = manifest["continuation_contract"]
    assert manifest["horizon_seconds"] == 1.0e-3
    assert len(manifest["output_times_seconds"]) == 11
    assert manifest["output_times_seconds"][-1] == 1.0e-3
    assert continuation["previous_history_time_seconds"] == 1.8e-4
    assert continuation["continuation_start_seconds"] == 2.0e-4
    assert continuation["previous_timestep_seconds"] == 2.0e-5
    assert continuation["continue_BDF2_without_new_BDF1_startup"] is True


def test_larger_step_preserves_ratio_and_correct_face_contract() -> None:
    manifest = _read_json(runner.MANIFEST_PATH)
    controller = manifest["main_controller"]
    assert manifest["coupling_face"] == 48
    assert controller["coupling_face_contract"][runner.LAYOUT] == 48
    assert controller["initial_timestep_seconds"] == 4.0e-5
    assert controller["maximum_timestep_seconds"] == 1.0e-4
    assert controller["maximum_BDF2_step_ratio"] == 2.0
    assert controller["error_estimator"]["local_tolerance"] == 2.5e-4


def test_restart_and_strict_shadow_are_prospective() -> None:
    manifest = _read_json(runner.MANIFEST_PATH)
    main = manifest["main_controller"]
    strict = manifest["strict_shadow"]["controller"]
    assert manifest["restart_time_seconds"] == 6.0e-4
    assert manifest["strict_shadow"]["start_time_seconds"] == 8.0e-4
    assert strict["maximum_timestep_seconds"] == 5.0e-5
    assert strict["maximum_timestep_seconds"] < main["maximum_timestep_seconds"]
    assert (
        strict["error_estimator"]["local_tolerance"]
        < main["error_estimator"]["local_tolerance"]
    )


def test_only_second_rung_propagation_is_authorized() -> None:
    summary = _read_json(runner.SUMMARY_PATH)
    assert summary["authorized_next"] == (
        "WP10c9d6c7c3b5c2_second_duration_rung_propagation"
    )
    assert summary["second_duration_rung_propagation_authorized"] is True
    assert summary["later_duration_rungs_authorized"] is False
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
