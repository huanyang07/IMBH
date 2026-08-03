from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_causal_inner_nonlinear_first_duration_rung_manifest_wp10c9d6c7c3b5c1a as runner


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_parent_controller_certificate_is_preserved() -> None:
    summary = _read_json(runner.SUMMARY_PATH)
    assert summary["parent_classification_preserved"] == (
        "short_horizon_variable_step_controller_certified_"
        "first_duration_rung_manifest_authorized"
    )


def test_manifest_is_definitions_only() -> None:
    summary = _read_json(runner.SUMMARY_PATH)
    assert summary["passed"] is True
    assert summary["propagation_executed"] is False
    assert summary["operator_changed"] is False
    assert summary["production_defaults_changed"] is False


def test_first_rung_outputs_and_restart_are_frozen() -> None:
    manifest = _read_json(runner.MANIFEST_PATH)
    assert manifest["horizon_seconds"] == 2.0e-4
    assert len(manifest["output_times_seconds"]) == 11
    assert manifest["output_times_seconds"][0] == 0.0
    assert manifest["output_times_seconds"][-1] == 2.0e-4
    assert manifest["restart_time_seconds"] == 1.0e-4
    assert manifest["strict_shadow"]["start_time_seconds"] == 1.6e-4


def test_controller_and_correct_face_contract_are_preserved() -> None:
    manifest = _read_json(runner.MANIFEST_PATH)
    controller = manifest["main_controller"]
    assert manifest["coupling_face"] == 48
    assert controller["coupling_face_contract"][runner.LAYOUT] == 48
    assert controller["maximum_BDF2_step_ratio"] == 2.0
    assert controller["error_estimator"]["local_tolerance"] == 2.5e-4
    assert manifest["main_rung_error_budget"][
        "maximum_sum_of_accepted_error_estimates"
    ] == 5.0e-3


def test_strict_shadow_is_prospectively_stricter() -> None:
    manifest = _read_json(runner.MANIFEST_PATH)
    main = manifest["main_controller"]
    strict = manifest["strict_shadow"]["controller"]
    assert strict["maximum_timestep_seconds"] < main["maximum_timestep_seconds"]
    assert (
        strict["error_estimator"]["local_tolerance"]
        < main["error_estimator"]["local_tolerance"]
    )
    assert manifest["strict_shadow"][
        "same_saved_state_and_BDF_history_as_main"
    ] is True


def test_only_first_rung_propagation_is_authorized() -> None:
    summary = _read_json(runner.SUMMARY_PATH)
    assert summary["authorized_next"] == (
        "WP10c9d6c7c3b5c1_first_duration_rung_propagation"
    )
    assert summary["first_duration_rung_propagation_authorized"] is True
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
