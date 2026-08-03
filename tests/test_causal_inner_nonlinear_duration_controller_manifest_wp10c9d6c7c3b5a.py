from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_causal_inner_nonlinear_duration_controller_manifest_wp10c9d6c7c3b5a as runner


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_parent_authorization_is_preserved() -> None:
    summary = _read_json(runner.SUMMARY_PATH)
    assert summary["parent_classification_preserved"] == (
        "heldout_spatial_export_failure_caused_by_active_face_alias_"
        "corrected_physical_face_contract_passes"
    )


def test_manifest_is_definitions_only() -> None:
    summary = _read_json(runner.SUMMARY_PATH)
    assert summary["passed"] is True
    assert summary["propagation_executed"] is False
    assert summary["operator_changed"] is False
    assert summary["production_defaults_changed"] is False


def test_controller_respects_variable_step_stability() -> None:
    contract = _read_json(runner.MANIFEST_PATH)["controller_contract"]
    assert contract["maximum_BDF2_step_ratio"] <= contract["analytic_stability_bound"]
    assert contract["minimum_timestep_seconds"] > 0.0
    assert contract["minimum_timestep_seconds"] <= contract["initial_timestep_seconds"]
    assert contract["initial_timestep_seconds"] <= contract["maximum_timestep_seconds"]
    assert contract["error_estimator"]["accepted_branch_error_multiplier"] == 4.0 / 3.0


def test_physical_face_indices_are_explicit() -> None:
    contract = _read_json(runner.MANIFEST_PATH)["controller_contract"]
    assert list(contract["coupling_face_contract"].values()) == [48, 96, 192]


def test_short_horizon_reference_is_independent_and_frozen() -> None:
    contract = _read_json(runner.MANIFEST_PATH)["short_horizon_validation_contract"]
    assert contract["background_and_perturbed_trajectories_required"] is True
    assert contract["independent_reference"]["timestep_seconds"] == 2.5e-6
    assert contract["maximum_controller_to_reference_scaled_state_difference"] == 5.0e-3
    assert contract["maximum_controller_to_reference_scaled_Tier_I_difference"] == 5.0e-3
    assert contract["correct_active_coupling_face_required"] is True


def test_duration_ladder_is_ordered_and_conditionally_blocked() -> None:
    manifest = _read_json(runner.MANIFEST_PATH)
    horizons = tuple(item["horizon_seconds"] for item in manifest["duration_ladder"])
    assert horizons == runner.DURATION_RUNGS_SECONDS
    assert all(right > left for left, right in zip(horizons, horizons[1:]))
    assert manifest["stage_authorization"]["duration_rungs_authorized_now"] is False
    assert manifest["stage_authorization"]["each_later_rung_requires_previous_binding_pass"] is True


def test_only_short_controller_validation_is_authorized() -> None:
    summary = _read_json(runner.SUMMARY_PATH)
    assert summary["authorized_next"] == (
        "WP10c9d6c7c3b5b_short_horizon_variable_step_controller_validation"
    )
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
