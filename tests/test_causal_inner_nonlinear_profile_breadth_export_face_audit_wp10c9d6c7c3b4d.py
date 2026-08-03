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

import run_causal_inner_nonlinear_profile_breadth_export_face_audit_wp10c9d6c7c3b4d as runner


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_historical_classifications_are_preserved() -> None:
    summary = _read_json(runner.SUMMARY_PATH)
    assert summary["parent_classification_preserved"] == (
        "spatial_failure_localized_to_layout_native_export_map_"
        "common_parent_map_passes"
    )
    assert summary["failed_b4b3_classification_preserved"] == (
        "heldout_profile_spatial_confirmation_failed_duration_extension_blocked"
    )


def test_package_is_analysis_only() -> None:
    config = _read_json(runner.CONFIG_PATH)
    summary = _read_json(runner.SUMMARY_PATH)
    assert config["propagation_executed"] is False
    assert summary["propagation_executed"] is False
    assert summary["operator_changed"] is False
    assert summary["production_defaults_changed"] is False


def test_active_face_mapping_matches_one_physical_radius() -> None:
    summary = _read_json(runner.SUMMARY_PATH)
    mappings = summary["audit"]["face_mappings"]
    assert [
        mappings[layout]["correct_active_coupling_face_index"]
        for layout in runner.LAYOUTS
    ] == [48, 96, 192]
    assert all(
        mappings[layout]["correct_radius_relative_defect"] <= 1.0e-15
        for layout in runner.LAYOUTS
    )
    assert mappings[runner.LAYOUTS[0]]["legacy_radius_relative_displacement"] == 0.0
    assert mappings[runner.LAYOUTS[1]]["legacy_radius_relative_displacement"] < -0.6
    assert mappings[runner.LAYOUTS[2]]["legacy_radius_relative_displacement"] < -0.7


def test_face_alias_cause_and_corrected_contract_pass() -> None:
    summary = _read_json(runner.SUMMARY_PATH)
    audit = summary["audit"]
    assert summary["passed"] is True
    assert audit["legacy_helper_hardcodes_parent_face"] is True
    assert audit["physical_face_contract_passed"] is True
    assert audit["all_corrected_spatial_contracts_passed"] is True
    assert audit["face_alias_cause_proved"] is True
    for report in audit["profiles"].values():
        assert report["passed"] is True
        assert report["instantaneous"]["corrected_physical_face"]["passed"] is True
        assert report["cumulative"]["corrected_physical_face"]["passed"] is True


def test_alias_attribution_closes() -> None:
    audit = _read_json(runner.SUMMARY_PATH)["audit"]
    assert audit["minimum_face_alias_alignment_with_legacy_error"] >= 0.95
    assert audit["maximum_corrected_to_legacy_error_ratio"] <= 0.25
    assert audit["maximum_error_decomposition_closure_defect"] <= 1.0e-12


def test_corrected_export_audits_remain_physical() -> None:
    audit = _read_json(runner.SUMMARY_PATH)["audit"]
    assert audit["maximum_corrected_export_ledger_defect"] <= 1.0e-9
    assert audit["maximum_corrected_incoming_excision_characteristics"] == 0


def test_decisive_arrays_cover_every_profile_and_layout() -> None:
    with np.load(runner.DECISIVE_ARRAYS, allow_pickle=False) as arrays:
        assert arrays["times_seconds"].shape == (5,)
        assert arrays["fixed_physical_observable_scales"].shape == (13,)
        assert arrays["correct_active_coupling_face_indices"].tolist() == [48, 96, 192]
        for layout in runner.LAYOUTS:
            for profile in runner.PROFILES:
                for suffix in (
                    "legacy_wrong_face_response",
                    "corrected_face_response",
                    "face_alias_defect",
                ):
                    assert arrays[f"{layout}__{profile}__{suffix}"].shape == (5, 13)


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


def test_only_duration_manifest_is_authorized() -> None:
    summary = _read_json(runner.SUMMARY_PATH)
    assert summary["heldout_spatial_convergence_certified"] is True
    assert summary["variable_step_duration_controller_manifest_authorized"] is True
    assert summary["long_nonlinear_physical_ladder_authorized"] is False
    assert summary["fixed_q_micro_solver_authorized"] is False
    assert summary["reduced_slow_evolution_authorized"] is False
