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

import run_causal_inner_nonlinear_profile_breadth_spatial_localization_wp10c9d6c7c3b4c as runner


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_parent_negative_classification_is_preserved() -> None:
    summary = _read_json(runner.PARENT_DIRECTORY / "summary.json")
    assert summary["classification"] == (
        "heldout_profile_spatial_confirmation_failed_duration_extension_blocked"
    )
    assert summary["passed"] is False
    assert summary["authorized_next"] == (
        "WP10c9d6c7c3b4b3_spatial_failure_localization"
    )


def test_localization_is_analysis_only() -> None:
    config = _read_json(runner.CONFIG_PATH)
    summary = _read_json(runner.SUMMARY_PATH)
    assert config["propagation_executed"] is False
    assert summary["propagation_executed"] is False
    assert summary["operator_changed"] is False
    assert summary["production_defaults_changed"] is False


def test_common_parent_map_decision_is_complete() -> None:
    summary = _read_json(runner.SUMMARY_PATH)
    result = summary["localization"]
    assert set(result["profiles"]) == set(runner.PROFILES)
    assert isinstance(result["common_parent_export_contract_passed"], bool)
    assert isinstance(result["localized_to_layout_native_export_map"], bool)
    for report in result["profiles"].values():
        for channel in ("instantaneous", "cumulative"):
            item = report[channel]
            assert set(item) == {
                "native",
                "common_parent_map",
                "layout_native_map_defect",
                "coarse_medium_attribution",
                "medium_fine_attribution",
            }


def test_error_decomposition_closes() -> None:
    summary = _read_json(runner.SUMMARY_PATH)
    result = summary["localization"]
    assert result["maximum_error_decomposition_closure_defect"] <= 1.0e-12
    for report in result["profiles"].values():
        for channel in ("instantaneous", "cumulative"):
            for pair in (
                "coarse_medium_attribution",
                "medium_fine_attribution",
            ):
                assert report[channel][pair]["decomposition_closure_defect"] <= 1.0e-12


def test_common_parent_audits_remain_physical() -> None:
    summary = _read_json(runner.SUMMARY_PATH)
    result = summary["localization"]
    assert result["maximum_common_parent_ledger_defect"] <= 1.0e-9
    assert result["maximum_common_parent_incoming_excision_characteristics"] == 0


def test_decisive_arrays_have_all_profile_layout_channels() -> None:
    with np.load(runner.DECISIVE_ARRAYS, allow_pickle=False) as arrays:
        assert arrays["times_seconds"].shape == (5,)
        assert arrays["fixed_physical_observable_scales"].shape == (13,)
        for layout in runner.LAYOUTS:
            for profile in runner.PROFILES:
                for suffix in (
                    "native_export_response",
                    "common_parent_export_response",
                    "layout_native_export_map_defect",
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


def test_reduction_and_duration_remain_blocked() -> None:
    summary = _read_json(runner.SUMMARY_PATH)
    assert summary["heldout_spatial_convergence_certified"] is False
    assert summary["variable_step_duration_controller_manifest_authorized"] is False
    assert summary["long_nonlinear_physical_ladder_authorized"] is False
    assert summary["fixed_q_micro_solver_authorized"] is False
    assert summary["reduced_slow_evolution_authorized"] is False
