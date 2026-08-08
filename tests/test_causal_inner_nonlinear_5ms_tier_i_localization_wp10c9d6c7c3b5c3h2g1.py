from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_nonlinear_5ms_tier_i_localization_wp10c9d6c7c3b5c3h2g1 as h2g1  # noqa: E402


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_operator_neutral_localization_completed():
    summary = _read(h2g1.SUMMARY_PATH)
    assert summary["passed"]
    assert not summary["operator_changed"]
    assert not summary["production_defaults_changed"]
    assert not summary["propagation_executed"]
    assert summary["localization"]["common_parent_map"][
        "maximum_common_parent_incoming_excision_characteristics"
    ] == 0


def test_balance_identities_and_decision_are_self_consistent():
    summary = _read(h2g1.SUMMARY_PATH)
    result = summary["localization"]
    for report in result["net_drive_balance"].values():
        assert report["identity_closure_defect"] <= 1.0e-12
        assert report["coarse_middle"]["decomposition_closure_defect"] <= 1.0e-12
        assert report["middle_fine"]["decomposition_closure_defect"] <= 1.0e-12
    if result["common_parent_map"]["localized_to_layout_native_export_map"]:
        assert "layout_native" in summary["classification"]
    elif result["stable_term_localizations"].get("mass") == "inner_flux":
        assert "inner_face" in summary["classification"]
        mass = result["net_drive_balance"]["mass"]
        for pair in ("coarse_middle", "middle_fine"):
            assert mass[pair]["dominant_term"] == "inner_flux"
            assert mass[pair]["dominant_fraction"] >= 0.99
            assert mass[pair]["dominant_alignment"] >= 0.99


def test_schedule_and_nonlinear_remainder_are_excluded():
    result = _read(h2g1.SUMMARY_PATH)["localization"]
    tangent = result["tangent_nonlinear_pair_error"]
    assert tangent["schedule_or_nonlinear_remainder_dominance_excluded"]
    assert tangent["nonlinear_remainder_pair_fraction"] <= 0.01
    assert tangent["tangent_actual_pair_error_cosine"] >= 0.999
    assert tangent["sampled_temporal_uncertainty_fraction"] <= 0.10
    windows = result["time_window_error_energy"]
    assert windows["stable_time_window_localization"]
    assert windows["selected_window"] == "late"


def test_reduction_and_later_duration_remain_blocked():
    summary = _read(h2g1.SUMMARY_PATH)
    assert not summary["five_ms_spatial_convergence_certified"]
    assert not summary["fourth_duration_rung_manifest_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    assert summary["authorized_next"].startswith("WP10c9d6c7c3b5c3h2h")


def test_canonical_hashes_close():
    entries = {}
    for line in (h2g1.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        digest, name = line.split("  ", 1)
        entries[name] = digest
    assert entries
    for name, digest in entries.items():
        assert _sha256(h2g1.CANONICAL_DIRECTORY / name) == digest
