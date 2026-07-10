from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
P0 = ROOT / "results/canonical/p0_validity_ledger_outer_manifold"
SUMMARY = P0 / "outer_manifold_summary.json"
PROFILES = P0 / "outer_match_seeds.json"
NOTE = ROOT / "docs/reports/current/CODEX_MDOT5_INDEPENDENT_OUTER_MANIFOLD_RESULTS.md"
FIGURE = P0 / "m5_eta_independent_outer_manifold_98p125_N164.png"


def _summary() -> dict:
    return json.loads(SUMMARY.read_text())


def test_outer_manifold_artifacts_and_validity_surface() -> None:
    for path in (SUMMARY, PROFILES, NOTE, FIGURE):
        assert path.exists()
        assert path.stat().st_size > 0

    result = _summary()
    assert 223.0 < result["target"]["validity_R_rg"] < 224.0
    assert result["atlas"]["state_count"] >= 100
    assert result["atlas"]["accepted_root_count"] > 0
    assert result["decision"]["reached_validity_surface_count"] >= 1


def test_best_outer_match_is_conservative_but_not_promoted_to_connection() -> None:
    result = _summary()
    decision = result["decision"]
    best = decision["best_match"]

    assert decision["outcome"] == "exploratory_near_match"
    assert decision["connected"] is False
    assert decision["physical_closure_certified"] is False
    assert 1.0e-3 < best["state_max"] < 3.0e-3
    assert best["flux_max"] < 1.0e-4
    assert abs(best["flux_delta"]["advected_internal_energy_scaled"]) < 1.0e-4
    assert abs(best["flux_delta"]["angular_flux_scaled"]) < 1.0e-4


def test_local_shooting_map_records_geometric_ill_conditioning() -> None:
    shooting = _summary()["shooting_jacobian"]
    assert shooting["available"] is True
    jacobian = np.asarray(shooting["jacobian"], dtype=float)
    singular_values = np.asarray(shooting["singular_values"], dtype=float)

    assert jacobian.shape == (3, 3)
    assert np.all(np.isfinite(jacobian))
    assert np.all(np.diff(singular_values) <= 0.0)
    assert shooting["condition"] > 100.0
    assert abs(shooting["u_T_direction_cosine"]) > 0.99


def test_outer_atlas_seeds_are_not_copied_from_inner_phase_endpoint() -> None:
    profiles = json.loads(PROFILES.read_text())
    inner = np.asarray(profiles["inner_match_state"]["z"][:3], dtype=float)
    nominal = profiles["nominal_outer_seeds"]
    assert nominal
    for row in nominal:
        seed = np.asarray(row["z"][:3], dtype=float)
        assert np.max(np.abs(seed - inner)) > 1.0e-2
