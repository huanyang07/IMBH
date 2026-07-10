from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs/tables/m5_eta_endpoint_validity_audit_98p125_N164.json"
PROFILES = ROOT / "outputs/tables/m5_eta_endpoint_validity_audit_98p125_N164_profiles.json"
NOTE = ROOT / "Note/CODEX_MDOT5_ENDPOINT_VALIDITY_AND_EXPONENT_AUDIT_RESULTS.md"
FIGURE = ROOT / "outputs/figures/m5_eta_endpoint_validity_audit_98p125_N164.png"


def _summary() -> dict:
    return json.loads(SUMMARY.read_text())


def test_endpoint_validity_artifacts_exist_and_numerical_rows_remain_finite() -> None:
    for path in (SUMMARY, PROFILES, NOTE, FIGURE):
        assert path.exists()
        assert path.stat().st_size > 0

    result = _summary()
    numerical = result["numerical_gates"]
    assert numerical["homogeneous_max"] <= 5.0e-5
    assert numerical["mass_homogeneous_max"] <= 3.0e-6
    assert numerical["vertical_tau_identity_max"] <= 1.0e-12
    assert numerical["mass_differential_max_conditioned"] > numerical["mass_homogeneous_max"]


def test_endpoint_first_model_validity_failure_precedes_formal_limit() -> None:
    result = _summary()
    gates = result["validity_gates"]
    first = gates["first_model_validity_failure"]
    assert first["metric"] == "L_u_over_H"
    assert 223.0 < float(first["R_rg"]) < 224.0
    assert float(first["value"]) < 1.0
    assert float(first["R_rg"]) < float(result["target"]["R_limit_rg"])

    assert gates["radially_optically_thick"] is None
    assert gates["vertically_optically_thick"] is None
    assert gates["non_self_gravitating"] is None
    vertical = gates["vertical_adjustment_tlayer_over_tdyn"]
    assert vertical is not None
    assert float(vertical["R_rg"]) > float(first["R_rg"])


def test_endpoint_common_window_fits_make_integrability_robust() -> None:
    result = _summary()
    fits = result["fit_summary"]
    assert fits["p_R"]["count"] == 24
    assert fits["Sigma"]["count"] == 24
    assert fits["Sigma"]["maximum"] < -0.99
    assert fits["p_R"]["minimum"] > 1.5

    mass = fits["annulus_mass_power_of_deltaR"]
    divergence = fits["Sigma_divergence_power_of_deltaR"]
    assert 0.4 < mass["minimum"] <= mass["median"] <= mass["maximum"] < 0.6
    assert 0.4 < divergence["minimum"] <= divergence["maximum"] < 0.6
    assert result["interpretation"]["local_annulus_mass_integrable"] is True
    assert result["interpretation"]["formal_endpoint_within_1d_validity"] is False
    assert result["interpretation"]["global_nonexistence_established"] is False


def test_endpoint_physical_path_is_ordered_toward_the_limit() -> None:
    result = json.loads(PROFILES.read_text())
    path = result["physical_path"]
    radii = np.asarray([row["R_rg"] for row in path], dtype=float)
    assert radii.size > 500
    assert np.all(np.isfinite(radii))
    assert np.min(np.diff(radii)) > -1.0e-3
    assert radii[0] < 195.0
    assert radii[-1] > 225.52
