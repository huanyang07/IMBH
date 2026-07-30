"""Canonical evidence checks for WP10c9d6c7c2b1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_one_way_uniform_scattering_wp10c9d6c7c2b1"
)


def _read_json(name: str) -> dict:
    return json.loads((DIRECTORY / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_canonical_evidence_is_present_and_self_consistent() -> None:
    required = (
        "config.json",
        "summary.json",
        "provenance.json",
        "decisive_arrays.npz",
        "SHA256SUMS.txt",
    )
    for name in required:
        assert (DIRECTORY / name).is_file()
    summary = _read_json("summary.json")
    config = _read_json("config.json")
    provenance = _read_json("provenance.json")
    assert summary["work_package"] == "WP10c9d6c7c2b1"
    assert config["work_package"] == "WP10c9d6c7c2b1"
    assert provenance["work_package"] == "WP10c9d6c7c2b1"
    assert summary["operator_changed"] is False
    assert summary["propagation_executed"] is True
    assert summary["reflection_coefficient_defined"] is False
    assert config["root_sum_square_used"] is False
    assert config["slow_impact_threshold_used"] is False
    assert summary["decisive_arrays_sha256"] == _sha256(
        DIRECTORY / "decisive_arrays.npz"
    )
    with np.load(DIRECTORY / "decisive_arrays.npz") as arrays:
        assert np.array_equal(arrays["reference_levels"], (98, 196, 392))
        assert arrays["primary_times_seconds"].shape == (513,)


def test_binding_decision_matches_measured_gates() -> None:
    summary = _read_json("summary.json")
    decision = summary["binding_decision"]
    expected = bool(
        decision["method_passed"]
        and decision["tier_I_passed"]
        and decision["tier_II_passed"]
        and decision["amplitude_and_null_controls_passed"]
    )
    assert summary["passed"] is expected
    assert decision["uniform_c2b1_passed"] is expected
    assert decision["one_way_embedded_c2c1_authorized"] is expected
    assert decision["bidirectional_scattering_authorized"] is False
    assert decision["nonlinear_authorized"] is False
    assert decision["fixed_Q_or_reduction_authorized"] is False
    if expected:
        assert (
            summary["classification"]
            == "one_way_uniform_scattering_certified_"
            "embedded_discrimination_authorized"
        )
    else:
        assert (
            summary["classification"]
            == "one_way_uniform_scattering_validation_failed_"
            "embedded_discrimination_blocked"
        )


def test_tier_two_ledgers_and_uncertainty_contract() -> None:
    summary = _read_json("summary.json")
    for report in summary["tier_II"].values():
        assert report["maximum_energy_ledger_relative_defect"] <= 1.0e-10
        uncertainty = report["uncertainty_components"]
        assert uncertainty["RSS_used"] is False
        assert np.isclose(
            uncertainty["conservative_sum"],
            uncertainty["algebraic_continuum_projection_subspace"]
            + uncertainty["window_and_time_sampling"]
            + uncertainty["restart_and_roundoff"],
        )
        direction = report["transmission"]["direction_classification"]
        assert direction in {
            "binding_pass",
            "binding_fail",
            "direction_not_certifying_because_error_is_below_observability",
        }
