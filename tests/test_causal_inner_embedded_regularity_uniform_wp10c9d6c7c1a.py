from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (
    causal_array_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = (
    ROOT
    / "results/canonical/"
    "causal_inner_embedded_regularity_uniform_wp10c9d6c7c1a"
)
SUMMARY = CANONICAL / "summary.json"
DECISIVE = CANONICAL / "decisive_arrays.npz"

MANIFEST_SHA256 = (
    "b230ce7a3c7e7546d0d706ee8f9bcfa3102c6c69be5f67a29aa451e1b5d9706b"
)
PROFILE_NAMES = {
    "p4__inward_shear",
    "p4__outward_shear",
    "p3_buffer45__inward_shear",
    "p3_buffer45__outward_shear",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_wp10c9d6c7c1a_preserves_frozen_contract() -> None:
    summary = _summary()
    assert summary["manifest_sha256"] == MANIFEST_SHA256
    assert summary["parent_classification_preserved"]
    assert summary["c7b_rejection_preserved"]
    assert not summary["operator_changed"]
    assert summary["propagation_executed"]
    assert summary["method_passed"]
    assert (
        summary["maximum_exact_integral_relative_solve_residual"]
        <= 1.0e-12
    )


def test_wp10c9d6c7c1a_evaluates_all_frozen_variants() -> None:
    summary = _summary()
    decision = summary["prospective_decision"]
    reports = decision["variant_reports"]
    assert len(reports) == 16
    assert {report["base_profile"] for report in reports.values()} == (
        PROFILE_NAMES
    )
    assert {
        abs(float(report["multiplier"])) for report in reports.values()
    } == {0.5, 1.0}
    assert all(
        report["route"] in {"direct_contract", "failed"}
        for report in reports.values()
    )


def test_wp10c9d6c7c1a_decision_and_authorization_are_consistent() -> None:
    summary = _summary()
    decision = summary["prospective_decision"]
    failed = sorted(
        name
        for name, report in decision["variant_reports"].items()
        if not report["passed"]
    )
    assert failed == decision["failed_variants"]
    assert decision["direct_variant_count"] == 16 - len(failed)
    assert decision["passed"] == (not failed)
    assert summary["passed"] == decision["passed"]
    assert summary["uniform_controls_certified"] == summary["passed"]
    assert (
        summary["embedded_regularity_discrimination_authorized"]
        == summary["passed"]
    )
    if summary["passed"]:
        assert summary["classification"] == (
            "endpoint_interface_regularity_uniform_controls_certified_"
            "embedded_discrimination_authorized"
        )
        assert (
            summary["authorized_next"]
            == "WP10c9d6c7c1b_embedded_regularity_discrimination"
        )
    else:
        assert summary["classification"] == (
            "endpoint_interface_regularity_uniform_controls_failed"
        )
        assert summary["authorized_next"] is None


def test_wp10c9d6c7c1a_certifies_every_uniform_control_directly() -> None:
    summary = _summary()
    decision = summary["prospective_decision"]
    comparison = summary["historical_direct_contract_report"]
    assert summary["passed"]
    assert decision["passed"]
    assert decision["direct_variant_count"] == 16
    assert decision["failed_variants"] == []
    assert comparison["passed"]
    assert comparison["all_packets_passed"]
    assert comparison["failed_packets"] == []
    assert all(
        report["route"] == "direct_contract"
        for report in decision["variant_reports"].values()
    )
    assert all(
        report["state_reference"]["passed"]
        and report["instantaneous_exports"]["passed"]
        and report["cumulative_exports"]["passed"]
        for report in comparison["packet_reports"].values()
    )


def test_wp10c9d6c7c1a_never_authorizes_downstream_physics() -> None:
    summary = _summary()
    assert not summary["bounded_nonlinear_common_mode_authorized"]
    assert not summary["nonlinear_physical_trajectory_authorized"]
    assert not summary["production_operator_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_wp10c9d6c7c1a_canonical_hashes() -> None:
    summary = _summary()
    assert _sha256(DECISIVE) == summary["decisive_arrays_sha256"]
    for relative, expected in summary[
        "implementation_source_hashes"
    ].items():
        assert _sha256(ROOT / relative) == expected
    with np.load(DECISIVE, allow_pickle=False) as source:
        assert set(source.files) == set(summary["decisive_array_hashes"])
        for name in source.files:
            assert (
                causal_array_sha256(source[name])
                == summary["decisive_array_hashes"][name]
            )
