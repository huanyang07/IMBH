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
    "causal_inner_embedded_regularity_wp10c9d6c7c1b"
)
SUMMARY = CANONICAL / "summary.json"
DECISIVE = CANONICAL / "decisive_arrays.npz"
MANIFEST_SHA256 = (
    "b230ce7a3c7e7546d0d706ee8f9bcfa3102c6c69be5f67a29aa451e1b5d9706b"
)
PROFILES = {
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


def test_wp10c9d6c7c1b_preserves_the_complete_evidence_chain() -> None:
    summary = _summary()
    assert summary["manifest_sha256"] == MANIFEST_SHA256
    assert summary["c7b_rejection_preserved"]
    assert summary["c7c0_manifest_preserved"]
    assert summary["c7c1a_uniform_certification_preserved"]
    assert summary["historical_classifications_preserved"]
    assert not summary["operator_changed"]


def test_wp10c9d6c7c1b_method_and_integral_gates_pass() -> None:
    summary = _summary()
    assert summary["propagation_executed"]
    assert summary["method_passed"]
    assert all(
        report["passed"] for report in summary["method_reports"].values()
    )
    assert (
        summary["maximum_exact_integral_relative_solve_residual"]
        <= 1.0e-12
    )


def test_wp10c9d6c7c1b_evaluates_all_frozen_variants() -> None:
    summary = _summary()
    direct = summary["historical_direct_contract_report"]
    decision = summary["prospective_decision"]
    assert len(direct["packet_reports"]) == 16
    assert set(decision["profile_reports"]) == PROFILES
    assert all(
        len(report["variant_ids"]) == 4
        for report in decision["profile_reports"].values()
    )
    assert decision["direct_variant_count"] == (
        16 - len(direct["failed_packets"])
    )
    assert decision["alternate_variant_count"] == 0


def test_wp10c9d6c7c1b_frozen_decision_table_is_consistent() -> None:
    summary = _summary()
    decision = summary["prospective_decision"]
    p4_passed = all(
        decision["profile_reports"][name]["passed"]
        for name in ("p4__inward_shear", "p4__outward_shear")
    )
    buffered_passed = all(
        decision["profile_reports"][name]["passed"]
        for name in (
            "p3_buffer45__inward_shear",
            "p3_buffer45__outward_shear",
        )
    )
    assert decision["p4_active_endpoint_class_passed"] == p4_passed
    assert (
        decision["p3_exact_zero_buffer_class_passed"]
        == buffered_passed
    )
    assert decision["passed"] == (p4_passed and buffered_passed)
    assert summary["passed"] == decision["passed"]
    assert (
        summary["embedded_regularized_profile_class_certified"]
        == summary["passed"]
    )


def test_wp10c9d6c7c1b_preserves_direct_pass_but_rejects_diagnostics() -> None:
    summary = _summary()
    direct = summary["historical_direct_contract_report"]
    coupling = summary["coupling_diagnostic_report"]
    decision = summary["prospective_decision"]
    assert direct["passed"]
    assert direct["all_packets_passed"]
    assert direct["failed_packets"] == []
    assert decision["direct_variant_count"] == 16
    assert not coupling["passed"]
    assert coupling["coupling_face_flux_convergence_passed"]
    assert not coupling["energy_convergence_passed"]
    assert not coupling["interface_state_convergence_passed"]
    assert not summary["passed"]
    assert (
        summary["classification"]
        == "no_regularized_embedded_profile_class_selected"
    )
    assert decision["decision_branch"] == "both_prospective_classes_fail"
    assert summary["authorized_next"] is None


def test_wp10c9d6c7c1b_failed_diagnostics_still_contract() -> None:
    coupling = _summary()["coupling_diagnostic_report"]
    for report in coupling["common_face_flux_reports"].values():
        assert report["48"]["passed"]
        assert not report["48"]["active"]
    failed_energy = [
        channel
        for report in coupling["energy_reports"].values()
        for channel in report["channels"].values()
        if not channel["passed"]
    ]
    assert len(failed_energy) == 2
    assert min(item["observed_rms_order"] for item in failed_energy) > 2.0
    failed_interfaces = [
        report
        for report in coupling["interface_state_reports"].values()
        if not report["passed"]
    ]
    assert len(failed_interfaces) == 3
    assert min(item["observed_rms_order"] for item in failed_interfaces) > 2.0


def test_wp10c9d6c7c1b_never_authorizes_downstream_physics() -> None:
    summary = _summary()
    assert not summary["bounded_nonlinear_common_mode_authorized"]
    assert not summary["nonlinear_physical_trajectory_authorized"]
    assert not summary["production_operator_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_wp10c9d6c7c1b_canonical_hashes() -> None:
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
