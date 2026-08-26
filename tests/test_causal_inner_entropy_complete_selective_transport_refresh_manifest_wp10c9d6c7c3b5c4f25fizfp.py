from __future__ import annotations

import hashlib

import run_causal_inner_entropy_complete_selective_transport_refresh_manifest_wp10c9d6c7c3b5c4f25fizfp as target


def test_transport_rejection_is_locked() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["classification"] == target.parent.FAIL_CLASSIFICATION
    assert not validated["summary"]["passed"]
    assert validated["metrics"]["all_truth_physical_gates_passed"]


def test_saved_evidence_selects_two_fields() -> None:
    diagnostics = target._evidence_diagnostics()
    assert diagnostics["minimal_selected_field_indices"] == [0, 1]
    assert diagnostics["minimal_selected_field_names"] == ["lnSigma", "beta_phi"]
    assert diagnostics["selected_maximum_probe_relative_defect"] <= 5.0e-2


def test_selective_refresh_is_prospective_and_fail_closed() -> None:
    contract = target._contract()
    assert contract["selective_colored_refresh"]["new_colored_truth_calls"] == 12
    assert contract["blind_validation"]["new_JVP_truth_calls"] == 8
    assert contract["acquisition_cost"]["selectively_refreshed_patch_truth_calls"] == 21
    assert contract["preserved_rejection"]["no_gate_relaxed"]
    assert not contract["claim_boundary"]["complete_cycle_execution_authorized"]


def test_canonical_package_closes_if_present() -> None:
    directory = target.CANONICAL_DIRECTORY
    if not directory.exists():
        return
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
    summary = target._utils()._read_json(directory / "summary.json")
    assert summary["unchanged_transport_rejection_preserved"]
    assert summary["selectively_refreshed_third_patch_execution_authorized"]
    assert not summary["complete_cycle_execution_authorized"]
