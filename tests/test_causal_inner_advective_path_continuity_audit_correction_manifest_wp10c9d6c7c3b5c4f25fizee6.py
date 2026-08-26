from __future__ import annotations

import hashlib
from pathlib import Path

import run_causal_inner_advective_path_continuity_audit_correction_manifest_wp10c9d6c7c3b5c4f25fizee6 as target


def test_parent_path_certificate_is_hash_locked() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["classification"] == target.parent.PASS_CLASSIFICATION
    assert validated["summary"]["audit_correction_manifest_authorized"]


def test_correction_uses_invariant_pointwise_gates() -> None:
    contract = target._contract()
    standard = contract["corrected_mathematical_standard"]
    gates = contract["binding_gates"]
    claims = contract["claim_boundary"]
    assert standard["coarse_neighbor_subspace_cosine"] == "diagnostic_only"
    assert standard["cluster_transport_offset_and_cluster_complement_gap_binding"]
    assert gates["advective_cluster_dimension"] == 3
    assert gates["maximum_advective_cluster_transport_offset_over_c"] == 1.0e-6
    assert gates["minimum_advective_cluster_complement_gap_over_c"] == 1.0e-4
    assert gates["all_points_and_all_gates_required"] and gates["fail_closed"]
    assert claims["corrected_full_local_audit_authorized"]
    assert not claims["trajectory_authorized"]


def test_all_prior_results_remain_preserved() -> None:
    preserved = target._contract()["preserved_results"]
    assert preserved["original_complex_split_failure"]["preserved"]
    assert preserved["coarse_neighbor_overlap_failure"]["preserved"]
    assert preserved["saved_product_rule_repair"]["preserved"]
    assert preserved["path_continuity_certificate"]["preserved"]
    assert preserved["no_parent_result_reclassified"]


def test_canonical_package_closes_if_present() -> None:
    directory = target.CANONICAL_DIRECTORY
    if not directory.exists():
        return
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        assert actual == expected
    summary = target._utils()._read_json(directory / "summary.json")
    assert summary["classification"] == target.CLASSIFICATION
    assert summary["definitions_only"]
    assert summary["all_parent_results_preserved"]
    assert summary["authorized_next"] == target.AUTHORIZED_NEXT
    assert not summary["local_architecture_certified"]


def test_declared_paths_are_workspace_relative() -> None:
    for relative in (
        target.THIS_RUNNER,
        target.THIS_TEST,
        target.PHYSICAL_SOURCE,
        target.PHYSICAL_TEST,
        target.REPORT_RELATIVE,
    ):
        assert not Path(relative).is_absolute()
