from __future__ import annotations

import hashlib

import run_causal_inner_entropy_complete_pathwise_macro_atlas_expansion_manifest_wp10c9d6c7c3b5c4f25fizfl as target


def test_parent_certifies_only_bounded_patch_expansion() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["exact_affine_macro_integrator_certified"]
    assert not validated["summary"]["complete_cycle_execution_authorized"]


def test_second_patch_budget_and_claim_boundary_are_frozen() -> None:
    contract = target._contract()
    assert contract["budgets"]["maximum_new_truth_operator_calls"] == 39
    assert contract["overlap_and_dynamic_validation"]["absolute_elapsed_endpoint_seconds"] == 8.0e-3
    assert contract["patch_2_construction"]["maximum_independent_JVP_relative_defect"] == 5.0e-2
    assert not contract["claim_boundary"]["unbounded_pathwise_continuation_authorized"]
    assert not contract["claim_boundary"]["complete_cycle_execution_authorized"]


def test_canonical_package_closes_if_present() -> None:
    directory = target.CANONICAL_DIRECTORY
    if not directory.exists(): return
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
    summary = target._utils()._read_json(directory / "summary.json")
    assert summary["definitions_only"]
    assert summary["second_patch_execution_authorized"]
    assert not summary["complete_cycle_execution_authorized"]
