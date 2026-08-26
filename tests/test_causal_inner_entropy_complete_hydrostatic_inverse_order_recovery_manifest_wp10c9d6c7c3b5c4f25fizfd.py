from __future__ import annotations

import hashlib

import run_causal_inner_entropy_complete_hydrostatic_inverse_order_recovery_manifest_wp10c9d6c7c3b5c4f25fizfd as target


def test_fixed_step_rejection_is_preserved() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["classification"] == target.parent.FAIL_CLASSIFICATION
    assert not validated["summary"]["passed"]


def test_order_recovery_is_independent_and_fail_closed() -> None:
    contract = target._contract()
    audit = contract["order_aware_inverse_audit"]
    assert tuple(audit["ordered_central_steps"]) == (8.0e-6, 4.0e-6, 2.0e-6)
    assert audit["minimum_global_worst_defect_order"] == 1.8
    assert audit["maximum_each_Richardson_JVP_relative_defect"] == 2.0e-5
    assert audit["maximum_Richardson_pair_relative_disagreement"] == 2.0e-5
    assert audit["new_seven_field_operator_calls"] == 0
    assert not contract["claim_boundary"]["slow_flux_atlas_authorized"]


def test_canonical_package_closes_if_present() -> None:
    directory = target.CANONICAL_DIRECTORY
    if not directory.exists():
        return
    for line in (directory / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
    summary = target._utils()._read_json(directory / "summary.json")
    assert summary["definitions_only"]
    assert summary["fixed_step_inverse_JVP_rejection_preserved"]
    assert not summary["slow_flux_atlas_authorized"]
