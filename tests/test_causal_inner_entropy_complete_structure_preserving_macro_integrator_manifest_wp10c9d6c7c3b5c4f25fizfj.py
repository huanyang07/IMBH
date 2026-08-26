from __future__ import annotations

import hashlib

import run_causal_inner_entropy_complete_structure_preserving_macro_integrator_manifest_wp10c9d6c7c3b5c4f25fizfj as target


def test_certified_atlas_authorizes_only_bounded_integrator_manifest() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["thermodynamic_chart_conservative_macro_atlas_certified"]
    assert not validated["summary"]["complete_cycle_execution_authorized"]


def test_contract_is_exact_affine_and_trust_bounded() -> None:
    contract = target._contract()
    assert contract["mathematical_system"]["augmented_generator"] == "K=[[B,c],[0,0]]"
    assert contract["bounded_pilot"]["accepted_macrosteps"] == 4
    assert contract["bounded_pilot"]["horizon_seconds"] == 4.0e-3
    assert contract["bounded_pilot"]["pilot_reserved_trust_coordinate_infinity"] < contract["bounded_pilot"]["atlas_absolute_trust_coordinate_infinity"]
    assert contract["bounded_pilot"]["maximum_new_truth_operator_calls"] == 1
    assert not contract["claim_boundary"]["complete_cycle_execution_authorized"]


def test_canonical_package_closes_if_present() -> None:
    directory = target.CANONICAL_DIRECTORY
    if not directory.exists():
        return
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
    summary = target._utils()._read_json(directory / "summary.json")
    contract = target._utils()._read_json(directory / "macro_integrator_contract.json")
    assert summary["definitions_only"]
    assert summary["bounded_macro_propagation_authorized"]
    assert not summary["complete_cycle_execution_authorized"]
    assert contract["online_cost"]["benchmark_macrosteps"] == 100000
