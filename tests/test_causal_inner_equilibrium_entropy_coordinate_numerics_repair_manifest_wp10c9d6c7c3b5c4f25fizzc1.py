import hashlib
import json

import pytest

import run_causal_inner_equilibrium_entropy_coordinate_numerics_repair_manifest_wp10c9d6c7c3b5c4f25fizzc1 as target


def test_rejected_parent_is_preserved_and_diagnosed_as_method_failure():
    validated = target._validate_parent(require_clean=False)
    assert not validated["summary"]["passed"]
    assert validated["metrics"]["maximum_complex_step_current_jacobian_relative_defect"] <= 1.0e-9
    contract = target._contract()
    assert not contract["diagnosis"]["physical_thermodynamic_failure_selected"]
    assert contract["preserved_rejection"]["retroactive_pass_forbidden"]


def test_repair_keeps_physics_and_gates_fixed():
    contract = target._contract()
    assert contract["prospective_repair"]["physical_EOS_unchanged"]
    assert contract["prospective_repair"]["entropy_field_count_unchanged"] == 5
    assert contract["binding_rerun"]["physical_current_relative_defect_maximum"] == 1.0e-10
    assert contract["binding_rerun"]["same_47_frozen_physical_witnesses"]
    assert not contract["claim_boundary"]["complete_cycle_execution_authorized"]


@pytest.mark.skipif(not target.CANONICAL_DIRECTORY.exists(), reason="repair manifest has not run")
def test_canonical_manifest_closes():
    summary = json.loads((target.CANONICAL_DIRECTORY / "summary.json").read_text(encoding="utf-8"))
    assert summary["passed"] and summary["definitions_only"]
    assert summary["equilibrium_rejection_preserved"]
    assert not summary["equilibrium_physical_potential_certified"]
    for line in (target.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((target.CANONICAL_DIRECTORY / name).read_bytes()).hexdigest() == expected
