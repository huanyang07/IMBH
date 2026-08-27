import hashlib
import json

import pytest

import run_causal_inner_equilibrium_compensated_coordinate_implementation_wp10c9d6c7c3b5c4f25fizzc2 as target


def test_repair_manifest_authorizes_binding_rerun():
    validated = target._validate_manifest(require_clean=False)
    assert validated["summary"]["authorized_next"] == target.WORK_PACKAGE
    assert validated["contract"]["binding_rerun"]["same_47_frozen_physical_witnesses"]


def test_original_rejection_remains_a_parent_not_a_pass():
    assert target.original.FAIL_CLASSIFICATION != target.PASS_CLASSIFICATION
    summary = target._utils()._read_json(target.original.CANONICAL_DIRECTORY / "summary.json")
    assert not summary["passed"]


@pytest.mark.skipif(not target.CANONICAL_DIRECTORY.exists(), reason="compensated certificate has not run")
def test_canonical_package_closes_before_dynamic_height():
    summary = json.loads((target.CANONICAL_DIRECTORY / "summary.json").read_text(encoding="utf-8"))
    assert summary["passed"]
    assert summary["original_equilibrium_rejection_preserved"]
    assert summary["equilibrium_physical_potential_certified"]
    assert not summary["dynamic_height_potential_certified"]
    assert not summary["eleven_field_trajectory_authorized"]
    assert not summary["complete_cycle_execution_authorized"]
    for line in (target.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((target.CANONICAL_DIRECTORY / name).read_bytes()).hexdigest() == expected
