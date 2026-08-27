import hashlib
import json

import numpy as np
import pytest

import run_causal_inner_equilibrium_column_thermodynamic_potential_implementation_wp10c9d6c7c3b5c4f25fizzc as target


def test_parent_authorizes_fixed_height_equilibrium_only():
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["authorized_next"] == target.WORK_PACKAGE
    assert validated["contract"]["next_package"]["fixed_height_only"]
    assert not validated["contract"]["next_package"]["add_height_or_shear_terms"]


def test_physical_relative_defect_is_scale_aware():
    expected = np.asarray((1.0e20, -2.0e20))
    actual = expected + np.asarray((1.0e10, -2.0e10))
    assert target._relative(actual, expected) == pytest.approx(1.0e-10)


def test_claim_boundary_stays_before_height_and_shear():
    assert target.AUTHORIZED_NEXT.startswith("definitions_only_")
    assert "dynamic_height" in target.AUTHORIZED_NEXT


@pytest.mark.skipif(
    not target.CANONICAL_DIRECTORY.exists(),
    reason="canonical equilibrium potential package has not yet run",
)
def test_canonical_package_closes_and_authorizes_no_trajectory():
    summary = json.loads(
        (target.CANONICAL_DIRECTORY / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["passed"]
    assert summary["equilibrium_physical_potential_certified"]
    assert not summary["dynamic_height_potential_certified"]
    assert not summary["full_shear_master_potential_certified"]
    assert not summary["eleven_field_trajectory_authorized"]
    assert not summary["complete_cycle_execution_authorized"]
    for line in (target.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256(
            (target.CANONICAL_DIRECTORY / name).read_bytes()
        ).hexdigest() == expected
