import hashlib
import json

import pytest

import run_causal_inner_eleven_field_nonlinear_master_potential_derivation_manifest_wp10c9d6c7c3b5c4f25fizzb as target


def test_parent_certifies_only_the_structural_kernel():
    parent = target._validate_parent(require_clean=False)
    assert parent["summary"]["five_STF_basis_certified"]
    assert parent["summary"]["quadratic_convex_normal_form_certified"]
    assert not parent["summary"]["nonlinear_physical_master_potential_derived"]


def test_prior_temperature_shear_curl_changes_derivation_order():
    failure = target._contract()["failure_avoidance"]
    assert failure["minimum_failed_relative_curl"] > 1.0
    assert failure["post_hoc_symmetrization_forbidden"]
    assert "master scalar first" in failure["repair_rule"]


def test_equilibrium_control_precedes_height_and_shear():
    contract = target._contract()
    assert contract["derivation_sequence"][0].startswith("exact fixed-height")
    equilibrium = contract["equilibrium_thermodynamic_control"]
    assert "surface-mass current" in equilibrium["required_derivatives"]
    assert equilibrium["gas_radiation_first_law_and_Gibbs_Duhem_binding"]


def test_full_shear_ansatz_retains_nonlinear_invariants():
    shear = target._contract()["full_shear_master_scalar"]
    assert "I2" in shear["prospective_ansatz"]
    assert "I3" in shear["prospective_ansatz"]
    assert shear["coefficient_calibration_after_differentiation"]
    assert shear["one_Rphi_projection_forbidden"]


def test_next_package_is_fixed_height_equilibrium_only():
    next_package = target._contract()["next_package"]
    assert next_package["fixed_height_only"]
    assert next_package["exact_physical_gas_radiation_EOS"]
    assert not next_package["add_height_or_shear_terms"]
    assert not next_package["trajectory"]


@pytest.mark.skipif(
    not target.CANONICAL_DIRECTORY.exists(),
    reason="canonical derivation manifest has not been frozen",
)
def test_canonical_package_closes_without_authorizing_execution():
    summary = json.loads(
        (target.CANONICAL_DIRECTORY / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["passed"] and summary["definitions_only"]
    assert summary["prior_entropy_curl_failure_preserved"]
    assert not summary["nonlinear_physical_master_potential_derived"]
    assert not summary["eleven_field_trajectory_authorized"]
    assert not summary["complete_cycle_execution_authorized"]
    for line in (target.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256(
            (target.CANONICAL_DIRECTORY / name).read_bytes()
        ).hexdigest() == expected
