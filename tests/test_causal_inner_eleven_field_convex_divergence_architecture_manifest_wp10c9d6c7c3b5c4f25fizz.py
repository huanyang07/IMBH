import hashlib
import json

import pytest

import run_causal_inner_eleven_field_convex_divergence_architecture_manifest_wp10c9d6c7c3b5c4f25fizz as target


def test_parent_selects_full_five_component_completion():
    parent = target._validate_parent(require_clean=False)
    assert parent["summary"]["full_five_component_shear_completion_selected"]
    assert parent["metrics"]["derivative_stable_complex_pair"]
    assert parent["metrics"]["full_tensor_screen_passed"]


def test_state_count_and_full_stf_representation_are_binding():
    contract = target._contract()
    state = contract["state_architecture"]
    shear = contract["covariant_shear_representation"]
    assert state["dimension"] == 11
    assert "five independent" in state["dissipative_completion"]
    assert state["one_Rphi_projection_forbidden"]
    assert shear["moving_basis_derivatives_included"]
    assert not shear["basis_frozen_inside_principal_part"]


def test_one_master_scalar_generates_every_principal_current():
    potential = target._contract()["single_master_potential"]
    for key in (
        "mass_current",
        "stress_energy",
        "height_current",
        "shear_current",
    ):
        assert "dchi" in potential[key]
    assert potential["temporal_hessian"].endswith("positive definite")
    assert potential["radial_hessian"].endswith("symmetric")


def test_local_and_reduction_gates_precede_cycle():
    contract = target._contract()
    sequence = contract["prospective_sequence"]
    claims = contract["claim_boundary"]
    assert "complete local envelope structural certificate" in sequence
    assert sequence[-1] == "definitions-only complete-cycle execution manifest"
    assert not claims["nonlinear_physical_master_potential_derived"]
    assert not claims["eleven_field_local_closure_certified"]
    assert not claims["complete_cycle_execution_authorized"]


def test_next_package_is_mathematical_not_physical_execution():
    next_package = target._contract()["next_package"]
    assert next_package["implement_covariant_five_STF_basis"]
    assert next_package["implement_quadratic_eleven_field_common_potential_normal_form"]
    assert not next_package["physical_coefficient_fit_or_trajectory"]


@pytest.mark.skipif(
    not target.CANONICAL_DIRECTORY.exists(),
    reason="canonical architecture manifest has not yet been frozen",
)
def test_canonical_package_closes_and_preserves_claim_boundary():
    summary = json.loads(
        (target.CANONICAL_DIRECTORY / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["eleven_field_architecture_selected"]
    assert not summary["eleven_field_physical_closure_certified"]
    assert not summary["complete_cycle_execution_authorized"]
    for line in (target.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256(
            (target.CANONICAL_DIRECTORY / name).read_bytes()
        ).hexdigest()
        assert actual == expected
