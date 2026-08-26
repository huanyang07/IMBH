from __future__ import annotations

import hashlib
import json

import pytest

import run_causal_inner_seven_field_entropy_architecture_stage1_wp10c9d6c7c3b5c4f25fizea as target


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_parent_boundary_remains_binding_and_authorizes_nothing() -> None:
    parent = target._validate_parent(require_clean=False)
    assert parent["summary"]["classification"] == target.PARENT_CLASSIFICATION
    assert parent["summary"]["authorized_next"] is None
    assert not parent["summary"]["trajectory_continuation_passed"]
    assert parent["metrics"]["stop_reason"] == "hyperbolicity_boundary"


def test_stage1_is_an_explicit_design_only_departure() -> None:
    contract = target._contract()
    authorization = contract["authorization_basis"]
    assert authorization["parent_authorized_next_is_null"]
    assert authorization["scope_is_model_design_only"]
    assert contract["budget"]["new_five_field_trajectory_steps"] == 0
    assert contract["budget"]["new_seven_field_trajectory_steps"] == 0


def test_state_promotes_height_and_vertical_momentum() -> None:
    state = target._contract()["seven_field_state"]
    assert state["dimension"] == 7
    assert state["vertical_equilibrium_dimension"] == 5
    assert state["primitive_names"][-2:] == (
        "log_height",
        "vertical_velocity_over_c",
    )
    assert state["height_and_vertical_momentum_are_independent"]
    assert state["shear_stress_remains_a_causal_relaxation_field"]


def test_energy_derivation_recovers_the_hydrostatic_relation() -> None:
    energy = target._contract()["total_energy_architecture"]
    assert energy["density_relation"] == "rho=Sigma/(2*H)"
    assert energy["vertical_force_at_fixed_entropy"] == (
        "-dE/dH=Pi/H-Sigma*Omega_perp**2*H"
    )
    assert energy["hydrostatic_equilibrium"] == (
        "Pi=Sigma*Omega_perp**2*H**2"
    )
    assert energy["matches_existing_gas_radiation_height_equation"]
    assert not energy["total_energy_loss_from_internal_relaxation"]


def test_entropy_potential_and_normal_form_obligations_are_binding() -> None:
    contract = target._contract()
    potential = contract["entropy_potential_architecture"]
    normal = contract["proved_local_normal_form"]
    assert potential["temporal_hessian"].endswith("positive definite")
    assert potential["spatial_hessian"].endswith("symmetric")
    assert potential["all_principal_derivative_couplings_must_come_from_phi"]
    assert potential["algebraic_height_derivative_foldback_forbidden"]
    assert normal["chart_temporal"] == "A0=L.T*M*L"
    assert normal["chart_spatial"] == "A1=L.T*K*L"
    assert "interlaces" in normal["subcharacteristic_theorem"]


def test_old_nonhyperbolic_limit_is_not_a_required_target() -> None:
    boundary = target._contract()["five_field_limit_boundary"]
    assert boundary["exact_equivalence_required_only_on_certified_pre_boundary_domain"]
    assert boundary["exact_equivalence_at_or_beyond_failed_face_forbidden"]
    assert boundary["finite_vertical_inertia_is_binding_beyond_old_validity_domain"]


def test_reference_normal_form_proves_only_the_structural_identities() -> None:
    reference, arrays = target._reference_payload()
    assert reference["passed"]
    assert not reference["fixture_is_physical_calibration"]
    assert arrays["temporal_matrix"].shape == (7, 7)
    assert arrays["reduced_temporal_matrix"].shape == (5, 5)
    audit = reference["audit"]
    assert audit["temporal_minimum_eigenvalue"] > 0.0
    assert audit["source_entropy_positive_part"] <= 1.0e-12
    assert audit["subcharacteristic_interlacing_violation"] <= 1.0e-12


def test_stage2_is_manifest_only_and_no_trajectory_is_authorized() -> None:
    contract = target._contract()
    obligations = contract["stage2_manifest_obligations"]
    claims = contract["claim_boundary"]
    assert obligations["definitions_only"]
    assert obligations["no_trajectory"]
    assert claims["quadratic_entropy_normal_form_certified"]
    assert not claims["physical_Kerr_Schild_seven_field_closure_certified"]
    assert not claims["nonlinear_global_symmetrizer_certified"]
    assert claims["definitions_only_stage2_manifest_authorized"]
    assert not claims["seven_field_trajectory_authorized"]
    assert not claims["complete_cycle_authorized"]


@pytest.mark.skipif(
    not target.CANONICAL_DIRECTORY.exists(),
    reason="canonical Stage-1 package has not yet been frozen",
)
def test_frozen_stage1_package_closes_and_preserves_claim_boundary() -> None:
    summary = _read(target.CANONICAL_DIRECTORY / "summary.json")
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["quadratic_entropy_normal_form_certified"]
    assert not summary["physical_Kerr_Schild_seven_field_closure_certified"]
    assert not summary["seven_field_trajectory_authorized"]
    assert summary["authorized_next"] == target.AUTHORIZED_NEXT
    for line in (target.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        assert _sha256(target.CANONICAL_DIRECTORY / name) == expected
