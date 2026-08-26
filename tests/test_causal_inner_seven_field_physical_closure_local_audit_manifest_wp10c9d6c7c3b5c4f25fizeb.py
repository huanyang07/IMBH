from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

import run_causal_inner_seven_field_physical_closure_local_audit_manifest_wp10c9d6c7c3b5c4f25fizeb as target


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage1_definitions_only_authorization_is_binding() -> None:
    parent = target._validate_stage1(require_clean=False)
    assert parent["summary"]["classification"] == target.STAGE1_CLASSIFICATION
    assert parent["summary"]["authorized_next"].startswith("definitions_only_")
    assert not parent["summary"][
        "physical_Kerr_Schild_seven_field_closure_certified"
    ]
    assert not parent["summary"]["seven_field_trajectory_authorized"]


def test_all_state_inputs_are_canonical_and_hash_locked() -> None:
    inputs = target._validate_state_inputs()
    assert set(inputs) == set(target.STATE_INPUTS)
    for label, entry in inputs.items():
        specification = target.STATE_INPUTS[label]
        assert entry["hashes"][specification["array_name"]] == specification[
            "array_sha256"
        ]
        assert entry["summary"]["classification"] == specification[
            "classification"
        ]
        assert entry["summary"]["passed"]


def test_envelope_reconstructs_committed_primary_and_heldout_states() -> None:
    inputs = target._validate_state_inputs()
    metadata, arrays = target._audit_envelope(inputs)
    for label, key in (
        ("primary_20ms", "primary_20ms_base_charts5"),
        ("heldout_16ms", "heldout_16ms_base_charts5"),
    ):
        data = target._load_npz(inputs, label)
        expected = data["bdf1_primitive_charts"] - data[
            "bdf1_primitive_increment"
        ]
        assert np.array_equal(arrays[key], expected)
    assert metadata["canonical_sources_only"]
    assert not metadata["mutable_scratch_files_used"]


def test_envelope_contains_every_declared_profile_and_failed_face() -> None:
    metadata, arrays = target._audit_envelope()
    assert arrays["accepted_trajectory_base_charts5"].shape == (72, 112, 5)
    assert arrays["rejected_full_step_base_charts5"].shape == (112, 5)
    assert arrays["failed_face_chart5"].shape == (5,)
    assert metadata["empirical_base_chart_count"] == 8401
    assert metadata["failed_face_index"] == 3
    assert metadata["old_failed_face_maximum_imaginary_speed_over_c"] > 1.0e-5
    assert np.array_equal(
        arrays["accepted_terminal_base_charts5"],
        arrays["accepted_trajectory_base_charts5"][-1],
    )


def test_frozen_stencil_is_physical_discrete_and_prospective() -> None:
    metadata, arrays = target._audit_envelope()
    witnesses = arrays["witness_charts5"]
    assert witnesses.shape[0] == metadata["witness_chart_count"]
    assert len(set(arrays["witness_labels"].tolist())) == witnesses.shape[0]
    assert np.max(witnesses[:, 1] ** 2 + witnesses[:, 2] ** 2) < 1.0
    assert np.array_equal(
        arrays["height_departure_stencil"], np.asarray([-0.10, 0.0, 0.10])
    )
    assert np.array_equal(
        arrays["vertical_velocity_over_c_stencil"],
        np.asarray([-0.03, 0.0, 0.03]),
    )
    assert metadata["no_hyperrectangle_claim"]
    assert metadata["stencil_is_discrete_and_prospective"]


def test_covariant_densitization_promotes_real_balance_law_fields() -> None:
    contract = target._contract()
    storage = contract["covariant_densitization"]
    assert storage["Valencia_rest_mass_storage"] == "D=Sigma*W"
    assert storage["conserved_height_content"] == "Z_H=D*H"
    assert storage["conserved_vertical_momentum"] == "P_H=D*w_H"
    assert storage["conserved_shear_coordinate"] == "R_pi=D*chi"
    assert storage["no_redundant_algebraic_height_unknown"]


def test_one_total_energy_generates_state_flux_and_geometry_sources() -> None:
    contract = target._contract()
    energy = contract["rest_frame_total_energy"]
    kerr_schild = contract["Kerr_Schild_state_and_flux"]
    assert "E_H+E_pi" in energy["surface_energy"]
    assert energy["reservoirs_contribute_to_relativistic_inertia"]
    assert "Killing projections" in kerr_schild["first_four_state_entries"]
    assert kerr_schild["geometry_derivatives"].startswith("lower-order")
    assert kerr_schild[
        "existing_Kerr_Schild_projection_code_is_the_pre_boundary_control"
    ]


def test_entropy_variables_must_generate_both_state_and_flux() -> None:
    entropy = target._contract()["physical_entropy_extension"]
    assert entropy["mathematical_entropy"] == "eta(U7)=-S_ext"
    assert entropy["state_identity"] == "U7=dpsi/dw"
    assert entropy["flux_identity"] == "F7=dphi/dw"
    assert entropy["temporal_symmetrizer"].endswith("positive definite")
    assert entropy["spatial_symmetrizer"].endswith("symmetric")
    assert entropy["no_post_hoc_matrix_symmetrization"]


def test_shear_coefficients_are_physical_and_not_boundary_fitted() -> None:
    shear = target._contract()["shear_relaxation_closure"]
    assert shear["alpha_target"] == "chi_alpha=alpha*Pi/(Sigma*c**2)"
    assert shear["signal_calibration"] == "c_nu/c=sqrt(alpha)*c_s/c"
    assert shear[
        "a_pi_is_fixed_by_entropy_conjugacy_and_the_same_nu_s_tau_pi_pair"
    ]
    assert shear["full_seven_field_spectrum_is_binding_not_isolated_c_nu"]
    assert shear["boundary_tuned_coefficient_or_floor_forbidden"]


def test_vertical_fields_obey_covariant_currents_and_heat_closure() -> None:
    contract = target._contract()
    vertical = contract["vertical_balance_laws"]
    ledgers = contract["source_and_energy_ledgers"]
    assert vertical["height_current"] == "nabla_a(Sigma*H*u^a)=Sigma*w_H"
    assert "Pi/H" in vertical["vertical_momentum_current"]
    assert vertical["structural_damping_calibration"] == (
        "gamma_H=alpha*Omega_perp"
    )
    assert vertical["vertical_damping_heats_internal_energy"]
    assert ledgers["internal_relaxation_total_energy_defect"] == (
        "identically zero"
    )
    assert ledgers["mathematical_entropy_source"] == "w dot S<=0"


def test_old_model_parity_stops_before_the_failed_face() -> None:
    parity = target._contract()["old_model_parity_boundary"]
    assert parity["pre_boundary_state_flux_source_parity_required"]
    assert parity["pre_boundary_five_field_compressed_principal_parity_required"]
    assert parity["failed_face_exact_principal_parity_forbidden"]
    assert parity["failed_face_old_complex_spectrum_is_a_negative_control"]
    assert parity[
        "failed_face_new_finite_inertia_spectrum_must_be_real_and_causal"
    ]


def test_derivative_and_local_structural_gates_fail_closed() -> None:
    contract = target._contract()
    derivative = contract["independent_derivative_audit"]
    gates = contract["binding_local_audit_gates"]
    assert derivative["independent_code_path_required"]
    assert derivative["differentiate_then_symmetrize_forbidden"]
    assert gates["A0_diagonally_equilibrated_minimum_eigenvalue_min"] > 0.0
    assert gates["maximum_absolute_characteristic_speed_over_c"] < 1.0
    assert gates["failed_face_old_model_imaginary_speed_over_c_min"] == 1.0e-5
    assert gates["all_points_and_all_gates_required"]
    assert gates["fail_closed"]


def test_stage3_order_derives_potentials_before_matrices() -> None:
    order = target._contract()["stage3_execution_order"]
    assert order.index(
        "derive eta, psi, entropy flux, and phi before assembling matrices"
    ) < order.index("evaluate primary and held-out equilibrium profiles")
    assert order[-1] == "freeze a positive or negative local structural certificate"


def test_stage2_authorizes_only_a_local_nontrajectory_audit() -> None:
    contract = target._contract()
    claims = contract["claim_boundary"]
    budget = contract["budget"]
    assert claims["physical_audit_envelope_frozen"]
    assert not claims["physical_Kerr_Schild_seven_field_closure_certified"]
    assert claims["local_structural_audit_authorized"]
    assert not claims["seven_field_spatial_discretization_authorized"]
    assert not claims["seven_field_trajectory_authorized"]
    assert not claims["complete_cycle_authorized"]
    assert not claims["reduced_slow_evolution_authorized"]
    assert budget["new_five_field_trajectory_steps"] == 0
    assert budget["new_seven_field_trajectory_steps"] == 0
    assert budget["new_nonlinear_roots"] == 0


@pytest.mark.skipif(
    not target.CANONICAL_DIRECTORY.exists(),
    reason="canonical Stage-2 manifest has not yet been frozen",
)
def test_frozen_stage2_package_closes_and_preserves_claim_boundary() -> None:
    summary = _read(target.CANONICAL_DIRECTORY / "summary.json")
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["canonical_audit_envelope_frozen"]
    assert not summary["physical_Kerr_Schild_seven_field_closure_certified"]
    assert summary["local_structural_audit_authorized"]
    assert not summary["seven_field_trajectory_authorized"]
    assert summary["authorized_next"] == target.AUTHORIZED_NEXT
    for line in (target.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        assert _sha256(target.CANONICAL_DIRECTORY / name) == expected
