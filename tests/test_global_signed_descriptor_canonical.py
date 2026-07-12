from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "results/canonical/global_signed_descriptor_prototype"


def _load(name: str):
    return json.loads((CANONICAL / name).read_text())


def test_global_signed_descriptor_canonical_gates() -> None:
    report = _load("rank_and_ledger.json")
    assert report["all_rank_gates_pass"]
    assert report["all_ledger_gates_pass"]
    assert report["all_primitive_recovery_gates_pass"]
    assert report["all_radial_balance_gates_pass"]
    for mesh in report["meshes"]:
        assert mesh["unknowns"] == 8 * mesh["n_cells"] + 4
        assert mesh["descriptor_rank"] == 4 * mesh["n_cells"]
        assert mesh["backward_euler_rank"] == mesh["unknowns"]
        assert mesh["mass_flux_has_both_signs"]
        assert mesh["mass_flux_has_zero_crossing"]
        assert mesh["maximum_relative_ledger_defect"] < 2.0e-15
        assert mesh["maximum_temperature_round_trip_error"] < 5.0e-12
        assert mesh["constant_pi_keplerian_balance_error"] < 2.0e-9
    assert min(report["radial_balance_error_ratios"]) > 7.0
    rusanov = report["rusanov_audit"]
    assert rusanov["accepted_step"]
    assert rusanov["ten_times_larger_step_rejected"]
    assert rusanov["rejected_state_is_unchanged"]
    assert rusanov["maximum_equilibrium_corrected_flux_difference"] < 1.0e-14
    assert rusanov["accepted_maximum_relative_ledger_defect"] < 1.0e-8
    temporal = report["source_free_temporal_audit"]
    assert temporal["all_steps_accepted"]
    assert temporal["error_ratio"] > 1.9
    assert temporal["maximum_storage_scaled_ledger_defect"] < 1.0e-14
    mesh = report["source_free_mesh_audit"]
    assert mesh["all_interior_mesh_gates_pass"]
    assert mesh["open_edge_response_is_bounded"]
    assert min(mesh["interior_drift_ratios"]) > 3.0
    assert all(
        item["maximum_drift_cell"] == item["n_cells"] - 1
        for item in mesh["meshes"]
    )
    stress = report["common_stress_flux_audit"]
    assert stress["outward_orientation"]
    assert stress["minimum_viscous_torque"] > 0.0
    assert stress["maximum_mass_flux_change"] == 0.0
    assert stress["maximum_radial_momentum_flux_change"] == 0.0
    assert stress["maximum_scaled_angular_pair_mismatch"] < 1.0e-14
    assert stress["maximum_scaled_energy_pair_mismatch"] < 1.0e-14
    assert stress["zero_torque_boundary_is_exact"]
    implicit = report["implicit_stress_audit"]
    assert implicit["all_steps_accepted"]
    assert implicit["error_ratio"] > 1.9
    assert implicit["maximum_scaled_residual"] < 1.0e-12
    assert implicit["maximum_storage_scaled_ledger_defect"] < 1.0e-14
    assert implicit["forced_rejection_returns_original_state"]
    monolithic = report["monolithic_backward_euler_audit"]
    assert len(monolithic["runs"]) == 3
    for run in monolithic["runs"]:
        assert run["steps_accepted"] == run["steps_attempted"]
        assert run["maximum_scaled_residual"] < 6.0e-9
        assert run["maximum_storage_scaled_ledger_defect"] < 4.0e-9
        assert run["minimum_surface_density"] > 0.0
        assert run["minimum_omega"] > 0.0
        assert run["minimum_temperature"] > 0.0
    temporal = monolithic["temporal_differences_4_vs_8"]
    assert temporal["maximum_surface_density_fraction"] < 3.0e-7
    assert temporal["maximum_temperature_fraction"] < 2.0e-4
    assert temporal["maximum_omega_fraction"] < 3.0e-8
    assert temporal["maximum_radial_velocity_difference_c"] < 4.0e-8
    colored = monolithic["colored_jacobian_audit"]
    assert not colored["accepted"]
    assert colored["pattern_shape"] == [32, 32]
    assert colored["pattern_nonzeros"] == 352
    assert colored["maximum_temperature_fraction"] > 1.0e-3
    assert monolithic["split_imex_steps_before_rejection"] < 8
    assert monolithic["split_imex_rejected_state_unchanged"]
    cooling = report["radiative_cooling_audit"]
    assert cooling["all_local_steps_accepted"]
    assert cooling["local_error_ratio"] > 1.8
    assert cooling["maximum_local_storage_scaled_ledger_defect"] < 1.0e-14
    assert cooling["monolithic_adiabatic_accepted"]
    assert cooling["monolithic_cooled_accepted"]
    assert cooling["monolithic_cooled_maximum_residual"] < 1.0e-8
    assert cooling["monolithic_cooled_storage_scaled_ledger_defect"] < 1.0e-12
    assert cooling["minimum_monolithic_temperature_drop"] > 0.0
    stream = _load("stream_preflight.json")
    assert stream["all_exact_moment_gates_pass"]
    assert stream["constant_injected_state"]
    assert [
        mesh["active_source_cells"] for mesh in stream["exact_moment_meshes"]
    ] == [6, 10, 19]
    step = stream["monolithic_step"]
    assert step["accepted"]
    assert step["maximum_scaled_residual"] < 1.0e-8
    assert step["maximum_storage_scaled_ledger_defect"] < 1.0e-8
    assert step["minimum_surface_density"] > 0.0
    assert step["minimum_temperature"] > 0.0
    physical = _load("physical_open_preflight.json")
    assert physical["input_closure"]["stream_rate_over_eddington"] == 5.0
    assert physical["input_closure"]["specific_radial_velocity"] == 0.0
    assert physical["input_closure"]["outer_radius_rg"] == 335.0
    for run in physical["runs"]:
        step = run["step"]
        assert step["all_steps_accepted"]
        assert step["maximum_scaled_residual"] < 1.0e-8
        assert step["maximum_storage_scaled_ledger_defect"] < 1.0e-12
        assert step["minimum_surface_density"] > 0.0
        assert step["minimum_temperature"] > 0.0
    temporal = physical["temporal_comparison_N16"]
    assert abs(temporal["inner_flux_difference"]) < 1.0e-6
    assert abs(temporal["outer_flux_difference"]) < 1.0e-8
    mesh_16_24 = physical["mesh_comparison_N16_N24_two_steps"]
    assert abs(mesh_16_24["outer_flux_difference"]) > 1.0
    mesh_24_32 = physical["mesh_comparison_N24_N32_two_steps"]
    assert abs(mesh_24_32["outer_flux_difference"]) > 0.05
    for mapping in physical["pointwise_mapping_only_meshes"]:
        assert mapping["accepted"]
        assert (
            mapping["maximum_face_mass_flux_change_from_stream_source"]
            == 0.0
        )
    conservative = physical["conservative_mapping_only_meshes"]
    assert [item["accepted"] for item in conservative] == [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    assert all(
        item["minimum_recovered_specific_internal_energy"] > 0.0
        for item in conservative
    )
    corrections = [
        item["maximum_absolute_specific_mechanical_energy_correction"]
        for item in conservative[-3:]
    ]
    assert corrections[2] < corrections[1] < corrections[0]
    quadrature = physical["mechanical_reference_quadrature_audit"]
    assert [item["n_cells"] for item in quadrature] == [64, 96, 128]
    for item in quadrature:
        assert item["maximum_mass_relative_difference"] < 5.0e-4
        assert item["maximum_radial_momentum_relative_difference"] < 1.2e-3
        assert item["maximum_angular_momentum_relative_difference"] < 5.0e-4
        assert item["maximum_total_energy_relative_difference"] < 5.0e-4
        assert item["maximum_correction_relative_difference"] < 7.0e-4
        assert item["maximum_temperature_relative_difference"] < 2.0e-4
    target_inner = physical["input_closure"][
        "target_inner_outward_mass_flux_over_supply"
    ]
    target_outer = physical["input_closure"][
        "target_outer_outward_mass_flux_over_supply"
    ]
    assert (
        abs(conservative[-2]["inner_mass_flux_over_supply"] - target_inner)
        < 5.0e-4
    )
    assert (
        abs(conservative[-2]["outer_mass_flux_over_supply"] - target_outer)
        < 2.0e-2
    )
    conservative_step = physical["conservative_N64_step"]["step"]
    assert conservative_step["all_steps_accepted"]
    assert conservative_step["maximum_scaled_residual"] < 1.0e-8
    assert (
        conservative_step["maximum_storage_scaled_ledger_defect"] < 1.0e-12
    )
    conservative_half = physical["conservative_N64_two_half_steps"]["step"]
    assert conservative_half["all_steps_accepted"]
    assert conservative_half["maximum_scaled_residual"] < 1.0e-8
    assert (
        conservative_half["maximum_storage_scaled_ledger_defect"] < 1.0e-12
    )
    conservative_temporal = physical["conservative_N64_temporal_comparison"]
    assert abs(conservative_temporal["inner_flux_difference"]) < 1.0e-6
    assert abs(conservative_temporal["outer_flux_difference"]) < 1.0e-8
    assert (
        abs(conservative_temporal["maximum_H_over_R_relative_difference"])
        < 1.0e-9
    )
    sparse_runs = physical["sparse_evolved_mesh_runs"]
    assert [run["initial"]["n_cells"] for run in sparse_runs] == [64, 96]
    for run in sparse_runs:
        assert run["step"]["all_steps_accepted"]
        audit = run["step"]["jacobian_audits"][0]
        assert audit["accepted"]
        assert audit["maximum_relative_defect"] == 0.0
        assert run["step"]["maximum_scaled_residual"] < 5.0e-12
    evolved_mesh = physical["sparse_evolved_N64_N96_comparison"]
    assert not evolved_mesh["flux_mesh_gate_pass"]
    assert abs(evolved_mesh["outer_flux_difference_over_supply"]) > 0.02
    assert abs(evolved_mesh["maximum_H_over_R_relative_difference"]) < 1.0e-3
    donor_mapping = physical["conserved_donor_mapping_only_meshes"]
    assert [item["n_cells"] for item in donor_mapping] == [64, 96, 128]
    for item in donor_mapping:
        assert item["accepted"]
        assert item["open_face_reconstruction"] == "conserved_donor"
        assert item["outer_radial_flux_donor_consistency"] == 0.0
        assert item["outer_angular_flux_donor_consistency"] == 0.0
        assert item["outer_energy_flux_donor_consistency"] == 0.0
        characteristic = item["outer_characteristic_audit"]
        assert 0.0 < characteristic["radial_mach_number"] < 0.02
        assert characteristic["incoming_characteristics"] == 1
        assert characteristic["eigenvalues"][0] < 0.0
        assert all(value > 0.0 for value in characteristic["eigenvalues"][1:])
        geometry = item["outer_boundary_geometry"]
        assert abs(geometry["outer_radius_over_hill_radius"] - 0.44852) < 1.0e-5
        assert not geometry["is_roche_saddle"]
        assert not geometry["exterior_thermodynamic_state_declared"]
        assert not geometry["characteristic_contract_closed"]
        inner = item["inner_characteristic_audit"]
        assert -1.0 < inner["radial_mach_number"] < 0.0
        assert inner["incoming_characteristics"] == 1
        assert all(value < 0.0 for value in inner["eigenvalues"][:3])
        assert inner["eigenvalues"][3] > 0.0
    assert (
        abs(
            donor_mapping[2]["outer_mass_flux_over_supply"]
            - donor_mapping[1]["outer_mass_flux_over_supply"]
        )
        < 0.01
    )
    donor_runs = physical["conserved_donor_evolved_mesh_runs"]
    for run in donor_runs:
        assert run["step"]["all_steps_accepted"]
        assert run["step"]["maximum_scaled_residual"] < 5.0e-12
        assert run["step"]["maximum_storage_scaled_ledger_defect"] < 1.0e-14
    donor_mesh = physical["conserved_donor_N64_N96_comparison"]
    assert donor_mesh["flux_mesh_gate_pass"]
    assert abs(donor_mesh["outer_flux_difference_over_supply"]) < 0.01
    assert abs(donor_mesh["inner_flux_difference_over_supply"]) < 0.005
    assert abs(donor_mesh["maximum_H_over_R_relative_difference"]) < 0.01
    column_runs = physical["column_energy_evolved_mesh_runs"]
    assert [run["initial"]["n_cells"] for run in column_runs] == [64, 96]
    for run in column_runs:
        assert run["initial"]["include_vertical_column_work"]
        assert 0.039 < abs(
            run["initial"][
                "integrated_vertical_work_over_eddington_luminosity"
            ]
        ) < 0.041
        assert run["step"]["all_steps_accepted"]
        assert run["step"]["maximum_scaled_residual"] < 1.0e-11
        assert run["step"]["maximum_storage_scaled_ledger_defect"] < 1.0e-14
    column_mesh = physical["column_energy_N64_N96_comparison"]
    assert column_mesh["flux_mesh_gate_pass"]
    assert abs(column_mesh["outer_flux_difference_over_supply"]) < 0.01
    characteristic_runs = physical["characteristic_inner_evolved_mesh_runs"]
    assert [run["initial"]["n_cells"] for run in characteristic_runs] == [64, 96]
    for run in characteristic_runs:
        assert run["initial"]["boundary_mode"] == (
            "characteristic_inner_open_outer"
        )
        assert run["step"]["all_steps_accepted"]
        assert run["step"]["maximum_scaled_residual"] < 1.0e-11
        assert run["step"]["maximum_storage_scaled_ledger_defect"] < 1.0e-14
        projection = run["step"]["inner_characteristic_projections"][0]
        incoming_scale = max(abs(projection["incoming_amplitude_before"]), 1.0)
        outgoing_scale = max(abs(projection["outgoing_amplitude_before"]), 1.0)
        assert abs(projection["incoming_amplitude_after"]) < 1.0e-7 * incoming_scale
        assert abs(
            projection["outgoing_amplitude_after"]
            - projection["outgoing_amplitude_before"]
        ) < 1.0e-7 * outgoing_scale
        assert run["step"]["inner_mass_flux_over_supply"] < 0.0
    for selected, baseline in zip(characteristic_runs, column_runs):
        assert abs(
            selected["step"]["inner_mass_flux_over_supply"]
            - baseline["step"]["inner_mass_flux_over_supply"]
        ) < 5.0e-5


def test_global_signed_descriptor_checksums_and_scope() -> None:
    expected = {}
    for line in (CANONICAL / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        expected[name] = digest
    assert expected
    for name, digest in expected.items():
        actual = hashlib.sha256((CANONICAL / name).read_bytes()).hexdigest()
        assert actual == digest
    provenance = _load("provenance.json")
    assert provenance["numerical_status"] == "DIAGNOSTIC ONLY"
    assert provenance["physical_status"] == "DIAGNOSTIC ONLY"
    assert "not implemented" in provenance["claim_scope"]
