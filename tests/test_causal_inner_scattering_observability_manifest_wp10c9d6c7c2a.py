from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (
    causal_array_sha256,
    causal_canonical_json_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = (
    ROOT
    / "results/canonical/"
    "causal_inner_scattering_observability_manifest_wp10c9d6c7c2a"
)
SUMMARY = CANONICAL / "summary.json"
MANIFEST = CANONICAL / "scattering_manifest.json"
DECISIVE = CANONICAL / "decisive_arrays.npz"
PROVENANCE = CANONICAL / "provenance.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_wp10c9d6c7c2a_preserves_every_historical_classification() -> None:
    summary = _summary()
    preserved = summary["historical_classifications_preserved"]
    assert preserved["c7b"] == {
        "classification": "prospective_embedded_profile_validation_failed",
        "passed": False,
    }
    assert preserved["c7c0"]["passed"]
    assert preserved["c7c1a"]["passed"]
    assert preserved["c7c1b"] == {
        "classification": "no_regularized_embedded_profile_class_selected",
        "passed": False,
    }
    assert summary["tier_I_direct_physics_status"] == (
        "passed_for_declared_c7c1b_profiles"
    )
    assert summary["tier_II_scattering_status"] == (
        "definitions_frozen_not_propagated"
    )


def test_wp10c9d6c7c2a_separates_certification_tiers() -> None:
    tiers = _manifest()["certification_tiers"]
    assert tiers["tier_I_primary_physics"][
        "one_shared_conservative_face_flux"
    ]
    assert tiers["tier_I_primary_physics"][
        "exact_prefix_and_global_ledgers"
    ]
    tier_II = tiers["tier_II_interface_scattering"]
    assert tier_II["integrated_flux_is_primary"]
    assert tier_II["pointwise_traction_requires_observability"]
    assert {
        "time_integrated_incident_energy_flux",
        "time_integrated_reflected_energy_flux",
        "time_integrated_transmitted_energy_flux",
        "complete_energy_ledger_residual",
    }.issubset(tier_II["primary_observables"])
    assert not tiers["tier_III_nonlinear"]["authorized"]


def test_wp10c9d6c7c2a_freezes_complete_energy_contract() -> None:
    contract = _manifest()["energy_and_scattering_contract"]
    assert contract["eigenvector_normalization_invariant"]
    assert contract[
        "balance_terms_must_be_derived_from_implemented_symmetrized_DAE"
    ]
    assert contract["each_physical_work_term_recorded_exactly_once"]
    assert contract["no_constant_coefficient_R_plus_T_equals_one_assumption"]
    assert "W_background" in contract["complete_balance"]
    assert "W_height" in contract["complete_balance"]
    assert contract["uniform_virtual_interface_parent_face"] == 48
    coefficients = contract["scattering_coefficients"]
    assert coefficients["R"] == "E_reflected / E_incident"
    assert coefficients["T"] == "E_transmitted / E_incident"
    assert "uniform_continuum_extrapolate" in coefficients[
        "interface_delta_R"
    ]


def test_wp10c9d6c7c2a_uncertainty_is_conservative_and_prospective() -> None:
    contract = _manifest()["uncertainty_and_observability_contract"]
    assert contract["default_combination"] == (
        "conservative_sum_or_direct_nuisance_sweep_envelope"
    )
    assert contract[
        "root_sum_square_forbidden_without_demonstrated_independence"
    ]
    assert contract[
        "covariance_combination_allowed_only_when_measured_and_stable"
    ]
    assert contract["minimum_signal_to_uncertainty_ratio"] == 5.0
    assert contract[
        "maximum_reference_uncertainty_to_medium_fine_difference"
    ] == 0.10
    direction = contract["direction_gate"]
    assert direction[
        "coarse_medium_error_norm_must_exceed_kappa_uncertainty"
    ]
    assert direction[
        "medium_fine_error_norm_must_exceed_kappa_uncertainty"
    ]
    assert direction["below_floor_is_neither_pass_nor_fail"]
    assert not contract["slow_impact_threshold_binding"]
    assert contract[
        "slow_impact_deferred_until_Q_macro_horizon_and_closure_exist"
    ]
    assert contract["c7c1b_not_reclassified"]


def test_wp10c9d6c7c2a_freezes_bidirectional_scattering_roles() -> None:
    profiles = _manifest()["requested_scattering_profiles"]
    assert len(profiles) == 7
    for direction in ("fine_to_coarse", "coarse_to_fine"):
        for family in ("shear", "acoustic", "mixed_shear_acoustic"):
            definition = profiles[f"{direction}__{family}"]
            assert definition["binding"]
            assert definition["endpoint_regularity"] == "C3_or_better"
            assert definition["compact_support"]
            assert definition["amplitude_factors"] == [0.5, 1.0]
            assert definition["signs"] == [-1, 1]
    null = profiles["null_selected_channel"]
    assert null["selected_incident_channel_exactly_zero"]
    assert null["role"] == "diagnostic_false_positive_floor"


def test_wp10c9d6c7c2a_detects_frozen_domain_geometry_blocker() -> None:
    summary = _summary()
    geometry = summary["geometry_feasibility"]
    assert geometry["parent_cell_count"] == 64
    assert geometry["coupling_parent_face"] == 48
    assert geometry["raw_inner_parent_cells"] == 48
    assert geometry["raw_outer_parent_cells"] == 16
    assert geometry["minimum_spectral_support_parent_cells"] >= 40
    assert geometry["raw_fine_to_coarse_capacity_passed"]
    assert not geometry["raw_coarse_to_fine_capacity_passed"]
    assert not geometry["clearance_fine_to_coarse_capacity_passed"]
    assert not geometry["clearance_coarse_to_fine_capacity_passed"]
    assert not geometry["bidirectional_compact_packet_class_feasible"]
    assert summary["classification"] == (
        "scattering_observability_contract_frozen_"
        "bidirectional_packet_preflight_blocked"
    )


def test_wp10c9d6c7c2a_authorizes_no_propagation_or_downstream_work() -> None:
    summary = _summary()
    assert summary["passed"]
    assert not summary["operator_changed"]
    assert not summary["propagation_executed"]
    assert not summary["uniform_scattering_propagation_authorized"]
    assert not summary["embedded_scattering_propagation_authorized"]
    assert not summary["bounded_nonlinear_common_mode_authorized"]
    assert not summary["production_operator_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    assert summary["authorized_next"] == (
        "WP10c9d6c7c2a1_operator_neutral_scattering_geometry_"
        "feasibility_design"
    )


def test_wp10c9d6c7c2a_uses_canonical_provenance_vocabulary() -> None:
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    assert provenance["source_parent_commit"] == (
        "c73102812b73f115c1e4f2771be952adc6ea4c00"
    )
    assert provenance["scientific_status"] == "DIAGNOSTIC ONLY"
    assert provenance["classification"] == (
        "scattering_observability_contract_frozen_"
        "bidirectional_packet_preflight_blocked"
    )


def test_wp10c9d6c7c2a_manifest_and_canonical_hashes() -> None:
    summary = _summary()
    manifest = _manifest()
    payload = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_sha256"
    }
    assert (
        causal_canonical_json_sha256(payload)
        == manifest["manifest_sha256"]
        == summary["manifest_sha256"]
    )
    assert _sha256(DECISIVE) == summary["decisive_arrays_sha256"]
    for relative, expected in summary[
        "implementation_source_hashes"
    ].items():
        assert _sha256(ROOT / relative) == expected
    with np.load(DECISIVE, allow_pickle=False) as source:
        assert set(source.files) == set(summary["decisive_array_hashes"])
        for name in source.files:
            assert (
                causal_array_sha256(source[name])
                == summary["decisive_array_hashes"][name]
            )
