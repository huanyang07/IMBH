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
    "causal_inner_scattering_geometry_wp10c9d6c7c2a1"
)
SUMMARY = CANONICAL / "summary.json"
MANIFEST = CANONICAL / "geometry_manifest.json"
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


def test_wp10c9d6c7c2a1_preserves_parent_and_changes_no_operator() -> None:
    summary = _summary()
    assert summary["parent_classification_preserved"] == (
        "scattering_observability_contract_frozen_"
        "bidirectional_packet_preflight_blocked"
    )
    assert summary["passed"]
    assert not summary["operator_changed"]
    assert not summary["propagation_executed"]
    assert summary["required_geometry"] == {
        "both_incidence_directions_required": True,
        "clearance_cells_each_end": 3,
        "packet_support_parent_cells": 43,
        "required_parent_cells_per_side": 49,
    }


def test_wp10c9d6c7c2a1_rejects_uncertified_physical_extension() -> None:
    report = _summary()["option_reports"]["extended_physical_domain"]
    assert report["geometry_passed"]
    assert report["cells_per_side"] == 49
    assert report["required_inner_radius_over_rg"] < 1.8
    assert report["required_outer_radius_over_rg"] > 90.0
    assert np.isclose(
        report["certified_background_inner_radius_over_rg"],
        1.8,
        rtol=2.0e-15,
        atol=0.0,
    )
    assert report["certified_background_outer_radius_over_rg"] < 13.0
    assert not report["certified_background_coverage_passed"]
    assert not report["new_inner_excision_causality_certified"]
    assert not report["independent_stationary_background_extension_available"]
    assert not report["passed"]
    assert not report["selected"]


def test_wp10c9d6c7c2a1_rejects_overlapping_existing_injection() -> None:
    report = _summary()["option_reports"][
        "existing_domain_characteristic_injection"
    ]
    assert report["minimum_resolved_pulse_extent_parent_cell_times"] == 43
    assert report["outer_cells_available"] == 16
    assert report[
        "outer_boundary_to_interface_roundtrip_parent_cell_times"
    ] == 32
    assert report[
        "postinterface_surface_roundtrip_parent_cell_times"
    ] == 6
    assert not report[
        "coarse_to_fine_nonoverlap_at_outer_boundary_passed"
    ]
    assert not report[
        "coarse_to_fine_nonoverlap_at_postinterface_surface_passed"
    ]
    assert not report["fine_to_coarse_injection_at_excision_allowed"]
    assert not report["passed"]
    assert not report["selected"]


def test_wp10c9d6c7c2a1_selects_only_manufactured_patch() -> None:
    summary = _summary()
    assert summary["classification"] == (
        "manufactured_interface_patch_geometry_selected_"
        "energy_preflight_authorized"
    )
    assert summary["selected_route"] == (
        "manufactured_variable_coefficient_interface_patch"
    )
    options = summary["option_reports"]
    selected = [name for name, report in options.items() if report["selected"]]
    assert selected == [
        "manufactured_variable_coefficient_interface_patch"
    ]
    patch = options[selected[0]]
    assert patch["method_level_interface_audit_only"]
    assert not patch["physical_background_claimed"]
    assert patch["parent_equivalent_cell_count"] == 98
    assert patch["parent_equivalent_cells_per_side"] == 49
    assert patch["interface_face"] == 49
    assert patch["minimum_packet_clearance_cells"] == 3
    assert patch["travel_window_nonoverlap_capacity_passed"]
    assert patch["geometry_passed"]
    assert patch["passed"]


def test_wp10c9d6c7c2a1_freezes_exact_core_and_C4_extension() -> None:
    contract = _manifest()["coefficient_extension_contract"]
    core = contract["exact_physical_core"]
    assert core["parent_faces"] == [42, 54]
    assert core["patch_faces"] == [43, 55]
    assert core[
        "interface_state_measure_and_principal_matrix_parity_required"
    ]
    assert core["maximum_interface_parity_defect"] == 1.0e-12
    assert contract["endpoint_jet_order"] == 4
    assert contract["transition_regularity"] == "C4"
    assert contract["transition_parent_cells"] == 12
    assert contract["degree9_smootherstep_coefficients_low_to_high"] == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        126.0,
        -420.0,
        540.0,
        -315.0,
        70.0,
    ]
    assert contract[
        "must_recompute_all_physical_matrices_from_extended_state"
    ]
    assert contract["direct_matrix_interpolation_forbidden"]
    assert contract["admissibility_required_everywhere"]
    assert contract[
        "hyperbolicity_and_separated_subspaces_required_everywhere"
    ]
    assert contract["no_residual_subtraction"]
    assert contract["no_fitted_coefficient"]


def test_wp10c9d6c7c2a1_authorizes_only_energy_method_preflight() -> None:
    summary = _summary()
    assert summary["authorized_next"] == (
        "WP10c9d6c7c2a2_manufactured_scattering_energy_preflight"
    )
    assert not summary["uniform_scattering_propagation_authorized"]
    assert not summary["embedded_scattering_propagation_authorized"]
    assert not summary["bounded_nonlinear_common_mode_authorized"]
    assert not summary["production_operator_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    limits = _manifest()["scientific_limits"]
    assert not limits["physical_radial_background_certified"]
    assert not limits["actual_physical_embedded_scattering_certified"]
    assert limits["c7c1b_strict_rejection_unchanged"]
    assert limits[
        "manufactured_patch_cannot_by_itself_authorize_nonlinear_work"
    ]


def test_wp10c9d6c7c2a1_decisive_geometry_arrays() -> None:
    with np.load(DECISIVE, allow_pickle=False) as source:
        patch_edges = source["manufactured_patch_edges"]
        support = source["manufactured_patch_support_faces"]
        measurement = source["manufactured_patch_measurement_faces"]
        mapping = source["manufactured_patch_core_face_map"]
    assert patch_edges.size == 99
    assert support.tolist() == [[3, 46], [52, 95]]
    assert measurement.tolist() == [6, 49, 92]
    assert mapping.tolist() == [[42, 48, 54], [43, 49, 55]]
    spacing = np.diff(np.log(patch_edges))
    assert float(np.max(spacing) - np.min(spacing)) <= 2.0e-14


def test_wp10c9d6c7c2a1_uses_canonical_provenance_vocabulary() -> None:
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    assert provenance["source_parent_commit"] == (
        "73f902622834d13981d36e22aa21e13fefb9df8b"
    )
    assert provenance["scientific_status"] == "DIAGNOSTIC ONLY"
    assert provenance["classification"] == (
        "manufactured_interface_patch_geometry_selected_"
        "energy_preflight_authorized"
    )


def test_wp10c9d6c7c2a1_manifest_and_canonical_hashes() -> None:
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
