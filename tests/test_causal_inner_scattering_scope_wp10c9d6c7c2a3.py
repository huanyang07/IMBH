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
    "causal_inner_scattering_scope_wp10c9d6c7c2a3"
)
SUMMARY = CANONICAL / "summary.json"
MANIFEST = CANONICAL / "scope_manifest.json"
DECISIVE = CANONICAL / "decisive_arrays.npz"
PROVENANCE = CANONICAL / "provenance.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_c2a3_selects_only_the_one_way_physical_core_route() -> None:
    summary = _read(SUMMARY)
    assert summary["passed"]
    assert summary["classification"] == (
        "one_way_physical_core_scattering_scope_frozen_"
        "uniform_validation_authorized"
    )
    assert summary["selected_route"] == (
        "one_way_coarse_to_fine_physical_core"
    )
    assert not summary["operator_changed"]
    assert not summary["propagation_executed"]
    assert summary["parent_classification_preserved"] == (
        "manufactured_interface_patch_rejected_"
        "unidirectional_characteristic_core"
    )


def test_c2a3_freezes_causal_packets_surfaces_and_windows() -> None:
    contract = _read(SUMMARY)["packet_and_window_contract"]
    assert contract["exact_physical_core_retained"]
    assert not contract["manufactured_extension_is_physical_background"]
    assert contract["all_characteristics_inward_over_patch"]
    assert contract["maximum_characteristic_speed_over_c"] < 0.0
    assert contract["positive_speed_family_count_everywhere"] == 0
    assert not contract["reflection_coefficient_defined"]
    assert contract["packet_support_faces"] == [52, 95]
    assert contract["interface_face"] == 49
    assert contract["downstream_measurement_face"] == 6
    assert contract["upstream_diagnostic_face"] == 92
    assert contract["windows_derived_before_propagation"]
    assert contract["observed_histories_may_not_move_windows"]
    for family in ("acoustic", "shear", "mixed_shear_acoustic"):
        interface = contract["interface_windows_seconds"][family]
        downstream = contract["downstream_windows_seconds"][family]
        assert 0.0 <= interface[0] < interface[1]
        assert 0.0 <= downstream[0] < downstream[1]
        assert downstream[0] > interface[0]
        assert downstream[1] > interface[1]


def test_c2a3_packets_have_the_frozen_family_content() -> None:
    with np.load(DECISIVE, allow_pickle=False) as source:
        acoustic = source[
            "initial_family_energy_fractions__acoustic"
        ]
        shear = source["initial_family_energy_fractions__shear"]
        mixed = source["initial_family_energy_fractions__mixed"]
        material = source[
            "initial_family_energy_fractions__material_null"
        ]
        assert acoustic[0] >= 1.0 - 1.0e-12
        assert shear[1] >= 1.0 - 1.0e-12
        assert material[2] >= 1.0 - 1.0e-12
        assert abs(mixed[0] - 0.5) <= 1.0e-12
        assert abs(mixed[1] - 0.5) <= 1.0e-12
        assert np.sum(mixed[2:]) <= 1.0e-12
        assert np.count_nonzero(source["packet__zero_null"]) == 0
        assert source["primary_time_samples_seconds"].shape == (513,)
        assert source["measurement_faces"].tolist() == [6, 49, 92]


def test_c2a3_uses_conservative_observability_not_slow_impact() -> None:
    manifest = _read(MANIFEST)
    uncertainty = manifest["uncertainty_and_observability"]
    assert uncertainty["default_combination"] == (
        "conservative_sum_of_deterministic_component_bounds"
    )
    assert uncertainty["RSS_forbidden_without_demonstrated_independence"]
    assert uncertainty["no_slow_impact_threshold"]
    assert uncertainty["observability_factor"] == 5.0
    tier_II = manifest["certification_tiers"][
        "tier_II_one_way_transport"
    ]
    assert tier_II["reflection_coefficient_is_not_defined"]
    assert "transmission_coefficient_T" in tier_II["primary_observables"]
    assert "embedded_minus_uniform_upstream_contamination" in (
        tier_II["secondary_observables"]
    )


def test_c2a3_authorizes_uniform_only_and_preserves_hard_stops() -> None:
    manifest = _read(MANIFEST)
    decision = manifest["binding_decision"]
    assert decision["definitions_only_scope_is_internally_consistent"]
    assert decision["exact_physical_core_retained"]
    assert decision["one_way_uniform_c2b1_authorized"]
    assert not decision["bidirectional_physical_propagation_authorized"]
    assert not decision["generic_bidirectional_method_test_authorized"]
    assert not decision["embedded_c2c1_authorized"]
    assert not decision["nonlinear_authorized"]
    assert not decision["fixed_Q_or_reduction_authorized"]
    assert manifest["authorized_next"] == (
        "WP10c9d6c7c2b1_one_way_uniform_scattering_validation"
    )
    assert "do_not_define_R_for_an_empty_positive_speed_subspace" in (
        manifest["hard_stops"]
    )


def test_c2a3_decisive_arrays_and_hashes_are_canonical() -> None:
    summary = _read(SUMMARY)
    with np.load(DECISIVE, allow_pickle=False) as source:
        assert set(source.files) == set(summary["decisive_array_hashes"])
        for name in source.files:
            assert causal_array_sha256(source[name]) == (
                summary["decisive_array_hashes"][name]
            )
    assert _sha256(DECISIVE) == summary["decisive_arrays_sha256"]


def test_c2a3_manifest_and_provenance_are_canonical() -> None:
    summary = _read(SUMMARY)
    manifest = _read(MANIFEST)
    payload = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_sha256"
    }
    assert causal_canonical_json_sha256(payload) == (
        manifest["manifest_sha256"]
    )
    assert summary["manifest_sha256"] == manifest["manifest_sha256"]
    provenance = _read(PROVENANCE)
    assert provenance["source_parent_commit"] == (
        "1f3570894fc6e41a0770289dc7134356402e17cb"
    )
    assert provenance["scientific_status"] == "CERTIFIED"
    assert provenance["classification"] == summary["classification"]
