"""Canonical contracts for WP10c9d6c7c2c1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (
    causal_canonical_json_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
CASE = (
    ROOT
    / "results/canonical/"
    "causal_inner_direct_continuum_embedded_manifest_wp10c9d6c7c2c1"
)
SUMMARY = CASE / "summary.json"
CONFIG = CASE / "config.json"
MANIFEST = CASE / "embedded_manifest.json"
ARRAYS = CASE / "decisive_arrays.npz"
SHA256SUMS = CASE / "SHA256SUMS.txt"

LABELS = (
    "N98_outer_N98_inner_f49",
    "N98_outer_N196_inner_f49",
    "N98_outer_N392_inner_f49",
)
BASES = {
    "acoustic",
    "shear",
    "mixed_shear_acoustic",
    "difference_shear_acoustic",
    "shear_weighted_shear_acoustic",
    "angle_22p5_acoustic_shear",
    "angle_67p5_acoustic_shear",
    "angle_112p5_acoustic_shear",
    "angle_157p5_acoustic_shear",
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_c2c1_freezes_reference_preflight_without_propagation() -> None:
    summary = _read_json(SUMMARY)
    decision = summary["binding_decision"]
    assert summary["classification"] == (
        "direct_continuum_embedded_contract_frozen_fixed_exterior_"
        "reference_preflight_authorized"
    )
    assert summary["passed"]
    assert not summary["operator_changed"]
    assert not summary["propagation_executed"]
    assert not summary["embedded_or_nonlinear_propagation_executed"]
    assert decision["uniform_direct_continuum_class_preserved"]
    assert decision["layout_and_profile_preflight_passed"]
    assert decision["matched_fixed_exterior_reference_required"]
    assert not decision["matched_fixed_exterior_reference_available_now"]
    assert decision["fixed_exterior_reference_preflight_authorized"]
    assert not decision["embedded_propagation_authorized"]
    assert not decision["operator_or_interface_redesign_authorized"]
    assert not decision["nonlinear_authorized"]
    assert not decision["fixed_Q_or_reduction_authorized"]
    assert summary["authorized_next"] == (
        "WP10c9d6c7c2c2_fixed_exterior_continuum_reference_preflight"
    )


def test_c2c1_layouts_retain_outer_grid_and_one_way_causality() -> None:
    summary = _read_json(SUMMARY)
    layouts = summary["layout_preflight"]["layouts"]
    assert set(layouts) == set(LABELS)
    expected = {
        LABELS[0]: (1, 49, 49, 98),
        LABELS[1]: (2, 98, 49, 147),
        LABELS[2]: (4, 196, 49, 245),
    }
    for label, (ratio, inner, outer, total) in expected.items():
        item = layouts[label]
        local = item["local_energy_and_causality"]
        assert item["passed"]
        assert item["refinement_ratio"] == ratio
        assert item["refined_inner_cells"] == inner
        assert item["fixed_outer_cells"] == outer
        assert item["total_cells"] == total
        assert item["grid_exterior_replay_defect"] == 0.0
        assert item["primitive_exterior_replay_defect"] <= 2.0e-12
        assert item["common_face_radius_replay_defect"] == 0.0
        assert local["passed"]
        assert local["all_characteristics_inward"]
        assert local["maximum_characteristic_speed_over_c"] < 0.0
        assert local["minimum_characteristic_gap"] >= 1.0e-6
        assert local["minimum_energy_eigenvalue"] > 0.0
        assert (
            local["maximum_projector_or_energy_algebra_defect"]
            <= 2.0e-9
        )


def test_c2c1_profiles_replay_and_restrict_exactly() -> None:
    profiles = _read_json(SUMMARY)["profile_preflight"]
    assert profiles["passed"]
    assert profiles["binding_base_count"] == 9
    assert profiles["binding_variant_count"] == 36
    assert set(profiles["per_profile"]) == BASES
    assert profiles["maximum_parent_restriction_defect"] <= 2.0e-12
    assert profiles["maximum_initial_inner_packet_norm"] == 0.0
    assert profiles["minimum_initial_target_family_fraction"] >= (
        1.0 - 1.0e-9
    )
    assert profiles["maximum_family_partition_relative_defect"] <= 2.0e-9
    for profile in profiles["per_profile"].values():
        assert set(profile["layouts"]) == set(LABELS)
        assert all(
            item["passed"] for item in profile["layouts"].values()
        )


def test_c2c1_requires_matched_fixed_exterior_reference() -> None:
    manifest = _read_json(MANIFEST)
    reference = manifest["matched_reference_contract"]
    direct = manifest["tier_II_direct_continuum_contract"]
    assert reference["required_before_embedded_propagation"]
    assert reference["reference_name"] == (
        "fixed_N98_exterior_driven_N769_inner_continuum"
    )
    assert reference["secondary_reference"] == (
        "fixed_N98_exterior_driven_N513_inner_continuum"
    )
    assert reference["continuum_inner"]["primary_nodes"] == 769
    assert reference["continuum_inner"]["secondary_nodes"] == 513
    assert (
        reference["mandatory_preflight_gates"][
            "incoming_interface_characteristic_count"
        ]
        == 5
    )
    assert (
        reference["mandatory_preflight_gates"][
            "incoming_inner_boundary_characteristic_count"
        ]
        == 0
    )
    assert reference["uniform_controls"][
        "full_uniform_continuum_is_not_the_binding_fixed_exterior_reference"
    ]
    assert direct["primary_reference"] == reference["reference_name"]
    assert not direct["pairwise_embedded_error_direction_binding"]
    assert direct["pairwise_embedded_error_direction_reported"]
    assert not direct["raw_local_opposite_family_energy_binding_alone"]
    recorded_hash = manifest.pop("manifest_sha256")
    assert causal_canonical_json_sha256(manifest) == recorded_hash


def test_c2c1_canonical_payload_hashes_and_shapes() -> None:
    summary = _read_json(SUMMARY)
    assert _sha256(CONFIG) == summary["config_sha256"]
    assert _sha256(MANIFEST) == summary["embedded_manifest_file_sha256"]
    assert _sha256(ARRAYS) == summary["decisive_arrays_sha256"]
    with np.load(ARRAYS, allow_pickle=False) as source:
        assert tuple(source["refinement_ratios"]) == (1, 2, 4)
        assert tuple(source["reference_levels"]) == (98, 196, 392)
        assert tuple(source["continuum_nodes"]) == (513, 769)
        assert tuple(source["common_parent_faces"]) == (
            0,
            6,
            49,
            52,
            92,
            95,
            98,
        )
        assert source["binding_variant_table"].shape == (36, 3)
        assert source[f"{LABELS[0]}__grid_edges"].shape == (99,)
        assert source[f"{LABELS[1]}__grid_edges"].shape == (148,)
        assert source[f"{LABELS[2]}__grid_edges"].shape == (246,)
        for base in BASES:
            assert source[f"{base}__{LABELS[0]}__packet"].shape == (98, 5)
            assert source[f"{base}__{LABELS[1]}__packet"].shape == (147, 5)
            assert source[f"{base}__{LABELS[2]}__packet"].shape == (245, 5)
    expected = {
        line.split("  ", 1)[1]: line.split("  ", 1)[0]
        for line in SHA256SUMS.read_text(encoding="utf-8").splitlines()
    }
    for name, digest in expected.items():
        assert _sha256(CASE / name) == digest
