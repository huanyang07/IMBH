"""Canonical contracts for WP10c9d6c7c2b6d."""

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
    "causal_inner_direct_continuum_contract_manifest_wp10c9d6c7c2b6d"
)
SUMMARY = CASE / "summary.json"
CONFIG = CASE / "config.json"
MANIFEST = CASE / "contract_manifest.json"
ARRAYS = CASE / "decisive_arrays.npz"
SHA256SUMS = CASE / "SHA256SUMS.txt"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_b6d_freezes_direct_continuum_contract_without_propagation() -> None:
    summary = _read_json(SUMMARY)
    decision = summary["binding_decision"]
    assert summary["classification"] == (
        "direct_continuum_arrival_contract_frozen_uniform_"
        "recertification_authorized"
    )
    assert summary["passed"]
    assert not summary["operator_changed"]
    assert not summary["propagation_executed"]
    assert decision["historical_classifications_preserved"]
    assert decision["profile_manifest_certified"]
    assert decision["direct_continuum_contract_frozen"]
    assert decision["uniform_b6e_recertification_authorized"]
    assert not decision["operator_or_interface_redesign_authorized"]
    assert not decision["embedded_authorized"]
    assert not decision["nonlinear_authorized"]
    assert not decision["fixed_Q_or_reduction_authorized"]


def test_b6d_manifest_uses_direct_errors_and_unseen_profiles() -> None:
    manifest = _read_json(MANIFEST)
    direct = manifest["direct_continuum_contract"]
    profiles = manifest["profile_manifest"]
    assert direct["primary_reference"] == (
        "independent_N769_continuum_history"
    )
    assert direct["secondary_reference"] == (
        "independent_N513_continuum_history"
    )
    assert direct["minimum_weighted_RMS_error_order"] == 0.75
    assert direct["minimum_maximum_error_order"] == 0.75
    assert direct["maximum_N392_response_relative_RMS_error"] == 0.05
    assert direct["maximum_N392_response_relative_maximum_error"] == 0.05
    assert direct["pairwise_error_direction_reported_not_binding"]
    assert profiles["binding_base_count"] == 9
    assert profiles["binding_variant_count"] == 36
    assert len(profiles["prospective_heldout_bases"]) == 4
    assert profiles["heldouts_frozen_before_propagation"]
    assert not manifest["scientific_interpretation"]["b6b_reclassified"]
    assert not manifest["scientific_interpretation"][
        "numerical_redesign_selected"
    ]
    recorded_hash = manifest.pop("manifest_sha256")
    assert causal_canonical_json_sha256(manifest) == recorded_hash


def test_b6d_profiles_are_distinct_normalized_angular_heldouts() -> None:
    summary = _read_json(SUMMARY)
    profiles = summary["profile_manifest"]
    coefficients = np.asarray(
        [
            profiles["per_profile"][name]["acoustic_shear_coefficients"]
            for name in profiles["prospective_heldout_bases"]
        ],
        dtype=float,
    )
    assert np.allclose(
        np.linalg.norm(coefficients, axis=1),
        1.0,
        rtol=0.0,
        atol=1.0e-15,
    )
    assert np.unique(np.round(coefficients, 14), axis=0).shape[0] == 4
    assert (
        profiles["maximum_initial_family_partition_relative_defect"]
        <= 2.0e-9
    )
    assert profiles["minimum_initial_target_family_fraction"] >= (
        1.0 - 1.0e-10
    )
    for name in profiles["prospective_heldout_bases"]:
        assert profiles["per_profile"][name]["role"] == (
            "prospective_heldout"
        )


def test_b6d_canonical_payload_hashes() -> None:
    summary = _read_json(SUMMARY)
    assert _sha256(CONFIG) == summary["config_sha256"]
    assert _sha256(MANIFEST) == summary["contract_manifest_file_sha256"]
    assert _sha256(ARRAYS) == summary["decisive_arrays_sha256"]
    with np.load(ARRAYS, allow_pickle=False) as source:
        assert tuple(source["reference_levels"]) == (98, 196, 392)
        assert tuple(source["continuum_nodes"]) == (513, 769)
        assert source["heldout_angles_degrees"].shape == (4,)
        assert source["binding_variant_table"].shape == (36, 3)
    expected = {
        line.split("  ", 1)[1]: line.split("  ", 1)[0]
        for line in SHA256SUMS.read_text(encoding="utf-8").splitlines()
    }
    for name, digest in expected.items():
        assert _sha256(CASE / name) == digest
