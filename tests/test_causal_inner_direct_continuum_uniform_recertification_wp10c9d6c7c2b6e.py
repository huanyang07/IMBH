"""Canonical contracts for WP10c9d6c7c2b6e."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASE = (
    ROOT
    / "results/canonical/"
    "causal_inner_direct_continuum_uniform_recertification_wp10c9d6c7c2b6e"
)
SUMMARY = CASE / "summary.json"
CONFIG = CASE / "config.json"
ARRAYS = CASE / "decisive_arrays.npz"
SHA256SUMS = CASE / "SHA256SUMS.txt"

EXPECTED_BASES = {
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
EXPECTED_HELDOUTS = {
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


def test_b6e_certifies_uniform_class_without_broadening_authority() -> None:
    summary = _read_json(SUMMARY)
    decision = summary["binding_decision"]
    assert summary["classification"] == (
        "direct_continuum_uniform_arrival_class_certified_"
        "embedded_manifest_authorized"
    )
    assert summary["passed"]
    assert summary["propagation_executed"]
    assert summary["independent_N769_N513_propagation_executed"]
    assert not summary["operator_changed"]
    assert not summary["embedded_or_nonlinear_propagation_executed"]
    assert decision["tier_I_passed"]
    assert decision["direct_continuum_arrival_passed"]
    assert decision["covariant_transfer_passed"]
    assert decision["independent_continuum_passed"]
    assert decision["projector_contract_passed"]
    assert decision["amplitude_sign_controls_passed"]
    assert decision["uniform_direct_continuum_class_certified"]
    assert decision["definitions_only_embedded_manifest_authorized"]
    assert not decision["embedded_propagation_authorized"]
    assert not decision["operator_or_interface_redesign_authorized"]
    assert not decision["nonlinear_authorized"]
    assert not decision["fixed_Q_or_reduction_authorized"]
    assert summary["authorized_next"] == (
        "WP10c9d6c7c2c1_direct_continuum_embedded_manifest"
    )


def test_b6e_all_calibration_and_heldout_profiles_pass() -> None:
    summary = _read_json(SUMMARY)
    tier_i = summary["tier_I"]
    tier_ii = summary["tier_II_direct_continuum_arrival"]
    transfer = summary["covariant_transfer"]["by_base"]
    assert set(tier_i) == EXPECTED_BASES
    assert set(tier_ii) == EXPECTED_BASES
    assert set(transfer) == EXPECTED_BASES
    assert {
        name for name, result in tier_i.items()
        if result["role"] == "prospective_heldout"
    } == EXPECTED_HELDOUTS
    assert all(result["passed"] for result in tier_i.values())
    assert all(result["passed"] for result in tier_ii.values())
    assert all(result["passed"] for result in transfer.values())
    for result in tier_ii.values():
        for observable in ("total", "target"):
            assert result[observable]["passed"]
            assert result[observable]["physical_gain_history"]["passed"]
            assert result[observable]["unit_shape_history"]["passed"]


def test_b6e_direct_continuum_extrema_pass_frozen_gates() -> None:
    extrema = _read_json(SUMMARY)["direct_continuum_extrema"]
    assert extrema["minimum_RMS_error_order"] >= 0.75
    assert extrema["minimum_maximum_error_order"] >= 0.75
    assert extrema["maximum_N392_relative_RMS_error"] <= 0.05
    assert extrema["maximum_N392_relative_maximum_error"] <= 0.05
    assert extrema["minimum_N392_continuum_history_cosine"] >= 0.90
    assert extrema["maximum_peak_response_relative_error"] <= 0.05
    assert extrema["maximum_time_average_response_relative_error"] <= 0.05
    assert extrema["maximum_peak_time_window_fraction"] <= 0.05
    assert extrema["maximum_reference_to_fine_error_ratio"] <= 0.10


def test_b6e_continuum_projector_transfer_and_scaling_contracts() -> None:
    summary = _read_json(SUMMARY)
    continuum = summary["independent_continuum"]
    projector = summary["projector_contract"]
    scaling = summary["amplitude_and_sign_controls"]
    transfer = summary["covariant_transfer"]
    assert continuum["passed"]
    assert continuum["primary_secondary_action_relative_difference"] <= 2.0e-5
    assert continuum["maximum_energy_algebra_defect"] <= 1.0e-10
    assert continuum["maximum_restart_defect"] <= 1.0e-10
    assert continuum["minimum_spectral_gap"] > 0.0
    assert projector["passed"]
    assert projector["maximum_projector_algebra_defect"] <= 1.0e-8
    assert projector["maximum_equivalent_local_projector_difference"] <= 1.0e-8
    assert projector["maximum_family_partition_defect"] <= 1.0e-10
    assert transfer["passed"]
    assert (
        transfer["maximum_exact_block_source_receiver_closure_defect"]
        <= 1.0e-10
    )
    assert scaling["passed"]
    assert scaling["variant_count"] == 36
    assert scaling["maximum_linear_state_or_flux_scaling_defect"] == 0.0
    assert scaling["maximum_quadratic_energy_scaling_defect"] == 0.0
    assert scaling["maximum_sign_symmetry_defect"] == 0.0


def test_b6e_canonical_payload_hashes_and_shapes() -> None:
    summary = _read_json(SUMMARY)
    config = _read_json(CONFIG)
    assert set(config["binding_bases"]) == set(summary["tier_I"])
    assert config["reference_levels"] == [98, 196, 392]
    assert config["continuum_nodes"] == [513, 769]
    assert _sha256(CONFIG) == summary["config_sha256"]
    assert _sha256(ARRAYS) == summary["decisive_arrays_sha256"]
    with np.load(ARRAYS, allow_pickle=False) as source:
        assert tuple(source["reference_levels"]) == (98, 196, 392)
        assert tuple(source["continuum_nodes"]) == (513, 769)
        assert source["primary_times_seconds"].shape == (513,)
        assert source["continuum_primary_action_rate"].shape == (2, 257, 5)
        assert source["continuum_secondary_action_rate"].shape == (2, 257, 5)
        for name in EXPECTED_BASES:
            assert source[f"{name}__N392_tier_I_exports"].shape == (513, 13)
            for observable in ("total", "target"):
                for kind in ("gain", "shape"):
                    for level in (
                        "coarse",
                        "medium",
                        "fine",
                        "continuum_primary",
                        "continuum_secondary",
                    ):
                        assert source[
                            f"{name}__{observable}__{kind}__{level}"
                        ].shape == (513,)
    expected = {
        line.split("  ", 1)[1]: line.split("  ", 1)[0]
        for line in SHA256SUMS.read_text(encoding="utf-8").splitlines()
    }
    for name, digest in expected.items():
        assert _sha256(CASE / name) == digest
