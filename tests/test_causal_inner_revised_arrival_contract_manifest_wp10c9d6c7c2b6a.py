import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASE = (
    ROOT
    / "results/canonical/"
    "causal_inner_revised_arrival_contract_manifest_wp10c9d6c7c2b6a"
)
SOURCE_PARENT = "7857051b4292c3101456019210f7437c73fd8621"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _json(name: str) -> dict:
    return json.loads((CASE / name).read_text(encoding="utf-8"))


def test_canonical_revised_contract_and_stop_gates():
    summary = _json("summary.json")
    decision = summary["binding_decision"]
    assert summary["analyzed_base_commit"] == SOURCE_PARENT
    assert summary["passed"]
    assert not summary["operator_changed"]
    assert not summary["propagation_executed"]
    assert (
        summary["classification"]
        == "revised_uniform_arrival_transfer_contract_frozen_"
        "recertification_authorized"
    )
    assert (
        summary["authorized_next"]
        == "WP10c9d6c7c2b6b_revised_uniform_arrival_transfer_"
        "recertification"
    )
    assert decision["historical_classifications_preserved"]
    assert decision["profile_manifest_certified"]
    assert decision["revised_uniform_contract_frozen"]
    assert decision["uniform_b6b_recertification_authorized"]
    assert not decision["raw_local_family_leakage_certifying"]
    assert not decision["embedded_authorized"]
    assert not decision["operator_or_interface_redesign_authorized"]
    assert not decision["nonlinear_authorized"]
    assert not decision["fixed_Q_or_reduction_authorized"]


def test_profiles_are_frozen_with_prospective_mixtures():
    summary = _json("summary.json")
    profiles = summary["profile_manifest"]
    assert profiles["passed"]
    assert profiles["binding_base_count"] == 5
    assert profiles["binding_variant_count"] == 20
    assert profiles["calibration_bases"] == [
        "acoustic",
        "shear",
        "mixed_shear_acoustic",
    ]
    assert profiles["prospective_heldout_bases"] == [
        "difference_shear_acoustic",
        "shear_weighted_shear_acoustic",
    ]
    assert profiles["minimum_initial_target_family_fraction"] >= 1 - 1e-10
    assert (
        profiles["maximum_initial_family_partition_relative_defect"]
        <= 2e-9
    )
    assert profiles["heldouts_frozen_before_recertification_propagation"]
    assert profiles[
        "amplitude_and_sign_variants_are_controls_not_independent_profiles"
    ]

    with np.load(CASE / "decisive_arrays.npz", allow_pickle=False) as arrays:
        assert np.array_equal(arrays["reference_levels"], (98, 196, 392))
        assert np.array_equal(arrays["source_band_faces_N98"], (52, 95))
        assert np.array_equal(arrays["receiving_band_faces_N98"], (6, 49))
        assert arrays["binding_variant_table"].shape == (20, 3)
        acoustic = arrays["packet__acoustic"]
        shear = arrays["packet__shear"]
        assert np.allclose(
            arrays["packet__difference_shear_acoustic"],
            (acoustic - shear) / np.sqrt(2.0),
            rtol=0.0,
            atol=0.0,
        )
        assert np.allclose(
            arrays["packet__shear_weighted_shear_acoustic"],
            0.5 * acoustic + np.sqrt(3.0) * 0.5 * shear,
            rtol=0.0,
            atol=0.0,
        )


def test_revised_observable_projector_and_uncertainty_contracts():
    manifest = _json("contract_manifest.json")
    interpretation = manifest["scientific_interpretation"]
    assert interpretation["old_absolute_arrival_history_contract_rejected"]
    assert not interpretation["uniform_operator_rejected"]
    assert not interpretation["positive_energy_rejected"]
    assert not interpretation["raw_local_opposite_family_leakage_certifying"]

    arrival = manifest["arrival_history_definition"]
    assert arrival["physical_gain_may_exceed_one"]
    assert "response-relative" in arrival["binding_accuracy_normalization"]
    assert arrival["old_initial_energy_absolute_0p05_gate_reused"] is False
    assert arrival["c2b4_result_reclassified"] is False
    assert arrival["reference"][
        "stop_if_independent_history_reference_unavailable"
    ]

    projector = manifest["projector_contract"]
    assert projector["raw_opposite_family_leakage_is_noncertifying"]
    assert projector["hard_stop_on_unresolved_spectral_cluster"]
    assert (
        projector["frozen_midpoint_projector"]
        == "rotation_diagnostic_only_not_equivalent_uncertainty"
    )

    transfer = manifest["covariant_transfer_contract"]
    assert transfer["binding_integrals"] == [
        "total_receiver_work",
        "target_family_receiver_work",
    ]
    assert "opposite_family_receiver_work" in transfer[
        "reported_nonbinding_integrals"
    ]

    uncertainty = manifest["uncertainty_contract"]
    assert "root-sum-square is forbidden" in uncertainty["combination"]
    assert uncertainty["observability_factor"] == 5.0
    assert uncertainty["no_slow_impact_threshold"]
    assert "non-certifying" in uncertainty["error_direction_rule"]


def test_canonical_hashes_and_sources_are_current():
    summary = _json("summary.json")
    provenance = _json("provenance.json")
    assert provenance["source_parent_commit"] == SOURCE_PARENT
    assert provenance["scientific_status"] == "CERTIFIED"
    assert summary["decisive_arrays_sha256"] == _sha256(
        CASE / "decisive_arrays.npz"
    )
    assert summary["config_sha256"] == _sha256(CASE / "config.json")
    for relative, expected in summary["implementation_source_hashes"].items():
        assert _sha256(ROOT / relative) == expected

    declared = {}
    for line in (CASE / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        digest, name = line.split("  ", 1)
        declared[name] = digest
    for name in (
        "config.json",
        "contract_manifest.json",
        "decisive_arrays.npz",
        "provenance.json",
        "summary.json",
    ):
        assert declared[name] == _sha256(CASE / name)
