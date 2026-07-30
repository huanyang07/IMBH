import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASE = (
    ROOT
    / "results/canonical/"
    "causal_inner_uniform_family_transfer_wp10c9d6c7c2b5b"
)
SOURCE_PARENT = "dc27efe1b414143cec67a050f8dc5c9ccff69ee4"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _json(name: str) -> dict:
    return json.loads((CASE / name).read_text(encoding="utf-8"))


def test_b5b_preserves_rejections_and_selects_only_definitions_manifest():
    summary = _json("summary.json")
    decision = summary["binding_decision"]
    assert summary["analyzed_base_commit"] == SOURCE_PARENT
    assert summary["passed"]
    assert (
        summary["classification"]
        == "raw_local_family_leakage_projector_rotation_sensitive_"
        "revised_transfer_observable_manifest_authorized"
    )
    assert decision["c2b4_and_c2b5a_classifications_preserved"]
    assert decision["exact_family_transfer_ledger_passed"]
    assert decision["equivalent_local_projectors_passed"]
    assert decision["independent_continuum_action_passed"]
    assert not decision["family_resolved_leakage_certifying"]
    assert decision["revised_uniform_manifest_authorized"]
    assert not decision["selected_block_local_audit_authorized"]
    assert not decision["uniform_recertification_propagation_authorized"]
    assert not decision["embedded_authorized"]
    assert not decision["operator_or_interface_redesign_authorized"]
    assert not decision["nonlinear_authorized"]
    assert not decision["fixed_Q_or_reduction_authorized"]
    assert (
        summary["authorized_next"]
        == "WP10c9d6c7c2b6a_revised_uniform_arrival_contract_manifest"
    )


def test_projectors_transfer_and_rotation_diagnostic_are_certified():
    summary = _json("summary.json")
    assert (
        summary["maximum_exact_transfer_closure_defect"] < 2.0e-9
    )
    for item in summary["projector_audit"].values():
        assert item["passed"]
        assert item["maximum_polynomial_algebra_defect"] < 2.0e-9
        assert (
            item["maximum_eigenvector_polynomial_projector_defect"]
            < 2.0e-8
        )
        assert item["minimum_spectral_gap"] > 0.0

    observables = summary["projector_observable"]
    local = observables["local_eigenvector"]["shear"]["leakage"]
    polynomial = observables["local_polynomial"]["shear"]["leakage"]
    common = observables["common_N392_field"]["shear"]["leakage"]
    frozen = observables[
        "frozen_receiving_band_midpoint_diagnostic"
    ]["shear"]["leakage"]
    np.testing.assert_allclose(
        local["values"],
        polynomial["values"],
        rtol=2.0e-13,
        atol=2.0e-11,
    )
    assert local["observed_order"] < 0.75
    assert common["observed_order"] < 0.75
    assert frozen["observed_order"] >= 0.75
    assert summary["common_cross_grid_projector_robust"]
    assert (
        summary[
            "shear_leakage_common_projector_"
            "difference_to_fine_spatial_ratio"
        ]
        <= 0.10
    )
    assert summary["raw_shear_leakage_projector_rotation_sensitive"]
    assert not summary["stable_noncontracting_numerical_block_selected"]


def test_independent_continuum_actions_contract_for_all_controls():
    summary = _json("summary.json")
    assert summary["continuum_action_audit_passed"]
    assert summary["independent_continuum_action_reference_available"]
    assert not summary["independent_continuum_history_reference_available"]
    for family in (
        "acoustic",
        "shear",
        "mixed_shear_acoustic",
    ):
        item = summary["continuum_action_audit"][family]
        assert (
            item["primary_secondary_rate_relative_difference"] < 2.0e-5
        )
        assert item["unsolved_DAE_truncation"]["observed_order"] >= 0.75
        assert item["mass_solved_rate_error"]["observed_order"] >= 0.75
        assert item["minimum_block_truncation_order"] >= 0.75
        assert not item["block_truncation"]["candidate_stream"]["active"]


def test_canonical_hashes_and_sources_are_current():
    summary = _json("summary.json")
    provenance = _json("provenance.json")
    assert provenance["source_parent_commit"] == SOURCE_PARENT
    assert provenance["scientific_status"] == "DIAGNOSTIC ONLY"
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
        "decisive_arrays.npz",
        "provenance.json",
        "summary.json",
    ):
        assert declared[name] == _sha256(CASE / name)

    with np.load(CASE / "decisive_arrays.npz", allow_pickle=False) as arrays:
        assert np.array_equal(arrays["reference_levels"], (98, 196, 392))
        assert arrays[
            "N392__shear__integrated_block_source_receiver_cell_work"
        ].ndim == 4
        assert arrays["N392__times_seconds"].shape == (513,)
