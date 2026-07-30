import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASE = (
    ROOT
    / "results/canonical/"
    "causal_inner_uniform_arrival_conditioning_wp10c9d6c7c2b5a"
)
SOURCE_PARENT = "dbfd8bdaf859fa23f530c5c4f00f78fa407137d3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _json(name: str) -> dict:
    return json.loads((CASE / name).read_text(encoding="utf-8"))


def test_b5a_preserves_rejection_and_authorizes_only_transfer_audit():
    summary = _json("summary.json")
    decision = summary["binding_decision"]
    assert summary["analyzed_base_commit"] == SOURCE_PARENT
    assert summary["passed"]
    assert (
        summary["classification"]
        == "arrival_history_conditioning_and_horizon_audit_complete_"
        "shear_family_transfer_audit_required"
    )
    assert (
        summary["parent_classification"]
        == "one_way_uniform_arrival_energy_validation_failed_"
        "embedded_discrimination_blocked"
    )
    assert decision["c2b4_rejection_preserved"]
    assert decision[
        "absolute_initial_energy_history_scale_identified_as_ill_conditioned"
    ]
    assert decision["acoustic_peak_is_convergent_but_old_gate_remains_failed"]
    assert decision["horizon_complete"]
    assert decision["shear_family_transfer_audit_authorized"]
    assert not decision["revised_uniform_recertification_authorized"]
    assert not decision["embedded_authorized"]
    assert not decision["operator_or_interface_redesign_authorized"]
    assert not decision["nonlinear_authorized"]
    assert not decision["fixed_Q_or_reduction_authorized"]
    assert (
        summary["authorized_next"]
        == "WP10c9d6c7c2b5b_shear_family_transfer_and_projector_audit"
    )


def test_gain_conditioning_peak_and_horizon_are_recorded():
    summary = _json("summary.json")
    families = summary["families"]
    acoustic = families["acoustic"]["total"]["conditioning"]
    assert acoustic["absolute_fine_maximum_difference"] > 100.0
    assert acoustic["response_relative_fine_maximum_difference"] < 0.06
    assert acoustic["weighted_rms_order"] > 1.9
    assert acoustic["shape_fine_maximum_difference"] < 0.06

    peak = families["acoustic"]["peak_location"]
    for level in ("N98", "N196", "N392"):
        item = peak[level]
        assert 3.0 < item["interpolated_time_seconds"] < 4.5
        assert item["interpolated_value"] > 3000.0
        assert 1.8 < item["energy_centroid_radius_rg"] < 15.0
        assert 1.8 < item["peak_cell_radius_rg"] < 15.0
        assert 0.0 < item["peak_cell_energy_fraction"] < 1.0

    assert summary["all_total_target_leakage_horizons_complete"]
    for family in (
        "acoustic",
        "shear",
        "mixed_shear_acoustic",
    ):
        for observable in ("total", "target", "leakage"):
            item = families[family][observable]
            assert not item["complete_uncertainty_contract_closed"]
            assert item["uncertainty_components_unresolved"] == [
                "independent_continuum_history_reference"
            ]
            for horizon in item["horizon"].values():
                assert horizon["complete"]
                assert horizon["terminal_to_peak"] < 0.01


def test_measured_nuisance_is_conservative_and_projector_not_zeroed():
    summary = _json("summary.json")
    item = summary["families"]["shear"]["leakage"][
        "windowed_measured_nuisance_envelope"
    ]
    cm = item["coarse_medium_components_l2"]
    mf = item["medium_fine_components_l2"]
    assert "projection_and_subspace" in cm
    assert "projection_and_subspace" in mf
    assert cm["projection_and_subspace"] > 0.0
    assert mf["projection_and_subspace"] > 0.0
    assert np.isclose(
        item["coarse_medium_conservative_l2"],
        sum(cm.values()),
    )
    assert np.isclose(
        item["medium_fine_conservative_l2"],
        sum(mf.values()),
    )
    assert summary["effective_independent_profile_count"] == 3
    assert summary["historical_case_count"] == 12
    assert summary[
        "historical_cases_are_three_profiles_plus_exact_amplitude_sign_controls"
    ]


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
        assert arrays["primary_times_seconds"].shape == (513,)
        assert arrays["N392__acoustic_peak_cell_energy"].ndim == 1
