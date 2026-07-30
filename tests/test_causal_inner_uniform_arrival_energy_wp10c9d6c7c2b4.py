import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASE = (
    ROOT
    / "results/canonical/"
    "causal_inner_uniform_arrival_energy_wp10c9d6c7c2b4"
)
SOURCE_PARENT = "f9fbdb866d8f692f482c2fbf7455d4a496a11867"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _json(name: str) -> dict:
    return json.loads((CASE / name).read_text(encoding="utf-8"))


def _representative(summary: dict, family: str) -> dict:
    return next(
        item
        for item in summary["arrival_energy"].values()
        if item["family"] == family
        and item["sign"] == 1
        and item["amplitude"] == 1.0
    )


def test_binding_rejection_and_stop_gates():
    summary = _json("summary.json")
    decision = summary["binding_decision"]
    assert summary["analyzed_base_commit"] == SOURCE_PARENT
    assert not summary["passed"]
    assert (
        summary["classification"]
        == "one_way_uniform_arrival_energy_validation_failed_"
        "embedded_discrimination_blocked"
    )
    assert summary["propagation_executed"]
    assert not summary["operator_changed"]
    assert decision["c2b1_rejection_preserved"]
    assert decision["c2b2_interpretation_preserved"]
    assert decision["method_passed"]
    assert decision["tier_I_passed"]
    assert not decision["all_arrival_energy_cases_passed"]
    assert decision["amplitude_positive_null_controls_passed"]
    assert not decision["uniform_c2b4_passed"]
    assert not decision["one_way_embedded_c2c2_authorized"]
    assert not decision["old_embedded_c2c1_authorized"]
    assert not decision["operator_or_interface_redesign_authorized"]
    assert not decision["nonlinear_authorized"]
    assert not decision["fixed_Q_or_reduction_authorized"]
    assert (
        summary["authorized_next"]
        == "WP10c9d6c7c2b5_frozen_arrival_energy_failure_audit"
    )


def test_method_tier_I_and_positive_controls_pass():
    summary = _json("summary.json")
    method = summary["method"]
    controls = summary["amplitude_and_positive_controls"]
    assert method["passed"]
    assert (
        method["maximum_positive_to_semidiscrete_stored_energy_defect"]
        <= 1.4e-15
    )
    assert controls["passed"]
    assert controls["maximum_quadratic_or_sign_scaling_defect"] == 0.0
    assert controls["maximum_zero_state_energy"] == 0.0
    assert controls["maximum_family_partition_relative_defect"] <= 4.5e-15
    for item in summary["tier_I"].values():
        assert item["passed"]
    for level in ("N98", "N196", "N392"):
        ledger = method["exact_semidiscrete_energy"][level]
        assert ledger["maximum_block_power_defect"] <= 4.5e-13
        assert ledger["maximum_face_power_defect"] <= 6.6e-15
        assert ledger["maximum_time_integrated_energy_defect"] <= 1.1e-11


def test_failure_localization_is_frozen():
    summary = _json("summary.json")
    assert len(summary["arrival_energy"]) == 12
    assert all(not item["passed"] for item in summary["arrival_energy"].values())

    acoustic = _representative(summary, "acoustic")
    shear = _representative(summary, "shear")
    mixed = _representative(summary, "mixed_shear_acoustic")

    acoustic_peak = acoustic["scalar_contracts"]["peak_total_arrival"]
    assert np.isclose(acoustic_peak["observed_order"], 1.9645272730886991)
    assert np.isclose(
        acoustic_peak["maximum_fine_normalized_difference"],
        0.05046022389380062,
    )
    assert not acoustic_peak["passed"]
    assert acoustic["scalar_contracts"]["total_arrival"]["passed"]
    assert acoustic["scalar_contracts"]["target_arrival"]["passed"]

    shear_leakage = shear["scalar_contracts"][
        "opposite_family_leakage"
    ]
    assert np.isclose(
        shear_leakage["observed_order"],
        -0.12740020074590755,
    )
    assert not shear_leakage["passed"]
    shear_history = shear["history_contracts"]["leakage"]
    assert np.isclose(
        shear_history["observed_RMS_order"],
        0.6291460163091752,
    )
    assert np.isclose(
        shear_history["refinement_error_cosine"],
        0.7090549022603131,
    )
    assert not shear_history["passed"]

    assert all(
        item["passed"]
        for item in mixed["scalar_contracts"].values()
    )
    assert not mixed["history_contracts"]["total"]["passed"]
    assert (
        mixed["history_contracts"]["total"][
            "maximum_fine_normalized_difference"
        ]
        > 50.0
    )


def test_canonical_hashes_and_sources_are_current():
    summary = _json("summary.json")
    provenance = _json("provenance.json")
    assert provenance["source_parent_commit"] == SOURCE_PARENT
    assert provenance["scientific_status"] == "REJECTED"
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
        for family in (
            "acoustic",
            "shear",
            "mixed_shear_acoustic",
        ):
            assert arrays[f"N392__{family}__total_history"].shape == (513,)
