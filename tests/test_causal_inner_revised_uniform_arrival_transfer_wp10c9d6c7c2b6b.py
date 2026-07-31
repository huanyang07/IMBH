"""Canonical contracts for WP10c9d6c7c2b6b."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASE = (
    ROOT
    / "results/canonical/"
    "causal_inner_revised_uniform_arrival_transfer_wp10c9d6c7c2b6b"
)
SUMMARY = CASE / "summary.json"
CONFIG = CASE / "config.json"
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


def test_b6b_canonical_rejection_and_stop_gates() -> None:
    summary = _read_json(SUMMARY)
    decision = summary["binding_decision"]
    assert summary["classification"] == (
        "revised_uniform_arrival_transfer_recertification_failed_"
        "embedded_blocked"
    )
    assert not summary["passed"]
    assert decision["tier_I_passed"]
    assert not decision["tier_II_arrival_passed"]
    assert decision["covariant_transfer_passed"]
    assert decision["independent_continuum_passed"]
    assert decision["projector_contract_passed"]
    assert decision["amplitude_sign_controls_passed"]
    assert not decision["definitions_only_embedded_manifest_authorized"]
    assert not decision["embedded_propagation_authorized"]
    assert not decision["operator_or_interface_redesign_authorized"]
    assert not decision["nonlinear_authorized"]
    assert not decision["fixed_Q_or_reduction_authorized"]


def test_b6b_records_exact_failure_set_and_noncertifying_raw_leakage() -> None:
    summary = _read_json(SUMMARY)
    arrival = summary["tier_II_arrival"]
    assert arrival["acoustic"]["total"]["passed"]
    assert not arrival["acoustic"]["target"]["passed"]
    assert arrival["mixed_shear_acoustic"]["passed"]
    assert not arrival["difference_shear_acoustic"]["total"]["passed"]
    assert not arrival["difference_shear_acoustic"]["target"]["passed"]
    assert not arrival["shear"]["total"]["passed"]
    assert not arrival["shear"]["target"]["passed"]
    assert arrival["shear_weighted_shear_acoustic"]["passed"]
    for base in arrival.values():
        assert not base["raw_opposite_family_stored_energy"]["certifying"]


def test_b6b_continuum_uncertainty_and_exact_transfer_contracts() -> None:
    summary = _read_json(SUMMARY)
    continuum = summary["independent_continuum"]
    assert continuum["primary_nodes"] == 769
    assert continuum["secondary_nodes"] == 513
    assert (
        continuum["primary_secondary_action_relative_difference"]
        <= 2.0e-5
    )
    assert (
        continuum[
            "maximum_action_to_independent_quintic_relative_difference"
        ]
        <= 2.0e-5
    )
    assert (
        summary["covariant_transfer"][
            "maximum_exact_block_source_receiver_closure_defect"
        ]
        <= 2.0e-9
    )
    for base in summary["tier_II_arrival"].values():
        for observable in ("total", "target"):
            for metric in (
                "physical_gain_history",
                "unit_shape_history",
                "time_average",
                "peak",
            ):
                report = base[observable][metric]
                assert (
                    report["continuum_reference_to_medium_fine_ratio"]
                    <= 0.1
                )
                assert not report["uncertainty"]["root_sum_square_used"]


def test_b6b_canonical_payload_hashes() -> None:
    summary = _read_json(SUMMARY)
    assert _sha256(CONFIG) == summary["config_sha256"]
    assert _sha256(ARRAYS) == summary["decisive_arrays_sha256"]
    with np.load(ARRAYS, allow_pickle=False) as source:
        assert tuple(source["reference_levels"]) == (98, 196, 392)
        assert tuple(source["continuum_nodes"]) == (513, 769)
        assert source["primary_times_seconds"].shape == (513,)
    expected = {
        line.split("  ", 1)[1]: line.split("  ", 1)[0]
        for line in SHA256SUMS.read_text(encoding="utf-8").splitlines()
    }
    for name, digest in expected.items():
        assert _sha256(CASE / name) == digest
