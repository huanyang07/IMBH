"""Canonical contracts for WP10c9d6c7c2b6c."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASE = (
    ROOT
    / "results/canonical/"
    "causal_inner_uniform_failure_localization_wp10c9d6c7c2b6c"
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


def test_b6c_classifies_direct_continuum_contraction_without_redesign() -> None:
    summary = _read_json(SUMMARY)
    decision = summary["binding_decision"]
    assert summary["classification"] == (
        "direct_continuum_arrival_errors_contract_pairwise_rotation_"
        "preasymptotic_no_redesign"
    )
    assert summary["passed"]
    assert decision["b6b_rejection_preserved"]
    assert decision["all_failed_channels_contract_directly_to_N769"]
    assert decision["equivalent_local_projectors_passed"]
    assert not decision["stable_noncontracting_DAE_mechanism_selected"]
    assert not decision["operator_or_interface_redesign_authorized"]
    assert decision["definitions_only_direct_continuum_manifest_authorized"]
    assert not decision["embedded_propagation_authorized"]
    assert not decision["nonlinear_authorized"]
    assert not decision["fixed_Q_or_reduction_authorized"]


def test_b6c_direct_errors_and_DAE_residuals_contract() -> None:
    summary = _read_json(SUMMARY)
    assert summary["minimum_direct_history_order"] >= 1.65
    assert summary["maximum_fine_direct_history_difference"] <= 0.05
    assert len(summary["direct_to_N769_history"]) == 5
    for result in summary["direct_to_N769_history"].values():
        assert min(result["direct_error_orders"]) >= 0.75
        assert result["fine_direct_response_relative_maximum"] <= 0.05
        assert result["direct_continuum_contract_passed"]
    dae = summary["DAE_localization"]
    assert not dae["stable_candidates"]
    assert not dae["stable_noncontracting_mechanism_selected"]
    for base in ("acoustic", "difference_shear_acoustic", "shear"):
        for result in dae[base].values():
            assert min(result["unsolved_DAE_residual_orders"]) >= 0.75
            assert min(result["mass_solved_rate_error_orders"]) >= 0.75
            assert not result["stable_noncontracting_mechanism_selected"]


def test_b6c_canonical_payload_hashes() -> None:
    summary = _read_json(SUMMARY)
    assert _sha256(CONFIG) == summary["config_sha256"]
    assert _sha256(ARRAYS) == summary["decisive_arrays_sha256"]
    with np.load(ARRAYS, allow_pickle=False) as source:
        assert tuple(source["reference_levels"]) == (98, 196, 392)
        assert source["selected_time_indices"].size == 5
        assert source["continuum_log_radii"].shape == (769,)
    expected = {
        line.split("  ", 1)[1]: line.split("  ", 1)[0]
        for line in SHA256SUMS.read_text(encoding="utf-8").splitlines()
    }
    for name, digest in expected.items():
        assert _sha256(CASE / name) == digest
