"""Canonical contracts for the c3a1 physical-background manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = (
    ROOT
    / "results/canonical/"
    "causal_inner_physical_background_nonlinear_readiness_"
    "manifest_wp10c9d6c7c3a1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_c3a1_certifies_physical_background_only() -> None:
    summary = json.loads(
        (CANONICAL / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["passed"]
    assert summary["classification"] == (
        "physical_embedded_background_nonlinear_ready_"
        "monolithic_bdf_method_preflight_authorized"
    )
    assert summary["authorized_next"] == (
        "WP10c9d6c7c3b1_monolithic_bdf_method_preflight"
    )
    assert not summary["operator_changed"]
    assert not summary["propagation_executed"]
    assert summary["c3a_rejection_preserved"]
    assert summary["c7c1b_strict_classification_preserved"]
    assert summary["c7c1b_tier_I_direct_passed"]
    assert summary["monolithic_bdf_method_preflight_authorized"]
    assert not summary["nonlinear_physical_ladder_authorized"]


def test_c3a1_physical_and_monolithic_gates_pass() -> None:
    summary = json.loads(
        (CANONICAL / "summary.json").read_text(encoding="utf-8")
    )
    audit = summary["physical_background_audit"]
    assert audit["variant_count"] == 48
    assert audit["maximum_h_over_r"] <= 0.25
    assert audit["minimum_scattering_optical_depth"] > 1.0
    assert audit["minimum_reconstruction_factor"] >= 1.0 - 1.0e-12
    assert audit["maximum_incoming_excision_characteristics"] == 0
    assert audit["maximum_coupling_trace_jump"] <= 1.0e-4
    assert audit["maximum_restriction_defect"] <= 2.0e-12
    assert audit["maximum_monolithic_block_ledger_defect"] <= 1.0e-12
    assert audit["maximum_center_broken_path_adjustment"] <= 2.0e-8


def test_c3a1_freezes_background_subtracted_response() -> None:
    manifest = json.loads(
        (CANONICAL / "physical_background_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert "unperturbed background" in manifest[
        "nonlinear_response_contract"
    ]["comparison"]
    assert manifest["tier_II_interface_observability"] == (
        "unresolved_nonpromoted"
    )
    method = manifest["method_preflight_contract"]
    assert method["maximum_scaled_residual"] == 1.0e-10
    assert method["BDF2_split_restart_replay"] == "bitwise"


def test_c3a1_canonical_hashes() -> None:
    summary = json.loads(
        (CANONICAL / "summary.json").read_text(encoding="utf-8")
    )
    assert _sha256(CANONICAL / "decisive_arrays.npz") == summary[
        "decisive_arrays_sha256"
    ]
    with np.load(CANONICAL / "decisive_arrays.npz") as source:
        assert source["maximum_h_over_r"].shape == (3, 4, 4)
        assert source["output_times_seconds"].shape == (65,)
        assert np.max(source["incoming_excision_characteristics"]) == 0
    for line in (CANONICAL / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        digest, name = line.split("  ", 1)
        assert _sha256(CANONICAL / name) == digest
