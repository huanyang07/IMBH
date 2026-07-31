"""Canonical contracts for the c3a bounded nonlinear manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = (
    ROOT
    / "results/canonical/"
    "causal_inner_bounded_nonlinear_manifest_wp10c9d6c7c3a"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_c3a_rejects_manufactured_background_without_propagation() -> None:
    summary = json.loads(
        (CANONICAL / "summary.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (CANONICAL / "bounded_nonlinear_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["work_package"] == "WP10c9d6c7c3a"
    assert not summary["passed"]
    assert summary["classification"] == (
        "bounded_nonlinear_manufactured_background_rejected_"
        "physical_background_readiness_manifest_authorized"
    )
    assert summary["authorized_next"] == (
        "WP10c9d6c7c3a1_physical_background_"
        "nonlinear_readiness_manifest"
    )
    assert not summary["operator_changed"]
    assert not summary["propagation_executed"]
    assert not summary["nonlinear_physical_ladder_authorized"]
    assert not summary["monolithic_bdf_method_preflight_authorized"]
    assert manifest["nonlinear_architecture"]["shared_MJE_face_flux"]
    assert not manifest["nonlinear_architecture"][
        "uses_production_generator"
    ]
    assert not manifest["nonlinear_architecture"][
        "uses_production_anchor_storage_derivative"
    ]


def test_c3a_initial_state_failure_is_the_physical_thickness_gate() -> None:
    summary = json.loads(
        (CANONICAL / "summary.json").read_text(encoding="utf-8")
    )
    audit = summary["initial_state_audit"]
    assert audit["variant_count"] == 96
    assert audit["maximum_h_over_r"] > 0.25
    assert audit["minimum_scattering_optical_depth"] > 1.0
    assert audit["minimum_reconstruction_admissibility_factor"] >= (
        1.0 - 1.0e-12
    )
    assert audit["maximum_incoming_excision_characteristics"] == 0
    assert audit["minimum_surface_density"] > 0.0
    assert audit["minimum_temperature"] > 0.0
    assert not audit["passed"]


def test_c3a_gates_and_background_subtraction_are_frozen() -> None:
    manifest = json.loads(
        (CANONICAL / "bounded_nonlinear_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    method = manifest["method_preflight_contract"]
    assert method["requirements"]["maximum_scaled_residual"] == 1.0e-10
    assert method["requirements"]["BDF2_split_restart_replay"] == "bitwise"
    physical = manifest["conditional_physical_ladder_contract"]
    assert physical["not_authorized_until_method_preflight_passes"]
    assert "unperturbed nonlinear background" in physical["comparison"]
    assert physical["spatial_gates"]["minimum_RMS_order"] == 0.75
    assert physical["spatial_gates"][
        "maximum_fine_normalized_difference"
    ] == 0.05
    assert physical["temporal_refinement"][
        "maximum_temporal_to_spatial_error_fraction"
    ] == 0.10


def test_c3a_canonical_payload_and_hashes() -> None:
    summary = json.loads(
        (CANONICAL / "summary.json").read_text(encoding="utf-8")
    )
    assert _sha256(CANONICAL / "decisive_arrays.npz") == summary[
        "decisive_arrays_sha256"
    ]
    with np.load(CANONICAL / "decisive_arrays.npz") as source:
        assert source["profile_angles_degrees"].shape == (8,)
        assert source["nonlinear_amplitudes"].tolist() == [
            0.05,
            -0.05,
            0.025,
            -0.025,
        ]
        assert source["maximum_initial_h_over_r"].shape == (3, 8, 4)
        assert source["physical_output_times_seconds"].shape == (513,)
    recorded = {
        line.split("  ", 1)[1]: line.split("  ", 1)[0]
        for line in (CANONICAL / "SHA256SUMS.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    }
    for name, digest in recorded.items():
        assert _sha256(CANONICAL / name) == digest
