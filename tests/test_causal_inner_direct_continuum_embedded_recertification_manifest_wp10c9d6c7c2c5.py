"""Canonical contracts for the c2c5 embedded recertification manifest."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASE = (
    ROOT
    / "results/canonical/"
    "causal_inner_direct_continuum_embedded_recertification_manifest_"
    "wp10c9d6c7c2c5"
)


def _read_json(name: str) -> dict:
    return json.loads((CASE / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_c2c5_freezes_only_the_definitions_manifest() -> None:
    summary = _read_json("summary.json")
    assert summary["passed"]
    assert summary["classification"] == (
        "direct_continuum_embedded_two_route_contract_frozen_"
        "recertification_authorized"
    )
    assert summary["authorized_next"] == (
        "WP10c9d6c7c2c6_direct_continuum_embedded_recertification"
    )
    assert not summary["operator_changed"]
    assert not summary["propagation_executed"]
    decision = summary["binding_decision"]
    assert decision["two_route_contract_frozen"]
    assert decision["embedded_recertification_authorized"]
    assert not decision["operator_or_interface_redesign_authorized"]
    assert not decision["nonlinear_propagation_authorized"]
    assert not decision["fixed_Q_or_reduced_evolution_authorized"]


def test_c2c5_profiles_are_unseen_common_parent_packets() -> None:
    summary = _read_json("summary.json")
    audit = summary["profile_audit"]
    config = _read_json("config.json")
    assert summary["profile_count"] == 8
    assert summary["variant_count"] == 32
    assert audit["passed"]
    assert (
        audit["maximum_restriction_defect"]
        <= config["gates"]["maximum_restriction_defect"]
    )
    assert (
        audit["maximum_initial_inner_activity"]
        <= config["gates"]["maximum_initial_inner_activity"]
    )
    assert (
        audit["coefficient_norm_defect"]
        <= config["gates"]["maximum_coefficient_norm_defect"]
    )
    with np.load(CASE / "decisive_arrays.npz", allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    assert arrays["angles_degrees"].shape == (8,)
    assert arrays["acoustic_shear_coefficients"].shape == (8, 2)
    assert arrays["common_parent_packets"].shape == (8, 98, 5)
    assert arrays["variant_multipliers"].shape == (4,)
    assert np.max(
        np.abs(
            np.linalg.norm(
                arrays["acoustic_shear_coefficients"], axis=1
            )
            - 1.0
        )
    ) <= 1.0e-15


def test_c2c5_two_route_contract_is_narrow_and_prospective() -> None:
    summary = _read_json("summary.json")
    manifest = _read_json("recertification_manifest.json")
    assert summary["manifest_sha256"] == manifest["manifest_sha256"]
    routes = manifest["component_order_routes"]
    assert routes["primary_route"]["minimum_order"] == 0.75
    alternate = routes["alternate_route"]
    assert alternate["requires_all_three_embedded_levels_inside_envelope"]
    assert alternate["maximum_fixed_scale_RMS_error"] == 0.05
    assert alternate["maximum_response_relative_maximum_error"] == 0.05
    assert np.isclose(
        alternate["maximum_reference_uncertainty_fixed_scale"], 0.005
    )
    assert np.isclose(
        alternate["maximum_reference_uncertainty_response_relative"],
        0.005,
    )
    assert not alternate["strict_direct_error_order_required"]
    limits = manifest["alternate_route_limits"]
    assert limits["may_replace_only_failed_significant_component_order"]
    assert limits["global_RMS_order_still_binding"]
    assert limits["global_refinement_error_cosine_still_binding"]
    assert limits["non_boundary_component_failure_is_binding_failure"]
    assert all(
        item["role"] == "prospective_unseen_heldout"
        for item in manifest["binding_profiles"]
    )


def test_c2c5_canonical_hashes_and_catalog() -> None:
    summary = _read_json("summary.json")
    assert summary["config_sha256"] == _sha256(CASE / "config.json")
    assert summary["recertification_manifest_file_sha256"] == _sha256(
        CASE / "recertification_manifest.json"
    )
    assert summary["decisive_arrays_sha256"] == _sha256(
        CASE / "decisive_arrays.npz"
    )
    sums = {}
    for line in (CASE / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        digest, name = line.split("  ", 1)
        sums[name] = digest
    for path in CASE.iterdir():
        if path.is_file() and path.name != "SHA256SUMS.txt":
            assert sums[path.name] == _sha256(path)
    with (
        ROOT / "results/manifests/canonical_artifacts.csv"
    ).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row["case"] == CASE.name]
    assert selected
    assert {row["path"] for row in selected} == {
        str(path.relative_to(ROOT))
        for path in CASE.iterdir()
        if path.is_file()
    }
