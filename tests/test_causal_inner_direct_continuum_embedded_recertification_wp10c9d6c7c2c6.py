"""Canonical contracts for the c2c6 embedded recertification."""

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
    "causal_inner_direct_continuum_embedded_recertification_"
    "wp10c9d6c7c2c6"
)
ALTERNATE_PROFILE = "unseen_angle_11p25_acoustic_shear"


def _read_json(name: str) -> dict:
    return json.loads((CASE / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_c2c6_certifies_only_the_frozen_linear_embedded_class() -> None:
    summary = _read_json("summary.json")
    assert summary["passed"]
    assert summary["classification"] == (
        "direct_continuum_embedded_linear_class_certified_"
        "bounded_nonlinear_manifest_authorized"
    )
    assert summary["authorized_next"] == (
        "WP10c9d6c7c3a_bounded_nonlinear_contract_manifest"
    )
    assert not summary["operator_changed"]
    assert not summary["new_state_propagation_executed"]
    assert summary["cached_basis_recombination_executed"]
    decision = summary["binding_decision"]
    assert decision["declared_frozen_linear_embedded_class_certified"]
    assert decision[
        "definitions_only_bounded_nonlinear_manifest_authorized"
    ]
    assert not decision["operator_or_interface_redesign_authorized"]
    assert not decision["nonlinear_propagation_authorized"]
    assert not decision["fixed_Q_or_reduced_evolution_authorized"]


def test_c2c6_all_unseen_profiles_pass_with_one_alternate_route() -> None:
    reports = _read_json("summary.json")["comparison"][
        "profile_reports"
    ]
    assert len(reports) == 8
    assert all(report["passed"] for report in reports.values())
    alternates = [
        (profile, kind, component, result)
        for profile, report in reports.items()
        for kind in ("instantaneous_exports", "cumulative_exports")
        for component, result in report[kind][
            "alternate_route"
        ].items()
    ]
    assert len(alternates) == 1
    profile, kind, component, result = alternates[0]
    assert profile == ALTERNATE_PROFILE
    assert kind == "cumulative_exports"
    assert component == "coupling_killing_energy_flux"
    assert result["passed"]
    assert max(
        item["response_relative_maximum_error"]
        for item in result["per_level"]
    ) < 0.05
    for profile, report in reports.items():
        assert report["state"]["passed"]
        assert report["instantaneous_exports"]["passed"]
        assert report["cumulative_exports"]["passed"]
        if profile != ALTERNATE_PROFILE:
            assert report["cumulative_exports"]["route_used"] == (
                "strict_pairwise_component_order"
            )


def test_c2c6_state_and_global_export_gates_remain_binding() -> None:
    summary = _read_json("summary.json")
    config = _read_json("config.json")
    contract = config["tier_I_global_contract"]
    assert summary["reference_exact_integral_solve_passed"]
    assert (
        summary["maximum_reference_exact_integral_solve_residual"]
        <= config["component_order_routes"]["alternate_route"][
            "maximum_exact_integral_solve_residual"
        ]
    )
    for report in summary["comparison"]["profile_reports"].values():
        assert (
            report["state"]["pairwise_observed_order"]
            >= contract["minimum_RMS_order"]
        )
        for kind in ("instantaneous_exports", "cumulative_exports"):
            metrics = report[kind]
            assert metrics["global_gates_passed"]
            assert (
                metrics["observed_rms_order"]
                >= contract["minimum_RMS_order"]
            )
            assert (
                metrics["observed_maximum_order"]
                >= contract["minimum_maximum_order"]
            )
            assert (
                metrics["refinement_error_cosine"]
                >= contract["minimum_refinement_error_cosine"]
            )


def test_c2c6_canonical_payload_and_hashes() -> None:
    summary = _read_json("summary.json")
    assert summary["config_sha256"] == _sha256(CASE / "config.json")
    assert summary["decisive_arrays_sha256"] == _sha256(
        CASE / "decisive_arrays.npz"
    )
    with np.load(CASE / "decisive_arrays.npz", allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    assert arrays["times_seconds"].shape == (513,)
    assert arrays["profile_pass_flags"].shape == (8,)
    assert np.all(arrays["profile_pass_flags"] == 1)
    assert arrays["acoustic_shear_coefficients"].shape == (8, 2)
    assert arrays["N769_state_endpoint"].shape == (8, 98, 5)
    assert all(np.all(np.isfinite(values)) for values in arrays.values())
    assert (CASE / "decisive_arrays.npz").stat().st_size < 5 * 1024 * 1024
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
