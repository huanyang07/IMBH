"""Canonical contracts for the embedded cumulative-flux diagnostic."""

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
    "causal_inner_embedded_cumulative_flux_diagnostic_"
    "wp10c9d6c7c2c4"
)
EXPECTED_BASES = {
    "acoustic",
    "shear",
    "mixed_shear_acoustic",
    "difference_shear_acoustic",
    "shear_weighted_shear_acoustic",
    "angle_22p5_acoustic_shear",
    "angle_67p5_acoustic_shear",
    "angle_112p5_acoustic_shear",
    "angle_157p5_acoustic_shear",
}
EXPECTED_CHANNELS = {
    "coupling_killing_energy_flux",
    "inner_angular_momentum_flux",
}


def _read_json(name: str) -> dict:
    return json.loads((CASE / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_c2c4_selects_manifest_without_amending_c2c3() -> None:
    summary = _read_json("summary.json")
    assert summary["passed"]
    assert summary["classification"] == (
        "cumulative_boundary_flux_absolute_envelope_supported_"
        "strict_order_unresolved_manifest_authorized"
    )
    assert summary["authorized_next"] == (
        "WP10c9d6c7c2c5_direct_continuum_embedded_"
        "recertification_manifest"
    )
    assert summary["historical_c2c3_classification_preserved"] == (
        "direct_continuum_embedded_discrimination_failed_"
        "nonlinear_blocked"
    )
    decision = summary["binding_decision"]
    assert decision["historical_c2c3_rejection_preserved"]
    assert not decision["strict_direct_order_convergence_demonstrated"]
    assert decision["absolute_direct_continuum_envelope_supported"]
    assert decision[
        "definitions_only_embedded_recertification_manifest_authorized"
    ]
    assert not decision["operator_or_interface_redesign_authorized"]
    assert not decision["nonlinear_propagation_authorized"]
    assert not decision["fixed_Q_or_reduced_evolution_authorized"]


def test_c2c4_all_absolute_envelopes_pass_but_strict_order_does_not() -> None:
    comparison = _read_json("summary.json")["comparison"]
    reports = comparison["profile_channel_reports"]
    assert set(reports) == EXPECTED_BASES
    assert comparison["all_absolute_direct_envelopes_passed"]
    assert not comparison["absolute_envelope_failed_profile_channels"]
    assert not comparison["all_strict_direct_order_channels_passed"]
    assert comparison["strict_order_failed_profile_channels"]
    for channels in reports.values():
        assert set(channels) == EXPECTED_CHANNELS
        for report in channels.values():
            assert report["absolute_envelope_passed"]
            assert report["instantaneous_direct_continuum"][
                "absolute_envelope_passed"
            ]
            assert report["cumulative_direct_continuum"][
                "absolute_envelope_passed"
            ]


def test_c2c4_historical_failures_are_single_signed_and_conditioned() -> None:
    reports = _read_json("summary.json")["comparison"][
        "profile_channel_reports"
    ]
    acoustic = reports["acoustic"]["coupling_killing_energy_flux"]
    difference = reports["difference_shear_acoustic"][
        "inner_angular_momentum_flux"
    ]
    assert acoustic["conditioning"]["minimum_physical_signal_sign_ratio"] > 0.99
    assert difference["conditioning"]["minimum_physical_signal_sign_ratio"] > 0.99
    assert acoustic["cumulative_direct_continuum"][
        "fine_response_relative_maximum_error"
    ] < 0.05
    assert difference["cumulative_direct_continuum"][
        "fine_response_relative_maximum_error"
    ] < 0.05
    assert acoustic["conditioning"][
        "cumulative_to_instantaneous_error_suppression"
    ][0] < 0.05
    assert difference["conditioning"][
        "cumulative_to_instantaneous_error_suppression"
    ][0] < 0.05
    assert difference["conditioning"][
        "cumulative_pairwise_error_cosine"
    ] < 0.90


def test_c2c4_reference_solves_and_payload() -> None:
    summary = _read_json("summary.json")
    config = _read_json("config.json")
    assert (
        summary["maximum_reference_exact_integral_solve_residual"]
        <= config["maximum_reference_solve_residual"]
    )
    with np.load(CASE / "decisive_arrays.npz", allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    assert arrays["times_seconds"].shape == (513,)
    assert arrays["direct_metric_matrix"].shape == (9, 2, 2, 8)
    assert arrays["N513_reference_signals"].shape == (513, 9, 2)
    assert arrays["N769_reference_signals"].shape == (513, 9, 2)
    assert arrays["N513_reference_cumulative_signals"].shape == (513, 9, 2)
    assert arrays["N769_reference_cumulative_signals"].shape == (513, 9, 2)
    assert arrays["profile_channel_pass_flags"].shape == (18,)
    assert 0 < int(np.sum(arrays["profile_channel_pass_flags"])) < 18
    assert all(np.all(np.isfinite(values)) for values in arrays.values())
    assert (CASE / "decisive_arrays.npz").stat().st_size < 5 * 1024 * 1024


def test_c2c4_canonical_hashes_and_catalog() -> None:
    summary = _read_json("summary.json")
    assert summary["config_sha256"] == _sha256(CASE / "config.json")
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
