"""Canonical contracts for the direct-continuum embedded discrimination."""

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
    "causal_inner_direct_continuum_embedded_discrimination_"
    "wp10c9d6c7c2c3"
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
FAILED_BASES = {"acoustic", "difference_shear_acoustic"}


def _read_json(name: str) -> dict:
    return json.loads((CASE / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_c2c3_preserves_binding_negative_classification() -> None:
    summary = _read_json("summary.json")
    assert not summary["passed"]
    assert summary["classification"] == (
        "direct_continuum_embedded_discrimination_failed_"
        "nonlinear_blocked"
    )
    assert summary["authorized_next"] == (
        "diagnose_direct_continuum_embedded_failure"
    )
    assert set(summary["comparison"]["failed_profiles"]) == FAILED_BASES
    decision = summary["binding_decision"]
    assert not decision["direct_continuum_embedded_class_certified"]
    assert not decision["bounded_nonlinear_manifest_authorized"]
    assert not decision["numerical_or_interface_redesign_authorized"]
    assert not decision["fixed_Q_or_reduced_evolution_authorized"]


def test_c2c3_failure_is_only_two_cumulative_component_orders() -> None:
    summary = _read_json("summary.json")
    reports = summary["comparison"]["profile_reports"]
    assert set(reports) == EXPECTED_BASES
    assert all(
        report["state"]["passed"]
        and report["instantaneous_exports"]["passed"]
        for report in reports.values()
    )
    assert {
        name
        for name, report in reports.items()
        if not report["cumulative_exports"]["passed"]
    } == FAILED_BASES
    assert (
        reports["acoustic"]["cumulative_exports"]["component_orders"][
            "coupling_killing_energy_flux"
        ]
        < 0.75
    )
    assert (
        reports["difference_shear_acoustic"]["cumulative_exports"][
            "component_orders"
        ]["inner_angular_momentum_flux"]
        < 0.75
    )
    for name, report in reports.items():
        if name not in FAILED_BASES:
            assert report["passed"]


def test_c2c3_direct_reference_and_method_gates_pass() -> None:
    summary = _read_json("summary.json")
    config = _read_json("config.json")
    contract = config["tier_I_contract"]
    assert all(
        method["passed"] for method in summary["method_reports"].values()
    )
    assert (
        summary["maximum_restart_replay_defect"]
        <= config["maximum_restart_defect"]
    )
    assert (
        summary["maximum_exact_integral_relative_solve_residual"]
        <= config["maximum_exact_integral_residual"]
    )
    for report in summary["comparison"]["profile_reports"].values():
        state = report["state"]
        assert state["pairwise_observed_order"] >= contract[
            "minimum_RMS_order"
        ]
        assert (
            state["pairwise_refinement_error_cosine"]
            >= contract["minimum_refinement_error_cosine"]
        )
        assert (
            state["reference_uncertainty_to_fine_direct_error_ratio"]
            <= config["maximum_reference_uncertainty_to_fine_error"]
        )


def test_c2c3_decisive_arrays_are_compact_finite_and_shaped() -> None:
    with np.load(CASE / "decisive_arrays.npz", allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    assert arrays["times_seconds"].shape == (513,)
    assert arrays["instantaneous_metric_matrix"].shape == (9, 6)
    assert arrays["cumulative_metric_matrix"].shape == (9, 6)
    assert arrays["state_metric_matrix"].shape == (9, 8)
    assert arrays["N769_reference_state_endpoint"].shape == (9, 98, 5)
    assert arrays["N513_reference_state_endpoint"].shape == (9, 98, 5)
    assert arrays["profile_pass_flags"].shape == (9,)
    assert int(np.sum(arrays["profile_pass_flags"])) == 7
    assert all(np.all(np.isfinite(values)) for values in arrays.values())
    assert (CASE / "decisive_arrays.npz").stat().st_size < 5 * 1024 * 1024


def test_c2c3_canonical_hashes_and_catalog() -> None:
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
