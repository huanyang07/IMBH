"""Canonical checks for the fixed-exterior continuum reference."""

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
    "causal_inner_fixed_exterior_continuum_reference_"
    "wp10c9d6c7c2c2"
)


def _read_json(name: str) -> dict:
    return json.loads((CASE / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes())
    return digest.hexdigest()


def test_fixed_exterior_reference_binding_classification() -> None:
    summary = _read_json("summary.json")
    assert summary["passed"]
    assert (
        summary["classification"]
        == "fixed_exterior_continuum_reference_certified_"
        "embedded_propagation_authorized"
    )
    assert (
        summary["authorized_next"]
        == "WP10c9d6c7c2c3_direct_continuum_embedded_discrimination"
    )
    decision = summary["binding_decision"]
    assert decision["matched_fixed_exterior_reference_certified"]
    assert decision["embedded_propagation_authorized"]
    assert not decision["numerical_redesign_authorized"]
    assert not decision["nonlinear_propagation_authorized"]
    assert not decision["fixed_Q_or_reduced_evolution_authorized"]


def test_fixed_exterior_reference_gates() -> None:
    summary = _read_json("summary.json")
    config = _read_json("config.json")
    gates = config["gates"]
    assert (
        summary["reference_comparison"]["action_difference"]["maximum"]
        <= gates["maximum_action_difference"]
    )
    assert (
        summary["maximum_trace_replay_defect"]
        <= gates["maximum_trace_replay_defect"]
    )
    assert (
        summary["maximum_characteristic_boundary_closure_defect"]
        <= gates["maximum_boundary_closure_defect"]
    )
    assert (
        summary["maximum_energy_ledger_defect"]
        <= gates["maximum_energy_ledger_defect"]
    )
    assert (
        summary["maximum_restart_replay_defect"]
        <= gates["restart_replay_tolerance"]
    )
    assert summary["characteristic_counts_passed"]
    for report in summary["per_reference"].values():
        assert report["incoming_interface_characteristic_count"] == 5
        assert report["incoming_inner_boundary_characteristic_count"] == 0


def test_fixed_exterior_decisive_arrays_are_finite_and_shaped() -> None:
    with np.load(CASE / "decisive_arrays.npz", allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    assert arrays["times_seconds"].shape == (513,)
    assert arrays["N513_common_state_endpoint"].shape == (9, 49, 5)
    assert arrays["N769_common_state_endpoint"].shape == (9, 49, 5)
    assert arrays[
        "N513_common_state_response_max_by_time_profile"
    ].shape == (513, 9)
    assert arrays[
        "N769_N513_common_state_difference_max_by_time_profile"
    ].shape == (513, 9)
    assert arrays["N513_manufactured_action"].shape == (2, 49, 5)
    assert arrays["N769_manufactured_action"].shape == (2, 49, 5)
    assert arrays["N513_total_energy"].shape == (513, 9)
    assert arrays["N769_total_energy"].shape == (513, 9)
    assert arrays["interface_trace_acoustic_shear"].shape == (513, 2, 5)
    assert arrays["interface_flux_acoustic_shear"].shape == (513, 2, 5)
    assert all(np.all(np.isfinite(values)) for values in arrays.values())


def test_fixed_exterior_canonical_hashes_and_catalog() -> None:
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
    selected = [
        row
        for row in rows
        if row["case"] == CASE.name
    ]
    assert selected
    assert {row["path"] for row in selected} == {
        str(path.relative_to(ROOT))
        for path in CASE.iterdir()
        if path.is_file()
    }
