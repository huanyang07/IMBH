from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "results/canonical"
MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_every_canonical_case_has_provenance_and_valid_checksums() -> None:
    cases = sorted(path for path in CANONICAL.iterdir() if path.is_dir())
    assert len(cases) >= 8
    for case in cases:
        provenance = json.loads((case / "provenance.json").read_text())
        assert "source_tag" in provenance or "source_parent_commit" in provenance
        status = provenance.get("scientific_status", provenance.get("numerical_status"))
        assert status in {
            "CERTIFIED",
            "SUPPORTED BUT NOT FULLY CERTIFIED",
            "DIAGNOSTIC ONLY",
            "REJECTED",
        }
        if "physical_status" in provenance:
            assert provenance["physical_status"] in {
                "SUPPORTED BUT NOT FULLY CERTIFIED",
                "DIAGNOSTIC ONLY",
                "REJECTED",
            }
        for line in (case / "SHA256SUMS.txt").read_text().splitlines():
            expected, filename = line.split("  ", 1)
            assert _sha256(case / filename) == expected


def test_canonical_manifest_matches_files() -> None:
    with MANIFEST.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_files = sum(
        1
        for case in CANONICAL.iterdir()
        if case.is_dir()
        for path in case.iterdir()
        if path.is_file()
    )
    assert len(rows) == expected_files
    for row in rows:
        path = ROOT / row["path"]
        assert path.is_file()
        assert path.stat().st_size == int(row["bytes"])
        assert _sha256(path) == row["sha256"]


def test_wp1_canonical_cases_preserve_legacy_and_closed_states() -> None:
    for name in (
        "signed_flux_legacy_53566fa_N512",
        "signed_flux_angular_closed_wp1_N512",
    ):
        case = CANONICAL / name
        assert case.is_dir()
        summary = json.loads((case / "summary.json").read_text())
        assert summary["tidal_wall"]["converged"]
        assert summary["zero_torque"]["converged"]
        for boundary in ("tidal_wall", "zero_torque"):
            assert (case / f"prescribed_{boundary}.npz").is_file()
            assert (case / f"thermoviscous_{boundary}.npz").is_file()


def test_wp2_canonical_cases_preserve_controls_and_failure_witness() -> None:
    accepted = CANONICAL / "signed_flux_total_energy_rin10_N512"
    summary = json.loads((accepted / "summary.json").read_text())
    assert summary["tidal_wall"]["converged"]
    assert summary["zero_torque"]["converged"]
    for boundary in ("tidal_wall", "zero_torque"):
        with np.load(accepted / f"{boundary}.npz", allow_pickle=False) as state:
            assert bool(state["converged"])
            assert float(state["total_energy_residual"]) < 1.0e-6

    rejected = CANONICAL / "signed_flux_total_energy_near_isco_failure"
    rows = json.loads((rejected / "failure_summary.json").read_text())
    assert any(not row["converged"] for row in rows)
    assert any(
        row["N"] == 512 and row["maximum_log_viscosity_change"] > 0.1
        for row in rows
    )


def test_canonical_numerical_anchors_are_accepted() -> None:
    paths = [
        CANONICAL / "no_wind_mdot5/state.npz",
        CANONICAL / "stream_no_wind_mdot2_fs080/state.npz",
        CANONICAL / "phase_dae_entry_N164/state.npz",
    ]
    for path in paths[:2]:
        with np.load(path, allow_pickle=False) as data:
            assert bool(data["accepted"])

    with np.load(paths[2], allow_pickle=False) as data:
        assert not bool(data["accepted"])
        assert int(data["n_nodes"]) == 164

    with np.load(paths[0], allow_pickle=False) as data:
        assert float(data["ratio"]) == 5.0
        assert int(data["n_nodes"]) == 768
    with np.load(paths[1], allow_pickle=False) as data:
        assert float(data["ratio"]) == 2.0
        assert float(data["stream_source_fraction"]) == 0.8
        assert int(data["n_nodes"]) == 896
