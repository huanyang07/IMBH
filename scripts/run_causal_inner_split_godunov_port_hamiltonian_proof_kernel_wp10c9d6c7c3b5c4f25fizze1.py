#!/usr/bin/env python3
"""Certify the split Godunov/port-Hamiltonian proof kernel."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_equilibrium_column_thermodynamic_potential_implementation_wp10c9d6c7c3b5c4f25fizzc as witnesses  # noqa: E402
import run_causal_inner_split_godunov_port_hamiltonian_architecture_manifest_wp10c9d6c7c3b5c4f25fizze as manifest  # noqa: E402
from imri_qpe.constants import C  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_split_godunov_port import (  # noqa: E402
    audit_split_godunov_port_hamiltonian_form,
    build_split_godunov_port_hamiltonian_form,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = manifest.AUTHORIZED_NEXT
PASS_CLASSIFICATION = "split_Godunov_port_Hamiltonian_kernel_certified"
FAIL_CLASSIFICATION = "split_Godunov_port_Hamiltonian_kernel_failed"
AUTHORIZED_NEXT = manifest.PASS_NEXT
ARTIFACT = (
    "causal_inner_split_godunov_port_hamiltonian_proof_kernel_"
    "wp10c9d6c7c3b5c4f25fizze1"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_SPLIT_GODUNOV_PORT_HAMILTONIAN_"
    "PROOF_KERNEL_WP10C9D6C7C3B5C4F25FIZZE1_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_split_godunov_port_hamiltonian_proof_kernel_"
    "wp10c9d6c7c3b5c4f25fizze1.py"
)
THIS_TEST = (
    "tests/test_causal_inner_split_godunov_port_hamiltonian_proof_kernel_"
    "wp10c9d6c7c3b5c4f25fizze1.py"
)
PHYSICAL_SOURCE = (
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_split_godunov_port.py"
)
PHYSICAL_TEST = "tests/test_causal_inner_split_godunov_port.py"
PARENT_SHA256 = "5896ce05a8f2e8e787e6ea16701b8362d9eb4a9882a1ba8d4c815215380deff7"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _u():
    return manifest._u()


def _validate_parent(*, require_clean: bool = False) -> tuple[dict, dict]:
    utils = _u()
    checksum = manifest.CANONICAL_DIRECTORY / "SHA256SUMS.txt"
    if utils._sha256(checksum) != PARENT_SHA256:
        raise RuntimeError("split architecture manifest checksum changed")
    hashes = utils._validate_checksums(manifest.CANONICAL_DIRECTORY)
    summary = utils._read_json(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = utils._read_json(
        manifest.CANONICAL_DIRECTORY / "architecture_contract.json"
    )
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or not summary["one_piece_height_rejection_preserved"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["field_decomposition"]["total_fields"] != 11
        or contract["claim_boundary"]["cycle_execution_authorized"]
    ):
        raise RuntimeError("split architecture contract changed")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("split proof kernel needs a clean tracked tree")
    return hashes, contract


def _certificate() -> tuple[dict, dict[str, np.ndarray]]:
    began = time.perf_counter()
    _, contract = _validate_parent()
    rows = []
    charts = []
    radii = []
    speed_maxima = []
    temporal_minima = []
    port_defects = []
    source_positive = []
    energy_defects = []
    spectra = []
    witness_began = time.perf_counter()
    physical_witnesses = list(witnesses._physical_witnesses())
    witness_seconds = time.perf_counter() - witness_began
    for index, label, radius, old_state, chart7 in physical_witnesses:
        height = float(np.exp(chart7[5]))
        temperature = float(np.exp(chart7[3]))
        surface_mass = float(np.exp(chart7[0]))
        omega = float(
            np.sqrt(
                old_state.thermodynamics.integrated_pressure
                / (surface_mass * height**2)
            )
        )
        alpha = float(
            (
                old_state.closure.viscous_signal_speed_over_c
                * C
                / old_state.thermodynamics.sound_speed
            )
            ** 2
        )
        transport = float(chart7[1])
        form = build_split_godunov_port_hamiltonian_form(
            proper_half_thickness=height,
            temperature=temperature,
            proper_vertical_frequency=omega,
            alpha=alpha,
            transport_speed_over_c=transport,
        )
        audit = audit_split_godunov_port_hamiltonian_form(form)
        row = {
            "index": index,
            "label": label,
            "radius_cm": radius,
            "alpha": alpha,
            "transport_speed_over_c": transport,
            "proper_vertical_frequency_per_second": omega,
            "audit": asdict(audit),
            "passed": audit.passed,
        }
        rows.append(row)
        charts.append(chart7)
        radii.append(radius)
        speed_maxima.append(audit.maximum_absolute_characteristic_speed_over_c)
        temporal_minima.append(audit.equilibrated_temporal_minimum_eigenvalue)
        port_defects.append(audit.port_skew_relative_defect)
        source_positive.append(audit.source_entropy_positive_part)
        energy_defects.append(
            max(
                audit.vertical_reversible_energy_relative_defect,
                audit.vertical_damping_heat_ledger_relative_defect,
            )
        )
        spectra.append(audit.characteristic_speeds_over_c)
    passed = bool(len(rows) == 47 and all(row["passed"] for row in rows))
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,
        "passed": passed,
        "physical_witness_count": len(rows),
        "passing_witness_count": sum(row["passed"] for row in rows),
        "minimum_equilibrated_temporal_eigenvalue": float(min(temporal_minima)),
        "maximum_absolute_characteristic_speed_over_c": float(max(speed_maxima)),
        "maximum_port_skew_relative_defect": float(max(port_defects)),
        "maximum_source_entropy_positive_part": float(max(source_positive)),
        "maximum_vertical_energy_ledger_relative_defect": float(max(energy_defects)),
        "one_piece_height_rejection_preserved": True,
        "fixed_height_equilibrium_potential_preserved": True,
        "nine_plus_two_field_count": True,
        "trajectory_steps": 0,
        "full_shear_physical_potential_certified": False,
        "split_discretization_certified": False,
        "complete_cycle_execution_authorized": False,
        "witness_construction_wall_seconds": witness_seconds,
        "certificate_wall_seconds": time.perf_counter() - began,
        "rows": rows,
        "authorized_next": AUTHORIZED_NEXT if passed else None,
    }
    arrays = {
        "witness_charts7": np.asarray(charts),
        "witness_radii_cm": np.asarray(radii),
        "characteristic_speeds_over_c": np.asarray(spectra),
        "equilibrated_temporal_minimum_eigenvalues": np.asarray(temporal_minima),
        "port_skew_relative_defects": np.asarray(port_defects),
        "source_entropy_positive_parts": np.asarray(source_positive),
        "vertical_energy_ledger_relative_defects": np.asarray(energy_defects),
    }
    del contract
    return metrics, arrays


def _update(summary: dict) -> None:
    utils = _u()
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    status = "SUPPORTED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": utils._sha256(path), "scientific_status": status})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    catalog = utils._read_json(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": summary["passed"]}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": utils._git("rev-parse", "HEAD"), "latest_work_package": WORK_PACKAGE})
    utils._write_json(CANONICAL_SUMMARY, catalog)


def _canonicalize(metrics: dict, arrays: dict[str, np.ndarray]) -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("split proof package already exists")
    hashes, _ = _validate_parent(require_clean=True); utils = _u(); CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "proof_metrics.json", metrics)
    np.savez_compressed(CANONICAL_DIRECTORY / "proof_arrays.npz", **arrays)
    summary = {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "classification": metrics["classification"], "passed": metrics["passed"], "one_piece_height_rejection_preserved": True, "split_kernel_certified": metrics["passed"], "full_shear_physical_potential_certified": False, "split_discretization_certified": False, "trajectory_authorized": False, "complete_cycle_execution_authorized": False, "authorized_next": metrics["authorized_next"]}
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utils._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"manifest_artifact": manifest.ARTIFACT, "manifest_checksum_manifest_sha256": PARENT_SHA256, "manifest_hashes": hashes})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Split Godunov/port-Hamiltonian proof certificate\n\n"
        f"Classification: `{metrics['classification']}`.\n\n"
        f"All {metrics['passing_witness_count']}/{metrics['physical_witness_count']} witnesses pass. "
        f"The maximum characteristic speed is `{metrics['maximum_absolute_characteristic_speed_over_c']:.6e} c`, "
        f"the port-skew defect is `{metrics['maximum_port_skew_relative_defect']:.6e}`, and the vertical energy ledger defect is `{metrics['maximum_vertical_energy_ledger_relative_defect']:.6e}`.\n\n"
        "This certifies the algebraic nine-plus-two split kernel only. The physical five-STF potential, split discretization, trajectory, and complete cycle remain unauthorized.\n\n"
        f"Authorized next: `{metrics['authorized_next']}`.\n",
        encoding="utf-8",
    )
    sources = (THIS_RUNNER, THIS_TEST, PHYSICAL_SOURCE, PHYSICAL_TEST, REPORT_RELATIVE)
    utils._write_json(CANONICAL_DIRECTORY / "provenance.json", {"implementation_commit": utils._git("rev-parse", "HEAD"), "source_hashes": {path: utils._sha256(ROOT / path) for path in sources}, "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "thread_environment": {name: os.environ.get(name, "") for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utils._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8")
    _update(summary); return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run", action="store_true"); args = parser.parse_args()
    if not args.run: parser.error("choose --run")
    metrics, arrays = _certificate(); print(json.dumps(metrics, indent=2, sort_keys=True), flush=True); summary = _canonicalize(metrics, arrays); return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
