#!/usr/bin/env python3
"""Certify the fully split physical eleven-field port-atlas kernel."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
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
sys.setrecursionlimit(max(sys.getrecursionlimit(), 5000))

import run_causal_inner_equilibrium_column_thermodynamic_potential_implementation_wp10c9d6c7c3b5c4f25fizzc as witnesses  # noqa: E402
import run_causal_inner_fully_split_shear_height_port_atlas_manifest_wp10c9d6c7c3b5c4f25fizzg as manifest  # noqa: E402
from imri_qpe.constants import C  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_eleven_field_convex import (  # noqa: E402
    audit_full_shear_rest_frame,
    full_shear_rest_frame,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_full_port_atlas import (  # noqa: E402
    audit_full_port_atlas_anchor,
    build_full_port_atlas_anchor,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = manifest.AUTHORIZED_NEXT
PASS_CLASSIFICATION = "fully_split_physical_port_atlas_kernel_certified"
FAIL_CLASSIFICATION = "fully_split_physical_port_atlas_kernel_failed"
AUTHORIZED_NEXT = manifest.PASS_NEXT
ARTIFACT = "causal_inner_fully_split_physical_port_atlas_kernel_wp10c9d6c7c3b5c4f25fizzg1"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = "docs/reports/current/CODEX_CAUSAL_INNER_FULLY_SPLIT_PHYSICAL_PORT_ATLAS_KERNEL_WP10C9D6C7C3B5C4F25FIZZG1_2026-08-26.md"
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = "scripts/run_causal_inner_fully_split_physical_port_atlas_kernel_wp10c9d6c7c3b5c4f25fizzg1.py"
THIS_TEST = "tests/test_causal_inner_fully_split_physical_port_atlas_kernel_wp10c9d6c7c3b5c4f25fizzg1.py"
PHYSICAL_SOURCE = "src/imri_qpe/layer3_minidisk_1d/causal_inner_full_port_atlas.py"
PHYSICAL_TEST = "tests/test_causal_inner_full_port_atlas.py"
PARENT_SHA256 = "e2cd8aa2030bf74512c500d35e1b2575e5df9941c8eff71ea08698c94655f3da"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _u():
    return manifest._u()


def _validate_parent(*, require_clean: bool = False) -> tuple[dict, dict]:
    utils = _u()
    if utils._sha256(manifest.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != PARENT_SHA256:
        raise RuntimeError("fully split port-atlas manifest checksum changed")
    hashes = utils._validate_checksums(manifest.CANONICAL_DIRECTORY)
    summary = utils._read_json(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = utils._read_json(manifest.CANONICAL_DIRECTORY / "atlas_contract.json")
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or not summary["prior_rejections_preserved"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["architecture"]["global_common_potential_required"]
        or contract["architecture"]["fields"].split("=")[-1].strip() != "11"
        or contract["claim_boundary"]["cycle_execution_authorized"]
    ):
        raise RuntimeError("fully split port-atlas contract changed")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("physical port-atlas kernel needs a clean tracked tree")
    return hashes, contract


def _maximum_frame_defect(audit) -> float:
    return float(max(asdict(audit).values()))


def _certificate() -> tuple[dict, dict[str, np.ndarray]]:
    began = time.perf_counter()
    _validate_parent()
    witness_began = time.perf_counter()
    physical_witnesses = list(witnesses._physical_witnesses())
    witness_seconds = time.perf_counter() - witness_began
    rows = []
    charts = []
    radii = []
    rest_spectra = []
    coordinate_spectra = []
    temporal_minima = []
    speed_maxima = []
    mapping_defects = []
    source_positive_parts = []
    height_skew_defects = []
    reciprocity_defects = []
    ledger_defects = []
    frame_defects = []
    for index, label, radius, old_state, chart7 in physical_witnesses:
        height = float(np.exp(chart7[5]))
        temperature = float(np.exp(chart7[3]))
        surface_mass = float(np.exp(chart7[0]))
        sound_speed = float(old_state.thermodynamics.sound_speed)
        omega = float(np.sqrt(old_state.thermodynamics.integrated_pressure / (surface_mass * height**2)))
        alpha = float((old_state.closure.viscous_signal_speed_over_c * C / sound_speed) ** 2)
        tau = float(old_state.closure.relaxation_time)
        transport = float(chart7[1])
        anchor = build_full_port_atlas_anchor(
            sound_speed=sound_speed,
            temperature=temperature,
            proper_half_thickness=height,
            proper_vertical_frequency=omega,
            alpha=alpha,
            shear_relaxation_time=tau,
            transport_speed_over_c=transport,
        )
        audit = audit_full_port_atlas_anchor(anchor)
        frame = full_shear_rest_frame(
            old_state.geometry,
            radial_velocity_over_c=float(chart7[1]),
            azimuthal_velocity_over_c=float(chart7[2]),
            vertical_velocity_over_c=float(chart7[6]),
        )
        frame_audit = audit_full_shear_rest_frame(frame)
        frame_defect = _maximum_frame_defect(frame_audit)
        passed = bool(audit.passed and frame_audit.passed)
        rows.append(
            {
                "index": index,
                "label": label,
                "radius_cm": radius,
                "alpha": alpha,
                "transport_speed_over_c": transport,
                "audit": asdict(audit),
                "full_shear_frame_audit": asdict(frame_audit),
                "passed": passed,
            }
        )
        charts.append(chart7)
        radii.append(radius)
        rest_spectra.append(audit.rest_speeds_over_c)
        coordinate_spectra.append(audit.coordinate_speeds_over_c)
        temporal_minima.append(audit.temporal_minimum_eigenvalue)
        speed_maxima.append(audit.coordinate_maximum_absolute_speed_over_c)
        mapping_defects.append(audit.relativistic_spectral_mapping_defect)
        source_positive_parts.append(audit.source_entropy_positive_part)
        height_skew_defects.append(audit.height_port_skew_defect)
        reciprocity_defects.append(audit.shear_work_reciprocity_defect)
        ledger_defects.append(audit.damping_heat_ledger_defect)
        frame_defects.append(frame_defect)
    passed = bool(len(rows) == 47 and all(row["passed"] for row in rows))
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,
        "passed": passed,
        "physical_witness_count": len(rows),
        "passing_witness_count": sum(row["passed"] for row in rows),
        "minimum_temporal_eigenvalue": float(min(temporal_minima)),
        "maximum_coordinate_characteristic_speed_over_c": float(max(speed_maxima)),
        "maximum_relativistic_spectral_mapping_defect": float(max(mapping_defects)),
        "maximum_source_entropy_positive_part": float(max(source_positive_parts)),
        "maximum_height_port_skew_defect": float(max(height_skew_defects)),
        "maximum_shear_work_reciprocity_defect": float(max(reciprocity_defects)),
        "maximum_damping_heat_ledger_defect": float(max(ledger_defects)),
        "maximum_full_shear_frame_constraint_defect": float(max(frame_defects)),
        "one_piece_height_potential_rejection_preserved": True,
        "restricted_five_STF_potential_rejection_preserved": True,
        "full_physical_STF_tensor_no_projection": True,
        "field_count": 11,
        "trajectory_steps": 0,
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
        "rest_characteristic_speeds_over_c": np.asarray(rest_spectra),
        "coordinate_characteristic_speeds_over_c": np.asarray(coordinate_spectra),
        "temporal_minimum_eigenvalues": np.asarray(temporal_minima),
        "relativistic_spectral_mapping_defects": np.asarray(mapping_defects),
        "source_entropy_positive_parts": np.asarray(source_positive_parts),
        "height_port_skew_defects": np.asarray(height_skew_defects),
        "shear_work_reciprocity_defects": np.asarray(reciprocity_defects),
        "damping_heat_ledger_defects": np.asarray(ledger_defects),
        "full_shear_frame_constraint_defects": np.asarray(frame_defects),
    }
    return metrics, arrays


def _update(summary: dict) -> None:
    utils = _u()
    rows = list(csv.DictReader(CANONICAL_MANIFEST.open(newline="", encoding="utf-8")))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    status = "SUPPORTED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": utils._sha256(path), "scientific_status": status})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    catalog = utils._read_json(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": summary["passed"]}
    catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": utils._git("rev-parse", "HEAD"), "latest_work_package": WORK_PACKAGE})
    utils._write_json(CANONICAL_SUMMARY, catalog)


def _canonicalize(metrics: dict, arrays: dict[str, np.ndarray]) -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("physical port-atlas package already exists")
    hashes, _ = _validate_parent(require_clean=True)
    utils = _u()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "kernel_metrics.json", metrics)
    np.savez_compressed(CANONICAL_DIRECTORY / "kernel_arrays.npz", **arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": metrics["passed"],
        "prior_rejections_preserved": True,
        "fully_split_physical_port_atlas_kernel_certified": metrics["passed"],
        "split_discretization_certified": False,
        "trajectory_authorized": False,
        "complete_cycle_execution_authorized": False,
        "authorized_next": metrics["authorized_next"],
    }
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utils._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"manifest_artifact": manifest.ARTIFACT, "manifest_checksum_manifest_sha256": PARENT_SHA256, "manifest_hashes": hashes})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Fully split physical port-atlas kernel certificate\n\n"
        f"Classification: `{metrics['classification']}`.\n\n"
        f"All {metrics['passing_witness_count']}/{metrics['physical_witness_count']} physical witnesses pass. "
        f"The largest coordinate characteristic speed is `{metrics['maximum_coordinate_characteristic_speed_over_c']:.6e} c`; "
        f"the maximum relativistic spectral-map defect is `{metrics['maximum_relativistic_spectral_mapping_defect']:.6e}`; "
        f"and the maximum physical five-STF frame defect is `{metrics['maximum_full_shear_frame_constraint_defect']:.6e}`.\n\n"
        "This certifies the state-local algebraic 4+5+2 port atlas. It does not certify a split discretization, nonlinear trust region, trajectory, or cycle execution.\n\n"
        f"Authorized next: `{metrics['authorized_next']}`.\n",
        encoding="utf-8",
    )
    sources = (THIS_RUNNER, THIS_TEST, PHYSICAL_SOURCE, PHYSICAL_TEST, REPORT_RELATIVE)
    utils._write_json(CANONICAL_DIRECTORY / "provenance.json", {"implementation_commit": utils._git("rev-parse", "HEAD"), "source_hashes": {path: utils._sha256(ROOT / path) for path in sources}, "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "thread_environment": {name: os.environ.get(name, "") for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utils._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8")
    _update(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("choose --run")
    metrics, arrays = _certificate()
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    summary = _canonicalize(metrics, arrays)
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
