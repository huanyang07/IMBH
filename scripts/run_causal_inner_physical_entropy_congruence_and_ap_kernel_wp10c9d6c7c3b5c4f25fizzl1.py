#!/usr/bin/env python3
"""Certify the physical entropy congruence and local AP port propagator."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_equilibrium_column_thermodynamic_potential_implementation_wp10c9d6c7c3b5c4f25fizzc as witnesses  # noqa: E402
import run_causal_inner_physical_entropy_congruence_and_ap_macrostep_manifest_wp10c9d6c7c3b5c4f25fizzl as manifest  # noqa: E402
from imri_qpe.constants import C  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_full_port_atlas import build_full_port_atlas_anchor  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_physical_entropy_congruence import (  # noqa: E402
    audit_ap_fast_propagator,
    audit_corrected_physical_port_atlas,
    audit_physical_entropy_congruence,
    build_corrected_physical_port_atlas,
    build_physical_entropy_congruence,
)


WORK_PACKAGE = manifest.AUTHORIZED_NEXT
PASS_CLASSIFICATION = "physical_entropy_congruence_and_AP_kernel_certified"
FAIL_CLASSIFICATION = "physical_entropy_congruence_and_AP_kernel_failed"
AUTHORIZED_NEXT = manifest.PASS_NEXT
ARTIFACT = (
    "causal_inner_physical_entropy_congruence_and_ap_kernel_"
    "wp10c9d6c7c3b5c4f25fizzl1"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_PHYSICAL_ENTROPY_CONGRUENCE_AND_AP_KERNEL_"
    "WP10C9D6C7C3B5C4F25FIZZL1_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_physical_entropy_congruence_and_ap_kernel_"
    "wp10c9d6c7c3b5c4f25fizzl1.py"
)
THIS_TEST = (
    "tests/test_causal_inner_physical_entropy_congruence_and_ap_kernel_"
    "wp10c9d6c7c3b5c4f25fizzl1.py"
)
PHYSICAL_SOURCE = (
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_physical_entropy_congruence.py"
)
PHYSICAL_TEST = "tests/test_causal_inner_physical_entropy_congruence.py"
PARENT_SHA256 = "e61637312b8daa877463c7d3b3257aa1a9e5e3ec9f1c0422ce7349dc5c850c4f"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
SELECTED_WITNESSES = (0, 10, 20, 30, 40, 46)


def _u():
    return manifest._u()


def _validate_parent(require_clean=False):
    utility = _u()
    if utility._sha256(manifest.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != PARENT_SHA256:
        raise RuntimeError("physical congruence manifest checksum changed")
    hashes = utility._validate_checksums(manifest.CANONICAL_DIRECTORY)
    summary = utility._read_json(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = utility._read_json(manifest.CANONICAL_DIRECTORY / "architecture_contract.json")
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["physical_entropy_congruence_certified"]
        or summary["AP_macrostep_certified"]
        or contract["claim_boundary"]["complete_cycle_execution_authorized"]
    ):
        raise RuntimeError("physical congruence manifest classification changed")
    if require_clean and utility._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("physical congruence kernel needs a clean tracked tree")
    return hashes, contract


def _certificate():
    began = time.perf_counter(); _, contract = _validate_parent()
    physical = {index: (label, radius, old, chart) for index, label, radius, old, chart in witnesses._physical_witnesses() if index in SELECTED_WITNESSES}
    rows = []; congruence_arrays = []; port_spectra = []
    for index in SELECTED_WITNESSES:
        label, radius, old, chart = physical[index]
        height = float(np.exp(chart[5])); sigma = float(np.exp(chart[0])); density = sigma / (2.0 * height); temperature = float(np.exp(chart[3])); radial = float(chart[1]); azimuthal = float(chart[2])
        congruence = build_physical_entropy_congruence(old.geometry, proper_half_thickness=height, density=density, temperature=temperature, radial_velocity_over_c=radial, azimuthal_velocity_over_c=azimuthal, primitive_step=contract["physical_congruence"]["primitive_step"])
        congruence_audit = audit_physical_entropy_congruence(congruence)
        omega = float(np.sqrt(old.thermodynamics.integrated_pressure / (sigma * height**2)))
        old_sound = float(old.thermodynamics.sound_speed)
        alpha = float((old.closure.viscous_signal_speed_over_c * C / old_sound) ** 2)
        anchor = build_full_port_atlas_anchor(sound_speed=congruence.sound_speed_over_c * C, temperature=temperature, proper_half_thickness=height, proper_vertical_frequency=omega, alpha=alpha, shear_relaxation_time=float(old.closure.relaxation_time), transport_speed_over_c=radial)
        corrected = build_corrected_physical_port_atlas(anchor, congruence, old.geometry)
        port_audit = audit_corrected_physical_port_atlas(corrected, anchor, congruence, old.geometry)
        ap_audit = audit_ap_fast_propagator(corrected, step_ratios=contract["AP_macrostep"]["kernel_step_ratios"])
        old_core_speeds = (radial + np.asarray((-anchor.sound_speed_over_c, 0.0, 0.0, anchor.sound_speed_over_c))) / (1.0 + radial * np.asarray((-anchor.sound_speed_over_c, 0.0, 0.0, anchor.sound_speed_over_c)))
        old_coordinate_fixture_defect = float(np.max(np.abs(np.sort(old_core_speeds) - congruence.analytic_speeds_over_c)))
        passed = bool(congruence_audit.passed and port_audit.passed and ap_audit.passed)
        rows.append({"witness_index": index, "witness_label": label, "radius_cm": radius, "congruence_audit": asdict(congruence_audit), "corrected_port_audit": asdict(port_audit), "AP_audit": asdict(ap_audit), "old_special_relativistic_coordinate_fixture_spectrum_defect": old_coordinate_fixture_defect, "passed": passed})
        congruence_arrays.append(np.stack((congruence.numerical_speeds_over_c, congruence.analytic_speeds_over_c)))
        port_spectra.append(np.stack((corrected.mapped_rest_speeds_over_c, corrected.coordinate_speeds_over_c)))
    passed = bool(len(rows) == 6 and all(row["passed"] for row in rows))
    metrics = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,
        "passed": passed,
        "physical_witness_count": len(rows),
        "passing_witness_count": sum(row["passed"] for row in rows),
        "minimum_scaled_entropy_metric_eigenvalue_ratio": float(min(row["congruence_audit"]["scaled_entropy_minimum_eigenvalue_ratio"] for row in rows)),
        "maximum_whitened_symmetry_relative_defect": float(max(row["congruence_audit"]["whitened_symmetry_relative_defect"] for row in rows)),
        "maximum_Valencia_spectrum_absolute_defect": float(max(row["congruence_audit"]["valencia_spectrum_absolute_defect"] for row in rows)),
        "maximum_core_reconstruction_relative_defect": float(max(row["congruence_audit"]["core_reconstruction_relative_defect"] for row in rows)),
        "maximum_corrected_port_symmetry_defect": float(max(row["corrected_port_audit"]["radial_symmetry_defect"] for row in rows)),
        "maximum_corrected_port_light_cone_violation": float(max(max(row["corrected_port_audit"]["lower_light_cone_violation"], row["corrected_port_audit"]["upper_light_cone_violation"]) for row in rows)),
        "maximum_AP_semigroup_expansivity": float(max(row["AP_audit"]["maximum_semigroup_expansivity"] for row in rows)),
        "maximum_AP_composition_defect": float(max(row["AP_audit"]["maximum_composition_defect"] for row in rows)),
        "maximum_AP_stiff_limit_defect": float(max(row["AP_audit"]["stiff_limit_defect"] for row in rows)),
        "minimum_stable_spectral_gap": float(min(row["AP_audit"]["stable_spectral_gap"] for row in rows)),
        "minimum_old_coordinate_fixture_spectrum_defect": float(min(row["old_special_relativistic_coordinate_fixture_spectrum_defect"] for row in rows)),
        "old_algebraic_kernel_preserved_as_rest_fixture": True,
        "bounded_AP_trajectory_steps": 0,
        "complete_cycle_execution_authorized": False,
        "certificate_wall_seconds": time.perf_counter() - began,
        "rows": rows,
        "authorized_next": AUTHORIZED_NEXT if passed else None,
    }
    arrays = {"core_spectra": np.asarray(congruence_arrays), "full_port_spectra": np.asarray(port_spectra), "selected_witness_indices": np.asarray(SELECTED_WITNESSES)}
    return metrics, arrays


def _update_catalog(summary):
    utility = _u(); rows = list(csv.DictReader(CANONICAL_MANIFEST.open(newline="", encoding="utf-8"))); rows = [row for row in rows if row.get("case") != ARTIFACT]; status = "SUPPORTED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file(): rows.append({"case": ARTIFACT, "path": str(path.relative_to(ROOT)), "bytes": str(path.stat().st_size), "sha256": utility._sha256(path), "scientific_status": status})
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    catalog = utility._read_json(CANONICAL_SUMMARY); catalog.setdefault("artifacts", {})[ARTIFACT] = {"path": str(CANONICAL_DIRECTORY.relative_to(ROOT)), "classification": summary["classification"], "passed": summary["passed"]}; catalog.update({"case_count": len({row["case"] for row in rows}), "file_count": len(rows), "total_bytes": sum(int(row["bytes"]) for row in rows), "all_payload_hashes_recorded": True, "latest_source_parent_commit": utility._git("rev-parse", "HEAD"), "latest_work_package": WORK_PACKAGE}); utility._write_json(CANONICAL_SUMMARY, catalog)


def _canonicalize(metrics, arrays):
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists(): raise RuntimeError("congruence/AP certificate exists")
    hashes, _ = _validate_parent(require_clean=True); utility = _u(); CANONICAL_DIRECTORY.mkdir(parents=True)
    utility._write_json(CANONICAL_DIRECTORY / "kernel_metrics.json", metrics); np.savez_compressed(CANONICAL_DIRECTORY / "kernel_arrays.npz", **arrays)
    summary = {"schema_version": 1, "work_package": WORK_PACKAGE, "classification": metrics["classification"], "passed": metrics["passed"], "physical_entropy_congruence_certified": metrics["passed"], "corrected_Kerr_Schild_eleven_field_port_certified": metrics["passed"], "AP_macrostep_kernel_certified": metrics["passed"], "old_algebraic_kernel_preserved_as_rest_fixture": True, "bounded_AP_coarse_trajectory_manifest_authorized": metrics["passed"], "complete_cycle_execution_authorized": False, "authorized_next": metrics["authorized_next"]}
    utility._write_json(CANONICAL_DIRECTORY / "summary.json", summary); utility._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"manifest_artifact": manifest.ARTIFACT, "manifest_checksum_manifest_sha256": PARENT_SHA256, "manifest_hashes": hashes})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True); REPORT_PATH.write_text(f"# Physical entropy-congruence and AP kernel certificate\n\nClassification: `{metrics['classification']}`.\n\n{metrics['passing_witness_count']}/{metrics['physical_witness_count']} physical witnesses pass. The maximum physical-core/Valencia spectral defect is `{metrics['maximum_Valencia_spectrum_absolute_defect']:.6e}` and the maximum AP composition defect is `{metrics['maximum_AP_composition_defect']:.6e}`. The old special-relativistic coordinate fixture differs from the physical Kerr-Schild spectrum by at least `{metrics['minimum_old_coordinate_fixture_spectrum_defect']:.6e}` and is retained only as a rest-frame algebraic certificate.\n\nThis certifies the local physical congruence and stiff fast propagator. No bounded AP trajectory or complete cycle was executed.\n", encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, PHYSICAL_SOURCE, PHYSICAL_TEST, REPORT_RELATIVE); utility._write_json(CANONICAL_DIRECTORY / "provenance.json", {"implementation_commit": utility._git("rev-parse", "HEAD"), "source_hashes": {source: utility._sha256(ROOT / source) for source in sources}, "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "thread_environment": {name: os.environ.get(name, "") for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()); (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utility._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8"); _update_catalog(summary); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run", action="store_true"); arguments = parser.parse_args()
    if not arguments.run: parser.error("choose --run")
    metrics, arrays = _certificate(); print(json.dumps(metrics, indent=2, sort_keys=True), flush=True); return 0 if _canonicalize(metrics, arrays)["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
