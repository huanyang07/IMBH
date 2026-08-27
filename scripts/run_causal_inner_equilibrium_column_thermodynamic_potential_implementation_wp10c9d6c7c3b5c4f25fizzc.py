#!/usr/bin/env python3
"""Certify the exact fixed-height equilibrium column potential."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
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

import run_causal_inner_eleven_field_nonlinear_master_potential_derivation_manifest_wp10c9d6c7c3b5c4f25fizzb as parent  # noqa: E402
import run_causal_inner_entropy_complete_projected_local_structural_audit_wp10c9d6c7c3b5c4f25fizee as frozen_audit  # noqa: E402
from imri_qpe.constants import C  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_eleven_field_convex import (  # noqa: E402
    full_shear_rest_frame,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_equilibrium_potential import (  # noqa: E402
    analytic_potential_current_jacobian,
    audit_equilibrium_column_potential,
    complex_step_potential_current_jacobian,
    entropy_variables_from_primitive,
    equilibrium_column_potential_state,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_geometry import (  # noqa: E402
    _column_stress_energy,
    valencia_column_state,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = parent.AUTHORIZED_NEXT
PASS_CLASSIFICATION = "equilibrium_column_thermodynamic_potential_certified"
FAIL_CLASSIFICATION = "equilibrium_column_thermodynamic_potential_failed"
AUTHORIZED_NEXT = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizzd_"
    "dynamic_height_convex_legendre_manifest"
)
ARTIFACT = (
    "causal_inner_equilibrium_column_thermodynamic_potential_implementation_"
    "wp10c9d6c7c3b5c4f25fizzc"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_EQUILIBRIUM_COLUMN_"
    "THERMODYNAMIC_POTENTIAL_WP10C9D6C7C3B5C4F25FIZZC_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_equilibrium_column_thermodynamic_potential_"
    "implementation_wp10c9d6c7c3b5c4f25fizzc.py"
)
THIS_TEST = (
    "tests/test_causal_inner_equilibrium_column_thermodynamic_potential_"
    "implementation_wp10c9d6c7c3b5c4f25fizzc.py"
)
PHYSICAL_SOURCE = (
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_equilibrium_potential.py"
)
PHYSICAL_TEST = "tests/test_causal_inner_equilibrium_potential.py"
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "d2c883418043cf277bded7882fab6c6a2bb0c6b92a4c5e2f5366d01753eccfd6"
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _utils():
    return parent._utils()


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    if (
        utils._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt")
        != PARENT_CHECKSUM_MANIFEST_SHA256
    ):
        raise RuntimeError("master-potential derivation manifest checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    contract = utils._read_json(
        parent.CANONICAL_DIRECTORY / "derivation_contract.json"
    )
    if (
        summary["classification"] != parent.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or summary["nonlinear_physical_master_potential_derived"]
        or summary["complete_cycle_execution_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["next_package"]["scope"] != WORK_PACKAGE
        or not contract["next_package"]["fixed_height_only"]
        or contract["next_package"]["add_height_or_shear_terms"]
    ):
        raise RuntimeError("master-potential derivation contract changed")
    for relative, expected in utils._read_json(
        parent.CANONICAL_DIRECTORY / "provenance.json"
    )["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"master-potential source changed: {relative}")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("equilibrium potential certificate needs a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "contract": contract}


def _relative(actual, expected) -> float:
    actual_array = np.asarray(actual, dtype=np.longdouble)
    expected_array = np.asarray(expected, dtype=np.longdouble)
    scale = max(float(np.max(np.abs(expected_array))), np.finfo(float).tiny)
    return float(np.max(np.abs(actual_array - expected_array)) / scale)


def _physical_witnesses():
    stage2 = frozen_audit.parent.parent.parent.parent
    envelope_meta = _utils()._read_json(
        stage2.CANONICAL_DIRECTORY / "audit_envelope.json"
    )
    with np.load(
        stage2.CANONICAL_DIRECTORY / "audit_envelope.npz", allow_pickle=False
    ) as archive:
        envelope = {name: np.array(archive[name], copy=True) for name in archive.files}
    source = (
        frozen_audit.parent.parent.parent.boundary_diagnostic.manifest.parent
        .engine.execution.source
    )
    context = source._initial_inputs()["base"]["configuration"]["context"]
    centers = np.asarray(context.grid.centers, dtype=float)
    entries = frozen_audit._base_entries(
        envelope,
        centers=centers,
        failed_face_radius=float(envelope_meta["failed_face_radius_cm"]),
    )
    radius_by_chart: dict[bytes, float] = {}
    for _label, _segment, _cell, radius, chart in entries:
        radius_by_chart.setdefault(np.asarray(chart, dtype=float).tobytes(), radius)
    for index, (label, chart5) in enumerate(
        zip(envelope["witness_labels"], envelope["witness_charts5"], strict=True)
    ):
        chart = np.asarray(chart5, dtype=float)
        radius = radius_by_chart.get(chart.tobytes())
        if radius is None and str(label) == "failed_face_003":
            radius = float(envelope_meta["failed_face_radius_cm"])
        if radius is None:
            raise RuntimeError(f"cannot recover witness radius: {label}")
        old_state, chart7 = frozen_audit._chart7_at_equilibrium(
            context, radius, chart
        )
        yield index, str(label), radius, old_state, np.asarray(chart7, dtype=float)


def _certificate() -> tuple[dict, dict[str, np.ndarray]]:
    began = time.perf_counter()
    _validate_parent(require_clean=False)
    rows = []
    charts = []
    radii = []
    complex_defects = []
    finite_defects = []
    physical_defects = []
    first_law_defects = []
    worst_payload = None
    context_began = time.perf_counter()
    witnesses = list(_physical_witnesses())
    context_seconds = time.perf_counter() - context_began
    for index, label, radius, old_state, chart7 in witnesses:
        height = float(np.exp(chart7[5]))
        surface_density = float(np.exp(chart7[0]))
        density = surface_density / (2.0 * height)
        temperature = float(np.exp(chart7[3]))
        frame = full_shear_rest_frame(
            old_state.geometry,
            radial_velocity_over_c=float(chart7[1]),
            azimuthal_velocity_over_c=float(chart7[2]),
            vertical_velocity_over_c=0.0,
        )
        audit = audit_equilibrium_column_potential(
            frame.metric,
            frame.four_velocity,
            density=density,
            temperature=temperature,
            proper_half_thickness=height,
        )
        alpha, beta = entropy_variables_from_primitive(
            frame.metric,
            frame.four_velocity,
            density=density,
            temperature=temperature,
        )
        state = equilibrium_column_potential_state(
            frame.metric, alpha, beta, proper_half_thickness=height
        )
        valencia = valencia_column_state(
            old_state.geometry.base,
            surface_density=old_state.primitive.surface_density,
            radial_velocity_over_c=old_state.primitive.radial_velocity_over_c,
            azimuthal_velocity_over_c=old_state.primitive.azimuthal_velocity_over_c,
            specific_internal_energy=old_state.primitive.specific_internal_energy,
            integrated_pressure=old_state.primitive.integrated_pressure,
        )
        old_velocity, old_stress_mass, _spatial = _column_stress_energy(
            old_state.geometry, old_state.primitive, valencia
        )
        parity = max(
            _relative(state.four_velocity[:3], old_velocity),
            _relative(state.column_stress_energy[:3, :3] / C**2, old_stress_mass),
            abs(2.0 * height * state.pressure - old_state.thermodynamics.integrated_pressure)
            / max(abs(old_state.thermodynamics.integrated_pressure), 1.0),
            abs(state.specific_internal_energy - old_state.thermodynamics.specific_internal_energy)
            / max(abs(old_state.thermodynamics.specific_internal_energy), 1.0),
        )
        first_law = max(
            audit.first_law_density_relative_defect,
            audit.first_law_temperature_relative_defect,
            audit.gibbs_duhem_density_relative_defect,
            audit.gibbs_duhem_temperature_relative_defect,
        )
        row = {
            "index": index,
            "label": label,
            "radius_cm": radius,
            "chart7": chart7.tolist(),
            "physical_current_relative_defect": parity,
            "audit": asdict(audit),
        }
        rows.append(row)
        charts.append(chart7)
        radii.append(radius)
        complex_defects.append(audit.complex_step_current_jacobian_relative_defect)
        finite_defects.append(audit.finite_difference_current_jacobian_relative_defect)
        physical_defects.append(parity)
        first_law_defects.append(first_law)
        if worst_payload is None or parity > worst_payload[0]:
            analytic = analytic_potential_current_jacobian(state)
            complex_jacobian = complex_step_potential_current_jacobian(
                frame.metric, alpha, beta, proper_half_thickness=height
            )
            worst_payload = (parity, analytic, complex_jacobian, frame.metric)
    maximum_physical = float(max(physical_defects))
    maximum_first_law = float(max(first_law_defects))
    maximum_complex = float(max(complex_defects))
    maximum_finite = float(max(finite_defects))
    all_local_pass = all(row["audit"]["density_affinity_roundtrip_relative_defect"] <= 2.0e-9 for row in rows)
    passed = bool(
        len(rows) == 47
        and all_local_pass
        and maximum_physical <= 1.0e-10
        and maximum_first_law <= 1.0e-11
        and maximum_complex <= 1.0e-9
        and maximum_finite <= 2.0e-5
    )
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,
        "passed": passed,
        "physical_witness_count": len(rows),
        "maximum_physical_current_relative_defect": maximum_physical,
        "maximum_first_law_or_gibbs_duhem_relative_defect": maximum_first_law,
        "maximum_complex_step_current_jacobian_relative_defect": maximum_complex,
        "maximum_sixth_order_current_jacobian_relative_defect": maximum_finite,
        "all_density_affinity_roundtrips_passed": all_local_pass,
        "fixed_height_only": True,
        "exact_gas_radiation_EOS": True,
        "mass_and_stress_energy_generated_by_one_potential": passed,
        "dynamic_height_potential_certified": False,
        "full_shear_master_potential_certified": False,
        "eleven_field_local_closure_certified": False,
        "eleven_field_trajectory_authorized": False,
        "complete_cycle_execution_authorized": False,
        "context_construction_wall_seconds": context_seconds,
        "certificate_wall_seconds": time.perf_counter() - began,
        "rows": rows,
        "authorized_next": AUTHORIZED_NEXT if passed else None,
    }
    assert worst_payload is not None
    arrays = {
        "witness_charts7": np.asarray(charts),
        "witness_radii_cm": np.asarray(radii),
        "physical_current_relative_defects": np.asarray(physical_defects),
        "complex_step_relative_defects": np.asarray(complex_defects),
        "sixth_order_relative_defects": np.asarray(finite_defects),
        "first_law_relative_defects": np.asarray(first_law_defects),
        "worst_analytic_current_jacobian4x5": np.asarray(worst_payload[1]),
        "worst_complex_step_current_jacobian4x5": np.asarray(worst_payload[2]),
        "worst_metric4x4": np.asarray(worst_payload[3]),
    }
    return _plain(metrics), arrays


def _update_catalog(summary: dict) -> None:
    utils = _utils()
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
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
        raise RuntimeError("equilibrium potential package already exists")
    validated = _validate_parent(require_clean=True)
    utils = _utils()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "certificate_metrics.json", metrics)
    np.savez_compressed(CANONICAL_DIRECTORY / "certificate_arrays.npz", **arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": metrics["passed"],
        "seven_field_rejection_preserved": True,
        "equilibrium_physical_potential_certified": metrics["passed"],
        "dynamic_height_potential_certified": False,
        "full_shear_master_potential_certified": False,
        "eleven_field_local_closure_certified": False,
        "eleven_field_trajectory_authorized": False,
        "complete_cycle_execution_authorized": False,
        "authorized_next": metrics["authorized_next"],
    }
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utils._write_json(CANONICAL_DIRECTORY / "input_lock.json", {"parent_artifact": parent.ARTIFACT, "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256, "parent_hashes": validated["hashes"], "frozen_envelope_artifact": frozen_audit.parent.parent.parent.parent.ARTIFACT})
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(("# Fixed-height equilibrium column potential certificate", "", f"Classification: `{metrics['classification']}`.", "", f"Across {metrics['physical_witness_count']} frozen physical witnesses, the exact gas+radiation potential current generates the surface-mass current and full perfect-fluid column stress-energy. The maximum physical-current parity defect is `{metrics['maximum_physical_current_relative_defect']:.6e}`.", "", f"The maximum first-law/Gibbs-Duhem defect is `{metrics['maximum_first_law_or_gibbs_duhem_relative_defect']:.6e}` and the independent complex-step derivative defect is `{metrics['maximum_complex_step_current_jacobian_relative_defect']:.6e}`.", "", "This certificate is fixed-height only. It adds neither a height current nor a shear potential, and authorizes no trajectory or complete-cycle execution.", "", f"Authorized next: `{metrics['authorized_next']}`.", "")), encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, PHYSICAL_SOURCE, PHYSICAL_TEST, REPORT_RELATIVE)
    utils._write_json(CANONICAL_DIRECTORY / "provenance.json", {"schema_version": SCHEMA_VERSION, "work_package": WORK_PACKAGE, "implementation_commit": utils._git("rev-parse", "HEAD"), "implementation_tree": utils._git("rev-parse", "HEAD^{tree}"), "source_hashes": {path: utils._sha256(ROOT / path) for path in sources}, "python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "thread_environment": {name: os.environ.get(name, "") for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}})
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text("".join(f"{utils._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8")
    _update_catalog(summary)
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
