#!/usr/bin/env python3
"""Certify the eleven-field STF basis and quadratic convex normal form."""

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

import run_causal_inner_eleven_field_convex_divergence_architecture_manifest_wp10c9d6c7c3b5c4f25fizz as manifest  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_eleven_field_convex import (  # noqa: E402
    N_ELEVEN_FIELDS,
    N_FULL_SHEAR_AMPLITUDES,
    audit_eleven_field_convex_normal_form,
    audit_full_shear_rest_frame,
    build_eleven_field_convex_normal_form,
    full_shear_rest_frame,
    one_Rphi_amplitude_embedding,
    reconstruct_full_shear_tensor,
    reference_eleven_field_parameters,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_geometry import (  # noqa: E402
    ValenciaPerfectFluidPrimitive,
    kerr_schild_column_geometry,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_stress import (  # noqa: E402
    causal_stress_column_state,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = manifest.AUTHORIZED_NEXT
PASS_CLASSIFICATION = "eleven_field_stf_basis_and_convex_normal_form_certified"
FAIL_CLASSIFICATION = "eleven_field_stf_or_convex_normal_form_failed"
AUTHORIZED_NEXT = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizzb_"
    "eleven_field_nonlinear_master_potential_derivation_manifest"
)
ARTIFACT = (
    "causal_inner_eleven_field_stf_convex_normal_form_implementation_"
    "wp10c9d6c7c3b5c4f25fizza"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ELEVEN_FIELD_STF_CONVEX_"
    "NORMAL_FORM_WP10C9D6C7C3B5C4F25FIZZA_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_eleven_field_stf_convex_normal_form_"
    "implementation_wp10c9d6c7c3b5c4f25fizza.py"
)
THIS_TEST = (
    "tests/test_causal_inner_eleven_field_stf_convex_normal_form_"
    "implementation_wp10c9d6c7c3b5c4f25fizza.py"
)
STRUCTURAL_SOURCE = (
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_eleven_field_convex.py"
)
STRUCTURAL_TEST = "tests/test_causal_inner_eleven_field_convex.py"
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "1a66e1c25a3b53688e33d176e410b0eac14d2b2e45e06bffcfc981ed1f64ffc5"
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _utils():
    return manifest._utils()


def _validate_manifest(*, require_clean: bool) -> dict:
    utils = _utils()
    if (
        utils._sha256(manifest.CANONICAL_DIRECTORY / "SHA256SUMS.txt")
        != PARENT_CHECKSUM_MANIFEST_SHA256
    ):
        raise RuntimeError("eleven-field architecture manifest checksum changed")
    hashes = utils._validate_checksums(manifest.CANONICAL_DIRECTORY)
    summary = utils._read_json(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = utils._read_json(
        manifest.CANONICAL_DIRECTORY / "architecture_contract.json"
    )
    if (
        summary["classification"] != manifest.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or not summary["seven_field_rejection_preserved"]
        or not summary["eleven_field_architecture_selected"]
        or summary["eleven_field_physical_closure_certified"]
        or summary["complete_cycle_execution_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["state_architecture"]["dimension"] != N_ELEVEN_FIELDS
        or not contract["state_architecture"]["one_Rphi_projection_forbidden"]
        or not contract["covariant_shear_representation"][
            "moving_basis_derivatives_included"
        ]
    ):
        raise RuntimeError("eleven-field architecture contract changed")
    for relative, expected in utils._read_json(
        manifest.CANONICAL_DIRECTORY / "provenance.json"
    )["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"architecture source changed: {relative}")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("STF normal-form certificate needs a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "contract": contract}


def _five_point_gradient(function, point: np.ndarray, step: float) -> np.ndarray:
    gradient = np.empty_like(point)
    for index in range(point.size):
        direction = np.zeros_like(point)
        direction[index] = step
        gradient[index] = (
            function(point - 2.0 * direction)
            - 8.0 * function(point - direction)
            + 8.0 * function(point + direction)
            - function(point + 2.0 * direction)
        ) / (12.0 * step)
    return gradient


def _basis_derivative_audit(geometry, chart: np.ndarray) -> tuple[dict, np.ndarray]:
    velocities = np.asarray((chart[1], chart[2], chart[6]), dtype=float)

    def basis_at(values):
        return full_shear_rest_frame(
            geometry,
            radial_velocity_over_c=float(values[0]),
            azimuthal_velocity_over_c=float(values[1]),
            vertical_velocity_over_c=float(values[2]),
        ).stf_basis

    derivatives = []
    half_derivatives = []
    for index in range(3):
        direction = np.zeros(3)
        direction[index] = 1.0e-6
        derivatives.append(
            (basis_at(velocities + direction) - basis_at(velocities - direction))
            / (2.0e-6)
        )
        half = 0.5 * direction
        half_derivatives.append(
            (basis_at(velocities + half) - basis_at(velocities - half)) / 1.0e-6
        )
    derivatives = np.asarray(derivatives)
    half_derivatives = np.asarray(half_derivatives)
    norms = np.linalg.norm(half_derivatives.reshape(3, -1), axis=1)
    defects = np.linalg.norm(
        (derivatives - half_derivatives).reshape(3, -1), axis=1
    ) / np.maximum(norms, np.finfo(float).tiny)
    return (
        {
            "derivative_norms": norms.tolist(),
            "step_halving_relative_defects": defects.tolist(),
            "minimum_derivative_norm": float(np.min(norms)),
            "maximum_step_halving_relative_defect": float(np.max(defects)),
            "basis_is_state_dependent": bool(np.min(norms) > 1.0e-6),
            "derivative_stable": bool(np.max(defects) <= 2.0e-7),
        },
        half_derivatives,
    )


def _certificate() -> tuple[dict, dict[str, np.ndarray]]:
    began = time.perf_counter()
    _validate_manifest(require_clean=False)
    blocker = manifest.parent
    boundary_diagnostic = blocker.manifest.parent
    truth_source, _adaptive = boundary_diagnostic._truth_modules()
    context_start = time.perf_counter()
    context, _profile, _initial = truth_source.fixed_q_implementation._primary_setup()
    context_seconds = time.perf_counter() - context_start
    with np.load(blocker.CANONICAL_DIRECTORY / "certificate_arrays.npz") as archive:
        chart = np.asarray(archive["witness_chart7"], dtype=float)
    metrics_parent = _utils()._read_json(
        blocker.CANONICAL_DIRECTORY / "certificate_metrics.json"
    )
    cell = int(metrics_parent["witness_cell"])
    radius = float(context.grid.centers[cell])
    geometry = kerr_schild_column_geometry(
        radius, context.grid.gravitational_radius
    )
    frame = full_shear_rest_frame(
        geometry,
        radial_velocity_over_c=float(chart[1]),
        azimuthal_velocity_over_c=float(chart[2]),
        vertical_velocity_over_c=float(chart[6]),
    )
    frame_audit = audit_full_shear_rest_frame(
        frame, old_specific_stress=float(chart[4])
    )
    basis_derivatives, derivative_arrays = _basis_derivative_audit(
        geometry, chart
    )

    sigma = float(np.exp(chart[0]))
    primitive = ValenciaPerfectFluidPrimitive(
        surface_density=sigma,
        radial_velocity_over_c=float(chart[1]),
        azimuthal_velocity_over_c=float(chart[2]),
        specific_internal_energy=0.0,
        integrated_pressure=0.0,
    )
    old = causal_stress_column_state(
        geometry, primitive, specific_stress=float(chart[4])
    ).viscous_stress_tensor
    reconstructed4 = reconstruct_full_shear_tensor(
        frame,
        one_Rphi_amplitude_embedding(float(chart[4])),
        stress_scale=sigma,
    )
    old_embedding_defect = float(
        np.linalg.norm(reconstructed4[:3, :3] - old)
        / max(float(np.linalg.norm(old)), np.finfo(float).tiny)
    )

    parameters = reference_eleven_field_parameters()
    form = build_eleven_field_convex_normal_form(parameters)
    normal_audit = audit_eleven_field_convex_normal_form(parameters)
    point = np.linspace(-0.17, 0.19, N_ELEVEN_FIELDS)
    state_fd = _five_point_gradient(form.state_potential, point, 2.0e-4)
    radial_fd = _five_point_gradient(form.radial_flux_potential, point, 2.0e-4)
    state_gradient_defect = float(
        np.linalg.norm(state_fd - form.state_current(point))
        / max(float(np.linalg.norm(form.state_current(point))), 1.0)
    )
    radial_gradient_defect = float(
        np.linalg.norm(radial_fd - form.radial_current(point))
        / max(float(np.linalg.norm(form.radial_current(point))), 1.0)
    )
    passed = bool(
        frame_audit.passed
        and basis_derivatives["basis_is_state_dependent"]
        and basis_derivatives["derivative_stable"]
        and old_embedding_defect <= 2.0e-13
        and normal_audit.passed
        and state_gradient_defect <= 2.0e-11
        and radial_gradient_defect <= 2.0e-11
    )
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,
        "passed": passed,
        "witness_cell": cell,
        "witness_radius_cm": radius,
        "witness_chart7": chart.tolist(),
        "field_count": N_ELEVEN_FIELDS,
        "full_shear_amplitude_count": N_FULL_SHEAR_AMPLITUDES,
        "frame_audit": asdict(frame_audit),
        "basis_derivative_audit": basis_derivatives,
        "old_Rphi_tensor_embedding_relative_defect": old_embedding_defect,
        "normal_form_audit": asdict(normal_audit),
        "independent_state_potential_gradient_relative_defect": state_gradient_defect,
        "independent_radial_potential_gradient_relative_defect": radial_gradient_defect,
        "fixture_is_physical_calibration": False,
        "nonlinear_physical_master_potential_derived": False,
        "eleven_field_physical_closure_certified": False,
        "eleven_field_trajectory_authorized": False,
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "context_construction_wall_seconds": context_seconds,
        "certificate_wall_seconds": time.perf_counter() - began,
        "authorized_next": AUTHORIZED_NEXT if passed else None,
    }
    arrays = {
        "witness_chart7": chart,
        "metric4": frame.metric,
        "four_velocity4": frame.four_velocity,
        "rest_triad3x4": frame.rest_triad,
        "stf_basis5x4x4": frame.stf_basis,
        "moving_basis_derivatives3x5x4x4": derivative_arrays,
        "old_projected_tensor3x3": old,
        "embedded_full_tensor4x4": reconstructed4,
        "temporal_hessian11x11": form.temporal_matrix,
        "radial_hessian11x11": form.radial_matrix,
        "source_matrix11x11": form.source_matrix,
        "vertical_equilibrium_embedding11x9": form.vertical_equilibrium_embedding,
        "full_characteristic_speeds": np.asarray(
            normal_audit.full_characteristic_speeds_over_c
        ),
        "reduced_characteristic_speeds": np.asarray(
            normal_audit.reduced_characteristic_speeds_over_c
        ),
    }
    return metrics, arrays


def _update_catalog(summary: dict) -> None:
    utils = _utils()
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    status = "SUPPORTED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": utils._sha256(path),
                    "scientific_status": status,
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "path", "bytes", "sha256", "scientific_status"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = utils._read_json(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": summary["passed"],
    }
    catalog.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": utils._git("rev-parse", "HEAD"),
            "latest_work_package": WORK_PACKAGE,
        }
    )
    utils._write_json(CANONICAL_SUMMARY, catalog)


def _canonicalize(metrics: dict, arrays: dict[str, np.ndarray]) -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("STF convex normal-form package already exists")
    validated = _validate_manifest(require_clean=True)
    utils = _utils()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "certificate_metrics.json", metrics)
    np.savez_compressed(CANONICAL_DIRECTORY / "certificate_arrays.npz", **arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": bool(metrics["passed"]),
        "seven_field_rejection_preserved": True,
        "five_STF_basis_certified": bool(metrics["passed"]),
        "quadratic_convex_normal_form_certified": bool(metrics["passed"]),
        "nonlinear_physical_master_potential_derived": False,
        "eleven_field_physical_closure_certified": False,
        "eleven_field_trajectory_authorized": False,
        "complete_cycle_execution_authorized": False,
        "authorized_next": metrics["authorized_next"],
    }
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utils._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_artifact": manifest.ARTIFACT,
            "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256,
            "parent_hashes": validated["hashes"],
            "blocker_artifact": manifest.parent.ARTIFACT,
            "blocker_checksum_manifest_sha256": utils._sha256(
                manifest.parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt"
            ),
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Eleven-field STF and convex-normal-form certificate",
                "",
                f"Classification: `{metrics['classification']}`.",
                "",
                f"The moving rest-frame STF basis passes all symmetry, trace, orthogonality, Gram, and amplitude-roundtrip gates. Its state dependence is resolved with maximum derivative step-halving defect `{metrics['basis_derivative_audit']['maximum_step_halving_relative_defect']:.6e}`.",
                "",
                f"The embedded one-amplitude R-phi tensor reproduces the prior unprojected stress tensor with relative defect `{metrics['old_Rphi_tensor_embedding_relative_defect']:.6e}`. This is compatibility on a subspace, not permission to project the new equations back to one amplitude.",
                "",
                f"The 11x11 quadratic common-potential fixture is symmetric hyperbolic and entropy dissipative; its maximum characteristic speed is `{metrics['normal_form_audit']['maximum_absolute_characteristic_speed_over_c']:.6e} c`. The fixture is not a physical calibration.",
                "",
                "The nonlinear gas+radiation Kerr-Schild master potential remains underived. No eleven-field trajectory or complete-cycle execution is authorized.",
                "",
                f"Authorized next: `{metrics['authorized_next']}`.",
                "",
            )
        ),
        encoding="utf-8",
    )
    sources = (
        THIS_RUNNER,
        THIS_TEST,
        STRUCTURAL_SOURCE,
        STRUCTURAL_TEST,
        REPORT_RELATIVE,
    )
    utils._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "implementation_commit": utils._git("rev-parse", "HEAD"),
            "implementation_tree": utils._git("rev-parse", "HEAD^{tree}"),
            "source_hashes": {
                path: utils._sha256(ROOT / path) for path in sources
            },
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "thread_environment": {
                name: os.environ.get(name, "")
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
        },
    )
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{utils._sha256(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
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
