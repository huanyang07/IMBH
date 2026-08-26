#!/usr/bin/env python3
"""Diagnose the 179.5 ms face-hyperbolicity boundary without propagation."""

from __future__ import annotations

import argparse
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

import run_causal_inner_entropy_complete_hyperbolicity_boundary_refinement_manifest_wp10c9d6c7c3b5c4f25fizv as manifest  # noqa: E402
from imri_qpe.layer3_minidisk_1d import causal_inner_generalized_maxwell_cattaneo_radial as radial  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo import (  # noqa: E402
    generalized_maxwell_cattaneo_principal,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_macro_integrator import (  # noqa: E402
    reconstruct_thermodynamic_macro_state,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_thermodynamic_macro_atlas import (  # noqa: E402
    thermodynamic_macro_chart_pullback,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_geometry import (  # noqa: E402
    kerr_schild_column_geometry,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = manifest.AUTHORIZED_NEXT
OVERSHOOT_CLASSIFICATION = (
    "entropy_complete_explicit_AB2_face_hyperbolicity_overshoot_confirmed"
)
BOUNDARY_CLASSIFICATION = (
    "entropy_complete_refined_candidate_hyperbolicity_failure_persists"
)
METHOD_CLASSIFICATION = (
    "entropy_complete_saved_face_hyperbolicity_failure_not_reproduced"
)
AUTHORIZED_NEXT = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizx_"
    "entropy_complete_event_aware_hyperbolicity_retry_recovery_manifest"
)
ARTIFACT = (
    "causal_inner_entropy_complete_hyperbolicity_boundary_refinement_"
    "diagnostic_wp10c9d6c7c3b5c4f25fizw"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_"
    "HYPERBOLICITY_BOUNDARY_REFINEMENT_DIAGNOSTIC_"
    "WP10C9D6C7C3B5C4F25FIZW_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_entropy_complete_hyperbolicity_boundary_"
    "refinement_diagnostic_wp10c9d6c7c3b5c4f25fizw.py"
)
THIS_TEST = (
    "tests/test_causal_inner_entropy_complete_hyperbolicity_boundary_"
    "refinement_diagnostic_wp10c9d6c7c3b5c4f25fizw.py"
)
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "b40c070f44f0faa31f7456bab286c7ef984500cac8a3e2fd93a3d171b0e92409"
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
        raise RuntimeError("boundary refinement manifest checksum changed")
    hashes = utils._validate_checksums(manifest.CANONICAL_DIRECTORY)
    summary = utils._read_json(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = utils._read_json(
        manifest.CANONICAL_DIRECTORY / "diagnostic_contract.json"
    )
    if (
        summary["classification"] != manifest.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or not summary["parent_rejection_preserved"]
        or not summary["nonpropagating_refinement_diagnostic_authorized"]
        or summary["complete_cycle_execution_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["authorized_next"] != WORK_PACKAGE
        or not contract["diagnostic_scope"]["nonpropagating"]
        or contract["diagnostic_scope"]["maximum_new_truth_operator_calls"] != 3
        or contract["diagnostic_scope"]["probe_timestep_seconds"]
        != [5.0e-4, 2.5e-4, 1.25e-4]
    ):
        raise RuntimeError("boundary refinement authorization changed")
    for relative, expected in utils._read_json(
        manifest.CANONICAL_DIRECTORY / "provenance.json"
    )["source_hashes"].items():
        if utils._sha256(ROOT / relative) != expected:
            raise RuntimeError(f"boundary refinement source changed: {relative}")
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("boundary refinement diagnostic needs a clean tracked tree")
    return {"hashes": hashes, "summary": summary, "contract": contract}


def _imaginary_ratio(values: np.ndarray) -> float:
    candidate = np.asarray(values)
    return float(
        np.max(np.abs(np.imag(candidate)))
        / max(float(np.max(np.abs(candidate))), 1.0)
    )


def _face_principal_audit(context, charts: np.ndarray) -> tuple[dict, dict]:
    primitive = np.asarray(charts, dtype=float)
    exterior = radial._outer_chart(context, primitive)
    rows = []
    eigenvalues = []
    for face, radius in enumerate(np.asarray(context.grid.edges, dtype=float)):
        if face == 0:
            chart = primitive[0]
        elif face == primitive.shape[0]:
            chart = 0.5 * (primitive[-1] + exterior)
        else:
            chart = 0.5 * (primitive[face - 1] + primitive[face])
        principal = generalized_maxwell_cattaneo_principal(
            kerr_schild_column_geometry(
                float(radius), context.grid.gravitational_radius
            ),
            chart,
            proper_vertical_frequency=float(
                context.vertical_frequency.frequency(float(radius))
            ),
            alpha=float(context.alpha),
            stress_factor=float(context.stress_factor),
        )
        values = np.asarray(principal.eigenvalues_over_c)
        ratio = _imaginary_ratio(values)
        rows.append(
            {
                "face": face,
                "radius": float(radius),
                "maximum_imaginary_ratio": ratio,
                "maximum_imaginary_speed_over_c": float(
                    np.max(np.abs(np.imag(values)))
                ),
                "maximum_real_speed_over_c": float(
                    np.max(np.abs(np.real(values)))
                ),
                "eigenvector_condition_number": float(
                    principal.eigenvector_condition_number
                ),
            }
        )
        eigenvalues.append(values)
    maximum = max(rows, key=lambda item: item["maximum_imaginary_ratio"])
    return {
        "maximum_imaginary_ratio": maximum["maximum_imaginary_ratio"],
        "first_nonreal_face": next(
            (
                item["face"]
                for item in rows
                if item["maximum_imaginary_ratio"] > 1.0e-10
            ),
            None,
        ),
        "maximum_imaginary_face": maximum["face"],
        "face_rows": rows,
    }, {"eigenvalues": np.asarray(eigenvalues)}


def _truth_modules():
    fizu = manifest.parent
    fizq = fizu.parent.parent.parent.parent
    truth_execution = fizq.rejected_execution.truth_execution
    return truth_execution.truth_source, fizu


def _diagnose() -> tuple[dict, dict[str, np.ndarray]]:
    began = time.perf_counter()
    validated = _validate_manifest(require_clean=False)
    contract = validated["contract"]
    scope = contract["diagnostic_scope"]
    gates = contract["binding_gates"]
    truth_source, fizu = _truth_modules()
    with np.load(manifest.PARENT_ARRAYS) as archive:
        seed = {name: np.asarray(archive[name]) for name in archive.files}
    current_state = np.array(seed["accepted_macro_states"][-1], copy=True)
    current_charts = np.array(seed["accepted_primitive_charts"][-1], copy=True)
    current_rate = np.array(seed["accepted_macro_rates_per_second"][-1], copy=True)
    previous_rate = np.array(seed["accepted_macro_rates_per_second"][-2], copy=True)
    previous_timestep = float(seed["terminal_previous_timestep_seconds"][0])
    context_start = time.perf_counter()
    context, _profile, _initial = truth_source.fixed_q_implementation._primary_setup()
    context_seconds = time.perf_counter() - context_start
    anchor, scales, _tangents, pullbacks = thermodynamic_macro_chart_pullback(
        context, current_charts, derivative_step=1.0e-5
    )
    anchor_defect = fizu._relative(anchor - current_state, anchor, current_state)
    cache: dict[float, dict] = {}

    def candidate(timestep: float) -> dict:
        key = float(timestep)
        if key in cache:
            return cache[key]
        macro = fizu._variable_ab2_candidate(
            current_state,
            current_rate,
            previous_rate,
            key,
            previous_timestep,
        )
        reconstruction = reconstruct_thermodynamic_macro_state(
            context,
            current_charts,
            macro,
            anchor_macro_state=anchor,
            macro_coordinate_scales=scales,
            macro_coordinate_pullbacks=pullbacks,
            derivative_step=1.0e-5,
            maximum_newton_corrections=8,
            relative_tolerance=gates["maximum_macro_roundtrip_relative_defect"],
            maximum_chart_coordinate_infinity=gates[
                "maximum_reconstruction_chart_coordinate"
            ],
        )
        face, face_arrays = _face_principal_audit(
            context, reconstruction.primitive_charts
        )
        result = {
            "timestep_seconds": key,
            "macro_state": macro,
            "primitive_charts": np.asarray(reconstruction.primitive_charts),
            "chart_coordinates": np.asarray(reconstruction.chart_coordinates),
            "maximum_chart_coordinate": float(
                np.max(np.abs(reconstruction.chart_coordinates))
            ),
            "macro_roundtrip_relative_defect": reconstruction.maximum_macro_state_roundtrip_relative_defect,
            "reconstruction_newton_corrections": reconstruction.newton_corrections,
            "face_audit": face,
            "face_eigenvalues": face_arrays["eigenvalues"],
        }
        cache[key] = result
        return result

    scan_rows = []
    scan_eigenvalues = []
    for timestep in scope["scan_timestep_seconds"]:
        item = candidate(float(timestep))
        scan_rows.append(
            {
                "timestep_seconds": float(timestep),
                "maximum_chart_coordinate": item["maximum_chart_coordinate"],
                "maximum_face_imaginary_ratio": item["face_audit"][
                    "maximum_imaginary_ratio"
                ],
                "first_nonreal_face": item["face_audit"]["first_nonreal_face"],
                "maximum_imaginary_face": item["face_audit"][
                    "maximum_imaginary_face"
                ],
            }
        )
        scan_eigenvalues.append(item["face_eigenvalues"])

    physical_gates = truth_source.fixed_q_implementation.parent._contract()[
        "binding_physical_gates"
    ]
    probe_rows = []
    truth_calls = 0
    for timestep in scope["probe_timestep_seconds"]:
        item = candidate(float(timestep))
        truth_calls += 1
        try:
            operator = radial.generalized_maxwell_cattaneo_radial_operator(
                context, item["primitive_charts"], quadrature_order=8
            )
            physical = truth_source._operator_record(operator)
            checks = truth_source._physical_checks(physical, physical_gates)
            checks["hydrostatic_embedding"] = (
                fizu.parent.parent.parent.parent.rejected_execution._hydrostatic_embedding_defect(
                    context, item["primitive_charts"]
                )
                <= 1.0e-10
            )
            truth_passed = bool(all(checks.values()))
            exception = None
        except Exception as exc:
            physical = None
            checks = None
            truth_passed = False
            exception = f"{type(exc).__name__}: {exc}"
        probe_rows.append(
            {
                "timestep_seconds": float(timestep),
                "maximum_chart_coordinate": item["maximum_chart_coordinate"],
                "macro_roundtrip_relative_defect": item[
                    "macro_roundtrip_relative_defect"
                ],
                "face_audit": item["face_audit"],
                "truth_operator_passed": truth_passed,
                "truth_exception": exception,
                "physical": physical,
                "physical_checks": checks,
            }
        )

    coarse = probe_rows[0]
    refined = probe_rows[1:]
    coarse_reproduced = bool(
        not coarse["truth_operator_passed"]
        and coarse["face_audit"]["maximum_imaginary_ratio"] > 1.0e-10
        and coarse["truth_exception"] is not None
        and "not real within the declared tolerance" in coarse["truth_exception"]
    )
    endpoint_hyperbolic = bool(
        scan_rows[0]["maximum_face_imaginary_ratio"]
        <= gates["maximum_face_imaginary_ratio_for_hyperbolic_probe"]
    )
    refined_pass = bool(
        all(item["truth_operator_passed"] for item in refined)
        and all(
            item["face_audit"]["maximum_imaginary_ratio"]
            <= gates["maximum_face_imaginary_ratio_for_hyperbolic_probe"]
            for item in refined
        )
        and all(
            item["maximum_chart_coordinate"]
            <= gates["maximum_reconstruction_chart_coordinate"]
            and item["macro_roundtrip_relative_defect"]
            <= gates["maximum_macro_roundtrip_relative_defect"]
            for item in refined
        )
    )
    overshoot = coarse_reproduced and endpoint_hyperbolic and refined_pass
    if overshoot:
        classification = OVERSHOOT_CLASSIFICATION
        passed = True
        authorized_next = AUTHORIZED_NEXT
    elif coarse_reproduced:
        classification = BOUNDARY_CLASSIFICATION
        passed = False
        authorized_next = None
    else:
        classification = METHOD_CLASSIFICATION
        passed = False
        authorized_next = None
    real_scan = [
        item
        for item in scan_rows
        if item["maximum_face_imaginary_ratio"] <= 1.0e-10
    ]
    nonreal_scan = [
        item
        for item in scan_rows
        if item["maximum_face_imaginary_ratio"] > 1.0e-10
    ]
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "authorized_next": authorized_next,
        "nonpropagating": True,
        "parent_rejection_preserved": True,
        "new_truth_operator_calls": truth_calls,
        "new_global_roots": 0,
        "fixed_Q_reaction_calls": 0,
        "context_construction_wall_seconds": context_seconds,
        "diagnostic_wall_seconds": time.perf_counter() - began,
        "anchor_roundtrip_relative_defect": anchor_defect,
        "coarse_failure_reproduced": coarse_reproduced,
        "accepted_endpoint_hyperbolic": endpoint_hyperbolic,
        "both_refined_truth_probes_passed": refined_pass,
        "probe_rows": probe_rows,
        "scan_rows": scan_rows,
        "largest_scanned_hyperbolic_timestep_seconds": max(
            (item["timestep_seconds"] for item in real_scan), default=None
        ),
        "smallest_scanned_nonreal_timestep_seconds": min(
            (item["timestep_seconds"] for item in nonreal_scan), default=None
        ),
        "failed_probe_propagated": False,
        "refined_probe_propagated": False,
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    arrays: dict[str, np.ndarray] = {
        "accepted_179ms_macro_state": current_state,
        "accepted_179ms_primitive_charts": current_charts,
        "accepted_179ms_current_macro_rate_per_second": current_rate,
        "accepted_179ms_previous_macro_rate_per_second": previous_rate,
        "scan_timestep_seconds": np.asarray(scope["scan_timestep_seconds"]),
        "scan_face_eigenvalues": np.asarray(scan_eigenvalues),
    }
    for index, timestep in enumerate(scope["probe_timestep_seconds"]):
        item = candidate(float(timestep))
        arrays[f"probe_{index}_macro_state"] = item["macro_state"]
        arrays[f"probe_{index}_primitive_charts"] = item["primitive_charts"]
        arrays[f"probe_{index}_face_eigenvalues"] = item["face_eigenvalues"]
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
        raise RuntimeError("boundary refinement diagnostic already exists")
    validated = _validate_manifest(require_clean=True)
    utils = _utils()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "diagnostic_metrics.json", metrics)
    np.savez_compressed(CANONICAL_DIRECTORY / "diagnostic_arrays.npz", **arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": bool(metrics["passed"]),
        "parent_rejection_preserved": True,
        "coarse_failure_reproduced": metrics["coarse_failure_reproduced"],
        "both_refined_truth_probes_passed": metrics[
            "both_refined_truth_probes_passed"
        ],
        "all_probes_nonpropagating": True,
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": metrics["authorized_next"],
    }
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utils._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_artifact": manifest.ARTIFACT,
            "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256,
            "parent_hashes": validated["hashes"],
            "trajectory_arrays_sha256": utils._sha256(manifest.PARENT_ARRAYS),
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Entropy-complete hyperbolicity-boundary refinement diagnostic",
                "",
                f"Classification: `{metrics['classification']}`.",
                "",
                f"The saved 0.5 ms failure reproduced: `{metrics['coarse_failure_reproduced']}`. Both 0.25 and 0.125 ms full-truth probes passed: `{metrics['both_refined_truth_probes_passed']}`.",
                "",
                f"The scan bracketed the boundary between `{metrics['largest_scanned_hyperbolic_timestep_seconds']}` and `{metrics['smallest_scanned_nonreal_timestep_seconds']}` s from the accepted 179 ms endpoint.",
                "",
                "Every probe was nonpropagating. The parent rejection remains binding and complete-cycle execution remains unauthorized.",
                "",
                f"Authorized next: `{metrics['authorized_next']}`.",
                "",
            )
        ),
        encoding="utf-8",
    )
    sources = (THIS_RUNNER, THIS_TEST, REPORT_RELATIVE)
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
    metrics, arrays = _diagnose()
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    summary = _canonicalize(metrics, arrays)
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
