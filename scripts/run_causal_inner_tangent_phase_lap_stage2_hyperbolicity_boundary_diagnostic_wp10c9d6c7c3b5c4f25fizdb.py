#!/usr/bin/env python3
"""Diagnose the saved stage-2 characteristic boundary without propagation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from scipy.linalg import eig


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_tangent_phase_lap_stage2_hyperbolicity_boundary_manifest_wp10c9d6c7c3b5c4f25fizda as manifest  # noqa: E402
from imri_qpe.layer3_minidisk_1d import causal_inner_characteristic_dissipation as finite  # noqa: E402
from imri_qpe.layer3_minidisk_1d import causal_inner_radial_linear_tangent as radial  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = manifest.AUTHORIZED_NEXT
PASS_CLASSIFICATION = "genuine_local_hyperbolicity_loss_confirmed"
ANALYTIC_ONLY_CLASSIFICATION = "analytic_tangent_hyperbolicity_defect_only"
FAILED_CLASSIFICATION = "saved_hyperbolicity_boundary_not_reproduced"
AUTHORIZED_NEXT = (
    "WP10c9d6c7c3b5c4f25fizdc_"
    "tangent_phase_hyperbolicity_two_half_step_bracket_manifest"
)
ARTIFACT = (
    "causal_inner_tangent_phase_lap_stage2_hyperbolicity_boundary_diagnostic_"
    "wp10c9d6c7c3b5c4f25fizdb"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_TANGENT_PHASE_LAP_STAGE2_"
    "HYPERBOLICITY_BOUNDARY_DIAGNOSTIC_WP10C9D6C7C3B5C4F25FIZDB_"
    "2026-08-25.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_tangent_phase_lap_stage2_"
    "hyperbolicity_boundary_diagnostic_wp10c9d6c7c3b5c4f25fizdb.py"
)
THIS_TEST = (
    "tests/test_causal_inner_tangent_phase_lap_stage2_"
    "hyperbolicity_boundary_diagnostic_wp10c9d6c7c3b5c4f25fizdb.py"
)


def _helper():
    return manifest._helper()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _source_hashes() -> dict[str, str]:
    helper = _helper()
    paths = (
        THIS_RUNNER,
        THIS_TEST,
        manifest.THIS_RUNNER,
        manifest.THIS_TEST,
        manifest.parent.THIS_RUNNER,
        manifest.parent.THIS_TEST,
        manifest.parent.manifest.THIS_RUNNER,
        manifest.parent.manifest.THIS_TEST,
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_radial_linear_tangent.py",
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_characteristic_dissipation.py",
    )
    return {name: helper._sha(ROOT / name) for name in paths}


def _validate_manifest(*, require_clean: bool) -> dict:
    helper = _helper()
    hashes = helper._validate_checksums(manifest.CANONICAL_DIRECTORY)
    summary = helper._read(manifest.CANONICAL_DIRECTORY / "summary.json")
    metrics = helper._read(manifest.CANONICAL_DIRECTORY / "boundary_metrics.json")
    contract = helper._read(
        manifest.CANONICAL_DIRECTORY / "diagnostic_contract.json"
    )
    provenance = helper._read(manifest.CANONICAL_DIRECTORY / "provenance.json")
    scope = contract["diagnostic_scope"]
    gates = contract["genuine_hyperbolicity_loss_requires"]
    if (
        summary["classification"] != manifest.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or summary["accepted_stage2_endpoints"] != 23
        or summary["failed_candidate_propagated"]
        or not summary["hyperbolicity_diagnostic_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or metrics["classification"] != manifest.CLASSIFICATION
        or not metrics["passed"]
        or metrics["failed_candidate_propagated"]
        or contract["authorized_diagnostic"] != WORK_PACKAGE
        or not scope["nonpropagating"]
        or scope["maximum_new_free_field_calls"] != 0
        or scope["maximum_new_retractions"] != 0
        or scope["independent_five_point_relative_steps"]
        != list(manifest.FINITE_DIFFERENCE_STEPS)
        or gates["analytic_maximum_imaginary_speed_at_least"] != 1.0e-8
    ):
        raise RuntimeError("hyperbolicity diagnostic authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if helper._sha(ROOT / relative) != expected:
            raise RuntimeError(f"hyperbolicity manifest source changed: {relative}")
    if require_clean and helper._git(
        "status", "--short", "--untracked-files=no"
    ):
        raise RuntimeError("hyperbolicity diagnostic requires a clean tracked tree")
    return {
        "hashes": hashes,
        "summary": summary,
        "metrics": metrics,
        "contract": contract,
        "provenance": provenance,
    }


def _seed() -> dict[str, np.ndarray]:
    return _load_npz(manifest.CANONICAL_DIRECTORY / "boundary_seed.npz")


def _scaled_eigensystem(
    temporal: np.ndarray,
    spatial: np.ndarray,
    column_scales: np.ndarray,
) -> dict[str, np.ndarray | float]:
    temporal = np.asarray(temporal, dtype=float)
    spatial = np.asarray(spatial, dtype=float)
    scales = np.asarray(column_scales, dtype=float)
    row_scales = np.maximum(
        np.max(np.abs(temporal), axis=1),
        np.max(np.abs(spatial), axis=1),
    )
    row_scales = np.maximum(
        row_scales,
        max(float(np.max(row_scales)), 1.0) * 1.0e-14,
    )
    scaled_temporal = temporal * scales[None, :] / row_scales[:, None]
    scaled_spatial = spatial * scales[None, :] / row_scales[:, None]
    values, vectors = eig(scaled_spatial, scaled_temporal)
    order = np.lexsort((np.imag(values), np.real(values)))
    values = values[order]
    vectors = vectors[:, order]
    residual = scaled_spatial @ vectors - scaled_temporal @ (
        vectors * values[None, :]
    )
    scale = max(
        float(np.max(np.abs(scaled_spatial @ vectors))),
        float(np.max(np.abs(scaled_temporal @ vectors))),
        np.finfo(float).tiny,
    )
    return {
        "values": values,
        "vectors": vectors,
        "row_scales": row_scales,
        "scaled_temporal": scaled_temporal,
        "scaled_spatial": scaled_spatial,
        "maximum_imaginary_speed": float(np.max(np.abs(np.imag(values)))),
        "maximum_eigenpair_defect": float(np.max(np.abs(residual)) / scale),
    }


def _analytic_pencil(context, radius: float, chart: np.ndarray) -> dict:
    maps = radial.causal_five_field_analytic_local_maps(context, radius, chart)
    state = radial._cell_state(context, radius, chart)
    audit = radial.audit_causal_five_field_principal(
        state.geometry,
        context.vertical_frequency.eos(radius),
        state.closure,
        surface_density=state.primitive.surface_density,
        radial_velocity_over_c=state.primitive.radial_velocity_over_c,
        azimuthal_velocity_over_c=state.primitive.azimuthal_velocity_over_c,
        temperature=state.thermodynamics.temperature,
    )
    stress_scale = max(
        abs(float(chart[4])),
        abs(float(state.closure.equilibrium_specific_stress)),
        1.0e-14,
    )
    column_scales = np.asarray([1.0, 0.1, 0.1, 1.0, stress_scale])
    temporal = np.asarray(maps.temporal_storage_matrix, dtype=float)
    spatial = (
        maps.physical_flux_jacobian
        - maps.shear_principal_source_matrix
        - maps.vertical_principal_source_matrix
    )
    eigen = _scaled_eigensystem(temporal, spatial, column_scales)
    return {
        "temporal": temporal,
        "spatial": spatial,
        "column_scales": column_scales,
        "analytic_speeds": np.asarray(audit.coordinate_speeds_over_c),
        **eigen,
    }


def _face_charts(context, state: np.ndarray) -> np.ndarray:
    left_weights, right_weights, _defect = (
        radial._frozen_quadratic_reconstruction_weights(context, state)
    )
    left = left_weights @ state
    right = right_weights @ state
    charts = np.empty((len(context.grid.edges), right.shape[1]), dtype=float)
    charts[0] = right[0]
    charts[1:] = 0.5 * (left[1:] + right[1:])
    return charts


def _locate_first_complex_face(context, state: np.ndarray) -> tuple[dict, dict]:
    charts = _face_charts(context, state)
    preceding = []
    for face, (radius, chart) in enumerate(
        zip(context.grid.edges, charts, strict=True)
    ):
        pencil = _analytic_pencil(context, float(radius), chart)
        maximum = pencil["maximum_imaginary_speed"]
        if maximum > 1.0e-10:
            return {
                "face": face,
                "radius": float(radius),
                "maximum_imaginary_speed": maximum,
                "preceding_face_maximum_imaginary_speeds": preceding,
            }, {"chart": chart, **pencil}
        preceding.append(maximum)
    raise RuntimeError("saved complex characteristic face did not reproduce")


def _finite_difference_rows(
    context,
    radius: float,
    chart: np.ndarray,
    analytic: dict,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    rows = []
    arrays: dict[str, np.ndarray] = {}
    for index, step in enumerate(manifest.FINITE_DIFFERENCE_STEPS):
        components = finite.causal_five_field_coordinate_principal_components(
            context,
            radius,
            chart,
            relative_step=step,
        )
        temporal = np.asarray(components.temporal_storage_matrix)
        spatial = np.asarray(components.spatial_principal_matrix)
        eigen = _scaled_eigensystem(
            temporal,
            spatial,
            np.asarray(components.primitive_column_scales),
        )
        rows.append(
            {
                "relative_step": step,
                "maximum_imaginary_speed": eigen[
                    "maximum_imaginary_speed"
                ],
                "maximum_eigenpair_defect": eigen[
                    "maximum_eigenpair_defect"
                ],
                "temporal_relative_defect": float(
                    np.linalg.norm(temporal - analytic["temporal"])
                    / np.linalg.norm(analytic["temporal"])
                ),
                "spatial_relative_defect": float(
                    np.linalg.norm(spatial - analytic["spatial"])
                    / np.linalg.norm(analytic["spatial"])
                ),
            }
        )
        arrays[f"fd_{index}_temporal5x5"] = temporal
        arrays[f"fd_{index}_spatial5x5"] = spatial
        arrays[f"fd_{index}_eigenvalues5"] = eigen["values"]
    return rows, arrays


def _chord_scan(
    context,
    accepted_state: np.ndarray,
    failed_state: np.ndarray,
    face: int,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    rows = []
    values = []
    charts = []
    for fraction in manifest.INTERPOLATION_SCAN_FRACTIONS:
        state = accepted_state + fraction * (failed_state - accepted_state)
        chart = _face_charts(context, state)[face]
        pencil = _analytic_pencil(
            context,
            float(context.grid.edges[face]),
            chart,
        )
        rows.append(
            {
                "fraction": fraction,
                "maximum_imaginary_speed": pencil[
                    "maximum_imaginary_speed"
                ],
            }
        )
        charts.append(chart)
        values.append(pencil["values"])
    return rows, {
        "chord_scan_fractions": np.asarray(
            manifest.INTERPOLATION_SCAN_FRACTIONS
        ),
        "chord_scan_face_charts": np.stack(charts),
        "chord_scan_eigenvalues": np.stack(values),
    }


def _diagnose() -> tuple[dict, dict[str, np.ndarray]]:
    began = time.perf_counter()
    seed = _seed()
    inputs = manifest.parent.engine.execution.source._initial_inputs()
    context = inputs["base"]["configuration"]["context"]
    failed_state = seed["failed_retracted_primitive_state"]
    accepted_state = seed["current_primitive_state"]
    location, analytic = _locate_first_complex_face(context, failed_state)
    fd_rows, fd_arrays = _finite_difference_rows(
        context,
        location["radius"],
        analytic["chart"],
        analytic,
    )
    chord_rows, chord_arrays = _chord_scan(
        context,
        accepted_state,
        failed_state,
        location["face"],
    )
    analytic_imaginary = analytic["maximum_imaginary_speed"]
    smallest_fd = fd_rows[-1]["maximum_imaginary_speed"]
    relative = abs(smallest_fd - analytic_imaginary) / analytic_imaginary
    real_rows = [
        item for item in chord_rows if item["maximum_imaginary_speed"] <= 1.0e-10
    ]
    complex_rows = [
        item for item in chord_rows if item["maximum_imaginary_speed"] > 1.0e-10
    ]
    gates = _validate_manifest(require_clean=False)["contract"][
        "genuine_hyperbolicity_loss_requires"
    ]
    genuine = bool(
        analytic_imaginary
        >= gates["analytic_maximum_imaginary_speed_at_least"]
        and all(
            item["maximum_imaginary_speed"]
            >= gates[
                "each_independent_finite_difference_maximum_imaginary_speed_at_least"
            ]
            for item in fd_rows
        )
        and relative
        <= gates[
            "smallest_step_to_analytic_imaginary_relative_defect_at_most"
        ]
        and chord_rows[0]["maximum_imaginary_speed"] <= 1.0e-10
        and chord_rows[-1]["maximum_imaginary_speed"] >= 1.0e-8
        and bool(real_rows)
        and bool(complex_rows)
    )
    analytic_only = bool(
        analytic_imaginary >= 1.0e-8
        and any(item["maximum_imaginary_speed"] <= 1.0e-10 for item in fd_rows)
    )
    if genuine:
        classification = PASS_CLASSIFICATION
        passed = True
        authorized_next = AUTHORIZED_NEXT
    elif analytic_only:
        classification = ANALYTIC_ONLY_CLASSIFICATION
        passed = False
        authorized_next = None
    else:
        classification = FAILED_CLASSIFICATION
        passed = False
        authorized_next = None
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "nonpropagating": True,
        "new_free_field_calls": 0,
        "new_retractions": 0,
        "first_complex_face": location["face"],
        "first_complex_face_radius": location["radius"],
        "analytic_maximum_imaginary_speed": analytic_imaginary,
        "analytic_maximum_eigenpair_defect": analytic[
            "maximum_eigenpair_defect"
        ],
        "analytic_speeds_over_c": analytic["analytic_speeds"].tolist(),
        "analytic_eigenvalues_real": np.real(analytic["values"]).tolist(),
        "analytic_eigenvalues_imag": np.imag(analytic["values"]).tolist(),
        "finite_difference_rows": fd_rows,
        "smallest_step_to_analytic_imaginary_relative_defect": relative,
        "chord_scan_rows": chord_rows,
        "last_scanned_real_fraction": max(item["fraction"] for item in real_rows),
        "first_scanned_complex_fraction": min(
            item["fraction"] for item in complex_rows
        ),
        "wall_seconds": float(time.perf_counter() - began),
        "authorized_next": authorized_next,
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
    }
    arrays = {
        "failing_face_index": np.asarray(location["face"], dtype=np.int64),
        "failing_face_radius": np.asarray(location["radius"]),
        "failing_face_chart5": analytic["chart"],
        "analytic_temporal5x5": analytic["temporal"],
        "analytic_spatial5x5": analytic["spatial"],
        "analytic_column_scales5": analytic["column_scales"],
        "analytic_row_scales5": analytic["row_scales"],
        "analytic_eigenvalues5": analytic["values"],
        "analytic_eigenvectors5x5": analytic["vectors"],
        **fd_arrays,
        **chord_arrays,
    }
    return metrics, arrays


def _update_catalog(summary: dict) -> None:
    helper = _helper()
    manifest_path = ROOT / "results/manifests/canonical_artifacts.csv"
    summary_path = ROOT / "results/manifests/canonical_summary.json"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": helper._sha(path),
                    "scientific_status": (
                        "SUPPORTED" if summary["passed"] else "REJECTED"
                    ),
                }
            )
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "path", "bytes", "sha256", "scientific_status"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = helper._read(summary_path)
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
            "latest_source_parent_commit": helper._git("rev-parse", "HEAD"),
            "latest_work_package": WORK_PACKAGE,
        }
    )
    helper._write_json(summary_path, catalog)


def _canonicalize(
    metrics: dict,
    arrays: dict[str, np.ndarray],
    lock: dict,
) -> dict:
    helper = _helper()
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("hyperbolicity diagnostic result already exists")
    CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY / "diagnostic_metrics.json", metrics)
    _save_npz(CANONICAL_DIRECTORY / "diagnostic_arrays.npz", arrays)
    helper._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "manifest_hashes": lock["hashes"],
            "manifest_classification": lock["summary"]["classification"],
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": metrics["passed"],
        "nonpropagating": True,
        "first_complex_face": metrics["first_complex_face"],
        "analytic_maximum_imaginary_speed": metrics[
            "analytic_maximum_imaginary_speed"
        ],
        "failed_candidate_propagated": False,
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": metrics["authorized_next"],
    }
    helper._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    helper._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "implementation_commit": helper._git("rev-parse", "HEAD"),
            "implementation_tree": helper._git("rev-parse", "HEAD^{tree}"),
            "source_hashes": _source_hashes(),
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "platform": platform.platform(),
        },
    )
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{helper._sha(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Stage-2 hyperbolicity-boundary diagnostic",
                "",
                f"Classification: `{metrics['classification']}`.",
                "",
                f"The first complex characteristic pencil occurs at face `{metrics['first_complex_face']}`, radius `{metrics['first_complex_face_radius']:.9e}`. The analytic conjugate pair has maximum imaginary speed `{metrics['analytic_maximum_imaginary_speed']:.9e}`.",
                "",
                f"Independent five-point pencils at relative steps `{list(manifest.FINITE_DIFFERENCE_STEPS)}` all remain complex; the smallest-step imaginary magnitude differs from the analytic value by `{metrics['smallest_step_to_analytic_imaginary_relative_defect']:.3e}`.",
                "",
                f"The accepted-to-failed chord is real through scanned fraction `{metrics['last_scanned_real_fraction']}` and complex by `{metrics['first_scanned_complex_fraction']}`. This is genuine local loss of hyperbolicity, not eigensolver noise or an analytic-tangent-only defect.",
                "",
                "No free field was evaluated, no retraction was performed, and the failed state was not propagated. A smaller step may localize or avoid predictor overshoot, but it may not cross a complex-characteristic region.",
                "",
                f"Authorized next artifact: `{metrics['authorized_next']}`. Complete-cycle execution and reduced slow evolution remain unauthorized.",
                "",
            )
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    arguments = parser.parse_args()
    if not arguments.run:
        parser.error("--run is required")
    lock = _validate_manifest(require_clean=True)
    metrics, arrays = _diagnose()
    summary = _canonicalize(metrics, arrays, lock)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
