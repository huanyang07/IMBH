#!/usr/bin/env python3
"""Run the nonpropagating seven-field semidiscrete relaxation audit."""

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
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_entropy_complete_semidiscrete_relaxation_manifest_wp10c9d6c7c3b5c4f25fizeh as parent  # noqa: E402
import run_causal_inner_invariant_cluster_local_structural_audit_wp10c9d6c7c3b5c4f25fizee7 as local_parent  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo import (  # noqa: E402
    default_primitive_steps,
    generalized_maxwell_cattaneo_principal,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_semidiscrete import (  # noqa: E402
    generalized_maxwell_cattaneo_hydrostatic_embedding,
    generalized_maxwell_cattaneo_lower_source,
    generalized_maxwell_cattaneo_periodic_operator,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = parent.AUTHORIZED_NEXT
PASS_CLASSIFICATION = "entropy_complete_semidiscrete_relaxation_audit_passed"
SOURCE_FAILURE = "entropy_complete_semidiscrete_relaxation_source_failed"
OPERATOR_FAILURE = "entropy_complete_semidiscrete_relaxation_operator_failed"
LIMIT_FAILURE = "entropy_complete_semidiscrete_relaxation_limit_failed"
HYPERBOLICITY_FAILURE = "entropy_complete_semidiscrete_relaxation_hyperbolicity_failed"
AUTHORIZED_NEXT_ON_PASS = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizej_"
    "entropy_complete_bounded_radial_crossing_manifest"
)
ARTIFACT = (
    "causal_inner_entropy_complete_semidiscrete_relaxation_audit_"
    "wp10c9d6c7c3b5c4f25fizei"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_"
    "SEMIDISCRETE_RELAXATION_AUDIT_WP10C9D6C7C3B5C4F25FIZEI_"
    "2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_entropy_complete_semidiscrete_relaxation_"
    "audit_wp10c9d6c7c3b5c4f25fizei.py"
)
THIS_TEST = (
    "tests/test_causal_inner_entropy_complete_semidiscrete_relaxation_"
    "audit_wp10c9d6c7c3b5c4f25fizei.py"
)
SOURCE = parent.SOURCE
SOURCE_TEST = parent.SOURCE_TEST
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "86b701930c247651169e781ccc31adc5b56c7d629fb16cd3e861918ae513a290"
)
SOURCE_SHA256 = "1a78b7096c2a146b389d79d4e54d3a1e26c2a490aa07a4e3571bd984e3dfa587"
SOURCE_TEST_SHA256 = "19fabe0adb7ff11cc01543583d6d6113f543a850b451db983a59a1b3bb996143"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
FAST_MULTIPLIERS = (1.0, 2.0, 4.0, 8.0)


def _utils():
    return parent._utils()


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    if utils._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != PARENT_CHECKSUM_MANIFEST_SHA256:
        raise RuntimeError("semidiscrete manifest checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    contract = utils._read_json(parent.CANONICAL_DIRECTORY / "relaxation_contract.json")
    if (
        summary["classification"] != parent.CLASSIFICATION
        or not summary["passed"]
        or not summary["nonpropagating_relaxation_audit_authorized"]
        or summary["radial_boundary_implementation_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["authorized_next"] != WORK_PACKAGE
        or not contract["binding_gates"]["fail_closed"]
    ):
        raise RuntimeError("semidiscrete manifest authorization changed")
    if utils._sha256(ROOT / SOURCE) != SOURCE_SHA256:
        raise RuntimeError("semidiscrete source changed")
    if utils._sha256(ROOT / SOURCE_TEST) != SOURCE_TEST_SHA256:
        raise RuntimeError("semidiscrete source test changed")
    local_hashes = utils._validate_checksums(local_parent.CANONICAL_DIRECTORY)
    if require_clean and utils._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("semidiscrete audit requires clean tracked tree")
    return {"hashes": hashes, "summary": summary, "contract": contract, "local_hashes": local_hashes}


def _context_and_envelope():
    stage2 = local_parent.frozen_audit.parent.parent.parent.parent
    with np.load(stage2.CANONICAL_DIRECTORY / "audit_envelope.npz", allow_pickle=False) as archive:
        envelope = {name: np.array(archive[name], copy=True) for name in archive.files}
    meta = _utils()._read_json(stage2.CANONICAL_DIRECTORY / "audit_envelope.json")
    source = (
        local_parent.frozen_audit.parent.parent.parent.boundary_diagnostic
        .manifest.parent.engine.execution.source
    )
    context = source._initial_inputs()["base"]["configuration"]["context"]
    return context, envelope, meta


def _selected_cases(context, envelope: dict, meta: dict) -> list[dict]:
    centers = np.asarray(context.grid.centers, dtype=float)
    raw = (
        ("primary_20ms_cell036", float(centers[36]), envelope["primary_20ms_base_charts5"][36]),
        ("heldout_16ms_cell036", float(centers[36]), envelope["heldout_16ms_base_charts5"][36]),
        ("accepted00_cell044", float(centers[44]), envelope["accepted_trajectory_base_charts5"][0, 44]),
        ("accepted53_cell000", float(centers[0]), envelope["accepted_trajectory_base_charts5"][53, 0]),
        ("accepted_terminal_cell055", float(centers[55]), envelope["accepted_terminal_base_charts5"][55]),
        ("old_failed_face", float(meta["failed_face_radius_cm"]), envelope["failed_face_chart5"]),
    )
    return [
        {
            "label": label,
            "radius_cm": radius,
            "chart5": np.asarray(chart5, dtype=float),
            "chart7": generalized_maxwell_cattaneo_hydrostatic_embedding(
                chart5,
                proper_vertical_frequency=float(context.vertical_frequency.frequency(radius)),
            ),
        }
        for label, radius, chart5 in raw
    ]


def _geometry(context, radius: float, chart5: np.ndarray):
    return local_parent.frozen_audit.parent.parent.parent.boundary_diagnostic.radial._cell_state(
        context, float(radius), np.asarray(chart5, dtype=float)
    ).geometry


def _source_jacobian(function, chart: np.ndarray, steps: np.ndarray) -> np.ndarray:
    result = np.empty((7, 7), dtype=float)
    for column, step in enumerate(steps):
        direction = np.zeros(7, dtype=float)
        direction[column] = step
        result[:, column] = (
            -function(chart - 3.0 * direction)
            + 9.0 * function(chart - 2.0 * direction)
            - 45.0 * function(chart - direction)
            + 45.0 * function(chart + direction)
            - 9.0 * function(chart + 2.0 * direction)
            + function(chart + 3.0 * direction)
        ) / (60.0 * step)
    return result


def _relative(defect, *references) -> float:
    scale = max(
        *(float(np.max(np.abs(np.asarray(item)))) for item in references),
        np.finfo(float).tiny,
    )
    return float(np.max(np.abs(np.asarray(defect))) / scale)


def _forced_vertical_limit(
    geometry,
    chart7: np.ndarray,
    *,
    omega: float,
    alpha: float,
    stress_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    equilibrium_height = float(np.exp(chart7[5]))
    base = generalized_maxwell_cattaneo_lower_source(
        geometry,
        chart7,
        proper_vertical_frequency=omega,
        alpha=alpha,
        stress_factor=stress_factor,
    )
    forcing = 1.0e-3 * omega**2 * equilibrium_height

    departures = []
    roots = []
    for multiplier in FAST_MULTIPLIERS:
        def balance(log_height):
            trial = np.array(chart7, copy=True)
            trial[5] = float(log_height)
            source = generalized_maxwell_cattaneo_lower_source(
                geometry,
                trial,
                proper_vertical_frequency=omega,
                alpha=alpha,
                stress_factor=stress_factor,
                fast_vertical_multiplier=multiplier,
            )
            return multiplier**2 * source.hydrostatic_force_acceleration_cm_per_s2 + forcing

        root = brentq(
            balance,
            float(chart7[5] - 0.1),
            float(chart7[5] + 0.1),
            xtol=1.0e-13,
            rtol=1.0e-14,
        )
        roots.append(root)
        departures.append(abs(root - float(chart7[5])))
    del base
    departures = np.asarray(departures, dtype=float)
    orders = np.log2(departures[:-1] / departures[1:])
    return departures, orders


def _audit() -> tuple[dict, dict[str, np.ndarray]]:
    validated = _validate_parent(require_clean=False)
    gates = validated["contract"]["binding_gates"]
    start = time.perf_counter()
    context, envelope, meta = _context_and_envelope()
    context_seconds = time.perf_counter() - start
    cases = _selected_cases(context, envelope, meta)
    case_metrics = []
    source_jacobians = []
    eigenvalues = []
    limit_departures = []
    limit_orders = []
    first_failure = None
    for index, case in enumerate(cases):
        radius = case["radius_cm"]
        omega = float(context.vertical_frequency.frequency(radius))
        geometry = _geometry(context, radius, case["chart5"])
        chart = case["chart7"]
        common = {
            "proper_vertical_frequency": omega,
            "alpha": float(context.alpha),
            "stress_factor": float(context.stress_factor),
        }
        local = generalized_maxwell_cattaneo_lower_source(geometry, chart, **common)
        principal = generalized_maxwell_cattaneo_principal(geometry, chart, **common)
        steps = default_primitive_steps(
            chart,
            equilibrium_specific_stress=principal.local_state.equilibrium_specific_stress,
        )
        jacobians = [
            _source_jacobian(
                lambda candidate: generalized_maxwell_cattaneo_lower_source(
                    geometry, candidate, **common
                ).source_per_cm,
                chart,
                factor * steps,
            )
            for factor in (2.0, 1.0, 0.5)
        ]
        derivative_defect = max(
            _relative(jacobians[0] - jacobians[1], jacobians[0], jacobians[1]),
            _relative(jacobians[1] - jacobians[2], jacobians[1], jacobians[2]),
        )
        gravity_scale = max(omega**2 * np.exp(chart[5]), 1.0)
        equilibrium_source_defect = max(
            abs(local.height_material_source_per_cm[5]),
            abs(local.vertical_momentum_source_per_cm[6]),
            abs(local.hydrostatic_force_acceleration_cm_per_s2) / gravity_scale,
        )
        departures, orders = _forced_vertical_limit(
            geometry,
            chart,
            omega=omega,
            alpha=float(context.alpha),
            stress_factor=float(context.stress_factor),
        )
        metrics = {
            "label": case["label"],
            "radius_cm": radius,
            "source_derivative_ladder_relative_defect": derivative_defect,
            "equilibrium_vertical_source_relative_defect": equilibrium_source_defect,
            "vertical_total_energy_ledger_relative_defect": local.source_ledger.vertical_total_energy_relative_defect,
            "vertical_reversible_exchange_relative_defect": local.source_ledger.vertical_reversible_exchange_relative_defect,
            "minimum_fast_relaxation_observed_order": float(np.min(orders)),
            "maximum_imaginary_speed_over_c": principal.maximum_imaginary_speed_over_c,
            "maximum_light_cone_excess_over_c": principal.maximum_light_cone_excess_over_c,
            "eigenvector_condition_number": principal.eigenvector_condition_number,
        }
        reasons = []
        checks = (
            ("source:derivative", derivative_defect, gates["source_derivative_ladder_relative_defect_max"]),
            ("source:vertical_equilibrium", equilibrium_source_defect, gates["equilibrium_vertical_source_relative_defect_max"]),
            ("source:energy_ledger", metrics["vertical_total_energy_ledger_relative_defect"], gates["vertical_total_energy_ledger_relative_defect_max"]),
            ("source:reversible_ledger", metrics["vertical_reversible_exchange_relative_defect"], gates["vertical_total_energy_ledger_relative_defect_max"]),
            ("hyperbolicity:imaginary", metrics["maximum_imaginary_speed_over_c"], gates["maximum_imaginary_speed_over_c"]),
            ("hyperbolicity:light_cone", metrics["maximum_light_cone_excess_over_c"], gates["maximum_light_cone_excess_over_c"]),
            ("hyperbolicity:condition", metrics["eigenvector_condition_number"], gates["eigenvector_condition_number_max"]),
        )
        for reason, value, maximum in checks:
            if value > maximum:
                reasons.append(reason)
        if metrics["minimum_fast_relaxation_observed_order"] < gates["minimum_fast_relaxation_observed_order"]:
            reasons.append("limit:fast_vertical_order")
        metrics["passed"] = not reasons
        metrics["failure_reasons"] = reasons
        case_metrics.append(metrics)
        source_jacobians.append(jacobians)
        eigenvalues.append(principal.eigenvalues_over_c)
        limit_departures.append(departures)
        limit_orders.append(orders)
        print(f"source case {index + 1}/{len(cases)} {case['label']}: {'passed' if not reasons else 'failed'}", flush=True)
        if reasons:
            first_failure = metrics
            break

    periodic_metrics = None
    periodic_arrays = {}
    if first_failure is None:
        case = cases[0]
        radius = case["radius_cm"]
        geometry = _geometry(context, radius, case["chart5"])
        omega = float(context.vertical_frequency.frequency(radius))
        base = case["chart7"]
        constant = generalized_maxwell_cattaneo_periodic_operator(
            geometry,
            np.repeat(base[None, :], 4, axis=0),
            cell_spacing_cm=1.0e7,
            proper_vertical_frequency=omega,
            alpha=float(context.alpha),
            stress_factor=float(context.stress_factor),
            include_lower_sources=False,
        )
        perturbations = np.asarray(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [1e-3, 1e-4, -1e-4, 8e-4, 1e-8, 5e-4, 1e-4],
                [-5e-4, -8e-5, 5e-5, -4e-4, -1e-8, -2e-4, -5e-5],
                [3e-4, 4e-5, 3e-5, 2e-4, 5e-9, 1e-4, 2e-5],
            ],
            dtype=float,
        )
        smooth = generalized_maxwell_cattaneo_periodic_operator(
            geometry,
            base + perturbations,
            cell_spacing_cm=1.0e7,
            proper_vertical_frequency=omega,
            alpha=float(context.alpha),
            stress_factor=float(context.stress_factor),
            include_lower_sources=True,
        )
        constant_defect = max(
            float(np.max(np.abs(constant.spatial_equation_residuals_per_cm))),
            float(np.max(np.abs(constant.primitive_rates_per_ct))),
        )
        periodic_metrics = {
            "constant_operator_absolute_defect": constant_defect,
            "global_exact_flux_ledger_relative_defect": smooth.global_exact_flux_ledger_relative_defect,
            "maximum_interface_split_relative_defect": smooth.maximum_interface_split_relative_defect,
            "maximum_temporal_solve_relative_residual": float(np.max(smooth.temporal_solve_relative_residuals)),
        }
        reasons = []
        checks = (
            ("operator:constant", constant_defect, gates["periodic_constant_operator_absolute_defect_max"]),
            ("operator:global_flux_ledger", periodic_metrics["global_exact_flux_ledger_relative_defect"], gates["periodic_exact_flux_global_ledger_relative_defect_max"]),
            ("operator:split", periodic_metrics["maximum_interface_split_relative_defect"], gates["periodic_signed_split_relative_defect_max"]),
            ("operator:temporal_solve", periodic_metrics["maximum_temporal_solve_relative_residual"], gates["temporal_solve_relative_residual_max"]),
        )
        for reason, value, maximum in checks:
            if value > maximum:
                reasons.append(reason)
        periodic_metrics["passed"] = not reasons
        periodic_metrics["failure_reasons"] = reasons
        if reasons:
            first_failure = periodic_metrics
        periodic_arrays = {
            "periodic_charts7": smooth.primitive_charts,
            "periodic_spatial_residuals_per_cm": smooth.spatial_equation_residuals_per_cm,
            "periodic_sources_per_cm": smooth.equation_sources_per_cm,
            "periodic_rates_per_ct": smooth.primitive_rates_per_ct,
            "periodic_interface_jumps_over_c": smooth.interface_total_jumps_over_c,
        }
        print(f"periodic semidiscrete operator: {'passed' if not reasons else 'failed'}", flush=True)

    all_reasons = [] if first_failure is None else list(first_failure["failure_reasons"])
    if not all_reasons:
        classification = PASS_CLASSIFICATION
    elif any(reason.startswith("hyperbolicity:") for reason in all_reasons):
        classification = HYPERBOLICITY_FAILURE
    elif any(reason.startswith("limit:") for reason in all_reasons):
        classification = LIMIT_FAILURE
    elif any(reason.startswith("operator:") for reason in all_reasons):
        classification = OPERATOR_FAILURE
    else:
        classification = SOURCE_FAILURE
    passed = first_failure is None and len(case_metrics) == len(cases)
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "cases_planned": len(cases),
        "cases_audited": len(case_metrics),
        "context_construction_wall_seconds": context_seconds,
        "audit_wall_seconds": time.perf_counter() - start,
        "fast_vertical_multipliers": list(FAST_MULTIPLIERS),
        "cases": case_metrics,
        "periodic_operator": periodic_metrics,
        "first_failure": first_failure,
    }
    arrays = {
        "case_labels": np.asarray([case["label"] for case in case_metrics], dtype="U120"),
        "case_radii_cm": np.asarray([case["radius_cm"] for case in cases[:len(case_metrics)]], dtype=float),
        "case_charts7": np.asarray([case["chart7"] for case in cases[:len(case_metrics)]], dtype=float),
        "source_derivative_jacobian_ladders": np.asarray(source_jacobians, dtype=float),
        "eigenvalues_over_c": np.asarray(eigenvalues),
        "fast_vertical_multipliers": np.asarray(FAST_MULTIPLIERS, dtype=float),
        "forced_vertical_log_height_departures": np.asarray(limit_departures, dtype=float),
        "forced_vertical_observed_orders": np.asarray(limit_orders, dtype=float),
        **periodic_arrays,
    }
    return metrics, arrays


def _report(metrics: dict) -> str:
    return "\n".join(
        (
            "# Entropy-complete semidiscrete relaxation audit",
            "",
            f"Classification: `{metrics['classification']}`.",
            "",
            f"Audited {metrics['cases_audited']} of {metrics['cases_planned']} local source/equilibrium cases and one fixed-geometry periodic semidiscrete stencil. No radial boundary or time step was constructed.",
            "",
            (
                f"Authorized next: `{AUTHORIZED_NEXT_ON_PASS}` only."
                if metrics["passed"]
                else "The audit failed closed; no bounded trajectory is authorized."
            ),
            "",
        )
    )


def _update_catalog(summary: dict) -> None:
    utils = _utils()
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({
                "case": ARTIFACT,
                "path": str(path.relative_to(ROOT)),
                "bytes": str(path.stat().st_size),
                "sha256": utils._sha256(path),
                "scientific_status": "SUPPORTED" if summary["passed"] else "REJECTED",
            })
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case", "path", "bytes", "sha256", "scientific_status"), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    catalog = utils._read_json(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": summary["passed"],
    }
    catalog.update({
        "case_count": len({row["case"] for row in rows}),
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "all_payload_hashes_recorded": True,
        "latest_source_parent_commit": utils._git("rev-parse", "HEAD"),
        "latest_work_package": WORK_PACKAGE,
    })
    utils._write_json(CANONICAL_SUMMARY, catalog)


def _canonicalize(metrics: dict, arrays: dict[str, np.ndarray]) -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("semidiscrete audit result already exists")
    validated = _validate_parent(require_clean=True)
    utils = _utils()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utils._write_json(CANONICAL_DIRECTORY / "audit_metrics.json", metrics)
    np.savez_compressed(CANONICAL_DIRECTORY / "audit_arrays.npz", **arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": bool(metrics["passed"]),
        "all_parent_results_preserved": True,
        "local_sources_certified": bool(metrics["passed"]),
        "periodic_semidiscrete_operator_certified": bool(metrics["passed"]),
        "hydrostatic_relaxation_limit_certified": bool(metrics["passed"]),
        "new_trajectory_steps": 0,
        "radial_boundary_implementation_authorized": False,
        "bounded_crossing_manifest_authorized": bool(metrics["passed"]),
        "bounded_crossing_trajectory_authorized": False,
        "fixed_Q_invariant_object_authorized": False,
        "complete_cycle_execution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT_ON_PASS if metrics["passed"] else None,
    }
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utils._write_json(CANONICAL_DIRECTORY / "input_lock.json", {
        "parent_artifact": parent.ARTIFACT,
        "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256,
        "parent_hashes": validated["hashes"],
        "local_parent_artifact": local_parent.ARTIFACT,
        "local_parent_hashes": validated["local_hashes"],
        "source_sha256": SOURCE_SHA256,
        "source_test_sha256": SOURCE_TEST_SHA256,
    })
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_report(metrics), encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, SOURCE, SOURCE_TEST, REPORT_RELATIVE)
    utils._write_json(CANONICAL_DIRECTORY / "provenance.json", {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "implementation_commit": utils._git("rev-parse", "HEAD"),
        "implementation_tree": utils._git("rev-parse", "HEAD^{tree}"),
        "source_hashes": {path: utils._sha256(ROOT / path) for path in sources},
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "thread_environment": {name: os.environ.get(name, "") for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
    })
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{utils._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names), encoding="utf-8"
    )
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("choose --run")
    metrics, arrays = _audit()
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    summary = _canonicalize(metrics, arrays)
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
