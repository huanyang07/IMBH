#!/usr/bin/env python3
"""Audit the frozen seven-field path-conservative interface operator.

This package evaluates only isolated interfaces.  It does not assemble a
cell residual, apply a boundary condition, or advance a trajectory.
"""

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

import run_causal_inner_entropy_complete_path_conservative_spatial_manifest_wp10c9d6c7c3b5c4f25fizef as parent  # noqa: E402
import run_causal_inner_invariant_cluster_local_structural_audit_wp10c9d6c7c3b5c4f25fizee7 as local_parent  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_spatial import (  # noqa: E402
    generalized_maxwell_cattaneo_path_jump,
    generalized_maxwell_cattaneo_signed_fluctuations,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = parent.AUTHORIZED_NEXT
PASS_CLASSIFICATION = "entropy_complete_path_conservative_interface_audit_passed"
FLUX_FAILURE = "entropy_complete_path_conservative_interface_flux_failed"
SPLIT_FAILURE = "entropy_complete_path_conservative_interface_split_failed"
HYPERBOLICITY_FAILURE = (
    "entropy_complete_path_conservative_interface_hyperbolicity_failed"
)
AUTHORIZED_NEXT_ON_PASS = (
    "definitions_only_WP10c9d6c7c3b5c4f25fizeh_"
    "entropy_complete_semidiscrete_relaxation_manifest"
)
ARTIFACT = (
    "causal_inner_entropy_complete_path_conservative_interface_audit_"
    "wp10c9d6c7c3b5c4f25fizeg"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ENTROPY_COMPLETE_PATH_"
    "CONSERVATIVE_INTERFACE_AUDIT_WP10C9D6C7C3B5C4F25FIZEG_"
    "2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_entropy_complete_path_conservative_interface_"
    "audit_wp10c9d6c7c3b5c4f25fizeg.py"
)
THIS_TEST = (
    "tests/test_causal_inner_entropy_complete_path_conservative_interface_"
    "audit_wp10c9d6c7c3b5c4f25fizeg.py"
)
SPATIAL_SOURCE = parent.SPATIAL_SOURCE
SPATIAL_TEST = parent.SPATIAL_TEST
PARENT_CHECKSUM_MANIFEST_SHA256 = (
    "834cb260105d3fca19f8cc431cef25498fb9e79f12f90537bd0dbdbc5b98bedd"
)
SPATIAL_SOURCE_SHA256 = (
    "5f713ddc0a893cdfdbb9482821a9e3beb696aa8a97e8545304a0672eaf68f70d"
)
SPATIAL_TEST_SHA256 = (
    "c674774e19b789c89f9cb372dca3aabddf4d3038200774ca9266c0f743628807"
)
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
EXACT_ROWS = np.asarray((0, 1, 2, 3, 5, 6), dtype=int)
QUADRATURE_ORDERS = (4, 8, 16)
SMOOTH_AMPLITUDES = (1.0e-3, 5.0e-4, 2.5e-4)


def _utils():
    return parent._utils()


def _validate_parent(*, require_clean: bool) -> dict:
    utils = _utils()
    if utils._sha256(parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != (
        PARENT_CHECKSUM_MANIFEST_SHA256
    ):
        raise RuntimeError("spatial manifest checksum changed")
    hashes = utils._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = utils._read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    contract = utils._read_json(
        parent.CANONICAL_DIRECTORY / "spatial_contract.json"
    )
    if (
        summary["classification"] != parent.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or not summary["nonpropagating_interface_audit_authorized"]
        or summary["semidiscrete_cell_operator_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["authorized_next"] != WORK_PACKAGE
        or not contract["binding_gates"]["fail_closed"]
    ):
        raise RuntimeError("spatial manifest authorization changed")
    if utils._sha256(ROOT / SPATIAL_SOURCE) != SPATIAL_SOURCE_SHA256:
        raise RuntimeError("spatial source changed")
    if utils._sha256(ROOT / SPATIAL_TEST) != SPATIAL_TEST_SHA256:
        raise RuntimeError("spatial unit test changed")
    local_hashes = utils._validate_checksums(local_parent.CANONICAL_DIRECTORY)
    local_summary = utils._read_json(
        local_parent.CANONICAL_DIRECTORY / "summary.json"
    )
    if (
        local_summary["classification"] != local_parent.PASS_CLASSIFICATION
        or not local_summary["passed"]
        or not local_summary["complete_reduced_principal_certified"]
    ):
        raise RuntimeError("local principal certificate changed")
    if require_clean and utils._git(
        "status", "--short", "--untracked-files=no"
    ):
        raise RuntimeError("interface audit requires a clean tracked tree")
    return {
        "hashes": hashes,
        "summary": summary,
        "contract": contract,
        "local_hashes": local_hashes,
        "local_summary": local_summary,
    }


def _context_and_envelope():
    stage2 = local_parent.frozen_audit.parent.parent.parent.parent
    with np.load(
        stage2.CANONICAL_DIRECTORY / "audit_envelope.npz", allow_pickle=False
    ) as archive:
        envelope = {name: np.array(archive[name], copy=True) for name in archive.files}
    envelope_meta = _utils()._read_json(
        stage2.CANONICAL_DIRECTORY / "audit_envelope.json"
    )
    source = (
        local_parent.frozen_audit.parent.parent.parent.boundary_diagnostic
        .manifest.parent.engine.execution.source
    )
    context = source._initial_inputs()["base"]["configuration"]["context"]
    return context, envelope, envelope_meta


def _equilibrium_chart7(context, radius: float, chart5: np.ndarray) -> np.ndarray:
    _state, chart7 = local_parent.frozen_audit._chart7_at_equilibrium(
        context, float(radius), np.asarray(chart5, dtype=float)
    )
    return np.asarray(chart7, dtype=float)


def _interface_cases(context, envelope: dict, envelope_meta: dict) -> list[dict]:
    """Return the exact prospectively selected finite interface set."""

    centers = np.asarray(context.grid.centers, dtype=float)
    cases: list[dict] = []

    def profile_pair(label: str, profile: np.ndarray, left_cell: int) -> None:
        right_cell = left_cell + 1
        radius = 0.5 * (float(centers[left_cell]) + float(centers[right_cell]))
        cases.append(
            {
                "label": label,
                "radius_cm": radius,
                "left": _equilibrium_chart7(
                    context, float(centers[left_cell]), profile[left_cell]
                ),
                "right": _equilibrium_chart7(
                    context, float(centers[right_cell]), profile[right_cell]
                ),
            }
        )

    profile_pair("primary_20ms_representative", envelope["primary_20ms_base_charts5"], 36)
    profile_pair("heldout_16ms_representative", envelope["heldout_16ms_base_charts5"], 36)
    profile_pair(
        "certified_large_jump_accepted00_cells44_45",
        envelope["accepted_trajectory_base_charts5"][0],
        44,
    )

    failed_radius = float(envelope_meta["failed_face_radius_cm"])
    failed = _equilibrium_chart7(
        context, failed_radius, envelope["failed_face_chart5"]
    )
    failed_delta = np.asarray(
        [1.0e-2, 2.0e-3, -2.0e-3, 5.0e-3, 2.0e-7, 1.0e-2, 2.0e-3],
        dtype=float,
    )
    cases.append(
        {
            "label": "saved_old_complex_split_point_symmetric_face",
            "radius_cm": failed_radius,
            "left": failed - 0.5 * failed_delta,
            "right": failed + 0.5 * failed_delta,
        }
    )

    with np.load(
        local_parent.CANONICAL_DIRECTORY / "audit_arrays.npz", allow_pickle=False
    ) as archive:
        witness_radii = np.asarray(archive["witness_radii_cm"], dtype=float)
        witness_charts = np.asarray(archive["witness_charts7"], dtype=float)
        witness_labels = np.asarray(archive["witness_labels"])
    selections = (0, 657, 1315, 1973, len(witness_charts) - 1)
    witness_delta = np.asarray(
        [1.0e-3, 1.0e-4, -1.0e-4, 1.0e-3, 1.0e-8, 1.0e-3, 1.0e-4],
        dtype=float,
    )
    for ordinal, selected in enumerate(selections):
        center = witness_charts[selected]
        cases.append(
            {
                "label": f"off_equilibrium_{ordinal:02d}_{witness_labels[selected]}",
                "radius_cm": float(witness_radii[selected]),
                "left": center - 0.5 * witness_delta,
                "right": center + 0.5 * witness_delta,
            }
        )
    return cases


def _relative(defect, *references) -> float:
    scale = max(
        *(float(np.max(np.abs(np.asarray(item)))) for item in references),
        np.finfo(float).tiny,
    )
    return float(np.max(np.abs(np.asarray(defect))) / scale)


def _audit_case(context, raw: dict) -> tuple[dict, dict[str, np.ndarray]]:
    radius = float(raw["radius_cm"])
    left = np.asarray(raw["left"], dtype=float)
    right = np.asarray(raw["right"], dtype=float)
    geometry = local_parent.frozen_audit.parent.parent.parent.boundary_diagnostic.radial._cell_state(
        context, radius, left[:5]
    ).geometry
    common = {
        "proper_vertical_frequency": float(
            context.vertical_frequency.frequency(radius)
        ),
        "alpha": float(context.alpha),
        "stress_factor": float(context.stress_factor),
    }
    jumps = [
        generalized_maxwell_cattaneo_path_jump(
            geometry, left, right, quadrature_order=order, **common
        )
        for order in QUADRATURE_ORDERS
    ]
    split = generalized_maxwell_cattaneo_signed_fluctuations(
        geometry, left, right, quadrature_order=8, **common
    )
    reverse = generalized_maxwell_cattaneo_signed_fluctuations(
        geometry, right, left, quadrature_order=8, **common
    )
    jump_scale = max(
        float(np.max(np.abs(split.path_jump.total_principal_jump_over_c))),
        np.finfo(float).tiny,
    )
    reversal = max(
        _relative(
            reverse.path_jump.total_principal_jump_over_c
            + split.path_jump.total_principal_jump_over_c,
            reverse.path_jump.total_principal_jump_over_c,
            split.path_jump.total_principal_jump_over_c,
        ),
        _relative(
            reverse.negative_fluctuation_over_c
            + split.negative_fluctuation_over_c,
            reverse.negative_fluctuation_over_c,
            split.negative_fluctuation_over_c,
        ),
        _relative(
            reverse.positive_fluctuation_over_c
            + split.positive_fluctuation_over_c,
            reverse.positive_fluctuation_over_c,
            split.positive_fluctuation_over_c,
        ),
    )
    quadrature = _relative(
        jumps[1].total_principal_jump_over_c
        - jumps[2].total_principal_jump_over_c,
        jumps[1].total_principal_jump_over_c,
        jumps[2].total_principal_jump_over_c,
    )
    direction = right - left
    smooth_defects = []
    for amplitude in SMOOTH_AMPLITUDES:
        small_left = 0.5 * (left + right) - 0.5 * amplitude * direction
        small_right = 0.5 * (left + right) + 0.5 * amplitude * direction
        small = generalized_maxwell_cattaneo_signed_fluctuations(
            geometry, small_left, small_right, quadrature_order=8, **common
        )
        linear = small.midpoint_principal.radial_matrix @ (
            small_right - small_left
        )
        smooth_defects.append(
            _relative(
                small.path_jump.total_principal_jump_over_c - linear,
                small.path_jump.total_principal_jump_over_c,
                linear,
            )
        )
    constant = generalized_maxwell_cattaneo_signed_fluctuations(
        geometry, left, left, quadrature_order=8, **common
    )
    constant_absolute = max(
        float(np.max(np.abs(constant.path_jump.total_principal_jump_over_c))),
        float(np.max(np.abs(constant.dissipation_over_c))),
        float(np.max(np.abs(constant.negative_fluctuation_over_c))),
        float(np.max(np.abs(constant.positive_fluctuation_over_c))),
    )
    principal = split.midpoint_principal
    metrics = {
        "label": str(raw["label"]),
        "radius_cm": radius,
        "chart_jump_maximum_absolute": float(np.max(np.abs(direction))),
        "path_jump_maximum_absolute": jump_scale,
        "constant_state_absolute_jump": constant_absolute,
        "exact_flux_parity_relative_defect": split.path_jump.exact_flux_parity_relative_defect,
        "split_closure_relative_defect": split.split_closure_relative_defect,
        "shared_exact_flux_relative_defect": split.shared_exact_flux_relative_defect,
        "path_reversal_relative_defect": reversal,
        "quadrature_middle_fine_relative_defect": quadrature,
        "maximum_smooth_limit_relative_defect": float(max(smooth_defects)),
        "maximum_imaginary_speed_over_c": principal.maximum_imaginary_speed_over_c,
        "maximum_light_cone_excess_over_c": principal.maximum_light_cone_excess_over_c,
        "eigenvector_condition_number": principal.eigenvector_condition_number,
        "characteristic_quadratic_dissipation": split.characteristic_quadratic_dissipation,
    }
    gates = parent._contract()["binding_gates"]
    reasons: list[str] = []
    checks = (
        ("flux:constant_state", constant_absolute, gates["constant_state_absolute_jump_max"]),
        ("flux:exact_flux_parity", metrics["exact_flux_parity_relative_defect"], gates["conservative_and_material_flux_parity_relative_defect_max"]),
        ("split:closure", metrics["split_closure_relative_defect"], gates["path_partition_and_split_relative_defect_max"]),
        ("flux:shared", metrics["shared_exact_flux_relative_defect"], gates["shared_flux_relative_defect_max"]),
        ("split:reversal", reversal, gates["path_reversal_relative_defect_max"]),
        ("flux:quadrature", quadrature, gates["quadrature_ladder_relative_defect_max"]),
        ("flux:smooth_limit", metrics["maximum_smooth_limit_relative_defect"], gates["smooth_limit_relative_defect_max"]),
        ("hyperbolicity:imaginary_speed", metrics["maximum_imaginary_speed_over_c"], gates["maximum_imaginary_speed_over_c"]),
        ("hyperbolicity:light_cone", metrics["maximum_light_cone_excess_over_c"], gates["maximum_light_cone_excess_over_c"]),
        ("hyperbolicity:eigenbasis_condition", metrics["eigenvector_condition_number"], gates["eigenvector_condition_number_max"]),
    )
    for reason, value, maximum in checks:
        if value > maximum:
            reasons.append(reason)
    if metrics["characteristic_quadratic_dissipation"] < gates[
        "characteristic_quadratic_dissipation_min"
    ]:
        reasons.append("split:negative_characteristic_dissipation")
    metrics["passed"] = not reasons
    metrics["failure_reasons"] = reasons
    arrays = {
        "left_chart": left,
        "right_chart": right,
        "path_jumps_over_c": np.asarray(
            [item.total_principal_jump_over_c for item in jumps], dtype=float
        ),
        "exact_flux_jump_over_c": split.path_jump.exact_flux_jump_over_c,
        "dissipation_over_c": split.dissipation_over_c,
        "negative_fluctuation_over_c": split.negative_fluctuation_over_c,
        "positive_fluctuation_over_c": split.positive_fluctuation_over_c,
        "eigenvalues_over_c": split.eigenvalues_over_c,
        "smooth_limit_relative_defects": np.asarray(smooth_defects, dtype=float),
    }
    return metrics, arrays


def _classification(reasons: list[str]) -> str:
    if not reasons:
        return PASS_CLASSIFICATION
    if any(reason.startswith("hyperbolicity:") for reason in reasons):
        return HYPERBOLICITY_FAILURE
    if any(reason.startswith("split:") for reason in reasons):
        return SPLIT_FAILURE
    return FLUX_FAILURE


def _audit() -> tuple[dict, dict[str, np.ndarray]]:
    _validate_parent(require_clean=False)
    start = time.perf_counter()
    context, envelope, envelope_meta = _context_and_envelope()
    context_seconds = time.perf_counter() - start
    cases = _interface_cases(context, envelope, envelope_meta)
    case_metrics = []
    array_blocks: dict[str, list[np.ndarray]] = {}
    first_failure = None
    for index, case in enumerate(cases):
        metrics, arrays = _audit_case(context, case)
        case_metrics.append(metrics)
        for name, values in arrays.items():
            array_blocks.setdefault(name, []).append(np.asarray(values))
        print(
            f"interface {index + 1}/{len(cases)} {case['label']}: "
            f"{'passed' if metrics['passed'] else 'failed'}",
            flush=True,
        )
        if not metrics["passed"]:
            first_failure = metrics
            break
    reasons = [] if first_failure is None else list(first_failure["failure_reasons"])
    passed = first_failure is None and len(case_metrics) == len(cases)
    maxima = {}
    for key in (
        "constant_state_absolute_jump",
        "exact_flux_parity_relative_defect",
        "split_closure_relative_defect",
        "shared_exact_flux_relative_defect",
        "path_reversal_relative_defect",
        "quadrature_middle_fine_relative_defect",
        "maximum_smooth_limit_relative_defect",
        "maximum_imaginary_speed_over_c",
        "maximum_light_cone_excess_over_c",
        "eigenvector_condition_number",
    ):
        maxima[key] = max(float(case[key]) for case in case_metrics)
    minimum_quadratic = min(
        float(case["characteristic_quadratic_dissipation"])
        for case in case_metrics
    )
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": _classification(reasons),
        "passed": passed,
        "cases_planned": len(cases),
        "cases_audited": len(case_metrics),
        "context_construction_wall_seconds": context_seconds,
        "audit_wall_seconds": time.perf_counter() - start,
        "quadrature_orders": list(QUADRATURE_ORDERS),
        "smooth_amplitudes": list(SMOOTH_AMPLITUDES),
        "maxima": maxima,
        "minimum_characteristic_quadratic_dissipation": minimum_quadratic,
        "first_failure": first_failure,
        "cases": case_metrics,
    }
    arrays_out = {
        "case_labels": np.asarray([case["label"] for case in case_metrics], dtype="U220"),
        "case_radii_cm": np.asarray([case["radius_cm"] for case in case_metrics], dtype=float),
        "quadrature_orders": np.asarray(QUADRATURE_ORDERS, dtype=int),
        "smooth_amplitudes": np.asarray(SMOOTH_AMPLITUDES, dtype=float),
    }
    for name, blocks in array_blocks.items():
        arrays_out[name] = np.asarray(blocks)
    return metrics, arrays_out


def _report(metrics: dict) -> str:
    return "\n".join(
        (
            "# Entropy-complete path-conservative interface audit",
            "",
            f"Classification: `{metrics['classification']}`.",
            "",
            f"Audited {metrics['cases_audited']} of {metrics['cases_planned']} prospectively selected isolated interfaces. No cell residual or trajectory step was constructed.",
            "",
            f"Maximum defects: `{json.dumps(metrics['maxima'], sort_keys=True)}`.",
            "",
            f"Minimum characteristic quadratic dissipation: `{metrics['minimum_characteristic_quadratic_dissipation']}`.",
            "",
            (
                f"Authorized next: `{AUTHORIZED_NEXT_ON_PASS}` only."
                if metrics["passed"]
                else "The interface audit failed closed; no downstream work is authorized."
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
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": utils._sha256(path),
                    "scientific_status": "SUPPORTED" if summary["passed"] else "REJECTED",
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
        raise RuntimeError("interface audit result already exists")
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
        "isolated_interfaces_only": True,
        "exact_flux_rows_certified": bool(metrics["passed"]),
        "complete_eigenbasis_split_certified": bool(metrics["passed"]),
        "new_trajectory_steps": 0,
        "semidiscrete_cell_operator_authorized": False,
        "relaxation_limit_audit_authorized": False,
        "seven_field_trajectory_authorized": False,
        "complete_cycle_execution_authorized": False,
        "authorized_next": AUTHORIZED_NEXT_ON_PASS if metrics["passed"] else None,
    }
    utils._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utils._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_artifact": parent.ARTIFACT,
            "parent_checksum_manifest_sha256": PARENT_CHECKSUM_MANIFEST_SHA256,
            "parent_hashes": validated["hashes"],
            "local_parent_artifact": local_parent.ARTIFACT,
            "local_parent_hashes": validated["local_hashes"],
            "spatial_source_sha256": SPATIAL_SOURCE_SHA256,
            "spatial_test_sha256": SPATIAL_TEST_SHA256,
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_report(metrics), encoding="utf-8")
    sources = (THIS_RUNNER, THIS_TEST, SPATIAL_SOURCE, SPATIAL_TEST, REPORT_RELATIVE)
    utils._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "implementation_commit": utils._git("rev-parse", "HEAD"),
            "implementation_tree": utils._git("rev-parse", "HEAD^{tree}"),
            "source_hashes": {path: utils._sha256(ROOT / path) for path in sources},
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "thread_environment": {
                name: os.environ.get(name, "")
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
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
    arguments = parser.parse_args()
    if not arguments.run:
        parser.error("choose --run")
    metrics, arrays = _audit()
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    summary = _canonicalize(metrics, arrays)
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
