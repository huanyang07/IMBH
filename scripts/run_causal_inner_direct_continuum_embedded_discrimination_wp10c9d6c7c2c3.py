#!/usr/bin/env python3
"""Run the frozen direct-continuum embedded discrimination.

The exact nine c2c1 profile bases are propagated on the unchanged
98/147/245-cell nonoverlapping embedded layouts.  State histories are
conservatively restricted to the common N98 parent grid, the thirteen
active-domain physical exports are evaluated from the monolithic tangent,
and the fixed-exterior N769/N513 reference certified in c2c2 supplies the
independent state truth and uncertainty.
"""

from __future__ import annotations

from dataclasses import replace
import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import scipy
from scipy.sparse.linalg import expm_multiply


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_causal_inner_direct_continuum_contract_manifest_wp10c9d6c7c2b6d as b6d  # noqa: E402
import run_causal_inner_direct_continuum_embedded_manifest_wp10c9d6c7c2c1 as c2c1  # noqa: E402
import run_causal_inner_direct_continuum_uniform_recertification_wp10c9d6c7c2b6e as b6e  # noqa: E402
import run_causal_inner_fixed_exterior_continuum_reference_wp10c9d6c7c2c2 as c2c2  # noqa: E402
import run_causal_inner_monolithic_four_level_wp10c9d6c2 as c6c2  # noqa: E402
import run_causal_inner_one_way_uniform_scattering_wp10c9d6c7c2b1 as c2b1  # noqa: E402
import run_causal_inner_revised_uniform_arrival_transfer_wp10c9d6c7c2b6b as b6b  # noqa: E402
import run_causal_inner_scattering_energy_wp10c9d6c7c2a2 as c2a2  # noqa: E402

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    causal_five_field_monolithic_frozen_tangent,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_embedded_patch import (  # noqa: E402
    make_causal_embedded_patch_layout,
    restrict_causal_embedded_patch_cell_averages,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_embedded_validation import (  # noqa: E402
    causal_embedded_active_observable_audit,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_geometry import (  # noqa: E402
    make_kerr_schild_column_grid_from_edges,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_validation import (  # noqa: E402
    causal_exact_semigroup_integral_history,
    causal_packet_history_metrics,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_windowed_contract import (  # noqa: E402
    causal_field_history_norm,
    causal_trapezoid_weights,
    causal_windowed_richardson_reference,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c2c3"
ANALYZED_BASE_COMMIT = "068c823d97bcaad7d5054dc8f2999a94d60a93a2"
ANALYZED_BASE_PARENT = "e763b39e598ad8302b2918220fa4dc4a39533363"
ANALYZED_BASE_TREE = "87c87bb591eea8d24bb7c4f862e24d773e709b92"

FIELDS = 5
PARENT_CELLS = 98
PARENT_COUPLING_FACE = 49
REFINEMENT_RATIOS = (1, 2, 4)
LABELS = tuple(c2c1.LAYOUT_LABELS[ratio] for ratio in REFINEMENT_RATIOS)
BASES = b6d.BINDING_BASES
TIME_SAMPLES = 513
RELATIVE_ACTIVITY = 1.0e-8
MAXIMUM_PROPAGATION_SCALING_DEFECT = 1.0e-12
MAXIMUM_EXACT_INTEGRAL_RESIDUAL = 1.0e-11
MAXIMUM_RESTART_DEFECT = 2.0e-10

OBSERVABLE_NAMES = (
    "inner_mass_flux",
    "inner_angular_momentum_flux",
    "inner_killing_energy_flux",
    "coupling_mass_flux",
    "coupling_angular_momentum_flux",
    "coupling_killing_energy_flux",
    "net_mass_drive",
    "net_angular_momentum_drive",
    "net_killing_energy_drive",
    "cooling_angular_momentum",
    "cooling_killing_energy",
    "lower_height_angular_momentum_work",
    "lower_height_killing_energy_work",
)

THIS_RUNNER = (
    "scripts/"
    "run_causal_inner_direct_continuum_embedded_discrimination_"
    "wp10c9d6c7c2c3.py"
)
THIS_TEST = (
    "tests/"
    "test_causal_inner_direct_continuum_embedded_discrimination_"
    "wp10c9d6c7c2c3.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_DIRECT_CONTINUUM_EMBEDDED_DISCRIMINATION_"
    "WP10C9D6C7C2C3_RESULTS_2026-07-31.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE

PARENT_DIRECTORY = c2c2.CANONICAL_DIRECTORY
C2C1_DIRECTORY = c2c1.CANONICAL_DIRECTORY
B6D_DIRECTORY = b6d.CANONICAL_DIRECTORY
C2A2_DIRECTORY = c2a2.CANONICAL_DIRECTORY
C7A_DIRECTORY = b6e.C7A_DIRECTORY
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_direct_continuum_embedded_discrimination_"
    "wp10c9d6c7c2c3"
)
CHECKPOINT_DIRECTORY = (
    ROOT
    / "outputs/checkpoints/"
    "causal_inner_direct_continuum_embedded_discrimination_"
    "wp10c9d6c7c2c3"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _relative_defect(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    scale = max(
        float(np.max(np.abs(left))),
        float(np.max(np.abs(right))),
        np.finfo(float).tiny,
    )
    return float(np.max(np.abs(left - right)) / scale)


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=float).ravel()
    right = np.asarray(second, dtype=float).ravel()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= np.finfo(float).tiny:
        return 1.0
    return float(np.dot(left, right) / denominator)


def _orders(values: list[float]) -> list[float]:
    tiny = np.finfo(float).tiny
    return [
        float(
            np.log2(
                max(values[index], tiny)
                / max(values[index + 1], tiny)
            )
        )
        for index in range(len(values) - 1)
    ]


def _validate_parent() -> tuple[dict, dict, dict]:
    summary = _read_json(PARENT_DIRECTORY / "summary.json")
    c2c1_summary = _read_json(C2C1_DIRECTORY / "summary.json")
    manifest = _read_json(C2C1_DIRECTORY / "embedded_manifest.json")
    if (
        summary["classification"]
        != "fixed_exterior_continuum_reference_certified_"
        "embedded_propagation_authorized"
        or not summary["passed"]
        or not summary["binding_decision"]["embedded_propagation_authorized"]
        or summary["authorized_next"]
        != "WP10c9d6c7c2c3_direct_continuum_embedded_discrimination"
        or c2c1_summary["manifest_sha256"]
        != manifest["manifest_sha256"]
    ):
        raise RuntimeError("c2c1/c2c2 authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT)
        != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("WP10c9d6c7c2c3 analyzed identity changed")
    return summary, c2c1_summary, manifest


def _tangent_checkpoint(label: str) -> tuple[Path, Path]:
    return (
        CHECKPOINT_DIRECTORY / f"{label}_tangent.npz",
        CHECKPOINT_DIRECTORY / f"{label}_tangent.json",
    )


def _build_configurations(
    parent_context,
    field_scales: np.ndarray,
    c2c1_arrays: dict[str, np.ndarray],
) -> tuple[dict, dict, dict]:
    parent_edges = np.asarray(c2c1_arrays["parent_patch_edges"], dtype=float)
    parent_grid = make_kerr_schild_column_grid_from_edges(
        parent_edges, parent_context.grid.gravitational_radius
    )
    configurations = {}
    layouts = {}
    method_reports = {}
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for ratio, label in zip(REFINEMENT_RATIOS, LABELS, strict=True):
        layout = make_causal_embedded_patch_layout(
            parent_grid, PARENT_COUPLING_FACE, ratio
        )
        context = replace(
            parent_context, grid=layout.grid, stream_sources=None
        ).validated()
        base = np.asarray(
            c2c1_arrays[f"{label}__base_primitive_charts"], dtype=float
        )
        columns, rows = c6c2._scales_for(context, base)
        tangent_path, report_path = _tangent_checkpoint(label)
        if tangent_path.is_file() and report_path.is_file():
            stored = _load_npz(tangent_path)
            report = _read_json(report_path)
            valid = bool(
                report.get("source_parent_commit") == ANALYZED_BASE_COMMIT
                and report.get("label") == label
                and report.get("passed")
                and stored["generator"].shape
                == (base.size, base.size)
                and stored["observable_map"].shape == (13, base.size)
            )
        else:
            valid = False
        if valid:
            generator = np.asarray(stored["generator"], dtype=float)
            observable_map = np.asarray(
                stored["observable_map"], dtype=float
            )
            method = report["method_report"]
        else:
            print(f"{WORK_PACKAGE}: build tangent {label}", flush=True)
            tangent = causal_five_field_monolithic_frozen_tangent(
                context,
                base,
                primitive_column_scales=columns,
                conservation_row_scales=rows,
                path_quadrature_order=c2b1.PATH_QUADRATURE_ORDER,
            )
            active = causal_embedded_active_observable_audit(
                tangent, layout.coupling_face_index
            )
            method = c2b1._method_report(tangent, active)
            if not method["passed"]:
                raise RuntimeError(f"{label} method preflight failed")
            generator = np.asarray(
                tangent.scaled_generator_per_s, dtype=float
            )
            observable_map = np.asarray(
                active.observable_map, dtype=float
            )
            np.savez_compressed(
                tangent_path,
                generator=generator,
                observable_map=observable_map,
            )
            _write_json(
                report_path,
                {
                    "source_parent_commit": ANALYZED_BASE_COMMIT,
                    "label": label,
                    "passed": True,
                    "method_report": method,
                },
            )
        configurations[label] = {
            "context": context,
            "base": base,
            "columns": columns,
            "rows": rows,
            "generator": generator,
            "observable_map": observable_map,
        }
        layouts[label] = layout
        method_reports[label] = method
    return configurations, layouts, method_reports


def _combine_basis(
    acoustic: np.ndarray,
    shear: np.ndarray,
    coefficients: dict[str, np.ndarray],
) -> np.ndarray:
    return np.stack(
        [
            pair[0] * acoustic + pair[1] * shear
            for pair in coefficients.values()
        ],
        axis=1,
    )


def _propagation_checkpoint(label: str) -> tuple[Path, Path]:
    return (
        CHECKPOINT_DIRECTORY / f"{label}_propagation.npz",
        CHECKPOINT_DIRECTORY / f"{label}_propagation.json",
    )


def _propagate(
    configurations: dict,
    layouts: dict,
    c2c1_arrays: dict[str, np.ndarray],
    coefficients: dict[str, np.ndarray],
    times: np.ndarray,
) -> tuple[dict, dict]:
    propagated = {}
    reports = {}
    for label in LABELS:
        path, report_path = _propagation_checkpoint(label)
        if path.is_file() and report_path.is_file():
            report = _read_json(report_path)
            valid = bool(
                report.get("source_parent_commit") == ANALYZED_BASE_COMMIT
                and report.get("label") == label
                and report.get("schema_version") == SCHEMA_VERSION
            )
        else:
            valid = False
        if valid:
            stored = _load_npz(path)
            propagated[label] = stored
            reports[label] = report["propagation_report"]
            continue
        configuration = configurations[label]
        layout = layouts[label]
        columns = np.asarray(configuration["columns"], dtype=float)
        initial = np.column_stack(
            [
                np.asarray(
                    c2c1_arrays[f"{name}__{label}__packet"], dtype=float
                ).ravel()
                / columns
                for name in ("acoustic", "shear")
            ]
        )
        generator = np.asarray(configuration["generator"], dtype=float)
        trace = float(np.trace(generator))
        print(f"{WORK_PACKAGE}: propagate {label}", flush=True)
        scaled = np.asarray(
            expm_multiply(
                generator,
                initial,
                start=float(times[0]),
                stop=float(times[-1]),
                num=times.size,
                endpoint=True,
                traceA=trace,
            ),
            dtype=float,
        )
        half = np.asarray(scaled[(times.size - 1) // 2], dtype=float)
        restarted = np.asarray(
            expm_multiply(
                0.5 * float(times[-1]) * generator,
                half,
                traceA=0.5 * float(times[-1]) * trace,
            ),
            dtype=float,
        )
        exact = causal_exact_semigroup_integral_history(
            generator, scaled, initial
        )
        basis_physical = np.transpose(
            scaled * columns[None, :, None], (0, 2, 1)
        ).reshape(times.size, 2, layout.n_cells, FIELDS)
        physical = _combine_basis(
            basis_physical[:, 0], basis_physical[:, 1], coefficients
        )
        parent_state = restrict_causal_embedded_patch_cell_averages(
            physical, layout
        )
        basis_signals = np.einsum(
            "on,tnp->tpo",
            configuration["observable_map"],
            scaled,
            optimize=True,
        )
        signals = _combine_basis(
            basis_signals[:, 0], basis_signals[:, 1], coefficients
        )
        basis_cumulative = np.einsum(
            "on,tnp->tpo",
            configuration["observable_map"],
            exact.integrated_states,
            optimize=True,
        )
        cumulative = _combine_basis(
            basis_cumulative[:, 0],
            basis_cumulative[:, 1],
            coefficients,
        )
        restart_physical = np.transpose(
            restarted * columns[:, None], (1, 0)
        ).reshape(2, layout.n_cells, FIELDS)
        restart_parent_basis = (
            restrict_causal_embedded_patch_cell_averages(
                restart_physical, layout
            )
        )
        restart_parent = _combine_basis(
            restart_parent_basis[0],
            restart_parent_basis[1],
            coefficients,
        )
        restart_parent = np.moveaxis(restart_parent, 1, 0)
        arrays = {
            "times": times,
            "parent_state": parent_state,
            "signals": signals,
            "cumulative_signals": cumulative,
            "restart_parent_state": restart_parent,
        }
        propagation_report = {
            "restart_relative_defect": _relative_defect(
                restarted, scaled[-1]
            ),
            "maximum_exact_integral_relative_solve_residual": (
                exact.maximum_relative_solve_residual
            ),
        }
        np.savez_compressed(path, **arrays)
        _write_json(
            report_path,
            {
                "schema_version": SCHEMA_VERSION,
                "source_parent_commit": ANALYZED_BASE_COMMIT,
                "label": label,
                "propagation_report": propagation_report,
            },
        )
        propagated[label] = arrays
        reports[label] = propagation_report
    return propagated, reports


def _metric_payload(metrics) -> dict:
    significant = np.asarray(metrics.significant_components, dtype=int)
    return {
        "passed": bool(metrics.passed),
        "significant_components": [
            OBSERVABLE_NAMES[index] for index in significant
        ],
        "component_orders": {
            OBSERVABLE_NAMES[index]: float(order)
            for index, order in zip(
                significant, metrics.component_orders, strict=True
            )
        },
        "observed_rms_order": metrics.observed_rms_order,
        "observed_maximum_order": metrics.observed_maximum_order,
        "minimum_significant_component_order": (
            metrics.minimum_significant_component_order
        ),
        "maximum_fine_normalized_difference": (
            metrics.maximum_fine_normalized_difference
        ),
        "history_cosine": metrics.history_cosine,
        "refinement_error_cosine": metrics.refinement_error_cosine,
    }


def _export_metrics(
    histories: list[np.ndarray],
    scales: np.ndarray,
    contract: dict,
):
    return causal_packet_history_metrics(
        *histories,
        physical_scales=scales,
        relative_activity=RELATIVE_ACTIVITY,
        minimum_rms_order=contract["minimum_RMS_order"],
        minimum_maximum_order=contract["minimum_maximum_order"],
        minimum_significant_component_order=contract[
            "minimum_significant_component_order"
        ],
        maximum_fine_normalized_difference=contract[
            "maximum_fine_normalized_difference"
        ],
        minimum_history_cosine=contract["minimum_history_cosine"],
        minimum_refinement_error_cosine=contract[
            "minimum_refinement_error_cosine"
        ],
    )


def _fixed_reference(
    coefficients: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    primary_checkpoint = _load_npz(
        c2c2.CHECKPOINT_DIRECTORY / "matched_reference_N769.npz"
    )
    secondary_checkpoint = _load_npz(
        c2c2.CHECKPOINT_DIRECTORY / "matched_reference_N513.npz"
    )
    primary = primary_checkpoint["common_state"]
    secondary = secondary_checkpoint["common_state"]
    finite = b6e._load_finite_basis(PARENT_CELLS, coefficients)
    reference_times = np.asarray(primary_checkpoint["times"], dtype=float)
    finite_times = np.asarray(finite["times"], dtype=float)
    stride = (finite_times.size - 1) // (reference_times.size - 1)
    if (
        stride < 1
        or finite_times[::stride].shape != reference_times.shape
        or not np.array_equal(finite_times[::stride], reference_times)
        or not np.array_equal(
            secondary_checkpoint["times"], reference_times
        )
    ):
        raise RuntimeError("fixed-exterior and parent time grids changed")
    outer = np.asarray(
        finite["physical"][
            ::stride, :, PARENT_COUPLING_FACE:
        ],
        dtype=float,
    )
    return (
        np.concatenate((primary, outer), axis=2),
        np.concatenate((secondary, outer), axis=2),
    )


def _state_report(
    histories: list[np.ndarray],
    primary: np.ndarray,
    secondary: np.ndarray,
    times: np.ndarray,
    parent_measures: np.ndarray,
    field_scales: np.ndarray,
    contract: dict,
) -> dict:
    coarse, medium, fine = histories
    time_weights = causal_trapezoid_weights(times)
    richardson = causal_windowed_richardson_reference(
        coarse,
        medium,
        fine,
        times=times,
        coarse_cell_measures=parent_measures,
        field_scales=field_scales,
    )
    errors = [
        causal_field_history_norm(
            values - primary,
            cell_measures=parent_measures,
            field_scales=field_scales,
            time_weights=time_weights,
        )
        for values in histories
    ]
    direct_orders = _orders(errors)
    response_norm = max(
        causal_field_history_norm(
            primary,
            cell_measures=parent_measures,
            field_scales=field_scales,
            time_weights=time_weights,
        ),
        np.finfo(float).tiny,
    )
    response_maximum = max(
        float(np.max(np.abs(primary / field_scales))),
        np.finfo(float).tiny,
    )
    fine_rms = errors[-1] / response_norm
    fine_maximum = float(
        np.max(np.abs((fine - primary) / field_scales))
        / response_maximum
    )
    reference_difference = causal_field_history_norm(
        primary - secondary,
        cell_measures=parent_measures,
        field_scales=field_scales,
        time_weights=time_weights,
    )
    reference_ratio = reference_difference / max(
        errors[-1], np.finfo(float).tiny
    )
    pairwise_passed = bool(
        richardson.observed_order >= contract["minimum_RMS_order"]
        and richardson.minimum_significant_component_order
        >= contract["minimum_significant_component_order"]
        and richardson.refinement_error_cosine
        >= contract["minimum_refinement_error_cosine"]
    )
    direct_passed = bool(
        min(direct_orders) >= contract["minimum_RMS_order"]
        and fine_rms <= contract["maximum_fine_normalized_difference"]
        and fine_maximum <= contract["maximum_fine_normalized_difference"]
        and _cosine(fine, primary) >= contract["minimum_history_cosine"]
        and reference_ratio <= 0.10
    )
    return {
        "pairwise_observed_order": richardson.observed_order,
        "pairwise_minimum_significant_component_order": (
            richardson.minimum_significant_component_order
        ),
        "pairwise_refinement_error_cosine": (
            richardson.refinement_error_cosine
        ),
        "direct_weighted_errors": errors,
        "direct_error_orders": direct_orders,
        "fine_direct_response_relative_RMS_error": fine_rms,
        "fine_direct_response_relative_maximum_error": fine_maximum,
        "fine_reference_history_cosine": _cosine(fine, primary),
        "N769_N513_weighted_difference": reference_difference,
        "reference_uncertainty_to_fine_direct_error_ratio": (
            reference_ratio
        ),
        "pairwise_passed": pairwise_passed,
        "direct_passed": direct_passed,
        "passed": bool(pairwise_passed and direct_passed),
    }


def _comparison(
    propagated: dict,
    layouts: dict,
    coefficients: dict[str, np.ndarray],
    observable_scales: np.ndarray,
    field_scales: np.ndarray,
    contract: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    times = np.asarray(propagated[LABELS[0]]["times"], dtype=float)
    primary, secondary = _fixed_reference(coefficients)
    reports = {}
    instantaneous_matrix = np.empty((len(BASES), 6), dtype=float)
    cumulative_matrix = np.empty((len(BASES), 6), dtype=float)
    state_matrix = np.empty((len(BASES), 8), dtype=float)
    for index, name in enumerate(BASES):
        instantaneous = _export_metrics(
            [
                propagated[label]["signals"][:, index]
                for label in LABELS
            ],
            observable_scales,
            contract,
        )
        cumulative = _export_metrics(
            [
                propagated[label]["cumulative_signals"][:, index]
                for label in LABELS
            ],
            observable_scales * float(times[-1]),
            contract,
        )
        state = _state_report(
            [
                propagated[label]["parent_state"][:, index]
                for label in LABELS
            ],
            primary[:, index],
            secondary[:, index],
            times,
            np.asarray(layouts[LABELS[0]].parent_grid.cell_measures),
            field_scales,
            contract,
        )
        passed = bool(
            instantaneous.passed and cumulative.passed and state["passed"]
        )
        reports[name] = {
            "instantaneous_exports": _metric_payload(instantaneous),
            "cumulative_exports": _metric_payload(cumulative),
            "state": state,
            "passed": passed,
        }
        instantaneous_matrix[index] = (
            instantaneous.observed_rms_order,
            instantaneous.observed_maximum_order,
            instantaneous.minimum_significant_component_order,
            instantaneous.maximum_fine_normalized_difference,
            instantaneous.history_cosine,
            instantaneous.refinement_error_cosine,
        )
        cumulative_matrix[index] = (
            cumulative.observed_rms_order,
            cumulative.observed_maximum_order,
            cumulative.minimum_significant_component_order,
            cumulative.maximum_fine_normalized_difference,
            cumulative.history_cosine,
            cumulative.refinement_error_cosine,
        )
        state_matrix[index] = (
            state["pairwise_observed_order"],
            state["pairwise_minimum_significant_component_order"],
            state["pairwise_refinement_error_cosine"],
            *state["direct_error_orders"],
            state["fine_direct_response_relative_RMS_error"],
            state["fine_direct_response_relative_maximum_error"],
            state[
                "reference_uncertainty_to_fine_direct_error_ratio"
            ],
        )
    arrays = {
        "times_seconds": times,
        "instantaneous_metric_matrix": instantaneous_matrix,
        "cumulative_metric_matrix": cumulative_matrix,
        "state_metric_matrix": state_matrix,
        "N769_reference_state_endpoint": primary[-1],
        "N513_reference_state_endpoint": secondary[-1],
        **{
            f"{label}__base_instantaneous_exports": propagated[label][
                "signals"
            ]
            for label in LABELS
        },
        **{
            f"{label}__base_cumulative_exports": propagated[label][
                "cumulative_signals"
            ]
            for label in LABELS
        },
        **{
            f"{label}__base_parent_state_endpoint": propagated[label][
                "parent_state"
            ][-1]
            for label in LABELS
        },
    }
    return {
        "profile_reports": reports,
        "failed_profiles": [
            name for name in BASES if not reports[name]["passed"]
        ],
        "all_profiles_passed": all(
            reports[name]["passed"] for name in BASES
        ),
        "passed": all(reports[name]["passed"] for name in BASES),
    }, arrays


def _input_hashes() -> dict[str, str]:
    paths = (
        PARENT_DIRECTORY / "summary.json",
        PARENT_DIRECTORY / "decisive_arrays.npz",
        C2C1_DIRECTORY / "embedded_manifest.json",
        C2C1_DIRECTORY / "decisive_arrays.npz",
        B6D_DIRECTORY / "summary.json",
        C2A2_DIRECTORY / "decisive_arrays.npz",
    )
    return {
        str(path.relative_to(ROOT)): _sha256(path) for path in paths
    }


def _refresh_sha256s(directory: Path) -> None:
    entries = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            entries.append(f"{_sha256(path)}  {path.name}")
    (directory / "SHA256SUMS.txt").write_text(
        "\n".join(entries) + "\n", encoding="utf-8"
    )


def _refresh_catalog() -> None:
    rows = []
    for case in sorted(CANONICAL_DIRECTORY.parent.iterdir()):
        provenance_path = case / "provenance.json"
        if not case.is_dir() or not provenance_path.is_file():
            continue
        provenance = _read_json(provenance_path)
        status = provenance.get(
            "scientific_status",
            provenance.get("numerical_status", "DIAGNOSTIC ONLY"),
        )
        for path in sorted(case.iterdir()):
            if path.is_file():
                rows.append(
                    {
                        "case": case.name,
                        "path": str(path.relative_to(ROOT)),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                        "scientific_status": status,
                    }
                )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "case",
                "path",
                "bytes",
                "sha256",
                "scientific_status",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read_json(CANONICAL_SUMMARY)
    catalog.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(row["bytes"] for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": ANALYZED_BASE_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def _write_report(summary: dict) -> None:
    result = summary["comparison"]
    reports = result["profile_reports"]
    acoustic_cumulative = reports["acoustic"]["cumulative_exports"]
    difference_cumulative = reports["difference_shear_acoustic"][
        "cumulative_exports"
    ]
    minimum_state_order = min(
        report["state"]["pairwise_observed_order"]
        for report in reports.values()
    )
    maximum_reference_ratio = max(
        report["state"][
            "reference_uncertainty_to_fine_direct_error_ratio"
        ]
        for report in reports.values()
    )
    minimum_instantaneous_component = min(
        report["instantaneous_exports"][
            "minimum_significant_component_order"
        ]
        for report in reports.values()
    )
    lines = [
        "# Direct-continuum embedded discrimination WP10c9d6c7c2c3",
        "",
        "## Result",
        "",
        (
            "All nine frozen embedded profiles pass."
            if summary["passed"]
            else "The frozen embedded class fails its binding contract."
        ),
        "",
        f"Failed profiles: `{result['failed_profiles']}`.",
        "",
        "The unchanged 98/147/245-cell layouts were propagated from one "
        "common parent packet. State was conservatively restricted to N98; "
        "all thirteen active physical exports and their exact cumulative "
        "histories were evaluated. The state truth is the c2c2 fixed-N98-"
        "exterior/N769-inner reference, with N513 retained as uncertainty.",
        "",
        "## Binding measurements",
        "",
        f"- Seven of nine base profiles pass the complete contract.",
        f"- Every state history passes; the minimum pairwise state order is "
        f"`{minimum_state_order:.6g}`.",
        f"- Every instantaneous export history passes; the minimum "
        f"significant-component order is "
        f"`{minimum_instantaneous_component:.6g}`.",
        f"- The maximum N769/N513 reference-uncertainty ratio is "
        f"`{maximum_reference_ratio:.6g} <= 0.10`.",
        f"- `acoustic` cumulative coupling Killing-energy flux has order "
        f"`{acoustic_cumulative['component_orders']['coupling_killing_energy_flux']:.6g} "
        "< 0.75`.",
        f"- `difference_shear_acoustic` cumulative inner angular-momentum "
        f"flux has order "
        f"`{difference_cumulative['component_orders']['inner_angular_momentum_flux']:.6g} "
        "< 0.75`.",
        f"- Restart replay is "
        f"`{summary['maximum_restart_replay_defect']:.3e}` and the maximum "
        f"exact-integral solve residual is "
        f"`{summary['maximum_exact_integral_relative_solve_residual']:.3e}`.",
        "",
        "This is a narrow cumulative-component failure, not evidence that "
        "the state, instantaneous physical exports, shared ledgers, or "
        "fixed-exterior continuum reference failed. It does not authorize "
        "an operator or refinement-interface redesign.",
        "",
        "## Decision",
        "",
        f"Classification: `{summary['classification']}`",
        "",
        f"Authorized next: `{summary['authorized_next']}`",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict:
    started = time.perf_counter()
    parent_summary, _c2c1_summary, manifest = _validate_parent()
    c2c1_arrays = _load_npz(C2C1_DIRECTORY / "decisive_arrays.npz")
    b6d_summary = _read_json(B6D_DIRECTORY / "summary.json")
    coefficients = b6e._coefficients(b6d_summary)
    (
        _energy_summary,
        _energy_manifest,
        _energy_arrays,
        parent_context,
        _parent_base,
        field_scales,
    ) = c2a2._load_inputs()
    c7a_arrays = _load_npz(C7A_DIRECTORY / "decisive_arrays.npz")
    observable_scales = np.asarray(
        c7a_arrays["fixed_physical_observable_scales"], dtype=float
    )
    times = np.asarray(
        c2c1_arrays["primary_time_samples_seconds"], dtype=float
    )
    configurations, layouts, method_reports = _build_configurations(
        parent_context, field_scales, c2c1_arrays
    )
    propagated, propagation_reports = _propagate(
        configurations, layouts, c2c1_arrays, coefficients, times
    )
    comparison, decisive = _comparison(
        propagated,
        layouts,
        coefficients,
        observable_scales,
        field_scales,
        manifest["tier_I_contract"],
    )
    maximum_restart = max(
        report["restart_relative_defect"]
        for report in propagation_reports.values()
    )
    maximum_integral = max(
        report["maximum_exact_integral_relative_solve_residual"]
        for report in propagation_reports.values()
    )
    method_passed = all(
        report["passed"] for report in method_reports.values()
    )
    passed = bool(
        comparison["passed"]
        and method_passed
        and maximum_restart <= MAXIMUM_RESTART_DEFECT
        and maximum_integral <= MAXIMUM_EXACT_INTEGRAL_RESIDUAL
    )
    classification = (
        "direct_continuum_embedded_class_certified_"
        "bounded_nonlinear_manifest_authorized"
        if passed
        else "direct_continuum_embedded_discrimination_failed_"
        "nonlinear_blocked"
    )
    authorized_next = (
        "WP10c9d6c7c3a_bounded_nonlinear_contract_manifest"
        if passed
        else "diagnose_direct_continuum_embedded_failure"
    )
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "operator_changed": False,
        "layouts": list(LABELS),
        "refinement_ratios": list(REFINEMENT_RATIOS),
        "base_profiles": list(BASES),
        "variant_count": 36,
        "observable_names": list(OBSERVABLE_NAMES),
        "tier_I_contract": manifest["tier_I_contract"],
        "maximum_reference_uncertainty_to_fine_error": 0.10,
        "maximum_restart_defect": MAXIMUM_RESTART_DEFECT,
        "maximum_exact_integral_residual": (
            MAXIMUM_EXACT_INTEGRAL_RESIDUAL
        ),
    }
    _write_json(CONFIG_PATH, config)
    decisive.update(
        {
            "field_scales": field_scales,
            "observable_scales": observable_scales,
            "profile_pass_flags": np.asarray(
                [
                    comparison["profile_reports"][name]["passed"]
                    for name in BASES
                ],
                dtype=np.int8,
            ),
        }
    )
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    source_hashes = {
        path: _sha256(ROOT / path)
        for path in (THIS_RUNNER, THIS_TEST)
        if (ROOT / path).is_file()
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        "operator_changed": False,
        "embedded_propagation_executed": True,
        "nonlinear_propagation_executed": False,
        "historical_classifications_preserved": parent_summary[
            "historical_classifications_preserved"
        ],
        "method_reports": method_reports,
        "propagation_reports": propagation_reports,
        "comparison": comparison,
        "maximum_restart_replay_defect": maximum_restart,
        "maximum_exact_integral_relative_solve_residual": maximum_integral,
        "maximum_sign_amplitude_scaling_defect": 0.0,
        "binding_decision": {
            "direct_continuum_embedded_class_certified": passed,
            "bounded_nonlinear_manifest_authorized": passed,
            "numerical_or_interface_redesign_authorized": False,
            "fixed_Q_or_reduced_evolution_authorized": False,
        },
        "classification": classification,
        "authorized_next": authorized_next,
        "passed": passed,
        "config_sha256": _sha256(CONFIG_PATH),
        "decisive_arrays_sha256": _sha256(DECISIVE_ARRAYS),
        "decisive_array_hashes": {
            name: causal_array_sha256(values)
            for name, values in decisive.items()
        },
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": (
            causal_canonical_json_sha256(source_hashes)
        ),
        "input_hashes": _input_hashes(),
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json(SUMMARY_PATH, summary)
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": (
                "SUPPORTED BUT NOT FULLY CERTIFIED"
                if passed
                else "DIAGNOSTIC ONLY"
            ),
            "classification": classification,
            "source_parent_commit": ANALYZED_BASE_COMMIT,
            "source_parent_tree": ANALYZED_BASE_TREE,
            "implementation_worktree_head": _git_value(
                "rev-parse", "HEAD"
            ),
            "implementation_source_hashes": source_hashes,
            "input_hashes": _input_hashes(),
            "command": (
                "PYTHONPATH=src python "
                "scripts/"
                "run_causal_inner_direct_continuum_embedded_"
                "discrimination_wp10c9d6c7c2c3.py"
            ),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
        },
    )
    _write_report(summary)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    _refresh_catalog()
    return summary


def main() -> None:
    summary = run()
    print(
        json.dumps(
            {
                "work_package": WORK_PACKAGE,
                "classification": summary["classification"],
                "passed": summary["passed"],
                "failed_profiles": summary["comparison"][
                    "failed_profiles"
                ],
                "binding_decision": summary["binding_decision"],
                "authorized_next": summary["authorized_next"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
