#!/usr/bin/env python3
"""Calibrate the complete discrete BDF tangent from committed trajectories."""

from __future__ import annotations

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


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_monolithic_bdf_base_preflight_wp10c9d6c7c3b1a as b1a  # noqa: E402
import run_causal_inner_nonlinear_profile_breadth_coarse_screen_wp10c9d6c7c3b4b1 as b4b1  # noqa: E402
import run_causal_inner_nonlinear_profile_breadth_spatial_wp10c9d6c7c3b4b3 as b4b3  # noqa: E402
import run_causal_inner_nonlinear_profile_breadth_export_face_audit_wp10c9d6c7c3b4d as b4d  # noqa: E402
import run_causal_inner_nonlinear_coarse_third_duration_rung_completion_wp10c9d6c7c3b5c3d as c3d  # noqa: E402
import run_causal_inner_nonlinear_coarse_heldout_third_duration_rung_screen_wp10c9d6c7c3b5c3f as c3f  # noqa: E402
import run_causal_inner_nonlinear_third_duration_rung_cost_bounded_spatial_confirmation_manifest_wp10c9d6c7c3b5c3g1 as c3g1  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_monolithic_discrete_tangent import (  # noqa: E402
    causal_five_field_monolithic_bdf_history_direction,
    causal_five_field_monolithic_bdf_history_from_interval,
    causal_five_field_monolithic_discrete_export_directions,
    causal_five_field_monolithic_discrete_step_matrix,
    causal_five_field_monolithic_discrete_tangent_step,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c3h1"
ANALYZED_BASE_COMMIT = "a868dc3c52c860399bd2af36d0f17f1d3fa5cac9"
ANALYZED_BASE_PARENT = "10f3665fb106843c6618a0e1e6e3ac61ace62d27"
ANALYZED_BASE_TREE = "5ffb9e1cba80215568fff338d467110ded3ba3da"

PROFILES = tuple(c3g1.PROFILE_NAMES)
GENERIC_PROFILE = c3g1.c3g.GENERIC_PROFILE
LAYOUTS = tuple(c3g1.LAYOUTS)
COARSE_LAYOUT, MIDDLE_LAYOUT, FINE_LAYOUT = LAYOUTS
GATES = dict(c3g1.TANGENT_GATES)
GATES.update(
    {
        "maximum_base_scaled_residual": 1.0e-10,
        "maximum_linear_solve_relative_defect": 1.0e-10,
        "maximum_matrix_component_closure_defect": 1.0e-12,
        "maximum_export_transport_telescoping_defect": 1.0e-12,
        "maximum_export_active_prefix_ledger_defect": 1.0e-12,
        "maximum_incoming_excision_characteristics": 0,
    }
)

ARTIFACT = (
    "causal_inner_nonlinear_discrete_bdf_tangent_calibration_"
    "wp10c9d6c7c3b5c3h1"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_discrete_bdf_tangent_"
    "calibration_wp10c9d6c7c3b5c3h1.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_discrete_bdf_tangent_"
    "calibration_wp10c9d6c7c3b5c3h1.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_DISCRETE_BDF_"
    "TANGENT_CALIBRATION_WP10C9D6C7C3B5C3H1_2026-08-05.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
MODULE_RELATIVE = (
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_monolithic_discrete_tangent.py"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
CHECKPOINT_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
CHECKPOINT_JSON = CHECKPOINT_DIRECTORY / "progress.json"
CHECKPOINT_ARRAYS = CHECKPOINT_DIRECTORY / "progress_arrays.npz"


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


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_parent() -> dict:
    parent = _read_json(c3g1.SUMMARY_PATH)
    if (
        not parent["passed"]
        or parent["classification"]
        != "third_duration_rung_cost_bounded_spatial_confirmation_manifest_"
        "frozen_discrete_bdf_tangent_calibration_authorized"
        or not parent["discrete_BDF_tangent_calibration_authorized"]
        or parent["middle_cost_bounded_propagation_authorized"]
    ):
        raise RuntimeError("h1 calibration authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_BASE_COMMIT) != ANALYZED_BASE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
        != ANALYZED_BASE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
        != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("h1 analyzed identity changed")
    return parent


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=float).ravel()
    b = np.asarray(right, dtype=float).ravel()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= np.finfo(float).tiny:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.dot(a, b) / denominator)


def _response_metrics(
    predicted: np.ndarray,
    actual: np.ndarray,
    scales: np.ndarray,
) -> dict:
    scale_shape = (1,) * (predicted.ndim - 1) + (scales.size,)
    predicted_scaled = np.asarray(predicted, dtype=float) / scales.reshape(scale_shape)
    actual_scaled = np.asarray(actual, dtype=float) / scales.reshape(scale_shape)
    difference = predicted_scaled - actual_scaled
    return {
        "maximum_scaled_discrepancy": float(np.max(np.abs(difference))),
        "rms_scaled_discrepancy": float(np.sqrt(np.mean(difference * difference))),
        "history_cosine": _cosine(predicted_scaled, actual_scaled),
        "maximum_scaled_predicted_response": float(
            np.max(np.abs(predicted_scaled))
        ),
        "maximum_scaled_actual_response": float(np.max(np.abs(actual_scaled))),
    }


def _actual_long_histories() -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    generic = _load_npz(c3d.DECISIVE_ARRAYS)
    heldout = _load_npz(c3f.DECISIVE_ARRAYS)
    base_states = generic["base__main__output_states"]
    base_exports = generic["base__main__output_exports"]
    states = {
        GENERIC_PROFILE: generic["perturbed__main__output_states"],
    }
    exports = {
        GENERIC_PROFILE: generic["perturbed__main__output_exports"],
    }
    for profile in PROFILES[1:]:
        states[profile] = heldout[f"{profile}__main__output_states"]
        exports[profile] = heldout[f"{profile}__main__output_exports"]
    return (
        base_states,
        base_exports,
        states,
        exports,
        generic["main_times_seconds"],
        generic["field_scales"],
    )


def _long_tail_calibration() -> tuple[dict, dict[str, np.ndarray]]:
    (
        base_states,
        base_exports,
        perturbed_states,
        perturbed_exports,
        times,
        field_scales,
    ) = _actual_long_histories()
    generic = _load_npz(c3d.DECISIVE_ARRAYS)
    export_scales = generic["export_scales"]
    configuration = c3d.c3b1a._configurations()[c3d.c2.LAYOUT]
    context = configuration["context"]
    columns = configuration["columns"]
    rows = configuration["rows"]
    coupling_face = c3g1.c3g.ACTIVE_COUPLING_FACE_INDICES[COARSE_LAYOUT]

    actual_response = np.asarray(
        [perturbed_states[name] - base_states for name in PROFILES],
        dtype=float,
    )
    actual_export_response = np.asarray(
        [perturbed_exports[name] - base_exports for name in PROFILES],
        dtype=float,
    )
    start = 1
    previous_dt = float(times[start] - times[start - 1])
    base_history = causal_five_field_monolithic_bdf_history_from_interval(
        context,
        base_states[start - 1],
        base_states[start],
        previous_dt,
    )
    history_directions = causal_five_field_monolithic_bdf_history_direction(
        context,
        base_states[start - 1],
        base_states[start],
        actual_response[:, start - 1],
        actual_response[:, start],
    )
    directions = np.asarray(actual_response[:, start], dtype=float)
    predicted_states = [np.array(directions, copy=True)]
    predicted_exports = []
    matrix_seconds = []
    step_seconds = []
    component_defects = []
    linear_defects = []
    base_residuals = []
    jvp_defects = []
    incoming = []
    export_telescoping_defects = []
    export_prefix_ledger_defects = []

    for target in range(start + 1, len(times)):
        old = target - 1
        dt = float(times[target] - times[old])
        began = time.perf_counter()
        matrix = causal_five_field_monolithic_discrete_step_matrix(
            context,
            base_states[old],
            base_states[target],
            dt,
            previous_dt,
            primitive_column_scales=columns,
            conservation_row_scales=rows,
        )
        matrix_seconds.append(time.perf_counter() - began)
        began = time.perf_counter()
        step = causal_five_field_monolithic_discrete_tangent_step(
            context,
            base_states[old],
            base_states[target],
            dt,
            base_history,
            directions,
            history_directions,
            primitive_column_scales=columns,
            conservation_row_scales=rows,
            analytic_step_matrix=matrix,
            audit_complete_residual=(target == start + 1),
        )
        step_seconds.append(time.perf_counter() - began)
        directions = step.new_primitive_directions
        history_directions = step.new_history_directions
        base_history = causal_five_field_monolithic_bdf_history_from_interval(
            context,
            base_states[old],
            base_states[target],
            dt,
        )
        previous_dt = dt
        predicted_states.append(np.array(directions, copy=True))
        export_direction, export_audit = (
            causal_five_field_monolithic_discrete_export_directions(
                matrix,
                directions,
                coupling_face,
            )
        )
        predicted_exports.append(export_direction)
        export_telescoping_defects.append(
            export_audit.conservative_transport_telescoping_defect
        )
        export_prefix_ledger_defects.append(
            export_audit.active_prefix_ledger_defect
        )
        component_defects.append(matrix.maximum_component_closure_defect)
        linear_defects.append(step.maximum_linear_solve_relative_defect)
        incoming.append(matrix.incoming_excision_characteristics)
        if np.isfinite(step.maximum_base_scaled_residual):
            base_residuals.append(step.maximum_base_scaled_residual)
        if np.isfinite(step.maximum_step_matrix_jvp_relative_defect):
            jvp_defects.append(step.maximum_step_matrix_jvp_relative_defect)
        print(
            f"h1: long tail {times[target]:.4e} s "
            f"matrix={matrix_seconds[-1]:.1f}s step={step_seconds[-1]:.1f}s",
            flush=True,
        )

    predicted_state_array = np.moveaxis(np.asarray(predicted_states), 1, 0)
    predicted_export_array = np.moveaxis(np.asarray(predicted_exports), 1, 0)
    actual_state_array = actual_response[:, start:]
    actual_export_array = actual_export_response[:, start + 1 :]
    long_times = times[start:]
    export_times = times[start + 1 :]
    predicted_cumulative = np.zeros_like(predicted_export_array)
    actual_cumulative = np.zeros_like(actual_export_array)
    for index in range(1, export_times.size):
        dt = float(export_times[index] - export_times[index - 1])
        predicted_cumulative[:, index] = (
            predicted_cumulative[:, index - 1]
            + 0.5
            * dt
            * (
                predicted_export_array[:, index - 1]
                + predicted_export_array[:, index]
            )
        )
        actual_cumulative[:, index] = (
            actual_cumulative[:, index - 1]
            + 0.5
            * dt
            * (
                actual_export_array[:, index - 1]
                + actual_export_array[:, index]
            )
        )
    state_metrics = _response_metrics(
        predicted_state_array,
        actual_state_array,
        field_scales,
    )
    export_metrics = _response_metrics(
        predicted_export_array,
        actual_export_array,
        export_scales,
    )
    cumulative_metrics = _response_metrics(
        predicted_cumulative,
        actual_cumulative,
        export_scales,
    )
    report = {
        "tail_start_seconds": float(long_times[0]),
        "tail_stop_seconds": float(long_times[-1]),
        "profiles": PROFILES,
        "steps": int(long_times.size - 1),
        "state": state_metrics,
        "instantaneous_Tier_I": export_metrics,
        "windowed_cumulative_Tier_I": cumulative_metrics,
        "maximum_base_scaled_residual": max(base_residuals, default=0.0),
        "maximum_step_matrix_jvp_relative_defect": max(jvp_defects, default=0.0),
        "maximum_linear_solve_relative_defect": max(linear_defects, default=0.0),
        "maximum_matrix_component_closure_defect": max(component_defects, default=0.0),
        "maximum_incoming_excision_characteristics": max(incoming, default=0),
        "maximum_export_transport_telescoping_defect": max(
            export_telescoping_defects,
            default=0.0,
        ),
        "maximum_export_active_prefix_ledger_defect": max(
            export_prefix_ledger_defects,
            default=0.0,
        ),
        "matrix_assembly_wall_seconds": matrix_seconds,
        "block_step_wall_seconds": step_seconds,
    }
    arrays = {
        "long_times_seconds": long_times,
        "long_export_times_seconds": export_times,
        "long_predicted_state_response": predicted_state_array,
        "long_actual_state_response": actual_state_array,
        "long_predicted_instantaneous_Tier_I_response": predicted_export_array,
        "long_actual_instantaneous_Tier_I_response": actual_export_array,
        "long_predicted_cumulative_Tier_I_response": predicted_cumulative,
        "long_actual_cumulative_Tier_I_response": actual_cumulative,
        "long_matrix_assembly_wall_seconds": np.asarray(matrix_seconds),
        "long_block_step_wall_seconds": np.asarray(step_seconds),
    }
    return report, arrays


def _short_perturbed_arrays(layout: str) -> tuple[np.ndarray, np.ndarray]:
    spatial = _load_npz(b4b3.DECISIVE_ARRAYS)
    corrected = _load_npz(b4d.DECISIVE_ARRAYS)
    if layout == COARSE_LAYOUT:
        source = _load_npz(b4b1.DECISIVE_ARRAYS)
        prefix = b4b1._task_id(GENERIC_PROFILE)
    else:
        source = spatial
        prefix = f"{layout}__{GENERIC_PROFILE}__p1__dt_1e-5"
    return (
        source[f"{prefix}__states"],
        corrected[
            f"{layout}__{GENERIC_PROFILE}__corrected_face_response"
        ],
    )


def _short_layout_calibration(
    layout: str,
    configuration: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    base_arrays = _load_npz(b1a.DECISIVE_ARRAYS)
    context = configuration["context"]
    columns = configuration["columns"]
    rows = configuration["rows"]
    base_states = base_arrays[f"{layout}__states"]
    perturbed_states, export_response = _short_perturbed_arrays(layout)
    response = perturbed_states - base_states
    previous_dt = 1.0e-5
    dt = 1.0e-5
    base_history = causal_five_field_monolithic_bdf_history_from_interval(
        context,
        base_states[0],
        base_states[1],
        previous_dt,
    )
    history_direction = causal_five_field_monolithic_bdf_history_direction(
        context,
        base_states[0],
        base_states[1],
        response[0:1],
        response[1:2],
    )
    began = time.perf_counter()
    matrix = causal_five_field_monolithic_discrete_step_matrix(
        context,
        base_states[1],
        base_states[2],
        dt,
        previous_dt,
        primitive_column_scales=columns,
        conservation_row_scales=rows,
    )
    matrix_seconds = time.perf_counter() - began
    began = time.perf_counter()
    step = causal_five_field_monolithic_discrete_tangent_step(
        context,
        base_states[1],
        base_states[2],
        dt,
        base_history,
        response[1:2],
        history_direction,
        primitive_column_scales=columns,
        conservation_row_scales=rows,
        analytic_step_matrix=matrix,
        audit_complete_residual=False,
    )
    step_seconds = time.perf_counter() - began
    predicted_export, export_audit = (
        causal_five_field_monolithic_discrete_export_directions(
            matrix,
            step.new_primitive_directions,
            c3g1.c3g.ACTIVE_COUPLING_FACE_INDICES[layout],
        )
    )
    field_scales = _load_npz(c3d.DECISIVE_ARRAYS)["field_scales"]
    export_scales = _load_npz(c3d.DECISIVE_ARRAYS)["export_scales"]
    state_metrics = _response_metrics(
        step.new_primitive_directions,
        response[2:3],
        field_scales,
    )
    export_metrics = _response_metrics(
        predicted_export,
        export_response[2:3],
        export_scales,
    )
    report = {
        "layout": layout,
        "state": state_metrics,
        "instantaneous_Tier_I": export_metrics,
        "maximum_linear_solve_relative_defect": (
            step.maximum_linear_solve_relative_defect
        ),
        "maximum_matrix_component_closure_defect": (
            matrix.maximum_component_closure_defect
        ),
        "maximum_incoming_excision_characteristics": (
            matrix.incoming_excision_characteristics
        ),
        "maximum_export_transport_telescoping_defect": (
            export_audit.conservative_transport_telescoping_defect
        ),
        "maximum_export_active_prefix_ledger_defect": (
            export_audit.active_prefix_ledger_defect
        ),
        "matrix_assembly_wall_seconds": matrix_seconds,
        "block_step_wall_seconds": step_seconds,
    }
    arrays = {
        f"{layout}__short_predicted_state_response": (
            step.new_primitive_directions[0]
        ),
        f"{layout}__short_actual_state_response": response[2],
        f"{layout}__short_predicted_Tier_I_response": predicted_export[0],
        f"{layout}__short_actual_Tier_I_response": export_response[2],
    }
    print(
        f"h1: short {layout} matrix={matrix_seconds:.1f}s "
        f"step={step_seconds:.3f}s",
        flush=True,
    )
    return report, arrays


def _save_progress(progress: dict, arrays: dict[str, np.ndarray]) -> None:
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(CHECKPOINT_JSON, progress)
    np.savez_compressed(CHECKPOINT_ARRAYS, **arrays)


def _load_progress() -> tuple[dict, dict[str, np.ndarray]]:
    if not CHECKPOINT_JSON.exists() or not CHECKPOINT_ARRAYS.exists():
        return {"long": None, "short": {}}, {}
    return _read_json(CHECKPOINT_JSON), _load_npz(CHECKPOINT_ARRAYS)


def _refresh_corrected_short_export_references(
    progress: dict,
    arrays: dict[str, np.ndarray],
) -> None:
    """Replace the historical wrong-face short export references."""

    if not progress.get("short"):
        return
    corrected = _load_npz(b4d.DECISIVE_ARRAYS)
    export_scales = _load_npz(c3d.DECISIVE_ARRAYS)["export_scales"]
    for layout, report in progress["short"].items():
        predicted_key = f"{layout}__short_predicted_Tier_I_response"
        actual_key = f"{layout}__short_actual_Tier_I_response"
        actual = corrected[
            f"{layout}__{GENERIC_PROFILE}__corrected_face_response"
        ][2]
        arrays[actual_key] = np.asarray(actual, dtype=float)
        report["instantaneous_Tier_I"] = _response_metrics(
            arrays[predicted_key][None, :],
            arrays[actual_key][None, :],
            export_scales,
        )
        report["export_reference"] = (
            "WP10c9d6c7c3b4d_corrected_active_coupling_face"
        )


def _passes(report: dict) -> bool:
    metrics = [
        report["long"]["state"],
        report["long"]["instantaneous_Tier_I"],
        report["long"]["windowed_cumulative_Tier_I"],
    ]
    for row in report["short"].values():
        metrics.extend((row["state"], row["instantaneous_Tier_I"]))
    state_metrics = [report["long"]["state"]] + [
        row["state"] for row in report["short"].values()
    ]
    export_metrics = [
        report["long"]["instantaneous_Tier_I"],
        report["long"]["windowed_cumulative_Tier_I"],
    ] + [row["instantaneous_Tier_I"] for row in report["short"].values()]
    method_rows = [report["long"], *report["short"].values()]
    return bool(
        max(row["maximum_scaled_discrepancy"] for row in state_metrics)
        <= GATES["maximum_scaled_state_response_discrepancy"]
        and max(row["maximum_scaled_discrepancy"] for row in export_metrics)
        <= GATES["maximum_scaled_Tier_I_response_discrepancy"]
        and min(row["history_cosine"] for row in state_metrics)
        >= GATES["minimum_state_response_history_cosine"]
        and min(row["history_cosine"] for row in export_metrics)
        >= GATES["minimum_Tier_I_response_history_cosine"]
        and report["long"]["maximum_step_matrix_jvp_relative_defect"]
        <= GATES["maximum_internal_discrete_residual_jvp_relative_defect"]
        and report["long"]["maximum_base_scaled_residual"]
        <= GATES["maximum_base_scaled_residual"]
        and max(
            row["maximum_linear_solve_relative_defect"] for row in method_rows
        )
        <= GATES["maximum_linear_solve_relative_defect"]
        and max(
            row["maximum_matrix_component_closure_defect"]
            for row in method_rows
        )
        <= GATES["maximum_matrix_component_closure_defect"]
        and max(
            row["maximum_incoming_excision_characteristics"]
            for row in method_rows
        )
        <= GATES["maximum_incoming_excision_characteristics"]
        and max(
            row["maximum_export_transport_telescoping_defect"]
            for row in method_rows
        )
        <= GATES["maximum_export_transport_telescoping_defect"]
        and max(
            row["maximum_export_active_prefix_ledger_defect"]
            for row in method_rows
        )
        <= GATES["maximum_export_active_prefix_ledger_defect"]
        and all(np.isfinite(row["maximum_scaled_discrepancy"]) for row in metrics)
    )


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
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
                    "sha256": _sha256(path),
                    "scientific_status": (
                        "CERTIFIED" if summary["passed"] else "REJECTED"
                    ),
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case", "path", "bytes", "sha256", "scientific_status"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read_json(CANONICAL_SUMMARY)
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
            "latest_source_parent_commit": ANALYZED_BASE_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def _report(summary: dict) -> str:
    long = summary["calibration"]["long"]
    short = summary["calibration"]["short"]
    lines = [
        "# Discrete BDF tangent calibration WP10c9d6c7c3b5c3h1",
        "",
        "## Classification",
        "",
        f"`{summary['classification']}`",
        "",
        "The complete analytic variable-step BDF tangent includes the new and old storage-path endpoints, primitive history, mapped-storage history, responsive-height history, and the monolithic stationary residual. It changes no production operator or integration default and launches no new physical trajectory.",
        "",
        "## Long-tail calibration",
        "",
        f"Five committed nonlinear responses are propagated together from `{long['tail_start_seconds']:.4g}` to `{long['tail_stop_seconds']:.4g} s`. The maximum scaled state discrepancy is `{long['state']['maximum_scaled_discrepancy']:.3e}`; instantaneous and cumulative Tier-I discrepancies are `{long['instantaneous_Tier_I']['maximum_scaled_discrepancy']:.3e}` and `{long['windowed_cumulative_Tier_I']['maximum_scaled_discrepancy']:.3e}`. The independent complete-residual JVP defect is `{long['maximum_step_matrix_jvp_relative_defect']:.3e}`.",
        "",
        "## Three-layout short-horizon calibration",
        "",
    ]
    for layout in LAYOUTS:
        row = short[layout]
        lines.append(
            f"- `{layout}`: state `{row['state']['maximum_scaled_discrepancy']:.3e}`, Tier-I `{row['instantaneous_Tier_I']['maximum_scaled_discrepancy']:.3e}`, matrix `{row['matrix_assembly_wall_seconds']:.1f} s`."
        )
    lines.extend(
        [
            "",
            "The short-layout Tier-I reference is the certified WP10c9d6c7c3b4d corrected active-coupling-face response. The superseded WP10c9d6c7c3b4b3 wrong-face response is retained as historical negative evidence and is not used for calibration.",
            "",
            "A pass authorizes only a definitions-only middle cost-bounded anchor manifest. Middle/fine propagation, the fourth duration rung, fixed-Q experiments, and reduced slow evolution remain blocked.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    started = time.perf_counter()
    parent = _validate_parent()
    progress, arrays = _load_progress()
    _refresh_corrected_short_export_references(progress, arrays)
    if progress["long"] is None:
        print("h1: calibrate committed 2.4-5.0 ms tail", flush=True)
        long_report, long_arrays = _long_tail_calibration()
        progress["long"] = long_report
        arrays.update(long_arrays)
        _save_progress(progress, arrays)
    short_configurations = (
        b1a._configurations()
        if any(layout not in progress["short"] for layout in LAYOUTS)
        else {}
    )
    for layout in LAYOUTS:
        if layout in progress["short"]:
            continue
        print(f"h1: calibrate short {layout}", flush=True)
        short_report, short_arrays = _short_layout_calibration(
            layout,
            short_configurations[layout],
        )
        progress["short"][layout] = short_report
        arrays.update(short_arrays)
        _save_progress(progress, arrays)

    calibration = {
        "long": progress["long"],
        "short": progress["short"],
    }
    passed = _passes(calibration)
    classification = (
        "complete_discrete_BDF_tangent_calibrated_middle_cost_bounded_"
        "anchor_manifest_authorized"
        if passed
        else "discrete_BDF_tangent_calibration_failed_bruteforce_c3g_only"
    )
    authorized_next = (
        "WP10c9d6c7c3b5c3h2_middle_cost_bounded_anchor_manifest"
        if passed
        else "historical_c3g_bruteforce_route_only"
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "operator_changed": False,
        "production_defaults_changed": False,
        "new_physical_trajectory_executed": False,
        "parent_classification_preserved": parent["classification"],
        "calibration": calibration,
        "gates": GATES,
        "analytic_complete_discrete_BDF_tangent_certified": passed,
        "middle_cost_bounded_anchor_manifest_authorized": passed,
        "middle_cost_bounded_propagation_authorized": False,
        "fine_cost_bounded_propagation_authorized": False,
        "third_duration_rung_spatial_convergence_certified": False,
        "fourth_duration_rung_manifest_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": authorized_next,
        "wall_seconds": time.perf_counter() - started,
    }
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "profiles": PROFILES,
        "layouts": LAYOUTS,
        "long_tail_start_index": 1,
        "long_tail_start_seconds": 2.4e-3,
        "long_tail_stop_seconds": 5.0e-3,
        "short_calibration_step": "10_to_20_microseconds",
        "gates": GATES,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DECISIVE_ARRAYS, **arrays)
    _write_json(CONFIG_PATH, config)
    _write_json(SUMMARY_PATH, summary)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_base_commit": ANALYZED_BASE_COMMIT,
        "analyzed_base_parent": ANALYZED_BASE_PARENT,
        "analyzed_base_tree": ANALYZED_BASE_TREE,
        "working_head": _git_value("rev-parse", "HEAD"),
        "working_tree": _git_value("rev-parse", "HEAD^{tree}"),
        "runner": THIS_RUNNER,
        "test": THIS_TEST,
        "report": REPORT_RELATIVE,
        "input_hashes": {
            "parent_summary": _sha256(c3g1.SUMMARY_PATH),
            "base_preflight_arrays": _sha256(b1a.DECISIVE_ARRAYS),
            "coarse_breadth_arrays": _sha256(b4b1.DECISIVE_ARRAYS),
            "three_layout_breadth_arrays": _sha256(b4b3.DECISIVE_ARRAYS),
            "corrected_export_face_arrays": _sha256(b4d.DECISIVE_ARRAYS),
            "coarse_generic_5ms_arrays": _sha256(c3d.DECISIVE_ARRAYS),
            "coarse_heldout_5ms_arrays": _sha256(c3f.DECISIVE_ARRAYS),
        },
        "implementation_source_hashes": {
            "runner": _sha256(ROOT / THIS_RUNNER),
            "module": _sha256(ROOT / MODULE_RELATIVE),
            "test": _sha256(ROOT / THIS_TEST),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "command": (
            "PYTHONPATH=src:scripts python scripts/"
            "run_causal_inner_nonlinear_discrete_bdf_tangent_"
            "calibration_wp10c9d6c7c3b5c3h1.py"
        ),
    }
    _write_json(PROVENANCE_PATH, provenance)
    REPORT_PATH.write_text(_report(summary), encoding="utf-8")
    names = (
        "config.json",
        "decisive_arrays.npz",
        "provenance.json",
        "summary.json",
    )
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    print(classification)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
