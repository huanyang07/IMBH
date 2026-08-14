#!/usr/bin/env python3
"""Run the fine-only six-mode dynamic-coordinate replay.

The committed middle trajectory is reused.  This package advances six
tangent directions through the committed fine nonlinear base history; it
does not advance a nonlinear state or apply a fixed-Q reaction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_face36_augmented_memory_screen_wp10c9d6c7c3b5c4f13 as c4f13  # noqa: E402
import run_causal_inner_face36_q_plus_a_reaction_coordinate_preflight_wp10c9d6c7c3b5c4f15 as c4f15  # noqa: E402
import run_causal_inner_face36_six_mode_dynamic_coordinate_replay_wp10c9d6c7c3b5c4f17 as c4f17  # noqa: E402
import run_causal_inner_face36_six_mode_numerical_audit_recovery_wp10c9d6c7c3b5c4f18 as c4f18  # noqa: E402
import run_causal_inner_face36_six_mode_fine_replay_manifest_wp10c9d6c7c3b5c4f19 as c4f19  # noqa: E402

from imri_qpe.layer3_minidisk_1d.causal_inner_embedded_patch import (  # noqa: E402
    restrict_causal_embedded_patch_cell_averages,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_monolithic_discrete_tangent import (  # noqa: E402
    CausalFiveFieldMonolithicBDFHistoryDirection,
    causal_five_field_monolithic_bdf_history_direction,
    causal_five_field_monolithic_discrete_step_matrix,
    causal_five_field_monolithic_discrete_tangent_step,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f20"
ARTIFACT = (
    "causal_inner_face36_six_mode_fine_dynamic_coordinate_replay_"
    "wp10c9d6c7c3b5c4f20"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_face36_six_mode_fine_dynamic_coordinate_replay_"
    "wp10c9d6c7c3b5c4f20.py"
)
THIS_TEST = (
    "tests/test_causal_inner_face36_six_mode_fine_dynamic_coordinate_replay_"
    "wp10c9d6c7c3b5c4f20.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/"
    "CODEX_CAUSAL_INNER_FACE36_SIX_MODE_FINE_DYNAMIC_COORDINATE_REPLAY_"
    "WP10C9D6C7C3B5C4F20_2026-08-13.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
CHECKPOINT_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
CHECKPOINT_PATH = CHECKPOINT_DIRECTORY / "fine.npz"
PROGRESS_PATH = CHECKPOINT_DIRECTORY / "fine.json"

MODE_DIMENSION = c4f19.MODE_DIMENSION
LEADING_DIMENSION = c4f19.LEADING_DIMENSION
AUDIT_TIME_IDS_MICROSECONDS = c4f19.AUDIT_TIME_IDS_MICROSECONDS
SELECTED_RELATIVE_STEPS = np.asarray(c4f19.SELECTED_RELATIVE_STEPS)
STAGES = ("preflight", "complete")


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


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        return {name: np.asarray(arrays[name]) for name in arrays.files}


def _save(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_identity() -> dict[str, str]:
    paths = (
        THIS_RUNNER,
        THIS_TEST,
        c4f19.THIS_RUNNER,
        c4f19.THIS_TEST,
        c4f18.THIS_RUNNER,
        c4f17.THIS_RUNNER,
        c4f15.THIS_RUNNER,
        c4f13.THIS_RUNNER,
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_discrete_tangent.py",
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_tangent.py",
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_radial_fluctuation.py",
    )
    return {path: _sha(ROOT / path) for path in paths if (ROOT / path).exists()}


def _authorization() -> dict:
    summary = _read(c4f19.SUMMARY_PATH)
    manifest = _read(c4f19.MANIFEST_PATH)
    expected = (
        "WP10c9d6c7c3b5c4f20_analysis_only_fine_six_mode_"
        "dynamic_coordinate_replay"
    )
    if (
        not summary["passed"]
        or not summary["fine_dynamic_coordinate_replay_authorized"]
        or not summary["middle_replay_forbidden"]
        or summary["authorized_next"] != expected
        or manifest["authorized_next"] != expected
        or manifest["authorized_fine_replay"]["reruns_middle_propagation"]
        or manifest["fixed_Q_micro_solver_authorized"]
        or manifest["nonlinear_retained_mode_pilot_authorized"]
    ):
        raise RuntimeError("c4f20 authorization changed")
    return manifest


def _stable_duals(label, layout, configuration, trajectory, basis):
    reaction = c4f15._reaction_preflight(
        label, 0, layout, configuration, trajectory
    )
    columns = np.asarray(configuration["columns"], dtype=float).reshape(
        trajectory["states"][0].shape
    )
    directions = c4f17.c4f1._initial_directions(
        configuration,
        trajectory,
        c4f13.PARENT_CORE_FACE * int(layout.refinement_ratio),
        trajectory["states"].shape[1],
    )["current"]
    scaled = c4f13._scaled_directions(directions, columns)
    state_lifts = scaled.T @ basis
    descriptor = reaction["descriptor"]
    reaction_lift = reaction["reaction_lift"]
    reaction_scale = np.linalg.norm(descriptor @ reaction_lift, axis=0)
    normalized_reaction_lift = reaction_lift / reaction_scale[None, :]
    trial = np.column_stack((scaled.T, normalized_reaction_lift))
    target = np.column_stack((basis.T, np.zeros((MODE_DIMENSION, 3))))
    descriptor_trial = descriptor @ trial

    q, r = np.linalg.qr(descriptor_trial, mode="reduced")
    dual_qr = target @ np.linalg.solve(r, q.T @ descriptor)
    u, singular, vt = np.linalg.svd(descriptor_trial, full_matrices=False)
    dual_svd = target @ ((vt.T / singular) @ (u.T @ descriptor))

    def metrics(dual):
        return {
            "biorthogonality_defect": float(
                np.max(np.abs(dual @ state_lifts - np.eye(MODE_DIMENSION)))
            ),
            "normalized_slow_lift_annihilation_defect": float(
                np.max(np.abs(dual @ normalized_reaction_lift))
            ),
            "initial_consensus_coefficient_defect": float(
                np.max(np.abs(dual @ scaled.T - basis.T))
            ),
        }

    scale = max(
        float(np.linalg.norm(dual_qr)),
        float(np.linalg.norm(dual_svd)),
        np.finfo(float).tiny,
    )
    return {
        "qr": dual_qr,
        "svd": dual_svd,
        "qr_metrics": metrics(dual_qr),
        "svd_metrics": metrics(dual_svd),
        "state_lift_Q3_defect": float(
            np.max(np.abs(reaction["q_scaled"] @ state_lifts))
        ),
        "relative_QR_SVD_difference": float(
            np.linalg.norm(dual_qr - dual_svd) / scale
        ),
        "descriptor_trial_condition_number": float(singular[0] / singular[-1]),
    }


def _relative_defect(analytic, reference) -> float:
    scale = max(
        float(np.linalg.norm(analytic)),
        float(np.linalg.norm(reference)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(analytic - reference) / scale)


def _face36_audit(context, state, directions, analytic_outputs, layout):
    central = np.empty((MODE_DIMENSION, SELECTED_RELATIVE_STEPS.size))
    five_point = np.empty_like(central)
    for mode in range(MODE_DIMENSION):
        direction = directions[mode]
        analytic = analytic_outputs[mode]
        for step_slot, step in enumerate(SELECTED_RELATIVE_STEPS):
            plus = c4f13._face36_flux(context, state + step * direction, layout)
            minus = c4f13._face36_flux(context, state - step * direction, layout)
            plus_two = c4f13._face36_flux(
                context, state + 2.0 * step * direction, layout
            )
            minus_two = c4f13._face36_flux(
                context, state - 2.0 * step * direction, layout
            )
            central[mode, step_slot] = _relative_defect(
                analytic, (plus - minus) / (2.0 * step)
            )
            five_point[mode, step_slot] = _relative_defect(
                analytic,
                (-plus_two + 8.0 * plus - 8.0 * minus + minus_two)
                / (12.0 * step),
            )
    return central, five_point


def _complement_diagnostics(directions, output_map, columns, layout):
    restricted = np.asarray(
        [
            restrict_causal_embedded_patch_cell_averages(direction, layout)
            for direction in directions
        ]
    )
    prolonged = restricted[:, layout.parent_cell_indices]
    complement = directions - prolonged
    full_scaled = c4f13._scaled_directions(directions, columns)
    complement_scaled = c4f13._scaled_directions(complement, columns)
    state_fraction = np.linalg.norm(complement_scaled, axis=1) / np.maximum(
        np.linalg.norm(full_scaled, axis=1), np.finfo(float).tiny
    )
    full_output = c4f13._apply_map(output_map, directions, columns)
    complement_output = c4f13._apply_map(output_map, complement, columns)
    output_fraction = np.linalg.norm(complement_output, axis=1) / np.maximum(
        np.linalg.norm(full_output, axis=1), np.finfo(float).tiny
    )
    return state_fraction, output_fraction


def _dual_passed(duals: dict, contract: dict) -> bool:
    return bool(
        all(
            metrics["biorthogonality_defect"]
            <= contract["maximum_biorthogonality_defect"]
            and metrics["normalized_slow_lift_annihilation_defect"]
            <= contract["maximum_normalized_slow_lift_annihilation_defect"]
            and metrics["initial_consensus_coefficient_defect"]
            <= contract["maximum_initial_consensus_coefficient_defect"]
            for metrics in (duals["qr_metrics"], duals["svd_metrics"])
        )
        and duals["relative_QR_SVD_difference"]
        <= contract["maximum_relative_QR_SVD_dual_difference"]
    )


def _initial_or_resume(manifest: dict):
    layout, configuration, trajectory = c4f13._layout_data("fine")
    context = configuration["context"]
    columns = np.asarray(configuration["columns"]).reshape(
        trajectory["states"].shape[1:]
    )
    rows = np.asarray(configuration["rows"]).reshape(columns.shape)
    source = _source_identity()

    if CHECKPOINT_PATH.exists() and PROGRESS_PATH.exists():
        progress = _read(PROGRESS_PATH)
        if progress.get("source_identity") != source:
            raise RuntimeError("c4f20 checkpoint source identity changed")
        arrays = _load(CHECKPOINT_PATH)
        return layout, configuration, trajectory, arrays, progress

    basis = c4f17._basis()
    current, previous, _initialized, orthogonality = c4f17._six_initial_directions(
        configuration, trajectory, layout, basis
    )
    previous_state = trajectory["states"][0] - trajectory["primitive_histories"][0]
    previous_dt = float(trajectory["previous_timesteps"][0])
    began = time.perf_counter()
    initial_matrix = causal_five_field_monolithic_discrete_step_matrix(
        context,
        previous_state,
        trajectory["states"][0],
        previous_dt,
        previous_dt,
        primitive_column_scales=columns,
        conservation_row_scales=rows,
    )
    history_direction = causal_five_field_monolithic_bdf_history_direction(
        context,
        previous_state,
        trajectory["states"][0],
        previous,
        current,
        analytic_step_matrix=initial_matrix,
    )
    duals = _stable_duals("fine", layout, configuration, trajectory, basis)
    output_map = c4f13._face36_output_map(initial_matrix, layout)
    outputs = c4f13._apply_map(output_map, current, columns)
    central, five_point = _face36_audit(
        context, trajectory["states"][0], current, outputs, layout
    )
    guard_mapped, guard_height = c4f13._guard_diagnostics(
        context,
        trajectory["states"][0],
        current,
        history_direction,
        columns,
        rows,
        layout,
    )
    constraint = c4f13._macro_constraint(
        context, trajectory["states"][0], columns, rows, layout
    )
    state_fraction, output_fraction = _complement_diagnostics(
        current, output_map, columns, layout
    )
    dual_contract = manifest["stable_dual_contract"]
    output_contract = manifest["face36_directional_JVP_contract"]
    preflight_passed = bool(
        _dual_passed(duals, dual_contract)
        and np.max(five_point)
        <= output_contract["maximum_relative_defect_at_each_selected_step"]
    )
    arrays = {
        "times": trajectory["times"][:1],
        "state_directions": current[None],
        "face36_outputs": outputs[None],
        "amplitude_transitions": (duals["qr"] @ c4f13._scaled_directions(current, columns).T)[None],
        "guard_mapped": guard_mapped[None],
        "guard_height_history": guard_height[None],
        "Q3_leakage": c4f13._relative_q3_leakage(
            current, columns, constraint
        )[None],
        "fine_complement_state_fractions": state_fraction[None],
        "fine_complement_face36_fractions": output_fraction[None],
        "face36_central_defects": central[None],
        "face36_five_point_defects": five_point[None],
        "current_directions": current,
        "current_primitive_history_directions": history_direction.previous_primitive_increment,
        "current_mapped_history_directions": history_direction.previous_mapped_storage_increment,
        "current_height_history_directions": history_direction.previous_responsive_height_storage_increment,
        "dual_QR": duals["qr"],
        "dual_SVD": duals["svd"],
        "dual_QR_metrics": np.asarray(list(duals["qr_metrics"].values())),
        "dual_SVD_metrics": np.asarray(list(duals["svd_metrics"].values())),
        "dual_relative_QR_SVD_difference": np.asarray(
            [duals["relative_QR_SVD_difference"]]
        ),
        "descriptor_trial_condition_number": np.asarray(
            [duals["descriptor_trial_condition_number"]]
        ),
        "initial_Q3_defect": np.asarray([duals["state_lift_Q3_defect"]]),
        "initial_orthogonality_defect": np.asarray([orthogonality]),
        "matrix_wall_seconds": np.asarray([time.perf_counter() - began]),
        "step_wall_seconds": np.empty(0),
        "JVP_defects": np.empty(0),
        "linear_solve_defects": np.empty(0),
        "component_closure_defects": np.asarray(
            [initial_matrix.maximum_component_closure_defect]
        ),
        "incoming_characteristics": np.asarray(
            [initial_matrix.incoming_excision_characteristics], dtype=np.int64
        ),
    }
    progress = {
        "source_identity": source,
        "steps_completed": 0,
        "preflight_passed": preflight_passed,
        "dual_QR_metrics": duals["qr_metrics"],
        "dual_SVD_metrics": duals["svd_metrics"],
        "dual_relative_QR_SVD_difference": duals[
            "relative_QR_SVD_difference"
        ],
        "descriptor_trial_condition_number": duals[
            "descriptor_trial_condition_number"
        ],
        "started_wall_seconds": time.time(),
    }
    _save(CHECKPOINT_PATH, **arrays)
    _write(PROGRESS_PATH, progress)
    return layout, configuration, trajectory, arrays, progress


def _run_fine(manifest: dict, through: str):
    layout, configuration, trajectory, arrays, progress = _initial_or_resume(manifest)
    if not progress["preflight_passed"] or through == "preflight":
        return arrays, progress, layout, configuration, trajectory

    context = configuration["context"]
    columns = np.asarray(configuration["columns"]).reshape(
        trajectory["states"].shape[1:]
    )
    rows = np.asarray(configuration["rows"]).reshape(columns.shape)
    current = arrays["current_directions"]
    dual = arrays["dual_QR"]
    history_direction = CausalFiveFieldMonolithicBDFHistoryDirection(
        previous_primitive_increment=arrays[
            "current_primitive_history_directions"
        ],
        previous_mapped_storage_increment=arrays[
            "current_mapped_history_directions"
        ],
        previous_responsive_height_storage_increment=arrays[
            "current_height_history_directions"
        ],
    ).validated(n_directions=MODE_DIMENSION, n_cells=current.shape[1])
    audit_ids = set(AUDIT_TIME_IDS_MICROSECONDS[1:])
    began = time.perf_counter()

    for step_index in range(
        int(progress["steps_completed"]), trajectory["timesteps"].size
    ):
        step_began = time.perf_counter()
        matrix_began = time.perf_counter()
        matrix = causal_five_field_monolithic_discrete_step_matrix(
            context,
            trajectory["states"][step_index],
            trajectory["states"][step_index + 1],
            float(trajectory["timesteps"][step_index]),
            float(trajectory["previous_timesteps"][step_index]),
            primitive_column_scales=columns,
            conservation_row_scales=rows,
        )
        matrix_wall = time.perf_counter() - matrix_began
        target_id = int(round(trajectory["times"][step_index + 1] * 1.0e6))
        tangent = causal_five_field_monolithic_discrete_tangent_step(
            context,
            trajectory["states"][step_index],
            trajectory["states"][step_index + 1],
            float(trajectory["timesteps"][step_index]),
            c4f13._history(trajectory, step_index),
            current,
            history_direction,
            primitive_column_scales=columns,
            conservation_row_scales=rows,
            analytic_step_matrix=matrix,
            audit_complete_residual=target_id in audit_ids,
        )
        current = tangent.new_primitive_directions
        history_direction = tangent.new_history_directions
        output_map = c4f13._face36_output_map(matrix, layout)
        outputs = c4f13._apply_map(output_map, current, columns)
        if target_id in audit_ids:
            central, five_point = _face36_audit(
                context,
                trajectory["states"][step_index + 1],
                current,
                outputs,
                layout,
            )
        else:
            central = np.full((MODE_DIMENSION, SELECTED_RELATIVE_STEPS.size), np.nan)
            five_point = np.full_like(central, np.nan)
        guard_mapped, guard_height = c4f13._guard_diagnostics(
            context,
            trajectory["states"][step_index + 1],
            current,
            history_direction,
            columns,
            rows,
            layout,
        )
        constraint = c4f13._macro_constraint(
            context, trajectory["states"][step_index + 1], columns, rows, layout
        )
        state_fraction, output_fraction = _complement_diagnostics(
            current, output_map, columns, layout
        )
        arrays["times"] = np.append(
            arrays["times"], trajectory["times"][step_index + 1]
        )
        for name, value in (
            ("state_directions", current),
            ("face36_outputs", outputs),
            (
                "amplitude_transitions",
                dual @ c4f13._scaled_directions(current, columns).T,
            ),
            ("guard_mapped", guard_mapped),
            ("guard_height_history", guard_height),
            (
                "Q3_leakage",
                c4f13._relative_q3_leakage(current, columns, constraint),
            ),
            ("fine_complement_state_fractions", state_fraction),
            ("fine_complement_face36_fractions", output_fraction),
            ("face36_central_defects", central),
            ("face36_five_point_defects", five_point),
        ):
            arrays[name] = np.concatenate((arrays[name], value[None]), axis=0)
        arrays["current_directions"] = current
        arrays["current_primitive_history_directions"] = (
            history_direction.previous_primitive_increment
        )
        arrays["current_mapped_history_directions"] = (
            history_direction.previous_mapped_storage_increment
        )
        arrays["current_height_history_directions"] = (
            history_direction.previous_responsive_height_storage_increment
        )
        arrays["matrix_wall_seconds"] = np.append(
            arrays["matrix_wall_seconds"], matrix_wall
        )
        arrays["step_wall_seconds"] = np.append(
            arrays["step_wall_seconds"], time.perf_counter() - step_began
        )
        arrays["JVP_defects"] = np.append(
            arrays["JVP_defects"], tangent.maximum_step_matrix_jvp_relative_defect
        )
        arrays["linear_solve_defects"] = np.append(
            arrays["linear_solve_defects"],
            tangent.maximum_linear_solve_relative_defect,
        )
        arrays["component_closure_defects"] = np.append(
            arrays["component_closure_defects"],
            matrix.maximum_component_closure_defect,
        )
        arrays["incoming_characteristics"] = np.append(
            arrays["incoming_characteristics"],
            matrix.incoming_excision_characteristics,
        )
        progress["steps_completed"] = step_index + 1
        _save(CHECKPOINT_PATH, **arrays)
        _write(PROGRESS_PATH, progress)
        print(
            f"c4f20-fine: {step_index + 1}/{trajectory['timesteps'].size} "
            f"t={trajectory['times'][step_index + 1]:.7f}s "
            f"matrix={matrix_wall:.1f}s step={arrays['step_wall_seconds'][-1]:.1f}s",
            flush=True,
        )
    progress["completion_wall_seconds"] = time.perf_counter() - began
    _save(CHECKPOINT_PATH, **arrays)
    _write(PROGRESS_PATH, progress)
    return arrays, progress, layout, configuration, trajectory


def _middle_with_recovered_amplitudes():
    middle = _load(c4f17.DECISIVE_ARRAYS)
    _layout, configuration, _trajectory = c4f13._layout_data("middle")
    columns = np.asarray(configuration["columns"]).reshape(
        middle["middle_state_directions"].shape[2:]
    )
    with np.load(c4f18.DECISIVE_ARRAYS, allow_pickle=False) as recovery:
        dual = np.asarray(recovery["dual_QR"])
    amplitudes = np.asarray(
        [
            dual @ c4f13._scaled_directions(directions, columns).T
            for directions in middle["middle_state_directions"]
        ]
    )
    return {
        "times": middle["times"],
        "state_directions": middle["middle_state_directions"],
        "face36_outputs": middle["middle_face36_outputs"],
        "amplitude_transitions": amplitudes,
        "Q3_leakage": middle["middle_Q3_leakage"],
        "guard_mapped": middle["middle_guard_mapped"],
        "guard_height_history": middle["middle_guard_height_history"],
    }


def _method_report(arrays, progress, manifest, trajectory):
    gates = manifest["single_layout_method_gates"]
    dual_contract = manifest["stable_dual_contract"]
    finite_jvp = arrays["JVP_defects"][np.isfinite(arrays["JVP_defects"])]
    finite_five = arrays["face36_five_point_defects"][
        np.isfinite(arrays["face36_five_point_defects"])
    ]
    duals = {
        "qr_metrics": progress["dual_QR_metrics"],
        "svd_metrics": progress["dual_SVD_metrics"],
        "relative_QR_SVD_difference": progress[
            "dual_relative_QR_SVD_difference"
        ],
    }
    complete = int(progress["steps_completed"]) == trajectory["timesteps"].size
    passed = bool(
        complete
        and progress["preflight_passed"]
        and _dual_passed(duals, dual_contract)
        and arrays["initial_Q3_defect"][0]
        <= gates["maximum_initial_state_lift_Q3_defect"]
        and arrays["initial_orthogonality_defect"][0]
        <= gates["maximum_initial_scaled_orthogonality_defect"]
        and finite_jvp.size == 4
        and np.max(finite_jvp)
        <= gates["maximum_step_matrix_JVP_relative_defect"]
        and np.max(arrays["linear_solve_defects"])
        <= gates["maximum_block_linear_solve_relative_defect"]
        and np.max(arrays["component_closure_defects"])
        <= gates["maximum_component_closure_defect"]
        and finite_five.size
        == len(AUDIT_TIME_IDS_MICROSECONDS)
        * MODE_DIMENSION
        * SELECTED_RELATIVE_STEPS.size
        and np.max(finite_five)
        <= manifest["face36_directional_JVP_contract"][
            "maximum_relative_defect_at_each_selected_step"
        ]
        and np.max(arrays["Q3_leakage"]) <= gates["maximum_Q3_leakage"]
        and np.max(arrays["incoming_characteristics"])
        == gates["incoming_excision_characteristics"]
    )
    return {
        "passed": passed,
        "complete": complete,
        "steps": int(progress["steps_completed"]),
        "directions": MODE_DIMENSION,
        "preflight_passed": bool(progress["preflight_passed"]),
        "dual_QR_metrics": progress["dual_QR_metrics"],
        "dual_SVD_metrics": progress["dual_SVD_metrics"],
        "dual_relative_QR_SVD_difference": progress[
            "dual_relative_QR_SVD_difference"
        ],
        "descriptor_trial_condition_number": progress[
            "descriptor_trial_condition_number"
        ],
        "maximum_JVP_defect": (
            float(np.max(finite_jvp)) if finite_jvp.size else None
        ),
        "maximum_linear_solve_defect": (
            float(np.max(arrays["linear_solve_defects"]))
            if arrays["linear_solve_defects"].size
            else None
        ),
        "maximum_component_closure_defect": float(
            np.max(arrays["component_closure_defects"])
        ),
        "maximum_face36_central_defect": float(
            np.nanmax(arrays["face36_central_defects"])
        ),
        "maximum_face36_five_point_defect": float(
            np.nanmax(arrays["face36_five_point_defects"])
        ),
        "maximum_Q3_leakage": float(np.max(arrays["Q3_leakage"])),
        "maximum_incoming_characteristics": int(
            np.max(arrays["incoming_characteristics"])
        ),
        "initial_Q3_defect": float(arrays["initial_Q3_defect"][0]),
        "initial_orthogonality_defect": float(
            arrays["initial_orthogonality_defect"][0]
        ),
        "maximum_fine_complement_state_fraction": float(
            np.max(arrays["fine_complement_state_fractions"])
        ),
        "maximum_fine_complement_face36_fraction": float(
            np.max(arrays["fine_complement_face36_fractions"])
        ),
        "wall_seconds": float(time.time() - progress["started_wall_seconds"]),
    }


def _catalog(summary):
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
                    "sha256": _sha(path),
                    "scientific_status": (
                        "SUPPORTED BUT NOT FULLY CERTIFIED"
                        if summary["passed"]
                        else "REJECTED CANDIDATE"
                    ),
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
    catalog = _read(CANONICAL_SUMMARY)
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
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write(CANONICAL_SUMMARY, catalog)


def _package(arrays, progress, layout, configuration, trajectory, manifest):
    fine_report = _method_report(arrays, progress, manifest, trajectory)
    cross = None
    static_output = _read(c4f15.SUMMARY_PATH)[
        "minimum_passing_dimension_output_reconstruction"
    ]
    if fine_report["passed"]:
        middle = _middle_with_recovered_amplitudes()
        cross = c4f17._cross_resolution(
            middle,
            arrays,
            c4f13._layout_data("middle")[0],
            layout,
            c4f13._layout_data("middle")[1],
            configuration,
        )
        gates = manifest["cross_resolution_contract"]
        cross_passed = bool(
            cross["minimum_leading_block_projector_cosine"]
            >= gates["minimum_leading_block_projector_cosine"]
            and cross["minimum_full_subspace_projector_cosine"]
            >= gates["minimum_full_subspace_projector_cosine"]
            and cross["amplitude_transition_history_cosine"]
            >= gates["minimum_stable_dual_amplitude_history_cosine"]
            and cross["amplitude_transition_relative_difference"]
            <= gates["maximum_stable_dual_amplitude_history_relative_difference"]
            and cross["face36_mode_history_cosine"]
            >= gates["minimum_face36_mode_history_cosine"]
            and cross["face36_mode_history_relative_difference"]
            <= gates["maximum_face36_mode_history_relative_difference"]
            and static_output["maximum_output_weighted_RMS_error"]
            <= gates["maximum_six_mode_output_weighted_RMS_error"]
            and static_output["maximum_significant_direction_error"]
            <= gates["maximum_six_mode_significant_direction_error"]
        )
    else:
        cross_passed = False

    passed = bool(fine_report["passed"] and cross_passed)
    if passed:
        classification = (
            "face36_six_mode_dynamic_coordinate_replay_certified_"
            "one_Q_manifest_authorized"
        )
        authorized_next = (
            "WP10c9d6c7c3b5c4f21_definitions_only_one_Q_constrained_"
            "nonlinear_pilot_manifest"
        )
    elif fine_report["passed"] and cross[
        "minimum_leading_block_projector_cosine"
    ] >= manifest["cross_resolution_contract"][
        "minimum_leading_block_projector_cosine"
    ]:
        classification = (
            "face36_six_mode_dynamic_coordinate_rejected_"
            "leading_two_plus_HMM_manifest_authorized"
        )
        authorized_next = (
            "WP10c9d6c7c3b5c4f21_definitions_only_leading_two_plus_"
            "HMM_closure_manifest"
        )
    elif fine_report["passed"]:
        classification = (
            "face36_dynamic_coordinate_basis_rejected_return_to_"
            "memory_localization"
        )
        authorized_next = "definitions_only_memory_basis_relocalization_manifest"
    else:
        classification = (
            "fine_six_mode_numerical_or_method_preflight_rejected_"
            "fine_audit_localization_manifest_authorized"
        )
        authorized_next = "definitions_only_fine_numerical_audit_localization_manifest"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "fine_executed": bool(fine_report["steps"] > 0),
        "fine": fine_report,
        "cross_resolution": (
            {
                key: value
                for key, value in cross.items()
                if not isinstance(value, np.ndarray)
            }
            if cross is not None
            else None
        ),
        "static_six_mode_output_reconstruction": static_output,
        "new_nonlinear_trajectory": False,
        "new_tangent_trajectory": bool(fine_report["steps"] > 0),
        "middle_replayed": False,
        "fixed_Q_reaction_applied": False,
        "fixed_Q_micro_solver_authorized": False,
        "nonlinear_retained_mode_pilot_authorized": False,
        "fifty_ms_propagation_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "physical_failure_detected": False,
        "guard_complement_retained": True,
        "raw_face48_export_rejection_preserved": True,
        "authorized_next": authorized_next,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    payload = {
        "times": arrays["times"],
        "fine_state_directions": arrays["state_directions"],
        "fine_face36_outputs": arrays["face36_outputs"],
        "fine_amplitude_transitions": arrays["amplitude_transitions"],
        "fine_Q3_leakage": arrays["Q3_leakage"],
        "fine_guard_mapped": arrays["guard_mapped"],
        "fine_guard_height_history": arrays["guard_height_history"],
        "fine_complement_state_fractions": arrays[
            "fine_complement_state_fractions"
        ],
        "fine_complement_face36_fractions": arrays[
            "fine_complement_face36_fractions"
        ],
        "fine_face36_central_defects": arrays["face36_central_defects"],
        "fine_face36_five_point_defects": arrays[
            "face36_five_point_defects"
        ],
        "fine_dual_QR": arrays["dual_QR"],
        "fine_dual_SVD": arrays["dual_SVD"],
    }
    if cross is not None:
        payload.update(
            {
                "leading_block_projector_cosines": cross[
                    "leading_block_projector_cosines"
                ],
                "full_subspace_projector_cosines": cross[
                    "full_subspace_projector_cosines"
                ],
                "weak_block_Procrustes_alignments": cross[
                    "weak_block_Procrustes_alignments"
                ],
                "fine_amplitude_transitions_aligned": cross[
                    "fine_amplitude_transitions_aligned"
                ],
                "fine_face36_outputs_aligned": cross[
                    "fine_face36_outputs_aligned"
                ],
            }
        )
    _save(DECISIVE_ARRAYS, **payload)
    _write(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "mode_dimension": MODE_DIMENSION,
            "leading_dimension": LEADING_DIMENSION,
            "audit_time_ids_microseconds": list(AUDIT_TIME_IDS_MICROSECONDS),
            "selected_relative_steps": SELECTED_RELATIVE_STEPS.tolist(),
            "single_layout_method_gates": manifest[
                "single_layout_method_gates"
            ],
            "stable_dual_contract": manifest["stable_dual_contract"],
            "face36_directional_JVP_contract": manifest[
                "face36_directional_JVP_contract"
            ],
            "cross_resolution_contract": manifest[
                "cross_resolution_contract"
            ],
        },
    )
    _write(SUMMARY_PATH, summary)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cross_text = (
        "Cross-resolution comparison was not interpreted because the fine "
        "single-layout gates failed."
        if cross is None
        else (
            "The minimum leading/full state-projector cosines are "
            f"`{cross['minimum_leading_block_projector_cosine']:.6f}` / "
            f"`{cross['minimum_full_subspace_projector_cosine']:.6f}`. The "
            "stable-dual amplitude history has cosine/difference "
            f"`{cross['amplitude_transition_history_cosine']:.6f}` / "
            f"`{cross['amplitude_transition_relative_difference']:.6f}`; "
            "the face-36 history has cosine/difference "
            f"`{cross['face36_mode_history_cosine']:.6f}` / "
            f"`{cross['face36_mode_history_relative_difference']:.6f}`."
        )
    )
    REPORT_PATH.write_text(
        "# Face-36 six-mode fine dynamic-coordinate replay\n\n"
        f"Classification: `{classification}`.\n\n"
        f"Fine method/coordinate gates pass: `{fine_report['passed']}`. The "
        "maximum complete-step JVP, five-point face-36, and Q3-leakage "
        f"defects are `{fine_report['maximum_JVP_defect']}`, "
        f"`{fine_report['maximum_face36_five_point_defect']:.3e}`, and "
        f"`{fine_report['maximum_Q3_leakage']:.6f}`.\n\n"
        f"{cross_text}\n\n"
        "The middle tangent history was not replayed. No nonlinear state, "
        "fixed-Q reaction, 50 ms trajectory, or reduced slow evolution was "
        "advanced. The guard complement and raw face-48 rejection remain "
        "binding.\n",
        encoding="utf-8",
    )
    _write(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "manifest_summary_sha256": _sha(c4f19.SUMMARY_PATH),
            "manifest_sha256": _sha(c4f19.MANIFEST_PATH),
            "middle_arrays_sha256": _sha(c4f17.DECISIVE_ARRAYS),
            "recovery_arrays_sha256": _sha(c4f18.DECISIVE_ARRAYS),
            "source_hashes": _source_identity(),
        },
    )
    files = (CONFIG_PATH, DECISIVE_ARRAYS, SUMMARY_PATH, PROVENANCE_PATH)
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )
    _catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--through", choices=STAGES, default="complete")
    args = parser.parse_args()
    manifest = _authorization()
    arrays, progress, layout, configuration, trajectory = _run_fine(
        manifest, args.through
    )
    preflight = {
        "preflight_passed": bool(progress["preflight_passed"]),
        "steps_completed": int(progress["steps_completed"]),
        "dual_QR_metrics": progress["dual_QR_metrics"],
        "dual_SVD_metrics": progress["dual_SVD_metrics"],
        "dual_relative_QR_SVD_difference": progress[
            "dual_relative_QR_SVD_difference"
        ],
        "maximum_initial_five_point_defect": float(
            np.max(arrays["face36_five_point_defects"][0])
        ),
    }
    if args.through == "preflight":
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return
    _package(arrays, progress, layout, configuration, trajectory, manifest)


if __name__ == "__main__":
    main()
