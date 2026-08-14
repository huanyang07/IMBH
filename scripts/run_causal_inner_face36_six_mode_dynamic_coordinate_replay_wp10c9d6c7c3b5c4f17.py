#!/usr/bin/env python3
"""Replay six retained directions through committed middle/fine histories.

This is an analysis-only tangent replay.  It advances no nonlinear state and
does not impose a fixed-Q reaction.
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

import run_causal_inner_absolute_baseline_observable_memory_screen_wp10c9d6c7c3b5c4f1 as c4f1  # noqa: E402
import run_causal_inner_face36_augmented_memory_screen_wp10c9d6c7c3b5c4f13 as c4f13  # noqa: E402
import run_causal_inner_face36_q_plus_a_reaction_coordinate_preflight_wp10c9d6c7c3b5c4f15 as c4f15  # noqa: E402
import run_causal_inner_face36_six_mode_coordinate_manifest_wp10c9d6c7c3b5c4f16 as c4f16  # noqa: E402

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
WORK_PACKAGE = "WP10c9d6c7c3b5c4f17"
ARTIFACT = "causal_inner_face36_six_mode_dynamic_coordinate_replay_wp10c9d6c7c3b5c4f17"
THIS_RUNNER = "scripts/run_causal_inner_face36_six_mode_dynamic_coordinate_replay_wp10c9d6c7c3b5c4f17.py"
THIS_TEST = "tests/test_causal_inner_face36_six_mode_dynamic_coordinate_replay_wp10c9d6c7c3b5c4f17.py"
REPORT_RELATIVE = "docs/reports/current/CODEX_CAUSAL_INNER_FACE36_SIX_MODE_DYNAMIC_COORDINATE_REPLAY_WP10C9D6C7C3B5C4F17_2026-08-13.md"
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
CHECKPOINT_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT

LAYOUTS = ("middle", "fine")
MODE_DIMENSION = 6
LEADING_DIMENSION = 2
AUDIT_TARGETS_SECONDS = (0.0054, 0.010, 0.016, 0.020)
STAGES = ("middle", "complete")


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
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _source_identity() -> dict[str, str]:
    paths = (
        THIS_RUNNER,
        THIS_TEST,
        c4f16.THIS_RUNNER,
        c4f16.THIS_TEST,
        c4f15.THIS_RUNNER,
        c4f13.THIS_RUNNER,
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_discrete_tangent.py",
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_tangent.py",
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_radial_fluctuation.py",
    )
    return {path: _sha(ROOT / path) for path in paths if (ROOT / path).exists()}


def _authorization():
    summary = _read(c4f16.SUMMARY_PATH)
    manifest = _read(c4f16.MANIFEST_PATH)
    if (
        not summary["passed"]
        or not summary["dynamic_coordinate_preflight_authorized"]
        or summary["authorized_next"]
        != "WP10c9d6c7c3b5c4f17_analysis_only_six_mode_dynamic_coordinate_replay"
        or not manifest["authorized_dynamic_coordinate_preflight"]["run_middle_first"]
        or manifest["fixed_Q_micro_solver_authorized"]
        or manifest["nonlinear_retained_mode_pilot_authorized"]
    ):
        raise RuntimeError("c4f17 authorization changed")
    return manifest


def _basis() -> np.ndarray:
    with np.load(c4f16.BASIS_PATH, allow_pickle=False) as arrays:
        basis = np.asarray(arrays["six_mode_consensus_direction_coefficients"])
    if basis.shape != (c4f13.TOTAL_DIRECTIONS, MODE_DIMENSION):
        raise RuntimeError("c4f17 six-mode basis changed")
    return basis


def _six_initial_directions(configuration, trajectory, layout, basis):
    initialized = c4f1._initial_directions(
        configuration,
        trajectory,
        c4f13.PARENT_CORE_FACE * int(layout.refinement_ratio),
        trajectory["states"].shape[1],
    )
    current = np.einsum("dm,dnf->mnf", basis, initialized["current"])
    previous = np.einsum("dm,dnf->mnf", basis, initialized["previous"])
    columns = np.asarray(configuration["columns"]).reshape(current.shape[1:])
    scaled = c4f13._scaled_directions(current, columns)
    orthogonality = float(
        np.max(np.abs(scaled @ scaled.T - np.eye(MODE_DIMENSION)))
    )
    return current, previous, initialized, orthogonality


def _audit_indices(trajectory):
    indices = set()
    time_ids_microseconds = np.rint(
        np.asarray(trajectory["times"][1:]) * 1.0e6
    ).astype(np.int64)
    for target in AUDIT_TARGETS_SECONDS:
        target_id = int(round(target * 1.0e6))
        match = np.flatnonzero(time_ids_microseconds == target_id)
        if match.size != 1:
            raise RuntimeError(f"c4f17 audit target {target} is not unique")
        indices.add(int(match[0]))
    return indices


def _run_layout(label: str, manifest: dict):
    layout, configuration, trajectory = c4f13._layout_data(label)
    context = configuration["context"]
    columns = np.asarray(configuration["columns"]).reshape(
        trajectory["states"].shape[1:]
    )
    rows = np.asarray(configuration["rows"]).reshape(columns.shape)
    basis = _basis()
    checkpoint = CHECKPOINT_DIRECTORY / f"{label}.npz"
    progress_path = CHECKPOINT_DIRECTORY / f"{label}.json"
    source = _source_identity()
    audits = _audit_indices(trajectory)
    began = time.perf_counter()

    if checkpoint.exists() and progress_path.exists():
        progress = _read(progress_path)
        if progress.get("source_identity") != source:
            raise RuntimeError(f"c4f17 {label} checkpoint source changed")
        arrays = _load(checkpoint)
        index = int(progress["steps_completed"])
        current = arrays["current_directions"]
        dual = arrays["fixed_descriptor_dual"]
        history_direction = CausalFiveFieldMonolithicBDFHistoryDirection(
            previous_primitive_increment=arrays["current_primitive_history_directions"],
            previous_mapped_storage_increment=arrays["current_mapped_history_directions"],
            previous_responsive_height_storage_increment=arrays["current_height_history_directions"],
        ).validated(n_directions=MODE_DIMENSION, n_cells=current.shape[1])
    else:
        current, previous, initialized, orthogonality = _six_initial_directions(
            configuration, trajectory, layout, basis
        )
        previous_state = trajectory["states"][0] - trajectory["primitive_histories"][0]
        previous_dt = float(trajectory["previous_timesteps"][0])
        initial_began = time.perf_counter()
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
        reaction = c4f15._reaction_preflight(
            label, 0, layout, configuration, trajectory
        )
        coordinate = c4f15._state_coordinate_preflight(
            layout,
            configuration,
            trajectory,
            reaction["descriptor"],
            reaction["q_scaled"],
            reaction["reaction_lift"],
            basis,
        )
        dual = coordinate["dual"]
        output_map = c4f13._face36_output_map(initial_matrix, layout)
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
        output_defect = c4f13._face36_map_defect(
            context,
            trajectory["states"][0],
            current[0],
            output_map,
            columns,
            layout,
        )
        scaled = c4f13._scaled_directions(current, columns)
        arrays = {
            "times": trajectory["times"][:1],
            "state_directions": current[None, ...],
            "face36_outputs": c4f13._apply_map(output_map, current, columns)[None, ...],
            "amplitude_transitions": (dual @ scaled.T)[None, ...],
            "guard_mapped": guard_mapped[None, ...],
            "guard_height_history": guard_height[None, ...],
            "Q3_leakage": c4f13._relative_q3_leakage(
                current, columns, constraint
            )[None, ...],
            "current_directions": current,
            "current_primitive_history_directions": history_direction.previous_primitive_increment,
            "current_mapped_history_directions": history_direction.previous_mapped_storage_increment,
            "current_height_history_directions": history_direction.previous_responsive_height_storage_increment,
            "fixed_descriptor_dual": dual,
            "matrix_wall_seconds": np.asarray([time.perf_counter() - initial_began]),
            "step_wall_seconds": np.empty(0),
            "JVP_defects": np.empty(0),
            "linear_solve_defects": np.empty(0),
            "component_closure_defects": np.asarray(
                [initial_matrix.maximum_component_closure_defect]
            ),
            "face36_output_map_defects": np.asarray([output_defect]),
            "incoming_characteristics": np.asarray(
                [initial_matrix.incoming_excision_characteristics], dtype=np.int64
            ),
            "initial_Q3_defect": np.asarray([coordinate["state_lift_Q3_defect"]]),
            "initial_orthogonality_defect": np.asarray([orthogonality]),
            "dual_biorthogonality_defect": np.asarray(
                [coordinate["dual_biorthogonality_defect"]]
            ),
            "dual_normalized_slow_annihilation_defect": np.asarray(
                [coordinate["dual_normalized_slow_lift_annihilation_defect"]]
            ),
        }
        index = 0
        progress = {
            "source_identity": source,
            "steps_completed": 0,
            "started_wall_seconds": time.time(),
        }
        _save(checkpoint, **arrays)
        _write(progress_path, progress)

    for step_index in range(index, trajectory["timesteps"].size):
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
            audit_complete_residual=step_index in audits,
        )
        current = tangent.new_primitive_directions
        history_direction = tangent.new_history_directions
        output_map = c4f13._face36_output_map(matrix, layout)
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
        output_defect = np.nan
        if step_index in audits:
            output_defect = c4f13._face36_map_defect(
                context,
                trajectory["states"][step_index + 1],
                current[0],
                output_map,
                columns,
                layout,
            )
        scaled = c4f13._scaled_directions(current, columns)
        arrays["times"] = np.append(arrays["times"], trajectory["times"][step_index + 1])
        arrays["state_directions"] = np.concatenate(
            (arrays["state_directions"], current[None, ...]), axis=0
        )
        arrays["face36_outputs"] = np.concatenate(
            (
                arrays["face36_outputs"],
                c4f13._apply_map(output_map, current, columns)[None, ...],
            ),
            axis=0,
        )
        arrays["amplitude_transitions"] = np.concatenate(
            (arrays["amplitude_transitions"], (dual @ scaled.T)[None, ...]), axis=0
        )
        arrays["guard_mapped"] = np.concatenate(
            (arrays["guard_mapped"], guard_mapped[None, ...]), axis=0
        )
        arrays["guard_height_history"] = np.concatenate(
            (arrays["guard_height_history"], guard_height[None, ...]), axis=0
        )
        arrays["Q3_leakage"] = np.concatenate(
            (
                arrays["Q3_leakage"],
                c4f13._relative_q3_leakage(current, columns, constraint)[None, ...],
            ),
            axis=0,
        )
        arrays["current_directions"] = current
        arrays["current_primitive_history_directions"] = history_direction.previous_primitive_increment
        arrays["current_mapped_history_directions"] = history_direction.previous_mapped_storage_increment
        arrays["current_height_history_directions"] = history_direction.previous_responsive_height_storage_increment
        arrays["matrix_wall_seconds"] = np.append(arrays["matrix_wall_seconds"], matrix_wall)
        arrays["step_wall_seconds"] = np.append(
            arrays["step_wall_seconds"], time.perf_counter() - step_began
        )
        arrays["JVP_defects"] = np.append(
            arrays["JVP_defects"], tangent.maximum_step_matrix_jvp_relative_defect
        )
        arrays["linear_solve_defects"] = np.append(
            arrays["linear_solve_defects"], tangent.maximum_linear_solve_relative_defect
        )
        arrays["component_closure_defects"] = np.append(
            arrays["component_closure_defects"], matrix.maximum_component_closure_defect
        )
        arrays["face36_output_map_defects"] = np.append(
            arrays["face36_output_map_defects"], output_defect
        )
        arrays["incoming_characteristics"] = np.append(
            arrays["incoming_characteristics"], matrix.incoming_excision_characteristics
        )
        progress["steps_completed"] = step_index + 1
        _save(checkpoint, **arrays)
        _write(progress_path, progress)
        print(
            f"c4f17-{label}: {step_index + 1}/{trajectory['timesteps'].size} "
            f"t={trajectory['times'][step_index + 1]:.7f}s "
            f"matrix={matrix_wall:.1f}s step={arrays['step_wall_seconds'][-1]:.1f}s",
            flush=True,
        )

    gates = manifest["prospective_dynamic_gates"]
    finite_jvp = arrays["JVP_defects"][np.isfinite(arrays["JVP_defects"])]
    finite_output = arrays["face36_output_map_defects"][
        np.isfinite(arrays["face36_output_map_defects"])
    ]
    method = bool(
        arrays["initial_Q3_defect"][0] <= gates["maximum_initial_state_lift_Q3_defect"]
        and arrays["initial_orthogonality_defect"][0] <= 1.0e-10
        and arrays["dual_biorthogonality_defect"][0]
        <= gates["maximum_dual_biorthogonality_defect"]
        and arrays["dual_normalized_slow_annihilation_defect"][0]
        <= gates["maximum_normalized_slow_lift_annihilation_defect"]
        and finite_jvp.size == len(AUDIT_TARGETS_SECONDS)
        and np.max(finite_jvp) <= gates["maximum_step_matrix_JVP_relative_defect"]
        and np.max(arrays["linear_solve_defects"])
        <= gates["maximum_block_linear_solve_relative_defect"]
        and finite_output.size == len(AUDIT_TARGETS_SECONDS) + 1
        and np.max(finite_output) <= gates["maximum_face36_output_map_relative_defect"]
        and np.max(arrays["Q3_leakage"]) <= gates["maximum_Q3_leakage"]
        and np.max(arrays["incoming_characteristics"]) == 0
    )
    report = {
        "passed_method_and_single_layout_coordinate_gates": method,
        "steps": int(trajectory["timesteps"].size),
        "directions": MODE_DIMENSION,
        "maximum_JVP_defect": float(np.max(finite_jvp)),
        "maximum_linear_solve_defect": float(np.max(arrays["linear_solve_defects"])),
        "maximum_component_closure_defect": float(
            np.max(arrays["component_closure_defects"])
        ),
        "maximum_face36_output_map_defect": float(np.max(finite_output)),
        "maximum_Q3_leakage": float(np.max(arrays["Q3_leakage"])),
        "maximum_incoming_characteristics": int(
            np.max(arrays["incoming_characteristics"])
        ),
        "initial_Q3_defect": float(arrays["initial_Q3_defect"][0]),
        "initial_orthogonality_defect": float(arrays["initial_orthogonality_defect"][0]),
        "dual_biorthogonality_defect": float(arrays["dual_biorthogonality_defect"][0]),
        "dual_normalized_slow_annihilation_defect": float(
            arrays["dual_normalized_slow_annihilation_defect"][0]
        ),
        "wall_seconds": float(time.perf_counter() - began),
    }
    _save(checkpoint, **arrays)
    _write(CHECKPOINT_DIRECTORY / f"{label}_summary.json", report)
    return report, arrays, layout, configuration, trajectory


def _restrict_modes(state_history: np.ndarray, layout) -> np.ndarray:
    return np.asarray(
        [
            [
                restrict_causal_embedded_patch_cell_averages(direction, layout)
                for direction in modes
            ]
            for modes in state_history
        ]
    )


def _cosine_and_difference(left, right):
    a = np.asarray(left).ravel()
    b = np.asarray(right).ravel()
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    scale = max(norm_a, norm_b, np.finfo(float).tiny)
    cosine = 1.0 if max(norm_a, norm_b) <= np.finfo(float).tiny else float(
        np.dot(a, b) / max(norm_a * norm_b, np.finfo(float).tiny)
    )
    return cosine, float(np.linalg.norm(a - b) / scale)


def _weak_block_alignment(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return the block-orthogonal map aligning fine columns to middle."""

    cross = right[:, LEADING_DIMENSION:].T @ left[:, LEADING_DIMENSION:]
    left_vectors, _singular, right_vectors_t = np.linalg.svd(cross)
    transform = np.eye(MODE_DIMENSION)
    transform[LEADING_DIMENSION:, LEADING_DIMENSION:] = (
        left_vectors @ right_vectors_t
    )
    return transform


def _cross_resolution(middle, fine, middle_layout, fine_layout, middle_cfg, fine_cfg):
    middle_states = _restrict_modes(middle["state_directions"], middle_layout)
    fine_states = _restrict_modes(fine["state_directions"], fine_layout)
    middle_columns = _restrict_modes(
        np.broadcast_to(
            np.asarray(middle_cfg["columns"]).reshape(middle["state_directions"].shape[2:]),
            (middle["times"].size, MODE_DIMENSION) + middle["state_directions"].shape[2:],
        ),
        middle_layout,
    )
    fine_columns = _restrict_modes(
        np.broadcast_to(
            np.asarray(fine_cfg["columns"]).reshape(fine["state_directions"].shape[2:]),
            (fine["times"].size, MODE_DIMENSION) + fine["state_directions"].shape[2:],
        ),
        fine_layout,
    )
    common_scale = 0.5 * (middle_columns[:, 0] + fine_columns[:, 0])
    middle_scaled = middle_states / common_scale[:, None]
    fine_scaled = fine_states / common_scale[:, None]
    leading_cosines = []
    full_cosines = []
    weak_alignments = []
    for left, right in zip(middle_scaled, fine_scaled, strict=True):
        left_matrix = left.reshape(MODE_DIMENSION, -1).T
        right_matrix = right.reshape(MODE_DIMENSION, -1).T
        q_left, _ = np.linalg.qr(left_matrix)
        q_right, _ = np.linalg.qr(right_matrix)
        full_cosines.append(
            np.linalg.svd(q_left.T @ q_right, compute_uv=False)
        )
        ql, _ = np.linalg.qr(left_matrix[:, :LEADING_DIMENSION])
        qr, _ = np.linalg.qr(right_matrix[:, :LEADING_DIMENSION])
        leading_cosines.append(np.linalg.svd(ql.T @ qr, compute_uv=False))
        weak_alignments.append(_weak_block_alignment(left_matrix, right_matrix))
    weak_alignments = np.asarray(weak_alignments)
    initial_alignment = weak_alignments[0]
    fine_amplitudes_aligned = np.asarray(
        [
            initial_alignment.T @ amplitudes @ alignment
            for amplitudes, alignment in zip(
                fine["amplitude_transitions"], weak_alignments, strict=True
            )
        ]
    )
    amplitude_cosine, amplitude_difference = _cosine_and_difference(
        middle["amplitude_transitions"], fine_amplitudes_aligned
    )
    middle_output = middle["face36_outputs"]
    fine_output = np.asarray(
        [
            alignment.T @ outputs
            for outputs, alignment in zip(
                fine["face36_outputs"], weak_alignments, strict=True
            )
        ]
    )
    output_scale = np.maximum(
        np.maximum(np.max(np.abs(middle_output), axis=(0, 1)), np.max(np.abs(fine_output), axis=(0, 1))),
        np.finfo(float).tiny,
    )
    output_cosine, output_difference = _cosine_and_difference(
        middle_output / output_scale, fine_output / output_scale
    )
    return {
        "minimum_leading_block_projector_cosine": float(np.min(leading_cosines)),
        "minimum_full_subspace_projector_cosine": float(np.min(full_cosines)),
        "amplitude_transition_history_cosine": amplitude_cosine,
        "amplitude_transition_relative_difference": amplitude_difference,
        "face36_mode_history_cosine": output_cosine,
        "face36_mode_history_relative_difference": output_difference,
        "leading_block_projector_cosines": np.asarray(leading_cosines),
        "full_subspace_projector_cosines": np.asarray(full_cosines),
        "weak_block_Procrustes_alignments": weak_alignments,
        "fine_amplitude_transitions_aligned": fine_amplitudes_aligned,
        "fine_face36_outputs_aligned": fine_output,
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
                    "scientific_status": "SUPPORTED BUT NOT FULLY CERTIFIED"
                    if summary["passed"] else "REJECTED CANDIDATE",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--through", choices=STAGES, default="complete")
    args = parser.parse_args()
    manifest = _authorization()
    middle_report, middle, middle_layout, middle_cfg, middle_traj = _run_layout(
        "middle", manifest
    )
    if not middle_report["passed_method_and_single_layout_coordinate_gates"]:
        raise RuntimeError("c4f17 middle failed; fine is blocked")
    if args.through == "middle":
        print(json.dumps(middle_report, indent=2, sort_keys=True))
        return
    fine_report, fine, fine_layout, fine_cfg, fine_traj = _run_layout("fine", manifest)
    cross = _cross_resolution(
        middle, fine, middle_layout, fine_layout, middle_cfg, fine_cfg
    )
    gates = manifest["prospective_dynamic_gates"]
    cross_passed = bool(
        cross["minimum_leading_block_projector_cosine"]
        >= gates["minimum_middle_fine_leading_block_projector_cosine"]
        and cross["minimum_full_subspace_projector_cosine"]
        >= gates["minimum_middle_fine_full_subspace_projector_cosine"]
        and cross["amplitude_transition_history_cosine"]
        >= gates["minimum_middle_fine_amplitude_transition_history_cosine"]
        and cross["amplitude_transition_relative_difference"]
        <= gates["maximum_middle_fine_amplitude_transition_relative_difference"]
        and cross["face36_mode_history_cosine"]
        >= gates["minimum_middle_fine_face36_mode_history_cosine"]
        and cross["face36_mode_history_relative_difference"]
        <= gates["maximum_middle_fine_face36_mode_history_relative_difference"]
    )
    passed = bool(
        middle_report["passed_method_and_single_layout_coordinate_gates"]
        and fine_report["passed_method_and_single_layout_coordinate_gates"]
        and cross_passed
    )
    if passed:
        classification = (
            "face36_six_mode_dynamic_coordinate_replay_certified_"
            "one_Q_manifest_authorized"
        )
        authorized_next = (
            "WP10c9d6c7c3b5c4f18_definitions_only_one_Q_"
            "constrained_nonlinear_pilot_manifest"
        )
    elif (
        cross["minimum_leading_block_projector_cosine"]
        >= gates["minimum_middle_fine_leading_block_projector_cosine"]
    ):
        classification = (
            "face36_six_mode_dynamic_coordinate_rejected_"
            "leading_two_plus_HMM_manifest_authorized"
        )
        authorized_next = (
            "WP10c9d6c7c3b5c4f18_definitions_only_leading_two_plus_"
            "HMM_closure_manifest"
        )
    else:
        classification = "face36_dynamic_coordinate_basis_rejected_return_to_memory_localization"
        authorized_next = "definitions_only_memory_basis_relocalization_manifest"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "middle": middle_report,
        "fine": fine_report,
        "cross_resolution": {key: value for key, value in cross.items() if not isinstance(value, np.ndarray)},
        "new_nonlinear_trajectory": False,
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
    _save(
        DECISIVE_ARRAYS,
        times=middle["times"],
        middle_state_directions=middle["state_directions"],
        fine_state_directions=fine["state_directions"],
        middle_face36_outputs=middle["face36_outputs"],
        fine_face36_outputs=fine["face36_outputs"],
        middle_amplitude_transitions=middle["amplitude_transitions"],
        fine_amplitude_transitions=fine["amplitude_transitions"],
        middle_Q3_leakage=middle["Q3_leakage"],
        fine_Q3_leakage=fine["Q3_leakage"],
        middle_guard_mapped=middle["guard_mapped"],
        fine_guard_mapped=fine["guard_mapped"],
        middle_guard_height_history=middle["guard_height_history"],
        fine_guard_height_history=fine["guard_height_history"],
        leading_block_projector_cosines=cross["leading_block_projector_cosines"],
        full_subspace_projector_cosines=cross["full_subspace_projector_cosines"],
        weak_block_Procrustes_alignments=cross[
            "weak_block_Procrustes_alignments"
        ],
        fine_amplitude_transitions_aligned=cross[
            "fine_amplitude_transitions_aligned"
        ],
        fine_face36_outputs_aligned=cross["fine_face36_outputs_aligned"],
    )
    _write(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "mode_dimension": MODE_DIMENSION,
            "leading_dimension": LEADING_DIMENSION,
            "audit_targets_seconds": list(AUDIT_TARGETS_SECONDS),
            "prospective_gates": gates,
        },
    )
    _write(SUMMARY_PATH, summary)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Face-36 six-mode dynamic coordinate replay\n\n"
        f"Classification: `{classification}`.\n\n"
        f"Middle/fine method gates: `{middle_report['passed_method_and_single_layout_coordinate_gates']}` / `{fine_report['passed_method_and_single_layout_coordinate_gates']}`.\n\n"
        f"The minimum leading/full state-projector cosines are `{cross['minimum_leading_block_projector_cosine']:.6f}` / `{cross['minimum_full_subspace_projector_cosine']:.6f}`. The fixed-dual amplitude-transition history has cosine `{cross['amplitude_transition_history_cosine']:.6f}` and relative difference `{cross['amplitude_transition_relative_difference']:.6f}`. The face-36 six-mode history has cosine `{cross['face36_mode_history_cosine']:.6f}` and relative difference `{cross['face36_mode_history_relative_difference']:.6f}`.\n\n"
        "No nonlinear state, fixed-Q reaction, 50 ms trajectory, or reduced slow evolution was advanced. The guard complement and raw face-48 rejection remain binding.\n",
        encoding="utf-8",
    )
    _write(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "source_parent_commit": _read(CANONICAL_SUMMARY)["latest_source_parent_commit"],
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "manifest_summary_sha256": _sha(c4f16.SUMMARY_PATH),
            "manifest_sha256": _sha(c4f16.MANIFEST_PATH),
            "basis_sha256": _sha(c4f16.BASIS_PATH),
            "source_hashes": _source_identity(),
        },
    )
    files = (CONFIG_PATH, DECISIVE_ARRAYS, SUMMARY_PATH, PROVENANCE_PATH)
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(path)}  {path.name}\n" for path in files), encoding="utf-8"
    )
    _catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
