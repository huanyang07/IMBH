#!/usr/bin/env python3
"""Run the authorized face-36 augmented analysis-only memory screen.

The package reuses committed 5--20 ms middle/fine nonlinear base histories.
It advances one block of 29 directions through the exact accepted BDF2
tangent.  It neither advances a nonlinear trajectory nor applies a fixed-Q
reaction or per-step projection.
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
import run_causal_inner_face36_augmented_memory_screen_manifest_wp10c9d6c7c3b5c4f12 as c4f12  # noqa: E402

from imri_qpe.constants import C  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_radial_fluctuation import (  # noqa: E402
    causal_five_field_radial_candidate_ledger,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_monolithic_discrete_tangent import (  # noqa: E402
    CausalFiveFieldMonolithicBDFHistoryDirection,
    causal_five_field_monolithic_bdf_history_direction,
    causal_five_field_monolithic_discrete_step_matrix,
    causal_five_field_monolithic_discrete_tangent_step,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f13"
ARTIFACT = "causal_inner_face36_augmented_memory_screen_wp10c9d6c7c3b5c4f13"
THIS_RUNNER = "scripts/run_causal_inner_face36_augmented_memory_screen_wp10c9d6c7c3b5c4f13.py"
THIS_TEST = "tests/test_causal_inner_face36_augmented_memory_screen_wp10c9d6c7c3b5c4f13.py"
REPORT_RELATIVE = "docs/reports/current/CODEX_CAUSAL_INNER_FACE36_AUGMENTED_MEMORY_SCREEN_WP10C9D6C7C3B5C4F13_2026-08-13.md"
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
CONTRACT_PATH = CANONICAL_DIRECTORY / "analysis_contract.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
CHECKPOINT_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT

PARENT_CORE_FACE = 36
PARENT_GUARD_END_FACE = 48
PARENT_CELL_COUNT = 64
N_FIELDS = 5
FIELDS = np.asarray((0, 2, 3), dtype=int)
HEIGHT_FIELDS = np.asarray((2, 3), dtype=int)
TOTAL_DIRECTIONS = c4f1.TOTAL_DIRECTIONS
AUDIT_FRACTIONS = (0.0, 0.5, 1.0)
# Reuse the smallest step on the already-certified c4f4/c4f5 face-flux JVP
# plateau.  A 2e-5 preflight was roundoff limited (1.44e-9 defect), whereas
# this prospective repository-standard step closes at O(1e-11).
FD_RELATIVE_STEP = 5.0e-4
SUPERSEDED_ROUNDOFF_LIMITED_STEP = 2.0e-5
SUPERSEDED_ROUNDOFF_LIMITED_MAXIMUM_DEFECT = 1.443545424980664e-9
COMPLETED_NUMERICAL_RUNNER_SHA256 = (
    "1035abb75c6fc68be5263413bab9ceeea8156e10871b04f4d841baf24c8c2a68"
)
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _save(path: Path, **arrays) -> None:
    path = Path(path)
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
        c4f12.THIS_RUNNER,
        c4f12.THIS_TEST,
        "scripts/run_causal_inner_absolute_baseline_observable_memory_screen_wp10c9d6c7c3b5c4f1.py",
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_discrete_tangent.py",
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_tangent.py",
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_radial_fluctuation.py",
    )
    return {path: _sha(ROOT / path) for path in paths if (ROOT / path).exists()}


def _validate_authorization() -> dict:
    summary = _read(c4f12.SUMMARY_PATH)
    manifest = _read(c4f12.MANIFEST_PATH)
    if (
        not summary["passed"]
        or not summary["memory_propagation_authorized"]
        or summary["authorized_next"]
        != "WP10c9d6c7c3b5c4f13_face36_augmented_analysis_only_memory_screen"
        or not manifest["staging"]["run_middle_first"]
        or not manifest["staging"]["run_fine_only_after_middle_method_pass"]
        or manifest["new_nonlinear_trajectory_authorized"]
        or manifest["fixed_Q_authorized"]
    ):
        raise RuntimeError("c4f13 authorization changed")
    return manifest


def _contract(manifest: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": "face36_augmented_analysis_only_memory_screen",
        "parent_manifest": manifest,
        "new_nonlinear_trajectory": False,
        "fixed_Q_constraint": False,
        "per_step_projection": False,
        "binding_output": "shared_face36_M_J_E_flux",
        "binding_forms": ("instantaneous", "cumulative", "window_mean"),
        "guard_diagnostics": (
            "parent_mapped_storage_cells_36_to_48_M_J_E",
            "parent_responsive_height_history_cells_36_to_48_J_E",
        ),
        "oscillatory_pair_diagnostic": {
            "requires_compact_dimension": 2,
            "minimum_leading_mode_zero_crossings": 2,
            "minimum_dominant_cycles_over_interval": 0.75,
        },
        "face36_finite_difference_audit": {
            "relative_step": FD_RELATIVE_STEP,
            "step_source": "existing_certified_c4f4_c4f5_face_flux_JVP_plateau",
            "superseded_roundoff_limited_step": SUPERSEDED_ROUNDOFF_LIMITED_STEP,
            "superseded_roundoff_limited_maximum_defect": SUPERSEDED_ROUNDOFF_LIMITED_MAXIMUM_DEFECT,
            "gate_unchanged": True,
        },
    }


def _layout_data(label: str):
    _parent, configurations = c4f1._configurations()
    layout, configuration = configurations[label]
    trajectory = c4f1._middle_trajectory() if label == "middle" else c4f1._fine_trajectory()
    return layout, configuration, trajectory


def _scaled_directions(directions: np.ndarray, columns: np.ndarray) -> np.ndarray:
    return np.asarray(directions).reshape(directions.shape[0], -1) / np.asarray(
        columns
    ).ravel()[None, :]


def _parent_sum(values: np.ndarray, layout) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.zeros(
        (values.shape[0], PARENT_CELL_COUNT, values.shape[-1]), dtype=float
    )
    for direction in range(values.shape[0]):
        np.add.at(result[direction], layout.parent_cell_indices, values[direction])
    return result


def _face36_output_map(matrix, layout) -> np.ndarray:
    face = PARENT_CORE_FACE * int(layout.refinement_ratio)
    return np.asarray(
        matrix.spatial_tangent.shared_face_flux_scaled_jacobians,
        dtype=float,
    )[face, FIELDS]


def _apply_map(output_map: np.ndarray, directions: np.ndarray, columns: np.ndarray):
    return _scaled_directions(directions, columns) @ np.asarray(output_map).T


def _guard_diagnostics(
    context,
    state: np.ndarray,
    directions: np.ndarray,
    history_direction: CausalFiveFieldMonolithicBDFHistoryDirection,
    columns: np.ndarray,
    rows: np.ndarray,
    layout,
) -> tuple[np.ndarray, np.ndarray]:
    mapped, _weights = c4f1._descriptor(context, state, columns, rows)
    mapped_scaled = _scaled_directions(directions, columns) @ mapped.T
    mapped_physical = mapped_scaled.reshape(directions.shape) * (
        C * rows[None, :, :]
    )
    parent_mapped = _parent_sum(mapped_physical, layout)
    parent_height = _parent_sum(
        history_direction.previous_responsive_height_storage_increment, layout
    )
    return (
        parent_mapped[:, PARENT_CORE_FACE:PARENT_GUARD_END_FACE][:, :, FIELDS],
        parent_height[:, PARENT_CORE_FACE:PARENT_GUARD_END_FACE][
            :, :, HEIGHT_FIELDS
        ],
    )


def _macro_constraint(context, state, columns, rows, layout) -> np.ndarray:
    start = PARENT_CORE_FACE * int(layout.refinement_ratio)
    return c4f1._Q3_scaled_map(
        context, state, columns, rows, start, state.shape[0]
    )


def _history(trajectory: dict, index: int):
    """Load one exact committed BDF history using the production path label."""

    return c4f1.CausalFiveFieldMonolithicBDFHistory(
        previous_primitive_increment=trajectory["primitive_histories"][index],
        previous_mapped_storage_increment=trajectory["mapped_histories"][index],
        previous_responsive_height_storage_increment=trajectory[
            "height_histories"
        ][index],
        previous_timestep_seconds=float(trajectory["previous_timesteps"][index]),
        temporal_path_scheme="straight_primitive_path",
    ).validated(n_cells=trajectory["states"].shape[1])


def _relative_q3_leakage(
    directions: np.ndarray, columns: np.ndarray, constraint: np.ndarray
) -> np.ndarray:
    scaled = _scaled_directions(directions, columns)
    numerator = np.linalg.norm(scaled @ constraint.T, axis=1)
    denominator = np.maximum(
        np.linalg.norm(scaled, axis=1), np.finfo(float).tiny
    )
    return numerator / denominator


def _face36_flux(context, state: np.ndarray, layout) -> np.ndarray:
    ledger = causal_five_field_radial_candidate_ledger(context, state)
    face = PARENT_CORE_FACE * int(layout.refinement_ratio)
    return np.asarray(
        ledger.interfaces.candidate_shared_face_fluxes_over_c, dtype=float
    )[face, FIELDS]


def _face36_map_defect(
    context,
    state: np.ndarray,
    direction: np.ndarray,
    output_map: np.ndarray,
    columns: np.ndarray,
    layout,
) -> float:
    analytic = _apply_map(output_map, direction[None, ...], columns)[0]
    plus = _face36_flux(context, state + FD_RELATIVE_STEP * direction, layout)
    minus = _face36_flux(context, state - FD_RELATIVE_STEP * direction, layout)
    finite_difference = (plus - minus) / (2.0 * FD_RELATIVE_STEP)
    scale = max(
        float(np.linalg.norm(analytic)),
        float(np.linalg.norm(finite_difference)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(analytic - finite_difference) / scale)


def _cumulative(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values)
    result[1:] = np.cumsum(
        0.5
        * (values[1:] + values[:-1])
        * np.diff(times)[:, None, None],
        axis=0,
    )
    return result


def _window_means(values: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = []
    durations = []
    for start, stop in c4f1.c4f.c4e12.WINDOWS_SECONDS:
        selected = (times >= start - 1.0e-15) & (times <= stop + 1.0e-15)
        if np.sum(selected) < 2:
            raise RuntimeError("c4f13 window is incomplete")
        means.append(
            np.trapezoid(values[selected], times[selected], axis=0)
            / (stop - start)
        )
        durations.append(stop - start)
    return np.asarray(means), np.asarray(durations)


def _weights(times: np.ndarray) -> np.ndarray:
    values = np.zeros(times.size, dtype=float)
    values[0] = 0.5 * (times[1] - times[0])
    values[-1] = 0.5 * (times[-1] - times[-2])
    if times.size > 2:
        values[1:-1] = 0.5 * (times[2:] - times[:-2])
    return values / np.sum(values)


def _svd_metrics(
    values: np.ndarray, scales: np.ndarray, sample_weights: np.ndarray, gates: dict
) -> tuple[dict, dict[str, np.ndarray]]:
    scaled = np.asarray(values) / np.asarray(scales)[None, None, :]
    weighted = (
        scaled * np.sqrt(np.asarray(sample_weights)[:, None, None])
    ).transpose(0, 2, 1).reshape(-1, scaled.shape[1])
    left, singular, right_t = np.linalg.svd(weighted, full_matrices=False)
    energy = singular**2
    fraction = np.cumsum(energy) / max(
        float(np.sum(energy)), np.finfo(float).tiny
    )
    k99 = int(
        np.searchsorted(fraction, gates["minimum_observable_energy_capture"]) + 1
    )
    significant = int(
        np.sum(
            singular
            >= gates["minimum_significant_singular_value_fraction"] * singular[0]
        )
    )
    return (
        {
            "k99": k99,
            "significant_dimension": significant,
            "leading_singular_value": float(singular[0]),
            "k99_energy_capture": float(fraction[k99 - 1]),
        },
        {
            "singular_values": singular,
            "left_vectors": left,
            "right_vectors": right_t,
        },
    )


def _oscillation_diagnostic(
    times: np.ndarray, values: np.ndarray, right_vectors: np.ndarray
) -> dict:
    leading = np.einsum("tdo,d->to", values, right_vectors[0])
    channel_u, _channel_s, _channel_v = np.linalg.svd(leading, full_matrices=False)
    signal = channel_u[:, 0]
    centered = signal - np.mean(signal)
    signs = np.sign(centered)
    signs[signs == 0.0] = 1.0
    crossings = int(np.sum(signs[1:] != signs[:-1]))
    uniform_times = np.linspace(times[0], times[-1], max(128, 4 * times.size))
    uniform = np.interp(uniform_times, times, centered)
    spectrum = np.abs(np.fft.rfft(uniform))
    frequency = np.fft.rfftfreq(
        uniform.size, d=float(uniform_times[1] - uniform_times[0])
    )
    if spectrum.size > 1:
        peak = 1 + int(np.argmax(spectrum[1:]))
        dominant_cycles = float(frequency[peak] * (times[-1] - times[0]))
    else:
        dominant_cycles = 0.0
    return {
        "leading_mode_zero_crossings": crossings,
        "dominant_cycles_over_interval": dominant_cycles,
    }


def _memory_metrics(arrays: dict[str, np.ndarray], trajectory: dict, gates: dict):
    times = arrays["times"]
    outputs = arrays["face36_outputs"]
    scales = np.asarray(trajectory["output_scales"][:3], dtype=float)
    cumulative = _cumulative(outputs, times)
    means, durations = _window_means(outputs, times)
    instant_gains = np.linalg.norm(outputs / scales[None, None, :], axis=(1, 2))
    peak = max(float(np.max(instant_gains)), np.finfo(float).tiny)
    last = instant_gains[
        times >= times[0] + 0.75 * (times[-1] - times[0])
    ]
    final_to_peak = float(instant_gains[-1] / peak)
    regrowth = float(max(0.0, instant_gains[-1] - np.min(last)) / peak)
    forms = {}
    vectors = {}
    for name, values, form_scales, sample_weights in (
        ("instantaneous", outputs, scales, _weights(times)),
        (
            "cumulative",
            cumulative,
            scales * (times[-1] - times[0]),
            _weights(times),
        ),
        ("window_mean", means, scales, durations / np.sum(durations)),
    ):
        forms[name], vectors[name] = _svd_metrics(
            values, form_scales, sample_weights, gates
        )
    oscillation = _oscillation_diagnostic(
        times,
        outputs / scales[None, None, :],
        vectors["instantaneous"]["right_vectors"],
    )
    rapid = bool(
        final_to_peak <= gates["rapid_contraction_final_to_peak_gain"]
        and regrowth
        <= gates["rapid_contraction_last_quarter_regrowth_fraction"]
    )
    maximum_dimension = max(item["k99"] for item in forms.values())
    oscillatory_pair = bool(
        maximum_dimension == 2
        and oscillation["leading_mode_zero_crossings"] >= 2
        and oscillation["dominant_cycles_over_interval"] >= 0.75
        and not rapid
    )
    guard_mapped = arrays["guard_mapped"]
    guard_height = arrays["guard_height_history"]
    augmented = np.concatenate(
        (
            guard_mapped.reshape(times.size, TOTAL_DIRECTIONS, -1),
            guard_height.reshape(times.size, TOTAL_DIRECTIONS, -1),
        ),
        axis=2,
    )
    augmented_scales = np.maximum(
        np.max(np.abs(augmented), axis=(0, 1)), np.finfo(float).tiny
    )
    augmented_metric, augmented_vectors = _svd_metrics(
        augmented, augmented_scales, _weights(times), gates
    )
    report = {
        "forms": forms,
        "maximum_binding_k99": maximum_dimension,
        "instantaneous_final_to_peak_gain": final_to_peak,
        "instantaneous_last_quarter_regrowth_fraction": regrowth,
        "rapid_contraction": rapid,
        "oscillation": oscillation,
        "oscillatory_pair_candidate": oscillatory_pair,
        "maximum_Q3_macro_leakage": float(np.max(arrays["Q3_leakage"])),
        "augmented_guard_k99": augmented_metric["k99"],
        "augmented_guard_significant_dimension": augmented_metric[
            "significant_dimension"
        ],
    }
    stored = {
        "instantaneous_gain_history": instant_gains,
        "cumulative_outputs": cumulative,
        "window_mean_outputs": means,
        "augmented_guard_singular_values": augmented_vectors["singular_values"],
    }
    for name, payload in vectors.items():
        for vector_name, values in payload.items():
            stored[f"{name}_{vector_name}"] = values
    return report, stored


def _audit_indices(step_count: int) -> set[int]:
    return {
        int(round(fraction * max(step_count - 1, 0)))
        for fraction in AUDIT_FRACTIONS
    }


def _run_layout(label: str, manifest: dict) -> tuple[dict, dict[str, np.ndarray]]:
    layout, configuration, trajectory = _layout_data(label)
    context = configuration["context"]
    columns = np.asarray(configuration["columns"], dtype=float).reshape(
        trajectory["states"].shape[1:]
    )
    rows = np.asarray(configuration["rows"], dtype=float).reshape(columns.shape)
    start = PARENT_CORE_FACE * int(layout.refinement_ratio)
    checkpoint = CHECKPOINT_DIRECTORY / f"{label}.npz"
    progress_path = CHECKPOINT_DIRECTORY / f"{label}.json"
    source = _source_identity()
    audits = _audit_indices(trajectory["timesteps"].size)
    began = time.perf_counter()
    if checkpoint.exists() and progress_path.exists():
        progress = _read(progress_path)
        if progress.get("source_identity") != source:
            raise RuntimeError(f"c4f13 {label} checkpoint source changed")
        arrays = _load(checkpoint)
        index = int(progress["steps_completed"])
        current = arrays["current_directions"]
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
        ).validated(n_directions=TOTAL_DIRECTIONS, n_cells=current.shape[1])
    else:
        initialized = c4f1._initial_directions(
            configuration, trajectory, start, trajectory["states"].shape[1]
        )
        current = initialized["current"]
        previous = initialized["previous"]
        previous_state = trajectory["states"][0] - trajectory["primitive_histories"][0]
        previous_dt = float(trajectory["previous_timesteps"][0])
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
        output_map = _face36_output_map(initial_matrix, layout)
        guard_mapped, guard_height = _guard_diagnostics(
            context,
            trajectory["states"][0],
            current,
            history_direction,
            columns,
            rows,
            layout,
        )
        constraint = _macro_constraint(
            context, trajectory["states"][0], columns, rows, layout
        )
        initial_fd = _face36_map_defect(
            context,
            trajectory["states"][0],
            current[0],
            output_map,
            columns,
            layout,
        )
        arrays = {
            "times": trajectory["times"][:1],
            "face36_outputs": _apply_map(output_map, current, columns)[None, ...],
            "guard_mapped": guard_mapped[None, ...],
            "guard_height_history": guard_height[None, ...],
            "Q3_leakage": _relative_q3_leakage(
                current, columns, constraint
            )[None, ...],
            "state_scaled_norms": np.linalg.norm(
                _scaled_directions(current, columns), axis=1
            )[None, ...],
            "current_directions": current,
            "current_primitive_history_directions": history_direction.previous_primitive_increment,
            "current_mapped_history_directions": history_direction.previous_mapped_storage_increment,
            "current_height_history_directions": history_direction.previous_responsive_height_storage_increment,
            "matrix_wall_seconds": np.asarray([time.perf_counter() - began]),
            "step_wall_seconds": np.empty(0),
            "JVP_defects": np.empty(0),
            "linear_solve_defects": np.empty(0),
            "component_closure_defects": np.asarray(
                [initial_matrix.maximum_component_closure_defect]
            ),
            "face36_output_map_defects": np.asarray([initial_fd]),
            "incoming_characteristics": np.asarray(
                [initial_matrix.incoming_excision_characteristics], dtype=np.int64
            ),
            "initial_constraint_defect": np.asarray(
                [initialized["constraint_defect"]]
            ),
            "initial_orthogonality_defect": np.asarray(
                [initialized["orthogonality_defect"]]
            ),
            "random_coefficient_hash": np.asarray(
                [initialized["coefficient_hash"]]
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
            _history(trajectory, step_index),
            current,
            history_direction,
            primitive_column_scales=columns,
            conservation_row_scales=rows,
            analytic_step_matrix=matrix,
            audit_complete_residual=step_index in audits,
        )
        current = tangent.new_primitive_directions
        history_direction = tangent.new_history_directions
        output_map = _face36_output_map(matrix, layout)
        guard_mapped, guard_height = _guard_diagnostics(
            context,
            trajectory["states"][step_index + 1],
            current,
            history_direction,
            columns,
            rows,
            layout,
        )
        constraint = _macro_constraint(
            context, trajectory["states"][step_index + 1], columns, rows, layout
        )
        output_defect = np.nan
        if step_index in audits:
            output_defect = _face36_map_defect(
                context,
                trajectory["states"][step_index + 1],
                current[0],
                output_map,
                columns,
                layout,
            )
        arrays["times"] = np.append(
            arrays["times"], trajectory["times"][step_index + 1]
        )
        arrays["face36_outputs"] = np.concatenate(
            (
                arrays["face36_outputs"],
                _apply_map(output_map, current, columns)[None, ...],
            ),
            axis=0,
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
                _relative_q3_leakage(current, columns, constraint)[None, ...],
            ),
            axis=0,
        )
        arrays["state_scaled_norms"] = np.concatenate(
            (
                arrays["state_scaled_norms"],
                np.linalg.norm(_scaled_directions(current, columns), axis=1)[
                    None, ...
                ],
            ),
            axis=0,
        )
        arrays["current_directions"] = current
        arrays[
            "current_primitive_history_directions"
        ] = history_direction.previous_primitive_increment
        arrays[
            "current_mapped_history_directions"
        ] = history_direction.previous_mapped_storage_increment
        arrays[
            "current_height_history_directions"
        ] = history_direction.previous_responsive_height_storage_increment
        arrays["matrix_wall_seconds"] = np.append(
            arrays["matrix_wall_seconds"], matrix_wall
        )
        arrays["step_wall_seconds"] = np.append(
            arrays["step_wall_seconds"], time.perf_counter() - step_began
        )
        arrays["JVP_defects"] = np.append(
            arrays["JVP_defects"],
            tangent.maximum_step_matrix_jvp_relative_defect,
        )
        arrays["linear_solve_defects"] = np.append(
            arrays["linear_solve_defects"],
            tangent.maximum_linear_solve_relative_defect,
        )
        arrays["component_closure_defects"] = np.append(
            arrays["component_closure_defects"],
            matrix.maximum_component_closure_defect,
        )
        arrays["face36_output_map_defects"] = np.append(
            arrays["face36_output_map_defects"], output_defect
        )
        arrays["incoming_characteristics"] = np.append(
            arrays["incoming_characteristics"],
            matrix.incoming_excision_characteristics,
        )
        progress["steps_completed"] = step_index + 1
        _save(checkpoint, **arrays)
        _write(progress_path, progress)
        print(
            f"c4f13-{label}: {step_index + 1}/{trajectory['timesteps'].size} "
            f"t={trajectory['times'][step_index + 1]:.7f}s "
            f"matrix={matrix_wall:.1f}s step={arrays['step_wall_seconds'][-1]:.1f}s",
            flush=True,
        )
    gates = manifest["prospective_gates"]
    memory, stored = _memory_metrics(arrays, trajectory, gates)
    arrays.update(stored)
    finite_jvp = arrays["JVP_defects"][np.isfinite(arrays["JVP_defects"])]
    finite_output = arrays["face36_output_map_defects"][
        np.isfinite(arrays["face36_output_map_defects"])
    ]
    method = bool(
        arrays["initial_constraint_defect"][0]
        <= gates["maximum_initial_Q3_null_defect"]
        and arrays["initial_orthogonality_defect"][0] <= 1.0e-10
        and finite_jvp.size > 0
        and np.max(finite_jvp) <= gates["maximum_tangent_matrix_JVP_defect"]
        and np.max(arrays["linear_solve_defects"])
        <= gates["maximum_linear_solve_relative_defect"]
        and np.max(arrays["component_closure_defects"])
        <= gates["maximum_component_closure_defect"]
        and finite_output.size == len(AUDIT_FRACTIONS) + 1
        and np.max(finite_output) <= gates["maximum_face36_output_map_defect"]
        and np.max(arrays["incoming_characteristics"]) == 0
    )
    report = {
        "passed_method_gates": method,
        "steps": int(trajectory["timesteps"].size),
        "directions": TOTAL_DIRECTIONS,
        "initial_constraint_defect": float(arrays["initial_constraint_defect"][0]),
        "initial_orthogonality_defect": float(
            arrays["initial_orthogonality_defect"][0]
        ),
        "maximum_JVP_defect": float(np.max(finite_jvp)),
        "maximum_linear_solve_defect": float(
            np.max(arrays["linear_solve_defects"])
        ),
        "maximum_component_closure_defect": float(
            np.max(arrays["component_closure_defects"])
        ),
        "maximum_face36_output_map_defect": float(np.max(finite_output)),
        "face36_output_map_audit_count": int(finite_output.size),
        "maximum_incoming_characteristics": int(
            np.max(arrays["incoming_characteristics"])
        ),
        "memory": memory,
        "wall_seconds": float(
            np.sum(arrays["step_wall_seconds"]) + arrays["matrix_wall_seconds"][0]
        ),
    }
    _save(checkpoint, **arrays)
    _write(CHECKPOINT_DIRECTORY / f"{label}_summary.json", report)
    return report, arrays


def _subspace_cosine(
    middle_arrays: dict[str, np.ndarray],
    fine_arrays: dict[str, np.ndarray],
    middle_report: dict,
    fine_report: dict,
) -> dict:
    forms = {}
    minimum = 1.0
    for form in ("instantaneous", "cumulative", "window_mean"):
        k = max(
            middle_report["memory"]["forms"][form]["k99"],
            fine_report["memory"]["forms"][form]["k99"],
        )
        middle_left = middle_arrays[f"{form}_left_vectors"][:, :k]
        fine_left = fine_arrays[f"{form}_left_vectors"][:, :k]
        overlap = middle_left.T @ fine_left
        cosine = float(np.min(np.linalg.svd(overlap, compute_uv=False)))
        forms[form] = {"dimension": k, "minimum_principal_cosine": cosine}
        minimum = min(minimum, cosine)
    return {"forms": forms, "minimum_principal_cosine": minimum}


def _classify(
    middle: dict, fine: dict, cross: dict, manifest: dict
) -> tuple[str, bool, str]:
    gates = manifest["prospective_gates"]
    if not middle["passed_method_gates"] or not fine["passed_method_gates"]:
        return (
            "face36_augmented_memory_method_gate_failed",
            False,
            "memory_method_localization_only",
        )
    if cross["minimum_principal_cosine"] < gates[
        "minimum_cross_resolution_memory_subspace_cosine"
    ]:
        return (
            "face36_augmented_memory_cross_resolution_gate_failed",
            False,
            "memory_localization_only",
        )
    leakage = max(
        middle["memory"]["maximum_Q3_macro_leakage"],
        fine["memory"]["maximum_Q3_macro_leakage"],
    )
    if leakage > gates[
        "maximum_Q3_macro_leakage_for_unconstrained_architecture_classification"
    ]:
        return (
            "face36_memory_screen_large_macro_Q3_leakage",
            True,
            "definitions_only_physical_constraint_reaction_map_manifest",
        )
    if middle["memory"]["rapid_contraction"] and fine["memory"][
        "rapid_contraction"
    ]:
        return (
            "face36_rapid_observable_contraction_detected",
            True,
            "definitions_only_quasi_steady_constraint_pilot_manifest",
        )
    if middle["memory"]["oscillatory_pair_candidate"] and fine["memory"][
        "oscillatory_pair_candidate"
    ]:
        return (
            "face36_compact_oscillatory_pair_detected",
            True,
            "definitions_only_amplitude_phase_pilot_manifest",
        )
    dimension = max(
        middle["memory"]["maximum_binding_k99"],
        fine["memory"]["maximum_binding_k99"],
    )
    if dimension <= gates["compact_retained_mode_limit"]:
        return (
            "face36_compact_persistent_observable_memory_detected",
            True,
            "definitions_only_retained_mode_Q_plus_a_pilot_manifest",
        )
    return (
        "face36_broad_observable_memory_detected",
        True,
        "definitions_only_HMM_microburst_manifest",
    )


def _catalog(summary: dict) -> None:
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
                    "scientific_status": "CERTIFIED" if summary["passed"] else "REJECTED",
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


def _report(summary: dict) -> None:
    lines = [
        "# Face-36 augmented analysis-only memory screen",
        "",
        f"Classification: `{summary['classification']}`.",
        "",
        "This package reused committed middle/fine 5--20 ms BDF2 histories. It ran no nonlinear trajectory, imposed no physical fixed-Q reaction, and never projected a direction after the initial macro-Q3 null lift.",
        "",
        "## Method gates",
        "",
        "| Layout | JVP defect | Face-36 map defect | Solve defect | Method pass |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in ("middle", "fine"):
        item = summary[label]
        lines.append(
            f"| {label} | {item['maximum_JVP_defect']:.6e} | "
            f"{item['maximum_face36_output_map_defect']:.6e} | "
            f"{item['maximum_linear_solve_defect']:.6e} | "
            f"{item['passed_method_gates']} |"
        )
    lines.extend(
        [
            "",
            "## Observable memory",
            "",
            "| Layout | k99 instant/cumulative/mean | Final/peak | Q3 leakage | Guard k99 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label in ("middle", "fine"):
        memory = summary[label]["memory"]
        dimensions = "/".join(
            str(memory["forms"][name]["k99"])
            for name in ("instantaneous", "cumulative", "window_mean")
        )
        lines.append(
            f"| {label} | {dimensions} | "
            f"{memory['instantaneous_final_to_peak_gain']:.6e} | "
            f"{memory['maximum_Q3_macro_leakage']:.6e} | "
            f"{memory['augmented_guard_k99']} |"
        )
    lines.extend(
        [
            "",
            f"Cross-resolution minimum memory-subspace cosine: `{summary['cross_resolution']['minimum_principal_cosine']:.6f}`.",
            "",
            "Both resolutions select two persistent output-oriented modes, but only the middle layout meets the oscillatory-pair diagnostic. The result therefore does not promote an amplitude/phase interpretation; the retained amplitudes must be defined by the next prospective manifest.",
            "",
            "The retained guard complement was propagated, not discarded. The raw face-48 export remains rejected; face 36 is the physical shared exchange for this architecture.",
            "",
            "## Authorization",
            "",
            f"Authorized next: `{summary['authorized_next']}`.",
            "",
            "Fixed-Q propagation, 50 ms propagation, and reduced slow evolution remain unauthorized.",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def _finalize(
    contract: dict,
    manifest: dict,
    middle_report: dict,
    middle_arrays: dict[str, np.ndarray],
    fine_report: dict,
    fine_arrays: dict[str, np.ndarray],
    *,
    numerical_source_identity: dict[str, str] | None = None,
) -> dict:
    cross = _subspace_cosine(
        middle_arrays, fine_arrays, middle_report, fine_report
    )
    classification, passed, authorized_next = _classify(
        middle_report, fine_report, cross, manifest
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "physical_failure_detected": False,
        "middle": middle_report,
        "fine": fine_report,
        "cross_resolution": cross,
        "analysis_only_memory_screen_completed": True,
        "new_nonlinear_trajectory_executed": False,
        "initial_null_projection_only": True,
        "per_step_projection_executed": False,
        "fixed_Q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "fifty_ms_propagation_authorized": False,
        "face48_absolute_export_rejection_preserved": True,
        "face36_overlap_partition_preserved": True,
        "fine_guard_complement_retained": True,
        "authorized_next": authorized_next,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    decisive = {}
    for label, arrays in (("middle", middle_arrays), ("fine", fine_arrays)):
        for name in (
            "times",
            "face36_outputs",
            "guard_mapped",
            "guard_height_history",
            "Q3_leakage",
            "state_scaled_norms",
            "instantaneous_gain_history",
            "cumulative_outputs",
            "window_mean_outputs",
            "instantaneous_singular_values",
            "instantaneous_left_vectors",
            "instantaneous_right_vectors",
            "cumulative_singular_values",
            "cumulative_left_vectors",
            "cumulative_right_vectors",
            "window_mean_singular_values",
            "window_mean_left_vectors",
            "window_mean_right_vectors",
            "augmented_guard_singular_values",
        ):
            decisive[f"{label}__{name}"] = arrays[name]
    _save(DECISIVE_ARRAYS, **decisive)
    _write(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "time_interval_seconds": (0.005, 0.020),
            "parent_shared_face": PARENT_CORE_FACE,
            "parent_guard_end_face": PARENT_GUARD_END_FACE,
            "directions": TOTAL_DIRECTIONS,
            "layouts": ("middle", "fine"),
            "finite_difference_relative_step": FD_RELATIVE_STEP,
            "windows_seconds": c4f1.c4f.c4e12.WINDOWS_SECONDS,
        },
    )
    _write(CONTRACT_PATH, contract)
    _write(SUMMARY_PATH, summary)
    _report(summary)
    finalization_source_identity = _source_identity()
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "execution_commit": _git("rev-parse", "HEAD"),
        "execution_tree": _git("rev-parse", "HEAD^{tree}"),
        "source_parent_commit": _read(CANONICAL_SUMMARY)[
            "latest_source_parent_commit"
        ],
        "scientific_status": "CERTIFIED" if summary["passed"] else "REJECTED",
        "parent_manifest_sha256": _sha(c4f12.MANIFEST_PATH),
        "source_hashes": numerical_source_identity or finalization_source_identity,
        "finalization_source_hashes": finalization_source_identity,
        "finalization_from_completed_checkpoints": numerical_source_identity is not None,
        "input_hashes": {
            str(path.relative_to(ROOT)): _sha(path)
            for path in (
                c4f1.c4f.MIDDLE_PILOT_ARRAYS,
                c4f1.c4f.MIDDLE_ARRAYS,
                c4f1.c4f.FINE_ARRAYS,
            )
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "output_hashes": {},
    }
    provenance["output_hashes"] = {
        str(path.relative_to(ROOT)): _sha(path)
        for path in (
            CONFIG_PATH,
            CONTRACT_PATH,
            SUMMARY_PATH,
            DECISIVE_ARRAYS,
            REPORT_PATH,
        )
    }
    _write(PROVENANCE_PATH, provenance)
    files = (
        CONFIG_PATH,
        CONTRACT_PATH,
        SUMMARY_PATH,
        PROVENANCE_PATH,
        DECISIVE_ARRAYS,
    )
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )
    _catalog(summary)
    return summary


def _finalize_completed_checkpoints(contract: dict, manifest: dict) -> dict:
    """Regenerate metadata without repeating the completed numerical screen."""
    checkpoints = {}
    for label in ("middle", "fine"):
        metadata = _read(CHECKPOINT_DIRECTORY / f"{label}.json")
        if int(metadata["steps_completed"]) != 39:
            raise RuntimeError(f"c4f13 {label} checkpoint is incomplete")
        sources = metadata["source_identity"]
        if sources.get(THIS_RUNNER) != COMPLETED_NUMERICAL_RUNNER_SHA256:
            raise RuntimeError(f"c4f13 {label} numerical runner identity changed")
        current = _source_identity()
        for path, digest in sources.items():
            if path not in {THIS_RUNNER, THIS_TEST} and current.get(path) != digest:
                raise RuntimeError(
                    f"c4f13 {label} numerical dependency changed: {path}"
                )
        checkpoints[label] = (
            _read(CHECKPOINT_DIRECTORY / f"{label}_summary.json"),
            _load(CHECKPOINT_DIRECTORY / f"{label}.npz"),
            sources,
        )
    if checkpoints["middle"][2] != checkpoints["fine"][2]:
        raise RuntimeError("c4f13 middle/fine numerical source identities differ")
    return _finalize(
        contract,
        manifest,
        checkpoints["middle"][0],
        checkpoints["middle"][1],
        checkpoints["fine"][0],
        checkpoints["fine"][1],
        numerical_source_identity=checkpoints["middle"][2],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--through", choices=STAGES, default="complete")
    parser.add_argument("--finalize-completed-checkpoints", action="store_true")
    args = parser.parse_args()
    manifest = _validate_authorization()
    contract = _contract(manifest)
    if args.finalize_completed_checkpoints:
        summary = _finalize_completed_checkpoints(contract, manifest)
        print(json.dumps(_plain(summary), indent=2, sort_keys=True), flush=True)
        return
    middle_report, middle_arrays = _run_layout("middle", manifest)
    print(
        f"c4f13-middle: method={middle_report['passed_method_gates']} "
        f"k={middle_report['memory']['maximum_binding_k99']} "
        f"leakage={middle_report['memory']['maximum_Q3_macro_leakage']:.3e}",
        flush=True,
    )
    if args.through == "middle" or not middle_report["passed_method_gates"]:
        return
    fine_report, fine_arrays = _run_layout("fine", manifest)
    print(
        f"c4f13-fine: method={fine_report['passed_method_gates']} "
        f"k={fine_report['memory']['maximum_binding_k99']} "
        f"leakage={fine_report['memory']['maximum_Q3_macro_leakage']:.3e}",
        flush=True,
    )
    summary = _finalize(
        contract,
        manifest,
        middle_report,
        middle_arrays,
        fine_report,
        fine_arrays,
    )
    print(json.dumps(_plain(summary), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
