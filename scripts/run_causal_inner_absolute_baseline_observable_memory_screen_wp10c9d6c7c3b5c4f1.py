#!/usr/bin/env python3
"""Run the authorized zero-trajectory Q3 observable-memory screen.

The package reuses the committed 5--20 ms base trajectories and differentiates
their accepted variable-step BDF2 residuals.  It does not advance a nonlinear
state, impose a fixed-Q constraint, or modify the certified physical operator.
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

import run_causal_inner_reduced_architecture_manifest_wp10c9d6c7c3b5c4f as c4f  # noqa: E402
import run_causal_inner_nonlinear_fine_20ms_generic_anchor_wp10c9d6c7c3b5c4e11 as c4e11  # noqa: E402

from imri_qpe.constants import C  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_monolithic_bdf import (  # noqa: E402
    CausalFiveFieldMonolithicBDFHistory,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_monolithic_dae import (  # noqa: E402
    _integrated_mapped_storage,
    _spatial_nodes,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_monolithic_discrete_tangent import (  # noqa: E402
    CausalFiveFieldMonolithicBDFHistoryDirection,
    causal_five_field_monolithic_bdf_history_direction,
    causal_five_field_monolithic_discrete_step_matrix,
    causal_five_field_monolithic_discrete_tangent_step,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_monolithic_tangent import (  # noqa: E402
    _descriptor_matrices,
    _node_reconstruction_weights,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f1"
ANALYZED_CERTIFICATE_COMMIT = c4f.ANALYZED_CERTIFICATE_COMMIT
ANALYZED_CERTIFICATE_PARENT = c4f.ANALYZED_CERTIFICATE_PARENT
ANALYZED_CERTIFICATE_TREE = c4f.ANALYZED_CERTIFICATE_TREE

ARTIFACT = (
    "causal_inner_absolute_baseline_observable_memory_screen_"
    "wp10c9d6c7c3b5c4f1"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_absolute_baseline_observable_memory_screen_"
    "wp10c9d6c7c3b5c4f1.py"
)
THIS_TEST = (
    "tests/test_causal_inner_absolute_baseline_observable_memory_screen_"
    "wp10c9d6c7c3b5c4f1.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ABSOLUTE_BASELINE_OBSERVABLE_"
    "MEMORY_SCREEN_WP10C9D6C7C3B5C4F1_2026-08-13.md"
)
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

LAYOUT_LABELS = tuple(c4f.LAYOUTS)
LAYOUT_KEYS = tuple(c4f.c4e12.c4e9.c4e4.b2b.LAYOUTS)
EXTRACTION_FACES = dict(zip(LAYOUT_LABELS, c4f.EXTRACTION_FACE_INDICES, strict=True))
COUPLING_FACES = dict(zip(LAYOUT_LABELS, c4f.COUPLING_FACE_INDICES, strict=True))
CONSERVATIVE_FIELDS = np.asarray(c4f.CONSERVATIVE_MAPPED_ROWS, dtype=int)
ENERGY_FIELDS = CONSERVATIVE_FIELDS[1:]
N_FIELDS = 5
RANDOM_COUNT = c4f.RANDOM_LIFT_COUNT
PROFILE_COUNT = len(c4f.c4e1.h2b1.PROFILES)
TOTAL_DIRECTIONS = RANDOM_COUNT + PROFILE_COUNT
AUDIT_FRACTIONS = (0.0, 0.5, 1.0)
STAGES = ("absolute", "middle", "fine", "complete")


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _source_identity() -> dict[str, str]:
    paths = (
        THIS_RUNNER,
        THIS_TEST,
        c4f.THIS_RUNNER,
        c4f.THIS_TEST,
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_discrete_tangent.py",
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_tangent.py",
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_monolithic_dae.py",
        "src/imri_qpe/layer3_minidisk_1d/causal_inner_radial_linear_tangent.py",
    )
    return {path: _sha256(ROOT / path) for path in paths if (ROOT / path).exists()}


def _contract() -> dict:
    parent = _read_json(c4f.MANIFEST_PATH)
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": "Q3_analysis_only_absolute_baseline_and_observable_memory_screen",
        "parent_classification": parent["classification"],
        "propagation": "accepted_discrete_BDF_tangent_only",
        "new_nonlinear_trajectory": False,
        "fixed_Q_dynamics": False,
        "slow_coordinate": parent["slow_coordinate_candidates"]["Q3"],
        "augmented_state": parent["augmented_discrete_state"],
        "lift_ensemble": {
            "stored_structured_profiles": PROFILE_COUNT,
            "seeded_smooth_random_profiles": RANDOM_COUNT,
            "total_block_directions": TOTAL_DIRECTIONS,
            "seed": c4f.RANDOM_SEED,
            "initial_Q3_null_projection_only": True,
            "per_step_projection": False,
        },
        "baseline_gates": parent["absolute_baseline_audit"],
        "memory_gates": parent["observable_memory_analysis"],
        "fine_complement_gates": parent["fine_complement_contract"],
        "staged_decision": {
            "Q3_rapid_contraction": "authorize_quasi_steady_pilot_manifest_only",
            "Q3_one_to_three_stable_modes": "authorize_retained_mode_pilot_manifest_only",
            "Q3_oscillatory_pair": "authorize_amplitude_phase_pilot_manifest_only",
            "Q3_noncompact_or_slow_leakage": "authorize_Q4_definitions_or_screen_manifest_only",
            "cross_resolution_failure": "authorize_localization_only",
        },
        "hard_stops": parent["hard_stops"],
    }


def _validate_authorization() -> tuple[dict, dict]:
    summary = _read_json(c4f.SUMMARY_PATH)
    manifest = _read_json(c4f.MANIFEST_PATH)
    if (
        not summary["passed"]
        or summary["authorized_next"]
        != "WP10c9d6c7c3b5c4f1_analysis_only_absolute_baseline_and_observable_memory_screen"
        or not manifest["definitions_only"]
        or manifest["propagation_executed"]
    ):
        raise RuntimeError("c4f1 authorization changed")
    if (
        _git_value("rev-parse", ANALYZED_CERTIFICATE_COMMIT)
        != ANALYZED_CERTIFICATE_COMMIT
        or _git_value("rev-parse", f"{ANALYZED_CERTIFICATE_COMMIT}^")
        != ANALYZED_CERTIFICATE_PARENT
        or _git_value("rev-parse", f"{ANALYZED_CERTIFICATE_COMMIT}^{{tree}}")
        != ANALYZED_CERTIFICATE_TREE
    ):
        raise RuntimeError("c4f1 analyzed certificate identity changed")
    return summary, manifest


def _configurations():
    parent_grid, layouts, configurations = (
        c4f.c4e12.c4e9.c4e4.b2b._layouts_and_contexts(
            c4f.c4e12.c4e9.c4e4.b2b._input_arrays()
        )
    )
    return parent_grid, {
        label: (layouts[key], configurations[key])
        for label, key in zip(LAYOUT_LABELS, LAYOUT_KEYS, strict=True)
    }


def _combine(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if first.shape[1:] != second.shape[1:] or not np.array_equal(first[-1], second[0]):
        raise RuntimeError("c4f1 trajectory chunks do not share an exact endpoint")
    return np.concatenate((first, second[1:]), axis=0)


def _middle_trajectory() -> dict[str, np.ndarray]:
    early = _load_npz(c4f.MIDDLE_PILOT_ARRAYS)
    late = _load_npz(c4f.MIDDLE_ARRAYS)
    result = {}
    for target, name in (
        ("times", "base__accepted_times"),
        ("states", "base__accepted_states"),
        ("primitive_histories", "base__accepted_primitive_histories"),
        ("mapped_histories", "base__accepted_mapped_histories"),
        ("height_histories", "base__accepted_height_histories"),
        ("previous_timesteps", "base__accepted_previous_timesteps"),
        ("structured_states", "tangent__state_directions"),
        ("structured_primitive_histories", "tangent__primitive_history_directions"),
    ):
        result[target] = _combine(early[name], late[name])
    result["timesteps"] = np.concatenate(
        (early["base__accepted_timesteps"], late["base__accepted_timesteps"])
    )
    result["all_structured_outputs"] = np.full(
        (result["times"].size, PROFILE_COUNT, 13), np.nan, dtype=float
    )
    result["all_structured_outputs"][: early["base__accepted_times"].size] = np.nan
    result["all_structured_outputs"][early["base__accepted_times"].size - 1 :] = late[
        "extraction__tangent_directions"
    ]
    result["columns"] = np.asarray(late["tangent__field_scales"], dtype=float)
    result["output_scales"] = np.asarray(late["tangent__export_scales"], dtype=float)
    return result


def _fine_trajectory() -> dict[str, np.ndarray]:
    values = _load_npz(c4f.FINE_ARRAYS)
    return {
        "times": values["base__accepted_times"],
        "states": values["base__accepted_states"],
        "primitive_histories": values["base__accepted_primitive_histories"],
        "mapped_histories": values["base__accepted_mapped_histories"],
        "height_histories": values["base__accepted_height_histories"],
        "previous_timesteps": values["base__accepted_previous_timesteps"],
        "timesteps": values["base__accepted_timesteps"],
        "structured_states": values["tangent__state_directions"],
        "structured_primitive_histories": values[
            "tangent__primitive_history_directions"
        ],
        "all_structured_outputs": values["extraction__tangent_directions"],
        "columns": values["tangent__field_scales"],
        "output_scales": values["tangent__export_scales"],
    }


def _coarse_absolute() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    early = _load_npz(c4f.COARSE_EARLY_ARRAYS)
    late = _load_npz(c4f.COARSE_ARRAYS)
    times = _combine(
        early["base_main__output_times"], late["base_main__output_times"]
    )
    states = _combine(
        early["base_main__output_states"], late["base_main__output_states"]
    )
    outputs = _combine(
        early["base_main__output_extraction_partition"],
        late["base_main__output_extraction_partition"],
    )
    perturbed = _combine(
        early["perturbed_main__output_states"],
        late["perturbed_main__output_states"],
    )
    return times, states, outputs, perturbed


def _indices(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_ids = np.rint(np.asarray(source) * 1.0e6).astype(np.int64)
    result = []
    for value in np.rint(np.asarray(target) * 1.0e6).astype(np.int64):
        matches = np.flatnonzero(source_ids == value)
        if matches.size != 1:
            raise RuntimeError(f"c4f1 time target {value} is not unique")
        result.append(int(matches[0]))
    return np.asarray(result, dtype=int)


def _cumulative(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values)
    result[1:] = np.cumsum(
        0.5 * (values[1:] + values[:-1]) * np.diff(times)[:, None], axis=0
    )
    return result


def _window_means(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    result = []
    for start, stop in c4f.c4e12.WINDOWS_SECONDS:
        selected = (times >= start - 1.0e-15) & (times <= stop + 1.0e-15)
        if np.sum(selected) < 2:
            raise RuntimeError("c4f1 absolute mean window is incomplete")
        result.append(np.trapezoid(values[selected], times[selected], axis=0) / (stop - start))
    return np.asarray(result)


def _triple_metric(values: tuple[np.ndarray, np.ndarray, np.ndarray], scales: np.ndarray) -> dict:
    normalized = tuple(np.asarray(item, dtype=float) / np.asarray(scales, dtype=float) for item in values)
    coarse_middle = normalized[1] - normalized[0]
    middle_fine = normalized[2] - normalized[1]
    coarse_norm = float(np.linalg.norm(coarse_middle))
    fine_norm = float(np.linalg.norm(middle_fine))
    signal = max(*(float(np.linalg.norm(item)) for item in normalized), np.finfo(float).tiny)
    relative = max(coarse_norm, fine_norm) / signal
    floor = float(_read_json(c4f.MANIFEST_PATH)["absolute_baseline_audit"]["relative_observability_floor"])
    if relative <= floor:
        return {
            "observable": False,
            "relative_difference": relative,
            "RMS_order": None,
            "error_direction_cosine": None,
            "passed": True,
        }
    order = float(np.log2(max(coarse_norm, np.finfo(float).tiny) / max(fine_norm, np.finfo(float).tiny)))
    cosine = float(
        np.vdot(coarse_middle.ravel(), middle_fine.ravel()).real
        / max(coarse_norm * fine_norm, np.finfo(float).tiny)
    )
    gates = _read_json(c4f.MANIFEST_PATH)["absolute_baseline_audit"]
    return {
        "observable": True,
        "relative_difference": relative,
        "RMS_order": order,
        "error_direction_cosine": cosine,
        "passed": bool(
            order >= gates["minimum_spatial_RMS_order"]
            and cosine >= gates["minimum_spatial_error_direction_cosine"]
        ),
    }


def _q3_value(context, state: np.ndarray, extraction: int, coupling: int) -> np.ndarray:
    mapped, factors, _nodes = _integrated_mapped_storage(
        context, state, _spatial_nodes(context)
    )
    if not np.array_equal(factors, np.ones_like(factors)):
        raise RuntimeError("c4f1 Q3 evaluation activated reconstruction scaling")
    return np.sum(mapped[extraction:coupling][:, CONSERVATIVE_FIELDS], axis=0)


def _absolute_audit() -> tuple[dict, dict[str, np.ndarray]]:
    began = time.perf_counter()
    parent_grid, configurations = _configurations()
    common_times = _load_npz(c4f.FINAL_ARRAYS)["common_times_seconds"]
    coarse_times, coarse_states_all, coarse_outputs_all, coarse_perturbed_all = _coarse_absolute()
    middle = _middle_trajectory()
    fine = _fine_trajectory()
    histories = {}
    for label, times, states, outputs in (
        ("coarse", coarse_times, coarse_states_all, coarse_outputs_all),
        ("middle", middle["times"], middle["states"], None),
        ("fine", fine["times"], fine["states"], None),
    ):
        selected = _indices(times, common_times)
        layout, configuration = configurations[label]
        selected_states = states[selected]
        restricted = c4f.c4e12.c4e9.c4e4._restrict(selected_states, layout)
        if outputs is None:
            source = _load_npz(c4f.MIDDLE_ARRAYS if label == "middle" else c4f.FINE_ARRAYS)
            if label == "middle":
                early = _load_npz(c4f.MIDDLE_PILOT_ARRAYS)["extraction__base"]
                late = source["extraction__base_values"]
                output_times = middle["times"]
                outputs_combined = _combine(early, late)
            else:
                output_times = fine["times"]
                outputs_combined = source["extraction__base_values"]
            selected_outputs = outputs_combined[_indices(output_times, common_times)]
        else:
            selected_outputs = outputs[selected]
        q3 = np.asarray(
            [
                _q3_value(
                    configuration["context"], state,
                    EXTRACTION_FACES[label], COUPLING_FACES[label]
                )
                for state in selected_states
            ]
        )
        histories[label] = {
            "state": restricted,
            "output": selected_outputs,
            "q3": q3,
        }
    field_scales = np.asarray(_load_npz(c4f.MIDDLE_ARRAYS)["tangent__field_scales"])
    state_scales = field_scales[None, None, :]
    output_scales = np.asarray(_load_npz(c4f.MIDDLE_ARRAYS)["tangent__export_scales"])[None, :]
    q_scales = np.maximum.reduce(
        [np.max(np.abs(histories[label]["q3"]), axis=0) for label in LAYOUT_LABELS]
    )[None, :]
    q_scales = np.maximum(q_scales, np.finfo(float).tiny)
    state_values = tuple(histories[label]["state"] for label in LAYOUT_LABELS)
    output_values = tuple(histories[label]["output"] for label in LAYOUT_LABELS)
    q_values = tuple(histories[label]["q3"] for label in LAYOUT_LABELS)
    cumulative = tuple(_cumulative(item, common_times) for item in output_values)
    means = tuple(_window_means(item, common_times) for item in output_values)
    metrics = {
        "absolute_state": _triple_metric(state_values, state_scales),
        "absolute_Q3": _triple_metric(q_values, q_scales),
        "instantaneous_extraction": _triple_metric(output_values, output_scales),
        "cumulative_extraction": _triple_metric(cumulative, output_scales),
        "window_mean_extraction": _triple_metric(means, output_scales),
    }
    component_metrics = {
        name: _triple_metric(
            tuple(item[:, index : index + 1] for item in output_values),
            output_scales[:, index : index + 1],
        )
        for index, name in enumerate(c4f.c4e12.OBSERVABLE_NAMES)
    }
    final = _load_npz(c4f.FINAL_ARRAYS)
    coarse_selected = _indices(coarse_times, common_times)
    coarse_response = c4f.c4e12.c4e9.c4e4._restrict(
        coarse_perturbed_all[coarse_selected] - coarse_states_all[coarse_selected],
        configurations["coarse"][0],
    )
    reconstruction_defect = float(
        np.max(
            np.abs(coarse_response - final["coarse_state_response"])
            / field_scales[None, None, :]
        )
    )
    gate = _read_json(c4f.MANIFEST_PATH)["absolute_baseline_audit"][
        "baseline_plus_response_relative_defect_gate"
    ]
    passed = bool(all(item["passed"] for item in metrics.values()) and reconstruction_defect <= gate)
    report = {
        "passed": passed,
        "metrics": metrics,
        "instantaneous_extraction_component_metrics": component_metrics,
        "baseline_plus_response_scaled_defect": reconstruction_defect,
        "common_times_seconds": common_times,
        "wall_seconds": time.perf_counter() - began,
    }
    arrays = {"common_times_seconds": common_times, "field_scales": field_scales, "output_scales": output_scales.ravel(), "Q3_scales": q_scales.ravel()}
    for label in LAYOUT_LABELS:
        for name, value in histories[label].items():
            arrays[f"{label}__absolute_{name}"] = value
    return report, arrays


def _descriptor(context, state, columns, rows) -> tuple[np.ndarray, np.ndarray]:
    weights, cells, radii, measures, reconstruction, partition = _node_reconstruction_weights(context, state)
    mapped, _height = _descriptor_matrices(
        context, state, columns, rows, weights, cells, radii, measures
    )
    if reconstruction > 1.0e-12 or partition > 1.0e-12:
        raise RuntimeError("c4f1 descriptor reconstruction contract failed")
    return mapped, weights


def _Q3_scaled_map(context, state, columns, rows, extraction, coupling) -> np.ndarray:
    mapped, _weights = _descriptor(context, state, columns, rows)
    row_values = np.asarray(rows, dtype=float).reshape(state.shape)
    result = []
    for component in CONSERVATIVE_FIELDS:
        selector = np.zeros(state.size, dtype=float)
        for cell in range(extraction, coupling):
            selector[N_FIELDS * cell + component] = C * row_values[cell, component]
        result.append(selector @ mapped)
    result = np.asarray(result)
    norms = np.linalg.norm(result, axis=1)
    if np.any(norms <= np.finfo(float).tiny):
        raise RuntimeError("c4f1 Q3 derivative is rank deficient")
    return result / norms[:, None]


def _project_null(directions: np.ndarray, constraint: np.ndarray) -> tuple[np.ndarray, float]:
    gram = constraint @ constraint.T
    projected = directions - (directions @ constraint.T) @ np.linalg.solve(gram, constraint)
    defect = float(np.max(np.abs(projected @ constraint.T)))
    return projected, defect


def _random_scaled_directions(context, count: int) -> np.ndarray:
    rng = np.random.default_rng(c4f.RANDOM_SEED)
    coefficients = rng.normal(size=(count, N_FIELDS, 6))
    log_r = np.log(context.grid.centers / context.grid.gravitational_radius)
    coordinate = (log_r - log_r.min()) / (log_r.max() - log_r.min())
    basis = np.column_stack(
        (
            np.ones_like(coordinate),
            np.sin(np.pi * coordinate),
            np.cos(np.pi * coordinate),
            np.sin(2.0 * np.pi * coordinate),
            np.cos(2.0 * np.pi * coordinate),
            np.sin(3.0 * np.pi * coordinate),
        )
    )
    values = np.einsum("dfk,nk->dnf", coefficients, basis)
    envelope = np.sin(np.pi * coordinate) ** 2
    values *= envelope[None, :, None]
    return values


def _initial_directions(configuration, trajectory, extraction, coupling):
    context = configuration["context"]
    columns = np.asarray(configuration["columns"], dtype=float).reshape(trajectory["states"].shape[1:])
    rows = np.asarray(configuration["rows"], dtype=float).reshape(columns.shape)
    current_state = trajectory["states"][0]
    previous_state = current_state - trajectory["primitive_histories"][0]
    structured_current = trajectory["structured_states"][0]
    structured_previous = structured_current - trajectory["structured_primitive_histories"][0]
    random_scaled = _random_scaled_directions(context, RANDOM_COUNT)
    current_scaled = np.concatenate(
        (structured_current / columns[None, :, :], random_scaled), axis=0
    ).reshape(TOTAL_DIRECTIONS, -1)
    previous_scaled = np.concatenate(
        (structured_previous / columns[None, :, :], random_scaled), axis=0
    ).reshape(TOTAL_DIRECTIONS, -1)
    current_constraint = _Q3_scaled_map(context, current_state, columns, rows, extraction, coupling)
    previous_constraint = _Q3_scaled_map(context, previous_state, columns, rows, extraction, coupling)
    current_projected, current_defect = _project_null(current_scaled, current_constraint)
    previous_projected, previous_defect = _project_null(previous_scaled, previous_constraint)
    q, r = np.linalg.qr(current_projected.T, mode="reduced")
    transform = np.linalg.solve(r.T, np.eye(TOTAL_DIRECTIONS))
    current_orthonormal = transform @ current_projected
    previous_transformed = transform @ previous_projected
    orthogonality = float(
        np.max(np.abs(current_orthonormal @ current_orthonormal.T - np.eye(TOTAL_DIRECTIONS)))
    )
    current_physical = current_orthonormal.reshape(TOTAL_DIRECTIONS, *columns.shape) * columns[None, :, :]
    previous_physical = previous_transformed.reshape(TOTAL_DIRECTIONS, *columns.shape) * columns[None, :, :]
    return {
        "current": current_physical,
        "previous": previous_physical,
        "constraint_defect": max(current_defect, previous_defect),
        "orthogonality_defect": orthogonality,
        "coefficient_hash": _array_sha256(random_scaled),
    }


def _history(values, index: int) -> CausalFiveFieldMonolithicBDFHistory:
    return CausalFiveFieldMonolithicBDFHistory(
        previous_primitive_increment=values["primitive_histories"][index],
        previous_mapped_storage_increment=values["mapped_histories"][index],
        previous_responsive_height_storage_increment=values["height_histories"][index],
        previous_timestep_seconds=float(values["previous_timesteps"][index]),
        temporal_path_scheme="straight_primitive_path_gauss_legendre",
    ).validated(n_cells=values["states"].shape[1])


def _exterior_output_map(matrix, extraction: int, coupling: int) -> tuple[np.ndarray, dict]:
    spatial = matrix.spatial_tangent
    cells = spatial.base_primitives.shape[0]
    dimensions = spatial.base_primitives.size
    rows = np.asarray(spatial.conservation_row_scales).reshape(cells, N_FIELDS)
    face = np.asarray(spatial.shared_face_flux_scaled_jacobians)
    stationary = (np.asarray(spatial.candidate_stationary_scaled_jacobian) * rows.ravel()[:, None]).reshape(cells, N_FIELDS, dimensions)
    cooling = (np.asarray(spatial.block_scaled_jacobians["candidate_cooling"]) * rows.ravel()[:, None]).reshape(cells, N_FIELDS, dimensions)
    height = (np.asarray(spatial.block_scaled_jacobians["candidate_lower_height_work"]) * rows.ravel()[:, None]).reshape(cells, N_FIELDS, dimensions)
    region = slice(extraction, coupling)
    observable = np.concatenate(
        (
            face[extraction, CONSERVATIVE_FIELDS],
            face[coupling, CONSERVATIVE_FIELDS],
            -np.sum(stationary[region][:, CONSERVATIVE_FIELDS], axis=0),
            -np.sum(cooling[region][:, ENERGY_FIELDS], axis=0),
            -np.sum(height[region][:, ENERGY_FIELDS], axis=0),
        ), axis=0,
    )
    transport = (np.asarray(spatial.block_scaled_jacobians["candidate_conservative_transport"]) * rows.ravel()[:, None]).reshape(cells, N_FIELDS, dimensions)
    telescoped = face[extraction, CONSERVATIVE_FIELDS] - face[coupling, CONSERVATIVE_FIELDS]
    direct = -np.sum(transport[region][:, CONSERVATIVE_FIELDS], axis=0)
    scale = max(float(np.linalg.norm(telescoped)), float(np.linalg.norm(direct)), np.finfo(float).tiny)
    return observable, {
        "transport_telescoping_defect": float(np.linalg.norm(telescoped - direct) / scale),
        "incoming_excision_characteristics": int(spatial.incoming_inner_characteristics),
    }


def _apply_output(output_map, physical_directions, columns):
    scaled = physical_directions.reshape(physical_directions.shape[0], -1) / np.asarray(columns).ravel()[None, :]
    return scaled @ output_map.T


def _memory_metrics(times, outputs, slow_leakage, output_scales, manifest):
    scaled = outputs / output_scales[None, None, :]
    gains = np.linalg.norm(scaled, axis=(1, 2))
    peak = max(float(np.max(gains)), np.finfo(float).tiny)
    final_to_peak = float(gains[-1] / peak)
    last = gains[times >= times[0] + 0.75 * (times[-1] - times[0])]
    regrowth = float(max(0.0, gains[-1] - np.min(last)) / peak)
    weights = np.zeros(times.size)
    weights[0] = 0.5 * (times[1] - times[0])
    weights[-1] = 0.5 * (times[-1] - times[-2])
    if times.size > 2:
        weights[1:-1] = 0.5 * (times[2:] - times[:-2])
    weighted = (scaled * np.sqrt(weights[:, None, None] / (times[-1] - times[0]))).transpose(0, 2, 1).reshape(-1, outputs.shape[1])
    left, singular, right_t = np.linalg.svd(weighted, full_matrices=False)
    energy = singular**2
    fractions = np.cumsum(energy) / max(float(np.sum(energy)), np.finfo(float).tiny)
    k99 = int(np.searchsorted(fractions, manifest["minimum_observable_energy_capture"]) + 1)
    significant = int(np.sum(singular >= manifest["minimum_significant_singular_value_fraction"] * singular[0])) if singular.size else 0
    rapid = bool(
        final_to_peak <= manifest["rapid_contraction_final_to_peak_gain"]
        and regrowth <= manifest["rapid_contraction_last_quarter_regrowth_fraction"]
    )
    return {
        "gain_history": gains,
        "final_to_peak_gain": final_to_peak,
        "last_quarter_regrowth_fraction": regrowth,
        "singular_values": singular,
        "left_vectors": left,
        "right_vectors": right_t,
        "k99": k99,
        "significant_dimension": significant,
        "rapid_contraction": rapid,
        "maximum_Q3_leakage": float(np.max(slow_leakage)),
    }


def _run_layout(label: str, configuration: dict, layout, trajectory: dict) -> tuple[dict, dict[str, np.ndarray]]:
    checkpoint = CHECKPOINT_DIRECTORY / f"{label}.npz"
    progress_path = CHECKPOINT_DIRECTORY / f"{label}.json"
    source = _source_identity()
    context = configuration["context"]
    columns = np.asarray(configuration["columns"], dtype=float).reshape(trajectory["states"].shape[1:])
    rows = np.asarray(configuration["rows"], dtype=float).reshape(columns.shape)
    extraction = EXTRACTION_FACES[label]
    coupling = COUPLING_FACES[label]
    audit_indices = {
        int(round(fraction * max(trajectory["timesteps"].size - 1, 0)))
        for fraction in AUDIT_FRACTIONS
    }
    began = time.perf_counter()
    if checkpoint.exists() and progress_path.exists():
        progress = _read_json(progress_path)
        if progress.get("source_identity") != source:
            raise RuntimeError(f"c4f1 {label} checkpoint source changed")
        arrays = _load_npz(checkpoint)
        index = int(progress["steps_completed"])
        current = arrays["current_directions"]
        history_direction = CausalFiveFieldMonolithicBDFHistoryDirection(
            previous_primitive_increment=arrays["current_primitive_history_directions"],
            previous_mapped_storage_increment=arrays["current_mapped_history_directions"],
            previous_responsive_height_storage_increment=arrays["current_height_history_directions"],
        ).validated(n_directions=TOTAL_DIRECTIONS, n_cells=current.shape[1])
    else:
        initialized = _initial_directions(configuration, trajectory, extraction, coupling)
        current = initialized["current"]
        previous = initialized["previous"]
        previous_state = trajectory["states"][0] - trajectory["primitive_histories"][0]
        previous_dt = float(trajectory["previous_timesteps"][0])
        initial_matrix = causal_five_field_monolithic_discrete_step_matrix(
            context, previous_state, trajectory["states"][0], previous_dt, previous_dt,
            primitive_column_scales=columns, conservation_row_scales=rows,
        )
        history_direction = causal_five_field_monolithic_bdf_history_direction(
            context, previous_state, trajectory["states"][0], previous, current,
            analytic_step_matrix=initial_matrix,
        )
        output_map, audit = _exterior_output_map(initial_matrix, extraction, coupling)
        outputs = _apply_output(output_map, current, columns)[None, ...]
        constraint = _Q3_scaled_map(context, trajectory["states"][0], columns, rows, extraction, coupling)
        slow = np.max(np.abs((current.reshape(TOTAL_DIRECTIONS, -1) / columns.ravel()[None, :]) @ constraint.T), axis=1)[None, :]
        structured_analytic = _apply_output(output_map, trajectory["structured_states"][0], columns)[None, ...]
        arrays = {
            "times": trajectory["times"][:1],
            "outputs": outputs,
            "Q3_leakage": slow,
            "state_scaled_norms": np.linalg.norm(current.reshape(TOTAL_DIRECTIONS, -1) / columns.ravel()[None, :], axis=1)[None, :],
            "structured_analytic_outputs": structured_analytic,
            "current_directions": current,
            "current_primitive_history_directions": history_direction.previous_primitive_increment,
            "current_mapped_history_directions": history_direction.previous_mapped_storage_increment,
            "current_height_history_directions": history_direction.previous_responsive_height_storage_increment,
            "matrix_wall_seconds": np.asarray([time.perf_counter() - began]),
            "step_wall_seconds": np.empty(0),
            "JVP_defects": np.empty(0),
            "linear_solve_defects": np.empty(0),
            "component_closure_defects": np.asarray([initial_matrix.maximum_component_closure_defect]),
            "transport_telescoping_defects": np.asarray([audit["transport_telescoping_defect"]]),
            "incoming_characteristics": np.asarray([audit["incoming_excision_characteristics"]], dtype=np.int64),
            "initial_constraint_defect": np.asarray([initialized["constraint_defect"]]),
            "initial_orthogonality_defect": np.asarray([initialized["orthogonality_defect"]]),
            "random_coefficient_hash": np.asarray([initialized["coefficient_hash"]]),
        }
        index = 0
        progress = {"source_identity": source, "steps_completed": 0, "started_wall_seconds": time.time()}
        _atomic_npz(checkpoint, **arrays)
        _write_json(progress_path, progress)
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
            audit_complete_residual=step_index in audit_indices,
        )
        current = tangent.new_primitive_directions
        history_direction = tangent.new_history_directions
        output_map, audit = _exterior_output_map(matrix, extraction, coupling)
        output = _apply_output(output_map, current, columns)
        constraint = _Q3_scaled_map(
            context, trajectory["states"][step_index + 1], columns, rows, extraction, coupling
        )
        current_scaled = current.reshape(TOTAL_DIRECTIONS, -1) / columns.ravel()[None, :]
        slow = np.max(np.abs(current_scaled @ constraint.T), axis=1)
        structured = _apply_output(
            output_map, trajectory["structured_states"][step_index + 1], columns
        )
        arrays["times"] = np.append(arrays["times"], trajectory["times"][step_index + 1])
        arrays["outputs"] = np.concatenate((arrays["outputs"], output[None, ...]), axis=0)
        arrays["Q3_leakage"] = np.concatenate((arrays["Q3_leakage"], slow[None, :]), axis=0)
        arrays["state_scaled_norms"] = np.concatenate((arrays["state_scaled_norms"], np.linalg.norm(current_scaled, axis=1)[None, :]), axis=0)
        arrays["structured_analytic_outputs"] = np.concatenate((arrays["structured_analytic_outputs"], structured[None, ...]), axis=0)
        arrays["current_directions"] = current
        arrays["current_primitive_history_directions"] = history_direction.previous_primitive_increment
        arrays["current_mapped_history_directions"] = history_direction.previous_mapped_storage_increment
        arrays["current_height_history_directions"] = history_direction.previous_responsive_height_storage_increment
        arrays["matrix_wall_seconds"] = np.append(arrays["matrix_wall_seconds"], matrix_wall)
        arrays["step_wall_seconds"] = np.append(arrays["step_wall_seconds"], time.perf_counter() - step_began)
        arrays["JVP_defects"] = np.append(arrays["JVP_defects"], tangent.maximum_step_matrix_jvp_relative_defect)
        arrays["linear_solve_defects"] = np.append(arrays["linear_solve_defects"], tangent.maximum_linear_solve_relative_defect)
        arrays["component_closure_defects"] = np.append(arrays["component_closure_defects"], matrix.maximum_component_closure_defect)
        arrays["transport_telescoping_defects"] = np.append(arrays["transport_telescoping_defects"], audit["transport_telescoping_defect"])
        arrays["incoming_characteristics"] = np.append(arrays["incoming_characteristics"], audit["incoming_excision_characteristics"])
        progress["steps_completed"] = step_index + 1
        _atomic_npz(checkpoint, **arrays)
        _write_json(progress_path, progress)
        print(
            f"c4f1-{label}: {step_index + 1}/{trajectory['timesteps'].size} "
            f"t={trajectory['times'][step_index + 1]:.7f}s matrix={matrix_wall:.1f}s "
            f"step={arrays['step_wall_seconds'][-1]:.1f}s",
            flush=True,
        )
    manifest = _read_json(c4f.MANIFEST_PATH)["observable_memory_analysis"]
    metrics = _memory_metrics(
        arrays["times"], arrays["outputs"], arrays["Q3_leakage"],
        trajectory["output_scales"], manifest,
    )
    finite_jvp = arrays["JVP_defects"][np.isfinite(arrays["JVP_defects"])]
    analytic_closure = float(
        np.nanmax(
            np.abs(
                arrays["structured_analytic_outputs"]
                - trajectory["all_structured_outputs"]
            )
            / trajectory["output_scales"][None, None, :]
        )
    ) if np.any(np.isfinite(trajectory["all_structured_outputs"])) else None
    report = {
        "passed_method_gates": bool(
            arrays["initial_constraint_defect"][0] <= manifest["constraint_null_defect_gate"]
            and arrays["initial_orthogonality_defect"][0] <= 1.0e-10
            and (finite_jvp.size == 0 or np.max(finite_jvp) <= manifest["existing_tangent_matrix_JVP_gate"])
            and np.max(arrays["linear_solve_defects"]) <= manifest["linear_solve_relative_defect_gate"]
            and np.max(arrays["component_closure_defects"]) <= 1.0e-12
            and np.max(arrays["transport_telescoping_defects"]) <= 1.0e-12
            and np.max(arrays["incoming_characteristics"]) == 0
        ),
        "steps": int(trajectory["timesteps"].size),
        "directions": TOTAL_DIRECTIONS,
        "initial_constraint_defect": float(arrays["initial_constraint_defect"][0]),
        "initial_orthogonality_defect": float(arrays["initial_orthogonality_defect"][0]),
        "maximum_JVP_defect": float(np.max(finite_jvp)) if finite_jvp.size else None,
        "maximum_linear_solve_defect": float(np.max(arrays["linear_solve_defects"])),
        "maximum_component_closure_defect": float(np.max(arrays["component_closure_defects"])),
        "maximum_transport_telescoping_defect": float(np.max(arrays["transport_telescoping_defects"])),
        "maximum_incoming_characteristics": int(np.max(arrays["incoming_characteristics"])),
        "structured_analytic_vs_committed_output_scaled_defect": analytic_closure,
        "memory": {key: value for key, value in metrics.items() if key not in {"gain_history", "singular_values", "left_vectors", "right_vectors"}},
        "wall_seconds": float(np.sum(arrays["step_wall_seconds"]) + arrays["matrix_wall_seconds"][0]),
    }
    arrays["memory_gain_history"] = metrics["gain_history"]
    arrays["memory_singular_values"] = metrics["singular_values"]
    arrays["memory_left_vectors"] = metrics["left_vectors"]
    arrays["memory_right_vectors"] = metrics["right_vectors"]
    _atomic_npz(checkpoint, **arrays)
    return report, arrays


def _fine_complement(fine_arrays, fine_trajectory, middle_arrays, fine_layout, columns, output_scales):
    total = fine_trajectory["structured_states"]
    restricted = c4f.c4e12.c4e9.c4e4._restrict(total.reshape((-1,) + total.shape[2:]), fine_layout).reshape(total.shape[0], total.shape[1], -1, N_FIELDS)
    prolonged = restricted[:, :, fine_layout.parent_cell_indices, :]
    complement = total - prolonged
    state_fraction = np.linalg.norm((complement / columns[None, None, :, :]).reshape(total.shape[0], total.shape[1], -1), axis=2) / np.maximum(
        np.linalg.norm((total / columns[None, None, :, :]).reshape(total.shape[0], total.shape[1], -1), axis=2), np.finfo(float).tiny
    )
    # The analytic complement output is reconstructed from the stored total and
    # the fine-only state fraction.  A targeted follow-up is required if the
    # conservative upper bound itself fails either prospective gate.
    total_output = fine_arrays["structured_analytic_outputs"]
    complement_bound = state_fraction[:, :, None] * np.abs(total_output)
    output_fraction = np.linalg.norm(complement_bound / output_scales[None, None, :], axis=2) / np.maximum(
        np.linalg.norm(total_output / output_scales[None, None, :], axis=2), np.finfo(float).tiny
    )
    middle_output = middle_arrays["structured_analytic_outputs"]
    spatial = np.linalg.norm((total_output - middle_output) / output_scales[None, None, :], axis=2)
    spatial_fraction = np.linalg.norm(complement_bound / output_scales[None, None, :], axis=2) / np.maximum(spatial, np.finfo(float).tiny)
    gates = _read_json(c4f.MANIFEST_PATH)["fine_complement_contract"]
    return {
        "maximum_state_energy_fraction": float(np.max(state_fraction)),
        "maximum_conservative_output_fraction_bound": float(np.max(output_fraction)),
        "maximum_fraction_of_middle_fine_spatial_difference_bound": float(np.max(spatial_fraction)),
        "passed": bool(
            np.max(output_fraction) <= gates["maximum_discarded_output_fraction"]
            and np.max(spatial_fraction) <= gates["maximum_fraction_of_middle_fine_spatial_difference"]
        ),
        "classification": "conservative_bound_requires_targeted_exact_JVP_if_failed",
    }


def _cross_resolution(middle_report, middle_arrays, fine_report, fine_arrays):
    k = max(int(middle_report["memory"]["k99"]), int(fine_report["memory"]["k99"]))
    k = min(k, middle_arrays["memory_left_vectors"].shape[1], fine_arrays["memory_left_vectors"].shape[1])
    overlap = middle_arrays["memory_left_vectors"][:, :k].T @ fine_arrays["memory_left_vectors"][:, :k]
    cosine = float(np.min(np.linalg.svd(overlap, compute_uv=False))) if k else 1.0
    gate = _read_json(c4f.MANIFEST_PATH)["observable_memory_analysis"]["minimum_cross_resolution_subspace_cosine"]
    return {"compared_dimension": k, "minimum_principal_cosine": cosine, "passed": bool(cosine >= gate)}


def _classify(absolute, middle, fine, cross, complement):
    manifest = _read_json(c4f.MANIFEST_PATH)["observable_memory_analysis"]
    method = absolute["passed"] and middle["passed_method_gates"] and fine["passed_method_gates"]
    if not method or not cross["passed"]:
        return "Q3_screen_failed_localization_only", False, "localization_manifest_only"
    if not complement["passed"]:
        return "Q3_screen_inconclusive_fine_complement_followup_required", False, "targeted_fine_complement_JVP_manifest"
    if middle["memory"]["rapid_contraction"] and fine["memory"]["rapid_contraction"]:
        return "Q3_rapid_unique_observable_contraction_detected", True, "definitions_only_quasi_steady_fixed_Q_pilot_manifest"
    compact = max(middle["memory"]["k99"], fine["memory"]["k99"]) <= manifest["compact_retained_mode_limit"]
    if compact:
        return "Q3_compact_persistent_observable_memory_detected", True, "definitions_only_retained_mode_Q_plus_a_pilot_manifest"
    return "Q3_noncompact_or_leaky_memory_Q4_screen_manifest_authorized", True, "definitions_only_Q4_observable_memory_screen_manifest"


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
                    "sha256": _sha256(path),
                    "scientific_status": "CERTIFIED" if summary["passed"] else "REJECTED",
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
        "classification": summary["classification"],
        "passed": summary["passed"],
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
    }
    catalog.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": ANALYZED_CERTIFICATE_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def _refresh_absolute_component_metrics(report: dict, arrays: dict[str, np.ndarray]) -> dict:
    if "instantaneous_extraction_component_metrics" in report:
        return report
    output_scales = arrays["output_scales"][None, :]
    values = tuple(
        arrays[f"{label}__absolute_output"] for label in LAYOUT_LABELS
    )
    report["instantaneous_extraction_component_metrics"] = {
        name: _triple_metric(
            tuple(item[:, index : index + 1] for item in values),
            output_scales[:, index : index + 1],
        )
        for index, name in enumerate(c4f.c4e12.OBSERVABLE_NAMES)
    }
    return report


def _write_absolute_stop(contract, absolute_report, absolute_arrays) -> dict:
    failing = [
        name
        for name, value in absolute_report["metrics"].items()
        if not value["passed"]
    ]
    component_failures = [
        name
        for name, value in absolute_report[
            "instantaneous_extraction_component_metrics"
        ].items()
        if not value["passed"]
    ]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "absolute_extraction_baseline_direction_gate_failed_"
            "observable_memory_propagation_not_executed"
        ),
        "passed": False,
        "physical_failure_detected": False,
        "absolute_baseline": absolute_report,
        "failed_absolute_channels": failing,
        "failed_instantaneous_components": component_failures,
        "absolute_state_and_Q3_storage_passed": bool(
            absolute_report["metrics"]["absolute_state"]["passed"]
            and absolute_report["metrics"]["absolute_Q3"]["passed"]
        ),
        "observable_memory_propagation_executed": False,
        "new_nonlinear_trajectory_executed": False,
        "Q3_screen_completed": False,
        "fixed_Q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "fifty_ms_propagation_authorized": False,
        "authorized_next": (
            "definitions_only_absolute_baseline_anchor_or_coupling_"
            "localization_manifest"
        ),
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _atomic_npz(DECISIVE_ARRAYS, **absolute_arrays)
    _write_json(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "analyzed_certificate_commit": ANALYZED_CERTIFICATE_COMMIT,
            "layouts": LAYOUT_LABELS,
            "memory_propagation_skipped_by_fail_fast_gate": True,
        },
    )
    _write_json(CONTRACT_PATH, contract)
    _write_json(SUMMARY_PATH, summary)
    component_rows = [
        "| Component | Order | Cosine | Pass |",
        "|---|---:|---:|---:|",
    ]
    for name, value in absolute_report[
        "instantaneous_extraction_component_metrics"
    ].items():
        component_rows.append(
            f"| {name} | {value['RMS_order']:.6f} | "
            f"{value['error_direction_cosine']:.6f} | {value['passed']} |"
        )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Q3 absolute-baseline and observable-memory screen",
                "",
                "## Result",
                "",
                f"Classification: `{summary['classification']}`.",
                "",
                "The fail-fast absolute audit stopped the package before any middle or fine memory propagation. No nonlinear trajectory or fixed-Q evolution ran.",
                "",
                "The absolute state and exact mapped Q3 storage converge at about second order. The extraction-surface flux, cooling, and responsive-height components also have consistent refinement directions. The coupling-face M/J/E flux and the derived net-drive components contract in magnitude but reverse direction between the coarse-middle and middle-fine pairs, so the predeclared absolute-baseline cosine gate fails.",
                "",
                "## Instantaneous component localization",
                "",
                *component_rows,
                "",
                "## Decision",
                "",
                "The certified 20 ms response result remains valid, but an absolute slow closure is not authorized. The next package must freeze either a fine-anchored absolute-baseline decomposition or a focused coupling-face baseline localization; it may not relax this result after inspection.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "analyzed_certificate_commit": ANALYZED_CERTIFICATE_COMMIT,
        "analyzed_certificate_parent": ANALYZED_CERTIFICATE_PARENT,
        "analyzed_certificate_tree": ANALYZED_CERTIFICATE_TREE,
        "analysis_execution_head": _git_value("rev-parse", "HEAD"),
        "source_identity": _source_identity(),
        "input_hashes": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (
                c4f.COARSE_EARLY_ARRAYS,
                c4f.COARSE_ARRAYS,
                c4f.MIDDLE_PILOT_ARRAYS,
                c4f.MIDDLE_ARRAYS,
                c4f.FINE_ARRAYS,
                c4f.FINAL_ARRAYS,
            )
        },
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "output_hashes": {},
    }
    provenance["output_hashes"] = {
        str(path.relative_to(ROOT)): _sha256(path)
        for path in (CONFIG_PATH, CONTRACT_PATH, SUMMARY_PATH, DECISIVE_ARRAYS, REPORT_PATH)
    }
    _write_json(PROVENANCE_PATH, provenance)
    checksums = CANONICAL_DIRECTORY / "SHA256SUMS.txt"
    checksums.write_text(
        "".join(
            f"{_sha256(path)}  {path.name}\n"
            for path in (
                CONFIG_PATH,
                CONTRACT_PATH,
                SUMMARY_PATH,
                PROVENANCE_PATH,
                DECISIVE_ARRAYS,
            )
        ),
        encoding="utf-8",
    )
    _catalog(summary)
    return summary


def _report(summary: dict) -> None:
    absolute = summary["absolute_baseline"]
    lines = [
        "# Q3 absolute-baseline and observable-memory screen",
        "",
        f"Analyzed certificate: `{ANALYZED_CERTIFICATE_COMMIT}`.",
        "",
        "## Result",
        "",
        f"Classification: `{summary['classification']}`.",
        "",
        "This package ran no nonlinear trajectory and imposed no fixed-Q constraint. It reused the accepted middle/fine BDF2 histories, projected the initial block directions into the exact mapped Q3 null space, and then propagated them without per-step reprojection.",
        "",
        "## Absolute baseline",
        "",
        f"Absolute baseline pass: `{absolute['passed']}`; baseline-plus-response scaled defect: `{absolute['baseline_plus_response_scaled_defect']:.6e}`.",
        "",
        "| Channel | Observable | RMS order | Error cosine | Pass |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metric in absolute["metrics"].items():
        order = "upper bound" if metric["RMS_order"] is None else f"{metric['RMS_order']:.6f}"
        cosine = "n/a" if metric["error_direction_cosine"] is None else f"{metric['error_direction_cosine']:.6f}"
        lines.append(f"| {name} | {metric['observable']} | {order} | {cosine} | {metric['passed']} |")
    lines.extend(["", "## Q3 memory", ""])
    for label in ("middle", "fine"):
        item = summary[label]
        memory = item["memory"]
        lines.extend([
            f"### {label.capitalize()}", "",
            f"Method gates: `{item['passed_method_gates']}`; directions: `{item['directions']}`; k99: `{memory['k99']}`; final/peak gain: `{memory['final_to_peak_gain']:.6e}`; maximum Q3 leakage: `{memory['maximum_Q3_leakage']:.6e}`.", "",
        ])
    lines.extend([
        "## Cross-resolution and fine complement", "",
        f"Cross-resolution minimum principal cosine: `{summary['cross_resolution']['minimum_principal_cosine']:.6f}`.", "",
        f"Fine-complement conservative output bound: `{summary['fine_complement']['maximum_conservative_output_fraction_bound']:.6e}`; fraction of middle-fine spatial difference bound: `{summary['fine_complement']['maximum_fraction_of_middle_fine_spatial_difference_bound']:.6e}`.", "",
        "## Authorization", "",
        f"Authorized next: `{summary['authorized_next']}`.", "",
        "Fixed-Q propagation, a 50/125 ms duration run, and reduced slow evolution remain unauthorized.", "",
    ])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def _finalize(contract, parent_summary, absolute_report, absolute_arrays, middle_report, middle_arrays, fine_report, fine_arrays, configurations):
    cross = _cross_resolution(middle_report, middle_arrays, fine_report, fine_arrays)
    fine_layout, fine_configuration = configurations["fine"]
    complement = _fine_complement(
        fine_arrays, _fine_trajectory(), middle_arrays, fine_layout,
        np.asarray(fine_configuration["columns"]).reshape(fine_layout.grid.centers.size, N_FIELDS),
        _fine_trajectory()["output_scales"],
    )
    classification, passed, authorized_next = _classify(
        absolute_report, middle_report, fine_report, cross, complement
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "physical_failure_detected": False,
        "absolute_baseline": absolute_report,
        "middle": middle_report,
        "fine": fine_report,
        "cross_resolution": cross,
        "fine_complement": complement,
        "Q3_screen_completed": True,
        "fixed_Q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "fifty_ms_propagation_authorized": False,
        "authorized_next": authorized_next,
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    arrays = dict(absolute_arrays)
    for label, values in (("middle", middle_arrays), ("fine", fine_arrays)):
        for name in (
            "times", "outputs", "Q3_leakage", "state_scaled_norms",
            "memory_gain_history", "memory_singular_values", "memory_left_vectors",
            "memory_right_vectors", "structured_analytic_outputs",
        ):
            arrays[f"{label}__{name}"] = values[name]
    _atomic_npz(DECISIVE_ARRAYS, **arrays)
    _write_json(CONFIG_PATH, {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "analyzed_certificate_commit": ANALYZED_CERTIFICATE_COMMIT,
        "directions": TOTAL_DIRECTIONS,
        "layouts": LAYOUT_LABELS,
    })
    _write_json(CONTRACT_PATH, contract)
    _write_json(SUMMARY_PATH, summary)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "analyzed_certificate_commit": ANALYZED_CERTIFICATE_COMMIT,
        "analyzed_certificate_parent": ANALYZED_CERTIFICATE_PARENT,
        "analyzed_certificate_tree": ANALYZED_CERTIFICATE_TREE,
        "analysis_execution_head": _git_value("rev-parse", "HEAD"),
        "source_identity": _source_identity(),
        "input_hashes": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (
                c4f.COARSE_EARLY_ARRAYS, c4f.COARSE_ARRAYS,
                c4f.MIDDLE_PILOT_ARRAYS, c4f.MIDDLE_ARRAYS,
                c4f.FINE_ARRAYS, c4f.FINAL_ARRAYS, c4e11.DECISIVE_ARRAYS,
            )
        },
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "output_hashes": {},
    }
    _report(summary)
    provenance["output_hashes"] = {
        str(path.relative_to(ROOT)): _sha256(path)
        for path in (CONFIG_PATH, CONTRACT_PATH, SUMMARY_PATH, DECISIVE_ARRAYS, REPORT_PATH)
    }
    _write_json(PROVENANCE_PATH, provenance)
    checksums = CANONICAL_DIRECTORY / "SHA256SUMS.txt"
    checksums.write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in (CONFIG_PATH, CONTRACT_PATH, SUMMARY_PATH, PROVENANCE_PATH, DECISIVE_ARRAYS)),
        encoding="utf-8",
    )
    _catalog(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--through", choices=STAGES, default="complete")
    args = parser.parse_args()
    parent_summary, _manifest = _validate_authorization()
    contract = _contract()
    parent_grid, configurations = _configurations()
    del parent_grid
    absolute_path = CHECKPOINT_DIRECTORY / "absolute.npz"
    absolute_report_path = CHECKPOINT_DIRECTORY / "absolute.json"
    if absolute_path.exists() and absolute_report_path.exists():
        absolute_arrays = _load_npz(absolute_path)
        absolute_report = _read_json(absolute_report_path)
        absolute_report = _refresh_absolute_component_metrics(
            absolute_report, absolute_arrays
        )
        _write_json(absolute_report_path, absolute_report)
    else:
        absolute_report, absolute_arrays = _absolute_audit()
        _atomic_npz(absolute_path, **absolute_arrays)
        _write_json(absolute_report_path, absolute_report)
    print(f"c4f1-absolute: passed={absolute_report['passed']}", flush=True)
    if not absolute_report["passed"]:
        summary = _write_absolute_stop(
            contract, absolute_report, absolute_arrays
        )
        print(json.dumps(_plain(summary), indent=2, sort_keys=True), flush=True)
        return
    if args.through == "absolute":
        return
    middle_report, middle_arrays = _run_layout(
        "middle", configurations["middle"][1], configurations["middle"][0], _middle_trajectory()
    )
    print(f"c4f1-middle: method={middle_report['passed_method_gates']} k99={middle_report['memory']['k99']}", flush=True)
    if args.through == "middle" or not middle_report["passed_method_gates"]:
        return
    fine_report, fine_arrays = _run_layout(
        "fine", configurations["fine"][1], configurations["fine"][0], _fine_trajectory()
    )
    print(f"c4f1-fine: method={fine_report['passed_method_gates']} k99={fine_report['memory']['k99']}", flush=True)
    if args.through == "fine" or not fine_report["passed_method_gates"]:
        return
    summary = _finalize(
        contract, parent_summary, absolute_report, absolute_arrays,
        middle_report, middle_arrays, fine_report, fine_arrays, configurations,
    )
    print(json.dumps(_plain(summary), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
