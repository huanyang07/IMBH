"""Run the WP10c8p synchronized natural-healing rapid screen.

The runner consumes the exact decisive N64/N128 equal-coordinate pairs from
WP10c8o.  Every lifted state discards its parent BDF history, takes one fresh
BDF1 startup step with a zero predictor, and then advances with fixed-step
BDF2.  Coarse and fine complete trajectories share exact physical output
times so pair-spread temporal uncertainty is measured without adaptive-grid
history contamination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
import scipy

import run_causal_moment_sufficiency_audit_wp10c8i as wp10c8i
import run_causal_nonlinear_fiber_audit_wp10c8o as wp10c8o
import run_causal_shell_closure_preflight_wp10c8h as wp10c8h
from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d import (
    CausalFiveFieldAdaptiveStepConfig,
    audit_causal_five_field_state_gates,
    causal_cumulative_trapezoid,
    causal_five_field_face_flux_decomposition,
    causal_five_field_moment_coordinate_values,
    causal_five_field_observable_snapshot,
    causal_five_field_path_temporal_storage_increment,
    causal_mesh_coincident_moment_shells,
    causal_refined_spread_upper_bound,
    causal_transport_rank_audit,
    evaluate_causal_five_field_dae,
    evolve_causal_five_field_fixed_bdf2,
    unpack_causal_five_field_state,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_dae_system import (
    causal_five_field_rusanov_control_diagnostics,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "bf6f3e0cc67c2091ed61b76fad316c2ee472478e"
WORK_PACKAGE = "WP10c8p"
SCHEMA_VERSION = 1
THIS_RUNNER = "scripts/run_causal_natural_healing_wp10c8p.py"
PARENT_JSON = (
    ROOT / "outputs/tables/causal_nonlinear_fiber_audit_wp10c8o.json"
)
PARENT_ARRAYS = (
    ROOT / "outputs/tables/causal_nonlinear_fiber_audit_wp10c8o_arrays.npz"
)
CHECKPOINT_DIRECTORY = (
    ROOT / "outputs/checkpoints/causal_five_field_wp10c8p"
)
DEFAULT_OUTPUT = ROOT / "outputs/tables/causal_natural_healing_wp10c8p.json"
DEFAULT_ARRAYS = (
    ROOT / "outputs/tables/causal_natural_healing_wp10c8p_arrays.npz"
)

TARGET_DURATION_SECONDS = 2.5e-2
OUTPUT_OFFSETS_SECONDS = (0.0, 2.5e-3, 5.0e-3, 1.0e-2, 2.5e-2)
COARSE_SUBDIVISIONS = 20
FINE_SUBDIVISIONS = 40
COARSE_TIMESTEP_SECONDS = TARGET_DURATION_SECONDS / COARSE_SUBDIVISIONS
FINE_TIMESTEP_SECONDS = TARGET_DURATION_SECONDS / FINE_SUBDIVISIONS
TEMPORAL_UNCERTAINTY_GATE = 2.5e-2
TEMPORAL_RELATIVE_UNCERTAINTY = 0.10
TEMPORAL_RELATIVE_UNCERTAINTY_FLOOR = 0.10
RAPID_HEALING_GATE = 0.10
MINIMUM_DECAY_FACTOR = 2.0
MAXIMUM_SHELL_LEDGER_DEFECT = 1.0e-3
MAXIMUM_FLUX_RECONSTRUCTION_DEFECT = 1.0e-12
MAXIMUM_RANK_ONE_SECONDARY_RATIO = 0.10
INTERFACE_INDEX = 4
MJE_COMPONENTS = (0, 2, 3)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-n128",
        action="store_true",
        help="Run N128 after a numerically interpretable N64 result.",
    )
    parser.add_argument(
        "--skip-fresh-rates",
        action="store_true",
        help="Development-only: omit expensive fresh coordinate-rate rows.",
    )
    parser.add_argument(
        "--skip-replay",
        action="store_true",
        help="Development-only: omit split deterministic replay.",
    )
    parser.add_argument(
        "--skip-n128-replay",
        action="store_true",
        help=(
            "Development-only: retain binding N64 replay but omit N128 "
            "replay while assembling N128 fresh-rate diagnostics."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute trajectory caches instead of reusing them.",
    )
    parser.add_argument(
        "--trajectory-only",
        choices=(
            "n64-coarse-minus",
            "n64-coarse-plus",
            "n64-fine-minus",
            "n64-fine-plus",
            "n64-replay-minus",
            "n64-replay-plus",
            "n128-coarse-minus",
            "n128-coarse-plus",
            "n128-fine-minus",
            "n128-fine-plus",
            "n128-replay-minus",
            "n128-replay-plus",
        ),
        default=None,
        help="Populate exactly one trajectory cache and exit.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arrays", type=Path, default=DEFAULT_ARRAYS)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _step_config() -> CausalFiveFieldAdaptiveStepConfig:
    return CausalFiveFieldAdaptiveStepConfig(
        minimum_dt=1.0e-9,
        maximum_dt=3.8436439949172674e-3,
        maximum_scaled_primitive_change=0.2,
        maximum_scaled_total_change=0.25,
        shrink_factor=0.5,
        growth_factor=1.5,
        maximum_retries=0,
        easy_iterations=3,
        residual_tolerance=1.0e-11,
        algebraic_residual_tolerance=1.0e-11,
        conservation_tolerance=1.0e-10,
        finite_difference_step=2.0e-6,
        maximum_newton_iterations=12,
    ).validated()


def _ledger_row(ledger) -> dict:
    return {
        "actual_conserved_storage": ledger.actual_conserved_storage,
        "actual_vertical_storage": ledger.actual_vertical_storage,
        "trapezoidal_boundary_transport": (
            ledger.trapezoidal_boundary_transport
        ),
        "trapezoidal_endogenous_source": (
            ledger.trapezoidal_endogenous_source
        ),
        "exact_prescribed_stream_source": (
            ledger.exact_prescribed_stream_source
        ),
        "closure_defect": ledger.closure_defect,
    }


def _result_row(result, wall_seconds: float) -> dict:
    return {
        "passed": bool(result.passed),
        "message": result.message,
        "subdivisions": result.subdivisions,
        "timestep_seconds": result.timestep_seconds,
        "completed_steps": result.completed_steps,
        "bdf1_steps": result.bdf1_steps,
        "bdf2_steps": result.bdf2_steps,
        "maximum_scaled_residual": result.maximum_scaled_residual,
        "maximum_scaled_algebraic_residual": (
            result.maximum_scaled_algebraic_residual
        ),
        "maximum_scaled_primitive_change": (
            result.maximum_scaled_primitive_change
        ),
        "maximum_scaled_total_change": result.maximum_scaled_total_change,
        "maximum_discrete_ledger_relative_defect": (
            result.maximum_discrete_ledger_relative_defect
        ),
        "cumulative_physical_ledger_relative_defect": (
            result.cumulative_physical_ledger_relative_defect
        ),
        "maximum_linear_residual": result.maximum_linear_residual,
        "maximum_newton_iterations": result.maximum_newton_iterations,
        "function_evaluations": result.function_evaluations,
        "jacobian_evaluations": result.jacobian_evaluations,
        "newton_iterations": result.newton_iterations,
        "state_gates": result.state_gates,
        "cumulative_physical_ledger": _ledger_row(
            result.cumulative_physical_ledger
        ),
        "wall_seconds": wall_seconds,
    }


def _trajectory_path(
    n_cells: int,
    resolution: str,
    side: str,
) -> Path:
    return (
        CHECKPOINT_DIRECTORY
        / f"N{n_cells:03d}_{resolution}_{side}_trajectory.npz"
    )


def _run_or_load_trajectory(
    *,
    context,
    initial_vector: np.ndarray,
    n_cells: int,
    resolution: str,
    side: str,
    subdivisions: int,
    force: bool,
) -> dict:
    path = _trajectory_path(n_cells, resolution, side)
    expected = {
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "n_cells": n_cells,
        "resolution": resolution,
        "side": side,
        "subdivisions": subdivisions,
        "duration_seconds": TARGET_DURATION_SECONDS,
        "initial_state_sha256": _array_sha256(initial_vector),
        "parent_arrays_sha256": _sha256(PARENT_ARRAYS),
        "startup": "one_bdf1_with_zero_predictor_then_fixed_bdf2",
    }
    if path.exists() and not force:
        with np.load(path, allow_pickle=False) as source:
            metadata = json.loads(str(source["metadata_json"].item()))
            states = np.asarray(source["states"], dtype=float)
        if all(metadata.get(key) == value for key, value in expected.items()):
            return {
                "path": path,
                "sha256": _sha256(path),
                "metadata": metadata,
                "summary": metadata["summary"],
                "states": states,
                "cached": True,
            }
        raise RuntimeError(f"stale WP10c8p trajectory cache: {path}")

    timestep = TARGET_DURATION_SECONDS / subdivisions
    snapshots = [np.asarray(initial_vector, dtype=float).copy()]

    def progress(_completed, _total, state, _history) -> None:
        snapshots.append(np.asarray(state, dtype=float).copy())
        print(
            f"WP10c8p N{n_cells} {resolution} {side}: "
            f"step {_completed}/{_total}",
            flush=True,
        )

    started = time.perf_counter()
    result = evolve_causal_five_field_fixed_bdf2(
        context,
        np.asarray(initial_vector, dtype=float),
        np.zeros_like(initial_vector, dtype=float),
        timestep,
        TARGET_DURATION_SECONDS,
        subdivisions,
        _step_config(),
        startup_with_bdf1=True,
        progress=progress,
    )
    wall_seconds = time.perf_counter() - started
    states = np.asarray(snapshots, dtype=float)
    if states.shape[0] != result.completed_steps + 1:
        raise RuntimeError("trajectory snapshot count is inconsistent")
    summary = _result_row(result, wall_seconds)
    metadata = {**expected, "summary": _plain(summary)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        states=states,
        metadata_json=np.asarray(
            json.dumps(metadata, sort_keys=True, allow_nan=False)
        ),
    )
    return {
        "path": path,
        "sha256": _sha256(path),
        "metadata": metadata,
        "summary": summary,
        "states": states,
        "cached": False,
    }


def _split_replay(
    *,
    context,
    initial_vector: np.ndarray,
    reference_states: np.ndarray,
    subdivisions: int,
    n_cells: int,
    side: str,
    force: bool,
) -> dict:
    if subdivisions % 2:
        raise ValueError("split replay requires an even subdivision count")
    path = CHECKPOINT_DIRECTORY / (
        f"N{n_cells:03d}_coarse_{side}_split_replay.json"
    )
    expected = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "base_commit": BASE_COMMIT,
        "n_cells": n_cells,
        "side": side,
        "subdivisions": subdivisions,
        "duration_seconds": TARGET_DURATION_SECONDS,
        "initial_state_sha256": _array_sha256(initial_vector),
        "reference_states_sha256": _array_sha256(reference_states),
        "parent_arrays_sha256": _sha256(PARENT_ARRAYS),
    }
    if path.exists() and not force:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if all(cached.get(key) == value for key, value in expected.items()):
            return {**cached["result"], "cached": True, "path": _relative(path)}
        raise RuntimeError(f"stale WP10c8p split-replay cache: {path}")
    timestep = TARGET_DURATION_SECONDS / subdivisions
    half_steps = subdivisions // 2
    first_states = [np.asarray(initial_vector, dtype=float).copy()]

    def first_progress(_completed, _total, state, _history) -> None:
        first_states.append(np.asarray(state, dtype=float).copy())

    started = time.perf_counter()
    first = evolve_causal_five_field_fixed_bdf2(
        context,
        np.asarray(initial_vector, dtype=float),
        np.zeros_like(initial_vector, dtype=float),
        timestep,
        0.5 * TARGET_DURATION_SECONDS,
        half_steps,
        _step_config(),
        startup_with_bdf1=True,
        progress=first_progress,
    )
    if not first.passed or first.history is None:
        return {
            "passed": False,
            "message": "first split segment failed",
            "wall_seconds": time.perf_counter() - started,
        }
    second_states = []

    def second_progress(_completed, _total, state, _history) -> None:
        second_states.append(np.asarray(state, dtype=float).copy())

    second = evolve_causal_five_field_fixed_bdf2(
        context,
        first.state_vector,
        first.history.previous_physical_increment,
        timestep,
        0.5 * TARGET_DURATION_SECONDS,
        half_steps,
        _step_config(),
        startup_with_bdf1=False,
        initial_history=first.history,
        progress=second_progress,
    )
    replay_states = np.asarray(first_states + second_states, dtype=float)
    equal = bool(
        second.passed
        and replay_states.shape == reference_states.shape
        and np.array_equal(replay_states, reference_states)
    )
    result = {
        "passed": equal,
        "message": (
            "split replay is bitwise equal"
            if equal
            else "split replay differs from uninterrupted trajectory"
        ),
        "maximum_absolute_state_difference": (
            float(np.max(np.abs(replay_states - reference_states)))
            if replay_states.shape == reference_states.shape
            else None
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    payload = {**expected, "result": _plain(result)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {**result, "cached": False, "path": _relative(path)}


def _output_indices(
    times: np.ndarray,
    output_offsets_seconds: tuple[float, ...] = OUTPUT_OFFSETS_SECONDS,
) -> np.ndarray:
    indices = []
    for target in output_offsets_seconds:
        index = int(np.argmin(np.abs(times - target)))
        if not np.isclose(times[index], target, rtol=0.0, atol=1.0e-14):
            raise RuntimeError("healing output is not on a fixed-step boundary")
        indices.append(index)
    return np.asarray(indices, dtype=int)


def _coordinate_scale_lookup(names: tuple[str, ...], scales: np.ndarray):
    return {name: float(scales[index]) for index, name in enumerate(names)}


def _shell_scale_matrix(
    coordinate_names: tuple[str, ...],
    coordinate_scales: np.ndarray,
    n_shells: int,
) -> np.ndarray:
    lookup = _coordinate_scale_lookup(coordinate_names, coordinate_scales)
    result = np.ones((n_shells, 5), dtype=float)
    for shell in range(n_shells):
        result[shell, 0] = lookup[f"shell_{shell}_rest_mass"]
        result[shell, 2] = lookup[f"shell_{shell}_angular_momentum"]
        result[shell, 3] = lookup[f"shell_{shell}_killing_energy"]
    return result


def _persistent_fresh_rate_cache(
    n_cells: int,
) -> dict[str, tuple[np.ndarray, dict]]:
    """Load exact-state fresh rates from an earlier nonbinding assembly."""

    if not DEFAULT_OUTPUT.exists() or not DEFAULT_ARRAYS.exists():
        return {}
    evidence = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    mesh_key = str(n_cells)
    if not (
        evidence.get("work_package") == WORK_PACKAGE
        and evidence.get("scope", {}).get("fresh_coordinate_rates_evaluated")
        and evidence.get("authorization", {}).get("wp10c8o_arrays_sha256")
        == _sha256(PARENT_ARRAYS)
        and mesh_key in evidence.get("meshes", {})
    ):
        return {}
    prefix = f"n{n_cells}_"
    result: dict[str, tuple[np.ndarray, dict]] = {}
    with np.load(DEFAULT_ARRAYS, allow_pickle=False) as arrays:
        for resolution in ("coarse", "fine"):
            for side in ("minus", "plus"):
                label = f"{resolution}_{side}"
                primitives = np.asarray(
                    arrays[f"{prefix}{label}_output_primitives"],
                    dtype=float,
                )
                rates = np.asarray(
                    arrays[f"{prefix}{label}_normalized_coordinate_rates"],
                    dtype=float,
                )
                audits = evidence["meshes"][mesh_key][
                    "trajectory_diagnostics"
                ][label]["fresh_rate_audits"]
                if not (
                    primitives.shape[0] == rates.shape[0] == len(audits)
                ):
                    raise RuntimeError("persistent fresh-rate cache is malformed")
                for primitive, rate, audit in zip(
                    primitives,
                    rates,
                    audits,
                    strict=True,
                ):
                    result[_array_sha256(primitive)] = (rate, audit)
    return result


def _trajectory_diagnostics(
    *,
    context,
    states: np.ndarray,
    subdivisions: int,
    shell_edges_rg: np.ndarray,
    baseline_snapshot,
    anchor_interface_scales: np.ndarray,
    coordinate_names: tuple[str, ...],
    coordinate_scales: np.ndarray,
    primitive_scales: np.ndarray,
    conservation_scales: np.ndarray,
    common_interpolation: np.ndarray,
    compute_fresh_rates: bool,
    rate_cache: dict[str, tuple[np.ndarray, dict]],
    duration_seconds: float = TARGET_DURATION_SECONDS,
    output_offsets_seconds: tuple[float, ...] = OUTPUT_OFFSETS_SECONDS,
) -> tuple[dict, dict[str, np.ndarray]]:
    n_cells = int(context.grid.centers.size)
    timestep = float(duration_seconds) / subdivisions
    times = timestep * np.arange(states.shape[0], dtype=float)
    output_indices = _output_indices(times, output_offsets_seconds)
    geometry = causal_mesh_coincident_moment_shells(
        context,
        shell_edges_rg,
    )
    n_shells = len(geometry.cell_masks)
    measures = np.asarray(context.grid.cell_measures, dtype=float)
    unpacked = [
        unpack_causal_five_field_state(vector, n_cells) for vector in states
    ]
    evaluations = [
        evaluate_causal_five_field_dae(vector, context) for vector in states
    ]
    physical_face_fluxes = C * np.asarray(
        [state.weighted_face_fluxes_over_c for state in unpacked],
        dtype=float,
    )
    source_rates = C * np.asarray(
        [evaluation.integrated_sources_per_ct for evaluation in evaluations],
        dtype=float,
    )
    source_component_names = tuple(
        sorted(evaluations[0].integrated_source_components_per_ct)
    )
    if any(
        tuple(sorted(row.integrated_source_components_per_ct))
        != source_component_names
        for row in evaluations
    ):
        raise RuntimeError("source-component schema changed with time")
    source_component_rates = {
        name: C
        * np.asarray(
            [
                row.integrated_source_components_per_ct[name]
                for row in evaluations
            ],
            dtype=float,
        )
        for name in source_component_names
    }
    macro_face_indices = np.asarray(geometry.edge_indices[1:-1], dtype=int)
    macro_fluxes = physical_face_fluxes[:, macro_face_indices]
    macro_flux_integrals = causal_cumulative_trapezoid(times, macro_fluxes)

    actual_storage = np.zeros((times.size, n_shells, 5), dtype=float)
    vertical_storage = np.zeros_like(actual_storage)
    boundary_rates = np.zeros_like(actual_storage)
    shell_source_rates = np.zeros_like(actual_storage)
    shell_source_component_rates = {
        name: np.zeros_like(actual_storage) for name in source_component_names
    }
    for shell, (mask, left, right) in enumerate(
        zip(
            geometry.cell_masks,
            geometry.edge_indices[:-1],
            geometry.edge_indices[1:],
            strict=True,
        )
    ):
        actual_storage[:, shell] = np.asarray(
            [
                np.sum(
                    measures[mask, None]
                    * (state.conserved[mask] - unpacked[0].conserved[mask]),
                    axis=0,
                )
                for state in unpacked
            ]
        )
        boundary_rates[:, shell] = (
            physical_face_fluxes[:, right] - physical_face_fluxes[:, left]
        )
        shell_source_rates[:, shell] = np.sum(
            source_rates[:, mask],
            axis=1,
        )
        for name in source_component_names:
            shell_source_component_rates[name][:, shell] = np.sum(
                source_component_rates[name][:, mask],
                axis=1,
            )
    for index in range(1, times.size):
        path = causal_five_field_path_temporal_storage_increment(
            context,
            unpacked[index - 1].primitives,
            unpacked[index].primitives,
        )
        for shell, mask in enumerate(geometry.cell_masks):
            increment = np.zeros(5, dtype=float)
            increment[:4] = np.sum(
                measures[mask, None]
                * path.vertical_killing_increment[mask],
                axis=0,
            )
            vertical_storage[index, shell] = (
                vertical_storage[index - 1, shell] + increment
            )
    boundary_transport = causal_cumulative_trapezoid(times, boundary_rates)
    integrated_sources = causal_cumulative_trapezoid(
        times,
        shell_source_rates,
    )
    integrated_source_components = {
        name: causal_cumulative_trapezoid(times, values)
        for name, values in shell_source_component_rates.items()
    }
    shell_defect = (
        actual_storage
        + vertical_storage
        + boundary_transport
        - integrated_sources
    )
    shell_scale = (
        np.abs(actual_storage)
        + np.abs(vertical_storage)
        + np.abs(boundary_transport)
        + np.abs(integrated_sources)
    )
    shell_relative_defect = np.abs(shell_defect) / np.maximum(
        shell_scale,
        np.finfo(float).tiny,
    )

    static_rows = []
    coordinate_rows = []
    rate_rows = []
    rate_audits = []
    interface4_total = []
    interface4_perfect = []
    interface4_stress = []
    interface4_rusanov = []
    interface4_controls = []
    state_gate_rows = []
    output_primitives = []
    output_conserved = []
    static_names = None
    static_gates = None
    face_row = int(
        np.flatnonzero(
            np.arange(1, n_cells, dtype=int)
            == geometry.edge_indices[INTERFACE_INDEX]
        )[0]
    )
    coordinate_map = wp10c8o._coordinate_evaluator(context, shell_edges_rg)
    for index in output_indices:
        vector = states[index]
        state = unpacked[index]
        static, gates, names, _blocks = wp10c8o._static_output_stack(
            context=context,
            vector=vector,
            baseline_snapshot=baseline_snapshot,
            anchor_interface_scales=anchor_interface_scales,
            shell_edges_rg=shell_edges_rg,
            common_interpolation=common_interpolation,
        )
        if static_names is None:
            static_names = names
            static_gates = gates
        elif static_names != names or not np.array_equal(static_gates, gates):
            raise RuntimeError("healing output schema changed with time")
        coordinates = causal_five_field_moment_coordinate_values(
            context,
            vector,
            shell_edges_rg,
        ).level(wp10c8o.LEVEL_NAME).coordinate_values
        if compute_fresh_rates:
            print(
                f"WP10c8p N{n_cells} fresh-rate audit: "
                f"t={times[index]:.6g} s",
                flush=True,
            )
            key = _array_sha256(state.primitives)
            if key not in rate_cache:
                rate, rate_audit, _rate_arrays = wp10c8o._fresh_coordinate_rate(
                    context=context,
                    primitives=np.asarray(state.primitives, dtype=float).ravel(),
                    coordinate_evaluator=coordinate_map,
                    primitive_scales=primitive_scales,
                    conservation_scales=conservation_scales,
                    coordinate_scales=coordinate_scales,
                    binding_dae_storage_audit=False,
                )
                rate_cache[key] = (rate, rate_audit)
            rate, rate_audit = rate_cache[key]
        else:
            rate = np.zeros(34, dtype=float)
            rate_audit = {"passed": True, "development_skip": True}
        split = causal_five_field_face_flux_decomposition(context, vector)
        control = causal_five_field_rusanov_control_diagnostics(
            context,
            state.primitives,
        )
        static_rows.append(static)
        coordinate_rows.append(coordinates)
        rate_rows.append(rate)
        rate_audits.append(rate_audit)
        interface4_total.append(
            C * split.production_weighted_face_fluxes_over_c[face_row]
        )
        interface4_perfect.append(
            C * split.central_perfect_weighted_face_fluxes_over_c[face_row]
        )
        interface4_stress.append(
            C * split.central_stress_weighted_face_fluxes_over_c[face_row]
        )
        interface4_rusanov.append(
            C * split.rusanov_weighted_face_fluxes_over_c[face_row]
        )
        interface4_controls.append(
            np.asarray(control["control_codes"], dtype=int)[face_row]
        )
        state_gate_rows.append(audit_causal_five_field_state_gates(context, vector))
        output_primitives.append(state.primitives)
        output_conserved.append(state.conserved)

    interface4_total = np.asarray(interface4_total)
    interface4_perfect = np.asarray(interface4_perfect)
    interface4_stress = np.asarray(interface4_stress)
    interface4_rusanov = np.asarray(interface4_rusanov)
    flux_reconstruction_scale = np.maximum(np.abs(interface4_total), 1.0)
    flux_reconstruction_defect = float(
        np.max(
            np.abs(
                interface4_total
                - interface4_perfect
                - interface4_stress
                - interface4_rusanov
            )
            / flux_reconstruction_scale
        )
    )
    summary = {
        "maximum_shell_ledger_relative_defect": float(
            np.max(shell_relative_defect[1:])
        ),
        "maximum_physical_mje_shell_ledger_relative_defect": float(
            np.max(shell_relative_defect[1:, :, MJE_COMPONENTS])
        ),
        "maximum_flux_reconstruction_defect": flux_reconstruction_defect,
        "all_output_state_gates_passed": bool(
            all(row["passed"] for row in state_gate_rows)
        ),
        "all_fresh_rate_audits_passed": bool(
            all(row["passed"] for row in rate_audits)
        ),
        "fresh_rates_evaluated": bool(compute_fresh_rates),
        "output_state_gates": state_gate_rows,
        "fresh_rate_audits": rate_audits,
    }
    arrays = {
        "times": times,
        "output_indices": output_indices,
        "output_times": times[output_indices],
        "static_output_names": np.asarray(static_names, dtype="U"),
        "static_output_gates": np.asarray(static_gates, dtype=float),
        "static_outputs": np.asarray(static_rows, dtype=float),
        "coordinate_names": np.asarray(coordinate_names, dtype="U"),
        "coordinate_scales": np.asarray(coordinate_scales, dtype=float),
        "coordinates": np.asarray(coordinate_rows, dtype=float),
        "normalized_coordinate_rates": np.asarray(rate_rows, dtype=float),
        "macro_face_indices": macro_face_indices,
        "macro_fluxes": macro_fluxes,
        "macro_flux_integrals": macro_flux_integrals,
        "shell_edge_indices": np.asarray(geometry.edge_indices, dtype=int),
        "shell_actual_storage": actual_storage,
        "shell_vertical_storage": vertical_storage,
        "shell_boundary_transport": boundary_transport,
        "shell_integrated_sources": integrated_sources,
        "source_component_names": np.asarray(source_component_names, dtype="U"),
        "shell_ledger_defect": shell_defect,
        "shell_ledger_relative_defect": shell_relative_defect,
        "interface4_total_flux": interface4_total,
        "interface4_perfect_flux": interface4_perfect,
        "interface4_stress_flux": interface4_stress,
        "interface4_rusanov_flux": interface4_rusanov,
        "interface4_control_codes": np.asarray(interface4_controls, dtype=int),
        "output_primitives": np.asarray(output_primitives, dtype=float),
        "output_conserved": np.asarray(output_conserved, dtype=float),
        **{
            f"shell_integrated_source_{name}": values
            for name, values in integrated_source_components.items()
        },
    }
    return summary, arrays


def _pair_diagnostics(
    *,
    minus: dict[str, np.ndarray],
    plus: dict[str, np.ndarray],
    coordinate_scales: np.ndarray,
    coordinate_names: tuple[str, ...],
) -> tuple[dict, dict[str, np.ndarray]]:
    if not np.array_equal(minus["output_times"], plus["output_times"]):
        raise RuntimeError("plus/minus output times differ")
    static_gates = np.asarray(minus["static_output_gates"], dtype=float)
    static_spreads = 0.5 * np.abs(
        plus["static_outputs"] - minus["static_outputs"]
    ) / static_gates[None, :]
    coordinate_spreads = np.abs(
        plus["coordinates"] - minus["coordinates"]
    ) / (2.0 * coordinate_scales)
    rate_spreads = 0.5 * np.abs(
        plus["normalized_coordinate_rates"]
        - minus["normalized_coordinate_rates"]
    )
    full_spreads = np.concatenate((static_spreads, rate_spreads), axis=1)
    full_names = np.concatenate(
        (
            minus["static_output_names"],
            np.asarray(
                [f"fresh_rate_{name}" for name in coordinate_names],
                dtype="U",
            ),
        )
    )

    interface4_half_difference = 0.5 * (
        plus["interface4_total_flux"] - minus["interface4_total_flux"]
    )
    transport_rank_inputs = interface4_half_difference[:, MJE_COMPONENTS]

    shell_scales = _shell_scale_matrix(
        coordinate_names,
        coordinate_scales,
        minus["shell_actual_storage"].shape[1],
    )
    shell_actual_pair = 0.5 * (
        plus["shell_actual_storage"] - minus["shell_actual_storage"]
    )
    shell_vertical_pair = 0.5 * (
        plus["shell_vertical_storage"] - minus["shell_vertical_storage"]
    )
    shell_boundary_pair = 0.5 * (
        plus["shell_boundary_transport"] - minus["shell_boundary_transport"]
    )
    shell_source_pair = 0.5 * (
        plus["shell_integrated_sources"] - minus["shell_integrated_sources"]
    )
    shell_defect_pair = (
        shell_actual_pair
        + shell_vertical_pair
        + shell_boundary_pair
        - shell_source_pair
    )
    shell_pair_scale = (
        np.abs(shell_actual_pair)
        + np.abs(shell_vertical_pair)
        + np.abs(shell_boundary_pair)
        + np.abs(shell_source_pair)
    )
    shell_pair_relative_defect = np.abs(shell_defect_pair) / np.maximum(
        shell_pair_scale,
        np.finfo(float).tiny,
    )
    shell_pair_scale_defect = np.abs(shell_defect_pair) / shell_scales[None, :, :]
    interface_impulse_pair = 0.5 * (
        plus["macro_flux_integrals"] - minus["macro_flux_integrals"]
    )
    interface4_impulse = interface_impulse_pair[:, INTERFACE_INDEX - 1]
    adjacent_impulse_normalized = []
    for shell in (INTERFACE_INDEX - 1, INTERFACE_INDEX):
        adjacent_impulse_normalized.append(
            np.abs(interface4_impulse[:, MJE_COMPONENTS])
            / shell_scales[shell, MJE_COMPONENTS]
        )
    adjacent_impulse_normalized = np.asarray(adjacent_impulse_normalized)

    arrays = {
        "times": minus["output_times"],
        "full_output_names": full_names,
        "static_spreads": static_spreads,
        "coordinate_spreads": coordinate_spreads,
        "rate_spreads": rate_spreads,
        "full_spreads": full_spreads,
        "interface4_transport_half_difference": interface4_half_difference,
        "transport_rank_inputs_unscaled": transport_rank_inputs,
        "shell_actual_pair": shell_actual_pair,
        "shell_vertical_pair": shell_vertical_pair,
        "shell_boundary_pair": shell_boundary_pair,
        "shell_source_pair": shell_source_pair,
        "shell_defect_pair": shell_defect_pair,
        "shell_pair_relative_defect": shell_pair_relative_defect,
        "shell_pair_scale_defect": shell_pair_scale_defect,
        "interface_impulse_pair": interface_impulse_pair,
        "adjacent_interface4_impulse_normalized": adjacent_impulse_normalized,
    }
    summary = {
        "initial_maximum_full_spread": float(np.max(full_spreads[0])),
        "final_maximum_full_spread": float(np.max(full_spreads[-1])),
        "final_maximum_coordinate_drift": float(
            np.max(coordinate_spreads[-1])
        ),
        "maximum_pair_shell_ledger_relative_defect": float(
            np.max(shell_pair_relative_defect[1:])
        ),
        "maximum_pair_physical_mje_shell_ledger_relative_defect": float(
            np.max(shell_pair_relative_defect[1:, :, MJE_COMPONENTS])
        ),
        "maximum_pair_physical_mje_shell_scale_defect": float(
            np.max(shell_pair_scale_defect[1:, :, MJE_COMPONENTS])
        ),
        "final_maximum_interface4_impulse_fraction": float(
            np.max(adjacent_impulse_normalized[:, -1])
        ),
    }
    return summary, arrays


def _temporal_and_healing_decision(
    *,
    coarse: dict[str, np.ndarray],
    fine: dict[str, np.ndarray],
    interface_flux_scales: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    if not (
        np.array_equal(coarse["times"], fine["times"])
        and np.array_equal(coarse["full_output_names"], fine["full_output_names"])
    ):
        raise RuntimeError("coarse/fine pair schemas differ")
    uncertainty, upper = causal_refined_spread_upper_bound(
        coarse["full_spreads"],
        fine["full_spreads"],
    )
    coordinate_uncertainty, coordinate_upper = (
        causal_refined_spread_upper_bound(
            coarse["coordinate_spreads"],
            fine["coordinate_spreads"],
        )
    )
    relative_mask = fine["full_spreads"] >= TEMPORAL_RELATIVE_UNCERTAINTY_FLOOR
    relative_passed = bool(
        np.all(
            uncertainty[relative_mask]
            <= TEMPORAL_RELATIVE_UNCERTAINTY
            * fine["full_spreads"][relative_mask]
        )
    )
    temporal_passed = bool(
        np.max(uncertainty) <= TEMPORAL_UNCERTAINTY_GATE
        and relative_passed
    )
    initial = upper[0]
    final = upper[-1]
    significant = initial > RAPID_HEALING_GATE
    decay_passed = bool(
        np.all(
            final[significant]
            <= initial[significant] / MINIMUM_DECAY_FACTOR
        )
    )
    no_late_regrowth = bool(
        np.all(
            fine["full_spreads"][-1, significant]
            <= fine["full_spreads"][-2, significant]
            + uncertainty[-1, significant]
            + uncertainty[-2, significant]
        )
    )
    coordinate_passed = bool(np.max(coordinate_upper[-1]) <= RAPID_HEALING_GATE)
    impulse_passed = bool(
        np.max(fine["adjacent_interface4_impulse_normalized"][:, -1])
        <= RAPID_HEALING_GATE
    )
    rapid_healing = bool(
        temporal_passed
        and np.max(final) <= RAPID_HEALING_GATE
        and decay_passed
        and no_late_regrowth
        and coordinate_passed
        and impulse_passed
    )

    normalized_transport = (
        fine["interface4_transport_half_difference"][:, MJE_COMPONENTS]
        / np.asarray(interface_flux_scales, dtype=float)[
            (INTERFACE_INDEX - 1) * 3 : INTERFACE_INDEX * 3
        ]
    )
    rank = causal_transport_rank_audit(
        normalized_transport,
        maximum_secondary_ratio=MAXIMUM_RANK_ONE_SECONDARY_RATIO,
    )
    summary = {
        "maximum_temporal_uncertainty": float(np.max(uncertainty)),
        "maximum_relative_temporal_uncertainty": float(
            np.max(
                uncertainty[relative_mask]
                / fine["full_spreads"][relative_mask]
            )
            if np.any(relative_mask)
            else 0.0
        ),
        "temporal_uncertainty_passed": temporal_passed,
        "final_maximum_upper_spread": float(np.max(final)),
        "factor_two_decay_passed": decay_passed,
        "no_late_regrowth_passed": no_late_regrowth,
        "final_maximum_coordinate_upper_spread": float(
            np.max(coordinate_upper[-1])
        ),
        "coordinate_drift_passed": coordinate_passed,
        "final_maximum_interface_impulse_fraction": float(
            np.max(fine["adjacent_interface4_impulse_normalized"][:, -1])
        ),
        "interface_impulse_passed": impulse_passed,
        "rapid_healing_passed": rapid_healing,
        "rank_one_transport_diagnostic": {
            "singular_values": rank.singular_values,
            "second_to_first_ratio": rank.second_to_first_ratio,
            "third_to_first_ratio": rank.third_to_first_ratio,
            "dominant_direction": rank.dominant_direction,
            "passed": rank.passed,
            "binding_for_auxiliary_authorization": False,
        },
    }
    arrays = {
        "temporal_uncertainty": uncertainty,
        "upper_spreads": upper,
        "coordinate_temporal_uncertainty": coordinate_uncertainty,
        "coordinate_upper_spreads": coordinate_upper,
        "normalized_interface4_transport_half_difference": normalized_transport,
    }
    return summary, arrays


def _mesh_contract(
    *,
    n_cells: int,
    parent: dict,
    parent_arrays,
    initial_by_mesh: dict,
    vectors_by_mesh: dict,
    shell_edges_rg: np.ndarray,
):
    pair_id = (
        parent["decisive_n64_pair"]
        if n_cells == 64
        else parent["n128_confirmation_pair"]
    )
    prefix = f"{pair_id}_"
    context = initial_by_mesh[n_cells]["context"]
    vector = vectors_by_mesh[n_cells][wp10c8o.PRIMARY_ANCHOR]
    cache, metadata, cache_path = wp10c8o._load_anchor_cache(
        n_cells,
        wp10c8o.PRIMARY_ANCHOR,
        vector,
    )
    coordinate_names = tuple(
        str(value)
        for value in np.asarray(parent_arrays[prefix + "coordinate_names"])
    )
    return {
        "pair_id": pair_id,
        "context": context,
        "anchor_vector": vector,
        "minus_vector": np.asarray(
            parent_arrays[prefix + "minus_state_vector"], dtype=float
        ),
        "plus_vector": np.asarray(
            parent_arrays[prefix + "plus_state_vector"], dtype=float
        ),
        "coordinate_names": coordinate_names,
        "coordinate_scales": np.asarray(
            parent_arrays[prefix + "coordinate_scales"], dtype=float
        ),
        "interface_flux_scales": np.asarray(
            parent_arrays[prefix + "interface_flux_scales"], dtype=float
        ),
        "primitive_scales": np.asarray(
            cache["primitive_column_scales"], dtype=float
        ),
        "conservation_scales": np.asarray(
            cache["conservation_row_scales"], dtype=float
        ),
        "cache_path": cache_path,
        "cache_metadata": metadata,
        "shell_edges_rg": shell_edges_rg,
    }


def _run_mesh(
    *,
    contract: dict,
    force: bool,
    compute_fresh_rates: bool,
    run_replay: bool,
) -> tuple[dict, dict[str, np.ndarray]]:
    context = contract["context"]
    n_cells = int(context.grid.centers.size)
    baseline_snapshot = causal_five_field_observable_snapshot(
        context,
        contract["anchor_vector"],
        cooling_inner_cutoff=6.0 * context.grid.gravitational_radius,
    )
    radius_rg = context.grid.centers / context.grid.gravitational_radius
    grid_edges_rg = context.grid.edges / context.grid.gravitational_radius
    _common_radius, common_interpolation = wp10c8i._common_log_h_interpolation(
        radius_rg,
        grid_edges_rg,
    )
    trajectories = {}
    diagnostics = {}
    all_arrays: dict[str, np.ndarray] = {}
    rate_cache = (
        _persistent_fresh_rate_cache(n_cells)
        if compute_fresh_rates
        else {}
    )
    for resolution, subdivisions in (
        ("coarse", COARSE_SUBDIVISIONS),
        ("fine", FINE_SUBDIVISIONS),
    ):
        for side in ("minus", "plus"):
            trajectory = _run_or_load_trajectory(
                context=context,
                initial_vector=contract[f"{side}_vector"],
                n_cells=n_cells,
                resolution=resolution,
                side=side,
                subdivisions=subdivisions,
                force=force,
            )
            trajectories[f"{resolution}_{side}"] = trajectory
            summary, arrays = _trajectory_diagnostics(
                context=context,
                states=trajectory["states"],
                subdivisions=subdivisions,
                shell_edges_rg=contract["shell_edges_rg"],
                baseline_snapshot=baseline_snapshot,
                anchor_interface_scales=contract["interface_flux_scales"],
                coordinate_names=contract["coordinate_names"],
                coordinate_scales=contract["coordinate_scales"],
                primitive_scales=contract["primitive_scales"],
                conservation_scales=contract["conservation_scales"],
                common_interpolation=common_interpolation,
                compute_fresh_rates=compute_fresh_rates,
                rate_cache=rate_cache,
            )
            diagnostics[f"{resolution}_{side}"] = summary
            all_arrays.update(
                {
                    f"{resolution}_{side}_{name}": value
                    for name, value in arrays.items()
                }
            )

    pair_summaries = {}
    pair_arrays = {}
    for resolution in ("coarse", "fine"):
        summary, arrays = _pair_diagnostics(
            minus={
                name.removeprefix(f"{resolution}_minus_"): value
                for name, value in all_arrays.items()
                if name.startswith(f"{resolution}_minus_")
            },
            plus={
                name.removeprefix(f"{resolution}_plus_"): value
                for name, value in all_arrays.items()
                if name.startswith(f"{resolution}_plus_")
            },
            coordinate_scales=contract["coordinate_scales"],
            coordinate_names=contract["coordinate_names"],
        )
        pair_summaries[resolution] = summary
        pair_arrays[resolution] = arrays
        all_arrays.update(
            {f"{resolution}_pair_{name}": value for name, value in arrays.items()}
        )
    decision, decision_arrays = _temporal_and_healing_decision(
        coarse=pair_arrays["coarse"],
        fine=pair_arrays["fine"],
        interface_flux_scales=contract["interface_flux_scales"],
    )
    all_arrays.update(
        {f"decision_{name}": value for name, value in decision_arrays.items()}
    )

    replay = {"evaluated": False, "passed": True}
    if run_replay:
        replay_rows = {}
        for side in ("minus", "plus"):
            replay_rows[side] = _split_replay(
                context=context,
                initial_vector=contract[f"{side}_vector"],
                reference_states=trajectories[f"coarse_{side}"]["states"],
                subdivisions=COARSE_SUBDIVISIONS,
                n_cells=n_cells,
                side=side,
                force=force,
            )
        replay = {
            "evaluated": True,
            "sides": replay_rows,
            "passed": bool(all(row["passed"] for row in replay_rows.values())),
        }

    trajectory_passed = bool(
        all(
            row["summary"]["passed"]
            for row in trajectories.values()
        )
    )
    diagnostics_passed = bool(
        all(
            row["maximum_physical_mje_shell_ledger_relative_defect"]
            <= MAXIMUM_SHELL_LEDGER_DEFECT
            and row["maximum_flux_reconstruction_defect"]
            <= MAXIMUM_FLUX_RECONSTRUCTION_DEFECT
            and row["all_output_state_gates_passed"]
            and row["all_fresh_rate_audits_passed"]
            for row in diagnostics.values()
        )
        and pair_summaries["fine"][
            "maximum_pair_physical_mje_shell_scale_defect"
        ]
        <= MAXIMUM_SHELL_LEDGER_DEFECT
    )
    numerically_interpretable = bool(
        trajectory_passed
        and diagnostics_passed
        and decision["temporal_uncertainty_passed"]
        and compute_fresh_rates
        and run_replay
        and replay["evaluated"]
        and replay["passed"]
    )
    if not numerically_interpretable:
        classification = "numerically_inconclusive"
    elif decision["rapid_healing_passed"]:
        classification = "rapid_healing_supported_through_0p025s"
    else:
        classification = "rapid_healing_rejected_through_0p025s_only"
    row = {
        "n_cells": n_cells,
        "pair_id": contract["pair_id"],
        "trajectory_provenance": {
            key: {
                "path": _relative(value["path"]),
                "sha256": value["sha256"],
                "cached": value["cached"],
                "summary": value["summary"],
            }
            for key, value in trajectories.items()
        },
        "trajectory_diagnostics": diagnostics,
        "pair_diagnostics": pair_summaries,
        "temporal_and_healing_decision": decision,
        "deterministic_split_replay": replay,
        "trajectory_contracts_passed": trajectory_passed,
        "diagnostic_contracts_passed": diagnostics_passed,
        "binding_diagnostics_complete": bool(
            compute_fresh_rates and run_replay and replay["evaluated"]
        ),
        "numerically_interpretable": numerically_interpretable,
        "classification": classification,
    }
    return row, all_arrays


def main() -> None:
    args = _arguments()
    started = time.perf_counter()
    if not PARENT_JSON.exists() or not PARENT_ARRAYS.exists():
        raise FileNotFoundError("WP10c8o parent evidence is missing")
    parent = json.loads(PARENT_JSON.read_text(encoding="utf-8"))
    if not (
        parent.get("work_package") == "WP10c8o"
        and parent.get("decision")
        == "wp10c8o_exact_nonlinear_fiber_counterexample_confirmed_n64_n128"
        and parent.get("artifacts", {}).get("arrays_sha256")
        == _sha256(PARENT_ARRAYS)
    ):
        raise RuntimeError("WP10c8o parent contract differs")
    initial_by_mesh, vectors_by_mesh, state_provenance = wp10c8i._load_states()
    shell_edges_rg = np.asarray(
        wp10c8h._common_shell_edges(initial_by_mesh)["five_shell"],
        dtype=float,
    )
    with np.load(PARENT_ARRAYS, allow_pickle=False) as parent_arrays:
        contract64 = _mesh_contract(
            n_cells=64,
            parent=parent,
            parent_arrays=parent_arrays,
            initial_by_mesh=initial_by_mesh,
            vectors_by_mesh=vectors_by_mesh,
            shell_edges_rg=shell_edges_rg,
        )
        if args.trajectory_only is not None:
            mesh_label, resolution, side = args.trajectory_only.split("-")
            n_cells = 64 if mesh_label == "n64" else 128
            contract = contract64
            if n_cells == 128:
                contract = _mesh_contract(
                    n_cells=128,
                    parent=parent,
                    parent_arrays=parent_arrays,
                    initial_by_mesh=initial_by_mesh,
                    vectors_by_mesh=vectors_by_mesh,
                    shell_edges_rg=shell_edges_rg,
                )
            if resolution == "replay":
                reference = _run_or_load_trajectory(
                    context=contract["context"],
                    initial_vector=contract[f"{side}_vector"],
                    n_cells=n_cells,
                    resolution="coarse",
                    side=side,
                    subdivisions=COARSE_SUBDIVISIONS,
                    force=False,
                )
                replay = _split_replay(
                    context=contract["context"],
                    initial_vector=contract[f"{side}_vector"],
                    reference_states=reference["states"],
                    subdivisions=COARSE_SUBDIVISIONS,
                    n_cells=n_cells,
                    side=side,
                    force=args.force,
                )
                print(json.dumps(_plain(replay), indent=2, sort_keys=True))
                return
            subdivisions = (
                COARSE_SUBDIVISIONS
                if resolution == "coarse"
                else FINE_SUBDIVISIONS
            )
            trajectory = _run_or_load_trajectory(
                context=contract["context"],
                initial_vector=contract[f"{side}_vector"],
                n_cells=n_cells,
                resolution=resolution,
                side=side,
                subdivisions=subdivisions,
                force=args.force,
            )
            print(
                json.dumps(
                    _plain(
                        {
                            "path": _relative(trajectory["path"]),
                            "sha256": trajectory["sha256"],
                            "cached": trajectory["cached"],
                            "summary": trajectory["summary"],
                        }
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return
        mesh64, arrays64 = _run_mesh(
            contract=contract64,
            force=args.force,
            compute_fresh_rates=not args.skip_fresh_rates,
            run_replay=not args.skip_replay,
        )
        meshes = {"64": mesh64}
        all_arrays = {f"n64_{name}": value for name, value in arrays64.items()}
        n128_authorized = bool(mesh64["numerically_interpretable"])
        if args.include_n128:
            if not n128_authorized:
                raise RuntimeError("N64 did not authorize N128 healing")
            contract128 = _mesh_contract(
                n_cells=128,
                parent=parent,
                parent_arrays=parent_arrays,
                initial_by_mesh=initial_by_mesh,
                vectors_by_mesh=vectors_by_mesh,
                shell_edges_rg=shell_edges_rg,
            )
            mesh128, arrays128 = _run_mesh(
                contract=contract128,
                force=args.force,
                compute_fresh_rates=not args.skip_fresh_rates,
                run_replay=(
                    not args.skip_replay and not args.skip_n128_replay
                ),
            )
            meshes["128"] = mesh128
            all_arrays.update(
                {f"n128_{name}": value for name, value in arrays128.items()}
            )

    if not mesh64["binding_diagnostics_complete"]:
        decision = "wp10c8p_development_diagnostics_nonbinding"
        next_action = "complete_fresh_rate_and_deterministic_replay_contracts"
    elif not mesh64["numerically_interpretable"]:
        decision = "wp10c8p_n64_rapid_healing_screen_numerically_inconclusive"
        next_action = "diagnose_n64_temporal_or_ledger_failure"
    elif mesh64["classification"] == "rapid_healing_supported_through_0p025s":
        decision = "wp10c8p_n64_rapid_healing_supported"
        next_action = (
            "confirm_n128_and_test_amplitudes_and_held_out_fibers_before_"
            "any_healed_closure"
        )
    else:
        decision = "wp10c8p_n64_rapid_healing_rejected_through_0p025s"
        next_action = "extend_n64_to_0p05_0p10_0p125_before_memory_architecture"
    if "128" in meshes:
        if not meshes["128"]["binding_diagnostics_complete"]:
            decision = "wp10c8p_development_diagnostics_nonbinding"
            next_action = (
                "complete_fresh_rate_and_deterministic_replay_contracts"
            )
        elif not meshes["128"]["numerically_interpretable"]:
            decision = "wp10c8p_n128_confirmation_numerically_inconclusive"
            next_action = "diagnose_n128_temporal_or_ledger_failure"
        elif all(
            row["classification"] == "rapid_healing_supported_through_0p025s"
            for row in meshes.values()
        ):
            decision = "wp10c8p_rapid_healing_supported_n64_n128"
            next_action = (
                "test_amplitudes_and_held_out_fibers_then_build_a_healed_"
                "derivative_prototype"
            )
        else:
            decision = "wp10c8p_rapid_healing_rejected_through_0p025s"
            next_action = "extend_n64_to_0p05_0p10_0p125_before_memory_architecture"

    arrays_path = _absolute(args.arrays)
    output_path = _absolute(args.output)
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **all_arrays)
    source_paths = (
        ROOT / THIS_RUNNER,
        ROOT / "src/imri_qpe/layer3_minidisk_1d/causal_inner_healing.py",
        ROOT / "src/imri_qpe/layer3_minidisk_1d/causal_inner_dae_system.py",
        ROOT / "src/imri_qpe/layer3_minidisk_1d/causal_inner_bdf_evolution.py",
        PARENT_JSON,
        PARENT_ARRAYS,
    )
    output = {
        "schema_version": SCHEMA_VERSION,
        "base_commit": BASE_COMMIT,
        "work_package": WORK_PACKAGE,
        "scope": {
            "rapid_healing_screen_only": True,
            "production_flux_changed": False,
            "production_descriptor_changed": False,
            "production_time_integrator_changed": False,
            "moment_ladder_changed": False,
            "new_coordinate_added": False,
            "memory_model_fit": False,
            "macrostep_run": False,
            "constrained_healing_run": False,
            "fresh_coordinate_rates_evaluated": not args.skip_fresh_rates,
        },
        "authorization": {
            "wp10c8o_path": _relative(PARENT_JSON),
            "wp10c8o_sha256": _sha256(PARENT_JSON),
            "wp10c8o_arrays_path": _relative(PARENT_ARRAYS),
            "wp10c8o_arrays_sha256": _sha256(PARENT_ARRAYS),
        },
        "frozen_contract": {
            "coordinate_level": wp10c8o.LEVEL_NAME,
            "coordinate_count": 34,
            "shell_edges_rg": shell_edges_rg,
            "duration_seconds": TARGET_DURATION_SECONDS,
            "output_offsets_seconds": OUTPUT_OFFSETS_SECONDS,
            "coarse_subdivisions": COARSE_SUBDIVISIONS,
            "fine_subdivisions": FINE_SUBDIVISIONS,
            "coarse_timestep_seconds": COARSE_TIMESTEP_SECONDS,
            "fine_timestep_seconds": FINE_TIMESTEP_SECONDS,
            "startup": "discard_parent_history_zero_predictor_bdf1_then_bdf2",
            "pair_policy": "identical_fixed_physical_timestep_schedule",
        },
        "gates": {
            "maximum_temporal_uncertainty_gate_units": (
                TEMPORAL_UNCERTAINTY_GATE
            ),
            "maximum_relative_temporal_uncertainty": (
                TEMPORAL_RELATIVE_UNCERTAINTY
            ),
            "relative_uncertainty_evaluation_floor": (
                TEMPORAL_RELATIVE_UNCERTAINTY_FLOOR
            ),
            "rapid_healing_upper_spread": RAPID_HEALING_GATE,
            "minimum_decay_factor": MINIMUM_DECAY_FACTOR,
            "maximum_physical_mje_shell_ledger_relative_defect": (
                MAXIMUM_SHELL_LEDGER_DEFECT
            ),
            "maximum_flux_reconstruction_defect": (
                MAXIMUM_FLUX_RECONSTRUCTION_DEFECT
            ),
            "maximum_rank_one_secondary_ratio_diagnostic": (
                MAXIMUM_RANK_ONE_SECONDARY_RATIO
            ),
        },
        "state_provenance": {
            key: state_provenance[key][wp10c8o.PRIMARY_ANCHOR]
            for key in meshes
        },
        "meshes": meshes,
        "n128_requested": bool(args.include_n128),
        "n128_authorized_by_n64": n128_authorized,
        "decision": decision,
        "next_action": next_action,
        "semantics": (
            "A 0.025 s failure rejects rapid healing only. It does not prove "
            "permanent memory, authorize one auxiliary, or mandate a coarse "
            "PDE. The rank-one transport row is diagnostic until multiple "
            "e-foldings, amplitudes, meshes, and held-out fibers agree."
        ),
        "artifacts": {
            "arrays_path": _relative(arrays_path),
            "arrays_sha256": _sha256(arrays_path),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "wall_seconds": time.perf_counter() - started,
        },
        "source_hashes": {
            _relative(path): _sha256(path) for path in source_paths
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_plain(output), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_plain(output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
