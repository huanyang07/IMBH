"""Run the bounded N32/N64 WP10c7i spatial-balance trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d import (
    CAUSAL_FIVE_FIELD_NAMES,
    CausalFiveFieldAdaptiveStepConfig,
    CausalFiveFieldBDFRestart,
    audit_causal_five_field_state_gates,
    causal_five_field_bdf_physical_ledger_relative_defects,
    causal_five_field_bdf_restarts_equal,
    causal_five_field_consistent_tangent_decomposition,
    causal_five_field_profile_fields,
    causal_five_field_reconstruct_face_charts,
    causal_five_field_regression_seed_parameters,
    causal_five_field_residual_terms,
    causal_five_field_state_summary,
    causal_restrict_cell_averages,
    causal_restrict_cell_integrals,
    evaluate_causal_five_field_dae,
    evolve_causal_five_field_fixed_bdf2,
    load_causal_five_field_bdf_restart,
    make_causal_five_field_regression_context,
    make_causal_five_field_seed,
    pack_causal_five_field_state,
    save_causal_five_field_bdf_restart,
    unpack_causal_five_field_state,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "ac05f352380616f2ec0e346adaf3613b054ee3e2"
WP10C7I_OUTPUT = (
    ROOT / "outputs/tables/causal_spatial_balance_wp10c7i.json"
)
WP10C7H_OUTPUT = (
    ROOT
    / "outputs/tables/causal_reconstructed_flux_trajectory_wp10c7h.json"
)
OUTPUT_CHECKPOINT_DIRECTORY = (
    ROOT / "outputs/checkpoints/causal_five_field_wp10c7j"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/tables/causal_spatial_balance_trajectory_wp10c7j.json"
)
DEFAULT_ARRAYS = (
    ROOT
    / "outputs/tables/"
    "causal_spatial_balance_trajectory_wp10c7j_arrays.npz"
)
SPATIAL_OPTIONS = {
    "spatial_reconstruction": "quadratic_admissible",
    "boundary_trace_reconstruction": "plm_one_sided",
    "cell_rate_scheme": "arithmetic_face",
    "cell_source_quadrature": "gauss_legendre_4_local_rates",
    "cell_storage_quadrature": "gauss_legendre_4",
}
SELECTED_WP10C7I_VARIANT = "quadratic_face_local_rate_high_order"
RESOLUTIONS = (32, 64)
SUBDIVISIONS = (32, 64)
SNAPSHOT_FRACTIONS = (
    ("t_1_8", 1, 8),
    ("t_1_4", 1, 4),
    ("t_1_2", 1, 2),
    ("t_1", 1, 1),
)
TARGET_DURATION_SECONDS = 1.537457597966907e-2
SPATIAL_RESPONSE_GATE = 5.0e-3
MAXIMUM_TEMPORAL_LOG_H_UNCERTAINTY = 2.5e-4
MAXIMUM_PHYSICAL_LEDGER_RELATIVE_DEFECT = 1.0e-3
SOURCE_RESTRICTION_TOLERANCE = 5.0e-13
THROUGHPUT_TOLERANCE = 5.0e-12
DIAGNOSED_INNER_RADIUS_RG = 15.0
DIAGNOSED_OUTER_RADIUS_RG = 60.0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-cells",
        type=int,
        action="append",
        choices=RESOLUTIONS,
        default=None,
        help="Repeat to run selected meshes; default selects N32 and N64.",
    )
    parser.add_argument(
        "--subdivisions",
        type=int,
        action="append",
        choices=SUBDIVISIONS,
        default=None,
        help="Repeat to run selected temporal rungs; default selects both.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--defer-aggregation",
        action="store_true",
        help="Run selected campaigns without aggregating all four rungs.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Load all completed campaigns and write canonical evidence.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate evidence and fresh initial states without evolving.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arrays", type=Path, default=DEFAULT_ARRAYS)
    return parser.parse_args()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _context(n_cells: int):
    return make_causal_five_field_regression_context(
        n_cells,
        **SPATIAL_OPTIONS,
    )


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
        jacobian_reuse_iterations=12,
    ).validated()


def _ledger_json(ledger) -> dict:
    relative = causal_five_field_bdf_physical_ledger_relative_defects(
        ledger
    )
    return {
        "actual_conserved_storage": [
            float(value) for value in ledger.actual_conserved_storage
        ],
        "actual_vertical_storage": [
            float(value) for value in ledger.actual_vertical_storage
        ],
        "trapezoidal_boundary_transport": [
            float(value)
            for value in ledger.trapezoidal_boundary_transport
        ],
        "trapezoidal_endogenous_source": [
            float(value)
            for value in ledger.trapezoidal_endogenous_source
        ],
        "exact_prescribed_stream_source": [
            float(value)
            for value in ledger.exact_prescribed_stream_source
        ],
        "closure_defect": [
            float(value) for value in ledger.closure_defect
        ],
        "component_relative_defects": [
            float(value) for value in relative
        ],
        "maximum_relative_defect": float(np.max(relative)),
    }


def _fixed_summary(result) -> dict:
    return {
        "subdivisions": int(result.subdivisions),
        "timestep_seconds": float(result.timestep_seconds),
        "completed_steps": int(result.completed_steps),
        "bdf1_steps": int(result.bdf1_steps),
        "bdf2_steps": int(result.bdf2_steps),
        "state_gates": result.state_gates,
        "maximum_scaled_residual": float(
            result.maximum_scaled_residual
        ),
        "maximum_scaled_algebraic_residual": float(
            result.maximum_scaled_algebraic_residual
        ),
        "maximum_scaled_primitive_change": float(
            result.maximum_scaled_primitive_change
        ),
        "maximum_scaled_total_change": float(
            result.maximum_scaled_total_change
        ),
        "maximum_discrete_ledger_relative_defect": float(
            result.maximum_discrete_ledger_relative_defect
        ),
        "maximum_linear_residual": float(
            result.maximum_linear_residual
        ),
        "maximum_newton_iterations": int(
            result.maximum_newton_iterations
        ),
        "work": {
            "implicit_solves": int(result.completed_steps),
            "function_evaluations": int(result.function_evaluations),
            "jacobian_evaluations": int(result.jacobian_evaluations),
            "newton_iterations": int(result.newton_iterations),
        },
        "cumulative_physical_ledger": _ledger_json(
            result.cumulative_physical_ledger
        ),
        "passed": bool(result.passed),
        "message": str(result.message),
    }


def _snapshot_steps(subdivisions: int) -> dict[str, int]:
    steps = {}
    for label, numerator, denominator in SNAPSHOT_FRACTIONS:
        if subdivisions % denominator != 0:
            raise ValueError("snapshot fractions do not divide subdivisions")
        steps[label] = subdivisions * numerator // denominator
    if len(set(steps.values())) != len(steps):
        raise ValueError("snapshot steps must be distinct")
    return steps


def _fixed_path(n_cells: int, subdivisions: int) -> Path:
    return (
        OUTPUT_CHECKPOINT_DIRECTORY
        / (
            f"causal_wp10c7j_N{n_cells:03d}_"
            f"balanced_bdf2_S{subdivisions:04d}.npz"
        )
    )


def _snapshot_path(n_cells: int, subdivisions: int) -> Path:
    return (
        OUTPUT_CHECKPOINT_DIRECTORY
        / (
            f"causal_wp10c7j_N{n_cells:03d}_"
            f"balanced_bdf2_S{subdivisions:04d}_snapshots.npz"
        )
    )


def _progress(
    n_cells: int,
    subdivisions: int,
    snapshots: dict[str, np.ndarray],
):
    step_labels = {
        step: label
        for label, step in _snapshot_steps(subdivisions).items()
    }
    interval = max(1, subdivisions // 8)

    def progress(completed, total, state, _history) -> None:
        if completed in step_labels:
            snapshots[step_labels[completed]] = np.array(
                state,
                copy=True,
            )
        if completed % interval == 0 or completed == total:
            print(
                json.dumps(
                    {
                        "mode": (
                            f"n{n_cells}_balanced_fixed_bdf2_"
                            f"s{subdivisions}"
                        ),
                        "completed_steps": completed,
                        "total_steps": total,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    return progress


def _initial_bundle(
    n_cells: int,
    seed_parameters: dict,
) -> dict:
    context = _context(n_cells)
    state = make_causal_five_field_seed(context, **seed_parameters)
    vector = pack_causal_five_field_state(state)
    tangent = causal_five_field_consistent_tangent_decomposition(
        context,
        vector,
    )
    physical_tangent = np.asarray(
        tangent["full"]["physical_tangent_per_s"],
        dtype=float,
    )
    reconstruction = causal_five_field_reconstruct_face_charts(
        context,
        state.primitives,
    )
    state_gates = audit_causal_five_field_state_gates(context, vector)
    if not state_gates["passed"]:
        raise RuntimeError(f"N{n_cells} balanced initial state failed")
    if context.stream_sources is None:
        raise RuntimeError("WP10c7j requires the exact stream source")
    stream_rate = float(np.sum(context.stream_sources.rest_mass))
    inner_rate = float(-C * state.weighted_face_fluxes_over_c[0, 0])
    throughput_ratio = inner_rate / stream_rate
    if abs(throughput_ratio - 1.0) > THROUGHPUT_TOLERANCE:
        raise RuntimeError(f"N{n_cells} initial throughput is incompatible")
    return {
        "context": context,
        "state": state,
        "vector": vector,
        "physical_tangent": physical_tangent,
        "vector_sha256": _array_sha256(vector),
        "state_gates": state_gates,
        "state_summary": causal_five_field_state_summary(
            context,
            vector,
        ),
        "throughput_ratio": throughput_ratio,
        "minimum_admissibility_factor": float(
            np.min(reconstruction.admissibility_factors)
        ),
        "admissibility_limited_cell_count": int(
            np.count_nonzero(
                reconstruction.admissibility_factors < 1.0 - 1.0e-12
            )
        ),
        "tangent_defects": {
            name: tangent[name]
            for name in (
                "maximum_scaled_consistency_defect",
                "maximum_residual_reconstruction_relative_defect",
                "maximum_tangent_reconstruction_relative_defect",
            )
        },
    }


def _save_snapshots(
    path: Path,
    initial: dict,
    subdivisions: int,
    snapshots: dict[str, np.ndarray],
) -> None:
    expected = _snapshot_steps(subdivisions)
    if set(snapshots) != set(expected):
        raise RuntimeError("fixed trajectory did not capture every snapshot")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray(1, dtype=np.int64),
        n_cells=np.asarray(initial["state"].n_cells, dtype=np.int64),
        subdivisions=np.asarray(subdivisions, dtype=np.int64),
        target_duration_seconds=np.asarray(TARGET_DURATION_SECONDS),
        initial_state_sha256=np.asarray(initial["vector_sha256"]),
        snapshot_labels=np.asarray(tuple(expected)),
        snapshot_steps=np.asarray(tuple(expected.values()), dtype=np.int64),
        **{
            f"state_{label}": np.asarray(snapshots[label], dtype=float)
            for label in expected
        },
    )


def _load_snapshots(
    path: Path,
    initial: dict,
    subdivisions: int,
    final_vector: np.ndarray,
) -> dict[str, np.ndarray]:
    expected = _snapshot_steps(subdivisions)
    with np.load(path, allow_pickle=False) as data:
        labels = tuple(str(value) for value in data["snapshot_labels"])
        steps = tuple(int(value) for value in data["snapshot_steps"])
        if not (
            int(data["schema_version"]) == 1
            and int(data["n_cells"]) == initial["state"].n_cells
            and int(data["subdivisions"]) == subdivisions
            and float(data["target_duration_seconds"])
            == TARGET_DURATION_SECONDS
            and str(data["initial_state_sha256"])
            == initial["vector_sha256"]
            and labels == tuple(expected)
            and steps == tuple(expected.values())
        ):
            raise RuntimeError("WP10c7j snapshot provenance failed")
        snapshots = {
            label: np.asarray(data[f"state_{label}"], dtype=float)
            for label in expected
        }
    shape = np.asarray(final_vector).shape
    if any(
        values.shape != shape or np.any(~np.isfinite(values))
        for values in snapshots.values()
    ):
        raise RuntimeError("WP10c7j snapshot state is invalid")
    if any(
        not audit_causal_five_field_state_gates(
            initial["context"],
            values,
        )["passed"]
        for values in snapshots.values()
    ):
        raise RuntimeError("WP10c7j snapshot failed a physical state gate")
    if not np.array_equal(snapshots["t_1"], final_vector):
        raise RuntimeError("WP10c7j final snapshot differs from restart")
    return snapshots


def _make_restart(
    initial: dict,
    result,
    subdivisions: int,
    wp10c7i_sha256: str,
) -> CausalFiveFieldBDFRestart:
    if result.history is None:
        raise RuntimeError("WP10c7j fixed trajectory lacks BDF history")
    return CausalFiveFieldBDFRestart(
        state_vector=np.asarray(result.state_vector, dtype=float),
        history=result.history,
        elapsed_time=TARGET_DURATION_SECONDS,
        dt_next=result.timestep_seconds,
        next_order=2,
        accepted_steps=result.completed_steps,
        rejected_attempts=0,
        provenance={
            "work_package": "WP10c7j",
            "role": "bounded_spatial_balance_fixed_bdf2",
            "base_commit": BASE_COMMIT,
            "wp10c7i_evidence_sha256": wp10c7i_sha256,
            "n_cells": int(initial["context"].grid.centers.size),
            "subdivisions": subdivisions,
            "target_duration_seconds": TARGET_DURATION_SECONDS,
            "spatial_options": dict(SPATIAL_OPTIONS),
            "initial_state_sha256": initial["vector_sha256"],
            "snapshot_steps": _snapshot_steps(subdivisions),
            "temporal_method": (
                "one BDF1 startup step then fixed equal-step BDF2"
            ),
            "result_summary": _fixed_summary(result),
        },
    )


def _load_fixed(
    initial: dict,
    subdivisions: int,
    wp10c7i_sha256: str,
) -> dict:
    context = initial["context"]
    n_cells = int(context.grid.centers.size)
    path = _fixed_path(n_cells, subdivisions)
    snapshot_path = _snapshot_path(n_cells, subdivisions)
    restart = load_causal_five_field_bdf_restart(path, context)
    provenance = restart.provenance
    summary = provenance.get("result_summary")
    timestep = TARGET_DURATION_SECONDS / subdivisions
    if not (
        provenance.get("work_package") == "WP10c7j"
        and provenance.get("role")
        == "bounded_spatial_balance_fixed_bdf2"
        and provenance.get("base_commit") == BASE_COMMIT
        and provenance.get("wp10c7i_evidence_sha256")
        == wp10c7i_sha256
        and provenance.get("n_cells") == n_cells
        and provenance.get("subdivisions") == subdivisions
        and provenance.get("target_duration_seconds")
        == TARGET_DURATION_SECONDS
        and provenance.get("spatial_options") == SPATIAL_OPTIONS
        and provenance.get("initial_state_sha256")
        == initial["vector_sha256"]
        and provenance.get("snapshot_steps")
        == _snapshot_steps(subdivisions)
        and isinstance(summary, dict)
        and summary.get("passed", False)
        and restart.elapsed_time == TARGET_DURATION_SECONDS
        and restart.history.previous_timestep_seconds == timestep
        and restart.dt_next == timestep
        and audit_causal_five_field_state_gates(
            context,
            restart.state_vector,
        )["passed"]
    ):
        raise RuntimeError(
            f"WP10c7j N{n_cells} S{subdivisions} provenance failed"
        )
    snapshots = _load_snapshots(
        snapshot_path,
        initial,
        subdivisions,
        restart.state_vector,
    )
    return {
        "restart": restart,
        "snapshots": snapshots,
        "summary": summary,
        "checkpoint": {
            "path": _relative(path),
            "sha256": _sha256(path),
            "snapshot_path": _relative(snapshot_path),
            "snapshot_sha256": _sha256(snapshot_path),
            "roundtrip_bitwise": True,
            "reused": True,
        },
    }


def _run_or_load_fixed(
    initial: dict,
    subdivisions: int,
    wp10c7i_sha256: str,
    *,
    force: bool,
) -> dict:
    context = initial["context"]
    n_cells = int(context.grid.centers.size)
    path = _fixed_path(n_cells, subdivisions)
    snapshot_path = _snapshot_path(n_cells, subdivisions)
    if path.exists() and snapshot_path.exists() and not force:
        return _load_fixed(initial, subdivisions, wp10c7i_sha256)
    timestep = TARGET_DURATION_SECONDS / subdivisions
    predictor = initial["physical_tangent"] * timestep
    snapshots: dict[str, np.ndarray] = {}
    result = evolve_causal_five_field_fixed_bdf2(
        context,
        initial["vector"],
        predictor,
        timestep,
        TARGET_DURATION_SECONDS,
        subdivisions,
        _step_config(),
        progress=_progress(n_cells, subdivisions, snapshots),
    )
    summary = _fixed_summary(result)
    if not result.passed:
        return {
            "restart": None,
            "snapshots": snapshots,
            "summary": summary,
            "checkpoint": None,
        }
    restart = _make_restart(
        initial,
        result,
        subdivisions,
        wp10c7i_sha256,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    save_causal_five_field_bdf_restart(path, context, restart)
    _save_snapshots(
        snapshot_path,
        initial,
        subdivisions,
        snapshots,
    )
    restored = load_causal_five_field_bdf_restart(path, context)
    if not causal_five_field_bdf_restarts_equal(restart, restored):
        raise RuntimeError("WP10c7j fixed restart is not bitwise")
    loaded = _load_fixed(initial, subdivisions, wp10c7i_sha256)
    loaded["checkpoint"]["reused"] = False
    return loaded


def _profile_response(initial: dict, final_vector: np.ndarray) -> dict:
    context = initial["context"]
    initial_profiles = causal_five_field_profile_fields(
        context,
        initial["vector"],
    )
    final_profiles = causal_five_field_profile_fields(
        context,
        final_vector,
    )
    return {
        name: np.asarray(
            final_profiles[name] - initial_profiles[name],
            dtype=float,
        )
        for name in initial_profiles
    }


def _selected_metrics(
    context,
    difference: np.ndarray,
    *,
    diagnosed_band: bool,
) -> dict:
    radius = (
        np.asarray(context.grid.centers, dtype=float)
        / context.grid.gravitational_radius
    )
    measures = np.asarray(context.grid.cell_measures, dtype=float)
    if diagnosed_band:
        selection = (
            (radius >= DIAGNOSED_INNER_RADIUS_RG)
            & (radius <= DIAGNOSED_OUTER_RADIUS_RG)
        )
        values = np.asarray(difference, dtype=float)[selection]
        selected_measures = measures[selection]
        selected_radius = radius[selection]
    else:
        values = np.asarray(difference, dtype=float)
        selected_measures = measures
        selected_radius = radius
    absolute = np.abs(values)
    peak = int(np.argmax(absolute))
    measure_sum = float(np.sum(selected_measures))
    return {
        "maximum_absolute_difference": float(absolute[peak]),
        "measure_weighted_l1_difference": float(
            np.sum(selected_measures * absolute) / measure_sum
        ),
        "measure_weighted_l2_difference": float(
            np.sqrt(
                np.sum(selected_measures * values**2) / measure_sum
            )
        ),
        "rms_difference": float(np.sqrt(np.mean(values**2))),
        "maximum_difference_radius_rg": float(selected_radius[peak]),
    }


def _profile_difference_rows(context, differences: dict) -> dict:
    return {
        name: {
            "full_domain": _selected_metrics(
                context,
                difference,
                diagnosed_band=False,
            ),
            "diagnosed_interior_band": _selected_metrics(
                context,
                difference,
                diagnosed_band=True,
            ),
        }
        for name, difference in differences.items()
    }


def _temporal_snapshot_comparison(
    initial: dict,
    coarse_fixed: dict,
    fine_fixed: dict,
) -> tuple[dict, dict]:
    rows = {}
    arrays = {}
    for label, _numerator, _denominator in SNAPSHOT_FRACTIONS:
        coarse_response = _profile_response(
            initial,
            coarse_fixed["snapshots"][label],
        )
        fine_response = _profile_response(
            initial,
            fine_fixed["snapshots"][label],
        )
        differences = {
            name: coarse_response[name] - fine_response[name]
            for name in coarse_response
        }
        rows[label] = _profile_difference_rows(
            initial["context"],
            differences,
        )
        for name, values in differences.items():
            arrays[f"temporal_{label}_{name}"] = values
    return rows, arrays


def _spatial_snapshot_comparison(
    coarse_initial: dict,
    fine_initial: dict,
    coarse_fixed: dict,
    fine_fixed: dict,
) -> tuple[dict, dict]:
    rows = {}
    arrays = {}
    for label, _numerator, _denominator in SNAPSHOT_FRACTIONS:
        coarse_response = _profile_response(
            coarse_initial,
            coarse_fixed["snapshots"][label],
        )
        fine_response = _profile_response(
            fine_initial,
            fine_fixed["snapshots"][label],
        )
        differences = {}
        for name, values in coarse_response.items():
            restricted = causal_restrict_cell_averages(
                coarse_initial["context"].grid,
                fine_initial["context"].grid,
                fine_response[name],
            )
            differences[name] = values - restricted
            arrays[f"spatial_{label}_n32_{name}"] = values
            arrays[f"spatial_{label}_restricted_n64_{name}"] = restricted
            arrays[f"spatial_{label}_difference_{name}"] = (
                differences[name]
            )
        rows[label] = _profile_difference_rows(
            coarse_initial["context"],
            differences,
        )
    return rows, arrays


def _term_density_response(initial: dict, vector: np.ndarray) -> dict:
    context = initial["context"]
    evaluation = evaluate_causal_five_field_dae(vector, context)
    terms = causal_five_field_residual_terms(
        context,
        vector,
        evaluation,
    )
    initial_evaluation = evaluate_causal_five_field_dae(
        initial["vector"],
        context,
    )
    initial_terms = causal_five_field_residual_terms(
        context,
        initial["vector"],
        initial_evaluation,
    )
    measures = np.asarray(context.grid.cell_measures, dtype=float)
    return {
        name: (
            np.asarray(values, dtype=float)
            - np.asarray(initial_terms[name], dtype=float)
        )
        / measures[:, None]
        for name, values in terms.items()
    }


def _term_snapshot_comparison(
    coarse_initial: dict,
    fine_initial: dict,
    coarse_fixed: dict,
    fine_fixed: dict,
) -> tuple[dict, dict]:
    rows = {}
    arrays = {}
    for label, _numerator, _denominator in SNAPSHOT_FRACTIONS:
        coarse = _term_density_response(
            coarse_initial,
            coarse_fixed["snapshots"][label],
        )
        fine = _term_density_response(
            fine_initial,
            fine_fixed["snapshots"][label],
        )
        rows[label] = {}
        for term, coarse_values in coarse.items():
            restricted = causal_restrict_cell_averages(
                coarse_initial["context"].grid,
                fine_initial["context"].grid,
                fine[term],
            )
            difference = coarse_values - restricted
            rows[label][term] = {
                field: {
                    "full_domain": _selected_metrics(
                        coarse_initial["context"],
                        difference[:, index],
                        diagnosed_band=False,
                    ),
                    "diagnosed_interior_band": _selected_metrics(
                        coarse_initial["context"],
                        difference[:, index],
                        diagnosed_band=True,
                    ),
                }
                for index, field in enumerate(CAUSAL_FIVE_FIELD_NAMES)
            }
            arrays[f"term_{label}_{term}_difference"] = difference
    return rows, arrays


def _limiter_summary(initial: dict, vector: np.ndarray) -> dict:
    state = unpack_causal_five_field_state(
        vector,
        initial["state"].n_cells,
    )
    reconstruction = causal_five_field_reconstruct_face_charts(
        initial["context"],
        state.primitives,
    )
    return {
        "minimum_admissibility_factor": float(
            np.min(reconstruction.admissibility_factors)
        ),
        "admissibility_limited_cell_count": int(
            np.count_nonzero(
                reconstruction.admissibility_factors < 1.0 - 1.0e-12
            )
        ),
    }


def _stream_matrix(context) -> np.ndarray:
    source = context.stream_sources
    if source is None:
        raise RuntimeError("WP10c7j requires the exact stream source")
    return np.column_stack(
        (
            source.rest_mass,
            source.radial_momentum_over_c,
            source.angular_momentum_over_c,
            source.killing_energy_over_c2,
            np.zeros_like(source.rest_mass),
        )
    )


def _source_restriction_audit(coarse_context, fine_context) -> dict:
    coarse = _stream_matrix(coarse_context)
    restricted = causal_restrict_cell_integrals(
        coarse_context.grid,
        fine_context.grid,
        _stream_matrix(fine_context),
    )
    scale = np.maximum(
        np.maximum(np.abs(coarse), np.abs(restricted)),
        1.0,
    )
    maximum = float(np.max(np.abs(coarse - restricted) / scale))
    return {
        "maximum_scaled_source_restriction_defect": maximum,
        "tolerance": SOURCE_RESTRICTION_TOLERANCE,
        "passed": bool(maximum <= SOURCE_RESTRICTION_TOLERANCE),
    }


def _validate_wp10c7i() -> tuple[dict, str]:
    if not WP10C7I_OUTPUT.exists():
        raise RuntimeError("WP10c7j requires the WP10c7i evidence")
    evidence = json.loads(WP10C7I_OUTPUT.read_text(encoding="utf-8"))
    decision = evidence.get("decision", {})
    passing = decision.get("passing_general_high_order_variants", [])
    variant_options = evidence.get("variants", {}).get(
        SELECTED_WP10C7I_VARIANT
    )
    normalized_options = {
        "spatial_reconstruction": "plm_smooth",
        "boundary_trace_reconstruction": "cell_centered",
        "cell_rate_scheme": "arithmetic_face",
        "cell_source_quadrature": "midpoint",
        "cell_storage_quadrature": "midpoint",
    }
    if isinstance(variant_options, dict):
        normalized_options.update(variant_options)
    if not (
        evidence.get("work_package") == "WP10c7i"
        and evidence.get("passed", False)
        and not evidence.get("trajectory_run", True)
        and decision.get("general_high_order_repair_sufficient", False)
        and not decision.get(
            "reference_state_fluctuation_operator_required",
            True,
        )
        and passing == [SELECTED_WP10C7I_VARIANT]
        and normalized_options == SPATIAL_OPTIONS
        and np.isclose(
            float(evidence.get("target_extension_seconds", np.nan)),
            TARGET_DURATION_SECONDS,
            rtol=2.0e-6,
            atol=0.0,
        )
    ):
        raise RuntimeError("WP10c7i did not authorize WP10c7j")
    return evidence, _sha256(WP10C7I_OUTPUT)


def _load_control() -> dict:
    if not WP10C7H_OUTPUT.exists():
        raise RuntimeError("WP10c7j requires the WP10c7h control")
    control = json.loads(WP10C7H_OUTPUT.read_text(encoding="utf-8"))
    if not (
        control.get("work_package") == "WP10c7h"
        and control.get("gates", {}).get(
            "all_fixed_campaigns_passed",
            False,
        )
    ):
        raise RuntimeError("WP10c7h control evidence is invalid")
    return {
        "path": _relative(WP10C7H_OUTPUT),
        "sha256": _sha256(WP10C7H_OUTPUT),
        "decision": control["decision"],
        "full_domain_log_h_over_r_difference": control[
            "primary_log_h_over_r_contract"
        ]["n32_n64_s64_spatial_difference"],
    }


def _aggregate(
    output_path: Path,
    arrays_path: Path,
    evidence: dict,
    wp10c7i_sha256: str,
    initial: dict,
    fixed: dict,
) -> dict:
    source_audit = _source_restriction_audit(
        initial[32]["context"],
        initial[64]["context"],
    )
    temporal = {}
    array_payload = {}
    for n_cells in RESOLUTIONS:
        rows, arrays = _temporal_snapshot_comparison(
            initial[n_cells],
            fixed[n_cells][32],
            fixed[n_cells][64],
        )
        temporal[str(n_cells)] = rows
        array_payload.update(
            {
                f"n{n_cells}_{name}": values
                for name, values in arrays.items()
            }
        )
    spatial, spatial_arrays = _spatial_snapshot_comparison(
        initial[32],
        initial[64],
        fixed[32][64],
        fixed[64][64],
    )
    array_payload.update(spatial_arrays)
    term_rows, term_arrays = _term_snapshot_comparison(
        initial[32],
        initial[64],
        fixed[32][64],
        fixed[64][64],
    )
    array_payload.update(term_arrays)

    temporal_uncertainties = {
        str(n_cells): {
            label: temporal[str(n_cells)][label]["log_h_over_r"][
                "full_domain"
            ]["maximum_absolute_difference"]
            for label, _numerator, _denominator in SNAPSHOT_FRACTIONS
        }
        for n_cells in RESOLUTIONS
    }
    spatial_differences = {
        label: spatial[label]["log_h_over_r"]["full_domain"][
            "maximum_absolute_difference"
        ]
        for label, _numerator, _denominator in SNAPSHOT_FRACTIONS
    }
    conservative_errors = {
        label: (
            spatial_differences[label]
            + temporal_uncertainties["32"][label]
            + temporal_uncertainties["64"][label]
        )
        for label, _numerator, _denominator in SNAPSHOT_FRACTIONS
    }
    maximum_temporal = max(
        value
        for rows in temporal_uncertainties.values()
        for value in rows.values()
    )
    maximum_spatial = max(spatial_differences.values())
    maximum_conservative = max(conservative_errors.values())
    temporal_passed = bool(
        maximum_temporal <= MAXIMUM_TEMPORAL_LOG_H_UNCERTAINTY
    )
    raw_spatial_passed = bool(
        maximum_spatial <= SPATIAL_RESPONSE_GATE
    )
    conservative_spatial_passed = bool(
        maximum_conservative <= SPATIAL_RESPONSE_GATE
    )
    ledger_maximum = max(
        fixed[n_cells][subdivisions]["summary"][
            "cumulative_physical_ledger"
        ]["maximum_relative_defect"]
        for n_cells in RESOLUTIONS
        for subdivisions in SUBDIVISIONS
    )
    ledgers_passed = bool(
        ledger_maximum <= MAXIMUM_PHYSICAL_LEDGER_RELATIVE_DEFECT
    )
    all_fixed_passed = all(
        fixed[n_cells][subdivisions]["summary"]["passed"]
        for n_cells in RESOLUTIONS
        for subdivisions in SUBDIVISIONS
    )
    initial_gates_passed = all(
        initial[n_cells]["state_gates"]["passed"]
        and abs(initial[n_cells]["throughput_ratio"] - 1.0)
        <= THROUGHPUT_TOLERANCE
        for n_cells in RESOLUTIONS
    )
    passed = bool(
        source_audit["passed"]
        and initial_gates_passed
        and all_fixed_passed
        and temporal_passed
        and ledgers_passed
        and raw_spatial_passed
        and conservative_spatial_passed
    )
    if passed:
        decision = "wp10c7j_bounded_n32_n64_trajectory_certified"
        next_authorization = "matched_adaptive_bdf2_confirmation"
    elif not (
        source_audit["passed"]
        and initial_gates_passed
        and all_fixed_passed
        and temporal_passed
        and ledgers_passed
    ):
        decision = "wp10c7j_numerical_contract_failed"
        next_authorization = "diagnose_failed_numerical_gate"
    else:
        decision = "wp10c7j_spatial_contract_failed"
        next_authorization = "return_to_spatial_balance_diagnosis"

    selected_row = evidence["decision"]["variant_rows"][
        SELECTED_WP10C7I_VARIANT
    ]
    projected = selected_row["projected_full_endpoint_difference"]
    measured = spatial_differences["t_1"]
    control = _load_control()
    control_difference = control[
        "full_domain_log_h_over_r_difference"
    ]

    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **array_payload)
    payload = {
        "work_package": "WP10c7j",
        "base_commit": BASE_COMMIT,
        "scope": (
            "fresh N32/N64 S32/S64 fixed-BDF2 confirmation of the "
            "WP10c7i general high-order spatial-balance operator"
        ),
        "trajectory_run": True,
        "spatial_options": dict(SPATIAL_OPTIONS),
        "wp10c7i_evidence": {
            "path": _relative(WP10C7I_OUTPUT),
            "sha256": wp10c7i_sha256,
            "selected_variant": SELECTED_WP10C7I_VARIANT,
            "projected_endpoint_difference": projected,
        },
        "wp10c7h_control": control,
        "initialization": {
            "policy": (
                "the exact WP10c7i shared source-compatible continuum "
                "is sampled independently on each mesh; selected "
                "algebraic maps and tangents are rebuilt; each fixed "
                "campaign creates fresh history with one BDF1 step"
            ),
            "meshes": {
                str(n_cells): {
                    "state_vector_sha256": initial[n_cells][
                        "vector_sha256"
                    ],
                    "state_gates": initial[n_cells]["state_gates"],
                    "state_summary": initial[n_cells]["state_summary"],
                    "throughput_ratio": initial[n_cells][
                        "throughput_ratio"
                    ],
                    "minimum_admissibility_factor": initial[n_cells][
                        "minimum_admissibility_factor"
                    ],
                    "admissibility_limited_cell_count": initial[n_cells][
                        "admissibility_limited_cell_count"
                    ],
                    "tangent_defects": initial[n_cells][
                        "tangent_defects"
                    ],
                }
                for n_cells in RESOLUTIONS
            },
        },
        "source_restriction_audit": source_audit,
        "fixed_campaigns": {
            str(n_cells): {
                str(subdivisions): {
                    "summary": fixed[n_cells][subdivisions]["summary"],
                    "checkpoint": fixed[n_cells][subdivisions][
                        "checkpoint"
                    ],
                    "final_limiter": _limiter_summary(
                        initial[n_cells],
                        fixed[n_cells][subdivisions][
                            "restart"
                        ].state_vector,
                    ),
                    "snapshot_limiters": {
                        label: _limiter_summary(
                            initial[n_cells],
                            fixed[n_cells][subdivisions][
                                "snapshots"
                            ][label],
                        )
                        for (
                            label,
                            _numerator,
                            _denominator,
                        ) in SNAPSHOT_FRACTIONS
                    },
                }
                for subdivisions in SUBDIVISIONS
            }
            for n_cells in RESOLUTIONS
        },
        "temporal_response_comparison": temporal,
        "spatial_s64_response_comparison": spatial,
        "spatial_s64_term_response_comparison": term_rows,
        "primary_log_h_over_r_contract": {
            "snapshot_spatial_differences": spatial_differences,
            "snapshot_n32_temporal_uncertainties": (
                temporal_uncertainties["32"]
            ),
            "snapshot_n64_temporal_uncertainties": (
                temporal_uncertainties["64"]
            ),
            "snapshot_conservative_errors": conservative_errors,
            "maximum_raw_temporal_uncertainty": maximum_temporal,
            "maximum_raw_spatial_difference": maximum_spatial,
            "maximum_spatial_plus_both_temporal_uncertainties": (
                maximum_conservative
            ),
            "endpoint_raw_spatial_difference": measured,
            "wp10c7i_projected_endpoint_difference": projected,
            "measured_to_projected_ratio": measured / projected,
            "wp10c7h_to_wp10c7j_endpoint_reduction": (
                control_difference / measured
            ),
            "gate": SPATIAL_RESPONSE_GATE,
            "maximum_temporal_uncertainty": (
                MAXIMUM_TEMPORAL_LOG_H_UNCERTAINTY
            ),
            "raw_spatial_passed": raw_spatial_passed,
            "conservative_spatial_passed": (
                conservative_spatial_passed
            ),
            "passed": bool(
                temporal_passed
                and raw_spatial_passed
                and conservative_spatial_passed
            ),
        },
        "gates": {
            "source_restriction_passed": source_audit["passed"],
            "initial_state_and_throughput_passed": initial_gates_passed,
            "all_fixed_campaigns_passed": all_fixed_passed,
            "all_snapshot_temporal_uncertainties_passed": (
                temporal_passed
            ),
            "physical_ledgers_passed": ledgers_passed,
            "maximum_physical_ledger_relative_defect": ledger_maximum,
            "all_snapshot_raw_spatial_differences_passed": (
                raw_spatial_passed
            ),
            "all_snapshot_conservative_spatial_budgets_passed": (
                conservative_spatial_passed
            ),
            "wp10c7j_passed": passed,
        },
        "decision": decision,
        "next_authorization": next_authorization,
        "artifacts": {
            "arrays_path": _relative(arrays_path),
            "arrays_sha256": _sha256(arrays_path),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    args = _arguments()
    output_path = _absolute(args.output)
    arrays_path = _absolute(args.arrays)
    evidence, evidence_sha256 = _validate_wp10c7i()

    baseline_context = make_causal_five_field_regression_context(
        16,
        spatial_reconstruction="plm_smooth",
    )
    seed_parameters = causal_five_field_regression_seed_parameters(
        baseline_context
    )
    selected_resolutions = tuple(
        sorted(set(args.n_cells or RESOLUTIONS))
    )
    selected_subdivisions = tuple(
        sorted(set(args.subdivisions or SUBDIVISIONS))
    )
    initial = {
        n_cells: _initial_bundle(n_cells, seed_parameters)
        for n_cells in (
            RESOLUTIONS
            if args.preflight or args.aggregate_only
            else selected_resolutions
        )
    }
    if args.preflight:
        source_audit = _source_restriction_audit(
            initial[32]["context"],
            initial[64]["context"],
        )
        print(
            json.dumps(
                {
                    "work_package": "WP10c7j",
                    "preflight_passed": bool(
                        source_audit["passed"]
                        and all(
                            initial[n]["state_gates"]["passed"]
                            for n in RESOLUTIONS
                        )
                    ),
                    "source_restriction_audit": source_audit,
                    "throughput_ratios": {
                        str(n): initial[n]["throughput_ratio"]
                        for n in RESOLUTIONS
                    },
                },
                sort_keys=True,
            )
        )
        return

    if not args.aggregate_only:
        for n_cells in selected_resolutions:
            for subdivisions in selected_subdivisions:
                completed = _run_or_load_fixed(
                    initial[n_cells],
                    subdivisions,
                    evidence_sha256,
                    force=args.force,
                )
                if completed["restart"] is None:
                    raise RuntimeError(
                        f"WP10c7j N{n_cells} S{subdivisions} failed: "
                        f"{completed['summary']['message']}"
                    )
        if args.defer_aggregation:
            print(
                json.dumps(
                    {
                        "work_package": "WP10c7j",
                        "campaigns_completed": {
                            "n_cells": selected_resolutions,
                            "subdivisions": selected_subdivisions,
                        },
                        "aggregation_deferred": True,
                    },
                    sort_keys=True,
                )
            )
            return

    for n_cells in RESOLUTIONS:
        if n_cells not in initial:
            initial[n_cells] = _initial_bundle(
                n_cells,
                seed_parameters,
            )
    fixed = {
        n_cells: {
            subdivisions: _load_fixed(
                initial[n_cells],
                subdivisions,
                evidence_sha256,
            )
            for subdivisions in SUBDIVISIONS
        }
        for n_cells in RESOLUTIONS
    }
    payload = _aggregate(
        output_path,
        arrays_path,
        evidence,
        evidence_sha256,
        initial,
        fixed,
    )
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "maximum_conservative_log_h_over_r_error": payload[
                    "primary_log_h_over_r_contract"
                ][
                    "maximum_spatial_plus_both_temporal_uncertainties"
                ],
                "output": _relative(output_path),
                "arrays": _relative(arrays_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
