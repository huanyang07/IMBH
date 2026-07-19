"""Run the matched N32/N64 adaptive-BDF2 WP10c7k confirmation."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d import (
    CAUSAL_FIVE_FIELD_NAMES,
    CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1,
    CausalFiveFieldAdaptiveBDF2Config,
    CausalFiveFieldAdaptiveBDF2Restart,
    CausalFiveFieldAdaptiveStepConfig,
    audit_causal_five_field_endpoint_with_reference_uncertainty,
    audit_causal_five_field_state_gates,
    causal_five_field_adaptive_bdf2_restarts_equal,
    causal_five_field_bdf_history,
    causal_five_field_bdf_physical_ledger_from_restart,
    causal_five_field_bdf_physical_ledger_relative_defects,
    causal_five_field_bdf_zero_physical_ledger,
    causal_five_field_consistent_tangent_decomposition,
    causal_five_field_profile_fields,
    causal_five_field_reconstruct_face_charts,
    causal_five_field_regression_seed_parameters,
    causal_five_field_residual_terms,
    causal_five_field_state_summary,
    causal_restrict_cell_averages,
    causal_restrict_cell_integrals,
    compare_causal_five_field_endpoint_vectors,
    evaluate_causal_five_field_dae,
    evolve_causal_five_field_adaptive_bdf2_campaign,
    load_causal_five_field_adaptive_bdf2_restart,
    load_causal_five_field_bdf_restart,
    make_causal_five_field_regression_context,
    make_causal_five_field_seed,
    pack_causal_five_field_state,
    save_causal_five_field_adaptive_bdf2_restart,
    unpack_causal_five_field_state,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "ac05f352380616f2ec0e346adaf3613b054ee3e2"
WP10C7J_OUTPUT = (
    ROOT
    / "outputs/tables/causal_spatial_balance_trajectory_wp10c7j.json"
)
OUTPUT_CHECKPOINT_DIRECTORY = (
    ROOT / "outputs/checkpoints/causal_five_field_wp10c7k"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/tables/causal_spatial_balance_adaptive_wp10c7k.json"
)
DEFAULT_ARRAYS = (
    ROOT
    / "outputs/tables/causal_spatial_balance_adaptive_wp10c7k_arrays.npz"
)
SPATIAL_OPTIONS = {
    "spatial_reconstruction": "quadratic_admissible",
    "boundary_trace_reconstruction": "plm_one_sided",
    "cell_rate_scheme": "arithmetic_face",
    "cell_source_quadrature": "gauss_legendre_4_local_rates",
    "cell_storage_quadrature": "gauss_legendre_4",
}
RESOLUTIONS = (32, 64)
FIXED_SUBDIVISIONS = (32, 64)
SNAPSHOT_FRACTIONS = (
    ("t_1_8", 1, 8),
    ("t_1_4", 1, 4),
    ("t_1_2", 1, 2),
    ("t_1", 1, 1),
)
REPLAY_SPLIT_LABEL = "t_1_2"
TARGET_DURATION_SECONDS = 1.537457597966907e-2
INITIAL_TIMESTEP_SECONDS = TARGET_DURATION_SECONDS / 64.0
SHARED_PASSING_CEILING_SECONDS = 1.9218219974586337e-3
COOLING_INNER_CUTOFF_RG = 6.0
TARGET_TIME_RELATIVE_TOLERANCE = 5.0e-14
LOCAL_ERROR_GATE_FRACTION = 0.25
PREDICTOR_ERROR_SCALE = 0.2
AUDIT_INTERVAL = 4
MAXIMUM_PHYSICAL_LEDGER_RELATIVE_DEFECT = 1.0e-3
MAXIMUM_ADAPTIVE_TO_FIXED_S64_JACOBIAN_FRACTION = 0.75
SPATIAL_RESPONSE_GATE = 5.0e-3
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
        help="Repeat to select meshes; default runs N32 and N64.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--defer-aggregation",
        action="store_true",
        help="Run selected meshes without requiring both for aggregation.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Load completed mesh campaigns and write canonical evidence.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate WP10c7j and all fresh/fixed inputs without evolving.",
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


def _plain(value):
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(name): _plain(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _plain(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


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


def _controller_config(context) -> CausalFiveFieldAdaptiveBDF2Config:
    return CausalFiveFieldAdaptiveBDF2Config(
        step_config=_step_config(),
        cooling_inner_cutoff=(
            COOLING_INNER_CUTOFF_RG
            * context.grid.gravitational_radius
        ),
        minimum_dt=1.0e-8,
        maximum_dt=SHARED_PASSING_CEILING_SECONDS,
        local_error_gate_fraction=LOCAL_ERROR_GATE_FRACTION,
        predictor_error_scale=PREDICTOR_ERROR_SCALE,
        safety_factor=0.8,
        minimum_factor=0.5,
        maximum_factor=2.0,
        maximum_retries=6,
        audit_interval=AUDIT_INTERVAL,
    ).validated()


def _controller_contract() -> dict:
    return {
        "initial_timestep_seconds": INITIAL_TIMESTEP_SECONDS,
        "minimum_timestep_seconds": 1.0e-8,
        "maximum_timestep_seconds": SHARED_PASSING_CEILING_SECONDS,
        "local_error_gate_fraction": LOCAL_ERROR_GATE_FRACTION,
        "predictor_error_scale": PREDICTOR_ERROR_SCALE,
        "safety_factor": 0.8,
        "minimum_factor": 0.5,
        "maximum_factor": 2.0,
        "maximum_retries": 6,
        "audit_interval": AUDIT_INTERVAL,
        "maximum_adaptive_to_fixed_s64_jacobian_fraction": (
            MAXIMUM_ADAPTIVE_TO_FIXED_S64_JACOBIAN_FRACTION
        ),
    }


def _snapshot_targets() -> dict[str, float]:
    return {
        label: TARGET_DURATION_SECONDS * numerator / denominator
        for label, numerator, denominator in SNAPSHOT_FRACTIONS
    }


def _checkpoint_path(n_cells: int, label: str) -> Path:
    return (
        OUTPUT_CHECKPOINT_DIRECTORY
        / f"causal_wp10c7k_N{n_cells:03d}_adaptive_{label}.npz"
    )


def _replay_path(n_cells: int) -> Path:
    return (
        OUTPUT_CHECKPOINT_DIRECTORY
        / f"causal_wp10c7k_N{n_cells:03d}_adaptive_replay_final.npz"
    )


def _initial_bundle(n_cells: int, seed_parameters: dict) -> dict:
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
        raise RuntimeError(f"N{n_cells} WP10c7k initial state failed")
    if context.stream_sources is None:
        raise RuntimeError("WP10c7k requires the exact stream source")
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
        "state_summary": causal_five_field_state_summary(context, vector),
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


def _validate_wp10c7j() -> tuple[dict, str]:
    if not WP10C7J_OUTPUT.exists():
        raise RuntimeError("WP10c7k requires canonical WP10c7j evidence")
    evidence = json.loads(WP10C7J_OUTPUT.read_text(encoding="utf-8"))
    artifact = evidence.get("artifacts", {})
    arrays_path = ROOT / str(artifact.get("arrays_path", ""))
    if not (
        evidence.get("work_package") == "WP10c7j"
        and evidence.get("decision")
        == "wp10c7j_bounded_n32_n64_trajectory_certified"
        and evidence.get("next_authorization")
        == "matched_adaptive_bdf2_confirmation"
        and evidence.get("spatial_options") == SPATIAL_OPTIONS
        and evidence.get("gates", {}).get("wp10c7j_passed", False)
        and arrays_path.exists()
        and _sha256(arrays_path) == artifact.get("arrays_sha256")
    ):
        raise RuntimeError("WP10c7j did not authorize WP10c7k")
    return evidence, _sha256(WP10C7J_OUTPUT)


def _load_fixed_snapshots(
    path: Path,
    initial: dict,
    subdivisions: int,
    final_vector: np.ndarray,
) -> dict[str, np.ndarray]:
    expected_labels = tuple(label for label, _, _ in SNAPSHOT_FRACTIONS)
    expected_steps = tuple(
        subdivisions * numerator // denominator
        for _, numerator, denominator in SNAPSHOT_FRACTIONS
    )
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
            and labels == expected_labels
            and steps == expected_steps
        ):
            raise RuntimeError("WP10c7j fixed snapshot provenance failed")
        snapshots = {
            label: np.asarray(data[f"state_{label}"], dtype=float)
            for label in expected_labels
        }
    if any(
        values.shape != np.asarray(final_vector).shape
        or np.any(~np.isfinite(values))
        or not audit_causal_five_field_state_gates(
            initial["context"],
            values,
        )["passed"]
        for values in snapshots.values()
    ):
        raise RuntimeError("WP10c7j fixed snapshot state is invalid")
    if not np.array_equal(snapshots["t_1"], final_vector):
        raise RuntimeError("WP10c7j fixed final snapshot differs")
    return snapshots


def _load_fixed_reference(
    evidence: dict,
    initial: dict,
    subdivisions: int,
) -> dict:
    n_cells = initial["state"].n_cells
    entry = evidence["fixed_campaigns"][str(n_cells)][str(subdivisions)]
    checkpoint = entry["checkpoint"]
    path = ROOT / checkpoint["path"]
    snapshot_path = ROOT / checkpoint["snapshot_path"]
    if not (
        path.exists()
        and snapshot_path.exists()
        and _sha256(path) == checkpoint["sha256"]
        and _sha256(snapshot_path) == checkpoint["snapshot_sha256"]
        and entry["summary"]["passed"]
    ):
        raise RuntimeError(
            f"WP10c7j N{n_cells} S{subdivisions} artifacts differ"
        )
    restart = load_causal_five_field_bdf_restart(
        path,
        initial["context"],
    )
    provenance = restart.provenance
    if not (
        provenance.get("work_package") == "WP10c7j"
        and provenance.get("role")
        == "bounded_spatial_balance_fixed_bdf2"
        and provenance.get("n_cells") == n_cells
        and provenance.get("subdivisions") == subdivisions
        and provenance.get("target_duration_seconds")
        == TARGET_DURATION_SECONDS
        and provenance.get("spatial_options") == SPATIAL_OPTIONS
        and provenance.get("initial_state_sha256")
        == initial["vector_sha256"]
        and restart.elapsed_time == TARGET_DURATION_SECONDS
        and audit_causal_five_field_state_gates(
            initial["context"],
            restart.state_vector,
        )["passed"]
    ):
        raise RuntimeError(
            f"WP10c7j N{n_cells} S{subdivisions} restart differs"
        )
    return {
        "restart": restart,
        "snapshots": _load_fixed_snapshots(
            snapshot_path,
            initial,
            subdivisions,
            restart.state_vector,
        ),
        "summary": entry["summary"],
        "checkpoint": checkpoint,
    }


def _initial_adaptive_restart(
    initial: dict,
    evidence_sha256: str,
) -> CausalFiveFieldAdaptiveBDF2Restart:
    predictor = (
        initial["physical_tangent"] * INITIAL_TIMESTEP_SECONDS
    )
    history = causal_five_field_bdf_history(
        initial["context"],
        initial["vector"],
        predictor,
        INITIAL_TIMESTEP_SECONDS,
    )
    zero = causal_five_field_bdf_zero_physical_ledger()
    n_cells = initial["state"].n_cells
    return CausalFiveFieldAdaptiveBDF2Restart(
        state_vector=np.asarray(initial["vector"], dtype=float),
        history=history,
        older_physical_increment=np.asarray(predictor, dtype=float),
        older_timestep_seconds=INITIAL_TIMESTEP_SECONDS,
        cumulative_actual_conserved_storage=(
            zero.actual_conserved_storage
        ),
        cumulative_actual_vertical_storage=(
            zero.actual_vertical_storage
        ),
        cumulative_boundary_transport=(
            zero.trapezoidal_boundary_transport
        ),
        cumulative_endogenous_source=(
            zero.trapezoidal_endogenous_source
        ),
        cumulative_stream_source=zero.exact_prescribed_stream_source,
        cumulative_closure_defect=zero.closure_defect,
        elapsed_time=0.0,
        dt_next=INITIAL_TIMESTEP_SECONDS,
        next_order=1,
        accepted_steps=0,
        accepted_bdf2_steps=0,
        rejected_attempts=0,
        audit_count=0,
        provenance={
            "work_package": "WP10c7k",
            "role": "matched_spatial_balance_adaptive_bdf2",
            "base_commit": BASE_COMMIT,
            "wp10c7j_evidence_sha256": evidence_sha256,
            "n_cells": n_cells,
            "target_duration_seconds": TARGET_DURATION_SECONDS,
            "spatial_options": dict(SPATIAL_OPTIONS),
            "initial_state_sha256": initial["vector_sha256"],
            "controller": _controller_contract(),
            "segments": [],
        },
    )


def _attempt_work(attempt) -> dict[str, int]:
    solved = [attempt.step]
    if attempt.independent_audit is not None:
        solved.append(attempt.independent_audit.first_half_step)
        if attempt.independent_audit.second_half_step is not None:
            solved.append(attempt.independent_audit.second_half_step)
    return {
        "implicit_solves": len(solved),
        "function_evaluations": sum(
            int(step.function_evaluations) for step in solved
        ),
        "jacobian_evaluations": sum(
            int(step.jacobian_evaluations) for step in solved
        ),
        "newton_iterations": sum(
            int(step.iterations) for step in solved
        ),
    }


def _attempt_json(attempt) -> dict:
    return {
        "timestep_seconds": float(attempt.timestep_seconds),
        "order": int(attempt.order),
        "accepted": bool(attempt.accepted),
        "failure_class": str(attempt.failure_class),
        "proposed_factor": float(attempt.proposed_factor),
        "maximum_scaled_residual": float(
            attempt.step.maximum_scaled_residual
        ),
        "maximum_scaled_algebraic_residual": float(
            attempt.step.maximum_scaled_algebraic_residual
        ),
        "maximum_discrete_ledger_relative_defect": float(
            attempt.step.maximum_discrete_ledger_relative_defect
        ),
        "local_gate_audit": attempt.local_gate_audit,
        "independent_audit": (
            {
                "passed": bool(attempt.independent_audit.passed),
                "temporal_gate_audit": (
                    attempt.independent_audit.temporal_gate_audit
                ),
            }
            if attempt.independent_audit is not None
            else None
        ),
        "work": _attempt_work(attempt),
    }


def _segment_summary(
    label: str,
    start: CausalFiveFieldAdaptiveBDF2Restart,
    campaign,
) -> dict:
    records = []
    work = {
        "implicit_solves": 0,
        "function_evaluations": 0,
        "jacobian_evaluations": 0,
        "newton_iterations": 0,
    }
    for result in campaign.steps:
        attempts = [_attempt_json(attempt) for attempt in result.attempts]
        for attempt in result.attempts:
            row = _attempt_work(attempt)
            for name in work:
                work[name] += row[name]
        records.append(
            {
                "accepted": bool(result.accepted),
                "order": int(result.order),
                "dt_used": float(result.dt_used),
                "dt_next": float(result.dt_next),
                "attempts": attempts,
            }
        )
    accepted = [row for row in records if row["accepted"]]
    return _plain(
        {
            "label": label,
            "start_elapsed_time_seconds": float(start.elapsed_time),
            "target_elapsed_time_seconds": float(
                campaign.restart.elapsed_time
            ),
            "passed": bool(campaign.passed),
            "message": str(campaign.message),
            "accepted_steps": int(
                campaign.restart.accepted_steps - start.accepted_steps
            ),
            "accepted_bdf2_steps": int(
                campaign.restart.accepted_bdf2_steps
                - start.accepted_bdf2_steps
            ),
            "rejected_attempts": int(
                campaign.restart.rejected_attempts
                - start.rejected_attempts
            ),
            "audit_count": int(
                campaign.restart.audit_count - start.audit_count
            ),
            "minimum_dt_used": min(
                row["dt_used"] for row in accepted
            ),
            "maximum_dt_used": max(
                row["dt_used"] for row in accepted
            ),
            "work": work,
            "records": records,
        }
    )


def _append_segment(
    restart: CausalFiveFieldAdaptiveBDF2Restart,
    segment: dict,
) -> CausalFiveFieldAdaptiveBDF2Restart:
    provenance = dict(restart.provenance)
    segments = list(provenance.get("segments", []))
    segments.append(_plain(segment))
    provenance["segments"] = segments
    return replace(restart, provenance=provenance)


def _validate_adaptive_restart(
    restart: CausalFiveFieldAdaptiveBDF2Restart,
    initial: dict,
    evidence_sha256: str,
    label: str,
) -> None:
    labels = [name for name, _, _ in SNAPSHOT_FRACTIONS]
    expected_prefix = labels[: labels.index(label) + 1]
    provenance = restart.provenance
    segments = provenance.get("segments", [])
    if not (
        provenance.get("work_package") == "WP10c7k"
        and provenance.get("role")
        == "matched_spatial_balance_adaptive_bdf2"
        and provenance.get("wp10c7j_evidence_sha256")
        == evidence_sha256
        and provenance.get("n_cells") == initial["state"].n_cells
        and provenance.get("target_duration_seconds")
        == TARGET_DURATION_SECONDS
        and provenance.get("spatial_options") == SPATIAL_OPTIONS
        and provenance.get("initial_state_sha256")
        == initial["vector_sha256"]
        and provenance.get("controller") == _controller_contract()
        and [row.get("label") for row in segments] == expected_prefix
        and all(row.get("passed", False) for row in segments)
        and restart.elapsed_time == _snapshot_targets()[label]
        and audit_causal_five_field_state_gates(
            initial["context"],
            restart.state_vector,
        )["passed"]
    ):
        raise RuntimeError(
            f"WP10c7k N{initial['state'].n_cells} {label} differs"
        )


def _load_adaptive_snapshot(
    initial: dict,
    evidence_sha256: str,
    label: str,
) -> CausalFiveFieldAdaptiveBDF2Restart:
    path = _checkpoint_path(initial["state"].n_cells, label)
    restart = load_causal_five_field_adaptive_bdf2_restart(
        path,
        initial["context"],
    )
    _validate_adaptive_restart(
        restart,
        initial,
        evidence_sha256,
        label,
    )
    return restart


def _progress(n_cells: int, label: str):
    def progress(relative_step, restart, result) -> None:
        print(
            json.dumps(
                {
                    "mode": f"n{n_cells}_wp10c7k_adaptive_{label}",
                    "relative_accepted_step": int(relative_step),
                    "accepted_steps": int(restart.accepted_steps),
                    "order": int(result.order),
                    "dt_used": float(result.dt_used),
                    "dt_next": float(result.dt_next),
                    "elapsed_time": float(restart.elapsed_time),
                    "audits": int(restart.audit_count),
                    "rejected_attempts": int(
                        restart.rejected_attempts
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    return progress


def _run_segment(
    initial: dict,
    start: CausalFiveFieldAdaptiveBDF2Restart,
    label: str,
):
    campaign = evolve_causal_five_field_adaptive_bdf2_campaign(
        initial["context"],
        start,
        _snapshot_targets()[label],
        _controller_config(initial["context"]),
        target_time_relative_tolerance=TARGET_TIME_RELATIVE_TOLERANCE,
        progress=_progress(initial["state"].n_cells, label),
    )
    if not campaign.passed:
        raise RuntimeError(
            f"WP10c7k N{initial['state'].n_cells} {label} failed: "
            f"{campaign.message}"
        )
    return _append_segment(
        campaign.restart,
        _segment_summary(label, start, campaign),
    )


def _run_or_load_mesh(
    initial: dict,
    evidence_sha256: str,
    *,
    force: bool,
) -> dict:
    n_cells = initial["state"].n_cells
    state = _initial_adaptive_restart(initial, evidence_sha256)
    snapshots = {}
    roundtrips = {}
    for label, _, _ in SNAPSHOT_FRACTIONS:
        path = _checkpoint_path(n_cells, label)
        if path.exists() and not force:
            state = _load_adaptive_snapshot(
                initial,
                evidence_sha256,
                label,
            )
            roundtrips[label] = True
        else:
            state = _run_segment(initial, state, label)
            path.parent.mkdir(parents=True, exist_ok=True)
            save_causal_five_field_adaptive_bdf2_restart(
                path,
                initial["context"],
                state,
            )
            restored = _load_adaptive_snapshot(
                initial,
                evidence_sha256,
                label,
            )
            roundtrips[label] = (
                causal_five_field_adaptive_bdf2_restarts_equal(
                    state,
                    restored,
                )
            )
            if not roundtrips[label]:
                raise RuntimeError(
                    f"WP10c7k N{n_cells} {label} is not bitwise"
                )
            state = restored
        snapshots[label] = state

    replay_path = _replay_path(n_cells)
    if replay_path.exists() and not force:
        replay = load_causal_five_field_adaptive_bdf2_restart(
            replay_path,
            initial["context"],
        )
        _validate_adaptive_restart(
            replay,
            initial,
            evidence_sha256,
            "t_1",
        )
    else:
        split = snapshots[REPLAY_SPLIT_LABEL]
        replay = _run_segment(initial, split, "t_1")
        save_causal_five_field_adaptive_bdf2_restart(
            replay_path,
            initial["context"],
            replay,
        )
        replay = load_causal_five_field_adaptive_bdf2_restart(
            replay_path,
            initial["context"],
        )
        _validate_adaptive_restart(
            replay,
            initial,
            evidence_sha256,
            "t_1",
        )
    replay_bitwise = causal_five_field_adaptive_bdf2_restarts_equal(
        snapshots["t_1"],
        replay,
    )
    if not replay_bitwise:
        raise RuntimeError(f"WP10c7k N{n_cells} replay differs")
    return {
        "snapshots": snapshots,
        "final": snapshots["t_1"],
        "roundtrips": roundtrips,
        "replay": replay,
        "replay_bitwise": replay_bitwise,
        "checkpoint_rows": {
            label: {
                "path": _relative(_checkpoint_path(n_cells, label)),
                "sha256": _sha256(_checkpoint_path(n_cells, label)),
                "roundtrip_bitwise": roundtrips[label],
            }
            for label, _, _ in SNAPSHOT_FRACTIONS
        },
        "replay_checkpoint": {
            "path": _relative(replay_path),
            "sha256": _sha256(replay_path),
            "endpoint_replay_bitwise": replay_bitwise,
        },
    }


def _sum_work(segments: list[dict]) -> dict[str, int]:
    total = {
        "implicit_solves": 0,
        "function_evaluations": 0,
        "jacobian_evaluations": 0,
        "newton_iterations": 0,
    }
    for segment in segments:
        for name in total:
            total[name] += int(segment["work"][name])
    return total


def _campaign_audit(adaptive: dict, fixed_s64: dict) -> dict:
    restart = adaptive["final"]
    segments = list(restart.provenance["segments"])
    attempts = [
        attempt
        for segment in segments
        for record in segment["records"]
        for attempt in record["attempts"]
    ]
    audits = [
        attempt["independent_audit"]
        for attempt in attempts
        if attempt["independent_audit"] is not None
    ]
    accepted_attempts = [
        attempt for attempt in attempts if attempt["accepted"]
    ]
    local_estimator_passed = all(
        attempt["local_gate_audit"] is None
        or attempt["local_gate_audit"]["passed"]
        for attempt in accepted_attempts
    )
    audits_passed = bool(
        audits and all(audit["passed"] for audit in audits)
    )
    work = _sum_work(segments)
    fixed_work = fixed_s64["summary"]["work"]
    jacobian_fraction = (
        work["jacobian_evaluations"]
        / fixed_work["jacobian_evaluations"]
    )
    function_fraction = (
        work["function_evaluations"]
        / fixed_work["function_evaluations"]
    )
    work_passed = bool(
        jacobian_fraction
        <= MAXIMUM_ADAPTIVE_TO_FIXED_S64_JACOBIAN_FRACTION
    )
    ledger = causal_five_field_bdf_physical_ledger_from_restart(
        restart
    )
    relative = causal_five_field_bdf_physical_ledger_relative_defects(
        ledger
    )
    maximum_ledger = float(np.max(relative))
    ledger_passed = bool(
        maximum_ledger <= MAXIMUM_PHYSICAL_LEDGER_RELATIVE_DEFECT
    )
    return {
        "segments": segments,
        "accepted_steps": int(restart.accepted_steps),
        "accepted_bdf2_steps": int(restart.accepted_bdf2_steps),
        "rejected_attempts": int(restart.rejected_attempts),
        "audit_count": int(restart.audit_count),
        "minimum_dt_used": min(
            segment["minimum_dt_used"] for segment in segments
        ),
        "maximum_dt_used": max(
            segment["maximum_dt_used"] for segment in segments
        ),
        "all_segments_passed": all(
            segment["passed"] for segment in segments
        ),
        "local_estimator_passed": local_estimator_passed,
        "independent_audits": {
            "count": len(audits),
            "maximum_normalized_error": max(
                float(
                    audit["temporal_gate_audit"][
                        "maximum_normalized_error"
                    ]
                )
                for audit in audits
                if audit["temporal_gate_audit"] is not None
            ),
            "passed": audits_passed,
        },
        "work_audit": {
            "adaptive": work,
            "fixed_s64": fixed_work,
            "adaptive_to_fixed_s64_jacobian_fraction": (
                jacobian_fraction
            ),
            "adaptive_to_fixed_s64_function_fraction": (
                function_fraction
            ),
            "maximum_jacobian_fraction": (
                MAXIMUM_ADAPTIVE_TO_FIXED_S64_JACOBIAN_FRACTION
            ),
            "passed": work_passed,
        },
        "physical_ledger": {
            "actual_conserved_storage": _plain(
                ledger.actual_conserved_storage
            ),
            "actual_vertical_storage": _plain(
                ledger.actual_vertical_storage
            ),
            "trapezoidal_boundary_transport": _plain(
                ledger.trapezoidal_boundary_transport
            ),
            "trapezoidal_endogenous_source": _plain(
                ledger.trapezoidal_endogenous_source
            ),
            "exact_prescribed_stream_source": _plain(
                ledger.exact_prescribed_stream_source
            ),
            "closure_defect": _plain(ledger.closure_defect),
            "component_relative_defects": _plain(relative),
            "maximum_relative_defect": maximum_ledger,
            "gate": MAXIMUM_PHYSICAL_LEDGER_RELATIVE_DEFECT,
            "passed": ledger_passed,
        },
        "restart_replay": {
            "snapshot_checkpoints": adaptive["checkpoint_rows"],
            "replay_checkpoint": adaptive["replay_checkpoint"],
            "all_snapshot_roundtrips_bitwise": all(
                adaptive["roundtrips"].values()
            ),
            "endpoint_replay_bitwise": adaptive["replay_bitwise"],
            "passed": bool(
                all(adaptive["roundtrips"].values())
                and adaptive["replay_bitwise"]
            ),
        },
        "passed": bool(
            all(segment["passed"] for segment in segments)
            and local_estimator_passed
            and audits_passed
            and work_passed
            and ledger_passed
            and all(adaptive["roundtrips"].values())
            and adaptive["replay_bitwise"]
        ),
    }


def _profile_response(initial: dict, vector: np.ndarray) -> dict:
    initial_profiles = causal_five_field_profile_fields(
        initial["context"],
        initial["vector"],
    )
    final_profiles = causal_five_field_profile_fields(
        initial["context"],
        vector,
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
        raise RuntimeError("WP10c7k requires the exact stream source")
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


def _temporal_rows(
    initial: dict,
    fixed: dict,
    adaptive: dict,
) -> tuple[dict, dict]:
    rows = {}
    array_payload = {}
    cutoff = (
        COOLING_INNER_CUTOFF_RG
        * initial["context"].grid.gravitational_radius
    )
    for label, _, _ in SNAPSHOT_FRACTIONS:
        adaptive_vector = adaptive["snapshots"][label].state_vector
        fixed_s32 = fixed[32]["snapshots"][label]
        fixed_s64 = fixed[64]["snapshots"][label]
        endpoint_errors = compare_causal_five_field_endpoint_vectors(
            initial["context"],
            initial["vector"],
            adaptive_vector,
            fixed_s64,
            cooling_inner_cutoff=cutoff,
        )
        reference_errors = compare_causal_five_field_endpoint_vectors(
            initial["context"],
            initial["vector"],
            fixed_s32,
            fixed_s64,
            cooling_inner_cutoff=cutoff,
        )
        audit = audit_causal_five_field_endpoint_with_reference_uncertainty(
            endpoint_errors,
            reference_errors,
            dict(CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1),
        )
        endpoint_profiles = _profile_response(
            initial,
            adaptive_vector,
        )
        reference_profiles = _profile_response(initial, fixed_s64)
        differences = {
            name: endpoint_profiles[name] - reference_profiles[name]
            for name in endpoint_profiles
        }
        rows[label] = {
            "adaptive_to_fixed_s64": endpoint_errors,
            "fixed_s32_to_s64_reference_uncertainty": (
                reference_errors
            ),
            "combined_audit": audit,
            "profile_differences": _profile_difference_rows(
                initial["context"],
                differences,
            ),
        }
        for name, values in differences.items():
            array_payload[f"{label}_adaptive_fixed_{name}"] = values
    return rows, array_payload


def _spatial_rows(
    initial: dict,
    adaptive: dict,
    temporal: dict,
) -> tuple[dict, dict, dict]:
    rows = {}
    arrays = {}
    conservative = {}
    for label, _, _ in SNAPSHOT_FRACTIONS:
        coarse_response = _profile_response(
            initial[32],
            adaptive[32]["snapshots"][label].state_vector,
        )
        fine_response = _profile_response(
            initial[64],
            adaptive[64]["snapshots"][label].state_vector,
        )
        differences = {}
        for name, coarse_values in coarse_response.items():
            restricted = causal_restrict_cell_averages(
                initial[32]["context"].grid,
                initial[64]["context"].grid,
                fine_response[name],
            )
            differences[name] = coarse_values - restricted
            arrays[f"{label}_n32_{name}"] = coarse_values
            arrays[f"{label}_restricted_n64_{name}"] = restricted
            arrays[f"{label}_spatial_difference_{name}"] = (
                differences[name]
            )
        rows[label] = _profile_difference_rows(
            initial[32]["context"],
            differences,
        )
        raw = rows[label]["log_h_over_r"]["full_domain"][
            "maximum_absolute_difference"
        ]
        n32_endpoint = temporal[32][label][
            "adaptive_to_fixed_s64"
        ]["maximum_log_h_over_r_profile"]
        n32_reference = temporal[32][label][
            "fixed_s32_to_s64_reference_uncertainty"
        ]["maximum_log_h_over_r_profile"]
        n64_endpoint = temporal[64][label][
            "adaptive_to_fixed_s64"
        ]["maximum_log_h_over_r_profile"]
        n64_reference = temporal[64][label][
            "fixed_s32_to_s64_reference_uncertainty"
        ]["maximum_log_h_over_r_profile"]
        conservative[label] = {
            "raw_adaptive_n32_n64_spatial_difference": raw,
            "n32_adaptive_to_fixed_s64": n32_endpoint,
            "n32_fixed_reference_uncertainty": n32_reference,
            "n64_adaptive_to_fixed_s64": n64_endpoint,
            "n64_fixed_reference_uncertainty": n64_reference,
            "conservative_total": (
                raw
                + n32_endpoint
                + n32_reference
                + n64_endpoint
                + n64_reference
            ),
        }
    return rows, conservative, arrays


def _endpoint_term_rows(
    initial: dict,
    adaptive: dict,
) -> tuple[dict, dict]:
    coarse = _term_density_response(
        initial[32],
        adaptive[32]["final"].state_vector,
    )
    fine = _term_density_response(
        initial[64],
        adaptive[64]["final"].state_vector,
    )
    rows = {}
    arrays = {}
    for term, coarse_values in coarse.items():
        restricted = causal_restrict_cell_averages(
            initial[32]["context"].grid,
            initial[64]["context"].grid,
            fine[term],
        )
        difference = coarse_values - restricted
        arrays[f"endpoint_term_{term}_difference"] = difference
        rows[term] = {
            field: {
                "full_domain": _selected_metrics(
                    initial[32]["context"],
                    difference[:, index],
                    diagnosed_band=False,
                ),
                "diagnosed_interior_band": _selected_metrics(
                    initial[32]["context"],
                    difference[:, index],
                    diagnosed_band=True,
                ),
            }
            for index, field in enumerate(CAUSAL_FIVE_FIELD_NAMES)
        }
    return rows, arrays


def _aggregate(
    output_path: Path,
    arrays_path: Path,
    evidence: dict,
    evidence_sha256: str,
    initial: dict,
    fixed: dict,
    adaptive: dict,
) -> dict:
    source_audit = _source_restriction_audit(
        initial[32]["context"],
        initial[64]["context"],
    )
    temporal = {}
    arrays = {}
    campaigns = {}
    for n_cells in RESOLUTIONS:
        temporal_rows, temporal_arrays = _temporal_rows(
            initial[n_cells],
            fixed[n_cells],
            adaptive[n_cells],
        )
        temporal[n_cells] = temporal_rows
        arrays.update(
            {
                f"n{n_cells}_{name}": values
                for name, values in temporal_arrays.items()
            }
        )
        campaigns[n_cells] = _campaign_audit(
            adaptive[n_cells],
            fixed[n_cells][64],
        )
    spatial, conservative, spatial_arrays = _spatial_rows(
        initial,
        adaptive,
        temporal,
    )
    arrays.update(spatial_arrays)
    term_rows, term_arrays = _endpoint_term_rows(initial, adaptive)
    arrays.update(term_arrays)

    temporal_passed = all(
        temporal[n_cells][label]["combined_audit"]["passed"]
        for n_cells in RESOLUTIONS
        for label, _, _ in SNAPSHOT_FRACTIONS
    )
    maximum_temporal_normalized = max(
        float(
            temporal[n_cells][label]["combined_audit"][
                "maximum_combined_normalized_error"
            ]
        )
        for n_cells in RESOLUTIONS
        for label, _, _ in SNAPSHOT_FRACTIONS
    )
    campaign_passed = all(
        campaigns[n_cells]["passed"] for n_cells in RESOLUTIONS
    )
    raw_spatial_passed = all(
        row["raw_adaptive_n32_n64_spatial_difference"]
        <= SPATIAL_RESPONSE_GATE
        for row in conservative.values()
    )
    conservative_spatial_passed = all(
        row["conservative_total"] <= SPATIAL_RESPONSE_GATE
        for row in conservative.values()
    )
    maximum_raw_spatial = max(
        row["raw_adaptive_n32_n64_spatial_difference"]
        for row in conservative.values()
    )
    maximum_conservative = max(
        row["conservative_total"] for row in conservative.values()
    )
    initial_passed = all(
        initial[n_cells]["state_gates"]["passed"]
        and abs(initial[n_cells]["throughput_ratio"] - 1.0)
        <= THROUGHPUT_TOLERANCE
        for n_cells in RESOLUTIONS
    )
    passed = bool(
        source_audit["passed"]
        and initial_passed
        and temporal_passed
        and campaign_passed
        and raw_spatial_passed
        and conservative_spatial_passed
    )
    if passed:
        decision = "wp10c7k_matched_adaptive_bdf2_certified"
        next_authorization = (
            "no_tide_duration_ladder_characteristic_clock"
        )
    elif not (
        source_audit["passed"]
        and initial_passed
        and temporal_passed
        and campaign_passed
    ):
        decision = "wp10c7k_adaptive_temporal_or_numerical_gate_failed"
        next_authorization = "diagnose_adaptive_controller_failure"
    else:
        decision = "wp10c7k_adaptive_spatial_budget_failed"
        next_authorization = "compare_adaptive_fixed_term_responses"

    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **arrays)
    payload = {
        "work_package": "WP10c7k",
        "base_commit": BASE_COMMIT,
        "scope": (
            "matched N32/N64 adaptive-BDF2 confirmation of the "
            "WP10c7j selected spatial-balance operator"
        ),
        "spatial_options": dict(SPATIAL_OPTIONS),
        "controller": _controller_contract(),
        "target_duration_seconds": TARGET_DURATION_SECONDS,
        "snapshot_fractions": {
            label: numerator / denominator
            for label, numerator, denominator in SNAPSHOT_FRACTIONS
        },
        "wp10c7j_evidence": {
            "path": _relative(WP10C7J_OUTPUT),
            "sha256": evidence_sha256,
            "decision": evidence["decision"],
            "fixed_endpoint_spatial_difference": evidence[
                "primary_log_h_over_r_contract"
            ]["endpoint_raw_spatial_difference"],
            "fixed_maximum_conservative_difference": evidence[
                "primary_log_h_over_r_contract"
            ][
                "maximum_spatial_plus_both_temporal_uncertainties"
            ],
        },
        "initialization": {
            "policy": (
                "the exact WP10c7j source-compatible continuum is "
                "resampled independently on N32/N64; selected maps and "
                "tangents are rebuilt; each adaptive mesh begins with "
                "one BDF1 step and the unchanged WP10c7c-d controller"
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
        "fixed_references": {
            str(n_cells): {
                str(subdivisions): {
                    "summary": fixed[n_cells][subdivisions]["summary"],
                    "checkpoint": fixed[n_cells][subdivisions][
                        "checkpoint"
                    ],
                }
                for subdivisions in FIXED_SUBDIVISIONS
            }
            for n_cells in RESOLUTIONS
        },
        "adaptive_campaigns": {
            str(n_cells): {
                **campaigns[n_cells],
                "snapshot_limiters": {
                    label: _limiter_summary(
                        initial[n_cells],
                        adaptive[n_cells]["snapshots"][
                            label
                        ].state_vector,
                    )
                    for label, _, _ in SNAPSHOT_FRACTIONS
                },
            }
            for n_cells in RESOLUTIONS
        },
        "temporal_comparison": {
            str(n_cells): temporal[n_cells]
            for n_cells in RESOLUTIONS
        },
        "adaptive_spatial_response_comparison": spatial,
        "adaptive_endpoint_term_response_comparison": term_rows,
        "primary_log_h_over_r_contract": {
            "snapshot_rows": conservative,
            "maximum_temporal_combined_normalized_error": (
                maximum_temporal_normalized
            ),
            "maximum_raw_adaptive_spatial_difference": (
                maximum_raw_spatial
            ),
            "maximum_conservative_spatial_error": (
                maximum_conservative
            ),
            "fixed_wp10c7j_endpoint_spatial_difference": evidence[
                "primary_log_h_over_r_contract"
            ]["endpoint_raw_spatial_difference"],
            "adaptive_to_fixed_endpoint_spatial_ratio": (
                conservative["t_1"][
                    "raw_adaptive_n32_n64_spatial_difference"
                ]
                / evidence["primary_log_h_over_r_contract"][
                    "endpoint_raw_spatial_difference"
                ]
            ),
            "gate": SPATIAL_RESPONSE_GATE,
            "temporal_passed": temporal_passed,
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
            "initial_state_and_throughput_passed": initial_passed,
            "both_adaptive_campaigns_passed": campaign_passed,
            "all_snapshot_temporal_audits_passed": temporal_passed,
            "all_snapshot_raw_spatial_differences_passed": (
                raw_spatial_passed
            ),
            "all_snapshot_conservative_spatial_budgets_passed": (
                conservative_spatial_passed
            ),
            "wp10c7k_passed": passed,
        },
        "decision": decision,
        "next_authorization": next_authorization,
        "artifacts": {
            "arrays_path": _relative(arrays_path),
            "arrays_sha256": _sha256(arrays_path),
        },
    }
    _write_json(output_path, payload)
    return payload


def main() -> None:
    args = _arguments()
    output_path = _absolute(args.output)
    arrays_path = _absolute(args.arrays)
    evidence, evidence_sha256 = _validate_wp10c7j()
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
    initial = {
        n_cells: _initial_bundle(n_cells, seed_parameters)
        for n_cells in (
            RESOLUTIONS
            if args.preflight or args.aggregate_only
            else selected_resolutions
        )
    }
    fixed = {
        n_cells: {
            subdivisions: _load_fixed_reference(
                evidence,
                initial[n_cells],
                subdivisions,
            )
            for subdivisions in FIXED_SUBDIVISIONS
        }
        for n_cells in initial
    }
    if args.preflight:
        source_audit = _source_restriction_audit(
            initial[32]["context"],
            initial[64]["context"],
        )
        expected_hashes = evidence["initialization"]["meshes"]
        initial_hashes_passed = all(
            initial[n_cells]["vector_sha256"]
            == expected_hashes[str(n_cells)]["state_vector_sha256"]
            for n_cells in RESOLUTIONS
        )
        print(
            json.dumps(
                {
                    "work_package": "WP10c7k",
                    "preflight_passed": bool(
                        source_audit["passed"]
                        and initial_hashes_passed
                        and all(
                            initial[n]["state_gates"]["passed"]
                            for n in RESOLUTIONS
                        )
                    ),
                    "wp10c7j_evidence_sha256": evidence_sha256,
                    "source_restriction_audit": source_audit,
                    "initial_hashes_passed": initial_hashes_passed,
                    "throughput_ratios": {
                        str(n): initial[n]["throughput_ratio"]
                        for n in RESOLUTIONS
                    },
                },
                sort_keys=True,
            )
        )
        return

    adaptive = {}
    if not args.aggregate_only:
        for n_cells in selected_resolutions:
            adaptive[n_cells] = _run_or_load_mesh(
                initial[n_cells],
                evidence_sha256,
                force=args.force,
            )
        if args.defer_aggregation:
            print(
                json.dumps(
                    {
                        "work_package": "WP10c7k",
                        "meshes_completed": selected_resolutions,
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
            fixed[n_cells] = {
                subdivisions: _load_fixed_reference(
                    evidence,
                    initial[n_cells],
                    subdivisions,
                )
                for subdivisions in FIXED_SUBDIVISIONS
            }
        if n_cells not in adaptive:
            adaptive[n_cells] = _run_or_load_mesh(
                initial[n_cells],
                evidence_sha256,
                force=False,
            )
    payload = _aggregate(
        output_path,
        arrays_path,
        evidence,
        evidence_sha256,
        initial,
        fixed,
        adaptive,
    )
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "maximum_conservative_log_h_over_r_error": payload[
                    "primary_log_h_over_r_contract"
                ]["maximum_conservative_spatial_error"],
                "output": _relative(output_path),
                "arrays": _relative(arrays_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
