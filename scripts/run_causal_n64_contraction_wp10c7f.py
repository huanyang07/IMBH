"""Run the single authorized N64 fixed-BDF2 contraction diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    CAUSAL_FIVE_FIELD_NAMES,
    CAUSAL_FIVE_FIELD_OBSERVABLE_SCHEMA_VERSION,
    CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1,
    CausalFiveFieldAdaptiveRestart,
    CausalFiveFieldAdaptiveStepConfig,
    CausalFiveFieldBDFRestart,
    audit_causal_five_field_state_gates,
    causal_coincident_fine_faces,
    causal_five_field_bdf_physical_ledger_relative_defects,
    causal_five_field_bdf_restarts_equal,
    causal_five_field_profile_fields,
    causal_five_field_temporal_error_ratio,
    causal_nested_refinement_ratio,
    causal_restrict_cell_averages,
    causal_restrict_cell_integrals,
    causal_spatial_contraction_order,
    causal_spatial_difference_metrics,
    compare_causal_five_field_endpoint_vectors,
    evaluate_causal_five_field_dae,
    evolve_causal_five_field_fixed_bdf2,
    load_causal_five_field_adaptive_restart,
    load_causal_five_field_bdf_restart,
    make_causal_five_field_regression_context,
    save_causal_five_field_bdf_restart,
    unpack_causal_five_field_state,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "c23c5d5acc8ef563e5831a08382e61ca20149c52"
N32_INITIAL_CHECKPOINT = (
    ROOT
    / "outputs/checkpoints/causal_five_field_wp10c5k"
    / "causal_wp10c5q_N032_final.npz"
)
N64_INITIAL_CHECKPOINT = (
    ROOT
    / "outputs/checkpoints/causal_five_field_wp10c5k"
    / "causal_wp10c5s_duration_N064_final.npz"
)
N32_FIXED_S64_CHECKPOINT = (
    ROOT
    / "outputs/checkpoints/causal_five_field_wp10c7d"
    / "causal_wp10c7d_N032_bdf2_S0064.npz"
)
WP10C7E_OUTPUT = (
    ROOT / "outputs/tables/causal_spatial_response_wp10c7e.json"
)
OUTPUT_CHECKPOINT_DIRECTORY = (
    ROOT / "outputs/checkpoints/causal_five_field_wp10c7f"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/tables/causal_n64_contraction_wp10c7f.json"
)
DEFAULT_ARRAYS = (
    ROOT / "outputs/tables/causal_n64_contraction_wp10c7f_arrays.npz"
)
FIXED_SUBDIVISIONS = (32, 64)
TARGET_DURATION_SECONDS = 1.537457597966907e-2
COOLING_INNER_CUTOFF_RG = 6.0
TARGET_TIME_RELATIVE_TOLERANCE = 5.0e-14
MAXIMUM_N64_TEMPORAL_LOG_H_UNCERTAINTY = 5.0e-4
PREFERRED_N64_TEMPORAL_LOG_H_UNCERTAINTY = 2.5e-4
MAXIMUM_PHYSICAL_LEDGER_RELATIVE_DEFECT = 1.0e-3
SOURCE_RESTRICTION_TOLERANCE = 5.0e-13
SPATIAL_RESPONSE_GATE = 5.0e-3
MINIMUM_SPATIAL_ORDER = 0.75
FIRST_ORDER_UPPER_BOUND = 1.25


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subdivisions",
        type=int,
        action="append",
        choices=FIXED_SUBDIVISIONS,
        default=None,
        help="Repeat to run a subset; default selects S32 and S64.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute selected N64 trajectories.",
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


def _fixed_path(subdivisions: int) -> Path:
    return (
        OUTPUT_CHECKPOINT_DIRECTORY
        / f"causal_wp10c7f_N064_bdf2_S{subdivisions:04d}.npz"
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
        "cumulative_physical_ledger": _ledger_json(
            result.cumulative_physical_ledger
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
        "passed": bool(result.passed),
        "message": str(result.message),
    }


def _load_initial(
    context,
) -> tuple[CausalFiveFieldAdaptiveRestart, str]:
    checksum = _sha256(N64_INITIAL_CHECKPOINT)
    restart = load_causal_five_field_adaptive_restart(
        N64_INITIAL_CHECKPOINT,
        context,
    )
    if not (
        restart.provenance.get("work_package") == "WP10c5s"
        and restart.provenance.get("role")
        == "bounded_billionth_loading_time_duration"
        and restart.provenance.get("n_cells") == 64
        and "exact circularized regression stream"
        in str(restart.provenance.get("source", ""))
        and audit_causal_five_field_state_gates(
            context,
            restart.state_vector,
        )["passed"]
    ):
        raise RuntimeError("N64 WP10c5s restart prerequisite failed")
    return restart, checksum


def _make_fixed_restart(
    initial: CausalFiveFieldAdaptiveRestart,
    initial_sha: str,
    result,
) -> CausalFiveFieldBDFRestart:
    if result.history is None:
        raise RuntimeError("fixed N64 BDF2 result lacks history")
    return CausalFiveFieldBDFRestart(
        state_vector=np.asarray(result.state_vector, dtype=float),
        history=result.history,
        elapsed_time=initial.elapsed_time + TARGET_DURATION_SECONDS,
        dt_next=result.timestep_seconds,
        next_order=2,
        accepted_steps=initial.accepted_steps + result.completed_steps,
        rejected_attempts=initial.rejected_attempts,
        provenance={
            "work_package": "WP10c7f",
            "role": "n64_fixed_bdf2_contraction_diagnostic",
            "base_commit": BASE_COMMIT,
            "n_cells": 64,
            "subdivisions": int(result.subdivisions),
            "target_duration_seconds": TARGET_DURATION_SECONDS,
            "source": "exact circularized regression stream",
            "initial_checkpoint_sha256": initial_sha,
            "observable_schema": (
                CAUSAL_FIVE_FIELD_OBSERVABLE_SCHEMA_VERSION
            ),
            "temporal_method": (
                "one BDF1 startup step then fixed equal-step BDF2"
            ),
            "result_summary": _fixed_summary(result),
        },
    )


def _load_fixed(
    context,
    initial: CausalFiveFieldAdaptiveRestart,
    initial_sha: str,
    subdivisions: int,
) -> dict:
    path = _fixed_path(subdivisions)
    restart = load_causal_five_field_bdf_restart(path, context)
    provenance = restart.provenance
    summary = provenance.get("result_summary")
    timestep = TARGET_DURATION_SECONDS / subdivisions
    target_time = initial.elapsed_time + TARGET_DURATION_SECONDS
    tolerance = max(
        1.0e-20,
        TARGET_TIME_RELATIVE_TOLERANCE * target_time,
    )
    if not (
        provenance.get("work_package") == "WP10c7f"
        and provenance.get("role")
        == "n64_fixed_bdf2_contraction_diagnostic"
        and provenance.get("base_commit") == BASE_COMMIT
        and provenance.get("n_cells") == 64
        and provenance.get("subdivisions") == subdivisions
        and provenance.get("initial_checkpoint_sha256") == initial_sha
        and provenance.get("observable_schema")
        == CAUSAL_FIVE_FIELD_OBSERVABLE_SCHEMA_VERSION
        and isinstance(summary, dict)
        and summary.get("passed", False)
        and abs(restart.elapsed_time - target_time) <= tolerance
        and restart.history.previous_timestep_seconds == timestep
        and restart.dt_next == timestep
        and audit_causal_five_field_state_gates(
            context,
            restart.state_vector,
        )["passed"]
    ):
        raise RuntimeError(
            f"N64 fixed S{subdivisions} provenance failed"
        )
    return {
        "restart": restart,
        "summary": summary,
        "checkpoint": {
            "path": _relative(path),
            "sha256": _sha256(path),
            "roundtrip_bitwise": True,
            "reused": True,
        },
    }


def _progress(subdivisions: int):
    interval = max(1, subdivisions // 8)

    def progress(completed, total, _state, _history) -> None:
        if completed % interval == 0 or completed == total:
            print(
                json.dumps(
                    {
                        "mode": f"n64_fixed_bdf2_s{subdivisions}",
                        "completed_steps": completed,
                        "total_steps": total,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    return progress


def _run_or_load_fixed(
    context,
    initial: CausalFiveFieldAdaptiveRestart,
    initial_sha: str,
    subdivisions: int,
    *,
    force: bool,
) -> dict:
    path = _fixed_path(subdivisions)
    if path.exists() and not force:
        return _load_fixed(
            context,
            initial,
            initial_sha,
            subdivisions,
        )
    result = evolve_causal_five_field_fixed_bdf2(
        context,
        initial.state_vector,
        initial.previous_physical_increment,
        initial.previous_dt,
        TARGET_DURATION_SECONDS,
        subdivisions,
        _step_config(),
        progress=_progress(subdivisions),
    )
    summary = _fixed_summary(result)
    if not result.passed:
        return {
            "restart": None,
            "summary": summary,
            "checkpoint": None,
        }
    restart = _make_fixed_restart(initial, initial_sha, result)
    save_causal_five_field_bdf_restart(path, context, restart)
    restored = load_causal_five_field_bdf_restart(path, context)
    if not causal_five_field_bdf_restarts_equal(restart, restored):
        raise RuntimeError("fixed N64 BDF2 restart is not bitwise")
    loaded = _load_fixed(
        context,
        initial,
        initial_sha,
        subdivisions,
    )
    loaded["checkpoint"]["reused"] = False
    return loaded


def _stream_matrix(context) -> np.ndarray:
    source = context.stream_sources
    if source is None:
        raise RuntimeError("WP10c7f requires the exact stream source")
    return np.column_stack(
        (
            source.rest_mass,
            source.radial_momentum_over_c,
            source.angular_momentum_over_c,
            source.killing_energy_over_c2,
            np.zeros_like(source.rest_mass),
        )
    )


def _source_restriction_audit(n32_context, n64_context) -> dict:
    coarse = _stream_matrix(n32_context)
    restricted = causal_restrict_cell_integrals(
        n32_context.grid,
        n64_context.grid,
        _stream_matrix(n64_context),
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


def _profile_response(context, initial_vector, final_vector) -> dict:
    initial = causal_five_field_profile_fields(context, initial_vector)
    final = causal_five_field_profile_fields(context, final_vector)
    return {
        name: np.asarray(final[name] - initial[name], dtype=float)
        for name in initial
    }


def _metric_pair(context, left, right) -> dict:
    radius = (
        np.asarray(context.grid.centers, dtype=float)
        / context.grid.gravitational_radius
    )
    measures = np.asarray(context.grid.cell_measures, dtype=float)
    return {
        "full_domain": causal_spatial_difference_metrics(
            left,
            right,
            measures,
            radius,
        ),
        "excluding_two_boundary_cells_per_side": (
            causal_spatial_difference_metrics(
                left,
                right,
                measures,
                radius,
                exclude_boundary_cells=2,
            )
        ),
    }


def _restricted_response_comparison(
    n32_context,
    n64_context,
    n32_initial,
    n64_initial,
    n32_final,
    n64_final,
    arrays: dict[str, np.ndarray],
) -> dict:
    coarse_profiles = _profile_response(
        n32_context,
        n32_initial,
        n32_final,
    )
    fine_profiles = _profile_response(
        n64_context,
        n64_initial,
        n64_final,
    )
    profile_rows = {}
    for name, coarse in coarse_profiles.items():
        restricted = causal_restrict_cell_averages(
            n32_context.grid,
            n64_context.grid,
            fine_profiles[name],
        )
        profile_rows[name] = _metric_pair(
            n32_context,
            coarse,
            restricted,
        )
        arrays[f"n32_{name}_response"] = coarse
        arrays[f"restricted_n64_{name}_response"] = restricted

    coarse_initial_state = unpack_causal_five_field_state(
        n32_initial,
        32,
    )
    coarse_final_state = unpack_causal_five_field_state(n32_final, 32)
    fine_initial_state = unpack_causal_five_field_state(n64_initial, 64)
    fine_final_state = unpack_causal_five_field_state(n64_final, 64)
    coarse_conserved = (
        coarse_final_state.conserved - coarse_initial_state.conserved
    )
    fine_conserved = (
        fine_final_state.conserved - fine_initial_state.conserved
    )
    restricted_conserved = causal_restrict_cell_averages(
        n32_context.grid,
        n64_context.grid,
        fine_conserved,
    )
    conserved_rows = {
        name: _metric_pair(
            n32_context,
            coarse_conserved[:, index],
            restricted_conserved[:, index],
        )
        for index, name in enumerate(CAUSAL_FIVE_FIELD_NAMES)
    }
    arrays["n32_conserved_response"] = coarse_conserved
    arrays["restricted_n64_conserved_response"] = restricted_conserved
    return {
        "method": (
            "exact nested Kerr-Schild measure restriction onto N32 "
            "control volumes"
        ),
        "profile_response": profile_rows,
        "conserved_response": conserved_rows,
    }


def _face_response_comparison(
    n32_context,
    n64_context,
    n32_initial,
    n64_initial,
    n32_final,
    n64_final,
    arrays: dict[str, np.ndarray],
) -> dict:
    evaluations = {
        "n32_initial": evaluate_causal_five_field_dae(
            n32_initial,
            n32_context,
        ),
        "n32_final": evaluate_causal_five_field_dae(
            n32_final,
            n32_context,
        ),
        "n64_initial": evaluate_causal_five_field_dae(
            n64_initial,
            n64_context,
        ),
        "n64_final": evaluate_causal_five_field_dae(
            n64_final,
            n64_context,
        ),
    }
    names = {
        "numerical": "numerical_weighted_face_fluxes_over_c",
        "central": "central_weighted_face_fluxes_over_c",
        "rusanov": "rusanov_dissipation_weighted_face_fluxes_over_c",
    }
    radius = (
        np.asarray(n32_context.grid.edges, dtype=float)
        / n32_context.grid.gravitational_radius
    )
    rows = {}
    for label, attribute in names.items():
        coarse = (
            np.asarray(
                getattr(evaluations["n32_final"], attribute),
                dtype=float,
            )
            - np.asarray(
                getattr(evaluations["n32_initial"], attribute),
                dtype=float,
            )
        )
        fine = (
            np.asarray(
                getattr(evaluations["n64_final"], attribute),
                dtype=float,
            )
            - np.asarray(
                getattr(evaluations["n64_initial"], attribute),
                dtype=float,
            )
        )
        restricted = causal_coincident_fine_faces(
            n32_context.grid,
            n64_context.grid,
            fine,
        )
        difference = coarse - restricted
        fields = {}
        for index, field in enumerate(CAUSAL_FIVE_FIELD_NAMES):
            absolute = np.abs(difference[:, index])
            peak = int(np.argmax(absolute))
            fields[field] = {
                "maximum_absolute_difference": float(absolute[peak]),
                "maximum_difference_radius_rg": float(radius[peak]),
            }
        rows[label] = {"fields": fields}
        arrays[f"n32_{label}_face_response"] = coarse
        arrays[f"coincident_n64_{label}_face_response"] = restricted
    return {
        "method": "exact coincident native faces without interpolation",
        "components": rows,
    }


def _load_parent_data(n32_context) -> dict:
    with WP10C7E_OUTPUT.open(encoding="utf-8") as source:
        wp10c7e = json.load(source)
    if not (
        wp10c7e.get("work_package") == "WP10c7e"
        and wp10c7e.get("gates", {}).get(
            "n64_diagnostic_authorized_for_next_wp"
        )
        and wp10c7e.get("locked_next_experiment", {}).get(
            "work_package"
        )
        == "WP10c7f"
    ):
        raise RuntimeError("WP10c7e did not authorize WP10c7f")
    n32_initial = load_causal_five_field_adaptive_restart(
        N32_INITIAL_CHECKPOINT,
        n32_context,
    )
    n32_fixed = load_causal_five_field_bdf_restart(
        N32_FIXED_S64_CHECKPOINT,
        n32_context,
    )
    if not (
        n32_initial.provenance.get("work_package") == "WP10c5q"
        and n32_fixed.provenance.get("work_package") == "WP10c7d"
        and n32_fixed.provenance.get("role")
        == "n32_fixed_bdf2_temporal_reference"
        and n32_fixed.provenance.get("subdivisions") == 64
    ):
        raise RuntimeError("N32 fixed-S64 parent provenance failed")
    coarse_difference = float(
        wp10c7e["comparison_independence"][
            "fixed_exact_measure_restriction"
        ]["profile_response"]["log_h_over_r"]["full_domain"][
            "maximum_absolute_difference"
        ]
    )
    return {
        "wp10c7e": wp10c7e,
        "n32_initial": n32_initial,
        "n32_fixed": n32_fixed,
        "coarse_difference": coarse_difference,
        "records": {
            "wp10c7e": {
                "path": _relative(WP10C7E_OUTPUT),
                "sha256": _sha256(WP10C7E_OUTPUT),
            },
            "n32_initial": {
                "path": _relative(N32_INITIAL_CHECKPOINT),
                "sha256": _sha256(N32_INITIAL_CHECKPOINT),
            },
            "n32_fixed_s64": {
                "path": _relative(N32_FIXED_S64_CHECKPOINT),
                "sha256": _sha256(N32_FIXED_S64_CHECKPOINT),
            },
        },
    }


def _temporal_audit(context, initial, runs: dict[int, dict]) -> dict:
    errors = compare_causal_five_field_endpoint_vectors(
        context,
        initial.state_vector,
        runs[32]["restart"].state_vector,
        runs[64]["restart"].state_vector,
        cooling_inner_cutoff=(
            COOLING_INNER_CUTOFF_RG
            * context.grid.gravitational_radius
        ),
    )
    gates = dict(CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1)
    gates["maximum_log_h_over_r_profile"] = (
        MAXIMUM_N64_TEMPORAL_LOG_H_UNCERTAINTY
    )
    gate_audit = causal_five_field_temporal_error_ratio(errors, gates)
    ledger_passed = all(
        run["summary"]["cumulative_physical_ledger"][
            "maximum_relative_defect"
        ]
        <= MAXIMUM_PHYSICAL_LEDGER_RELATIVE_DEFECT
        for run in runs.values()
    )
    preferred = bool(
        float(errors["maximum_log_h_over_r_profile"])
        <= PREFERRED_N64_TEMPORAL_LOG_H_UNCERTAINTY
    )
    return {
        "raw_s32_s64_errors": errors,
        "gates": gates,
        "gate_audit": gate_audit,
        "preferred_log_h_uncertainty": (
            PREFERRED_N64_TEMPORAL_LOG_H_UNCERTAINTY
        ),
        "preferred_log_h_uncertainty_passed": preferred,
        "maximum_physical_ledger_relative_defect": (
            MAXIMUM_PHYSICAL_LEDGER_RELATIVE_DEFECT
        ),
        "physical_ledgers_passed": ledger_passed,
        "passed": bool(gate_audit["passed"] and ledger_passed),
    }


def _spatial_decision(
    coarse_difference: float,
    fine_difference: float,
) -> dict:
    order = causal_spatial_contraction_order(
        coarse_difference,
        fine_difference,
    )
    gate_passed = bool(fine_difference <= SPATIAL_RESPONSE_GATE)
    minimum_order_passed = bool(order >= MINIMUM_SPATIAL_ORDER)
    first_order_band = bool(
        MINIMUM_SPATIAL_ORDER <= order <= FIRST_ORDER_UPPER_BOUND
    )
    contraction_factor = float(2.0**order)
    projected_n64_n128_difference = float(
        fine_difference / contraction_factor
    )
    required_n64_n128_order = float(
        np.log2(fine_difference / SPATIAL_RESPONSE_GATE)
    )
    n128_useful_for_direct_certification = bool(
        projected_n64_n128_difference <= SPATIAL_RESPONSE_GATE
    )
    if gate_passed:
        decision = "bounded_spatial_gate_certified_at_n64"
    elif not minimum_order_passed:
        decision = "stop_noncontracting_spatial_response"
    elif first_order_band or not n128_useful_for_direct_certification:
        decision = "stop_before_n128_design_spatial_upgrade"
    else:
        decision = "n128_candidate_requires_explicit_post_wp10c7f_review"
    return {
        "n16_n32_difference": float(coarse_difference),
        "n32_n64_difference": float(fine_difference),
        "observed_spatial_order": float(order),
        "measured_contraction_factor": contraction_factor,
        "minimum_spatial_order": MINIMUM_SPATIAL_ORDER,
        "minimum_spatial_order_passed": minimum_order_passed,
        "first_order_upper_bound": FIRST_ORDER_UPPER_BOUND,
        "in_first_order_band": first_order_band,
        "spatial_response_gate": SPATIAL_RESPONSE_GATE,
        "spatial_response_gate_passed": gate_passed,
        "projected_n64_n128_difference_at_measured_order": (
            projected_n64_n128_difference
        ),
        "projected_n64_n128_gate_fraction": (
            projected_n64_n128_difference / SPATIAL_RESPONSE_GATE
        ),
        "required_n64_n128_order_for_direct_gate": (
            required_n64_n128_order
        ),
        "n128_useful_for_direct_certification": (
            n128_useful_for_direct_certification
        ),
        "decision": decision,
        "n128_automatically_authorized": False,
    }


def _write_json(path: Path, output: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            output,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = _arguments()
    selected = (
        list(FIXED_SUBDIVISIONS)
        if args.subdivisions is None
        else list(dict.fromkeys(args.subdivisions))
    )
    output_path = _absolute(args.output)
    arrays_path = _absolute(args.arrays)
    n32_context = make_causal_five_field_regression_context(32)
    n64_context = make_causal_five_field_regression_context(64)
    refinement_ratio = causal_nested_refinement_ratio(
        n32_context.grid,
        n64_context.grid,
    )
    initial, initial_sha = _load_initial(n64_context)
    parent = _load_parent_data(n32_context)
    exact_common_initial_time = bool(
        parent["n32_initial"].elapsed_time == initial.elapsed_time
    )
    if not exact_common_initial_time:
        raise RuntimeError("N32 and N64 do not share the initial time")
    source_restriction = _source_restriction_audit(
        n32_context,
        n64_context,
    )
    if not source_restriction["passed"]:
        raise RuntimeError("N32/N64 exact stream restriction failed")

    runs = {
        subdivisions: _run_or_load_fixed(
            n64_context,
            initial,
            initial_sha,
            subdivisions,
            force=args.force,
        )
        for subdivisions in selected
    }
    selected_passed = all(
        run["summary"]["passed"] and run["restart"] is not None
        for run in runs.values()
    )
    complete_ladder = set(runs) == set(FIXED_SUBDIVISIONS)
    base_output = {
        "work_package": "WP10c7f",
        "scope": (
            "single authorized N64 fixed-BDF2 temporal and "
            "N32/N64 spatial contraction diagnostic"
        ),
        "base_commit": BASE_COMMIT,
        "construction": {
            "n_cells": 64,
            "fixed_subdivisions": list(FIXED_SUBDIVISIONS),
            "selected_subdivisions": selected,
            "target_duration_seconds": TARGET_DURATION_SECONDS,
            "temporal_method": (
                "one BDF1 startup step then fixed equal-step BDF2"
            ),
            "observable_schema": (
                CAUSAL_FIVE_FIELD_OBSERVABLE_SCHEMA_VERSION
            ),
            "no_operator_change": True,
            "no_adaptivity": True,
            "no_n128_or_duration_extension": True,
        },
        "parent_records": parent["records"],
        "initial_checkpoint": {
            "path": _relative(N64_INITIAL_CHECKPOINT),
            "sha256": initial_sha,
            "elapsed_time_seconds": float(initial.elapsed_time),
            "provenance": initial.provenance,
        },
        "grid_and_source_contract": {
            "refinement_ratio": refinement_ratio,
            "exact_common_initial_time": exact_common_initial_time,
            "source_restriction": source_restriction,
        },
        "fixed_runs": {
            str(subdivisions): {
                "summary": run["summary"],
                "checkpoint": run["checkpoint"],
            }
            for subdivisions, run in runs.items()
        },
    }
    if not complete_ladder or not selected_passed:
        output = {
            **base_output,
            "temporal_audit": None,
            "spatial_comparison": None,
            "spatial_contraction": None,
            "decision": (
                "partial_n64_ladder_complete_no_spatial_decision"
                if selected_passed
                else "stop_n64_fixed_trajectory_failed"
            ),
            "passed": False,
        }
        _write_json(output_path, output)
        print(
            json.dumps(
                {
                    "output": str(output_path),
                    "decision": output["decision"],
                    "passed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    temporal = _temporal_audit(n64_context, initial, runs)
    if not temporal["passed"]:
        output = {
            **base_output,
            "temporal_audit": temporal,
            "spatial_comparison": None,
            "spatial_contraction": None,
            "decision": "stop_n64_temporal_uncertainty_gate_failed",
            "passed": False,
        }
        _write_json(output_path, output)
        print(
            json.dumps(
                {
                    "output": str(output_path),
                    "decision": output["decision"],
                    "passed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    arrays: dict[str, np.ndarray] = {
        "n32_radius_rg": (
            np.asarray(n32_context.grid.centers, dtype=float)
            / n32_context.grid.gravitational_radius
        ),
        "n64_radius_rg": (
            np.asarray(n64_context.grid.centers, dtype=float)
            / n64_context.grid.gravitational_radius
        ),
    }
    spatial = _restricted_response_comparison(
        n32_context,
        n64_context,
        parent["n32_initial"].state_vector,
        initial.state_vector,
        parent["n32_fixed"].state_vector,
        runs[64]["restart"].state_vector,
        arrays,
    )
    face = _face_response_comparison(
        n32_context,
        n64_context,
        parent["n32_initial"].state_vector,
        initial.state_vector,
        parent["n32_fixed"].state_vector,
        runs[64]["restart"].state_vector,
        arrays,
    )
    fine_difference = float(
        spatial["profile_response"]["log_h_over_r"][
            "full_domain"
        ]["maximum_absolute_difference"]
    )
    contraction = _spatial_decision(
        parent["coarse_difference"],
        fine_difference,
    )
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **arrays)
    output = {
        **base_output,
        "temporal_audit": temporal,
        "spatial_comparison": {
            "exact_measure_restriction": spatial,
            "native_coincident_faces": face,
        },
        "spatial_contraction": contraction,
        "evidence_arrays": {
            "path": _relative(arrays_path),
            "sha256": _sha256(arrays_path),
        },
        "authorization": {
            "wp10c7f_diagnostic_complete": True,
            "bounded_n32_n64_spatial_gate_certified": contraction[
                "spatial_response_gate_passed"
            ],
            "n128_automatically_authorized": False,
            "operator_change_implemented": False,
            "longer_duration_authorized": False,
            "tide_authorized": False,
            "wind_authorized": False,
            "stability_hot_state_or_cycle_authorized": False,
            "next_work_package": (
                "WP10c7g operator-level second-order spatial "
                "reconstruction audit"
            ),
        },
        "decision": contraction["decision"],
        "passed": True,
    }
    _write_json(output_path, output)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "decision": output["decision"],
                "n64_temporal_log_h_uncertainty": temporal[
                    "raw_s32_s64_errors"
                ]["maximum_log_h_over_r_profile"],
                "n32_n64_difference": fine_difference,
                "observed_spatial_order": contraction[
                    "observed_spatial_order"
                ],
                "passed": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
