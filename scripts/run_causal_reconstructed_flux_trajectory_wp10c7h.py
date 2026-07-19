"""Run the bounded N32/N64 reconstructed-flux trajectory certification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    CausalFiveFieldAdaptiveStepConfig,
    CausalFiveFieldBDFRestart,
    audit_causal_five_field_state_gates,
    causal_five_field_bdf_physical_ledger_relative_defects,
    causal_five_field_bdf_restarts_equal,
    causal_five_field_consistent_tangent_decomposition,
    causal_five_field_profile_fields,
    causal_five_field_reconstruct_face_charts,
    causal_five_field_regression_seed_parameters,
    causal_five_field_state_summary,
    causal_restrict_cell_averages,
    causal_restrict_cell_integrals,
    evolve_causal_five_field_fixed_bdf2,
    load_causal_five_field_bdf_restart,
    make_causal_five_field_regression_context,
    make_causal_five_field_seed,
    pack_causal_five_field_state,
    save_causal_five_field_bdf_restart,
    unpack_causal_five_field_state,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "3b835e269137b3df79e13efc85baa01532a624c5"
WP10C7G_OUTPUT = (
    ROOT / "outputs/tables/causal_spatial_reconstruction_wp10c7g.json"
)
OUTPUT_CHECKPOINT_DIRECTORY = (
    ROOT / "outputs/checkpoints/causal_five_field_wp10c7h"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/tables/causal_reconstructed_flux_trajectory_wp10c7h.json"
)
DEFAULT_ARRAYS = (
    ROOT
    / "outputs/tables/"
    "causal_reconstructed_flux_trajectory_wp10c7h_arrays.npz"
)
SPATIAL_RECONSTRUCTION = "plm_smooth"
RESOLUTIONS = (32, 64)
SUBDIVISIONS = (32, 64)
TARGET_DURATION_SECONDS = 1.537457597966907e-2
SPATIAL_RESPONSE_GATE = 5.0e-3
MAXIMUM_TEMPORAL_LOG_H_UNCERTAINTY = 5.0e-4
PREFERRED_TEMPORAL_LOG_H_UNCERTAINTY = 2.5e-4
MAXIMUM_PHYSICAL_LEDGER_RELATIVE_DEFECT = 1.0e-3
SOURCE_RESTRICTION_TOLERANCE = 5.0e-13
DIAGNOSED_INNER_RADIUS_RG = 15.0
DIAGNOSED_OUTER_RADIUS_RG = 60.0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
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


def _fixed_path(n_cells: int, subdivisions: int) -> Path:
    return (
        OUTPUT_CHECKPOINT_DIRECTORY
        / (
            f"causal_wp10c7h_N{n_cells:03d}_"
            f"plm_bdf2_S{subdivisions:04d}.npz"
        )
    )


def _progress(n_cells: int, subdivisions: int):
    interval = max(1, subdivisions // 8)

    def progress(completed, total, _state, _history) -> None:
        if completed % interval == 0 or completed == total:
            print(
                json.dumps(
                    {
                        "mode": (
                            f"n{n_cells}_plm_fixed_bdf2_"
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
    context = make_causal_five_field_regression_context(
        n_cells,
        spatial_reconstruction=SPATIAL_RECONSTRUCTION,
    )
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
        raise RuntimeError(f"N{n_cells} reconstructed initial state failed")
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


def _make_restart(
    initial: dict,
    result,
    subdivisions: int,
) -> CausalFiveFieldBDFRestart:
    if result.history is None:
        raise RuntimeError("WP10c7h fixed trajectory lacks BDF history")
    return CausalFiveFieldBDFRestart(
        state_vector=np.asarray(result.state_vector, dtype=float),
        history=result.history,
        elapsed_time=TARGET_DURATION_SECONDS,
        dt_next=result.timestep_seconds,
        next_order=2,
        accepted_steps=result.completed_steps,
        rejected_attempts=0,
        provenance={
            "work_package": "WP10c7h",
            "role": "bounded_reconstructed_flux_fixed_bdf2",
            "base_commit": BASE_COMMIT,
            "n_cells": int(initial["context"].grid.centers.size),
            "subdivisions": subdivisions,
            "target_duration_seconds": TARGET_DURATION_SECONDS,
            "spatial_reconstruction": SPATIAL_RECONSTRUCTION,
            "initial_state_sha256": initial["vector_sha256"],
            "temporal_method": (
                "one BDF1 startup step then fixed equal-step BDF2"
            ),
            "result_summary": _fixed_summary(result),
        },
    )


def _load_fixed(
    initial: dict,
    subdivisions: int,
) -> dict:
    context = initial["context"]
    n_cells = int(context.grid.centers.size)
    path = _fixed_path(n_cells, subdivisions)
    restart = load_causal_five_field_bdf_restart(path, context)
    provenance = restart.provenance
    summary = provenance.get("result_summary")
    timestep = TARGET_DURATION_SECONDS / subdivisions
    if not (
        provenance.get("work_package") == "WP10c7h"
        and provenance.get("role")
        == "bounded_reconstructed_flux_fixed_bdf2"
        and provenance.get("base_commit") == BASE_COMMIT
        and provenance.get("n_cells") == n_cells
        and provenance.get("subdivisions") == subdivisions
        and provenance.get("target_duration_seconds")
        == TARGET_DURATION_SECONDS
        and provenance.get("spatial_reconstruction")
        == SPATIAL_RECONSTRUCTION
        and provenance.get("initial_state_sha256")
        == initial["vector_sha256"]
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
            f"WP10c7h N{n_cells} S{subdivisions} provenance failed"
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


def _run_or_load_fixed(
    initial: dict,
    subdivisions: int,
    *,
    force: bool,
) -> dict:
    context = initial["context"]
    n_cells = int(context.grid.centers.size)
    path = _fixed_path(n_cells, subdivisions)
    if path.exists() and not force:
        return _load_fixed(initial, subdivisions)
    timestep = TARGET_DURATION_SECONDS / subdivisions
    predictor = initial["physical_tangent"] * timestep
    result = evolve_causal_five_field_fixed_bdf2(
        context,
        initial["vector"],
        predictor,
        timestep,
        TARGET_DURATION_SECONDS,
        subdivisions,
        _step_config(),
        progress=_progress(n_cells, subdivisions),
    )
    summary = _fixed_summary(result)
    if not result.passed:
        return {
            "restart": None,
            "summary": summary,
            "checkpoint": None,
        }
    restart = _make_restart(initial, result, subdivisions)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_causal_five_field_bdf_restart(path, context, restart)
    restored = load_causal_five_field_bdf_restart(path, context)
    if not causal_five_field_bdf_restarts_equal(restart, restored):
        raise RuntimeError("WP10c7h fixed restart is not bitwise")
    loaded = _load_fixed(initial, subdivisions)
    loaded["checkpoint"]["reused"] = False
    return loaded


def _stream_matrix(context) -> np.ndarray:
    source = context.stream_sources
    if source is None:
        raise RuntimeError("WP10c7h requires the exact stream source")
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
        "maximum_difference_radius": float(selected_radius[peak]),
        "excluded_boundary_cells_per_side": 0,
    }


def _temporal_comparison(
    initial: dict,
    coarse_fixed: dict,
    fine_fixed: dict,
) -> tuple[dict, dict]:
    coarse_response = _profile_response(
        initial,
        coarse_fixed["restart"].state_vector,
    )
    fine_response = _profile_response(
        initial,
        fine_fixed["restart"].state_vector,
    )
    rows = {}
    arrays = {}
    for name in coarse_response:
        difference = coarse_response[name] - fine_response[name]
        rows[name] = {
            "full_domain": _selected_metrics(
                initial["context"],
                difference,
                diagnosed_band=False,
            ),
            "diagnosed_interior_band": _selected_metrics(
                initial["context"],
                difference,
                diagnosed_band=True,
            ),
        }
        arrays[f"temporal_n{initial['state'].n_cells}_{name}"] = (
            difference
        )
    return rows, arrays


def _spatial_comparison(
    coarse_initial: dict,
    fine_initial: dict,
    coarse_fixed: dict,
    fine_fixed: dict,
) -> tuple[dict, dict]:
    coarse_response = _profile_response(
        coarse_initial,
        coarse_fixed["restart"].state_vector,
    )
    fine_response = _profile_response(
        fine_initial,
        fine_fixed["restart"].state_vector,
    )
    rows = {}
    arrays = {}
    for name in coarse_response:
        restricted = causal_restrict_cell_averages(
            coarse_initial["context"].grid,
            fine_initial["context"].grid,
            fine_response[name],
        )
        difference = coarse_response[name] - restricted
        rows[name] = {
            "full_domain": _selected_metrics(
                coarse_initial["context"],
                difference,
                diagnosed_band=False,
            ),
            "diagnosed_interior_band": _selected_metrics(
                coarse_initial["context"],
                difference,
                diagnosed_band=True,
            ),
        }
        arrays[f"spatial_n32_{name}"] = coarse_response[name]
        arrays[f"spatial_restricted_n64_{name}"] = restricted
        arrays[f"spatial_difference_{name}"] = difference
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


def main() -> None:
    args = _arguments()
    output_path = _absolute(args.output)
    arrays_path = _absolute(args.arrays)
    if not WP10C7G_OUTPUT.exists():
        raise RuntimeError("WP10c7h requires the WP10c7g evidence")
    wp10c7g = json.loads(WP10C7G_OUTPUT.read_text(encoding="utf-8"))
    if not (
        wp10c7g.get("work_package") == "WP10c7g"
        and wp10c7g.get("wp10c7h_authorized", False)
        and wp10c7g.get("decision")
        == "wp10c7h_reconstructed_flux_trajectory_authorized"
    ):
        raise RuntimeError("WP10c7g did not authorize WP10c7h")

    parameter_context = make_causal_five_field_regression_context(
        32,
        spatial_reconstruction=SPATIAL_RECONSTRUCTION,
    )
    seed_parameters = causal_five_field_regression_seed_parameters(
        parameter_context
    )
    initial = {
        n_cells: _initial_bundle(n_cells, seed_parameters)
        for n_cells in RESOLUTIONS
    }
    source_audit = _source_restriction_audit(
        initial[32]["context"],
        initial[64]["context"],
    )
    fixed = {}
    for n_cells in RESOLUTIONS:
        fixed[n_cells] = {}
        for subdivisions in SUBDIVISIONS:
            fixed[n_cells][subdivisions] = _run_or_load_fixed(
                initial[n_cells],
                subdivisions,
                force=args.force,
            )
            if fixed[n_cells][subdivisions]["restart"] is None:
                raise RuntimeError(
                    f"WP10c7h N{n_cells} S{subdivisions} failed"
                )

    temporal = {}
    temporal_arrays = {}
    for n_cells in RESOLUTIONS:
        rows, arrays = _temporal_comparison(
            initial[n_cells],
            fixed[n_cells][32],
            fixed[n_cells][64],
        )
        temporal[str(n_cells)] = rows
        temporal_arrays.update(arrays)
    spatial, spatial_arrays = _spatial_comparison(
        initial[32],
        initial[64],
        fixed[32][64],
        fixed[64][64],
    )
    temporal_uncertainties = {
        str(n_cells): temporal[str(n_cells)]["log_h_over_r"][
            "full_domain"
        ]["maximum_absolute_difference"]
        for n_cells in RESOLUTIONS
    }
    temporal_confound = max(temporal_uncertainties.values())
    spatial_log_h = spatial["log_h_over_r"]["full_domain"][
        "maximum_absolute_difference"
    ]
    combined_log_h = spatial_log_h + temporal_confound
    temporal_passed = bool(
        all(
            value <= MAXIMUM_TEMPORAL_LOG_H_UNCERTAINTY
            for value in temporal_uncertainties.values()
        )
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
    primary_spatial_passed = bool(
        combined_log_h <= SPATIAL_RESPONSE_GATE
    )
    passed = bool(
        source_audit["passed"]
        and all_fixed_passed
        and temporal_passed
        and ledgers_passed
        and primary_spatial_passed
    )
    decision = (
        "reconstructed_n32_n64_bounded_trajectory_certified"
        if passed
        else "reconstructed_n32_n64_bounded_trajectory_not_certified"
    )

    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arrays_path,
        **temporal_arrays,
        **spatial_arrays,
    )
    payload = {
        "work_package": "WP10c7h",
        "base_commit": BASE_COMMIT,
        "scope": (
            "single bounded reconstructed-flux N32/N64 trajectory; "
            "same no-tide/no-wind physics and duration as WP10c7f"
        ),
        "wp10c7g_evidence": {
            "path": _relative(WP10C7G_OUTPUT),
            "sha256": _sha256(WP10C7G_OUTPUT),
            "decision": wp10c7g["decision"],
        },
        "spatial_reconstruction": SPATIAL_RECONSTRUCTION,
        "initialization": {
            "policy": (
                "same analytic source-compatible continuum sampled on "
                "each mesh; reconstructed face fluxes rebuilt exactly; "
                "DAE-consistent tangent predictor; one BDF1 startup "
                "step creates history independently in each campaign"
            ),
            "n32": {
                "state_vector_sha256": initial[32]["vector_sha256"],
                "state_gates": initial[32]["state_gates"],
                "state_summary": initial[32]["state_summary"],
                "minimum_admissibility_factor": initial[32][
                    "minimum_admissibility_factor"
                ],
                "admissibility_limited_cell_count": initial[32][
                    "admissibility_limited_cell_count"
                ],
                "tangent_defects": initial[32]["tangent_defects"],
            },
            "n64": {
                "state_vector_sha256": initial[64]["vector_sha256"],
                "state_gates": initial[64]["state_gates"],
                "state_summary": initial[64]["state_summary"],
                "minimum_admissibility_factor": initial[64][
                    "minimum_admissibility_factor"
                ],
                "admissibility_limited_cell_count": initial[64][
                    "admissibility_limited_cell_count"
                ],
                "tangent_defects": initial[64]["tangent_defects"],
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
                }
                for subdivisions in SUBDIVISIONS
            }
            for n_cells in RESOLUTIONS
        },
        "temporal_response_comparison": temporal,
        "spatial_s64_response_comparison": spatial,
        "primary_log_h_over_r_contract": {
            "n32_n64_s64_spatial_difference": spatial_log_h,
            "n32_s32_s64_temporal_uncertainty": (
                temporal_uncertainties["32"]
            ),
            "n64_s32_s64_temporal_uncertainty": (
                temporal_uncertainties["64"]
            ),
            "maximum_temporal_confound": temporal_confound,
            "spatial_plus_temporal_confound": combined_log_h,
            "gate": SPATIAL_RESPONSE_GATE,
            "maximum_temporal_uncertainty": (
                MAXIMUM_TEMPORAL_LOG_H_UNCERTAINTY
            ),
            "preferred_temporal_uncertainty": (
                PREFERRED_TEMPORAL_LOG_H_UNCERTAINTY
            ),
            "passed": primary_spatial_passed and temporal_passed,
        },
        "gates": {
            "source_restriction_passed": source_audit["passed"],
            "all_fixed_campaigns_passed": all_fixed_passed,
            "temporal_uncertainty_passed": temporal_passed,
            "physical_ledgers_passed": ledgers_passed,
            "maximum_physical_ledger_relative_defect": ledger_maximum,
            "primary_spatial_contract_passed": primary_spatial_passed,
            "wp10c7h_passed": passed,
        },
        "decision": decision,
        "artifacts": {
            "arrays_path": _relative(arrays_path),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload["artifacts"]["arrays_sha256"] = _sha256(arrays_path)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "combined_log_h_over_r_error": combined_log_h,
                "output": _relative(output_path),
                "arrays": _relative(arrays_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
