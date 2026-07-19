"""Localize the matched N16/N32 causal spatial-response discrepancy."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    CAUSAL_FIVE_FIELD_NAMES,
    CausalFiveFieldAdaptiveStepConfig,
    audit_causal_five_field_state_gates,
    causal_coincident_fine_faces,
    causal_five_field_consistent_tangent_decomposition,
    causal_five_field_profile_fields,
    causal_five_field_residual_terms,
    causal_nested_refinement_ratio,
    causal_restrict_cell_averages,
    causal_restrict_cell_integrals,
    causal_spatial_difference_metrics,
    evaluate_causal_five_field_dae,
    evaluate_causal_five_field_increment_bdf,
    evolve_causal_five_field_fixed_bdf2,
    load_causal_five_field_adaptive_bdf2_restart,
    load_causal_five_field_adaptive_restart,
    load_causal_five_field_bdf_restart,
    make_causal_five_field_regression_context,
    unpack_causal_five_field_state,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "1de2df363330f81a383a0ac1356c48b8df4c966a"
TARGET_DURATION_SECONDS = 0.01537457597966907
FIXED_SUBDIVISIONS = 64
SNAPSHOT_STEPS = (0, 1, 2, 4, 8, 16, 32, 64)
REPLAY_PREFIX_STEPS = 32
SPATIAL_RESPONSE_GATE = 5.0e-3
TEMPORAL_CONFOUND_GATE = 1.0e-3
N64_MAXIMUM_TEMPORAL_UNCERTAINTY = 5.0e-4
PREFERRED_N64_TEMPORAL_UNCERTAINTY = 2.5e-4
SOURCE_RESTRICTION_TOLERANCE = 5.0e-13
TANGENT_CONSISTENCY_TOLERANCE = 1.0e-9
TANGENT_RECONSTRUCTION_TOLERANCE = 1.0e-8
RESIDUAL_RECONSTRUCTION_TOLERANCE = 1.0e-12

INITIAL_PATHS = {
    16: (
        ROOT
        / "outputs/checkpoints/causal_five_field_wp10c5k"
        / "causal_wp10c5q_N016_final.npz"
    ),
    32: (
        ROOT
        / "outputs/checkpoints/causal_five_field_wp10c5k"
        / "causal_wp10c5q_N032_final.npz"
    ),
}
FIXED_PATHS = {
    16: (
        ROOT
        / "outputs/checkpoints/causal_five_field_wp10c7b"
        / "causal_wp10c7b_N016_bdf2_S0064.npz"
    ),
    32: (
        ROOT
        / "outputs/checkpoints/causal_five_field_wp10c7d"
        / "causal_wp10c7d_N032_bdf2_S0064.npz"
    ),
}
ADAPTIVE_PATHS = {
    16: (
        ROOT
        / "outputs/checkpoints/causal_five_field_wp10c7c"
        / "causal_wp10c7c_N016_final.npz"
    ),
    32: (
        ROOT
        / "outputs/checkpoints/causal_five_field_wp10c7d"
        / "causal_wp10c7d_N032_final.npz"
    ),
}
PRIOR_SPATIAL_AUDIT = (
    ROOT
    / "outputs/tables"
    / "causal_five_field_mesh_common_spatial_response_wp10c5r.json"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/tables/causal_spatial_response_wp10c7e.json"
)
DEFAULT_ARRAYS = (
    ROOT / "outputs/tables/causal_spatial_response_wp10c7e_arrays.npz"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arrays", type=Path, default=DEFAULT_ARRAYS)
    return parser.parse_args()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


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


def _load_inputs() -> dict:
    contexts = {
        n_cells: make_causal_five_field_regression_context(n_cells)
        for n_cells in (16, 32)
    }
    initial = {
        n_cells: load_causal_five_field_adaptive_restart(
            INITIAL_PATHS[n_cells],
            contexts[n_cells],
        )
        for n_cells in (16, 32)
    }
    fixed = {
        n_cells: load_causal_five_field_bdf_restart(
            FIXED_PATHS[n_cells],
            contexts[n_cells],
        )
        for n_cells in (16, 32)
    }
    adaptive = {
        n_cells: load_causal_five_field_adaptive_bdf2_restart(
            ADAPTIVE_PATHS[n_cells],
            contexts[n_cells],
        )
        for n_cells in (16, 32)
    }
    expected = {
        "initial": {
            16: ("WP10c5q", "bounded_billionth_loading_time_duration"),
            32: ("WP10c5q", "bounded_billionth_loading_time_duration"),
        },
        "fixed": {
            16: ("WP10c7b", "fixed_bdf2_reference"),
            32: ("WP10c7d", "n32_fixed_bdf2_temporal_reference"),
        },
        "adaptive": {
            16: ("WP10c7c", "adaptive_bdf2_campaign"),
            32: ("WP10c7d", "matched_n32_adaptive_bdf2"),
        },
    }
    records = {}
    for kind, values in (
        ("initial", initial),
        ("fixed", fixed),
        ("adaptive", adaptive),
    ):
        records[kind] = {}
        paths = {
            "initial": INITIAL_PATHS,
            "fixed": FIXED_PATHS,
            "adaptive": ADAPTIVE_PATHS,
        }[kind]
        for n_cells, restart in values.items():
            work_package, role = expected[kind][n_cells]
            provenance_passed = bool(
                restart.provenance.get("work_package") == work_package
                and restart.provenance.get("role") == role
                and restart.provenance.get("n_cells") == n_cells
            )
            records[kind][str(n_cells)] = {
                "path": _relative(paths[n_cells]),
                "sha256": _sha256(paths[n_cells]),
                "work_package": restart.provenance.get("work_package"),
                "role": restart.provenance.get("role"),
                "elapsed_time_seconds": float(restart.elapsed_time),
                "provenance_passed": provenance_passed,
            }
    initial_time = initial[16].elapsed_time
    final_time = fixed[16].elapsed_time
    common_times = bool(
        initial[32].elapsed_time == initial_time
        and fixed[32].elapsed_time == final_time
        and adaptive[16].elapsed_time == final_time
        and adaptive[32].elapsed_time == final_time
        and np.isclose(
            final_time - initial_time,
            TARGET_DURATION_SECONDS,
            rtol=2.0e-15,
            atol=0.0,
        )
    )
    if not all(
        record["provenance_passed"]
        for kind in records.values()
        for record in kind.values()
    ):
        raise RuntimeError("WP10c7e checkpoint provenance failed")
    if not common_times:
        raise RuntimeError("WP10c7e checkpoints do not share exact times")
    return {
        "contexts": contexts,
        "initial": initial,
        "fixed": fixed,
        "adaptive": adaptive,
        "records": records,
        "exact_common_times": common_times,
    }


def _stream_matrix(context) -> np.ndarray:
    source = context.stream_sources
    if source is None:
        raise RuntimeError("WP10c7e requires the exact stream source")
    values = np.column_stack(
        (
            source.rest_mass,
            source.radial_momentum_over_c,
            source.angular_momentum_over_c,
            source.killing_energy_over_c2,
            np.zeros_like(source.rest_mass),
        )
    )
    return np.asarray(values, dtype=float)


def _source_audit(contexts: dict) -> dict:
    coarse = _stream_matrix(contexts[16])
    fine = _stream_matrix(contexts[32])
    restricted = causal_restrict_cell_integrals(
        contexts[16].grid,
        contexts[32].grid,
        fine,
    )
    scale = np.maximum(
        np.maximum(np.abs(coarse), np.abs(restricted)),
        1.0,
    )
    relative = np.abs(coarse - restricted) / scale
    maximum = float(np.max(relative))
    return {
        "method": "exact sum of nested fine-cell source integrals",
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


def _metric_pair(
    context,
    left: np.ndarray,
    right: np.ndarray,
) -> dict:
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
    contexts: dict,
    initial: dict,
    final: dict,
    *,
    label: str,
    arrays: dict,
) -> dict:
    coarse = _profile_response(
        contexts[16],
        initial[16].state_vector,
        final[16].state_vector,
    )
    fine = _profile_response(
        contexts[32],
        initial[32].state_vector,
        final[32].state_vector,
    )
    profile_rows = {}
    for name in coarse:
        restricted = causal_restrict_cell_averages(
            contexts[16].grid,
            contexts[32].grid,
            fine[name],
        )
        profile_rows[name] = _metric_pair(
            contexts[16],
            coarse[name],
            restricted,
        )
        arrays[f"{label}_n16_{name}"] = coarse[name]
        arrays[f"{label}_restricted_n32_{name}"] = restricted

    coarse_initial = unpack_causal_five_field_state(
        initial[16].state_vector,
        16,
    )
    coarse_final = unpack_causal_five_field_state(
        final[16].state_vector,
        16,
    )
    fine_initial = unpack_causal_five_field_state(
        initial[32].state_vector,
        32,
    )
    fine_final = unpack_causal_five_field_state(
        final[32].state_vector,
        32,
    )
    coarse_conserved = coarse_final.conserved - coarse_initial.conserved
    fine_conserved = fine_final.conserved - fine_initial.conserved
    restricted_conserved = causal_restrict_cell_averages(
        contexts[16].grid,
        contexts[32].grid,
        fine_conserved,
    )
    conserved_rows = {
        name: _metric_pair(
            contexts[16],
            coarse_conserved[:, index],
            restricted_conserved[:, index],
        )
        for index, name in enumerate(CAUSAL_FIVE_FIELD_NAMES)
    }
    arrays[f"{label}_n16_conserved_response"] = coarse_conserved
    arrays[
        f"{label}_restricted_n32_conserved_response"
    ] = restricted_conserved
    return {
        "method": (
            "exact nested Kerr-Schild measure restriction onto N16 "
            "control volumes"
        ),
        "profile_response": profile_rows,
        "conserved_response": conserved_rows,
    }


def _reconstruct_log_radius(context, values, sample_log_radius):
    centers = np.log(np.asarray(context.grid.centers, dtype=float))
    profile = np.asarray(values, dtype=float)
    sample = np.asarray(sample_log_radius, dtype=float)
    reconstructed = np.interp(sample, centers, profile)
    left = sample < centers[0]
    right = sample > centers[-1]
    reconstructed[left] = profile[0] + (
        (profile[1] - profile[0]) / (centers[1] - centers[0])
    ) * (sample[left] - centers[0])
    reconstructed[right] = profile[-1] + (
        (profile[-1] - profile[-2])
        / (centers[-1] - centers[-2])
    ) * (sample[right] - centers[-1])
    return reconstructed


def _interpolated_h_response(
    contexts: dict,
    initial: dict,
    final: dict,
    *,
    label: str,
    arrays: dict,
) -> dict:
    sample = np.linspace(
        np.log(float(contexts[32].grid.edges[0])),
        np.log(float(contexts[32].grid.edges[-1])),
        129,
    )
    responses = {}
    for n_cells in (16, 32):
        before = causal_five_field_profile_fields(
            contexts[n_cells],
            initial[n_cells].state_vector,
        )["log_h_over_r"]
        after = causal_five_field_profile_fields(
            contexts[n_cells],
            final[n_cells].state_vector,
        )["log_h_over_r"]
        responses[n_cells] = _reconstruct_log_radius(
            contexts[n_cells],
            after - before,
            sample,
        )
    difference = responses[16] - responses[32]
    index = int(np.argmax(np.abs(difference)))
    arrays[f"{label}_interpolation_radius_rg"] = (
        np.exp(sample) / contexts[32].grid.gravitational_radius
    )
    arrays[f"{label}_interpolation_n16_log_h_response"] = responses[16]
    arrays[f"{label}_interpolation_n32_log_h_response"] = responses[32]
    return {
        "method": (
            "WP10c7d log-linear cell-center reconstruction with "
            "one-cell edge extrapolation on 129 shared radii"
        ),
        "maximum_absolute_difference": float(
            np.abs(difference[index])
        ),
        "rms_difference": float(np.sqrt(np.mean(difference**2))),
        "maximum_difference_radius_rg": float(
            np.exp(sample[index])
            / contexts[32].grid.gravitational_radius
        ),
    }


def _face_response_comparison(
    contexts: dict,
    initial: dict,
    final: dict,
    *,
    label: str,
    arrays: dict,
) -> dict:
    split_names = (
        "numerical_weighted_face_fluxes_over_c",
        "central_weighted_face_fluxes_over_c",
        "rusanov_dissipation_weighted_face_fluxes_over_c",
    )
    responses = {}
    for n_cells in (16, 32):
        before = evaluate_causal_five_field_dae(
            initial[n_cells].state_vector,
            contexts[n_cells],
        )
        after = evaluate_causal_five_field_dae(
            final[n_cells].state_vector,
            contexts[n_cells],
        )
        responses[n_cells] = {
            name: (
                np.asarray(getattr(after, name), dtype=float)
                - np.asarray(getattr(before, name), dtype=float)
            )
            for name in split_names
        }
    rows = {}
    coarse_radius = (
        contexts[16].grid.edges
        / contexts[16].grid.gravitational_radius
    )
    for split in split_names:
        fine = causal_coincident_fine_faces(
            contexts[16].grid,
            contexts[32].grid,
            responses[32][split],
        )
        coarse = responses[16][split]
        difference = coarse - fine
        rows[split] = {}
        for index, field in enumerate(CAUSAL_FIVE_FIELD_NAMES):
            absolute = np.abs(difference[:, index])
            peak = int(np.argmax(absolute))
            rows[split][field] = {
                "maximum_absolute_difference": float(absolute[peak]),
                "rms_difference": float(
                    np.sqrt(np.mean(difference[:, index] ** 2))
                ),
                "maximum_difference_radius_rg": float(
                    coarse_radius[peak]
                ),
            }
        arrays[f"{label}_n16_{split}"] = coarse
        arrays[f"{label}_coincident_n32_{split}"] = fine
    return {
        "method": "exact coincident native N16/N32 faces",
        "splits": rows,
    }


def _within_mesh_history_audit(
    contexts: dict,
    initial: dict,
    fixed: dict,
    adaptive: dict,
) -> dict:
    result = {}
    for n_cells in (16, 32):
        fixed_response = _profile_response(
            contexts[n_cells],
            initial[n_cells].state_vector,
            fixed[n_cells].state_vector,
        )
        adaptive_response = _profile_response(
            contexts[n_cells],
            initial[n_cells].state_vector,
            adaptive[n_cells].state_vector,
        )
        result[str(n_cells)] = {
            name: _metric_pair(
                contexts[n_cells],
                fixed_response[name],
                adaptive_response[name],
            )
            for name in fixed_response
        }
    return result


def _profile_snapshot(context, vector) -> dict:
    return causal_five_field_profile_fields(context, vector)


def _replay_fixed(
    context,
    initial,
    retained,
    *,
    n_cells: int,
) -> tuple[dict, dict]:
    snapshots = {
        0: {
            "state_vector": np.array(initial.state_vector, copy=True),
            "profiles": _profile_snapshot(context, initial.state_vector),
            "terms": None,
            "state_gates": audit_causal_five_field_state_gates(
                context,
                initial.state_vector,
            ),
        }
    }
    previous_history = None

    def progress(completed, total, state, history) -> None:
        nonlocal previous_history
        increment = np.asarray(
            history.previous_physical_increment,
            dtype=float,
        )
        old = np.asarray(state, dtype=float) - increment
        order = 1 if completed == 1 else 2
        evaluation = evaluate_causal_five_field_increment_bdf(
            increment,
            context,
            old_vector=old,
            timestep_seconds=(
                TARGET_DURATION_SECONDS / FIXED_SUBDIVISIONS
            ),
            order=order,
            history=previous_history if order == 2 else None,
        )
        if completed in SNAPSHOT_STEPS:
            snapshots[completed] = {
                "state_vector": np.array(state, copy=True),
                "profiles": _profile_snapshot(context, state),
                "terms": causal_five_field_residual_terms(
                    context,
                    state,
                    evaluation,
                ),
                "state_gates": audit_causal_five_field_state_gates(
                    context,
                    state,
                ),
            }
            print(
                json.dumps(
                    {
                        "mode": f"replay_n{n_cells}",
                        "completed_steps": completed,
                        "total_steps": total,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        previous_history = history

    result = evolve_causal_five_field_fixed_bdf2(
        context,
        initial.state_vector,
        initial.previous_physical_increment,
        initial.previous_dt,
        (
            REPLAY_PREFIX_STEPS
            * TARGET_DURATION_SECONDS
            / FIXED_SUBDIVISIONS
        ),
        REPLAY_PREFIX_STEPS,
        _step_config(),
        progress=progress,
    )
    snapshots[FIXED_SUBDIVISIONS] = {
        "state_vector": np.array(retained.state_vector, copy=True),
        "profiles": _profile_snapshot(context, retained.state_vector),
        "terms": None,
        "state_gates": audit_causal_five_field_state_gates(
            context,
            retained.state_vector,
        ),
    }
    prefix_passed = bool(
        result.passed
        and result.completed_steps == REPLAY_PREFIX_STEPS
        and all(step in snapshots for step in SNAPSHOT_STEPS)
    )
    return snapshots, {
        "passed": bool(result.passed),
        "completed_steps": int(result.completed_steps),
        "scheduled_prefix_steps": REPLAY_PREFIX_STEPS,
        "prefix_replay_passed": prefix_passed,
        "retained_s64_endpoint_used_for_final_snapshot": True,
        "function_evaluations": int(result.function_evaluations),
        "jacobian_evaluations": int(result.jacobian_evaluations),
    }


def _replay_worker(n_cells: int) -> tuple[dict, dict]:
    context = make_causal_five_field_regression_context(n_cells)
    initial = load_causal_five_field_adaptive_restart(
        INITIAL_PATHS[n_cells],
        context,
    )
    retained = load_causal_five_field_bdf_restart(
        FIXED_PATHS[n_cells],
        context,
    )
    return _replay_fixed(
        context,
        initial,
        retained,
        n_cells=n_cells,
    )


def _scaled_term_matrix_difference(
    coarse: np.ndarray,
    fine: np.ndarray,
    radius_rg: np.ndarray,
) -> dict:
    field_scale = np.maximum(
        np.maximum(
            np.max(np.abs(coarse), axis=0),
            np.max(np.abs(fine), axis=0),
        ),
        1.0,
    )
    scaled = np.abs(coarse - fine) / field_scale[None, :]
    flat = int(np.argmax(scaled))
    cell, field = np.unravel_index(flat, scaled.shape)
    return {
        "maximum_field_scaled_difference": float(scaled[cell, field]),
        "controlling_field": CAUSAL_FIVE_FIELD_NAMES[field],
        "controlling_radius_rg": float(radius_rg[cell]),
        "field_scaled_rms_difference": float(
            np.sqrt(np.mean(scaled**2))
        ),
    }


def _snapshot_audit(
    contexts: dict,
    coarse_snapshots: dict,
    fine_snapshots: dict,
    arrays: dict,
) -> dict:
    rows = {}
    radius_rg = (
        contexts[16].grid.centers
        / contexts[16].grid.gravitational_radius
    )
    measures = contexts[16].grid.cell_measures
    initial_coarse = coarse_snapshots[0]["profiles"]
    initial_fine = fine_snapshots[0]["profiles"]
    for step in SNAPSHOT_STEPS:
        coarse = coarse_snapshots[step]
        fine = fine_snapshots[step]
        profile_rows = {}
        for name in coarse["profiles"]:
            coarse_response = (
                coarse["profiles"][name] - initial_coarse[name]
            )
            fine_response = fine["profiles"][name] - initial_fine[name]
            restricted = causal_restrict_cell_averages(
                contexts[16].grid,
                contexts[32].grid,
                fine_response,
            )
            profile_rows[name] = causal_spatial_difference_metrics(
                coarse_response,
                restricted,
                measures,
                radius_rg,
            )
            if name in (
                "log_h_over_r",
                "log_temperature",
                "log_surface_density",
            ):
                arrays[f"snapshot_{step:02d}_n16_{name}"] = (
                    coarse_response
                )
                arrays[f"snapshot_{step:02d}_restricted_n32_{name}"] = (
                    restricted
                )
        term_rows = {}
        if coarse["terms"] is not None and fine["terms"] is not None:
            for name, coarse_term in coarse["terms"].items():
                restricted_integral = causal_restrict_cell_integrals(
                    contexts[16].grid,
                    contexts[32].grid,
                    fine["terms"][name],
                )
                coarse_density = (
                    np.asarray(coarse_term, dtype=float)
                    / measures[:, None]
                )
                fine_density = restricted_integral / measures[:, None]
                term_rows[name] = _scaled_term_matrix_difference(
                    coarse_density,
                    fine_density,
                    radius_rg,
                )
        rows[str(step)] = {
            "elapsed_extension_seconds": float(
                step * TARGET_DURATION_SECONDS / FIXED_SUBDIVISIONS
            ),
            "profile_response": profile_rows,
            "bdf_residual_term_difference": term_rows,
            "n16_state_gates": coarse["state_gates"],
            "n32_state_gates": fine["state_gates"],
        }
    h_rows = {
        step: rows[str(step)]["profile_response"]["log_h_over_r"][
            "maximum_absolute_difference"
        ]
        for step in SNAPSHOT_STEPS
    }
    positive_steps = [step for step in SNAPSHOT_STEPS if step > 0]
    earliest_gate = next(
        (
            step
            for step in positive_steps
            if h_rows[step] > SPATIAL_RESPONSE_GATE
        ),
        None,
    )
    normalized_early = [
        h_rows[step] / step
        for step in positive_steps[:4]
        if h_rows[step] > 0.0
    ]
    early_linear_spread = (
        float(max(normalized_early) / min(normalized_early))
        if normalized_early
        else None
    )
    return {
        "method": (
            "exact fixed-S64 schedule replay through T/2 plus retained "
            "checksummed S64 endpoints at T; fine profiles restricted "
            "with Kerr-Schild measures"
        ),
        "snapshot_rows": rows,
        "first_nonzero_step_has_discrepancy": bool(h_rows[1] > 0.0),
        "earliest_step_exceeding_spatial_gate": earliest_gate,
        "early_difference_over_step_spread": early_linear_spread,
    }


def _combined_tangent_components(decomposition: dict) -> dict:
    components = {
        name: np.asarray(
            values["log_h_over_r_tangent_per_s"],
            dtype=float,
        )
        for name, values in decomposition["components"].items()
    }
    components["total_face_transport"] = (
        components["central_face_transport"]
        + components["rusanov_face_transport"]
        + components["flux_primary_closure"]
    )
    return components


def _tangent_worker(n_cells: int, vector: np.ndarray) -> dict:
    context = make_causal_five_field_regression_context(n_cells)
    return causal_five_field_consistent_tangent_decomposition(
        context,
        vector,
    )


def _tangent_pair(
    contexts: dict,
    coarse_vector: np.ndarray,
    fine_vector: np.ndarray,
    *,
    label: str,
    arrays: dict,
) -> dict:
    for n_cells in (16, 32):
        print(
            json.dumps(
                {
                    "mode": f"{label}_tangent_n{n_cells}",
                    "status": "started",
                },
                sort_keys=True,
            ),
            flush=True,
        )

    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {
            n_cells: executor.submit(_tangent_worker, n_cells, vector)
            for n_cells, vector in (
                (16, coarse_vector),
                (32, fine_vector),
            )
        }
        decompositions = {
            n_cells: future.result()
            for n_cells, future in futures.items()
        }
    coarse = decompositions[16]
    fine = decompositions[32]
    full_restricted = causal_restrict_cell_averages(
        contexts[16].grid,
        contexts[32].grid,
        fine["full"]["log_h_over_r_tangent_per_s"],
    )
    full_metrics = _metric_pair(
        contexts[16],
        coarse["full"]["log_h_over_r_tangent_per_s"],
        full_restricted,
    )
    arrays[f"{label}_tangent_n16_full_log_h"] = coarse["full"][
        "log_h_over_r_tangent_per_s"
    ]
    arrays[f"{label}_tangent_restricted_n32_full_log_h"] = (
        full_restricted
    )
    coarse_components = _combined_tangent_components(coarse)
    fine_components = _combined_tangent_components(fine)
    component_rows = {}
    for name in coarse_components:
        restricted = causal_restrict_cell_averages(
            contexts[16].grid,
            contexts[32].grid,
            fine_components[name],
        )
        metrics = _metric_pair(
            contexts[16],
            coarse_components[name],
            restricted,
        )
        component_rows[name] = metrics
        arrays[f"{label}_tangent_n16_{name}"] = coarse_components[name]
        arrays[f"{label}_tangent_restricted_n32_{name}"] = restricted
    ranking = sorted(
        (
            {
                "term": name,
                **metrics["full_domain"],
            }
            for name, metrics in component_rows.items()
            if name
            not in (
                "central_face_transport",
                "rusanov_face_transport",
                "flux_primary_closure",
            )
        ),
        key=lambda row: row["maximum_absolute_difference"],
        reverse=True,
    )
    defects_passed = bool(
        all(
            decomposition["maximum_scaled_consistency_defect"]
            <= TANGENT_CONSISTENCY_TOLERANCE
            and decomposition[
                "maximum_tangent_reconstruction_relative_defect"
            ]
            <= TANGENT_RECONSTRUCTION_TOLERANCE
            and decomposition[
                "maximum_residual_reconstruction_relative_defect"
            ]
            <= RESIDUAL_RECONSTRUCTION_TOLERANCE
            for decomposition in decompositions.values()
        )
    )
    return {
        "method": (
            "DAE-consistent descriptor tangent with separate central, "
            "Rusanov, flux-closure, and physical-source forcings"
        ),
        "full_log_h_over_r_tangent": full_metrics,
        "component_log_h_over_r_tangent": component_rows,
        "component_difference_ranking": ranking,
        "n16_defects": {
            name: value
            for name, value in coarse.items()
            if name.startswith("maximum_")
        },
        "n32_defects": {
            name: value
            for name, value in fine.items()
            if name.startswith("maximum_")
        },
        "n16_outer_boundary_choked": coarse["outer_boundary_choked"],
        "n32_outer_boundary_choked": fine["outer_boundary_choked"],
        "n16_outer_incoming_characteristics": (
            coarse["outer_incoming_characteristics"]
        ),
        "n32_outer_incoming_characteristics": (
            fine["outer_incoming_characteristics"]
        ),
        "defects_passed": defects_passed,
    }


def _prior_spatial_evidence() -> dict:
    source = json.loads(PRIOR_SPATIAL_AUDIT.read_text(encoding="utf-8"))
    manufactured = source["manufactured_transport_convergence"]
    return {
        "path": _relative(PRIOR_SPATIAL_AUDIT),
        "sha256": _sha256(PRIOR_SPATIAL_AUDIT),
        "decision": source["decision"],
        "manufactured_spatial_response_passed": bool(
            source["gates"]["manufactured_spatial_response_passed"]
        ),
        "minimum_central_observed_order": float(
            manufactured["gates"]["minimum_central_observed_order"]
        ),
        "minimum_rusanov_observed_order": float(
            manufactured["gates"][
                "minimum_rusanov_dissipation_observed_order"
            ]
        ),
        "minimum_total_observed_order": float(
            manufactured["gates"]["minimum_total_observed_order"]
        ),
        "prior_tangent_controlling_term": source["mesh_comparison"][
            "component_cell_average_difference_ranking"
        ][0]["term"],
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = _arguments()
    output_path = _absolute(args.output)
    arrays_path = _absolute(args.arrays)
    inputs = _load_inputs()
    contexts = inputs["contexts"]
    initial = inputs["initial"]
    fixed = inputs["fixed"]
    adaptive = inputs["adaptive"]
    arrays = {
        "n16_radius_rg": (
            contexts[16].grid.centers
            / contexts[16].grid.gravitational_radius
        ),
        "n32_radius_rg": (
            contexts[32].grid.centers
            / contexts[32].grid.gravitational_radius
        ),
        "n16_edges_rg": (
            contexts[16].grid.edges
            / contexts[16].grid.gravitational_radius
        ),
        "n32_edges_rg": (
            contexts[32].grid.edges
            / contexts[32].grid.gravitational_radius
        ),
    }
    nesting_ratio = causal_nested_refinement_ratio(
        contexts[16].grid,
        contexts[32].grid,
    )
    source_audit = _source_audit(contexts)
    fixed_restriction = _restricted_response_comparison(
        contexts,
        initial,
        fixed,
        label="fixed",
        arrays=arrays,
    )
    adaptive_restriction = _restricted_response_comparison(
        contexts,
        initial,
        adaptive,
        label="adaptive",
        arrays=arrays,
    )
    interpolation = {
        "fixed": _interpolated_h_response(
            contexts,
            initial,
            fixed,
            label="fixed",
            arrays=arrays,
        ),
        "adaptive": _interpolated_h_response(
            contexts,
            initial,
            adaptive,
            label="adaptive",
            arrays=arrays,
        ),
    }
    face_response = {
        "fixed": _face_response_comparison(
            contexts,
            initial,
            fixed,
            label="fixed",
            arrays=arrays,
        ),
        "adaptive": _face_response_comparison(
            contexts,
            initial,
            adaptive,
            label="adaptive",
            arrays=arrays,
        ),
    }
    history_audit = _within_mesh_history_audit(
        contexts,
        initial,
        fixed,
        adaptive,
    )
    with ProcessPoolExecutor(max_workers=2) as executor:
        replay_futures = {
            n_cells: executor.submit(_replay_worker, n_cells)
            for n_cells in (16, 32)
        }
        coarse_snapshots, coarse_replay = replay_futures[16].result()
        fine_snapshots, fine_replay = replay_futures[32].result()
    snapshots = _snapshot_audit(
        contexts,
        coarse_snapshots,
        fine_snapshots,
        arrays,
    )
    tangents = {
        "initial_checkpoint": _tangent_pair(
            contexts,
            initial[16].state_vector,
            initial[32].state_vector,
            label="initial",
            arrays=arrays,
        ),
        "fixed_s64_endpoint": _tangent_pair(
            contexts,
            fixed[16].state_vector,
            fixed[32].state_vector,
            label="fixed_final",
            arrays=arrays,
        ),
    }
    prior = _prior_spatial_evidence()
    first_step_rusanov = snapshots["snapshot_rows"]["1"][
        "bdf_residual_term_difference"
    ]["rusanov_face_transport"]

    fixed_h = fixed_restriction["profile_response"]["log_h_over_r"][
        "full_domain"
    ]["maximum_absolute_difference"]
    adaptive_h = adaptive_restriction["profile_response"]["log_h_over_r"][
        "full_domain"
    ]["maximum_absolute_difference"]
    fixed_interpolated_h = interpolation["fixed"][
        "maximum_absolute_difference"
    ]
    adaptive_interpolated_h = interpolation["adaptive"][
        "maximum_absolute_difference"
    ]
    temporal_confound = max(
        history_audit[str(n_cells)]["log_h_over_r"]["full_domain"][
            "maximum_absolute_difference"
        ]
        for n_cells in (16, 32)
    )
    replay_passed = bool(
        coarse_replay["prefix_replay_passed"]
        and fine_replay["prefix_replay_passed"]
    )
    mapping_passed = bool(
        fixed_h > SPATIAL_RESPONSE_GATE
        and adaptive_h > SPATIAL_RESPONSE_GATE
        and abs(fixed_h - fixed_interpolated_h) <= SPATIAL_RESPONSE_GATE
        and abs(adaptive_h - adaptive_interpolated_h)
        <= SPATIAL_RESPONSE_GATE
    )
    temporal_confound_passed = bool(
        temporal_confound <= TEMPORAL_CONFOUND_GATE
    )
    tangent_passed = all(
        audit["defects_passed"] for audit in tangents.values()
    )
    initial_ranking = tangents["initial_checkpoint"][
        "component_difference_ranking"
    ]
    final_ranking = tangents["fixed_s64_endpoint"][
        "component_difference_ranking"
    ]
    transport_controls_current_tangent = bool(
        initial_ranking[0]["term"] == "total_face_transport"
        or final_ranking[0]["term"] == "total_face_transport"
    )
    initial_components = tangents["initial_checkpoint"][
        "component_log_h_over_r_tangent"
    ]
    initial_rusanov_exceeds_central = bool(
        initial_components["rusanov_face_transport"]["full_domain"][
            "maximum_absolute_difference"
        ]
        >= initial_components["central_face_transport"]["full_domain"][
            "maximum_absolute_difference"
        ]
    )
    inherited_transport_evidence = bool(
        prior["manufactured_spatial_response_passed"]
        and prior["decision"]
        == "ordinary_first_order_rusanov_truncation_quantified"
        and prior["prior_tangent_controlling_term"] == "face_transport"
    )
    robust_spatial_discrepancy = bool(
        nesting_ratio == 2
        and source_audit["passed"]
        and mapping_passed
        and temporal_confound_passed
        and replay_passed
        and tangent_passed
    )
    mechanism_identified = bool(
        robust_spatial_discrepancy
        and transport_controls_current_tangent
        and initial_rusanov_exceeds_central
        and inherited_transport_evidence
    )
    n64_authorized = mechanism_identified
    if mechanism_identified:
        decision = (
            "robust_grid_dependent_response_traced_to_inherited_"
            "first_order_rusanov_transport_truncation"
        )
    elif robust_spatial_discrepancy:
        decision = "robust_spatial_discrepancy_mechanism_unresolved"
    else:
        decision = "spatial_discrepancy_not_yet_comparison_independent"

    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **arrays)
    output = {
        "work_package": "WP10c7e",
        "scope": (
            "localized N16/N32 spatial-response audit at the locked "
            "WP10c7d duration; no operator change and no N64 evolution"
        ),
        "base_commit": BASE_COMMIT,
        "decision": decision,
        "construction": {
            "resolutions": [16, 32],
            "fixed_subdivisions": FIXED_SUBDIVISIONS,
            "replayed_prefix_steps": REPLAY_PREFIX_STEPS,
            "snapshot_steps": list(SNAPSHOT_STEPS),
            "target_duration_seconds": TARGET_DURATION_SECONDS,
            "spatial_response_gate": SPATIAL_RESPONSE_GATE,
            "temporal_confound_gate": TEMPORAL_CONFOUND_GATE,
            "operator_modified": False,
            "n64_evolution_executed": False,
            "duration_extended": False,
            "physics_added": False,
        },
        "checkpoint_provenance": inputs["records"],
        "exact_common_times": inputs["exact_common_times"],
        "grid_and_source_contract": {
            "exact_refinement_ratio": nesting_ratio,
            "coarse_faces_equal_fine_faces_stride_two": bool(
                np.array_equal(
                    contexts[16].grid.edges,
                    contexts[32].grid.edges[::2],
                )
            ),
            "source_restriction": source_audit,
        },
        "comparison_independence": {
            "fixed_exact_measure_restriction": fixed_restriction,
            "adaptive_exact_measure_restriction": adaptive_restriction,
            "wp10c7d_interpolation_regression": interpolation,
            "native_coincident_face_response": face_response,
            "within_mesh_fixed_adaptive_response": history_audit,
            "maximum_within_mesh_log_h_temporal_confound": (
                temporal_confound
            ),
            "mapping_passed": mapping_passed,
            "temporal_confound_passed": temporal_confound_passed,
        },
        "fixed_replay": {
            "n16": coarse_replay,
            "n32": fine_replay,
            "passed": replay_passed,
        },
        "time_localization": snapshots,
        "characteristic_family_qualification": {
            "numerical_dissipation": (
                "scalar local Lax-Friedrichs/Rusanov maximum-speed "
                "envelope applied to the full conserved jump"
            ),
            "unique_acoustic_contact_shear_split_defined": False,
            "reason": (
                "the implemented Rusanov flux is not assembled as a "
                "sum of characteristic-family fluctuations"
            ),
            "first_step_controlling_conservation_channel": (
                first_step_rusanov["controlling_field"]
            ),
            "first_step_rusanov_field_scaled_difference": (
                first_step_rusanov["maximum_field_scaled_difference"]
            ),
            "first_step_controlling_radius_rg": (
                first_step_rusanov["controlling_radius_rg"]
            ),
        },
        "dae_consistent_tangent_decomposition": tangents,
        "inherited_common_state_operator_evidence": prior,
        "classification": {
            "robust_spatial_discrepancy": robust_spatial_discrepancy,
            "transport_controls_current_tangent": (
                transport_controls_current_tangent
            ),
            "initial_rusanov_difference_exceeds_central": (
                initial_rusanov_exceeds_central
            ),
            "inherited_first_order_rusanov_evidence": (
                inherited_transport_evidence
            ),
            "specific_mechanism_identified": mechanism_identified,
        },
        "gates": {
            "checkpoint_provenance_passed": True,
            "exact_nesting_passed": nesting_ratio == 2,
            "source_restriction_passed": source_audit["passed"],
            "comparison_mapping_passed": mapping_passed,
            "fixed_adaptive_history_confound_excluded": (
                temporal_confound_passed
            ),
            "exact_fixed_schedule_prefix_replay_passed": replay_passed,
            "tangent_decomposition_passed": tangent_passed,
            "n64_diagnostic_authorized_for_next_wp": n64_authorized,
            "n64_diagnostic_executed": False,
            "operator_change_authorized": False,
            "longer_duration_authorized": False,
            "tide_authorized": False,
            "wind_authorized": False,
        },
        "locked_next_experiment": {
            "work_package": "WP10c7f",
            "description": (
                "one N64 fixed-BDF2 bounded contraction diagnostic at "
                "the identical physical horizon"
            ),
            "authorized": n64_authorized,
            "n64_fixed_subdivisions": [32, 64],
            "selected_comparison": "N32 fixed S64 versus N64 fixed S64",
            "maximum_n64_temporal_log_h_uncertainty": (
                N64_MAXIMUM_TEMPORAL_UNCERTAINTY
            ),
            "preferred_n64_temporal_log_h_uncertainty": (
                PREFERRED_N64_TEMPORAL_UNCERTAINTY
            ),
            "spatial_order_formula": (
                "log2(D_N16_N32 / D_N32_N64)"
            ),
            "stop_if_spatial_order_below": 0.75,
            "no_n128_without_post_n64_decision": True,
        },
        "evidence_arrays": {
            "path": _relative(arrays_path),
            "sha256": _sha256(arrays_path),
            "array_count": len(arrays),
        },
    }
    _write_json(output_path, output)
    print(
        json.dumps(
            {
                "decision": decision,
                "fixed_restricted_log_h_difference": fixed_h,
                "adaptive_restricted_log_h_difference": adaptive_h,
                "maximum_temporal_confound": temporal_confound,
                "initial_controlling_term": initial_ranking[0]["term"],
                "final_controlling_term": final_ranking[0]["term"],
                "n64_diagnostic_authorized": n64_authorized,
                "output": _relative(output_path),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
