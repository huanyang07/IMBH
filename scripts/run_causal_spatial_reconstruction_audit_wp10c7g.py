"""Certify the optional PLM causal face reconstruction before evolution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    CAUSAL_FIVE_FIELD_NAMES,
    causal_five_field_colored_central_jacobian,
    causal_five_field_consistent_tangent_decomposition,
    causal_five_field_dae_jacobian_color_groups,
    causal_five_field_dae_jacobian_sparsity,
    causal_five_field_dae_scaling,
    causal_five_field_reconstruct_face_charts,
    causal_five_field_regression_seed_parameters,
    causal_restrict_cell_averages,
    evaluate_causal_five_field_dae,
    evaluate_causal_five_field_increment_backward_euler,
    make_causal_five_field_regression_context,
    make_causal_five_field_seed,
    pack_causal_five_field_state,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "3b835e269137b3df79e13efc85baa01532a624c5"
RESOLUTIONS = (16, 32, 64, 128)
MODES = ("piecewise_constant", "plm_unlimited", "plm_smooth")
MINIMUM_RECONSTRUCTED_ORDER = 1.8
MINIMUM_TANGENT_REDUCTION = 5.0
MAXIMUM_JACOBIAN_RELATIVE_DEFECT = 2.0e-8
MAXIMUM_JACOBIAN_STEP_SPREAD = 5.0e-3
MAXIMUM_ALGEBRAIC_MAP_RESIDUAL = 1.0e-12
MAXIMUM_RECONSTRUCTION_ORACLE_DEFECT = 1.0e-10
RANK_RELATIVE_THRESHOLD = 1.0e-11
DIAGNOSED_INNER_RADIUS_RG = 15.0
DIAGNOSED_OUTER_RADIUS_RG = 60.0
DEFAULT_OUTPUT = (
    ROOT / "outputs/tables/causal_spatial_reconstruction_wp10c7g.json"
)
DEFAULT_ARRAYS = (
    ROOT
    / "outputs/tables/causal_spatial_reconstruction_wp10c7g_arrays.npz"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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


def _source_compatible_seed_parameters() -> tuple[dict, dict]:
    context = make_causal_five_field_regression_context(16)
    parameters = causal_five_field_regression_seed_parameters(context)
    return parameters, {
        "profile": (
            "shared source-compatible C2 continuum profile in log radius"
        ),
        "inner_surface_density": parameters["inner_surface_density"],
        "inner_temperature_k": parameters["inner_temperature"],
        "target_inner_h_over_r": 0.1,
        "inner_plateau_radius_rg": 6.0,
        "outer_plateau_radius_rg": 240.0,
    }


def _reconstruction_oracles(context, charts: np.ndarray) -> dict:
    n_cells = int(context.grid.centers.size)
    middle = np.asarray(charts[n_cells // 2], dtype=float)
    constant = np.repeat(middle[None, :], n_cells, axis=0)
    constant_result = causal_five_field_reconstruct_face_charts(
        context,
        constant,
    )
    constant_exact = np.repeat(
        middle[None, :],
        n_cells + 1,
        axis=0,
    )
    constant_defect = max(
        float(
            np.max(
                np.abs(
                    constant_result.left_face_charts
                    - constant_exact
                )
            )
        ),
        float(
            np.max(
                np.abs(
                    constant_result.right_face_charts
                    - constant_exact
                )
            )
        ),
    )

    log_centers = np.log(context.grid.centers)
    log_edges = np.log(context.grid.edges)
    origin = float(np.mean(log_centers))
    span = float(log_centers[-2] - log_centers[1])
    slope = 0.1 * (charts[-2] - charts[1]) / span
    linear = middle + (log_centers - origin)[:, None] * slope
    linear_result = causal_five_field_reconstruct_face_charts(
        context,
        linear,
    )
    exact = middle + (log_edges - origin)[:, None] * slope
    interior = slice(2, -2)
    linear_defect = max(
        float(
            np.max(
                np.abs(
                    linear_result.left_face_charts[interior]
                    - exact[interior]
                )
            )
        ),
        float(
            np.max(
                np.abs(
                    linear_result.right_face_charts[interior]
                    - exact[interior]
                )
            )
        ),
    )
    return {
        "constant_left_right_maximum_absolute_defect": constant_defect,
        "linear_log_radius_maximum_absolute_defect": linear_defect,
    }


def _operator_artifact(
    mode: str,
    n_cells: int,
    seed_parameters: dict,
) -> tuple[dict, dict]:
    context = make_causal_five_field_regression_context(
        n_cells,
        spatial_reconstruction=mode,
    )
    state = make_causal_five_field_seed(context, **seed_parameters)
    vector = pack_causal_five_field_state(state)
    evaluation = evaluate_causal_five_field_dae(vector, context)
    reconstruction = causal_five_field_reconstruct_face_charts(
        context,
        state.primitives,
    )
    algebraic_maximum = max(
        float(np.max(np.abs(evaluation.primitive_map_rows))),
        float(np.max(np.abs(evaluation.interior_flux_rows))),
        float(np.max(np.abs(evaluation.inner_flux_rows))),
        float(np.max(np.abs(evaluation.outer_flux_rows))),
    )
    numerical = np.asarray(
        evaluation.numerical_weighted_face_fluxes_over_c,
        dtype=float,
    )
    central = np.asarray(
        evaluation.central_weighted_face_fluxes_over_c,
        dtype=float,
    )
    dissipation = np.asarray(
        evaluation.rusanov_dissipation_weighted_face_fluxes_over_c,
        dtype=float,
    )
    flux_scale = np.maximum(np.abs(numerical), 1.0)
    split_defect = float(
        np.max(np.abs(central + dissipation - numerical) / flux_scale)
    )
    oracles = _reconstruction_oracles(context, state.primitives)
    summary = {
        "mode": mode,
        "n_cells": n_cells,
        "maximum_algebraic_map_residual": algebraic_maximum,
        "maximum_relative_flux_split_defect": split_defect,
        "minimum_admissibility_factor": float(
            np.min(reconstruction.admissibility_factors)
        ),
        "admissibility_limited_cell_count": int(
            np.count_nonzero(
                reconstruction.admissibility_factors < 1.0 - 1.0e-12
            )
        ),
        **oracles,
    }
    return summary, {
        "grid_edges_rg": (
            context.grid.edges / context.grid.gravitational_radius
        ),
        "central": central,
        "rusanov": dissipation,
        "total": numerical,
        "admissibility_factors": reconstruction.admissibility_factors,
    }


def _norm_metrics(values: np.ndarray) -> dict:
    absolute = np.abs(np.asarray(values, dtype=float))
    return {
        "scaled_l1_error": float(np.mean(absolute)),
        "scaled_l2_error": float(np.sqrt(np.mean(absolute**2))),
        "scaled_linf_error": float(np.max(absolute)),
    }


def _operator_pair_error(coarse: dict, fine: dict) -> dict:
    coarse_edges = np.asarray(coarse["grid_edges_rg"], dtype=float)
    fine_edges = np.asarray(fine["grid_edges_rg"], dtype=float)
    n_coarse = coarse_edges.size - 1
    if not np.array_equal(coarse_edges, fine_edges[::2]):
        raise RuntimeError("WP10c7g operator grids are not exactly nested")
    field_scale = np.maximum(
        np.max(np.abs(fine["central"][1:-1]), axis=0),
        1.0,
    )
    face_indices = np.arange(2, n_coarse - 1)
    cell_indices = np.arange(2, n_coarse - 2)

    def component(name: str) -> dict:
        coarse_faces = np.asarray(coarse[name], dtype=float)
        fine_faces = np.asarray(fine[name], dtype=float)
        shared = (
            coarse_faces[face_indices]
            - fine_faces[2 * face_indices]
        ) / field_scale
        coarse_balance = coarse_faces[1:] - coarse_faces[:-1]
        fine_balance = fine_faces[1:] - fine_faces[:-1]
        restricted_fine = np.sum(
            fine_balance.reshape(n_coarse, 2, 5),
            axis=1,
        )
        balance = (
            coarse_balance[cell_indices]
            - restricted_fine[cell_indices]
        ) / field_scale
        return {
            "shared_face": _norm_metrics(shared),
            "cell_balance": _norm_metrics(balance),
            "shared_face_field_linf_errors": {
                name: float(value)
                for name, value in zip(
                    CAUSAL_FIVE_FIELD_NAMES,
                    np.max(np.abs(shared), axis=0),
                    strict=True,
                )
            },
        }

    return {
        "coarse_cells": n_coarse,
        "fine_cells": 2 * n_coarse,
        "central": component("central"),
        "rusanov": component("rusanov"),
        "total": component("total"),
    }


def _orders(pair_errors: list[dict]) -> dict:
    result = {}
    for component in ("central", "rusanov", "total"):
        result[component] = {}
        for location in ("shared_face", "cell_balance"):
            result[component][location] = {}
            for metric in ("scaled_l1_error", "scaled_l2_error"):
                errors = [
                    float(pair[component][location][metric])
                    for pair in pair_errors
                ]
                result[component][location][metric] = [
                    float(np.log2(first / second))
                    for first, second in zip(
                        errors[:-1],
                        errors[1:],
                        strict=True,
                    )
                ]
    return result


def _manufactured_audit(seed_parameters: dict) -> tuple[dict, dict]:
    summaries = {}
    artifacts = {}
    arrays = {}
    for mode in MODES:
        summaries[mode] = {}
        artifacts[mode] = {}
        for n_cells in RESOLUTIONS:
            summary, artifact = _operator_artifact(
                mode,
                n_cells,
                seed_parameters,
            )
            summaries[mode][str(n_cells)] = summary
            artifacts[mode][n_cells] = artifact
            for name in ("central", "rusanov", "total"):
                arrays[
                    f"operator_{mode}_n{n_cells}_{name}"
                ] = artifact[name]
        print(
            json.dumps(
                {
                    "mode": mode,
                    "stage": "manufactured_operator",
                    "status": "complete",
                },
                sort_keys=True,
            ),
            flush=True,
        )
    pair_errors = {
        mode: [
            _operator_pair_error(
                artifacts[mode][coarse],
                artifacts[mode][2 * coarse],
            )
            for coarse in RESOLUTIONS[:-1]
        ]
        for mode in MODES
    }
    observed_orders = {
        mode: _orders(pair_errors[mode])
        for mode in MODES
    }
    controlled_asymptotic_orders = [
        values[-1]
        for component in ("central", "rusanov", "total")
        for location in ("shared_face", "cell_balance")
        for values in observed_orders["plm_smooth"][component][
            location
        ].values()
    ]
    unlimited_asymptotic_orders = [
        values[-1]
        for component in ("central", "rusanov", "total")
        for location in ("shared_face", "cell_balance")
        for values in observed_orders["plm_unlimited"][component][
            location
        ].values()
    ]
    controlled_all_orders = [
        value
        for component in ("central", "rusanov", "total")
        for location in ("shared_face", "cell_balance")
        for values in observed_orders["plm_smooth"][component][
            location
        ].values()
        for value in values
    ]
    unlimited_all_orders = [
        value
        for component in ("central", "rusanov", "total")
        for location in ("shared_face", "cell_balance")
        for values in observed_orders["plm_unlimited"][component][
            location
        ].values()
        for value in values
    ]
    oracle_defect = max(
        max(
            summary[
                "constant_left_right_maximum_absolute_defect"
            ],
            summary[
                "linear_log_radius_maximum_absolute_defect"
            ],
        )
        for mode in ("plm_unlimited", "plm_smooth")
        for summary in summaries[mode].values()
    )
    algebraic_defect = max(
        summary["maximum_algebraic_map_residual"]
        for mode in MODES
        for summary in summaries[mode].values()
    )
    passed = bool(
        min(controlled_asymptotic_orders)
        >= MINIMUM_RECONSTRUCTED_ORDER
        and min(unlimited_asymptotic_orders)
        >= MINIMUM_RECONSTRUCTED_ORDER
        and oracle_defect <= MAXIMUM_RECONSTRUCTION_ORACLE_DEFECT
        and algebraic_defect <= MAXIMUM_ALGEBRAIC_MAP_RESIDUAL
    )
    return {
        "resolutions": list(RESOLUTIONS),
        "resolution_summaries": summaries,
        "nested_pair_errors": pair_errors,
        "observed_orders": observed_orders,
        "minimum_smooth_plm_asymptotic_observed_order": min(
            controlled_asymptotic_orders
        ),
        "minimum_unlimited_plm_asymptotic_observed_order": min(
            unlimited_asymptotic_orders
        ),
        "minimum_smooth_plm_all_pair_observed_order": min(
            controlled_all_orders
        ),
        "minimum_unlimited_plm_all_pair_observed_order": min(
            unlimited_all_orders
        ),
        "maximum_oracle_absolute_defect": oracle_defect,
        "maximum_algebraic_map_residual": algebraic_defect,
        "required_minimum_order": MINIMUM_RECONSTRUCTED_ORDER,
        "passed": passed,
    }, arrays


def _dense_central_jacobian(residual, values, step: float) -> np.ndarray:
    result = np.empty((values.size, values.size), dtype=float)
    for column in range(values.size):
        plus = np.array(values, copy=True)
        minus = np.array(values, copy=True)
        plus[column] += step
        minus[column] -= step
        result[:, column] = (
            residual(plus) - residual(minus)
        ) / (2.0 * step)
    return result


def _jacobian_audit(seed_parameters: dict) -> dict:
    n_cells = 8
    context = make_causal_five_field_regression_context(
        n_cells,
        spatial_reconstruction="plm_smooth",
    )
    state = make_causal_five_field_seed(context, **seed_parameters)
    old_vector = pack_causal_five_field_state(state)
    stationary = evaluate_causal_five_field_dae(old_vector, context)
    scaling = causal_five_field_dae_scaling(state, stationary)
    zero = np.zeros_like(old_vector)
    pattern = causal_five_field_dae_jacobian_sparsity(
        n_cells,
        spatial_reconstruction=context.spatial_reconstruction,
    )
    groups = causal_five_field_dae_jacobian_color_groups(pattern)

    def residual(scaled_increment: np.ndarray) -> np.ndarray:
        return (
            evaluate_causal_five_field_increment_backward_euler(
                scaling.column_scales * scaled_increment,
                context,
                old_vector=old_vector,
                timestep_seconds=2.0e-4,
            ).residual
            / scaling.row_scales
        )

    rng = np.random.default_rng(1729)
    direction = rng.normal(size=zero.size)
    direction /= np.linalg.norm(direction)
    steps = (1.0e-5, 2.0e-6, 5.0e-7)
    rows = {}
    colored_matrices = {}
    for step in steps:
        dense = _dense_central_jacobian(residual, zero, step)
        colored = causal_five_field_colored_central_jacobian(
            residual,
            zero,
            pattern,
            finite_difference_step=step,
        ).toarray()
        colored_matrices[step] = colored
        allowed = pattern.toarray().astype(bool)
        row_scale = np.maximum(np.max(np.abs(dense), axis=1), 1.0e-14)
        direct_directional = (
            residual(step * direction)
            - residual(-step * direction)
        ) / (2.0 * step)
        directional_scale = max(
            float(np.max(np.abs(direct_directional))),
            1.0e-14,
        )
        rows[f"{step:.1e}"] = {
            "maximum_omitted_relative_entry": float(
                np.max(
                    np.abs(np.where(allowed, 0.0, dense))
                    / row_scale[:, None]
                )
            ),
            "maximum_colored_dense_relative_entry": float(
                np.max(
                    np.abs(colored - dense) / row_scale[:, None]
                )
            ),
            "maximum_directional_relative_defect": float(
                np.max(
                    np.abs(colored @ direction - direct_directional)
                    / directional_scale
                )
            ),
        }
    reference = colored_matrices[2.0e-6]
    reference_row_scale = np.maximum(
        np.max(np.abs(reference), axis=1),
        1.0e-14,
    )
    step_spread = max(
        float(
            np.max(
                np.abs(matrix - reference)
                / reference_row_scale[:, None]
            )
        )
        for step, matrix in colored_matrices.items()
        if step != 2.0e-6
    )
    maximum_defect = max(
        max(
            row["maximum_omitted_relative_entry"],
            row["maximum_colored_dense_relative_entry"],
            row["maximum_directional_relative_defect"],
        )
        for row in rows.values()
    )
    passed = bool(
        maximum_defect <= MAXIMUM_JACOBIAN_RELATIVE_DEFECT
        and step_spread <= MAXIMUM_JACOBIAN_STEP_SPREAD
    )
    return {
        "n_cells": n_cells,
        "dimensions": list(pattern.shape),
        "pattern_nonzeros": int(pattern.nnz),
        "color_count": len(groups),
        "finite_difference_steps": list(steps),
        "step_rows": rows,
        "maximum_relative_parity_defect": maximum_defect,
        "maximum_cross_step_relative_spread": step_spread,
        "required_maximum_relative_parity_defect": (
            MAXIMUM_JACOBIAN_RELATIVE_DEFECT
        ),
        "required_maximum_cross_step_relative_spread": (
            MAXIMUM_JACOBIAN_STEP_SPREAD
        ),
        "passed": passed,
    }


def _tangent_worker(payload: tuple[str, int, dict]) -> dict:
    mode, n_cells, seed_parameters = payload
    context = make_causal_five_field_regression_context(
        n_cells,
        spatial_reconstruction=mode,
    )
    state = make_causal_five_field_seed(context, **seed_parameters)
    vector = pack_causal_five_field_state(state)
    decomposition = causal_five_field_consistent_tangent_decomposition(
        context,
        vector,
        rank_relative_threshold=(
            RANK_RELATIVE_THRESHOLD
            if mode == "plm_smooth" and n_cells in (16, 32)
            else None
        ),
    )
    components = decomposition["components"]
    total_transport = (
        components["central_face_transport"][
            "log_h_over_r_tangent_per_s"
        ]
        + components["rusanov_face_transport"][
            "log_h_over_r_tangent_per_s"
        ]
        + components["flux_primary_closure"][
            "log_h_over_r_tangent_per_s"
        ]
    )
    return {
        "mode": mode,
        "n_cells": n_cells,
        "radius_rg": decomposition["radius_rg"],
        "cell_measures": decomposition["cell_measures"],
        "full": decomposition["full"][
            "log_h_over_r_tangent_per_s"
        ],
        "central": components["central_face_transport"][
            "log_h_over_r_tangent_per_s"
        ],
        "rusanov": components["rusanov_face_transport"][
            "log_h_over_r_tangent_per_s"
        ],
        "total_transport": total_transport,
        "maximum_scaled_consistency_defect": decomposition[
            "maximum_scaled_consistency_defect"
        ],
        "maximum_residual_reconstruction_relative_defect": decomposition[
            "maximum_residual_reconstruction_relative_defect"
        ],
        "maximum_tangent_reconstruction_relative_defect": decomposition[
            "maximum_tangent_reconstruction_relative_defect"
        ],
        "consistency_dimensions": decomposition[
            "consistency_dimensions"
        ],
        "consistency_numerical_rank": decomposition[
            "consistency_numerical_rank"
        ],
        "consistency_condition_estimate": decomposition[
            "consistency_condition_estimate"
        ],
    }


def _tangent_pair(coarse: dict, fine: dict) -> dict:
    coarse_context = make_causal_five_field_regression_context(
        coarse["n_cells"],
        spatial_reconstruction=coarse["mode"],
    )
    fine_context = make_causal_five_field_regression_context(
        fine["n_cells"],
        spatial_reconstruction=fine["mode"],
    )
    measures = np.asarray(coarse["cell_measures"], dtype=float)

    radius = np.asarray(coarse["radius_rg"], dtype=float)
    diagnosed = (
        (radius >= DIAGNOSED_INNER_RADIUS_RG)
        & (radius <= DIAGNOSED_OUTER_RADIUS_RG)
    )

    def selected_metrics(
        difference: np.ndarray,
        selection: np.ndarray,
    ) -> dict:
        selected_difference = difference[selection]
        selected_measures = measures[selection]
        selected_radius = radius[selection]
        absolute = np.abs(selected_difference)
        peak = int(np.argmax(absolute))
        return {
            "maximum_absolute_difference_per_s": float(absolute[peak]),
            "measure_weighted_l2_difference_per_s": float(
                np.sqrt(
                    np.sum(
                        selected_measures * selected_difference**2
                    )
                    / np.sum(selected_measures)
                )
            ),
            "maximum_difference_radius_rg": float(
                selected_radius[peak]
            ),
        }

    def metrics(name: str) -> dict:
        restricted = causal_restrict_cell_averages(
            coarse_context.grid,
            fine_context.grid,
            fine[name],
        )
        difference = np.asarray(coarse[name]) - restricted
        return {
            "full_domain": selected_metrics(
                difference,
                np.ones(difference.size, dtype=bool),
            ),
            "diagnosed_interior_band": selected_metrics(
                difference,
                diagnosed,
            ),
        }

    return {
        "coarse_cells": coarse["n_cells"],
        "fine_cells": fine["n_cells"],
        **{
            name: metrics(name)
            for name in (
                "full",
                "central",
                "rusanov",
                "total_transport",
            )
        },
    }


def _tangent_audit(seed_parameters: dict) -> tuple[dict, dict]:
    payloads = [
        (mode, n_cells, seed_parameters)
        for mode in ("piecewise_constant", "plm_smooth")
        for n_cells in (16, 32, 64)
    ]
    results = []
    for payload in payloads:
        result = _tangent_worker(payload)
        results.append(result)
        print(
            json.dumps(
                {
                    "stage": "physical_tangent",
                    "mode": result["mode"],
                    "n_cells": result["n_cells"],
                    "status": "complete",
                },
                sort_keys=True,
            ),
            flush=True,
        )
    rows = {
        mode: {
            result["n_cells"]: result
            for result in results
            if result["mode"] == mode
        }
        for mode in ("piecewise_constant", "plm_smooth")
    }
    pairs = {
        mode: [
            _tangent_pair(rows[mode][16], rows[mode][32]),
            _tangent_pair(rows[mode][32], rows[mode][64]),
        ]
        for mode in rows
    }
    orders = {
        mode: {
            region: {
                name: float(
                    np.log2(
                        pairs[mode][0][name][region][
                            "maximum_absolute_difference_per_s"
                        ]
                        / pairs[mode][1][name][region][
                            "maximum_absolute_difference_per_s"
                        ]
                    )
                )
                for name in (
                    "full",
                    "central",
                    "rusanov",
                    "total_transport",
                )
            }
            for region in (
                "full_domain",
                "diagnosed_interior_band",
            )
        }
        for mode in rows
    }
    reduction = {
        region: {
            name: float(
                pairs["piecewise_constant"][1][name][region][
                    "maximum_absolute_difference_per_s"
                ]
                / pairs["plm_smooth"][1][name][region][
                    "maximum_absolute_difference_per_s"
                ]
            )
            for name in (
                "full",
                "central",
                "rusanov",
                "total_transport",
            )
        }
        for region in (
            "full_domain",
            "diagnosed_interior_band",
        )
    }
    maximum_defect = max(
        result[name]
        for result in results
        for name in (
            "maximum_scaled_consistency_defect",
            "maximum_residual_reconstruction_relative_defect",
            "maximum_tangent_reconstruction_relative_defect",
        )
    )
    rank_rows = {
        str(n_cells): {
            "dimensions": list(
                rows["plm_smooth"][n_cells][
                    "consistency_dimensions"
                ]
            ),
            "numerical_rank": rows["plm_smooth"][n_cells][
                "consistency_numerical_rank"
            ],
            "condition_estimate": rows["plm_smooth"][n_cells][
                "consistency_condition_estimate"
            ],
        }
        for n_cells in (16, 32)
    }
    rank_passed = all(
        rank["numerical_rank"] == rank["dimensions"][0]
        for rank in rank_rows.values()
    )
    passed = bool(
        orders["plm_smooth"]["diagnosed_interior_band"][
            "total_transport"
        ]
        >= MINIMUM_RECONSTRUCTED_ORDER
        and orders["plm_smooth"]["diagnosed_interior_band"]["full"]
        >= MINIMUM_RECONSTRUCTED_ORDER
        and reduction["diagnosed_interior_band"]["full"]
        >= MINIMUM_TANGENT_REDUCTION
        and maximum_defect <= 1.0e-8
        and rank_passed
    )
    arrays = {}
    for result in results:
        prefix = f"{result['mode']}_n{result['n_cells']}"
        arrays[f"{prefix}_radius_rg"] = result["radius_rg"]
        for name in ("full", "central", "rusanov", "total_transport"):
            arrays[f"{prefix}_{name}"] = result[name]
    summaries = {
        mode: {
            str(n_cells): {
                key: value
                for key, value in rows[mode][n_cells].items()
                if key
                not in (
                    "radius_rg",
                    "cell_measures",
                    "full",
                    "central",
                    "rusanov",
                    "total_transport",
                )
            }
            for n_cells in (16, 32, 64)
        }
        for mode in rows
    }
    return {
        "resolution_summaries": summaries,
        "nested_pair_differences": pairs,
        "observed_orders": orders,
        "n32_n64_reduction_factors": reduction,
        "diagnosed_interior_band_rg": [
            DIAGNOSED_INNER_RADIUS_RG,
            DIAGNOSED_OUTER_RADIUS_RG,
        ],
        "minimum_required_total_transport_order": (
            MINIMUM_RECONSTRUCTED_ORDER
        ),
        "minimum_required_full_tangent_reduction": (
            MINIMUM_TANGENT_REDUCTION
        ),
        "maximum_decomposition_defect": maximum_defect,
        "plm_smooth_consistency_rank": rank_rows,
        "passed": passed,
    }, arrays


def main() -> None:
    args = _arguments()
    output_path = _absolute(args.output)
    arrays_path = _absolute(args.arrays)
    seed_parameters, seed_summary = _source_compatible_seed_parameters()
    manufactured, manufactured_arrays = _manufactured_audit(
        seed_parameters
    )
    jacobian = _jacobian_audit(seed_parameters)
    print(
        json.dumps(
            {
                "stage": "jacobian_audit",
                "passed": jacobian["passed"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    tangents, tangent_arrays = _tangent_audit(seed_parameters)
    print(
        json.dumps(
            {
                "stage": "physical_tangent_audit",
                "passed": tangents["passed"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    passed = bool(
        manufactured["passed"]
        and jacobian["passed"]
        and tangents["passed"]
    )
    decision = (
        "wp10c7h_reconstructed_flux_trajectory_authorized"
        if passed
        else "wp10c7h_blocked_by_reconstruction_method_gate"
    )
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arrays_path,
        **manufactured_arrays,
        **tangent_arrays,
    )
    payload = {
        "work_package": "WP10c7g",
        "base_commit": BASE_COMMIT,
        "scope": (
            "method-only optional PLM reconstruction audit; no disk "
            "trajectory and no physics, boundary, source, or Riemann "
            "solver change"
        ),
        "seed": seed_summary,
        "reconstruction_contract": {
            "chart": (
                "lnSigma,betaR,betaPhi,lnT,specific causal stress"
            ),
            "coordinate": "ln(R)",
            "production_mode": "plm_smooth",
            "oracle_mode": "plm_unlimited",
            "frozen_control": "piecewise_constant",
            "boundary_policy": (
                "unchanged one-sided piecewise-constant physical maps"
            ),
            "rusanov_policy": (
                "reconstructed left/right charts feed conserved states, "
                "central fluxes, characteristic envelope, and jump"
            ),
        },
        "manufactured_operator_audit": manufactured,
        "jacobian_audit": jacobian,
        "physical_common_state_tangent_audit": tangents,
        "gates": {
            "manufactured_operator_passed": manufactured["passed"],
            "jacobian_passed": jacobian["passed"],
            "physical_tangent_passed": tangents["passed"],
            "wp10c7g_passed": passed,
        },
        "decision": decision,
        "wp10c7h_authorized": passed,
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
                "output": _relative(output_path),
                "arrays": _relative(arrays_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
