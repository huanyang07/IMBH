"""Run the bounded WP10c5d consistent-data and tiny-step gate."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.linalg.lapack import dgeequ, dgesvx

from imri_qpe.constants import C, G
from imri_qpe.layer3_minidisk_1d import (
    KERR_SCHILD_HILL_ENERGY_ZERO,
    CausalFiveFieldDAEContext,
    GasRadiationHillRocheNozzleProvider,
    SchwarzschildCurvatureVerticalFrequency,
    audit_causal_five_field_consistent_initial_data,
    audit_causal_five_field_dae_jacobian,
    causal_five_field_dae_scaling,
    causal_five_field_endpoint_temporal_storage_increment,
    causal_five_field_path_temporal_storage_increment,
    causal_five_field_reduced_backward_euler_residual,
    causal_five_field_state_from_primitives,
    evaluate_causal_five_field_dae,
    fiducial_hill_roche_nozzle_geometry,
    make_causal_five_field_seed,
    make_kerr_schild_column_grid,
    pack_causal_five_field_state,
)
from imri_qpe.parameters import FiducialParams


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/tables/causal_five_field_consistent_step_wp10c5d.json"
)
RANK_THRESHOLD = 1.0e-11
FINITE_DIFFERENCE_STEP = 2.0e-6
TARGET_SCALED_PRIMITIVE_CHANGES = (1.0e-4, 1.0e-3)
FIELD_NAMES = (
    "rest_mass",
    "radial_momentum",
    "angular_momentum",
    "killing_energy",
    "relaxing_stress",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--temporal-storage-scheme",
        choices=("endpoint", "path_integrated"),
        default="endpoint",
    )
    parser.add_argument(
        "--linear-precision-audit",
        action="store_true",
    )
    parser.add_argument(
        "--directional-consistency-audit",
        action="store_true",
    )
    return parser.parse_args()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _context(n_cells: int) -> CausalFiveFieldDAEContext:
    mass = FiducialParams().M2_g
    gravitational_radius = G * mass / C**2
    grid = make_kerr_schild_column_grid(
        1.8 * gravitational_radius,
        335.0 * gravitational_radius,
        n_cells,
        gravitational_radius,
    )
    geometry = replace(
        fiducial_hill_roche_nozzle_geometry(),
        energy_zero=KERR_SCHILD_HILL_ENERGY_ZERO,
    )
    return CausalFiveFieldDAEContext(
        grid=grid,
        vertical_frequency=SchwarzschildCurvatureVerticalFrequency(
            gravitational_radius
        ),
        outer_boundary_provider=GasRadiationHillRocheNozzleProvider(
            geometry,
            transverse_quadrature_zones=24,
        ),
        include_radiative_cooling=True,
    ).validated()


def _ledger_defect(
    new_state,
    evaluation,
) -> tuple[float, list[float]]:
    telescoped = np.asarray(
        [
            math.fsum(evaluation.conservation_rows[:, field])
            for field in range(5)
        ],
        dtype=float,
    )
    boundary = (
        new_state.weighted_face_fluxes_over_c[-1]
        - new_state.weighted_face_fluxes_over_c[0]
    )
    cell_terms = (
        -evaluation.integrated_sources_per_ct
        + evaluation.temporal_conserved_storage
    )
    cell_terms[:, :4] += evaluation.temporal_vertical_storage
    expected = boundary + np.asarray(
        [
            math.fsum(cell_terms[:, field])
            for field in range(5)
        ],
        dtype=float,
    )
    scale = np.maximum(
        np.abs(new_state.weighted_face_fluxes_over_c[-1])
        + np.abs(new_state.weighted_face_fluxes_over_c[0])
        + np.sum(
            np.abs(evaluation.integrated_sources_per_ct),
            axis=0,
        )
        + np.sum(
            np.abs(evaluation.temporal_conserved_storage),
            axis=0,
        ),
        1.0,
    )
    scale[:4] += np.sum(
        np.abs(evaluation.temporal_vertical_storage),
        axis=0,
    )
    component_defect = (telescoped - expected) / scale
    return (
        float(np.max(np.abs(component_defect))),
        [float(value) for value in component_defect],
    )


def _field_norms(values: np.ndarray, n_cells: int) -> list[float]:
    fields = np.asarray(values, dtype=float).reshape(n_cells, 5)
    return [float(value) for value in np.linalg.norm(fields, axis=0)]


def _storage_rate(
    context: CausalFiveFieldDAEContext,
    increment,
    timestep_seconds: float,
) -> np.ndarray:
    combined = np.array(increment.conserved_increment, copy=True)
    combined[:, :4] += increment.vertical_killing_increment
    return (
        context.grid.cell_measures[:, None]
        * combined
        / (C * timestep_seconds)
    )


def _storage_comparison(
    left_rate: np.ndarray,
    right_rate: np.ndarray,
    conservation_scale: np.ndarray,
) -> dict:
    scaled = (left_rate - right_rate) / conservation_scale
    flat = int(np.argmax(np.abs(scaled)))
    cell, field = np.unravel_index(flat, scaled.shape)
    field_names = (
        "rest_mass",
        "radial_momentum",
        "angular_momentum",
        "killing_energy",
        "relaxing_stress",
    )
    return {
        "maximum_scaled_rate_defect": float(np.max(np.abs(scaled))),
        "component_maximum_scaled_rate_defects": [
            float(value)
            for value in np.max(np.abs(scaled), axis=0)
        ],
        "controlling_cell": int(cell),
        "controlling_field": field_names[field],
        "controlling_scaled_rate_defect": float(scaled[cell, field]),
    }


def _storage_increment_audit(
    context: CausalFiveFieldDAEContext,
    old_primitives: np.ndarray,
    new_primitives: np.ndarray,
    timestep_seconds: float,
    conservation_scale: np.ndarray,
) -> dict:
    endpoint = causal_five_field_endpoint_temporal_storage_increment(
        context,
        old_primitives,
        new_primitives,
    )
    path2 = causal_five_field_path_temporal_storage_increment(
        context,
        old_primitives,
        new_primitives,
        quadrature_order=2,
        directional_step=1.0e-3,
    )
    path4 = causal_five_field_path_temporal_storage_increment(
        context,
        old_primitives,
        new_primitives,
        quadrature_order=4,
        directional_step=1.0e-3,
    )
    path8 = causal_five_field_path_temporal_storage_increment(
        context,
        old_primitives,
        new_primitives,
        quadrature_order=8,
        directional_step=1.0e-3,
    )
    path8_half = causal_five_field_path_temporal_storage_increment(
        context,
        old_primitives,
        new_primitives,
        quadrature_order=8,
        directional_step=5.0e-4,
    )
    path8_double = causal_five_field_path_temporal_storage_increment(
        context,
        old_primitives,
        new_primitives,
        quadrature_order=8,
        directional_step=2.0e-3,
    )
    rates = {
        "endpoint": _storage_rate(
            context,
            endpoint,
            timestep_seconds,
        ),
        "path_order_2": _storage_rate(
            context,
            path2,
            timestep_seconds,
        ),
        "path_order_4": _storage_rate(
            context,
            path4,
            timestep_seconds,
        ),
        "path_order_8": _storage_rate(
            context,
            path8,
            timestep_seconds,
        ),
        "path_order_8_step_half": _storage_rate(
            context,
            path8_half,
            timestep_seconds,
        ),
        "path_order_8_step_double": _storage_rate(
            context,
            path8_double,
            timestep_seconds,
        ),
    }
    comparisons = {
        "endpoint_vs_path_order_8": _storage_comparison(
            rates["endpoint"],
            rates["path_order_8"],
            conservation_scale,
        ),
        "path_order_2_vs_8": _storage_comparison(
            rates["path_order_2"],
            rates["path_order_8"],
            conservation_scale,
        ),
        "path_order_4_vs_8": _storage_comparison(
            rates["path_order_4"],
            rates["path_order_8"],
            conservation_scale,
        ),
        "path_step_half_vs_base": _storage_comparison(
            rates["path_order_8_step_half"],
            rates["path_order_8"],
            conservation_scale,
        ),
        "path_step_double_vs_base": _storage_comparison(
            rates["path_order_8_step_double"],
            rates["path_order_8"],
            conservation_scale,
        ),
    }
    convergence_gate = 5.0e-9
    passed = all(
        comparisons[name]["maximum_scaled_rate_defect"]
        <= convergence_gate
        for name in (
            "path_order_4_vs_8",
            "path_step_half_vs_base",
            "path_step_double_vs_base",
        )
    )
    return {
        "convergence_gate": convergence_gate,
        "comparisons": comparisons,
        "maximum_absolute_endpoint_vertical_work_per_area": float(
            np.max(np.abs(endpoint.vertical_work_per_area))
        ),
        "maximum_absolute_path_vertical_work_per_area": float(
            np.max(np.abs(path8.vertical_work_per_area))
        ),
        "passed": passed,
    }


def _bounded_newton(
    residual,
    jacobian,
    initial: np.ndarray,
    *,
    bound: float,
    residual_tolerance: float,
    maximum_iterations: int = 12,
    linear_solver: str = "direct",
) -> tuple[np.ndarray, dict, np.ndarray | None, np.ndarray]:
    state = np.asarray(initial, dtype=float)
    history = []
    message = "maximum iterations reached"
    success = False
    jacobian_evaluations = 0
    function_evaluations = 0
    last_matrix = None
    last_values = np.asarray(residual(state), dtype=float)
    function_evaluations += 1
    for iteration in range(maximum_iterations + 1):
        values = last_values
        maximum_residual = float(np.max(np.abs(values)))
        row = {
            "iteration": iteration,
            "maximum_residual": maximum_residual,
        }
        history.append(row)
        if maximum_residual <= residual_tolerance:
            message = "residual gate passed"
            success = True
            break
        if iteration == maximum_iterations:
            break
        matrix = np.asarray(jacobian(state), dtype=float)
        last_matrix = matrix
        jacobian_evaluations += 1
        singular = np.linalg.svd(matrix, compute_uv=False)
        row["jacobian_condition_estimate"] = float(
            singular[0] / max(singular[-1], np.finfo(float).tiny)
        )
        try:
            if linear_solver == "direct":
                correction = np.linalg.solve(matrix, -values)
            elif linear_solver == "dgesvx":
                result = dgesvx(
                    matrix,
                    (-values).reshape(-1, 1),
                    fact="E",
                )
                correction = np.asarray(result[7], dtype=float).ravel()
                if int(result[11]) != 0:
                    raise np.linalg.LinAlgError(
                        f"dgesvx failed with info={int(result[11])}"
                    )
            else:
                raise ValueError("unknown reduced Newton linear solver")
        except np.linalg.LinAlgError:
            message = "reduced Newton Jacobian is singular"
            break
        row["raw_correction_maximum"] = float(
            np.max(np.abs(correction))
        )
        alpha = 1.0
        positive = correction > 0.0
        negative = correction < 0.0
        if np.any(positive):
            alpha = min(
                alpha,
                float(
                    np.min(
                        (bound - state[positive]) / correction[positive]
                    )
                ),
            )
        if np.any(negative):
            alpha = min(
                alpha,
                float(
                    np.min(
                        (-bound - state[negative]) / correction[negative]
                    )
                ),
            )
        alpha = min(1.0, max(0.0, 0.99 * alpha))
        accepted = False
        for _line_search in range(14):
            candidate = state + alpha * correction
            candidate_values = np.asarray(residual(candidate), dtype=float)
            function_evaluations += 1
            if np.max(np.abs(candidate_values)) < maximum_residual:
                state = candidate
                last_values = candidate_values
                row["accepted_alpha"] = alpha
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            message = "bound-aware line search failed"
            break
    return state, {
        "success": success,
        "linear_solver": linear_solver,
        "message": message,
        "iterations": len(history) - 1,
        "function_evaluations": function_evaluations,
        "jacobian_evaluations": jacobian_evaluations,
        "history": history,
    }, last_matrix, last_values


def _linear_residual(
    matrix: np.ndarray,
    correction: np.ndarray,
    values: np.ndarray,
) -> float:
    return float(
        np.max(np.abs(matrix @ correction + values))
        / max(np.max(np.abs(values)), np.finfo(float).tiny)
    )


def _primitive_mode_localization(
    vector: np.ndarray,
    n_cells: int,
) -> dict:
    fields = np.asarray(vector, dtype=float).reshape(n_cells, 5)
    cell_norms = np.linalg.norm(fields, axis=1)
    total = max(float(np.linalg.norm(fields)), np.finfo(float).tiny)
    return {
        "field_norms": [
            float(value) for value in np.linalg.norm(fields, axis=0)
        ],
        "maximum_cell": int(np.argmax(cell_norms)),
        "maximum_cell_fraction": float(np.max(cell_norms) / total),
        "outermost_cell_fraction": float(cell_norms[-1] / total),
    }


def _longdouble_difference(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        np.asarray(left, dtype=np.longdouble)
        - np.asarray(right, dtype=np.longdouble),
        dtype=float,
    )


def _longdouble_stencil(
    minus_two: np.ndarray,
    minus_one: np.ndarray,
    plus_one: np.ndarray,
    plus_two: np.ndarray,
    *,
    denominator: float,
) -> np.ndarray:
    return np.asarray(
        (
            -np.asarray(plus_two, dtype=np.longdouble)
            + 8.0 * np.asarray(plus_one, dtype=np.longdouble)
            - 8.0 * np.asarray(minus_one, dtype=np.longdouble)
            + np.asarray(minus_two, dtype=np.longdouble)
        )
        / np.longdouble(denominator),
        dtype=float,
    )


def _component_cell_rows(components: dict[str, np.ndarray]) -> dict:
    face_fluxes = np.asarray(components["face_fluxes"], dtype=float)
    return {
        "face_flux_difference": face_fluxes[1:] - face_fluxes[:-1],
        "geometric_thermal_sources": np.asarray(
            components["geometric_thermal_sources"],
            dtype=float,
        ),
        "path_conserved_storage": np.asarray(
            components["path_conserved_storage"],
            dtype=float,
        ),
        "responsive_height_work": np.asarray(
            components["responsive_height_work"],
            dtype=float,
        ),
    }


def _component_difference(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    delta_faces = _longdouble_difference(
        left["face_fluxes"],
        right["face_fluxes"],
    )
    return {
        "face_flux_difference": _longdouble_difference(
            delta_faces[1:],
            delta_faces[:-1],
        ),
        "geometric_thermal_sources": _longdouble_difference(
            left["geometric_thermal_sources"],
            right["geometric_thermal_sources"],
        ),
        "path_conserved_storage": _longdouble_difference(
            left["path_conserved_storage"],
            right["path_conserved_storage"],
        ),
        "responsive_height_work": _longdouble_difference(
            left["responsive_height_work"],
            right["responsive_height_work"],
        ),
    }


def _component_directional_stencil(
    minus_two: dict[str, np.ndarray],
    minus_one: dict[str, np.ndarray],
    plus_one: dict[str, np.ndarray],
    plus_two: dict[str, np.ndarray],
    *,
    denominator: float,
) -> dict[str, np.ndarray]:
    face_derivative = _longdouble_stencil(
        minus_two["face_fluxes"],
        minus_one["face_fluxes"],
        plus_one["face_fluxes"],
        plus_two["face_fluxes"],
        denominator=denominator,
    )
    result = {
        "face_flux_difference": _longdouble_difference(
            face_derivative[1:],
            face_derivative[:-1],
        )
    }
    for name in (
        "geometric_thermal_sources",
        "path_conserved_storage",
        "responsive_height_work",
    ):
        result[name] = _longdouble_stencil(
            minus_two[name],
            minus_one[name],
            plus_one[name],
            plus_two[name],
            denominator=denominator,
        )
    return result


def _component_second_order_directional_stencil(
    minus_one: dict[str, np.ndarray],
    plus_one: dict[str, np.ndarray],
    *,
    denominator: float,
) -> dict[str, np.ndarray]:
    delta_faces = _longdouble_difference(
        plus_one["face_fluxes"],
        minus_one["face_fluxes"],
    ) / denominator
    result = {
        "face_flux_difference": _longdouble_difference(
            delta_faces[1:],
            delta_faces[:-1],
        )
    }
    for name in (
        "geometric_thermal_sources",
        "path_conserved_storage",
        "responsive_height_work",
    ):
        result[name] = (
            _longdouble_difference(
                plus_one[name],
                minus_one[name],
            )
            / denominator
        )
    return result


def _sum_components(components: dict[str, np.ndarray]) -> np.ndarray:
    names = (
        "face_flux_difference",
        "geometric_thermal_sources",
        "path_conserved_storage",
        "responsive_height_work",
    )
    values = np.zeros_like(components[names[0]], dtype=np.longdouble)
    for name in names:
        values += np.asarray(components[name], dtype=np.longdouble)
    return np.asarray(values, dtype=float)


def _scaled_components(
    components: dict[str, np.ndarray],
    row_scales: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(values, dtype=float) / row_scales
        for name, values in components.items()
    }


def _array_defect_summary(
    left: np.ndarray,
    right: np.ndarray,
) -> dict:
    defect = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    flat = int(np.argmax(np.abs(defect)))
    cell, field = np.unravel_index(flat, defect.shape)
    denominator = max(
        float(np.max(np.abs(left))),
        float(np.max(np.abs(right))),
        np.finfo(float).tiny,
    )
    return {
        "maximum_absolute_defect": float(np.max(np.abs(defect))),
        "relative_maximum_defect": float(
            np.max(np.abs(defect)) / denominator
        ),
        "controlling_cell": int(cell),
        "controlling_field": FIELD_NAMES[field],
        "controlling_defect": float(defect[cell, field]),
        "field_maximum_absolute_defects": [
            float(value) for value in np.max(np.abs(defect), axis=0)
        ],
    }


def _component_prediction_summary(
    actual: np.ndarray,
    second_order: np.ndarray,
    fourth_order: np.ndarray,
) -> dict:
    return {
        "maximum_actual_increment": float(np.max(np.abs(actual))),
        "maximum_second_order_prediction": float(
            np.max(np.abs(second_order))
        ),
        "maximum_fourth_order_prediction": float(
            np.max(np.abs(fourth_order))
        ),
        "actual_vs_second_order": _array_defect_summary(
            actual,
            second_order,
        ),
        "actual_vs_fourth_order": _array_defect_summary(
            actual,
            fourth_order,
        ),
        "second_vs_fourth_order": _array_defect_summary(
            second_order,
            fourth_order,
        ),
    }


def _directional_consistency_audit(
    residual,
    component_evaluation,
    state: np.ndarray,
    values: np.ndarray,
    second_order_jacobian: np.ndarray,
    row_scales: np.ndarray,
) -> dict:
    correction = np.linalg.solve(second_order_jacobian, -values)
    correction_norm = float(np.linalg.norm(correction))
    if correction_norm <= np.finfo(float).tiny:
        raise RuntimeError("directional audit correction is zero")
    direction = correction / correction_norm
    step = FINITE_DIFFERENCE_STEP

    base = component_evaluation(state)
    corrected = component_evaluation(state + correction)
    minus_two = component_evaluation(state - 2.0 * step * direction)
    minus_one = component_evaluation(state - step * direction)
    plus_one = component_evaluation(state + step * direction)
    plus_two = component_evaluation(state + 2.0 * step * direction)

    base_rows = _scaled_components(
        _component_cell_rows(base),
        row_scales,
    )
    corrected_rows = _scaled_components(
        _component_cell_rows(corrected),
        row_scales,
    )
    actual = _scaled_components(
        _component_difference(corrected, base),
        row_scales,
    )
    second = _scaled_components(
        _component_second_order_directional_stencil(
            minus_one,
            plus_one,
            denominator=2.0 * step,
        ),
        row_scales,
    )
    fourth = _scaled_components(
        _component_directional_stencil(
            minus_two,
            minus_one,
            plus_one,
            plus_two,
            denominator=12.0 * step,
        ),
        row_scales,
    )
    for component_set in (second, fourth):
        for name in component_set:
            component_set[name] *= correction_norm

    base_sum = _sum_components(base_rows)
    corrected_sum = _sum_components(corrected_rows)
    actual_sum = _sum_components(actual)
    second_sum = _sum_components(second)
    fourth_sum = _sum_components(fourth)
    base_residual = np.asarray(residual(state), dtype=float).reshape(
        row_scales.shape
    )
    corrected_residual = np.asarray(
        residual(state + correction),
        dtype=float,
    ).reshape(row_scales.shape)
    residual_delta = _longdouble_difference(
        corrected_residual,
        base_residual,
    )
    jacobian_prediction = (
        second_order_jacobian @ correction
    ).reshape(row_scales.shape)

    component_summaries = {
        name: _component_prediction_summary(
            actual[name],
            second[name],
            fourth[name],
        )
        for name in actual
    }
    convergence_gate = 5.0e-9
    nonlinear_identity_gate = 1.0e-8
    failing_components = [
        name
        for name, summary in component_summaries.items()
        if summary["second_vs_fourth_order"][
            "maximum_absolute_defect"
        ]
        <= convergence_gate
        and summary["actual_vs_fourth_order"][
            "maximum_absolute_defect"
        ]
        > nonlinear_identity_gate
    ]
    uniquely_identified = (
        failing_components[0] if len(failing_components) == 1 else None
    )
    return {
        "correction_norm": correction_norm,
        "maximum_absolute_correction": float(
            np.max(np.abs(correction))
        ),
        "directional_finite_difference_step": step,
        "compensated_differences_are_diagnostic_only": True,
        "component_summaries": component_summaries,
        "identities": {
            "base_component_sum_vs_residual": _array_defect_summary(
                base_sum,
                base_residual,
            ),
            "corrected_component_sum_vs_residual": (
                _array_defect_summary(
                    corrected_sum,
                    corrected_residual,
                )
            ),
            "actual_component_increment_sum_vs_residual_delta": (
                _array_defect_summary(
                    actual_sum,
                    residual_delta,
                )
            ),
            "second_order_component_sum_vs_jacobian_prediction": (
                _array_defect_summary(
                    second_sum,
                    jacobian_prediction,
                )
            ),
            "fourth_order_component_sum_vs_jacobian_prediction": (
                _array_defect_summary(
                    fourth_sum,
                    jacobian_prediction,
                )
            ),
            "base_plus_jacobian_prediction": {
                "maximum_absolute_linearized_residual": float(
                    np.max(
                        np.abs(
                            base_residual + jacobian_prediction
                        )
                    )
                )
            },
            "corrected_nonlinear_residual": {
                "maximum_absolute_residual": float(
                    np.max(np.abs(corrected_residual))
                ),
                "controlling_cell": int(
                    np.unravel_index(
                        int(np.argmax(np.abs(corrected_residual))),
                        corrected_residual.shape,
                    )[0]
                ),
                "controlling_field": FIELD_NAMES[
                    np.unravel_index(
                        int(np.argmax(np.abs(corrected_residual))),
                        corrected_residual.shape,
                    )[1]
                ],
            },
        },
        "gates": {
            "directional_convergence_gate": convergence_gate,
            "nonlinear_component_identity_gate": (
                nonlinear_identity_gate
            ),
            "failing_components": failing_components,
            "uniquely_identified_failing_component": uniquely_identified,
            "production_repair_authorized": uniquely_identified is not None,
        },
    }


def _fourth_order_jacobian(
    residual,
    state: np.ndarray,
) -> tuple[np.ndarray, int]:
    size = state.size
    matrix = np.empty((size, size), dtype=float)
    for index in range(size):
        plus_one = np.array(state, copy=True)
        minus_one = np.array(state, copy=True)
        plus_two = np.array(state, copy=True)
        minus_two = np.array(state, copy=True)
        plus_one[index] += FINITE_DIFFERENCE_STEP
        minus_one[index] -= FINITE_DIFFERENCE_STEP
        plus_two[index] += 2.0 * FINITE_DIFFERENCE_STEP
        minus_two[index] -= 2.0 * FINITE_DIFFERENCE_STEP
        matrix[:, index] = (
            -np.asarray(residual(plus_two), dtype=float)
            + 8.0 * np.asarray(residual(plus_one), dtype=float)
            - 8.0 * np.asarray(residual(minus_one), dtype=float)
            + np.asarray(residual(minus_two), dtype=float)
        ) / (12.0 * FINITE_DIFFERENCE_STEP)
    return matrix, 4 * size


def _linear_precision_audit(
    residual,
    state: np.ndarray,
    values: np.ndarray,
    second_order_jacobian: np.ndarray,
    *,
    bound: float,
    n_cells: int,
) -> dict:
    fourth_order, function_evaluations = _fourth_order_jacobian(
        residual,
        state,
    )
    left, singular, right = np.linalg.svd(
        second_order_jacobian,
        full_matrices=False,
    )
    fourth_singular = np.linalg.svd(
        fourth_order,
        compute_uv=False,
    )
    direct = np.linalg.solve(second_order_jacobian, -values)
    (
        equilibrated_matrix,
        _lu,
        _pivots,
        equilibrated,
        row_scale,
        column_scale,
        _scaled_rhs,
        equilibrated_correction_matrix,
        reciprocal_condition,
        forward_error,
        backward_error,
        info,
    ) = dgesvx(
        second_order_jacobian,
        (-values).reshape(-1, 1),
        fact="E",
    )
    if int(info) != 0:
        raise RuntimeError(f"dgesvx precision audit failed with info={info}")
    equilibrated_correction = np.asarray(
        equilibrated_correction_matrix,
        dtype=float,
    ).ravel()
    fourth_correction = np.linalg.solve(fourth_order, -values)
    (
        _rows,
        _columns,
        row_condition,
        column_condition,
        maximum_entry,
        equilibration_info,
    ) = dgeequ(second_order_jacobian)
    candidate_corrections = {
        "direct_second_order": direct,
        "dgesvx_second_order": equilibrated_correction,
        "direct_fourth_order": fourth_correction,
    }
    nonlinear = {}
    for name, correction in candidate_corrections.items():
        within_bound = bool(
            np.max(np.abs(state + correction)) <= bound
        )
        nonlinear[name] = {
            "maximum_correction": float(np.max(np.abs(correction))),
            "within_bound": within_bound,
            "maximum_residual_after_full_correction": (
                float(np.max(np.abs(residual(state + correction))))
                if within_bound
                else np.inf
            ),
        }
    jacobian_difference = fourth_order - second_order_jacobian
    relative_frobenius_defect = float(
        np.linalg.norm(jacobian_difference)
        / max(
            np.linalg.norm(fourth_order),
            np.linalg.norm(second_order_jacobian),
            np.finfo(float).tiny,
        )
    )
    correction_difference = float(
        np.linalg.norm(equilibrated_correction - direct)
        / max(np.linalg.norm(direct), np.finfo(float).tiny)
    )
    fourth_correction_difference = float(
        np.linalg.norm(fourth_correction - direct)
        / max(np.linalg.norm(direct), np.finfo(float).tiny)
    )
    recoverable = bool(
        nonlinear["dgesvx_second_order"][
            "maximum_residual_after_full_correction"
        ]
        <= 1.0e-8
        and correction_difference >= 1.0e-8
        and backward_error[0] <= 1.0e-12
    )
    return {
        "dimensions": list(second_order_jacobian.shape),
        "function_evaluations": function_evaluations,
        "second_order_condition_estimate": float(
            singular[0] / max(singular[-1], np.finfo(float).tiny)
        ),
        "fourth_order_condition_estimate": float(
            fourth_singular[0]
            / max(fourth_singular[-1], np.finfo(float).tiny)
        ),
        "second_order_smallest_singular_value": float(singular[-1]),
        "fourth_order_smallest_singular_value": float(
            fourth_singular[-1]
        ),
        "weakest_right": _primitive_mode_localization(
            right[-1],
            n_cells,
        ),
        "weakest_left": _primitive_mode_localization(
            left[:, -1],
            n_cells,
        ),
        "equilibration": {
            "lapack_equed": equilibrated.decode(),
            "row_scale_minimum": float(np.min(row_scale)),
            "row_scale_maximum": float(np.max(row_scale)),
            "column_scale_minimum": float(np.min(column_scale)),
            "column_scale_maximum": float(np.max(column_scale)),
            "row_condition": float(row_condition),
            "column_condition": float(column_condition),
            "maximum_matrix_entry": float(maximum_entry),
            "dgeequ_info": int(equilibration_info),
            "equilibrated_condition_estimate": float(
                np.linalg.cond(equilibrated_matrix)
            ),
        },
        "dgesvx": {
            "reciprocal_condition_estimate": float(
                reciprocal_condition
            ),
            "forward_error_bound": float(forward_error[0]),
            "backward_error": float(backward_error[0]),
        },
        "linear_relative_residuals": {
            name: _linear_residual(
                second_order_jacobian
                if name != "direct_fourth_order"
                else fourth_order,
                correction,
                values,
            )
            for name, correction in candidate_corrections.items()
        },
        "relative_direct_dgesvx_correction_difference": (
            correction_difference
        ),
        "relative_second_fourth_correction_difference": (
            fourth_correction_difference
        ),
        "relative_second_fourth_jacobian_frobenius_defect": (
            relative_frobenius_defect
        ),
        "nonlinear_full_correction": nonlinear,
        "recoverable_precision_demonstrated": recoverable,
    }


def _run_resolution(
    n_cells: int,
    target_scaled_primitive_change: float,
    *,
    temporal_storage_scheme: str = "endpoint",
    linear_solver: str = "direct",
    include_linear_precision_audit: bool = False,
    include_directional_consistency_audit: bool = False,
) -> dict:
    context = _context(n_cells)
    old_state = make_causal_five_field_seed(context)
    old_vector = pack_causal_five_field_state(old_state)
    stationary_evaluation = evaluate_causal_five_field_dae(
        old_vector,
        context,
    )
    scaling = causal_five_field_dae_scaling(
        old_state,
        stationary_evaluation,
    )
    audit_kwargs = {
        "finite_difference_step": FINITE_DIFFERENCE_STEP,
        "rank_relative_threshold": RANK_THRESHOLD,
    }
    stationary = audit_causal_five_field_dae_jacobian(
        lambda trial: evaluate_causal_five_field_dae(
            trial,
            context,
        ).residual,
        old_vector,
        scaling,
        **audit_kwargs,
    )
    descriptor_timestep = 1.0
    backward_euler = audit_causal_five_field_dae_jacobian(
        lambda trial: evaluate_causal_five_field_dae(
            trial,
            context,
            old_vector=old_vector,
            timestep_seconds=descriptor_timestep,
        ).residual,
        old_vector,
        scaling,
        **audit_kwargs,
    )
    consistent = audit_causal_five_field_consistent_initial_data(
        context,
        old_state,
        stationary,
        backward_euler,
        scaling=scaling,
        descriptor_timestep_seconds=descriptor_timestep,
        rank_relative_threshold=RANK_THRESHOLD,
    )
    n_differential = 5 * n_cells
    primitive_columns = slice(n_differential, 2 * n_differential)
    primitive_tangent = np.asarray(
        consistent.scaled_tangent[primitive_columns],
        dtype=float,
    )
    timestep = float(
        target_scaled_primitive_change
        / max(
            np.max(np.abs(primitive_tangent)),
            np.finfo(float).tiny,
        )
    )
    primitive_scale = scaling.column_scales[primitive_columns]
    old_primitives = np.asarray(old_state.primitives, dtype=float).ravel()
    initial_increment = timestep * primitive_tangent

    def residual(scaled_increment: np.ndarray) -> np.ndarray:
        primitives = old_primitives + primitive_scale * scaled_increment
        return (
            causal_five_field_reduced_backward_euler_residual(
                primitives,
                context,
                old_vector=old_vector,
                timestep_seconds=timestep,
                temporal_storage_scheme=temporal_storage_scheme,
            )
            / scaling.row_scales[:n_differential]
        )

    def jacobian(scaled_increment: np.ndarray) -> np.ndarray:
        columns = np.empty((n_differential, n_differential), dtype=float)
        for index in range(n_differential):
            plus = np.array(scaled_increment, copy=True)
            minus = np.array(scaled_increment, copy=True)
            plus[index] += FINITE_DIFFERENCE_STEP
            minus[index] -= FINITE_DIFFERENCE_STEP
            columns[:, index] = (
                residual(plus) - residual(minus)
            ) / (2.0 * FINITE_DIFFERENCE_STEP)
        return columns

    conservation_row_scales = scaling.row_scales[
        :n_differential
    ].reshape(n_cells, 5)

    def component_evaluation(
        scaled_increment: np.ndarray,
    ) -> dict[str, np.ndarray]:
        primitives = old_primitives + primitive_scale * scaled_increment
        trial_state = causal_five_field_state_from_primitives(
            context,
            primitives.reshape(n_cells, 5),
        )
        trial_evaluation = evaluate_causal_five_field_dae(
            pack_causal_five_field_state(trial_state),
            context,
            old_vector=old_vector,
            timestep_seconds=timestep,
            temporal_storage_scheme=temporal_storage_scheme,
        )
        height_work = np.zeros((n_cells, 5), dtype=float)
        height_work[:, :4] = (
            trial_evaluation.temporal_vertical_storage
        )
        return {
            "face_fluxes": (
                trial_evaluation.numerical_weighted_face_fluxes_over_c
            ),
            "geometric_thermal_sources": (
                -trial_evaluation.integrated_sources_per_ct
            ),
            "path_conserved_storage": (
                trial_evaluation.temporal_conserved_storage
            ),
            "responsive_height_work": height_work,
        }

    nonlinear_bound = 1.25 * target_scaled_primitive_change
    final_increment, solver, last_matrix, last_values = _bounded_newton(
        residual,
        jacobian,
        initial_increment,
        bound=nonlinear_bound,
        residual_tolerance=1.0e-8,
        linear_solver=linear_solver,
    )
    linear_precision = None
    directional_consistency = None
    if (
        include_linear_precision_audit
        or include_directional_consistency_audit
    ):
        last_values = np.asarray(
            residual(final_increment),
            dtype=float,
        )
        last_matrix = jacobian(final_increment)
        linear_precision = _linear_precision_audit(
            residual,
            final_increment,
            last_values,
            last_matrix,
            bound=nonlinear_bound,
            n_cells=n_cells,
        )
        if include_directional_consistency_audit:
            directional_consistency = (
                _directional_consistency_audit(
                    residual,
                    component_evaluation,
                    final_increment,
                    last_values,
                    last_matrix,
                    conservation_row_scales,
                )
            )
    new_primitives = old_primitives + primitive_scale * final_increment
    new_state = causal_five_field_state_from_primitives(
        context,
        new_primitives.reshape(n_cells, 5),
    )
    new_vector = pack_causal_five_field_state(new_state)
    evaluation = evaluate_causal_five_field_dae(
        new_vector,
        context,
        old_vector=old_vector,
        timestep_seconds=timestep,
        temporal_storage_scheme=temporal_storage_scheme,
    )
    maximum_scaled_residual = float(
        np.max(np.abs(evaluation.residual / scaling.row_scales))
    )
    scaled_conservation = (
        evaluation.conservation_rows
        / scaling.row_scales[:n_differential].reshape(n_cells, 5)
    )
    controlling_flat_index = int(np.argmax(np.abs(scaled_conservation)))
    controlling_cell, controlling_field = np.unravel_index(
        controlling_flat_index,
        scaled_conservation.shape,
    )
    maximum_scaled_conservation_residual = float(
        np.max(
            np.abs(
                evaluation.conservation_rows.ravel()
                / scaling.row_scales[:n_differential]
            )
        )
    )
    maximum_algebraic_residual = float(
        max(
            np.max(np.abs(evaluation.primitive_map_rows)),
            np.max(np.abs(evaluation.interior_flux_rows)),
            np.max(np.abs(evaluation.inner_flux_rows)),
            np.max(np.abs(evaluation.outer_flux_rows)),
        )
    )
    ledger_defect, component_ledger_defects = _ledger_defect(
        new_state,
        evaluation,
    )
    storage_increment_audit = _storage_increment_audit(
        context,
        old_state.primitives,
        new_state.primitives,
        timestep,
        scaling.row_scales[:n_differential].reshape(n_cells, 5),
    )
    maximum_change = float(np.max(np.abs(final_increment)))
    consistency_passed = (
        consistent.full_rank
        and consistent.descriptor_full_row_rank
        and consistent.maximum_initial_algebraic_residual <= 1.0e-12
        and consistent.maximum_scaled_consistency_residual <= 1.0e-10
        and consistent.storage_balance_residual_norm <= 1.0e-9
        and consistent.algebraic_tangent_residual_norm <= 1.0e-9
    )
    step_passed = (
        solver["success"]
        and maximum_scaled_residual <= 1.0e-8
        and maximum_algebraic_residual <= 1.0e-10
        and maximum_change <= 1.25 * target_scaled_primitive_change
        and evaluation.outer_boundary_choked
        == stationary_evaluation.outer_boundary_choked
        and np.min(evaluation.scattering_optical_depths) > 1.0
        and ledger_defect <= 1.0e-10
        and (
            temporal_storage_scheme == "endpoint"
            or storage_increment_audit["passed"]
        )
    )
    return {
        "n_cells": n_cells,
        "temporal_storage_scheme": temporal_storage_scheme,
        "linear_solver": linear_solver,
        "seed_is_stationary_root": False,
        "seed_maximum_scaled_conservation_residual": float(
            np.max(
                np.abs(
                    stationary_evaluation.conservation_rows.ravel()
                    / scaling.row_scales[:n_differential]
                )
            )
        ),
        "consistent_initial_data": {
            "dimensions": list(consistent.dimensions),
            "numerical_rank": consistent.numerical_rank,
            "full_rank": consistent.full_rank,
            "condition_estimate": consistent.condition_estimate,
            "smallest_singular_value": float(
                consistent.singular_values[-1]
            ),
            "descriptor_dimensions": list(
                consistent.descriptor_dimensions
            ),
            "descriptor_numerical_rank": (
                consistent.descriptor_numerical_rank
            ),
            "descriptor_full_row_rank": (
                consistent.descriptor_full_row_rank
            ),
            "maximum_initial_algebraic_residual": (
                consistent.maximum_initial_algebraic_residual
            ),
            "maximum_scaled_consistency_residual": (
                consistent.maximum_scaled_consistency_residual
            ),
            "storage_balance_residual_norm": (
                consistent.storage_balance_residual_norm
            ),
            "algebraic_tangent_residual_norm": (
                consistent.algebraic_tangent_residual_norm
            ),
            "maximum_scaled_tangent_per_s": (
                consistent.maximum_scaled_tangent
            ),
            "maximum_scaled_primitive_tangent_per_s": (
                consistent.maximum_scaled_primitive_tangent
            ),
            "primitive_tangent_field_norms_per_s": _field_norms(
                primitive_tangent,
                n_cells,
            ),
            "passed": consistency_passed,
        },
        "tiny_step": {
            "timestep_seconds": timestep,
            "target_scaled_primitive_change": (
                target_scaled_primitive_change
            ),
            "maximum_scaled_primitive_change": maximum_change,
            "tangent_predictor_maximum_scaled_change": float(
                np.max(np.abs(initial_increment))
            ),
            "solver_success": solver["success"],
            "solver_message": solver["message"],
            "solver_iterations": solver["iterations"],
            "function_evaluations": solver["function_evaluations"],
            "jacobian_evaluations": solver["jacobian_evaluations"],
            "solver_history": solver["history"],
            "maximum_scaled_residual": maximum_scaled_residual,
            "maximum_scaled_conservation_residual": (
                maximum_scaled_conservation_residual
            ),
            "controlling_residual_cell": int(controlling_cell),
            "controlling_residual_field": FIELD_NAMES[controlling_field],
            "controlling_scaled_residual": float(
                scaled_conservation[controlling_cell, controlling_field]
            ),
            "maximum_absolute_temporal_vertical_storage": float(
                np.max(np.abs(evaluation.temporal_vertical_storage))
            ),
            "maximum_absolute_integrated_source": float(
                np.max(np.abs(evaluation.integrated_sources_per_ct))
            ),
            "maximum_absolute_algebraic_residual": (
                maximum_algebraic_residual
            ),
            "outer_boundary_choked_before": (
                stationary_evaluation.outer_boundary_choked
            ),
            "outer_boundary_choked_after": (
                evaluation.outer_boundary_choked
            ),
            "minimum_scattering_optical_depth": float(
                np.min(evaluation.scattering_optical_depths)
            ),
            "conservation_telescoping_relative_defect": ledger_defect,
            "component_conservation_defects": component_ledger_defects,
            "storage_increment_audit": storage_increment_audit,
            "linear_precision_audit": linear_precision,
            "directional_consistency_audit": directional_consistency,
            "passed": step_passed,
        },
        "resolution_passed": consistency_passed and step_passed,
    }


def main() -> None:
    args = _arguments()
    n16_attempts = []
    for target_change in TARGET_SCALED_PRIMITIVE_CHANGES:
        attempt = _run_resolution(
            16,
            target_change,
            temporal_storage_scheme=args.temporal_storage_scheme,
            include_linear_precision_audit=(
                (
                    args.linear_precision_audit
                    or args.directional_consistency_audit
                )
                and target_change
                == TARGET_SCALED_PRIMITIVE_CHANGES[-1]
            ),
            include_directional_consistency_audit=(
                args.directional_consistency_audit
                and target_change
                == TARGET_SCALED_PRIMITIVE_CHANGES[-1]
            ),
        )
        n16_attempts.append(attempt)
        if attempt["resolution_passed"]:
            break
    selected_n16 = n16_attempts[-1]
    linear_precision = selected_n16["tiny_step"][
        "linear_precision_audit"
    ]
    precision_rerun = None
    if (
        args.linear_precision_audit
        and linear_precision is not None
        and linear_precision["recoverable_precision_demonstrated"]
    ):
        precision_rerun = _run_resolution(
            16,
            selected_n16["tiny_step"][
                "target_scaled_primitive_change"
            ],
            temporal_storage_scheme=args.temporal_storage_scheme,
            linear_solver="dgesvx",
        )
        selected_n16 = precision_rerun
    n32_result = (
        _run_resolution(
            32,
            selected_n16["tiny_step"][
                "target_scaled_primitive_change"
            ],
            temporal_storage_scheme=args.temporal_storage_scheme,
        )
        if selected_n16["resolution_passed"]
        else None
    )
    all_passed = (
        selected_n16["resolution_passed"]
        and n32_result is not None
        and n32_result["resolution_passed"]
    )
    output = {
        "work_package": (
            "WP10c5g"
            if args.directional_consistency_audit
            else (
                "WP10c5f"
                if args.linear_precision_audit
                else (
                    "WP10c5d"
                    if args.temporal_storage_scheme == "endpoint"
                    else "WP10c5e"
                )
            )
        ),
        "scope": (
            "index-one consistent initial data and one tangent-sized "
            "backward-Euler step"
        ),
        "rank_relative_threshold": RANK_THRESHOLD,
        "finite_difference_step": FINITE_DIFFERENCE_STEP,
        "temporal_storage_scheme": args.temporal_storage_scheme,
        "n16_attempts": n16_attempts,
        "precision_rerun": precision_rerun,
        "n32_result": n32_result,
        "gates": {
            "n16_passed": selected_n16["resolution_passed"],
            "n32_attempted": n32_result is not None,
            "n32_passed": (
                n32_result["resolution_passed"]
                if n32_result is not None
                else False
            ),
            "early_time_numerical_gate_passed": all_passed,
            "physical_evolution_certified": False,
            "stability_certified": False,
            "tide_authorized": False,
            "wind_authorized": False,
        },
        "decision": (
            "consistent_initial_step_passed_n16_n32"
            if all_passed
            else "stop_before_physical_evolution"
        ),
    }
    output_path = _absolute(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        output,
        indent=2,
        sort_keys=True,
        default=_json_default,
    )
    output_path.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
