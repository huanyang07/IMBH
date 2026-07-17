"""Run the bounded WP10c5d consistent-data and tiny-step gate."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy.linalg.lapack import dgeequ, dgesvx
from scipy.sparse import issparse
from scipy.sparse.csgraph import structural_rank

from imri_qpe.constants import C, G
from imri_qpe.layer3_minidisk_1d import (
    KERR_SCHILD_HILL_ENERGY_ZERO,
    CausalFiveFieldAdaptiveRestart,
    CausalFiveFieldAdaptiveStepConfig,
    CausalFiveFieldDAEContext,
    GasRadiationHillRocheNozzleProvider,
    SchwarzschildCurvatureVerticalFrequency,
    ValenciaPerfectFluidPrimitive,
    audit_causal_five_field_consistent_initial_data,
    audit_causal_five_field_dae_jacobian,
    advance_causal_five_field_adaptive_backward_euler,
    causal_five_field_colored_central_jacobian,
    causal_five_field_dae_jacobian_color_groups,
    causal_five_field_dae_jacobian_sparsity,
    causal_five_field_dae_scaling,
    causal_five_field_equilibrated_sparse_solve,
    causal_five_field_h_over_r_profile,
    causal_five_field_loading_time,
    causal_five_field_state_summary,
    causal_five_field_endpoint_temporal_storage_increment,
    causal_five_field_path_temporal_storage_increment,
    causal_five_field_reduced_backward_euler_residual,
    causal_five_field_state_from_primitives,
    evaluate_causal_five_field_dae,
    evaluate_causal_five_field_increment_backward_euler,
    exact_kerr_schild_compact_stream_sources,
    fiducial_hill_roche_nozzle_geometry,
    kerr_schild_column_geometry,
    kerr_schild_stream_injection,
    load_causal_five_field_adaptive_restart,
    make_causal_five_field_seed,
    make_kerr_schild_column_grid,
    pack_causal_five_field_state,
    save_causal_five_field_adaptive_restart,
    unpack_causal_five_field_state,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/tables/causal_five_field_consistent_step_wp10c5d.json"
)
DEFAULT_SOURCE_ON_OUTPUT = (
    ROOT
    / "outputs/tables/causal_five_field_source_on_startup_wp10c5i.json"
)
DEFAULT_SPARSE_BACKEND_OUTPUT = (
    ROOT
    / "outputs/tables/causal_five_field_sparse_backend_wp10c5j.json"
)
DEFAULT_REPEATED_SOURCE_ON_OUTPUT = (
    ROOT
    / "outputs/tables/causal_five_field_repeated_source_on_wp10c5k.json"
)
DEFAULT_RESTART_DIRECTORY = (
    ROOT / "outputs/checkpoints/causal_five_field_wp10c5k"
)
RANK_THRESHOLD = 1.0e-11
FINITE_DIFFERENCE_STEP = 2.0e-6
TARGET_SCALED_PRIMITIVE_CHANGES = (1.0e-4, 1.0e-3)
STREAM_CENTER_RG = 240.0
STREAM_LOG_WIDTH = 0.08
STREAM_MDOT_EDD = 5.0
STREAM_SURFACE_DENSITY = 1.0e5
STREAM_TEMPERATURE = 1.0e6
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
    parser.add_argument(
        "--increment-primary-audit",
        action="store_true",
    )
    parser.add_argument(
        "--increment-primary-source-on-audit",
        action="store_true",
    )
    parser.add_argument(
        "--increment-primary-sparse-backend-audit",
        action="store_true",
    )
    parser.add_argument(
        "--increment-primary-repeated-source-on-audit",
        action="store_true",
    )
    return parser.parse_args()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _exact_regression_stream(
    context: CausalFiveFieldDAEContext,
    mass: float,
    gravitational_radius: float,
):
    radius = STREAM_CENTER_RG * gravitational_radius
    geometry = kerr_schild_column_geometry(
        radius,
        gravitational_radius,
    )
    thermodynamics = context.vertical_frequency.eos(
        radius
    ).from_surface_density_temperature(
        STREAM_SURFACE_DENSITY,
        STREAM_TEMPERATURE,
    )
    primitive = ValenciaPerfectFluidPrimitive(
        surface_density=STREAM_SURFACE_DENSITY,
        radial_velocity_over_c=(
            2.0 * gravitational_radius / radius
        ),
        azimuthal_velocity_over_c=float(
            np.sqrt(gravitational_radius / radius)
            / geometry.base.lapse
        ),
        specific_internal_energy=(
            thermodynamics.specific_internal_energy
        ),
        integrated_pressure=thermodynamics.integrated_pressure,
    )
    injection = kerr_schild_stream_injection(
        geometry,
        primitive,
        rest_mass_rate=STREAM_MDOT_EDD * eddington_mdot(mass),
    )
    return exact_kerr_schild_compact_stream_sources(
        context.grid,
        injection,
        center=radius,
        log_width=STREAM_LOG_WIDTH,
        shape="compact_c2",
    )


def _context(
    n_cells: int,
    *,
    include_stream: bool = False,
) -> CausalFiveFieldDAEContext:
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
    context = CausalFiveFieldDAEContext(
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
    if include_stream:
        context = replace(
            context,
            stream_sources=_exact_regression_stream(
                context,
                mass,
                gravitational_radius,
            ),
        ).validated()
    return context


def _stream_summary(
    context: CausalFiveFieldDAEContext,
) -> dict:
    source = context.stream_sources
    if source is None:
        return {
            "enabled": False,
            "source_role": "none",
        }
    totals = np.sum(source.matrix, axis=0)
    active = np.flatnonzero(source.rest_mass > 0.0)
    mass_rate = float(totals[0])
    expected_mass_rate = (
        STREAM_MDOT_EDD * eddington_mdot(FiducialParams().M2_g)
    )
    return {
        "enabled": True,
        "source_role": (
            "exact circularized regression stream; not a ballistic "
            "Layer-1 calibration"
        ),
        "shape": "compact_c2",
        "center_rg": STREAM_CENTER_RG,
        "log_width": STREAM_LOG_WIDTH,
        "supply_mdot_edd": STREAM_MDOT_EDD,
        "injection_surface_density_g_cm2": STREAM_SURFACE_DENSITY,
        "injection_temperature_k": STREAM_TEMPERATURE,
        "active_cell_count": int(active.size),
        "active_cell_indices": [int(value) for value in active],
        "rest_mass_rate_g_s": mass_rate,
        "specific_radial_momentum_over_c": float(
            totals[1] / mass_rate
        ),
        "specific_angular_momentum_over_c_cm": float(
            totals[2] / mass_rate
        ),
        "specific_killing_energy_over_c2": float(
            totals[3] / mass_rate
        ),
        "source_normalization_relative_defect": float(
            abs(mass_rate - expected_mass_rate)
            / expected_mass_rate
        ),
    }


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
) -> tuple[np.ndarray, dict, object | None, np.ndarray]:
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
        raw_matrix = jacobian(state)
        matrix = (
            raw_matrix.tocsr()
            if issparse(raw_matrix)
            else np.asarray(raw_matrix, dtype=float)
        )
        last_matrix = matrix
        jacobian_evaluations += 1
        try:
            if linear_solver == "direct":
                singular = np.linalg.svd(matrix, compute_uv=False)
                row["jacobian_condition_estimate"] = float(
                    singular[0]
                    / max(singular[-1], np.finfo(float).tiny)
                )
                correction = np.linalg.solve(matrix, -values)
            elif linear_solver == "dgesvx":
                singular = np.linalg.svd(matrix, compute_uv=False)
                row["jacobian_condition_estimate"] = float(
                    singular[0]
                    / max(singular[-1], np.finfo(float).tiny)
                )
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
            elif linear_solver == "equilibrated_splu":
                if not issparse(matrix):
                    raise ValueError(
                        "equilibrated_splu requires a sparse Jacobian"
                    )
                correction, linear_audit = (
                    causal_five_field_equilibrated_sparse_solve(
                        matrix,
                        -values,
                    )
                )
                row["jacobian_nonzeros"] = int(matrix.nnz)
                row["linear_solver_audit"] = asdict(linear_audit)
            else:
                raise ValueError("unknown reduced Newton linear solver")
        except (np.linalg.LinAlgError, RuntimeError):
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
    final_increment, solver, last_matrix, _last_values = _bounded_newton(
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


def _rank_summary(matrix: np.ndarray) -> dict:
    def summary(values: np.ndarray) -> dict:
        singular = np.linalg.svd(values, compute_uv=False)
        largest = float(singular[0])
        smallest = float(singular[-1])
        threshold = max(
            RANK_THRESHOLD * largest,
            np.finfo(float).eps * max(values.shape) * largest,
        )
        rank = int(np.sum(singular > threshold))
        return {
            "dimensions": [int(value) for value in values.shape],
            "numerical_rank": rank,
            "full_rank": rank == min(values.shape),
            "rank_threshold": threshold,
            "largest_singular_value": largest,
            "smallest_singular_value": smallest,
            "condition_estimate": float(
                largest / max(smallest, np.finfo(float).tiny)
            ),
            "smallest_six_singular_values": [
                float(value) for value in singular[-6:]
            ],
        }

    result = summary(matrix)
    (
        row_scale,
        column_scale,
        row_condition,
        column_condition,
        maximum_entry,
        info,
    ) = dgeequ(matrix)
    if int(info) != 0:
        raise RuntimeError(f"dgeequ rank audit failed with info={info}")
    equilibrated = (
        row_scale[:, None]
        * matrix
        * column_scale[None, :]
    )
    result["equilibration"] = {
        "row_scale_minimum": float(np.min(row_scale)),
        "row_scale_maximum": float(np.max(row_scale)),
        "column_scale_minimum": float(np.min(column_scale)),
        "column_scale_maximum": float(np.max(column_scale)),
        "row_condition": float(row_condition),
        "column_condition": float(column_condition),
        "maximum_matrix_entry": float(maximum_entry),
        "dgeequ_info": int(info),
        **summary(equilibrated),
    }
    return result


def _run_increment_primary_resolution(
    n_cells: int,
    target_scaled_primitive_change: float,
    *,
    include_stream: bool = False,
) -> tuple[dict, dict]:
    context = _context(n_cells, include_stream=include_stream)
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
    initial_increment = timestep * np.asarray(
        consistent.scaled_tangent,
        dtype=float,
    )

    def residual(scaled_increment: np.ndarray) -> np.ndarray:
        physical_increment = (
            scaling.column_scales
            * np.asarray(scaled_increment, dtype=float)
        )
        return (
            evaluate_causal_five_field_increment_backward_euler(
                physical_increment,
                context,
                old_vector=old_vector,
                timestep_seconds=timestep,
                temporal_height_scheme="path_integrated",
            ).residual
            / scaling.row_scales
        )

    def central_jacobian(scaled_increment: np.ndarray) -> np.ndarray:
        size = scaled_increment.size
        columns = np.empty((size, size), dtype=float)
        for index in range(size):
            plus = np.array(scaled_increment, copy=True)
            minus = np.array(scaled_increment, copy=True)
            plus[index] += FINITE_DIFFERENCE_STEP
            minus[index] -= FINITE_DIFFERENCE_STEP
            columns[:, index] = (
                residual(plus) - residual(minus)
            ) / (2.0 * FINITE_DIFFERENCE_STEP)
        return columns

    initial_jacobian = central_jacobian(initial_increment)
    initial_rank = _rank_summary(initial_jacobian)
    initial_matrix_available = True

    def jacobian(scaled_increment: np.ndarray) -> np.ndarray:
        nonlocal initial_matrix_available
        if (
            initial_matrix_available
            and np.array_equal(scaled_increment, initial_increment)
        ):
            initial_matrix_available = False
            return initial_jacobian
        return central_jacobian(scaled_increment)

    nonlinear_bound = 1.25 * target_scaled_primitive_change
    final_increment, solver, last_matrix, _last_values = _bounded_newton(
        residual,
        jacobian,
        initial_increment,
        bound=nonlinear_bound,
        residual_tolerance=1.0e-8,
        linear_solver="direct",
    )
    physical_increment = scaling.column_scales * final_increment
    new_state = unpack_causal_five_field_state(
        old_vector + physical_increment,
        n_cells,
    )
    evaluation = evaluate_causal_five_field_increment_backward_euler(
        physical_increment,
        context,
        old_vector=old_vector,
        timestep_seconds=timestep,
        temporal_height_scheme="path_integrated",
    )
    scaled_residual = evaluation.residual / scaling.row_scales
    scaled_conservation = scaled_residual[:n_differential].reshape(
        n_cells,
        5,
    )
    controlling_flat_index = int(np.argmax(np.abs(scaled_conservation)))
    controlling_cell, controlling_field = np.unravel_index(
        controlling_flat_index,
        scaled_conservation.shape,
    )
    algebraic_scaled_residual = scaled_residual[n_differential:]
    ledger_defect, component_ledger_defects = _ledger_defect(
        new_state,
        evaluation,
    )
    if last_matrix is None:
        last_matrix = initial_jacobian
    final_rank = _rank_summary(np.asarray(last_matrix, dtype=float))
    block_maxima = {
        "conserved": float(
            np.max(np.abs(final_increment[:n_differential]))
        ),
        "primitive": float(
            np.max(
                np.abs(
                    final_increment[
                        n_differential : 2 * n_differential
                    ]
                )
            )
        ),
        "face_flux": float(
            np.max(np.abs(final_increment[2 * n_differential :]))
        ),
    }
    maximum_change = float(np.max(np.abs(final_increment)))
    maximum_scaled_residual = float(np.max(np.abs(scaled_residual)))
    maximum_scaled_conservation_residual = float(
        np.max(np.abs(scaled_conservation))
    )
    maximum_scaled_algebraic_residual = float(
        np.max(np.abs(algebraic_scaled_residual))
    )
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
        and initial_rank["equilibration"]["full_rank"]
        and final_rank["equilibration"]["full_rank"]
        and maximum_scaled_residual <= 1.0e-8
        and maximum_scaled_algebraic_residual <= 1.0e-10
        and maximum_change <= nonlinear_bound
        and evaluation.outer_boundary_choked
        == stationary_evaluation.outer_boundary_choked
        and np.min(evaluation.scattering_optical_depths) > 1.0
        and ledger_defect <= 1.0e-10
    )
    report = {
        "n_cells": n_cells,
        "unknown_count": int(final_increment.size),
        "residual_count": int(scaled_residual.size),
        "coordinate": "primary physical increments",
        "temporal_height_scheme": "path_integrated",
        "stream": _stream_summary(context),
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
            "tangent_predictor_maximum_scaled_change": float(
                np.max(np.abs(initial_increment))
            ),
            "maximum_scaled_change": maximum_change,
            "maximum_scaled_block_changes": block_maxima,
            "solver_success": solver["success"],
            "solver_message": solver["message"],
            "solver_iterations": solver["iterations"],
            "function_evaluations": solver["function_evaluations"],
            "jacobian_evaluations": solver["jacobian_evaluations"],
            "solver_history": solver["history"],
            "initial_jacobian": initial_rank,
            "final_newton_jacobian": final_rank,
            "maximum_scaled_residual": maximum_scaled_residual,
            "maximum_scaled_conservation_residual": (
                maximum_scaled_conservation_residual
            ),
            "maximum_scaled_algebraic_residual": (
                maximum_scaled_algebraic_residual
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
            "passed": step_passed,
        },
        "resolution_passed": consistency_passed and step_passed,
    }
    artifacts = {
        "context": context,
        "old_vector": old_vector,
        "scaling": scaling,
        "initial_scaled_increment": initial_increment,
        "initial_dense_jacobian": initial_jacobian,
        "timestep_seconds": timestep,
        "physical_increment": physical_increment,
        "final_scaled_increment": final_increment,
        "final_dense_jacobian": last_matrix,
        "new_vector": old_vector + physical_increment,
    }
    return report, artifacts


def _solve_increment_primary_substep(
    context: CausalFiveFieldDAEContext,
    old_vector: np.ndarray,
    *,
    timestep_seconds: float,
    initial_physical_increment: np.ndarray,
    scaled_change_bound: float,
) -> tuple[dict, dict]:
    n_cells = int(context.grid.centers.size)
    n_differential = 5 * n_cells
    old_state = unpack_causal_five_field_state(old_vector, n_cells)
    stationary_evaluation = evaluate_causal_five_field_dae(
        old_vector,
        context,
    )
    scaling = causal_five_field_dae_scaling(
        old_state,
        stationary_evaluation,
    )
    initial_scaled_increment = (
        np.asarray(initial_physical_increment, dtype=float)
        / scaling.column_scales
    )

    def residual(scaled_increment: np.ndarray) -> np.ndarray:
        physical_increment = (
            scaling.column_scales
            * np.asarray(scaled_increment, dtype=float)
        )
        return (
            evaluate_causal_five_field_increment_backward_euler(
                physical_increment,
                context,
                old_vector=old_vector,
                timestep_seconds=timestep_seconds,
                temporal_height_scheme="path_integrated",
            ).residual
            / scaling.row_scales
        )

    def central_jacobian(scaled_increment: np.ndarray) -> np.ndarray:
        size = scaled_increment.size
        columns = np.empty((size, size), dtype=float)
        for index in range(size):
            plus = np.array(scaled_increment, copy=True)
            minus = np.array(scaled_increment, copy=True)
            plus[index] += FINITE_DIFFERENCE_STEP
            minus[index] -= FINITE_DIFFERENCE_STEP
            columns[:, index] = (
                residual(plus) - residual(minus)
            ) / (2.0 * FINITE_DIFFERENCE_STEP)
        return columns

    initial_jacobian = central_jacobian(initial_scaled_increment)
    initial_rank = _rank_summary(initial_jacobian)
    initial_matrix_available = True

    def jacobian(scaled_increment: np.ndarray) -> np.ndarray:
        nonlocal initial_matrix_available
        if (
            initial_matrix_available
            and np.array_equal(
                scaled_increment,
                initial_scaled_increment,
            )
        ):
            initial_matrix_available = False
            return initial_jacobian
        return central_jacobian(scaled_increment)

    (
        final_scaled_increment,
        solver,
        last_matrix,
        _last_values,
    ) = _bounded_newton(
        residual,
        jacobian,
        initial_scaled_increment,
        bound=scaled_change_bound,
        residual_tolerance=1.0e-8,
        linear_solver="direct",
    )
    physical_increment = (
        scaling.column_scales * final_scaled_increment
    )
    new_vector = old_vector + physical_increment
    new_state = unpack_causal_five_field_state(
        new_vector,
        n_cells,
    )
    evaluation = evaluate_causal_five_field_increment_backward_euler(
        physical_increment,
        context,
        old_vector=old_vector,
        timestep_seconds=timestep_seconds,
        temporal_height_scheme="path_integrated",
    )
    scaled_residual = evaluation.residual / scaling.row_scales
    algebraic_scaled_residual = scaled_residual[n_differential:]
    ledger_defect, component_ledger_defects = _ledger_defect(
        new_state,
        evaluation,
    )
    if last_matrix is None:
        last_matrix = initial_jacobian
    final_rank = _rank_summary(np.asarray(last_matrix, dtype=float))
    maximum_change = float(
        np.max(np.abs(final_scaled_increment))
    )
    maximum_scaled_residual = float(
        np.max(np.abs(scaled_residual))
    )
    maximum_scaled_algebraic_residual = float(
        np.max(np.abs(algebraic_scaled_residual))
    )
    passed = (
        solver["success"]
        and initial_rank["equilibration"]["full_rank"]
        and final_rank["equilibration"]["full_rank"]
        and maximum_scaled_residual <= 1.0e-8
        and maximum_scaled_algebraic_residual <= 1.0e-10
        and maximum_change <= scaled_change_bound
        and evaluation.outer_boundary_choked
        == stationary_evaluation.outer_boundary_choked
        and np.min(evaluation.scattering_optical_depths) > 1.0
        and ledger_defect <= 1.0e-10
    )
    report = {
        "timestep_seconds": timestep_seconds,
        "scaled_change_bound": scaled_change_bound,
        "predictor_maximum_scaled_change": float(
            np.max(np.abs(initial_scaled_increment))
        ),
        "maximum_scaled_change": maximum_change,
        "solver_success": solver["success"],
        "solver_message": solver["message"],
        "solver_iterations": solver["iterations"],
        "function_evaluations": solver["function_evaluations"],
        "jacobian_evaluations": solver["jacobian_evaluations"],
        "solver_history": solver["history"],
        "initial_jacobian": initial_rank,
        "final_newton_jacobian": final_rank,
        "maximum_scaled_residual": maximum_scaled_residual,
        "maximum_scaled_algebraic_residual": (
            maximum_scaled_algebraic_residual
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
        "passed": passed,
    }
    artifacts = {
        "scaling": scaling,
        "physical_increment": physical_increment,
        "new_vector": new_vector,
    }
    return report, artifacts


def _temporal_refinement_comparison(
    full_step_artifacts: dict,
    target_scaled_change: float,
) -> dict:
    context = full_step_artifacts["context"]
    old_vector = np.asarray(
        full_step_artifacts["old_vector"],
        dtype=float,
    )
    base_scaling = full_step_artifacts["scaling"]
    timestep = float(full_step_artifacts["timestep_seconds"])
    predictor = np.asarray(
        full_step_artifacts["initial_scaled_increment"],
        dtype=float,
    )
    half_bound = 0.75 * target_scaled_change
    first_report, first_artifacts = _solve_increment_primary_substep(
        context,
        old_vector,
        timestep_seconds=0.5 * timestep,
        initial_physical_increment=(
            0.5 * base_scaling.column_scales * predictor
        ),
        scaled_change_bound=half_bound,
    )
    if not first_report["passed"]:
        return {
            "n_cells": int(context.grid.centers.size),
            "full_timestep_seconds": timestep,
            "relative_error_gate": 0.05,
            "first_half": first_report,
            "second_half": None,
            "passed": False,
            "decision": "first_half_step_failed",
        }
    second_report, second_artifacts = _solve_increment_primary_substep(
        context,
        first_artifacts["new_vector"],
        timestep_seconds=0.5 * timestep,
        initial_physical_increment=(
            first_artifacts["physical_increment"]
        ),
        scaled_change_bound=half_bound,
    )
    full_new_vector = np.asarray(
        full_step_artifacts["new_vector"],
        dtype=float,
    )
    two_half_new_vector = np.asarray(
        second_artifacts["new_vector"],
        dtype=float,
    )
    scaled_difference = (
        two_half_new_vector - full_new_vector
    ) / base_scaling.column_scales
    scaled_full_change = (
        full_new_vector - old_vector
    ) / base_scaling.column_scales
    n_cells = int(context.grid.centers.size)
    n_differential = 5 * n_cells
    block_slices = {
        "conserved": slice(0, n_differential),
        "primitive": slice(
            n_differential,
            2 * n_differential,
        ),
        "face_flux": slice(2 * n_differential, None),
    }
    block_errors = {
        name: float(np.max(np.abs(scaled_difference[block])))
        for name, block in block_slices.items()
    }
    maximum_error = float(np.max(np.abs(scaled_difference)))
    maximum_full_change = float(
        np.max(np.abs(scaled_full_change))
    )
    relative_error = float(
        maximum_error
        / max(maximum_full_change, np.finfo(float).tiny)
    )
    relative_error_gate = 0.05
    passed = (
        first_report["passed"]
        and second_report["passed"]
        and relative_error <= relative_error_gate
    )
    return {
        "n_cells": n_cells,
        "full_timestep_seconds": timestep,
        "half_timestep_seconds": 0.5 * timestep,
        "maximum_scaled_full_step_change": maximum_full_change,
        "maximum_scaled_full_vs_two_half_error": maximum_error,
        "maximum_scaled_block_errors": block_errors,
        "relative_full_vs_two_half_error": relative_error,
        "relative_error_gate": relative_error_gate,
        "first_half": first_report,
        "second_half": second_report,
        "passed": passed,
        "decision": (
            "temporal_refinement_gate_passed"
            if passed
            else "temporal_refinement_gate_failed"
        ),
    }


def _increment_primary_residual_from_artifacts(artifacts: dict):
    context = artifacts["context"]
    old_vector = np.asarray(artifacts["old_vector"], dtype=float)
    scaling = artifacts["scaling"]
    timestep = float(artifacts["timestep_seconds"])

    def residual(scaled_increment: np.ndarray) -> np.ndarray:
        physical_increment = (
            scaling.column_scales
            * np.asarray(scaled_increment, dtype=float)
        )
        return (
            evaluate_causal_five_field_increment_backward_euler(
                physical_increment,
                context,
                old_vector=old_vector,
                timestep_seconds=timestep,
                temporal_height_scheme="path_integrated",
            ).residual
            / scaling.row_scales
        )

    return residual


def _sparse_matrix_parity(
    residual,
    values: np.ndarray,
    dense_matrix: np.ndarray,
    sparse_matrix,
    pattern,
) -> dict:
    dense = np.asarray(dense_matrix, dtype=float)
    sparse = sparse_matrix.toarray()
    allowed = pattern.toarray().astype(bool)
    row_scale = np.maximum(
        np.max(np.abs(dense), axis=1),
        1.0e-14,
    )
    omitted = np.where(allowed, 0.0, dense)
    matrix_difference = sparse - dense
    maximum_directional_defect = 0.0
    coordinates = np.arange(values.size, dtype=float) + 1.0
    directional_step = 5.0e-7
    for index in range(3):
        direction = np.sin((index + 1.0) * coordinates)
        direction += 0.5 * np.cos((index + 2.0) * coordinates)
        direction /= max(np.max(np.abs(direction)), 1.0)
        finite_difference = (
            residual(values + directional_step * direction)
            - residual(values - directional_step * direction)
        ) / (2.0 * directional_step)
        product = sparse_matrix @ direction
        scale = max(
            np.max(np.abs(finite_difference)),
            np.max(np.abs(product)),
            1.0e-14,
        )
        maximum_directional_defect = max(
            maximum_directional_defect,
            float(
                np.max(np.abs(product - finite_difference)) / scale
            ),
        )
    groups = causal_five_field_dae_jacobian_color_groups(pattern)
    return {
        "dimensions": list(dense.shape),
        "pattern_nonzeros": int(pattern.nnz),
        "pattern_density": float(pattern.nnz / dense.size),
        "structural_rank": int(structural_rank(pattern)),
        "color_count": len(groups),
        "maximum_color_size": int(max(len(group) for group in groups)),
        "dense_central_residual_evaluations_per_jacobian": int(
            2 * values.size
        ),
        "colored_central_residual_evaluations_per_jacobian": int(
            2 * len(groups)
        ),
        "assembly_evaluation_reduction_factor": float(
            values.size / len(groups)
        ),
        "maximum_omitted_absolute_derivative": float(
            np.max(np.abs(omitted))
        ),
        "maximum_omitted_row_relative_derivative": float(
            np.max(np.abs(omitted) / row_scale[:, None])
        ),
        "maximum_colored_absolute_matrix_defect": float(
            np.max(np.abs(matrix_difference))
        ),
        "maximum_colored_row_relative_matrix_defect": float(
            np.max(
                np.abs(matrix_difference) / row_scale[:, None]
            )
        ),
        "maximum_directional_relative_defect": (
            maximum_directional_defect
        ),
    }


def _run_sparse_backend_resolution(
    n_cells: int,
    target_scaled_primitive_change: float,
) -> tuple[dict, bool]:
    dense_report, artifacts = _run_increment_primary_resolution(
        n_cells,
        target_scaled_primitive_change,
        include_stream=True,
    )
    if not dense_report["resolution_passed"]:
        return {
            "n_cells": n_cells,
            "dense_reference_passed": False,
            "sparse_attempted": False,
            "passed": False,
            "decision": "dense_reference_failed",
        }, False

    context = artifacts["context"]
    old_vector = np.asarray(artifacts["old_vector"], dtype=float)
    scaling = artifacts["scaling"]
    residual = _increment_primary_residual_from_artifacts(artifacts)
    initial = np.asarray(
        artifacts["initial_scaled_increment"],
        dtype=float,
    )
    dense_matrix = np.asarray(
        artifacts["initial_dense_jacobian"],
        dtype=float,
    )
    pattern = causal_five_field_dae_jacobian_sparsity(n_cells)
    initial_sparse = causal_five_field_colored_central_jacobian(
        residual,
        initial,
        pattern,
        finite_difference_step=FINITE_DIFFERENCE_STEP,
    )
    matrix_parity = _sparse_matrix_parity(
        residual,
        initial,
        dense_matrix,
        initial_sparse,
        pattern,
    )
    initial_rank = _rank_summary(initial_sparse.toarray())
    initial_matrix_available = True

    def jacobian(scaled_increment: np.ndarray):
        nonlocal initial_matrix_available
        if (
            initial_matrix_available
            and np.array_equal(scaled_increment, initial)
        ):
            initial_matrix_available = False
            return initial_sparse
        return causal_five_field_colored_central_jacobian(
            residual,
            scaled_increment,
            pattern,
            finite_difference_step=FINITE_DIFFERENCE_STEP,
        )

    nonlinear_bound = 1.25 * target_scaled_primitive_change
    (
        sparse_increment,
        solver,
        last_matrix,
        _last_values,
    ) = _bounded_newton(
        residual,
        jacobian,
        initial,
        bound=nonlinear_bound,
        residual_tolerance=1.0e-8,
        linear_solver="equilibrated_splu",
    )
    if last_matrix is None:
        last_matrix = initial_sparse
    final_rank = _rank_summary(last_matrix.toarray())
    physical_increment = scaling.column_scales * sparse_increment
    new_vector = old_vector + physical_increment
    new_state = unpack_causal_five_field_state(
        new_vector,
        n_cells,
    )
    evaluation = evaluate_causal_five_field_increment_backward_euler(
        physical_increment,
        context,
        old_vector=old_vector,
        timestep_seconds=float(artifacts["timestep_seconds"]),
        temporal_height_scheme="path_integrated",
    )
    scaled_residual = evaluation.residual / scaling.row_scales
    n_differential = 5 * n_cells
    maximum_residual = float(np.max(np.abs(scaled_residual)))
    maximum_algebraic_residual = float(
        np.max(np.abs(scaled_residual[n_differential:]))
    )
    ledger_defect, component_ledger_defects = _ledger_defect(
        new_state,
        evaluation,
    )
    dense_increment = np.asarray(
        artifacts["final_scaled_increment"],
        dtype=float,
    )
    maximum_root_defect = float(
        np.max(np.abs(sparse_increment - dense_increment))
    )
    relative_root_defect = float(
        maximum_root_defect
        / max(
            np.max(np.abs(dense_increment)),
            np.finfo(float).tiny,
        )
    )
    matrix_passed = (
        matrix_parity[
            "maximum_omitted_row_relative_derivative"
        ]
        <= 1.0e-10
        and matrix_parity[
            "maximum_colored_row_relative_matrix_defect"
        ]
        <= 1.0e-10
        and matrix_parity["maximum_directional_relative_defect"]
        <= 1.0e-6
        and matrix_parity["structural_rank"] == initial.size
        and initial_rank["equilibration"]["full_rank"]
        and final_rank["equilibration"]["full_rank"]
    )
    root_passed = (
        solver["success"]
        and maximum_residual <= 1.0e-8
        and maximum_algebraic_residual <= 1.0e-10
        and relative_root_defect <= 1.0e-5
        and ledger_defect <= 1.0e-10
        and np.min(evaluation.scattering_optical_depths) > 1.0
    )
    passed = matrix_passed and root_passed
    report = {
        "n_cells": n_cells,
        "unknown_count": int(initial.size),
        "stream": _stream_summary(context),
        "dense_reference": {
            "passed": dense_report["resolution_passed"],
            "timestep_seconds": dense_report["tiny_step"][
                "timestep_seconds"
            ],
            "maximum_scaled_residual": dense_report["tiny_step"][
                "maximum_scaled_residual"
            ],
            "maximum_scaled_algebraic_residual": dense_report[
                "tiny_step"
            ]["maximum_scaled_algebraic_residual"],
        },
        "matrix_parity": matrix_parity,
        "initial_sparse_jacobian": initial_rank,
        "final_sparse_jacobian": final_rank,
        "sparse_root": {
            "solver": solver,
            "maximum_scaled_residual": maximum_residual,
            "maximum_scaled_algebraic_residual": (
                maximum_algebraic_residual
            ),
            "maximum_scaled_dense_root_defect": maximum_root_defect,
            "relative_scaled_dense_root_defect": relative_root_defect,
            "conservation_telescoping_relative_defect": ledger_defect,
            "component_conservation_defects": (
                component_ledger_defects
            ),
            "minimum_scattering_optical_depth": float(
                np.min(evaluation.scattering_optical_depths)
            ),
        },
        "matrix_gate_passed": matrix_passed,
        "root_gate_passed": root_passed,
        "passed": passed,
        "decision": (
            "sparse_backend_parity_passed"
            if passed
            else "sparse_backend_parity_failed"
        ),
    }
    return report, passed


def _run_sparse_backend_audit(args: argparse.Namespace) -> None:
    n16, n16_passed = _run_sparse_backend_resolution(
        16,
        TARGET_SCALED_PRIMITIVE_CHANGES[0],
    )
    n32 = None
    n32_passed = False
    if n16_passed:
        n32, n32_passed = _run_sparse_backend_resolution(
            32,
            TARGET_SCALED_PRIMITIVE_CHANGES[0],
        )
    passed = n16_passed and n32 is not None and n32_passed
    output = {
        "work_package": "WP10c5j",
        "scope": (
            "dense-certified local colored-central Jacobian with "
            "max-norm-equilibrated sparse LU"
        ),
        "finite_difference_step": FINITE_DIFFERENCE_STEP,
        "n16": n16,
        "n32": n32,
        "gates": {
            "n16_sparse_parity_passed": n16_passed,
            "n32_attempted": n32 is not None,
            "n32_sparse_parity_passed": n32_passed,
            "practical_backend_certified": passed,
            "repeated_source_on_evolution_authorized": passed,
            "tide_authorized": False,
            "wind_authorized": False,
        },
        "decision": (
            "practical_sparse_backend_certified"
            if passed
            else "stop_before_repeated_evolution"
        ),
    }
    output_path = _absolute(
        DEFAULT_SPARSE_BACKEND_OUTPUT
        if args.output == DEFAULT_OUTPUT
        else args.output
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        output,
        indent=2,
        sort_keys=True,
        default=_json_default,
    )
    output_path.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


def _adaptive_step_row(
    accepted_step: int,
    elapsed_time: float,
    result,
) -> dict:
    step = result.step
    return {
        "accepted_step": accepted_step,
        "elapsed_time_seconds": elapsed_time,
        "dt_used_seconds": result.dt_used,
        "dt_next_seconds": result.dt_next,
        "attempts": [asdict(attempt) for attempt in result.attempts],
        "maximum_scaled_residual": step.maximum_scaled_residual,
        "maximum_scaled_algebraic_residual": (
            step.maximum_scaled_algebraic_residual
        ),
        "maximum_scaled_primitive_change": (
            step.maximum_scaled_primitive_change
        ),
        "maximum_scaled_total_change": (
            step.maximum_scaled_total_change
        ),
        "conservation_telescoping_relative_defect": (
            step.conservation_telescoping_relative_defect
        ),
        "minimum_scattering_optical_depth": (
            step.minimum_scattering_optical_depth
        ),
        "outer_boundary_choked": step.outer_boundary_choked_after,
        "iterations": step.iterations,
        "function_evaluations": step.function_evaluations,
        "jacobian_evaluations": step.jacobian_evaluations,
        "maximum_linear_residual": step.maximum_linear_residual,
        "jacobian_nonzeros": step.jacobian_nonzeros,
        "jacobian_color_count": step.jacobian_color_count,
    }


def _integrated_rest_mass_increment(
    context: CausalFiveFieldDAEContext,
    physical_increment: np.ndarray,
) -> float:
    n_cells = int(context.grid.centers.size)
    count = 15 * n_cells + 5
    increment = np.asarray(
        physical_increment,
        dtype=float,
    )
    if increment.shape != (count,) or np.any(~np.isfinite(increment)):
        raise ValueError("physical increment has the wrong shape or value")
    conserved_increment = increment[: 5 * n_cells].reshape(
        n_cells,
        5,
    )
    weighted = (
        np.asarray(context.grid.cell_measures, dtype=np.longdouble)
        * np.asarray(
            conserved_increment[:, 0],
            dtype=np.longdouble,
        )
    )
    return float(np.sum(weighted, dtype=np.longdouble))


def _reconstructed_log_h_over_r(
    context: CausalFiveFieldDAEContext,
    vector: np.ndarray,
    sample_log_radius: np.ndarray,
) -> np.ndarray:
    radius = np.asarray(context.grid.centers, dtype=float)
    h_over_r = causal_five_field_h_over_r_profile(context, vector)
    log_radius = np.log(radius)
    log_h_over_r = np.log(h_over_r)
    reconstructed = np.interp(
        sample_log_radius,
        log_radius,
        log_h_over_r,
    )
    left = sample_log_radius < log_radius[0]
    right = sample_log_radius > log_radius[-1]
    reconstructed[left] = (
        log_h_over_r[0]
        + (
            (log_h_over_r[1] - log_h_over_r[0])
            / (log_radius[1] - log_radius[0])
        )
        * (sample_log_radius[left] - log_radius[0])
    )
    reconstructed[right] = (
        log_h_over_r[-1]
        + (
            (log_h_over_r[-1] - log_h_over_r[-2])
            / (log_radius[-1] - log_radius[-2])
        )
        * (sample_log_radius[right] - log_radius[-1])
    )
    return reconstructed


def _reconstructed_h_over_r_summary(
    context: CausalFiveFieldDAEContext,
    vector: np.ndarray,
) -> dict:
    h_over_r = causal_five_field_h_over_r_profile(context, vector)
    sample_log_radius = np.linspace(
        np.log(float(context.grid.edges[0])),
        np.log(float(context.grid.edges[-1])),
        1025,
    )
    reconstructed = np.exp(
        _reconstructed_log_h_over_r(
            context,
            vector,
            sample_log_radius,
        )
    )
    maximum_index = int(np.argmax(reconstructed))
    return {
        "raw_cell_center_maximum": float(np.max(h_over_r)),
        "reconstructed_common_domain_maximum": float(
            reconstructed[maximum_index]
        ),
        "reconstructed_maximum_radius_rg": float(
            np.exp(sample_log_radius[maximum_index])
            / context.grid.gravitational_radius
        ),
    }


def _h_over_r_response_summary(
    context: CausalFiveFieldDAEContext,
    initial_vector: np.ndarray,
    final_vector: np.ndarray,
) -> dict:
    sample_log_radius = np.linspace(
        np.log(float(context.grid.edges[0])),
        np.log(float(context.grid.edges[-1])),
        129,
    )
    initial = _reconstructed_log_h_over_r(
        context,
        initial_vector,
        sample_log_radius,
    )
    final = _reconstructed_log_h_over_r(
        context,
        final_vector,
        sample_log_radius,
    )
    response = final - initial
    return {
        "method": (
            "baseline-subtracted Delta log(H/R) from log-linear "
            "cell-center reconstruction and one-cell edge extrapolation "
            "on a shared 129-point log-radius grid"
        ),
        "sample_radius_rg": [
            float(np.exp(value) / context.grid.gravitational_radius)
            for value in sample_log_radius
        ],
        "delta_log_h_over_r": [float(value) for value in response],
        "maximum_absolute_delta_log_h_over_r": float(
            np.max(np.abs(response))
        ),
        "rms_delta_log_h_over_r": float(
            np.sqrt(np.mean(response**2))
        ),
    }


def _restart_payload(
    state_vector: np.ndarray,
    previous_increment: np.ndarray,
    *,
    elapsed_time: float,
    dt_next: float,
    previous_dt: float,
    accepted_steps: int,
    rejected_attempts: int,
    n_cells: int,
    role: str,
) -> CausalFiveFieldAdaptiveRestart:
    return CausalFiveFieldAdaptiveRestart(
        state_vector=np.asarray(state_vector, dtype=float),
        previous_physical_increment=np.asarray(
            previous_increment,
            dtype=float,
        ),
        elapsed_time=elapsed_time,
        dt_next=dt_next,
        previous_dt=previous_dt,
        accepted_steps=accepted_steps,
        rejected_attempts=rejected_attempts,
        provenance={
            "work_package": "WP10c5k",
            "n_cells": n_cells,
            "role": role,
            "source": (
                "exact circularized regression stream; not ballistic "
                "Layer-1 calibration"
            ),
        },
    )


def _restart_roundtrip_is_bitwise(
    original: CausalFiveFieldAdaptiveRestart,
    restored: CausalFiveFieldAdaptiveRestart,
) -> bool:
    return bool(
        np.array_equal(
            original.state_vector,
            restored.state_vector,
        )
        and np.array_equal(
            original.previous_physical_increment,
            restored.previous_physical_increment,
        )
        and original.elapsed_time == restored.elapsed_time
        and original.dt_next == restored.dt_next
        and original.previous_dt == restored.previous_dt
        and original.accepted_steps == restored.accepted_steps
        and original.rejected_attempts == restored.rejected_attempts
        and original.provenance == restored.provenance
    )


def _adaptive_results_are_bitwise(left, right) -> bool:
    scalar_step_fields = (
        "accepted",
        "timestep_seconds",
        "maximum_scaled_residual",
        "maximum_scaled_algebraic_residual",
        "maximum_scaled_primitive_change",
        "maximum_scaled_total_change",
        "conservation_telescoping_relative_defect",
        "component_conservation_defects",
        "minimum_scattering_optical_depth",
        "outer_boundary_choked_before",
        "outer_boundary_choked_after",
        "iterations",
        "function_evaluations",
        "jacobian_evaluations",
        "maximum_linear_residual",
        "jacobian_nonzeros",
        "jacobian_color_count",
        "message",
    )
    return bool(
        left.accepted == right.accepted
        and left.dt_used == right.dt_used
        and left.dt_next == right.dt_next
        and left.message == right.message
        and np.array_equal(left.state_vector, right.state_vector)
        and np.array_equal(
            left.physical_increment,
            right.physical_increment,
        )
        and np.array_equal(
            left.step.state_vector,
            right.step.state_vector,
        )
        and np.array_equal(
            left.step.physical_increment,
            right.step.physical_increment,
        )
        and all(
            getattr(left.step, field) == getattr(right.step, field)
            for field in scalar_step_fields
        )
        and left.attempts == right.attempts
    )


def _run_repeated_source_on_resolution(
    n_cells: int,
    *,
    accepted_step_target: int | None,
    elapsed_time_target: float | None,
    perform_restart_resume_audit: bool,
) -> tuple[dict, bool]:
    initialization, artifacts = _run_increment_primary_resolution(
        n_cells,
        TARGET_SCALED_PRIMITIVE_CHANGES[0],
        include_stream=True,
    )
    if not initialization["resolution_passed"]:
        return {
            "n_cells": n_cells,
            "initialization_passed": False,
            "passed": False,
            "decision": "source_on_initialization_failed",
        }, False
    if (accepted_step_target is None) == (elapsed_time_target is None):
        raise ValueError(
            "repeated run requires exactly one duration target"
        )

    context = artifacts["context"]
    initial_vector = np.asarray(artifacts["old_vector"], dtype=float)
    state_vector = np.asarray(artifacts["new_vector"], dtype=float)
    previous_increment = np.asarray(
        artifacts["physical_increment"],
        dtype=float,
    )
    previous_dt = float(artifacts["timestep_seconds"])
    elapsed_time = previous_dt
    dt_next = 1.5 * previous_dt
    accepted_steps = 1
    rejected_attempts = 0
    config = CausalFiveFieldAdaptiveStepConfig(
        minimum_dt=previous_dt / 128.0,
        maximum_dt=16.0 * previous_dt,
        maximum_scaled_primitive_change=5.0e-4,
        maximum_scaled_total_change=1.0e-3,
        shrink_factor=0.5,
        growth_factor=1.5,
        maximum_retries=6,
        easy_iterations=3,
        residual_tolerance=1.0e-8,
        algebraic_residual_tolerance=1.0e-10,
        conservation_tolerance=1.0e-10,
        finite_difference_step=FINITE_DIFFERENCE_STEP,
        maximum_newton_iterations=12,
    ).validated()
    if context.stream_sources is None:
        raise RuntimeError("repeated source-on run has no stream")
    source_rate = float(np.sum(context.stream_sources.rest_mass))
    initial_summary = causal_five_field_state_summary(
        context,
        initial_vector,
    )
    initial_h_over_r = _reconstructed_h_over_r_summary(
        context,
        initial_vector,
    )
    first_summary = causal_five_field_state_summary(
        context,
        state_vector,
    )
    loading_time = causal_five_field_loading_time(
        context,
        initial_vector,
    )
    actual_mass_increments = [
        _integrated_rest_mass_increment(
            context,
            artifacts["physical_increment"],
        )
    ]
    expected_mass_increments = [
        previous_dt
        * (
            source_rate
            + first_summary["inner_face_rates"][0]
            - first_summary["outer_face_rates"][0]
        )
    ]
    step_rows = [
        {
            "accepted_step": 1,
            "elapsed_time_seconds": elapsed_time,
            "dt_used_seconds": previous_dt,
            "dt_next_seconds": dt_next,
            "attempts": [],
            "maximum_scaled_residual": initialization["tiny_step"][
                "maximum_scaled_residual"
            ],
            "maximum_scaled_algebraic_residual": initialization[
                "tiny_step"
            ]["maximum_scaled_algebraic_residual"],
            "maximum_scaled_primitive_change": initialization[
                "tiny_step"
            ]["maximum_scaled_block_changes"]["primitive"],
            "maximum_scaled_total_change": initialization["tiny_step"][
                "maximum_scaled_change"
            ],
            "conservation_telescoping_relative_defect": initialization[
                "tiny_step"
            ]["conservation_telescoping_relative_defect"],
            "minimum_scattering_optical_depth": initialization[
                "tiny_step"
            ]["minimum_scattering_optical_depth"],
            "outer_boundary_choked": initialization["tiny_step"][
                "outer_boundary_choked_after"
            ],
            "iterations": initialization["tiny_step"][
                "solver_iterations"
            ],
            "function_evaluations": initialization["tiny_step"][
                "function_evaluations"
            ],
            "jacobian_evaluations": initialization["tiny_step"][
                "jacobian_evaluations"
            ],
            "backend": "dense_reference_startup",
        }
    ]
    restart_directory = DEFAULT_RESTART_DIRECTORY
    restart_directory.mkdir(parents=True, exist_ok=True)
    midpoint_path = (
        restart_directory
        / f"causal_wp10c5k_N{n_cells:03d}_midpoint.npz"
    )
    final_path = (
        restart_directory
        / f"causal_wp10c5k_N{n_cells:03d}_final.npz"
    )
    restart_roundtrip_bitwise = not perform_restart_resume_audit
    restart_resume_step_bitwise = not perform_restart_resume_audit
    restart_audited = False
    terminal_message = "target reached"
    target_tolerance = (
        0.0
        if elapsed_time_target is None
        else max(1.0e-20, 5.0e-14 * elapsed_time_target)
    )

    while True:
        if accepted_step_target is not None:
            if accepted_steps >= accepted_step_target:
                break
            requested_dt = dt_next
        else:
            assert elapsed_time_target is not None
            remaining = elapsed_time_target - elapsed_time
            if abs(remaining) <= target_tolerance:
                break
            if remaining <= 0.0:
                terminal_message = "elapsed-time target overshot"
                break
            requested_dt = min(dt_next, remaining)
        local_config = config
        if requested_dt < config.minimum_dt:
            local_config = replace(
                config,
                minimum_dt=requested_dt,
            ).validated()

        if (
            perform_restart_resume_audit
            and not restart_audited
            and accepted_steps == 4
        ):
            midpoint = _restart_payload(
                state_vector,
                previous_increment,
                elapsed_time=elapsed_time,
                dt_next=dt_next,
                previous_dt=previous_dt,
                accepted_steps=accepted_steps,
                rejected_attempts=rejected_attempts,
                n_cells=n_cells,
                role="midpoint_restart_resume_audit",
            )
            save_causal_five_field_adaptive_restart(
                midpoint_path,
                context,
                midpoint,
            )
            restored = load_causal_five_field_adaptive_restart(
                midpoint_path,
                context,
            )
            restart_roundtrip_bitwise = (
                _restart_roundtrip_is_bitwise(midpoint, restored)
            )
            original_result = (
                advance_causal_five_field_adaptive_backward_euler(
                    context,
                    midpoint.state_vector,
                    requested_dt,
                    midpoint.previous_physical_increment,
                    midpoint.previous_dt,
                    local_config,
                )
            )
            result = advance_causal_five_field_adaptive_backward_euler(
                context,
                restored.state_vector,
                requested_dt,
                restored.previous_physical_increment,
                restored.previous_dt,
                local_config,
            )
            restart_resume_step_bitwise = (
                _adaptive_results_are_bitwise(
                    original_result,
                    result,
                )
            )
            restart_audited = True
            if (
                not restart_roundtrip_bitwise
                or not restart_resume_step_bitwise
            ):
                terminal_message = "restart parity gate failed"
                break
        else:
            result = advance_causal_five_field_adaptive_backward_euler(
                context,
                state_vector,
                requested_dt,
                previous_increment,
                previous_dt,
                local_config,
            )

        rejected_attempts += max(0, len(result.attempts) - 1)
        if not result.accepted:
            terminal_message = result.message
            break
        state_vector = np.asarray(result.state_vector, dtype=float)
        previous_increment = np.asarray(
            result.physical_increment,
            dtype=float,
        )
        previous_dt = result.dt_used
        dt_next = result.dt_next
        elapsed_time += result.dt_used
        accepted_steps += 1
        accepted_summary = causal_five_field_state_summary(
            context,
            state_vector,
        )
        actual_mass_increments.append(
            _integrated_rest_mass_increment(
                context,
                result.physical_increment,
            )
        )
        expected_mass_increments.append(
            result.dt_used
            * (
                source_rate
                + accepted_summary["inner_face_rates"][0]
                - accepted_summary["outer_face_rates"][0]
            )
        )
        step_rows.append(
            _adaptive_step_row(
                accepted_steps,
                elapsed_time,
                result,
            )
        )

    final_restart = _restart_payload(
        state_vector,
        previous_increment,
        elapsed_time=elapsed_time,
        dt_next=dt_next,
        previous_dt=previous_dt,
        accepted_steps=accepted_steps,
        rejected_attempts=rejected_attempts,
        n_cells=n_cells,
        role="final_repeated_source_on_state",
    )
    save_causal_five_field_adaptive_restart(
        final_path,
        context,
        final_restart,
    )
    restored_final = load_causal_five_field_adaptive_restart(
        final_path,
        context,
    )
    final_restart_roundtrip_bitwise = (
        _restart_roundtrip_is_bitwise(
            final_restart,
            restored_final,
        )
    )
    final_summary = causal_five_field_state_summary(
        context,
        state_vector,
    )
    final_h_over_r = _reconstructed_h_over_r_summary(
        context,
        state_vector,
    )
    h_over_r_response = _h_over_r_response_summary(
        context,
        initial_vector,
        state_vector,
    )
    endpoint_subtraction_mass_change = (
        final_summary["integrated_conserved"][0]
        - initial_summary["integrated_conserved"][0]
    )
    actual_mass_change = math.fsum(actual_mass_increments)
    expected_mass_change = math.fsum(expected_mass_increments)
    mass_budget_relative_defect = float(
        abs(actual_mass_change - expected_mass_change)
        / max(
            abs(actual_mass_change),
            abs(expected_mass_change),
            source_rate * elapsed_time,
            1.0,
        )
    )
    if accepted_step_target is not None:
        target_reached = accepted_steps == accepted_step_target
    else:
        assert elapsed_time_target is not None
        target_reached = (
            abs(elapsed_time - elapsed_time_target)
            <= target_tolerance
        )
    all_step_gates_passed = all(
        row["maximum_scaled_residual"] <= 1.0e-8
        and row["maximum_scaled_algebraic_residual"] <= 1.0e-10
        and row["maximum_scaled_primitive_change"] <= 5.0e-4
        and row["maximum_scaled_total_change"] <= 1.0e-3
        and row["conservation_telescoping_relative_defect"]
        <= 1.0e-10
        and row["minimum_scattering_optical_depth"] > 1.0
        for row in step_rows
    )
    passed = bool(
        target_reached
        and all_step_gates_passed
        and restart_roundtrip_bitwise
        and restart_resume_step_bitwise
        and final_restart_roundtrip_bitwise
        and mass_budget_relative_defect <= 1.0e-10
    )
    report = {
        "n_cells": n_cells,
        "initialization_passed": True,
        "target": {
            "accepted_steps": accepted_step_target,
            "elapsed_time_seconds": elapsed_time_target,
        },
        "accepted_steps": accepted_steps,
        "rejected_attempts": rejected_attempts,
        "elapsed_time_seconds": elapsed_time,
        "loading_time_seconds": loading_time,
        "elapsed_loading_time_fraction": elapsed_time / loading_time,
        "source_rate_g_s": source_rate,
        "initial_state": initial_summary,
        "final_state": final_summary,
        "h_over_r_reconstruction": {
            "method": (
                "log-linear cell-center reconstruction and one-cell "
                "edge extrapolation on a shared 1025-point log-radius grid"
            ),
            "initial": initial_h_over_r,
            "final": final_h_over_r,
        },
        "h_over_r_response": h_over_r_response,
        "mass_budget": {
            "cancellation_safe_actual_change_g": actual_mass_change,
            "endpoint_subtraction_change_g": (
                endpoint_subtraction_mass_change
            ),
            "endpoint_subtraction_relative_defect": float(
                abs(
                    endpoint_subtraction_mass_change
                    - actual_mass_change
                )
                / max(abs(actual_mass_change), 1.0)
            ),
            "expected_change_g": expected_mass_change,
            "injected_mass_g": source_rate * elapsed_time,
            "relative_defect": mass_budget_relative_defect,
        },
        "restart": {
            "midpoint_path": (
                str(midpoint_path.relative_to(ROOT))
                if perform_restart_resume_audit
                else None
            ),
            "roundtrip_bitwise": restart_roundtrip_bitwise,
            "resume_step_bitwise": restart_resume_step_bitwise,
            "final_path": str(final_path.relative_to(ROOT)),
            "final_roundtrip_bitwise": (
                final_restart_roundtrip_bitwise
            ),
        },
        "steps": step_rows,
        "target_reached": target_reached,
        "all_step_gates_passed": all_step_gates_passed,
        "passed": passed,
        "terminal_message": terminal_message,
        "decision": (
            "short_repeated_source_on_gate_passed"
            if passed
            else "short_repeated_source_on_gate_failed"
        ),
    }
    return report, passed


def _repeated_mesh_comparison(n16: dict, n32: dict) -> dict:
    supply16 = n16["source_rate_g_s"]
    supply32 = n32["source_rate_g_s"]

    def metrics(run: dict, supply: float) -> dict:
        injected = supply * run["elapsed_time_seconds"]
        return {
            "mass_response_per_injected_mass": (
                run["mass_budget"][
                    "cancellation_safe_actual_change_g"
                ]
                / injected
            ),
            "inner_mass_flux_over_supply": (
                run["final_state"]["inner_face_rates"][0] / supply
            ),
            "outer_mass_flux_over_supply": (
                run["final_state"]["outer_face_rates"][0] / supply
            ),
            "maximum_h_over_r": run["final_state"]["maximum_h_over_r"],
            "reconstructed_maximum_h_over_r": (
                run["h_over_r_reconstruction"]["final"][
                    "reconstructed_common_domain_maximum"
                ]
            ),
            "maximum_absolute_delta_log_h_over_r": (
                run["h_over_r_response"][
                    "maximum_absolute_delta_log_h_over_r"
                ]
            ),
            "rms_delta_log_h_over_r": (
                run["h_over_r_response"]["rms_delta_log_h_over_r"]
            ),
        }

    left = metrics(n16, supply16)
    right = metrics(n32, supply32)
    left_radius = np.asarray(
        n16["h_over_r_response"]["sample_radius_rg"],
        dtype=float,
    )
    right_radius = np.asarray(
        n32["h_over_r_response"]["sample_radius_rg"],
        dtype=float,
    )
    if not np.array_equal(left_radius, right_radius):
        raise RuntimeError("H/R response samples do not share radii")
    left_response = np.asarray(
        n16["h_over_r_response"]["delta_log_h_over_r"],
        dtype=float,
    )
    right_response = np.asarray(
        n32["h_over_r_response"]["delta_log_h_over_r"],
        dtype=float,
    )
    response_difference = left_response - right_response
    differences = {
        "mass_response_per_injected_mass": abs(
            left["mass_response_per_injected_mass"]
            - right["mass_response_per_injected_mass"]
        ),
        "inner_mass_flux_over_supply": abs(
            left["inner_mass_flux_over_supply"]
            - right["inner_mass_flux_over_supply"]
        ),
        "outer_mass_flux_over_supply": abs(
            left["outer_mass_flux_over_supply"]
            - right["outer_mass_flux_over_supply"]
        ),
        "reconstructed_maximum_h_over_r_relative": abs(
            left["reconstructed_maximum_h_over_r"]
            - right["reconstructed_maximum_h_over_r"]
        )
        / max(
            abs(left["reconstructed_maximum_h_over_r"]),
            abs(right["reconstructed_maximum_h_over_r"]),
            np.finfo(float).tiny,
        ),
        "raw_cell_center_maximum_h_over_r_relative": abs(
            left["maximum_h_over_r"] - right["maximum_h_over_r"]
        )
        / max(
            abs(left["maximum_h_over_r"]),
            abs(right["maximum_h_over_r"]),
            np.finfo(float).tiny,
        ),
        "maximum_delta_log_h_over_r_response_difference": float(
            np.max(np.abs(response_difference))
        ),
        "rms_delta_log_h_over_r_response_difference": float(
            np.sqrt(np.mean(response_difference**2))
        ),
    }
    gates = {
        "mass_response_per_injected_mass": 0.05,
        "inner_mass_flux_over_supply": 0.05,
        "outer_mass_flux_over_supply": 0.05,
        "maximum_delta_log_h_over_r_response_difference": 5.0e-3,
    }
    passed = all(
        differences[name] <= limit
        for name, limit in gates.items()
    )
    return {
        "n16": left,
        "n32": right,
        "absolute_or_relative_differences": differences,
        "gates": gates,
        "passed": passed,
    }


def _run_repeated_source_on_audit(args: argparse.Namespace) -> None:
    n16, n16_passed = _run_repeated_source_on_resolution(
        16,
        accepted_step_target=8,
        elapsed_time_target=None,
        perform_restart_resume_audit=True,
    )
    n32 = None
    n32_passed = False
    mesh = None
    if n16_passed:
        n32, n32_passed = _run_repeated_source_on_resolution(
            32,
            accepted_step_target=None,
            elapsed_time_target=n16["elapsed_time_seconds"],
            perform_restart_resume_audit=False,
        )
    if n32_passed:
        mesh = _repeated_mesh_comparison(n16, n32)
    passed = bool(
        n16_passed
        and n32_passed
        and mesh is not None
        and mesh["passed"]
    )
    output = {
        "work_package": "WP10c5k",
        "scope": (
            "short adaptive exact-stream no-tide repeated startup with "
            "restart and N16/N32 equal-time gates"
        ),
        "n16": n16,
        "n32": n32,
        "mesh_comparison": mesh,
        "gates": {
            "n16_repeated_passed": n16_passed,
            "n32_attempted": n32 is not None,
            "n32_equal_time_passed": n32_passed,
            "mesh_gate_passed": (
                mesh["passed"] if mesh is not None else False
            ),
            "short_no_tide_startup_certified": passed,
            "long_evolution_authorized": False,
            "stability_certified": False,
            "hot_state_certified": False,
            "limit_cycle_certified": False,
            "tide_authorized": False,
            "wind_authorized": False,
        },
        "decision": (
            "short_repeated_source_on_mesh_gate_passed"
            if passed
            else "stop_before_long_or_forced_evolution"
        ),
    }
    output_path = _absolute(
        DEFAULT_REPEATED_SOURCE_ON_OUTPUT
        if args.output == DEFAULT_OUTPUT
        else args.output
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        output,
        indent=2,
        sort_keys=True,
        default=_json_default,
    )
    output_path.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


def _run_increment_primary_audit(
    args: argparse.Namespace,
    *,
    include_stream: bool = False,
) -> None:
    n16_attempts = []
    selected_n16_artifacts = None
    for target_change in TARGET_SCALED_PRIMITIVE_CHANGES:
        attempt, artifacts = _run_increment_primary_resolution(
            16,
            target_change,
            include_stream=include_stream,
        )
        n16_attempts.append(attempt)
        selected_n16_artifacts = artifacts
        if attempt["resolution_passed"]:
            break
    selected_n16 = n16_attempts[-1]
    n32_result = None
    n32_artifacts = None
    if selected_n16["resolution_passed"]:
        n32_result, n32_artifacts = _run_increment_primary_resolution(
            32,
            selected_n16["tiny_step"][
                "target_scaled_primitive_change"
            ],
            include_stream=include_stream,
        )
    all_passed = (
        selected_n16["resolution_passed"]
        and n32_result is not None
        and n32_result["resolution_passed"]
    )
    temporal_n16 = None
    temporal_n32 = None
    if all_passed:
        temporal_n16 = _temporal_refinement_comparison(
            selected_n16_artifacts,
            selected_n16["tiny_step"][
                "target_scaled_primitive_change"
            ],
        )
        if temporal_n16["passed"]:
            temporal_n32 = _temporal_refinement_comparison(
                n32_artifacts,
                n32_result["tiny_step"][
                    "target_scaled_primitive_change"
                ],
            )
    temporal_passed = (
        temporal_n16 is not None
        and temporal_n16["passed"]
        and temporal_n32 is not None
        and temporal_n32["passed"]
    )
    output = {
        "work_package": "WP10c5i" if include_stream else "WP10c5h",
        "scope": (
            (
                "full-DAE primary-increment backward Euler with direct "
                "conserved storage and exact circularized stream moments"
            )
            if include_stream
            else (
                "full-DAE primary-increment backward Euler with direct "
                "conserved storage"
            )
        ),
        "rank_relative_threshold": RANK_THRESHOLD,
        "finite_difference_step": FINITE_DIFFERENCE_STEP,
        "n16_attempts": n16_attempts,
        "n32_result": n32_result,
        "temporal_refinement": {
            "n16": temporal_n16,
            "n32": temporal_n32,
        },
        "gates": {
            "exact_stream_enabled": include_stream,
            "n16_passed": selected_n16["resolution_passed"],
            "n32_attempted": n32_result is not None,
            "n32_passed": (
                n32_result["resolution_passed"]
                if n32_result is not None
                else False
            ),
            "temporal_comparison_attempted": (
                temporal_n16 is not None
            ),
            "temporal_comparison_passed": temporal_passed,
            "early_time_numerical_gate_passed": temporal_passed,
            "physical_evolution_certified": False,
            "stability_certified": False,
            "tide_authorized": False,
            "wind_authorized": False,
        },
        "decision": (
            (
                "source_on_increment_primary_startup_gate_passed"
                if include_stream
                else "increment_primary_startup_gate_passed"
            )
            if temporal_passed
            else (
                "stop_after_temporal_refinement"
                if all_passed
                else "stop_before_temporal_refinement"
            )
        ),
    }
    output_path = _absolute(
        DEFAULT_SOURCE_ON_OUTPUT
        if include_stream and args.output == DEFAULT_OUTPUT
        else args.output
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        output,
        indent=2,
        sort_keys=True,
        default=_json_default,
    )
    output_path.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


def main() -> None:
    args = _arguments()
    if args.increment_primary_repeated_source_on_audit:
        _run_repeated_source_on_audit(args)
        return
    if args.increment_primary_sparse_backend_audit:
        _run_sparse_backend_audit(args)
        return
    if args.increment_primary_source_on_audit:
        _run_increment_primary_audit(args, include_stream=True)
        return
    if args.increment_primary_audit:
        _run_increment_primary_audit(args)
        return
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
