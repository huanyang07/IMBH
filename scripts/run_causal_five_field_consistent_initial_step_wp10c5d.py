"""Run the bounded WP10c5d consistent-data and tiny-step gate."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy.linalg.lapack import dgeequ, dgesvx
from scipy.interpolate import PchipInterpolator
from scipy.sparse import issparse
from scipy.sparse.csgraph import structural_rank

from imri_qpe.constants import C, G
from imri_qpe.layer3_minidisk_1d import (
    KERR_SCHILD_HILL_ENERGY_ZERO,
    CausalFiveFieldAdaptiveRestart,
    CausalFiveFieldAdaptiveStepConfig,
    CausalFiveFieldPhysicalStepLedger,
    CausalFiveFieldDAEContext,
    GasRadiationHillRocheNozzleProvider,
    SchwarzschildCurvatureVerticalFrequency,
    ValenciaPerfectFluidPrimitive,
    audit_causal_five_field_consistent_initial_data,
    audit_causal_five_field_dae_jacobian,
    audit_causal_five_field_principal,
    advance_causal_five_field_adaptive_backward_euler,
    advance_causal_five_field_increment_backward_euler,
    calibrate_causal_alpha_shear,
    causal_five_field_colored_central_jacobian,
    causal_five_field_dae_jacobian_color_groups,
    causal_five_field_dae_jacobian_sparsity,
    causal_five_field_dae_scaling,
    causal_five_field_equilibrated_sparse_solve,
    causal_five_field_h_over_r_profile,
    causal_five_field_loading_time,
    causal_five_field_physical_step_ledger,
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
DEFAULT_MATCHED_SOURCE_CONTROL_OUTPUT = (
    ROOT
    / "outputs/tables/causal_five_field_matched_source_control_wp10c5l.json"
)
DEFAULT_SOURCE_COMPATIBLE_STARTUP_OUTPUT = (
    ROOT
    / "outputs/tables/causal_five_field_source_compatible_startup_wp10c5m.json"
)
DEFAULT_SOURCE_COMPATIBLE_DURATION_OUTPUT = (
    ROOT
    / "outputs/tables/causal_five_field_source_compatible_duration_wp10c5n.json"
)
DEFAULT_MESH_COMMON_STARTUP_DURATION_OUTPUT = (
    ROOT
    / "outputs/tables/causal_five_field_mesh_common_startup_duration_wp10c5op.json"
)
DEFAULT_MESH_COMMON_TEMPORAL_PARITY_OUTPUT = (
    ROOT
    / "outputs/tables/causal_five_field_mesh_common_temporal_parity_wp10c5q.json"
)
DEFAULT_MESH_COMMON_SPATIAL_RESPONSE_OUTPUT = (
    ROOT
    / "outputs/tables/causal_five_field_mesh_common_spatial_response_wp10c5r.json"
)
DEFAULT_MESH_COMMON_N64_CONFIRMATION_OUTPUT = (
    ROOT
    / "outputs/tables/causal_five_field_mesh_common_n64_confirmation_wp10c5s.json"
)
DEFAULT_MESH_COMMON_N64_LEDGER_REPLAY_OUTPUT = (
    ROOT
    / "outputs/tables/causal_five_field_mesh_common_n64_ledger_replay_wp10c5t.json"
)
DEFAULT_MESH_COMMON_N128_CONFIRMATION_OUTPUT = (
    ROOT
    / "outputs/tables/causal_five_field_mesh_common_n128_confirmation_wp10c5u.json"
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
    parser.add_argument(
        "--increment-primary-matched-source-control-audit",
        action="store_true",
    )
    parser.add_argument(
        "--increment-primary-source-compatible-startup-audit",
        action="store_true",
    )
    parser.add_argument(
        "--increment-primary-source-compatible-duration-audit",
        action="store_true",
    )
    parser.add_argument(
        "--increment-primary-mesh-common-startup-duration-audit",
        action="store_true",
    )
    parser.add_argument(
        "--increment-primary-mesh-common-temporal-parity-audit",
        action="store_true",
    )
    parser.add_argument(
        "--increment-primary-mesh-common-spatial-response-audit",
        action="store_true",
    )
    parser.add_argument(
        "--increment-primary-mesh-common-n64-confirmation-audit",
        action="store_true",
    )
    parser.add_argument(
        "--increment-primary-mesh-common-n64-ledger-replay-audit",
        action="store_true",
    )
    parser.add_argument(
        "--increment-primary-mesh-common-n128-confirmation-audit",
        action="store_true",
    )
    return parser.parse_args()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
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


def _rank_summary(
    matrix: np.ndarray,
    *,
    equilibrate: bool = True,
) -> dict:
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
    if not equilibrate:
        return result
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
    seed_kwargs: dict | None = None,
) -> tuple[dict, dict]:
    context = _context(n_cells, include_stream=include_stream)
    seed_parameters = dict(seed_kwargs or {})
    old_state = make_causal_five_field_seed(
        context,
        **seed_parameters,
    )
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
        "seed_parameters": seed_parameters,
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
        "stationary_evaluation": stationary_evaluation,
        "stationary_jacobian_audit": stationary,
        "backward_euler_jacobian_audit": backward_euler,
        "consistent_initial_data_audit": consistent,
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
    work_package: str,
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
            "work_package": work_package,
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
    seed_kwargs: dict | None = None,
    restart_label: str = "wp10c5k",
    restart_work_package: str = "WP10c5k",
    initialization_bundle: tuple[dict, dict] | None = None,
    step_residual_tolerance: float = 1.0e-8,
    step_algebraic_tolerance: float = 1.0e-10,
    mass_budget_tolerance: float = 1.0e-10,
) -> tuple[dict, bool]:
    if initialization_bundle is None:
        initialization, artifacts = _run_increment_primary_resolution(
            n_cells,
            TARGET_SCALED_PRIMITIVE_CHANGES[0],
            include_stream=True,
            seed_kwargs=seed_kwargs,
        )
    else:
        initialization, artifacts = initialization_bundle
        if initialization.get("n_cells") != n_cells:
            raise ValueError(
                "repeated-run initialization resolution does not match"
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
        residual_tolerance=step_residual_tolerance,
        algebraic_residual_tolerance=step_algebraic_tolerance,
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
        / f"causal_{restart_label}_N{n_cells:03d}_midpoint.npz"
    )
    final_path = (
        restart_directory
        / f"causal_{restart_label}_N{n_cells:03d}_final.npz"
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
                work_package=restart_work_package,
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
        work_package=restart_work_package,
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
        row["maximum_scaled_residual"] <= step_residual_tolerance
        and row["maximum_scaled_algebraic_residual"]
        <= step_algebraic_tolerance
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
        and mass_budget_relative_defect <= mass_budget_tolerance
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
        "acceptance_tolerances": {
            "scaled_residual": step_residual_tolerance,
            "scaled_algebraic_residual": (
                step_algebraic_tolerance
            ),
            "mass_budget_relative_defect": mass_budget_tolerance,
        },
        "passed": passed,
        "terminal_message": terminal_message,
        "decision": (
            "short_repeated_source_on_gate_passed"
            if passed
            else "short_repeated_source_on_gate_failed"
        ),
    }
    return report, passed


def _repeated_mesh_comparison(
    left_run: dict,
    right_run: dict,
    *,
    left_label: str = "n16",
    right_label: str = "n32",
) -> dict:
    left_supply = left_run["source_rate_g_s"]
    right_supply = right_run["source_rate_g_s"]

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

    left = metrics(left_run, left_supply)
    right = metrics(right_run, right_supply)
    left_radius = np.asarray(
        left_run["h_over_r_response"]["sample_radius_rg"],
        dtype=float,
    )
    right_radius = np.asarray(
        right_run["h_over_r_response"]["sample_radius_rg"],
        dtype=float,
    )
    if not np.array_equal(left_radius, right_radius):
        raise RuntimeError("H/R response samples do not share radii")
    left_response = np.asarray(
        left_run["h_over_r_response"]["delta_log_h_over_r"],
        dtype=float,
    )
    right_response = np.asarray(
        right_run["h_over_r_response"]["delta_log_h_over_r"],
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
        left_label: left,
        right_label: right,
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


def _matched_source_step_ledger(
    source_on_context: CausalFiveFieldDAEContext,
    source_off_context: CausalFiveFieldDAEContext,
    source_on_old: np.ndarray,
    source_off_old: np.ndarray,
    source_on_increment: np.ndarray,
    source_off_increment: np.ndarray,
    timestep_seconds: float,
) -> dict:
    source_on = causal_five_field_physical_step_ledger(
        source_on_context,
        source_on_old,
        source_on_increment,
        timestep_seconds,
    )
    source_off = causal_five_field_physical_step_ledger(
        source_off_context,
        source_off_old,
        source_off_increment,
        timestep_seconds,
    )
    term_names = (
        "conserved_storage_change",
        "vertical_storage_change",
        "boundary_transport",
        "endogenous_source",
        "prescribed_stream_source",
    )
    differential = {
        name: (
            np.asarray(getattr(source_on, name), dtype=float)
            - np.asarray(getattr(source_off, name), dtype=float)
        )
        for name in term_names
    }
    inferred_stream = (
        differential["conserved_storage_change"]
        + differential["vertical_storage_change"]
        + differential["boundary_transport"]
        - differential["endogenous_source"]
    )
    prescribed_stream = differential["prescribed_stream_source"]
    recovery_defect = inferred_stream - prescribed_stream
    closure_difference = (
        np.asarray(source_on.closure_defect, dtype=float)
        - np.asarray(source_off.closure_defect, dtype=float)
    )
    return {
        "timestep_seconds": timestep_seconds,
        "differential_terms": {
            name: [float(value) for value in values]
            for name, values in differential.items()
        },
        "inferred_stream_source": [
            float(value) for value in inferred_stream
        ],
        "prescribed_stream_source": [
            float(value) for value in prescribed_stream
        ],
        "recovery_defect": [
            float(value) for value in recovery_defect
        ],
        "closure_difference": [
            float(value) for value in closure_difference
        ],
        "recovery_closure_identity_maximum_absolute_defect": float(
            np.max(np.abs(recovery_defect - closure_difference))
        ),
    }


def _aggregate_matched_source_ledgers(step_ledgers: list[dict]) -> dict:
    term_names = (
        "conserved_storage_change",
        "vertical_storage_change",
        "boundary_transport",
        "endogenous_source",
        "prescribed_stream_source",
    )
    aggregated = {
        name: np.asarray(
            [
                math.fsum(
                    row["differential_terms"][name][field]
                    for row in step_ledgers
                )
                for field in range(5)
            ],
            dtype=float,
        )
        for name in term_names
    }
    inferred = (
        aggregated["conserved_storage_change"]
        + aggregated["vertical_storage_change"]
        + aggregated["boundary_transport"]
        - aggregated["endogenous_source"]
    )
    prescribed = aggregated["prescribed_stream_source"]
    defect = inferred - prescribed
    closure_difference = np.asarray(
        [
            math.fsum(row["closure_difference"][field] for row in step_ledgers)
            for field in range(5)
        ],
        dtype=float,
    )
    balanced_scale = (
        np.abs(aggregated["conserved_storage_change"])
        + np.abs(aggregated["vertical_storage_change"])
        + np.abs(aggregated["boundary_transport"])
        + np.abs(aggregated["endogenous_source"])
        + np.abs(prescribed)
    )
    balanced_scale = np.maximum(balanced_scale, 1.0)
    source_scale = np.maximum(np.abs(prescribed), 1.0)
    source_relative = np.abs(defect) / source_scale
    balanced_relative = np.abs(defect) / balanced_scale
    return {
        "differential_terms": {
            name: [float(value) for value in values]
            for name, values in aggregated.items()
        },
        "inferred_stream_source": [float(value) for value in inferred],
        "prescribed_stream_source": [
            float(value) for value in prescribed
        ],
        "recovery_defect": [float(value) for value in defect],
        "source_relative_recovery_defect": [
            float(value) for value in source_relative
        ],
        "balanced_relative_recovery_defect": [
            float(value) for value in balanced_relative
        ],
        "maximum_source_relative_recovery_defect_first_four": float(
            np.max(source_relative[:4])
        ),
        "maximum_balanced_relative_recovery_defect": float(
            np.max(balanced_relative)
        ),
        "maximum_balanced_relative_recovery_defect_first_four": float(
            np.max(balanced_relative[:4])
        ),
        "closure_difference": [
            float(value) for value in closure_difference
        ],
        "recovery_closure_identity_maximum_absolute_defect": float(
            np.max(np.abs(defect - closure_difference))
        ),
    }


def _matched_source_profile_response(
    context: CausalFiveFieldDAEContext,
    source_on_vector: np.ndarray,
    source_off_vector: np.ndarray,
) -> dict:
    radii = np.geomspace(
        2.0 * context.grid.gravitational_radius,
        330.0 * context.grid.gravitational_radius,
        129,
    )
    log_radii = np.log(radii)
    source_on = _reconstructed_log_h_over_r(
        context,
        source_on_vector,
        log_radii,
    )
    source_off = _reconstructed_log_h_over_r(
        context,
        source_off_vector,
        log_radii,
    )
    response = source_on - source_off
    return {
        "sample_radius_rg": [
            float(value / context.grid.gravitational_radius)
            for value in radii
        ],
        "delta_log_h_over_r": [float(value) for value in response],
        "maximum_absolute_delta_log_h_over_r": float(
            np.max(np.abs(response))
        ),
        "rms_delta_log_h_over_r": float(
            np.sqrt(np.mean(response**2))
        ),
    }


def _fixed_step_summary(result) -> dict:
    return {
        "accepted": result.accepted,
        "maximum_scaled_residual": result.maximum_scaled_residual,
        "maximum_scaled_algebraic_residual": (
            result.maximum_scaled_algebraic_residual
        ),
        "maximum_scaled_primitive_change": (
            result.maximum_scaled_primitive_change
        ),
        "maximum_scaled_total_change": (
            result.maximum_scaled_total_change
        ),
        "conservation_telescoping_relative_defect": (
            result.conservation_telescoping_relative_defect
        ),
        "minimum_scattering_optical_depth": (
            result.minimum_scattering_optical_depth
        ),
        "outer_boundary_choked_before": (
            result.outer_boundary_choked_before
        ),
        "outer_boundary_choked_after": (
            result.outer_boundary_choked_after
        ),
        "iterations": result.iterations,
        "function_evaluations": result.function_evaluations,
        "jacobian_evaluations": result.jacobian_evaluations,
        "maximum_linear_residual": result.maximum_linear_residual,
        "message": result.message,
    }


def _run_matched_source_control_resolution(
    n_cells: int,
    *,
    accepted_step_target: int | None,
    elapsed_time_target: float | None,
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
            "matched control requires exactly one duration target"
        )

    source_on_context = artifacts["context"]
    source_off_context = _context(n_cells, include_stream=False)
    source_on_vector = np.asarray(artifacts["old_vector"], dtype=float)
    source_off_vector = pack_causal_five_field_state(
        make_causal_five_field_seed(source_off_context)
    )
    initial_vectors_bitwise_equal = np.array_equal(
        source_on_vector,
        source_off_vector,
    )
    if not initial_vectors_bitwise_equal:
        return {
            "n_cells": n_cells,
            "initialization_passed": True,
            "initial_vectors_bitwise_equal": False,
            "passed": False,
            "decision": "matched_control_initial_states_differ",
        }, False
    if source_on_context.stream_sources is None:
        raise RuntimeError("matched source-on context has no stream")

    base_dt = float(artifacts["timestep_seconds"])
    config = CausalFiveFieldAdaptiveStepConfig(
        minimum_dt=base_dt / 128.0,
        maximum_dt=16.0 * base_dt,
        maximum_scaled_primitive_change=5.0e-4,
        maximum_scaled_total_change=1.0e-3,
        shrink_factor=0.5,
        growth_factor=1.5,
        maximum_retries=6,
        easy_iterations=3,
        residual_tolerance=1.0e-10,
        algebraic_residual_tolerance=1.0e-11,
        conservation_tolerance=1.0e-10,
        finite_difference_step=FINITE_DIFFERENCE_STEP,
        maximum_newton_iterations=12,
    ).validated()
    source_on_previous_increment = np.asarray(
        artifacts["physical_increment"],
        dtype=float,
    )
    source_off_previous_increment = np.array(
        source_on_previous_increment,
        copy=True,
    )
    previous_dt = base_dt
    dt_next = base_dt
    elapsed_time = 0.0
    accepted_steps = 0
    rejected_pair_attempts = 0
    step_rows: list[dict] = []
    step_ledgers: list[dict] = []
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

        trial_dt = requested_dt
        pair_attempts: list[dict] = []
        source_on_step = None
        source_off_step = None
        for retry in range(config.maximum_retries + 1):
            source_on_predictor = (
                source_on_previous_increment
                * (trial_dt / previous_dt)
            )
            source_off_predictor = (
                source_off_previous_increment
                * (trial_dt / previous_dt)
            )
            source_on_trial = (
                advance_causal_five_field_increment_backward_euler(
                    source_on_context,
                    source_on_vector,
                    trial_dt,
                    source_on_predictor,
                    config,
                )
            )
            source_off_trial = (
                advance_causal_five_field_increment_backward_euler(
                    source_off_context,
                    source_off_vector,
                    trial_dt,
                    source_off_predictor,
                    config,
                )
            )
            pair_accepted = bool(
                source_on_trial.accepted
                and source_off_trial.accepted
            )
            pair_attempts.append(
                {
                    "retry": retry,
                    "timestep_seconds": trial_dt,
                    "accepted": pair_accepted,
                    "source_on": _fixed_step_summary(source_on_trial),
                    "source_off": _fixed_step_summary(source_off_trial),
                }
            )
            if pair_accepted:
                source_on_step = source_on_trial
                source_off_step = source_off_trial
                break
            rejected_pair_attempts += 1
            next_dt = trial_dt * config.shrink_factor
            if next_dt < config.minimum_dt:
                break
            trial_dt = next_dt
        if source_on_step is None or source_off_step is None:
            terminal_message = (
                "lockstep retries exhausted without two accepted states"
            )
            break

        source_on_old = source_on_vector
        source_off_old = source_off_vector
        source_on_vector = np.asarray(
            source_on_step.state_vector,
            dtype=float,
        )
        source_off_vector = np.asarray(
            source_off_step.state_vector,
            dtype=float,
        )
        source_on_previous_increment = np.asarray(
            source_on_step.physical_increment,
            dtype=float,
        )
        source_off_previous_increment = np.asarray(
            source_off_step.physical_increment,
            dtype=float,
        )
        previous_dt = trial_dt
        elapsed_time += trial_dt
        accepted_steps += 1
        ledger = _matched_source_step_ledger(
            source_on_context,
            source_off_context,
            source_on_old,
            source_off_old,
            source_on_step.physical_increment,
            source_off_step.physical_increment,
            trial_dt,
        )
        step_ledgers.append(ledger)
        easy = bool(
            source_on_step.iterations <= config.easy_iterations
            and source_off_step.iterations <= config.easy_iterations
            and source_on_step.maximum_scaled_primitive_change
            <= 0.5 * config.maximum_scaled_primitive_change
            and source_off_step.maximum_scaled_primitive_change
            <= 0.5 * config.maximum_scaled_primitive_change
            and source_on_step.maximum_scaled_total_change
            <= 0.5 * config.maximum_scaled_total_change
            and source_off_step.maximum_scaled_total_change
            <= 0.5 * config.maximum_scaled_total_change
        )
        dt_next = min(
            (
                trial_dt * config.growth_factor
                if easy
                else trial_dt
            ),
            config.maximum_dt,
        )
        step_rows.append(
            {
                "accepted_step": accepted_steps,
                "elapsed_time_seconds": elapsed_time,
                "dt_used_seconds": trial_dt,
                "dt_next_seconds": dt_next,
                "attempts": pair_attempts,
                "source_ledger": ledger,
            }
        )

    if accepted_step_target is not None:
        target_reached = accepted_steps == accepted_step_target
    else:
        assert elapsed_time_target is not None
        target_reached = (
            abs(elapsed_time - elapsed_time_target)
            <= target_tolerance
        )
    aggregate = _aggregate_matched_source_ledgers(step_ledgers)
    source_on_summary = causal_five_field_state_summary(
        source_on_context,
        source_on_vector,
    )
    source_off_summary = causal_five_field_state_summary(
        source_off_context,
        source_off_vector,
    )
    source_rate = float(
        np.sum(source_on_context.stream_sources.rest_mass)
    )
    source_profile = _matched_source_profile_response(
        source_on_context,
        source_on_vector,
        source_off_vector,
    )
    exact_timestep_history = bool(
        all(
            row["attempts"][-1]["source_on"]["accepted"]
            and row["attempts"][-1]["source_off"]["accepted"]
            for row in step_rows
        )
    )
    source_relative_gate = 1.0e-4
    balanced_relative_gate = 1.0e-4
    passed = bool(
        target_reached
        and exact_timestep_history
        and aggregate[
            "maximum_source_relative_recovery_defect_first_four"
        ]
        <= source_relative_gate
        and aggregate[
            "maximum_balanced_relative_recovery_defect_first_four"
        ]
        <= balanced_relative_gate
    )
    return {
        "n_cells": n_cells,
        "initialization_passed": True,
        "initial_vectors_bitwise_equal": initial_vectors_bitwise_equal,
        "target": {
            "accepted_steps": accepted_step_target,
            "elapsed_time_seconds": elapsed_time_target,
        },
        "accepted_steps": accepted_steps,
        "rejected_pair_attempts": rejected_pair_attempts,
        "elapsed_time_seconds": elapsed_time,
        "loading_time_seconds": causal_five_field_loading_time(
            source_on_context,
            np.asarray(artifacts["old_vector"], dtype=float),
        ),
        "source_rate_g_s": source_rate,
        "source_on_final_state": source_on_summary,
        "source_off_final_state": source_off_summary,
        "source_isolated_response": {
            "cancellation_safe_integrated_conserved_change": (
                aggregate["differential_terms"][
                    "conserved_storage_change"
                ]
            ),
            "inner_face_rate_difference": [
                float(left - right)
                for left, right in zip(
                    source_on_summary["inner_face_rates"],
                    source_off_summary["inner_face_rates"],
                    strict=True,
                )
            ],
            "outer_face_rate_difference": [
                float(left - right)
                for left, right in zip(
                    source_on_summary["outer_face_rates"],
                    source_off_summary["outer_face_rates"],
                    strict=True,
                )
            ],
            "h_over_r": source_profile,
        },
        "aggregate_source_moment_ledger": aggregate,
        "gates": {
            "target_reached": target_reached,
            "exact_shared_timestep_history": exact_timestep_history,
            "source_relative_recovery_defect_first_four": (
                source_relative_gate
            ),
            "balanced_relative_recovery_defect_first_four": (
                balanced_relative_gate
            ),
        },
        "steps": step_rows,
        "passed": passed,
        "terminal_message": terminal_message,
        "decision": (
            "matched_source_control_passed"
            if passed
            else "matched_source_control_failed"
        ),
    }, passed


def _matched_source_mesh_comparison(n16: dict, n32: dict) -> dict:
    def metrics(run: dict) -> dict:
        source_rate = run["source_rate_g_s"]
        injected_mass = source_rate * run["elapsed_time_seconds"]
        response = run["source_isolated_response"]
        return {
            "conserved_mass_response_per_injected_mass": (
                response[
                    "cancellation_safe_integrated_conserved_change"
                ][0]
                / injected_mass
            ),
            "inner_mass_flux_response_over_supply": (
                response["inner_face_rate_difference"][0]
                / source_rate
            ),
            "outer_mass_flux_response_over_supply": (
                response["outer_face_rate_difference"][0]
                / source_rate
            ),
            "maximum_absolute_delta_log_h_over_r": (
                response["h_over_r"][
                    "maximum_absolute_delta_log_h_over_r"
                ]
            ),
            "rms_delta_log_h_over_r": (
                response["h_over_r"]["rms_delta_log_h_over_r"]
            ),
        }

    left = metrics(n16)
    right = metrics(n32)
    left_profile = np.asarray(
        n16["source_isolated_response"]["h_over_r"][
            "delta_log_h_over_r"
        ],
        dtype=float,
    )
    right_profile = np.asarray(
        n32["source_isolated_response"]["h_over_r"][
            "delta_log_h_over_r"
        ],
        dtype=float,
    )
    profile_difference = left_profile - right_profile
    differences = {
        "conserved_mass_response_per_injected_mass": abs(
            left["conserved_mass_response_per_injected_mass"]
            - right["conserved_mass_response_per_injected_mass"]
        ),
        "inner_mass_flux_response_over_supply": abs(
            left["inner_mass_flux_response_over_supply"]
            - right["inner_mass_flux_response_over_supply"]
        ),
        "outer_mass_flux_response_over_supply": abs(
            left["outer_mass_flux_response_over_supply"]
            - right["outer_mass_flux_response_over_supply"]
        ),
        "maximum_delta_log_h_over_r_response_difference": float(
            np.max(np.abs(profile_difference))
        ),
        "rms_delta_log_h_over_r_response_difference": float(
            np.sqrt(np.mean(profile_difference**2))
        ),
    }
    gates = {
        "conserved_mass_response_per_injected_mass": 0.05,
        "inner_mass_flux_response_over_supply": 0.05,
        "outer_mass_flux_response_over_supply": 0.05,
        "maximum_delta_log_h_over_r_response_difference": 5.0e-3,
    }
    passed = all(
        differences[name] <= limit
        for name, limit in gates.items()
    )
    return {
        "n16": left,
        "n32": right,
        "absolute_differences": differences,
        "gates": gates,
        "passed": passed,
    }


def _run_matched_source_control_audit(args: argparse.Namespace) -> None:
    n16, n16_passed = _run_matched_source_control_resolution(
        16,
        accepted_step_target=8,
        elapsed_time_target=None,
    )
    n32 = None
    n32_passed = False
    mesh = None
    if n16_passed:
        n32, n32_passed = _run_matched_source_control_resolution(
            32,
            accepted_step_target=None,
            elapsed_time_target=n16["elapsed_time_seconds"],
        )
    if n32_passed:
        mesh = _matched_source_mesh_comparison(n16, n32)
    passed = bool(
        n16_passed
        and n32_passed
        and mesh is not None
        and mesh["passed"]
    )
    output = {
        "work_package": "WP10c5l",
        "scope": (
            "lockstep source-on/source-off causal control with exact "
            "differential five-field source-moment recovery"
        ),
        "n16": n16,
        "n32": n32,
        "mesh_comparison": mesh,
        "gates": {
            "n16_matched_control_passed": n16_passed,
            "n32_attempted": n32 is not None,
            "n32_equal_time_control_passed": n32_passed,
            "mesh_gate_passed": (
                mesh["passed"] if mesh is not None else False
            ),
            "isolated_stream_response_certified": passed,
            "long_evolution_authorized": False,
            "source_compatible_initial_datum_certified": False,
            "tide_authorized": False,
            "wind_authorized": False,
        },
        "decision": (
            "matched_source_on_off_control_mesh_gate_passed"
            if passed
            else "stop_before_source_compatible_initialization"
        ),
    }
    output_path = _absolute(
        DEFAULT_MATCHED_SOURCE_CONTROL_OUTPUT
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


def _source_compatible_seed_parameters(
    n_cells: int,
) -> tuple[dict, dict]:
    context = _context(n_cells, include_stream=True)
    if context.stream_sources is None:
        raise RuntimeError("source-compatible datum requires a stream")
    source_rate = float(np.sum(context.stream_sources.rest_mass))
    unit_state = make_causal_five_field_seed(
        context,
        inner_surface_density=1.0,
        inner_temperature=1.0e6,
    )
    unit_summary = causal_five_field_state_summary(
        context,
        pack_causal_five_field_state(unit_state),
    )
    unit_inner_rate = float(unit_summary["inner_face_rates"][0])
    if unit_inner_rate >= 0.0:
        raise RuntimeError(
            "source-compatible seed requires inward inner flux"
        )
    inner_surface_density = source_rate / abs(unit_inner_rate)

    target_inner_h_over_r = 0.1
    inner_radius = float(context.grid.centers[0])
    eos = context.vertical_frequency.eos(inner_radius)

    def thickness(log_temperature: float) -> float:
        thermodynamics = eos.from_surface_density_temperature(
            inner_surface_density,
            float(np.exp(log_temperature)),
        )
        return (
            thermodynamics.proper_half_thickness / inner_radius
        )

    lower = float(np.log(1.0e5))
    upper = float(np.log(1.0e7))
    if (
        thickness(lower) >= target_inner_h_over_r
        or thickness(upper) <= target_inner_h_over_r
    ):
        raise RuntimeError(
            "source-compatible thickness target is not bracketed"
        )
    for _iteration in range(80):
        midpoint = 0.5 * (lower + upper)
        if thickness(midpoint) < target_inner_h_over_r:
            lower = midpoint
        else:
            upper = midpoint
    inner_temperature = float(np.exp(0.5 * (lower + upper)))
    parameters = {
        "inner_surface_density": inner_surface_density,
        "outer_surface_density": 1.0e5,
        "inner_temperature": inner_temperature,
        "outer_temperature": 8.0e5,
        "inner_radial_velocity_over_c": -0.40,
        "inner_azimuthal_velocity_over_c": 0.60,
        "outer_radial_velocity_margin_over_c": 1.0e-5,
    }
    return parameters, {
        "target_absolute_inner_mass_flux_over_supply": 1.0,
        "target_inner_h_over_r": target_inner_h_over_r,
        "surface_density_construction": (
            "exact linear rest-mass face-flux inversion"
        ),
        "temperature_construction": (
            "log-temperature bisection at the first cell center"
        ),
        "temperature_bracket_k": [1.0e5, 1.0e7],
    }


def _mesh_common_source_compatible_seed_parameters() -> tuple[
    dict,
    dict,
]:
    context = _context(16, include_stream=True)
    if context.stream_sources is None:
        raise RuntimeError("mesh-common datum requires a stream")
    gravitational_radius = context.grid.gravitational_radius
    inner_plateau = 6.0 * gravitational_radius
    outer_plateau = STREAM_CENTER_RG * gravitational_radius
    source_rate = float(np.sum(context.stream_sources.rest_mass))
    unit_state = make_causal_five_field_seed(
        context,
        inner_surface_density=1.0,
        inner_temperature=1.0e6,
        profile_inner_plateau_radius=inner_plateau,
        profile_outer_plateau_radius=outer_plateau,
    )
    unit_summary = causal_five_field_state_summary(
        context,
        pack_causal_five_field_state(unit_state),
    )
    unit_inner_rate = float(unit_summary["inner_face_rates"][0])
    if unit_inner_rate >= 0.0:
        raise RuntimeError("mesh-common datum requires inner inflow")
    inner_surface_density = source_rate / abs(unit_inner_rate)

    target_inner_h_over_r = 0.1
    physical_inner_radius = float(context.grid.edges[0])
    eos = context.vertical_frequency.eos(physical_inner_radius)

    def thickness(log_temperature: float) -> float:
        thermodynamics = eos.from_surface_density_temperature(
            inner_surface_density,
            float(np.exp(log_temperature)),
        )
        return (
            thermodynamics.proper_half_thickness
            / physical_inner_radius
        )

    lower = float(np.log(1.0e5))
    upper = float(np.log(1.0e7))
    if (
        thickness(lower) >= target_inner_h_over_r
        or thickness(upper) <= target_inner_h_over_r
    ):
        raise RuntimeError(
            "mesh-common thickness target is not bracketed"
        )
    for _iteration in range(80):
        midpoint = 0.5 * (lower + upper)
        if thickness(midpoint) < target_inner_h_over_r:
            lower = midpoint
        else:
            upper = midpoint
    inner_temperature = float(np.exp(0.5 * (lower + upper)))
    outer_surface_density = 1.0e5
    outer_temperature = 8.0e5
    physical_outer_radius = float(context.grid.edges[-1])
    target_outer_h_over_r = (
        context.vertical_frequency.eos(
            physical_outer_radius
        ).from_surface_density_temperature(
            outer_surface_density,
            outer_temperature,
        ).proper_half_thickness
        / physical_outer_radius
    )
    parameters = {
        "inner_surface_density": inner_surface_density,
        "outer_surface_density": outer_surface_density,
        "inner_temperature": inner_temperature,
        "outer_temperature": outer_temperature,
        "inner_radial_velocity_over_c": -0.40,
        "inner_azimuthal_velocity_over_c": 0.60,
        "outer_radial_velocity_margin_over_c": 1.0e-5,
        "profile_inner_plateau_radius": inner_plateau,
        "profile_outer_plateau_radius": outer_plateau,
        "profile_interpolate_log_h_over_r": True,
    }
    return parameters, {
        "target_absolute_inner_mass_flux_over_supply": 1.0,
        "target_physical_inner_face_h_over_r": (
            target_inner_h_over_r
        ),
        "target_physical_outer_face_h_over_r": (
            target_outer_h_over_r
        ),
        "inner_plateau_radius_rg": 6.0,
        "outer_plateau_radius_rg": STREAM_CENTER_RG,
        "profile": (
            "common C2 smootherstep in log radius for log Sigma, "
            "velocities, and log H/R between fixed physical plateaus"
        ),
        "surface_density_construction": (
            "exact linear rest-mass face-flux inversion on the "
            "shared inner plateau"
        ),
        "temperature_construction": (
            "shared physical-face temperature anchors converted to "
            "log H/R, followed by local EOS inversion"
        ),
        "temperature_bracket_k": [1.0e5, 1.0e7],
    }


def _inner_principal_summary(
    context: CausalFiveFieldDAEContext,
    primitive_chart: np.ndarray,
) -> dict:
    radius = float(context.grid.edges[0])
    geometry = kerr_schild_column_geometry(
        radius,
        context.grid.gravitational_radius,
    )
    sigma = float(np.exp(primitive_chart[0]))
    temperature = float(np.exp(primitive_chart[3]))
    thermodynamics = context.vertical_frequency.eos(
        radius
    ).from_surface_density_temperature(
        sigma,
        temperature,
    )
    primitive = ValenciaPerfectFluidPrimitive(
        surface_density=sigma,
        radial_velocity_over_c=float(primitive_chart[1]),
        azimuthal_velocity_over_c=float(primitive_chart[2]),
        specific_internal_energy=(
            thermodynamics.specific_internal_energy
        ),
        integrated_pressure=thermodynamics.integrated_pressure,
    )
    closure = calibrate_causal_alpha_shear(
        primitive,
        alpha=context.alpha,
        stress_factor=context.stress_factor,
        reference_positive_shear_rate=(
            1.5 * context.vertical_frequency.frequency(radius)
        ),
        viscous_signal_speed_over_c=(
            np.sqrt(context.alpha)
            * thermodynamics.sound_speed
            / C
        ),
    )
    audit = audit_causal_five_field_principal(
        geometry,
        context.vertical_frequency.eos(radius),
        closure,
        surface_density=sigma,
        radial_velocity_over_c=primitive.radial_velocity_over_c,
        azimuthal_velocity_over_c=primitive.azimuthal_velocity_over_c,
        temperature=temperature,
    )
    return {
        "radius_rg": radius / context.grid.gravitational_radius,
        "incoming_inner_characteristics": (
            audit.incoming_inner_characteristics
        ),
        "coordinate_speeds_over_c": [
            float(value) for value in audit.coordinate_speeds_over_c
        ],
        "maximum_light_cone_excess": audit.maximum_light_cone_excess,
    }


def _sparse_source_compatible_increment_initialization(
    n_cells: int,
    seed_parameters: dict,
) -> tuple[dict, dict]:
    """Build one bounded increment using the certified colored backend."""

    seed_audit, artifacts = _mesh_common_spatial_seed_artifacts(
        n_cells,
        seed_parameters,
    )
    context = artifacts["context"]
    old_vector = np.asarray(artifacts["old_vector"], dtype=float)
    scaling = artifacts["scaling"]
    tangent = np.asarray(
        artifacts["consistent_scaled_tangent"],
        dtype=float,
    )
    n_differential = 5 * n_cells
    primitive_tangent = tangent[
        n_differential : 2 * n_differential
    ]
    timestep = float(
        TARGET_SCALED_PRIMITIVE_CHANGES[0]
        / max(
            np.max(np.abs(primitive_tangent)),
            np.finfo(float).tiny,
        )
    )
    predictor = (
        np.asarray(scaling.column_scales, dtype=float)
        * timestep
        * tangent
    )
    bound = 1.25 * TARGET_SCALED_PRIMITIVE_CHANGES[0]
    config = CausalFiveFieldAdaptiveStepConfig(
        minimum_dt=timestep / 128.0,
        maximum_dt=16.0 * timestep,
        maximum_scaled_primitive_change=bound,
        maximum_scaled_total_change=bound,
        shrink_factor=0.5,
        growth_factor=1.5,
        maximum_retries=0,
        easy_iterations=3,
        residual_tolerance=1.0e-8,
        algebraic_residual_tolerance=1.0e-10,
        conservation_tolerance=1.0e-10,
        finite_difference_step=FINITE_DIFFERENCE_STEP,
        maximum_newton_iterations=12,
    ).validated()
    step = advance_causal_five_field_increment_backward_euler(
        context,
        old_vector,
        timestep,
        predictor,
        config,
    )
    physical_increment = np.asarray(
        step.physical_increment,
        dtype=float,
    )
    scaled_increment = (
        physical_increment
        / np.asarray(scaling.column_scales, dtype=float)
    )
    block_maxima = {
        "conserved": float(
            np.max(np.abs(scaled_increment[:n_differential]))
        ),
        "primitive": float(
            np.max(
                np.abs(
                    scaled_increment[
                        n_differential : 2 * n_differential
                    ]
                )
            )
        ),
        "face_flux": float(
            np.max(
                np.abs(scaled_increment[2 * n_differential :])
            )
        ),
    }
    consistency = seed_audit["consistency_rank"]["equilibration"]
    descriptor = seed_audit["descriptor_rank"]
    consistent_initial_data = {
        "dimensions": [old_vector.size, old_vector.size],
        "numerical_rank": consistency["numerical_rank"],
        "full_rank": consistency["full_rank"],
        "condition_estimate": consistency["condition_estimate"],
        "descriptor_dimensions": [n_differential, old_vector.size],
        "descriptor_numerical_rank": descriptor["numerical_rank"],
        "descriptor_full_row_rank": bool(
            descriptor["numerical_rank"] == n_differential
        ),
        "maximum_initial_algebraic_residual": (
            seed_audit["maximum_initial_algebraic_residual"]
        ),
        "maximum_scaled_consistency_residual": (
            seed_audit["maximum_scaled_consistency_defect"]
        ),
        "maximum_scaled_tangent_per_s": float(
            np.max(np.abs(tangent))
        ),
        "maximum_scaled_primitive_tangent_per_s": float(
            np.max(np.abs(primitive_tangent))
        ),
        "backend": (
            "exact_18_color_central_jacobian_with_dense_rank_and_"
            "consistency_solve"
        ),
        "passed": seed_audit["passed"],
    }
    tiny_step = {
        "timestep_seconds": timestep,
        "target_scaled_primitive_change": (
            TARGET_SCALED_PRIMITIVE_CHANGES[0]
        ),
        "tangent_predictor_maximum_scaled_change": float(
            np.max(
                np.abs(
                    predictor
                    / np.asarray(scaling.column_scales, dtype=float)
                )
            )
        ),
        "maximum_scaled_change": (
            step.maximum_scaled_total_change
        ),
        "maximum_scaled_block_changes": block_maxima,
        "solver_success": step.accepted,
        "solver_message": step.message,
        "solver_iterations": step.iterations,
        "function_evaluations": step.function_evaluations,
        "jacobian_evaluations": step.jacobian_evaluations,
        "solver_history": [],
        "maximum_scaled_residual": step.maximum_scaled_residual,
        "maximum_scaled_algebraic_residual": (
            step.maximum_scaled_algebraic_residual
        ),
        "conservation_telescoping_relative_defect": (
            step.conservation_telescoping_relative_defect
        ),
        "component_conservation_defects": (
            step.component_conservation_defects
        ),
        "minimum_scattering_optical_depth": (
            step.minimum_scattering_optical_depth
        ),
        "outer_boundary_choked_before": (
            step.outer_boundary_choked_before
        ),
        "outer_boundary_choked_after": (
            step.outer_boundary_choked_after
        ),
        "backend": "equilibrated_sparse_colored_startup",
        "passed": step.accepted,
    }
    initialization = {
        "n_cells": n_cells,
        "unknown_count": int(old_vector.size),
        "residual_count": int(old_vector.size),
        "coordinate": "primary physical increments",
        "temporal_height_scheme": "path_integrated",
        "stream": _stream_summary(context),
        "seed_parameters": seed_parameters,
        "seed_is_stationary_root": False,
        "consistent_initial_data": consistent_initial_data,
        "tiny_step": tiny_step,
        "resolution_passed": bool(
            seed_audit["passed"] and step.accepted
        ),
        "initialization_backend": "certified_colored_sparse",
    }
    return initialization, {
        **artifacts,
        "timestep_seconds": timestep,
        "physical_increment": physical_increment,
        "new_vector": old_vector + physical_increment,
    }


def _source_compatible_initialization(
    n_cells: int,
    *,
    seed_parameters_override: dict | None = None,
    construction_override: dict | None = None,
    use_sparse_colored_initialization: bool = False,
) -> tuple[dict, tuple[dict, dict], dict]:
    if seed_parameters_override is None:
        seed_parameters, construction = (
            _source_compatible_seed_parameters(n_cells)
        )
    else:
        seed_parameters = dict(seed_parameters_override)
        construction = dict(construction_override or {})
    if use_sparse_colored_initialization:
        initialization, artifacts = (
            _sparse_source_compatible_increment_initialization(
                n_cells,
                seed_parameters,
            )
        )
    else:
        initialization, artifacts = _run_increment_primary_resolution(
            n_cells,
            TARGET_SCALED_PRIMITIVE_CHANGES[0],
            include_stream=True,
            seed_kwargs=seed_parameters,
        )
    reference_tiny_step = dict(initialization["tiny_step"])
    base_dt = float(artifacts["timestep_seconds"])
    polish_config = CausalFiveFieldAdaptiveStepConfig(
        minimum_dt=base_dt / 128.0,
        maximum_dt=16.0 * base_dt,
        maximum_scaled_primitive_change=5.0e-4,
        maximum_scaled_total_change=1.0e-3,
        shrink_factor=0.5,
        growth_factor=1.5,
        maximum_retries=0,
        easy_iterations=3,
        residual_tolerance=1.0e-10,
        algebraic_residual_tolerance=1.0e-11,
        conservation_tolerance=1.0e-10,
        finite_difference_step=FINITE_DIFFERENCE_STEP,
        maximum_newton_iterations=12,
    ).validated()
    polished_step = advance_causal_five_field_increment_backward_euler(
        artifacts["context"],
        np.asarray(artifacts["old_vector"], dtype=float),
        base_dt,
        np.asarray(artifacts["physical_increment"], dtype=float),
        polish_config,
    )
    if polished_step.accepted:
        artifacts = {
            **artifacts,
            "physical_increment": np.asarray(
                polished_step.physical_increment,
                dtype=float,
            ),
            "new_vector": np.asarray(
                polished_step.state_vector,
                dtype=float,
            ),
        }
        polished_tiny_step = dict(initialization["tiny_step"])
        polished_tiny_step.update(
            {
                "maximum_scaled_residual": (
                    polished_step.maximum_scaled_residual
                ),
                "maximum_scaled_algebraic_residual": (
                    polished_step.maximum_scaled_algebraic_residual
                ),
                "maximum_scaled_change": (
                    polished_step.maximum_scaled_total_change
                ),
                "maximum_scaled_block_changes": {
                    **polished_tiny_step[
                        "maximum_scaled_block_changes"
                    ],
                    "primitive": (
                        polished_step.maximum_scaled_primitive_change
                    ),
                },
                "solver_success": True,
                "solver_message": polished_step.message,
                "solver_iterations": polished_step.iterations,
                "function_evaluations": (
                    polished_step.function_evaluations
                ),
                "jacobian_evaluations": (
                    polished_step.jacobian_evaluations
                ),
                "conservation_telescoping_relative_defect": (
                    polished_step
                    .conservation_telescoping_relative_defect
                ),
                "minimum_scattering_optical_depth": (
                    polished_step.minimum_scattering_optical_depth
                ),
                "outer_boundary_choked_after": (
                    polished_step.outer_boundary_choked_after
                ),
                "passed": True,
                "backend": "equilibrated_sparse_polish",
            }
        )
        initialization = {
            **initialization,
            "tiny_step": polished_tiny_step,
            "resolution_passed": True,
        }
    context = artifacts["context"]
    old_vector = np.asarray(artifacts["old_vector"], dtype=float)
    old_state = unpack_causal_five_field_state(old_vector, n_cells)
    evaluation = evaluate_causal_five_field_dae(
        old_vector,
        context,
    )
    summary = causal_five_field_state_summary(context, old_vector)
    h_over_r = causal_five_field_h_over_r_profile(
        context,
        old_vector,
    )
    if context.stream_sources is None:
        raise RuntimeError("source-compatible audit lost its stream")
    source_rate = float(np.sum(context.stream_sources.rest_mass))
    throughput_ratio = (
        summary["inner_face_rates"][0] / source_rate
    )
    algebraic_blocks = (
        evaluation.primitive_map_rows,
        evaluation.interior_flux_rows,
        evaluation.inner_flux_rows,
        evaluation.outer_flux_rows,
    )
    maximum_algebraic_residual = float(
        max(np.max(np.abs(block)) for block in algebraic_blocks)
    )
    principal = _inner_principal_summary(
        context,
        old_state.primitives[0],
    )
    gates = {
        "absolute_inner_mass_flux_over_supply": [0.95, 1.05],
        "maximum_h_over_r": 0.25,
        "minimum_scattering_optical_depth": 1.0,
        "inner_incoming_characteristics": 0,
        "outer_channel_choked": False,
        "outer_incoming_characteristics": 2,
        "maximum_initial_algebraic_residual": 1.0e-12,
        "increment_primary_initialization_passed": True,
    }
    passed = bool(
        0.95 <= abs(throughput_ratio) <= 1.05
        and float(np.max(h_over_r)) <= 0.25
        and float(np.min(evaluation.scattering_optical_depths)) > 1.0
        and principal["incoming_inner_characteristics"] == 0
        and not evaluation.outer_boundary_choked
        and evaluation.outer_incoming_characteristics == 2
        and maximum_algebraic_residual <= 1.0e-12
        and initialization["resolution_passed"]
        and polished_step.accepted
    )
    report = {
        "n_cells": n_cells,
        "construction": construction,
        "seed_parameters": seed_parameters,
        "inner_mass_flux_over_supply": throughput_ratio,
        "minimum_scattering_optical_depth": float(
            np.min(evaluation.scattering_optical_depths)
        ),
        "maximum_h_over_r": float(np.max(h_over_r)),
        "inner_h_over_r": float(h_over_r[0]),
        "outer_h_over_r": float(h_over_r[-1]),
        "outer_boundary_choked": evaluation.outer_boundary_choked,
        "outer_incoming_characteristics": (
            evaluation.outer_incoming_characteristics
        ),
        "inner_principal": principal,
        "maximum_initial_algebraic_residual": (
            maximum_algebraic_residual
        ),
        "consistent_initial_data": (
            initialization["consistent_initial_data"]
        ),
        "tiny_step": initialization["tiny_step"],
        "initialization_backend": (
            initialization.get(
                "initialization_backend",
                "dense_reference",
            )
        ),
        "dense_reference_tiny_step": (
            None
            if use_sparse_colored_initialization
            else reference_tiny_step
        ),
        "sparse_reference_tiny_step": (
            reference_tiny_step
            if use_sparse_colored_initialization
            else None
        ),
        "sparse_initial_polish": _fixed_step_summary(polished_step),
        "gates": gates,
        "passed": passed,
        "decision": (
            "source_compatible_initial_datum_passed"
            if passed
            else "source_compatible_initial_datum_failed"
        ),
    }
    return report, (initialization, artifacts), seed_parameters


def _source_compatible_state_audit(
    context: CausalFiveFieldDAEContext,
    vector: np.ndarray,
) -> tuple[dict, bool]:
    n_cells = int(context.grid.centers.size)
    values = np.asarray(vector, dtype=float)
    state = unpack_causal_five_field_state(values, n_cells)
    evaluation = evaluate_causal_five_field_dae(values, context)
    summary = causal_five_field_state_summary(context, values)
    scaling = causal_five_field_dae_scaling(
        state,
        evaluation,
    )
    principal = _inner_principal_summary(
        context,
        state.primitives[0],
    )
    algebraic_blocks = (
        evaluation.primitive_map_rows,
        evaluation.interior_flux_rows,
        evaluation.inner_flux_rows,
        evaluation.outer_flux_rows,
    )
    maximum_raw_algebraic_residual = float(
        max(np.max(np.abs(block)) for block in algebraic_blocks)
    )
    n_differential = 5 * n_cells
    maximum_scaled_algebraic_residual = float(
        np.max(
            np.abs(
                evaluation.residual[n_differential:]
                / scaling.row_scales[n_differential:]
            )
        )
    )
    gates = {
        "maximum_h_over_r": 0.25,
        "minimum_scattering_optical_depth": 1.0,
        "inner_incoming_characteristics": 0,
        "maximum_inner_light_cone_excess": 1.0e-10,
        "outer_channel_choked": False,
        "outer_incoming_characteristics": 2,
        "maximum_scaled_algebraic_map_residual": 1.0e-11,
    }
    passed = bool(
        summary["maximum_h_over_r"] <= gates["maximum_h_over_r"]
        and float(np.min(evaluation.scattering_optical_depths))
        > gates["minimum_scattering_optical_depth"]
        and principal["incoming_inner_characteristics"]
        == gates["inner_incoming_characteristics"]
        and principal["maximum_light_cone_excess"]
        <= gates["maximum_inner_light_cone_excess"]
        and evaluation.outer_boundary_choked
        == gates["outer_channel_choked"]
        and evaluation.outer_incoming_characteristics
        == gates["outer_incoming_characteristics"]
        and maximum_scaled_algebraic_residual
        <= gates["maximum_scaled_algebraic_map_residual"]
    )
    return {
        "state": summary,
        "inner_principal": principal,
        "outer_boundary_choked": evaluation.outer_boundary_choked,
        "outer_incoming_characteristics": (
            evaluation.outer_incoming_characteristics
        ),
        "minimum_scattering_optical_depth": float(
            np.min(evaluation.scattering_optical_depths)
        ),
        "maximum_raw_algebraic_map_residual": (
            maximum_raw_algebraic_residual
        ),
        "maximum_scaled_algebraic_map_residual": (
            maximum_scaled_algebraic_residual
        ),
        "gates": gates,
        "passed": passed,
    }, passed


def _reconstructed_primitive_profile(
    context: CausalFiveFieldDAEContext,
    vector: np.ndarray,
    sample_log_radius: np.ndarray,
) -> np.ndarray:
    state = unpack_causal_five_field_state(
        np.asarray(vector, dtype=float),
        int(context.grid.centers.size),
    )
    log_radius = np.log(context.grid.centers)
    sample = np.asarray(sample_log_radius, dtype=float)
    reconstructed = np.empty((sample.size, 4), dtype=float)
    for field in range(4):
        values = state.primitives[:, field]
        reconstructed[:, field] = PchipInterpolator(
            log_radius,
            values,
            extrapolate=True,
        )(sample)
    return reconstructed


def _pchip_reconstructed_log_h_over_r(
    context: CausalFiveFieldDAEContext,
    vector: np.ndarray,
    sample_log_radius: np.ndarray,
) -> np.ndarray:
    values = np.log(
        causal_five_field_h_over_r_profile(
            context,
            np.asarray(vector, dtype=float),
        )
    )
    return np.asarray(
        PchipInterpolator(
            np.log(context.grid.centers),
            values,
            extrapolate=True,
        )(np.asarray(sample_log_radius, dtype=float)),
        dtype=float,
    )


def _mesh_common_initial_profile_audit(
    left_artifacts: dict,
    right_artifacts: dict,
    construction: dict,
    *,
    left_label: str = "n16",
    right_label: str = "n32",
) -> dict:
    left_context = left_artifacts["context"]
    right_context = right_artifacts["context"]
    left_vector = np.asarray(left_artifacts["old_vector"], dtype=float)
    right_vector = np.asarray(right_artifacts["old_vector"], dtype=float)
    left_n_cells = int(left_context.grid.centers.size)
    right_n_cells = int(right_context.grid.centers.size)
    left_state = unpack_causal_five_field_state(
        left_vector,
        left_n_cells,
    )
    right_state = unpack_causal_five_field_state(
        right_vector,
        right_n_cells,
    )
    sample_log_radius = np.linspace(
        np.log(float(left_context.grid.edges[0])),
        np.log(float(left_context.grid.edges[-1])),
        257,
    )
    left_profile = _reconstructed_primitive_profile(
        left_context,
        left_vector,
        sample_log_radius,
    )
    right_profile = _reconstructed_primitive_profile(
        right_context,
        right_vector,
        sample_log_radius,
    )
    inner_plateau = (
        construction["inner_plateau_radius_rg"]
        * left_context.grid.gravitational_radius
    )
    outer_plateau = (
        construction["outer_plateau_radius_rg"]
        * left_context.grid.gravitational_radius
    )
    radius = np.exp(sample_log_radius)
    coordinate = np.clip(
        (
            np.log(radius / inner_plateau)
            / np.log(outer_plateau / inner_plateau)
        ),
        0.0,
        1.0,
    )
    fraction = coordinate**3 * (
        10.0 - 15.0 * coordinate + 6.0 * coordinate**2
    )
    reference_primitives = (
        (1.0 - fraction[:, None])
        * left_state.primitives[0, :3]
        + fraction[:, None] * left_state.primitives[-1, :3]
    )
    field_names = (
        "log_surface_density",
        "radial_velocity_over_c",
        "azimuthal_velocity_over_c",
        "log_temperature",
    )
    analytic_field_names = field_names[:3]
    cross_defects = np.max(
        np.abs(left_profile - right_profile),
        axis=0,
    )
    left_reference_defects = np.max(
        np.abs(left_profile[:, :3] - reference_primitives),
        axis=0,
    )
    right_reference_defects = np.max(
        np.abs(right_profile[:, :3] - reference_primitives),
        axis=0,
    )
    left_log_h = _pchip_reconstructed_log_h_over_r(
        left_context,
        left_vector,
        sample_log_radius,
    )
    right_log_h = _pchip_reconstructed_log_h_over_r(
        right_context,
        right_vector,
        sample_log_radius,
    )
    reference_log_h = (
        (1.0 - fraction)
        * np.log(
            construction["target_physical_inner_face_h_over_r"]
        )
        + fraction
        * np.log(
            construction["target_physical_outer_face_h_over_r"]
        )
    )
    gates = {
        "maximum_cross_mesh_field_defects": {
            "log_surface_density": 5.0e-2,
            "radial_velocity_over_c": 1.0e-2,
            "azimuthal_velocity_over_c": 1.0e-2,
            "log_temperature": 2.0e-2,
        },
        "maximum_reference_field_defects": {
            "log_surface_density": 5.0e-2,
            "radial_velocity_over_c": 1.0e-2,
            "azimuthal_velocity_over_c": 1.0e-2,
        },
        "maximum_cross_mesh_log_h_over_r_defect": 1.0e-2,
        "maximum_reference_log_h_over_r_defect": 1.0e-2,
        "inner_plateau_kinematic_primitives_bitwise": True,
        "outer_plateau_kinematic_primitives_bitwise": True,
    }
    cross = {
        name: float(value)
        for name, value in zip(
            field_names,
            cross_defects,
            strict=True,
        )
    }
    left_reference = {
        name: float(value)
        for name, value in zip(
            analytic_field_names,
            left_reference_defects,
            strict=True,
        )
    }
    right_reference = {
        name: float(value)
        for name, value in zip(
            analytic_field_names,
            right_reference_defects,
            strict=True,
        )
    }
    inner_bitwise = np.array_equal(
        left_state.primitives[0, :3],
        right_state.primitives[0, :3],
    )
    outer_bitwise = np.array_equal(
        left_state.primitives[-1, :3],
        right_state.primitives[-1, :3],
    )
    maximum_log_h_defect = float(
        np.max(np.abs(left_log_h - right_log_h))
    )
    left_log_h_reference_defect = float(
        np.max(np.abs(left_log_h - reference_log_h))
    )
    right_log_h_reference_defect = float(
        np.max(np.abs(right_log_h - reference_log_h))
    )
    passed = bool(
        inner_bitwise
        and outer_bitwise
        and all(
            cross[name]
            <= gates["maximum_cross_mesh_field_defects"][name]
            for name in field_names
        )
        and all(
            left_reference[name]
            <= gates["maximum_reference_field_defects"][name]
            and right_reference[name]
            <= gates["maximum_reference_field_defects"][name]
            for name in analytic_field_names
        )
        and maximum_log_h_defect
        <= gates["maximum_cross_mesh_log_h_over_r_defect"]
        and left_log_h_reference_defect
        <= gates["maximum_reference_log_h_over_r_defect"]
        and right_log_h_reference_defect
        <= gates["maximum_reference_log_h_over_r_defect"]
    )
    return {
        "method": (
            "shape-preserving cubic reconstruction of one fixed-anchor "
            "analytic kinematic/log-H/R profile on 257 shared "
            "log-radius points"
        ),
        "sample_radius_rg": [
            float(value / left_context.grid.gravitational_radius)
            for value in radius
        ],
        "left_label": left_label,
        "right_label": right_label,
        "left_n_cells": left_n_cells,
        "right_n_cells": right_n_cells,
        "inner_plateau_kinematic_primitives_bitwise": (
            inner_bitwise
        ),
        "outer_plateau_kinematic_primitives_bitwise": (
            outer_bitwise
        ),
        "maximum_cross_mesh_field_defects": cross,
        f"maximum_{left_label}_reference_field_defects": (
            left_reference
        ),
        f"maximum_{right_label}_reference_field_defects": (
            right_reference
        ),
        "maximum_cross_mesh_log_h_over_r_defect": (
            maximum_log_h_defect
        ),
        f"maximum_{left_label}_reference_log_h_over_r_defect": (
            left_log_h_reference_defect
        ),
        f"maximum_{right_label}_reference_log_h_over_r_defect": (
            right_log_h_reference_defect
        ),
        "gates": gates,
        "passed": passed,
    }


def _log_h_over_r_tangent(
    context: CausalFiveFieldDAEContext,
    vector: np.ndarray,
    primitive_tangent: np.ndarray,
) -> np.ndarray:
    n_cells = int(context.grid.centers.size)
    state = unpack_causal_five_field_state(
        np.asarray(vector, dtype=float),
        n_cells,
    )
    tangent = np.asarray(primitive_tangent, dtype=float)
    if tangent.shape != (n_cells, 5):
        raise ValueError("primitive tangent has the wrong shape")
    result = np.empty(n_cells, dtype=float)
    for index, radius in enumerate(context.grid.centers):
        sigma = float(np.exp(state.primitives[index, 0]))
        temperature = float(np.exp(state.primitives[index, 3]))
        derivatives = context.vertical_frequency.eos(
            float(radius)
        ).derivatives(
            sigma,
            temperature,
        )
        result[index] = (
            derivatives.height_log_surface_density
            * tangent[index, 0]
            + derivatives.height_log_temperature
            * tangent[index, 3]
        )
    return result


def _mesh_common_spatial_seed_artifacts(
    n_cells: int,
    seed_parameters: dict,
) -> tuple[dict, dict]:
    context = _context(n_cells, include_stream=True)
    state = make_causal_five_field_seed(
        context,
        **seed_parameters,
    )
    vector = pack_causal_five_field_state(state)
    evaluation = evaluate_causal_five_field_dae(
        vector,
        context,
    )
    scaling = causal_five_field_dae_scaling(
        state,
        evaluation,
    )
    size = vector.size
    zero = np.zeros(size, dtype=float)
    pattern = causal_five_field_dae_jacobian_sparsity(n_cells)

    def scaled_residual(
        scaled_increment: np.ndarray,
        *,
        backward_euler: bool,
    ) -> np.ndarray:
        trial = (
            vector
            + np.asarray(scaling.column_scales, dtype=float)
            * np.asarray(scaled_increment, dtype=float)
        )
        if backward_euler:
            trial_evaluation = evaluate_causal_five_field_dae(
                trial,
                context,
                old_vector=vector,
                timestep_seconds=1.0,
            )
        else:
            trial_evaluation = evaluate_causal_five_field_dae(
                trial,
                context,
            )
        return (
            trial_evaluation.residual
            / np.asarray(scaling.row_scales, dtype=float)
        )

    stationary = causal_five_field_colored_central_jacobian(
        lambda increment: scaled_residual(
            increment,
            backward_euler=False,
        ),
        zero,
        pattern,
        finite_difference_step=FINITE_DIFFERENCE_STEP,
    ).toarray()
    backward_euler = causal_five_field_colored_central_jacobian(
        lambda increment: scaled_residual(
            increment,
            backward_euler=True,
        ),
        zero,
        pattern,
        finite_difference_step=FINITE_DIFFERENCE_STEP,
    ).toarray()
    n_differential = 5 * n_cells
    descriptor_rows = (
        backward_euler - stationary
    )[:n_differential]
    algebraic_tangent = stationary[n_differential:]
    consistency_matrix = np.vstack(
        (descriptor_rows, algebraic_tangent)
    )
    scaled_stationary_residual = (
        evaluation.residual
        / np.asarray(scaling.row_scales, dtype=float)
    )
    right_hand_side = np.concatenate(
        (
            -scaled_stationary_residual[:n_differential],
            np.zeros(size - n_differential, dtype=float),
        )
    )
    scaled_tangent = np.linalg.solve(
        consistency_matrix,
        right_hand_side,
    )
    consistency_defect = float(
        np.max(
            np.abs(
                consistency_matrix @ scaled_tangent
                - right_hand_side
            )
        )
    )
    descriptor_rank = _rank_summary(
        descriptor_rows,
        equilibrate=False,
    )
    consistency_rank = _rank_summary(consistency_matrix)
    state_audit, state_passed = _source_compatible_state_audit(
        context,
        vector,
    )
    maximum_initial_algebraic_residual = float(
        np.max(
            np.abs(
                scaled_stationary_residual[n_differential:]
            )
        )
    )
    passed = bool(
        state_passed
        and descriptor_rank["numerical_rank"] == n_differential
        and consistency_rank["equilibration"]["full_rank"]
        and maximum_initial_algebraic_residual <= 1.0e-12
        and consistency_defect <= 1.0e-10
    )
    return {
        "n_cells": n_cells,
        "state_audit": state_audit,
        "descriptor_rank": descriptor_rank,
        "consistency_rank": consistency_rank,
        "maximum_initial_algebraic_residual": (
            maximum_initial_algebraic_residual
        ),
        "maximum_scaled_consistency_defect": consistency_defect,
        "passed": passed,
    }, {
        "context": context,
        "old_vector": vector,
        "scaling": scaling,
        "stationary_evaluation": evaluation,
        "stationary_scaled_jacobian": stationary,
        "backward_euler_scaled_jacobian": backward_euler,
        "consistent_scaled_tangent": scaled_tangent,
    }


def _mesh_common_semidiscrete_tangent_resolution(
    initialization: dict,
    artifacts: dict,
) -> dict:
    context = artifacts["context"]
    vector = np.asarray(artifacts["old_vector"], dtype=float)
    n_cells = int(context.grid.centers.size)
    n_differential = 5 * n_cells
    scaling = artifacts["scaling"]
    if "stationary_scaled_jacobian" in artifacts:
        stationary_matrix = np.asarray(
            artifacts["stationary_scaled_jacobian"],
            dtype=float,
        )
        backward_euler_matrix = np.asarray(
            artifacts["backward_euler_scaled_jacobian"],
            dtype=float,
        )
        full_scaled_tangent = np.asarray(
            artifacts["consistent_scaled_tangent"],
            dtype=float,
        )
    else:
        stationary_matrix = np.asarray(
            artifacts["stationary_jacobian_audit"].scaled_jacobian,
            dtype=float,
        )
        backward_euler_matrix = np.asarray(
            artifacts[
                "backward_euler_jacobian_audit"
            ].scaled_jacobian,
            dtype=float,
        )
        full_scaled_tangent = np.asarray(
            artifacts[
                "consistent_initial_data_audit"
            ].scaled_tangent,
            dtype=float,
        )
    evaluation = artifacts["stationary_evaluation"]
    state = unpack_causal_five_field_state(vector, n_cells)

    descriptor = (
        backward_euler_matrix - stationary_matrix
    )
    descriptor_rows = descriptor[:n_differential]
    algebraic_tangent = np.asarray(
        stationary_matrix[n_differential:],
        dtype=float,
    )
    consistency_matrix = np.vstack(
        (descriptor_rows, algebraic_tangent)
    )
    transport = (
        evaluation.numerical_weighted_face_fluxes_over_c[1:]
        - evaluation.numerical_weighted_face_fluxes_over_c[:-1]
    )
    residual_terms = {"face_transport": transport}
    residual_terms.update(
        {
            name: -np.asarray(values, dtype=float)
            for name, values in (
                evaluation.integrated_source_components_per_ct.items()
            )
        }
    )
    term_names = tuple(residual_terms)
    row_scale = np.asarray(
        scaling.row_scales[:n_differential],
        dtype=float,
    ).reshape(n_cells, 5)
    right_hand_sides = np.zeros(
        (consistency_matrix.shape[0], len(term_names)),
        dtype=float,
    )
    for index, name in enumerate(term_names):
        right_hand_sides[:n_differential, index] = (
            -np.asarray(residual_terms[name], dtype=float).ravel()
            / row_scale.ravel()
        )
    scaled_component_tangents = np.linalg.solve(
        consistency_matrix,
        right_hand_sides,
    )
    physical_component_tangents = (
        np.asarray(scaling.column_scales, dtype=float)[:, None]
        * scaled_component_tangents
    )
    full_physical_tangent = (
        np.asarray(scaling.column_scales, dtype=float)
        * full_scaled_tangent
    )
    component_sum = np.sum(
        physical_component_tangents,
        axis=1,
    )
    tangent_scale = np.maximum(
        np.abs(full_physical_tangent),
        np.sum(np.abs(physical_component_tangents), axis=1),
    )
    tangent_scale = np.maximum(tangent_scale, 1.0e-300)
    tangent_reconstruction_relative_defect = float(
        np.max(
            np.abs(component_sum - full_physical_tangent)
            / tangent_scale
        )
    )
    residual_reconstruction = np.sum(
        np.asarray(list(residual_terms.values()), dtype=float),
        axis=0,
    )
    residual_scale = np.maximum(
        np.abs(evaluation.conservation_rows),
        np.abs(transport)
        + np.abs(evaluation.integrated_sources_per_ct),
    )
    residual_scale = np.maximum(residual_scale, 1.0)
    residual_reconstruction_relative_defect = float(
        np.max(
            np.abs(
                residual_reconstruction
                - evaluation.conservation_rows
            )
            / residual_scale
        )
    )

    primitive_slice = slice(
        n_differential,
        2 * n_differential,
    )
    full_primitive_tangent = full_physical_tangent[
        primitive_slice
    ].reshape(n_cells, 5)
    full_conserved_tangent = full_physical_tangent[
        :n_differential
    ].reshape(n_cells, 5)
    full_log_h_tangent = _log_h_over_r_tangent(
        context,
        vector,
        full_primitive_tangent,
    )
    components = {}
    for index, name in enumerate(term_names):
        physical = physical_component_tangents[:, index]
        primitive = physical[primitive_slice].reshape(n_cells, 5)
        conserved = physical[:n_differential].reshape(n_cells, 5)
        log_h_tangent = _log_h_over_r_tangent(
            context,
            vector,
            primitive,
        )
        components[name] = {
            "primitive_tangent_per_s": primitive,
            "conserved_tangent_per_s": conserved,
            "log_h_over_r_tangent_per_s": log_h_tangent,
            "maximum_absolute_log_h_over_r_tangent_per_s": float(
                np.max(np.abs(log_h_tangent))
            ),
            "primitive_field_maxima_per_s": {
                name: float(value)
                for name, value in zip(
                    (
                        "log_surface_density",
                        "radial_velocity_over_c",
                        "azimuthal_velocity_over_c",
                        "log_temperature",
                        "specific_stress",
                    ),
                    np.max(np.abs(primitive), axis=0),
                    strict=True,
                )
            },
        }
    passed = bool(
        initialization["passed"]
        and residual_reconstruction_relative_defect <= 1.0e-12
        and tangent_reconstruction_relative_defect <= 5.0e-10
    )
    return {
        "n_cells": n_cells,
        "radius_rg": (
            np.asarray(context.grid.centers, dtype=float)
            / context.grid.gravitational_radius
        ),
        "cell_measures": np.asarray(
            context.grid.cell_measures,
            dtype=float,
        ),
        "grid_edges_rg": (
            np.asarray(context.grid.edges, dtype=float)
            / context.grid.gravitational_radius
        ),
        "full": {
            "primitive_tangent_per_s": full_primitive_tangent,
            "conserved_tangent_per_s": full_conserved_tangent,
            "log_h_over_r_tangent_per_s": full_log_h_tangent,
            "maximum_absolute_log_h_over_r_tangent_per_s": float(
                np.max(np.abs(full_log_h_tangent))
            ),
        },
        "components": components,
        "term_names": list(term_names),
        "residual_reconstruction_relative_defect": (
            residual_reconstruction_relative_defect
        ),
        "tangent_reconstruction_relative_defect": (
            tangent_reconstruction_relative_defect
        ),
        "passed": passed,
    }


def _linear_center_reconstruction(
    log_radius: np.ndarray,
    values: np.ndarray,
    sample_log_radius: np.ndarray,
) -> np.ndarray:
    centers = np.asarray(log_radius, dtype=float)
    field = np.asarray(values, dtype=float)
    sample = np.asarray(sample_log_radius, dtype=float)
    reconstructed = np.interp(sample, centers, field)
    left = sample < centers[0]
    right = sample > centers[-1]
    reconstructed[left] = (
        field[0]
        + (field[1] - field[0])
        / (centers[1] - centers[0])
        * (sample[left] - centers[0])
    )
    reconstructed[right] = (
        field[-1]
        + (field[-1] - field[-2])
        / (centers[-1] - centers[-2])
        * (sample[right] - centers[-1])
    )
    return reconstructed


def _tangent_field_difference(
    first: np.ndarray,
    second: np.ndarray,
    radius_rg: np.ndarray,
) -> dict:
    first_values = np.asarray(first, dtype=float)
    second_values = np.asarray(second, dtype=float)
    difference = first_values - second_values
    maximum_index = int(np.argmax(np.abs(difference)))
    amplitude = max(
        float(np.max(np.abs(first_values))),
        float(np.max(np.abs(second_values))),
        np.finfo(float).tiny,
    )
    return {
        "maximum_absolute_difference_per_s": float(
            np.max(np.abs(difference))
        ),
        "rms_difference_per_s": float(
            np.sqrt(np.mean(difference**2))
        ),
        "maximum_difference_relative_to_profile_amplitude": float(
            np.max(np.abs(difference)) / amplitude
        ),
        "maximum_difference_radius_rg": float(
            np.asarray(radius_rg, dtype=float)[maximum_index]
        ),
    }


def _mesh_common_semidiscrete_tangent_comparison(
    n16: dict,
    n32: dict,
) -> dict:
    coarse_edges = np.asarray(n16["grid_edges_rg"], dtype=float)
    fine_edges = np.asarray(n32["grid_edges_rg"], dtype=float)
    nested = bool(np.array_equal(coarse_edges, fine_edges[::2]))
    coarse_measures = np.asarray(n16["cell_measures"], dtype=float)
    fine_measures = np.asarray(n32["cell_measures"], dtype=float)
    restriction_weights = fine_measures.reshape(16, 2)

    def restrict(values: np.ndarray) -> np.ndarray:
        fine = np.asarray(values, dtype=float)
        reshaped = fine.reshape((16, 2) + fine.shape[1:])
        weights = restriction_weights
        while weights.ndim < reshaped.ndim:
            weights = weights[..., None]
        return np.sum(weights * reshaped, axis=1) / (
            coarse_measures.reshape(
                (16,) + (1,) * (fine.ndim - 1)
            )
        )

    sample_log_radius = np.linspace(
        np.log(coarse_edges[0]),
        np.log(coarse_edges[-1]),
        257,
    )
    sample_radius = np.exp(sample_log_radius)
    coarse_log_radius = np.log(
        np.asarray(n16["radius_rg"], dtype=float)
    )
    fine_log_radius = np.log(
        np.asarray(n32["radius_rg"], dtype=float)
    )

    def compare_log_h(
        coarse_values: np.ndarray,
        fine_values: np.ndarray,
    ) -> dict:
        coarse = np.asarray(coarse_values, dtype=float)
        fine = np.asarray(fine_values, dtype=float)
        restricted = restrict(fine)
        linear_coarse = _linear_center_reconstruction(
            coarse_log_radius,
            coarse,
            sample_log_radius,
        )
        linear_fine = _linear_center_reconstruction(
            fine_log_radius,
            fine,
            sample_log_radius,
        )
        pchip_coarse = PchipInterpolator(
            coarse_log_radius,
            coarse,
            extrapolate=True,
        )(sample_log_radius)
        pchip_fine = PchipInterpolator(
            fine_log_radius,
            fine,
            extrapolate=True,
        )(sample_log_radius)
        return {
            "fine_to_coarse_cell_average": _tangent_field_difference(
                coarse,
                restricted,
                np.asarray(n16["radius_rg"], dtype=float),
            ),
            "log_linear_shared_radius": _tangent_field_difference(
                linear_coarse,
                linear_fine,
                sample_radius,
            ),
            "pchip_shared_radius": _tangent_field_difference(
                pchip_coarse,
                pchip_fine,
                sample_radius,
            ),
            "n16_reconstruction_spread": _tangent_field_difference(
                linear_coarse,
                pchip_coarse,
                sample_radius,
            ),
            "n32_reconstruction_spread": _tangent_field_difference(
                linear_fine,
                pchip_fine,
                sample_radius,
            ),
        }

    full_log_h = compare_log_h(
        n16["full"]["log_h_over_r_tangent_per_s"],
        n32["full"]["log_h_over_r_tangent_per_s"],
    )
    component_log_h = {
        name: compare_log_h(
            n16["components"][name][
                "log_h_over_r_tangent_per_s"
            ],
            n32["components"][name][
                "log_h_over_r_tangent_per_s"
            ],
        )
        for name in n16["term_names"]
    }
    coarse_conserved = np.asarray(
        n16["full"]["conserved_tangent_per_s"],
        dtype=float,
    )
    restricted_conserved = restrict(
        n32["full"]["conserved_tangent_per_s"]
    )
    conserved_restriction = {
        name: _tangent_field_difference(
            coarse_conserved[:, index],
            restricted_conserved[:, index],
            np.asarray(n16["radius_rg"], dtype=float),
        )
        for index, name in enumerate(FIELD_NAMES)
    }
    component_ranking = sorted(
        (
            {
                "term": name,
                **metrics["fine_to_coarse_cell_average"],
            }
            for name, metrics in component_log_h.items()
        ),
        key=lambda row: row["maximum_absolute_difference_per_s"],
        reverse=True,
    )
    return {
        "grids_are_exactly_nested": nested,
        "full_log_h_over_r_tangent": full_log_h,
        "component_log_h_over_r_tangent": component_log_h,
        "component_cell_average_difference_ranking": component_ranking,
        "full_conserved_tangent_fine_to_coarse": (
            conserved_restriction
        ),
        "passed": bool(
            nested
            and n16["passed"]
            and n32["passed"]
        ),
    }


def _mesh_common_transport_flux_resolution(
    n_cells: int,
    seed_parameters: dict,
) -> tuple[dict, dict]:
    """Evaluate the production Rusanov split on one analytic-profile mesh."""

    context = _context(n_cells, include_stream=True)
    state = make_causal_five_field_seed(
        context,
        **seed_parameters,
    )
    evaluation = evaluate_causal_five_field_dae(
        pack_causal_five_field_state(state),
        context,
    )
    central = np.asarray(
        evaluation.central_weighted_face_fluxes_over_c,
        dtype=float,
    )
    dissipation = np.asarray(
        evaluation.rusanov_dissipation_weighted_face_fluxes_over_c,
        dtype=float,
    )
    numerical = np.asarray(
        evaluation.numerical_weighted_face_fluxes_over_c,
        dtype=float,
    )
    scale = np.maximum(np.abs(numerical), 1.0)
    reconstruction_defect = float(
        np.max(
            np.abs(central + dissipation - numerical)
            / scale
        )
    )
    passed = bool(
        reconstruction_defect <= 5.0e-15
        and np.all(dissipation[[0, -1]] == 0.0)
    )
    return {
        "n_cells": n_cells,
        "maximum_relative_flux_split_reconstruction_defect": (
            reconstruction_defect
        ),
        "boundary_dissipation_is_exactly_zero": bool(
            np.all(dissipation[[0, -1]] == 0.0)
        ),
        "passed": passed,
    }, {
        "grid_edges_rg": (
            np.asarray(context.grid.edges, dtype=float)
            / context.grid.gravitational_radius
        ),
        "central": central,
        "rusanov_dissipation": dissipation,
        "total": numerical,
        "source_components": {
            name: np.asarray(values, dtype=float)
            for name, values in (
                evaluation.integrated_source_components_per_ct.items()
            )
        },
    }


def _mesh_common_transport_pair_error(
    coarse: dict,
    fine: dict,
) -> dict:
    """Compare one nested pair using shared faces and cell balances."""

    coarse_edges = np.asarray(coarse["grid_edges_rg"], dtype=float)
    fine_edges = np.asarray(fine["grid_edges_rg"], dtype=float)
    n_coarse = coarse_edges.size - 1
    nested = bool(
        fine_edges.size == 2 * n_coarse + 1
        and np.array_equal(coarse_edges, fine_edges[::2])
    )
    if not nested:
        raise RuntimeError("manufactured transport grids are not nested")
    field_scale = np.maximum(
        np.max(np.abs(fine["central"][1:-1]), axis=0),
        np.finfo(float).tiny,
    )

    def metrics(name: str) -> dict:
        coarse_faces = np.asarray(coarse[name], dtype=float)
        fine_faces = np.asarray(fine[name], dtype=float)
        shared_difference = (
            coarse_faces[1:-1] - fine_faces[2:-1:2]
        )
        normalized_shared = shared_difference / field_scale
        coarse_balance = coarse_faces[1:] - coarse_faces[:-1]
        fine_balance = fine_faces[1:] - fine_faces[:-1]
        restricted_fine_balance = np.sum(
            fine_balance.reshape(n_coarse, 2, 5),
            axis=1,
        )
        balance_difference = (
            coarse_balance[1:-1]
            - restricted_fine_balance[1:-1]
        )
        normalized_balance = balance_difference / field_scale
        return {
            "shared_face_scaled_l2_error": float(
                np.sqrt(np.mean(normalized_shared**2))
            ),
            "shared_face_scaled_linf_error": float(
                np.max(np.abs(normalized_shared))
            ),
            "cell_balance_scaled_l2_error": float(
                np.sqrt(np.mean(normalized_balance**2))
            ),
            "cell_balance_scaled_linf_error": float(
                np.max(np.abs(normalized_balance))
            ),
            "shared_face_field_linf_errors": {
                field: float(value)
                for field, value in zip(
                    FIELD_NAMES,
                    np.max(np.abs(normalized_shared), axis=0),
                    strict=True,
                )
            },
            "cell_balance_field_linf_errors": {
                field: float(value)
                for field, value in zip(
                    FIELD_NAMES,
                    np.max(np.abs(normalized_balance), axis=0),
                    strict=True,
                )
            },
        }

    return {
        "coarse_cells": n_coarse,
        "fine_cells": 2 * n_coarse,
        "grids_are_exactly_nested": nested,
        "central": metrics("central"),
        "rusanov_dissipation": metrics("rusanov_dissipation"),
        "total": metrics("total"),
    }


def _observed_pair_orders(
    pair_errors: list[dict],
    component: str,
    metric: str,
) -> list[float]:
    errors = [
        float(pair[component][metric])
        for pair in pair_errors
    ]
    return [
        float(np.log2(first / second))
        for first, second in zip(errors[:-1], errors[1:], strict=True)
        if first > 0.0 and second > 0.0
    ]


def _mesh_common_source_pair_error(
    coarse: dict,
    fine: dict,
) -> dict:
    """Compare integrated production sources on one nested grid pair."""

    coarse_edges = np.asarray(coarse["grid_edges_rg"], dtype=float)
    fine_edges = np.asarray(fine["grid_edges_rg"], dtype=float)
    n_coarse = coarse_edges.size - 1
    nested = bool(
        fine_edges.size == 2 * n_coarse + 1
        and np.array_equal(coarse_edges, fine_edges[::2])
    )
    if not nested:
        raise RuntimeError("manufactured source grids are not nested")
    result = {
        "coarse_cells": n_coarse,
        "fine_cells": 2 * n_coarse,
        "grids_are_exactly_nested": nested,
        "components": {},
    }
    for name, coarse_values in coarse["source_components"].items():
        coarse_component = np.asarray(coarse_values, dtype=float)
        fine_component = np.asarray(
            fine["source_components"][name],
            dtype=float,
        )
        restricted = np.sum(
            fine_component.reshape(n_coarse, 2, 5),
            axis=1,
        )
        difference = (
            coarse_component[1:-1] - restricted[1:-1]
        )
        amplitude_l2 = max(
            float(np.linalg.norm(coarse_component[1:-1])),
            float(np.linalg.norm(restricted[1:-1])),
            np.finfo(float).tiny,
        )
        amplitude_linf = max(
            float(np.max(np.abs(coarse_component[1:-1]))),
            float(np.max(np.abs(restricted[1:-1]))),
            np.finfo(float).tiny,
        )
        result["components"][name] = {
            "scaled_l2_error": float(
                np.linalg.norm(difference) / amplitude_l2
            ),
            "scaled_linf_error": float(
                np.max(np.abs(difference)) / amplitude_linf
            ),
            "maximum_absolute_error": float(
                np.max(np.abs(difference))
            ),
        }
    return result


def _mesh_common_manufactured_transport_convergence(
    seed_parameters: dict,
) -> dict:
    """Certify the declared central-plus-Rusanov spatial order."""

    resolutions = (16, 32, 64, 128)
    summaries = {}
    artifacts = {}
    for n_cells in resolutions:
        summary, artifact = _mesh_common_transport_flux_resolution(
            n_cells,
            seed_parameters,
        )
        summaries[str(n_cells)] = summary
        artifacts[n_cells] = artifact
    pair_errors = [
        _mesh_common_transport_pair_error(
            artifacts[coarse],
            artifacts[2 * coarse],
        )
        for coarse in resolutions[:-1]
    ]
    source_pair_errors = [
        _mesh_common_source_pair_error(
            artifacts[coarse],
            artifacts[2 * coarse],
        )
        for coarse in resolutions[:-1]
    ]
    metrics = (
        "shared_face_scaled_l2_error",
        "cell_balance_scaled_l2_error",
    )
    observed_orders = {
        component: {
            metric: _observed_pair_orders(
                pair_errors,
                component,
                metric,
            )
            for metric in metrics
        }
        for component in (
            "central",
            "rusanov_dissipation",
            "total",
        )
    }
    central_orders = [
        value
        for metric in metrics
        for value in observed_orders["central"][metric]
    ]
    dissipation_orders = [
        value
        for metric in metrics
        for value in observed_orders[
            "rusanov_dissipation"
        ][metric]
    ]
    total_orders = [
        value
        for metric in metrics
        for value in observed_orders["total"][metric]
    ]
    source_observed_orders = {
        name: _observed_pair_orders(
            [
                {
                    "source": pair["components"][name],
                }
                for pair in source_pair_errors
            ],
            "source",
            "scaled_l2_error",
        )
        for name in artifacts[16]["source_components"]
        if name != "stream"
    }
    local_source_names = (
        "perfect_fluid_geometry",
        "stress_geometry",
        "radiative_cooling",
    )
    derivative_source_names = (
        "vertical_work",
        "stress_relaxation",
    )
    local_source_orders = [
        value
        for name in local_source_names
        for value in source_observed_orders[name]
    ]
    derivative_source_orders = [
        value
        for name in derivative_source_names
        for value in source_observed_orders[name]
    ]
    maximum_stream_scaled_linf_error = max(
        pair["components"]["stream"]["scaled_linf_error"]
        for pair in source_pair_errors
    )
    gates = {
        "all_flux_splits_reconstruct": all(
            summary["passed"] for summary in summaries.values()
        ),
        "minimum_central_observed_order": (
            min(central_orders) if central_orders else float("-inf")
        ),
        "minimum_rusanov_dissipation_observed_order": (
            min(dissipation_orders)
            if dissipation_orders
            else float("-inf")
        ),
        "minimum_total_observed_order": (
            min(total_orders) if total_orders else float("-inf")
        ),
        "required_minimum_central_order": 1.5,
        "required_minimum_rusanov_dissipation_order": 0.75,
        "required_minimum_total_order": 0.75,
        "minimum_local_source_observed_order": (
            min(local_source_orders)
            if local_source_orders
            else float("-inf")
        ),
        "minimum_derivative_source_observed_order": (
            min(derivative_source_orders)
            if derivative_source_orders
            else float("-inf")
        ),
        "maximum_stream_scaled_linf_error": (
            maximum_stream_scaled_linf_error
        ),
        "required_minimum_local_source_order": 1.5,
        "required_minimum_derivative_source_order": 0.75,
        "required_maximum_stream_scaled_linf_error": 5.0e-13,
    }
    passed = bool(
        gates["all_flux_splits_reconstruct"]
        and gates["minimum_central_observed_order"]
        >= gates["required_minimum_central_order"]
        and gates[
            "minimum_rusanov_dissipation_observed_order"
        ]
        >= gates["required_minimum_rusanov_dissipation_order"]
        and gates["minimum_total_observed_order"]
        >= gates["required_minimum_total_order"]
        and gates["minimum_local_source_observed_order"]
        >= gates["required_minimum_local_source_order"]
        and gates["minimum_derivative_source_observed_order"]
        >= gates["required_minimum_derivative_source_order"]
        and gates["maximum_stream_scaled_linf_error"]
        <= gates["required_maximum_stream_scaled_linf_error"]
    )
    return {
        "profile": (
            "the same fixed-anchor C2 continuum primitive profile used "
            "by the mesh-common causal startup"
        ),
        "scope": (
            "operator-only face-flux evaluation; N64 and N128 are not "
            "physical evolution runs"
        ),
        "resolutions": list(resolutions),
        "resolution_summaries": summaries,
        "nested_pair_errors": pair_errors,
        "nested_source_pair_errors": source_pair_errors,
        "observed_orders": observed_orders,
        "source_observed_orders": source_observed_orders,
        "gates": gates,
        "passed": passed,
        "interpretation": (
            "declared first-order Rusanov truncation is demonstrated"
            if passed
            else "the production face operator fails its declared order"
        ),
    }


def _run_mesh_common_spatial_response_audit(
    args: argparse.Namespace,
) -> None:
    common_parameters, construction = (
        _mesh_common_source_compatible_seed_parameters()
    )
    n16_initial, n16_artifacts = (
        _mesh_common_spatial_seed_artifacts(
            16,
            common_parameters,
        )
    )
    n32_initial, n32_artifacts = (
        _mesh_common_spatial_seed_artifacts(
            32,
            common_parameters,
        )
    )
    profile = _mesh_common_initial_profile_audit(
        n16_artifacts,
        n32_artifacts,
        construction,
    )
    prerequisites_passed = bool(
        n16_initial["passed"]
        and n32_initial["passed"]
        and profile["passed"]
    )
    n16 = None
    n32 = None
    comparison = None
    manufactured = None
    if prerequisites_passed:
        n16 = _mesh_common_semidiscrete_tangent_resolution(
            n16_initial,
            n16_artifacts,
        )
        n32 = _mesh_common_semidiscrete_tangent_resolution(
            n32_initial,
            n32_artifacts,
        )
        if n16["passed"] and n32["passed"]:
            comparison = (
                _mesh_common_semidiscrete_tangent_comparison(
                    n16,
                    n32,
                )
            )
            if comparison["passed"]:
                manufactured = (
                    _mesh_common_manufactured_transport_convergence(
                        common_parameters,
                    )
                )
    passed = bool(
        prerequisites_passed
        and comparison is not None
        and comparison["passed"]
        and manufactured is not None
        and manufactured["passed"]
    )
    output = {
        "work_package": "WP10c5r",
        "scope": (
            "no-evolution term-resolved semidiscrete response on the "
            "fixed mesh-common causal datum"
        ),
        "construction": construction,
        "common_seed_parameters": common_parameters,
        "initial_profile_mesh_audit": profile,
        "n16": n16,
        "n32": n32,
        "mesh_comparison": comparison,
        "manufactured_transport_convergence": manufactured,
        "gates": {
            "common_initial_data_passed": prerequisites_passed,
            "n16_tangent_decomposition_passed": bool(
                n16 is not None and n16["passed"]
            ),
            "n32_tangent_decomposition_passed": bool(
                n32 is not None and n32["passed"]
            ),
            "nested_cell_average_comparison_passed": bool(
                comparison is not None and comparison["passed"]
            ),
            "manufactured_spatial_response_complete": bool(
                manufactured is not None
            ),
            "manufactured_spatial_response_passed": bool(
                manufactured is not None and manufactured["passed"]
            ),
            "operator_correction_authorized": False,
            "operator_correction_applied": False,
            "n64_physical_evolution_executed": False,
            "n64_physical_evolution_authorized_for_next_wp": passed,
            "long_evolution_authorized": False,
            "tide_authorized": False,
            "wind_authorized": False,
        },
        "passed": passed,
        "decision": (
            "ordinary_first_order_rusanov_truncation_quantified"
            if passed
            else "stop_before_evolution_and_review_face_operator"
        ),
    }
    output_path = _absolute(
        DEFAULT_MESH_COMMON_SPATIAL_RESPONSE_OUTPUT
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


def _source_compatible_consistency_rank_audit(
    context: CausalFiveFieldDAEContext,
    vector: np.ndarray,
) -> dict:
    n_cells = int(context.grid.centers.size)
    values = np.asarray(vector, dtype=float)
    state = unpack_causal_five_field_state(values, n_cells)
    stationary_evaluation = evaluate_causal_five_field_dae(
        values,
        context,
    )
    scaling = causal_five_field_dae_scaling(
        state,
        stationary_evaluation,
    )
    pattern = causal_five_field_dae_jacobian_sparsity(n_cells)
    zero = np.zeros_like(values)

    def scaled_residual(
        scaled_increment: np.ndarray,
        *,
        backward_euler: bool,
    ) -> np.ndarray:
        trial = (
            values
            + scaling.column_scales
            * np.asarray(scaled_increment, dtype=float)
        )
        if backward_euler:
            evaluation = evaluate_causal_five_field_dae(
                trial,
                context,
                old_vector=values,
                timestep_seconds=1.0,
            )
        else:
            evaluation = evaluate_causal_five_field_dae(
                trial,
                context,
            )
        return evaluation.residual / scaling.row_scales

    stationary = causal_five_field_colored_central_jacobian(
        lambda increment: scaled_residual(
            increment,
            backward_euler=False,
        ),
        zero,
        pattern,
        finite_difference_step=FINITE_DIFFERENCE_STEP,
    ).toarray()
    backward_euler = causal_five_field_colored_central_jacobian(
        lambda increment: scaled_residual(
            increment,
            backward_euler=True,
        ),
        zero,
        pattern,
        finite_difference_step=FINITE_DIFFERENCE_STEP,
    ).toarray()
    n_differential = 5 * n_cells
    descriptor_rows = (
        backward_euler - stationary
    )[:n_differential]
    algebraic_tangent = stationary[n_differential:]
    consistency = np.vstack(
        (descriptor_rows, algebraic_tangent)
    )
    descriptor = _rank_summary(
        descriptor_rows,
        equilibrate=False,
    )
    complete = _rank_summary(consistency)
    passed = bool(
        descriptor["numerical_rank"] == n_differential
        and complete["equilibration"]["full_rank"]
    )
    return {
        "descriptor": descriptor,
        "consistency": complete,
        "passed": passed,
    }


def _aggregate_physical_step_ledgers(
    ledgers: list[CausalFiveFieldPhysicalStepLedger],
) -> dict:
    term_names = (
        "conserved_storage_change",
        "vertical_storage_change",
        "boundary_transport",
        "endogenous_source",
        "prescribed_stream_source",
    )
    aggregated = {
        name: np.asarray(
            [
                math.fsum(
                    float(getattr(ledger, name)[field])
                    for ledger in ledgers
                )
                for field in range(5)
            ],
            dtype=float,
        )
        for name in term_names
    }
    balance = (
        aggregated["conserved_storage_change"]
        + aggregated["vertical_storage_change"]
        + aggregated["boundary_transport"]
        - aggregated["endogenous_source"]
        - aggregated["prescribed_stream_source"]
    )
    summed_closure = np.asarray(
        [
            math.fsum(
                float(ledger.closure_defect[field])
                for ledger in ledgers
            )
            for field in range(5)
        ],
        dtype=float,
    )
    absolute_term_scale = np.asarray(
        [
            math.fsum(
                abs(float(getattr(ledger, name)[field]))
                for ledger in ledgers
                for name in term_names
            )
            for field in range(5)
        ],
        dtype=float,
    )
    relative_defect = np.abs(balance) / np.maximum(
        absolute_term_scale,
        1.0,
    )
    return {
        "step_count": len(ledgers),
        "terms": {
            name: [float(value) for value in values]
            for name, values in aggregated.items()
        },
        "balance_defect": [float(value) for value in balance],
        "summed_step_closure_defect": [
            float(value) for value in summed_closure
        ],
        "balance_identity_maximum_absolute_defect": float(
            np.max(np.abs(balance - summed_closure))
        ),
        "relative_balance_defect": [
            float(value) for value in relative_defect
        ],
        "maximum_relative_balance_defect": float(
            np.max(relative_defect)
        ),
    }


def _run_source_compatible_startup_audit(
    args: argparse.Namespace,
) -> None:
    n16_initial, n16_bundle, n16_parameters = (
        _source_compatible_initialization(16)
    )
    n16_repeated = None
    n16_repeated_passed = False
    if n16_initial["passed"]:
        n16_repeated, n16_repeated_passed = (
            _run_repeated_source_on_resolution(
                16,
                accepted_step_target=8,
                elapsed_time_target=None,
                perform_restart_resume_audit=True,
                seed_kwargs=n16_parameters,
                restart_label="wp10c5m",
                restart_work_package="WP10c5m",
                initialization_bundle=n16_bundle,
                step_residual_tolerance=1.0e-10,
                step_algebraic_tolerance=1.0e-11,
                mass_budget_tolerance=1.0e-10,
            )
        )

    n32_initial = None
    n32_repeated = None
    n32_repeated_passed = False
    mesh = None
    if n16_repeated_passed:
        n32_initial, n32_bundle, n32_parameters = (
            _source_compatible_initialization(32)
        )
        if n32_initial["passed"]:
            n32_repeated, n32_repeated_passed = (
                _run_repeated_source_on_resolution(
                    32,
                    accepted_step_target=None,
                    elapsed_time_target=(
                        n16_repeated["elapsed_time_seconds"]
                    ),
                    perform_restart_resume_audit=False,
                    seed_kwargs=n32_parameters,
                    restart_label="wp10c5m",
                    restart_work_package="WP10c5m",
                    initialization_bundle=n32_bundle,
                    step_residual_tolerance=1.0e-10,
                    step_algebraic_tolerance=1.0e-11,
                    mass_budget_tolerance=1.0e-10,
                )
            )
    if n32_repeated_passed:
        mesh = _repeated_mesh_comparison(
            n16_repeated,
            n32_repeated,
        )
    passed = bool(
        n16_initial["passed"]
        and n16_repeated_passed
        and n32_initial is not None
        and n32_initial["passed"]
        and n32_repeated_passed
        and mesh is not None
        and mesh["passed"]
    )
    output = {
        "work_package": "WP10c5m",
        "scope": (
            "co-tuned source-compatible causal datum and short adaptive "
            "exact-stream no-tide startup"
        ),
        "n16_initial_datum": n16_initial,
        "n16_repeated_startup": n16_repeated,
        "n32_initial_datum": n32_initial,
        "n32_repeated_startup": n32_repeated,
        "mesh_comparison": mesh,
        "gates": {
            "n16_initial_datum_passed": n16_initial["passed"],
            "n16_repeated_startup_passed": n16_repeated_passed,
            "n32_initial_datum_attempted": n32_initial is not None,
            "n32_initial_datum_passed": (
                n32_initial["passed"]
                if n32_initial is not None
                else False
            ),
            "n32_repeated_startup_passed": n32_repeated_passed,
            "mesh_gate_passed": (
                mesh["passed"] if mesh is not None else False
            ),
            "source_compatible_short_startup_certified": passed,
            "duration_extension_authorized": passed,
            "long_evolution_certified": False,
            "hot_state_certified": False,
            "limit_cycle_certified": False,
            "tide_authorized": False,
            "wind_authorized": False,
        },
        "decision": (
            "source_compatible_short_startup_mesh_gate_passed"
            if passed
            else "stop_before_duration_extension"
        ),
    }
    output_path = _absolute(
        DEFAULT_SOURCE_COMPATIBLE_STARTUP_OUTPUT
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


def _continue_source_compatible_duration_resolution(
    n_cells: int,
    *,
    initialization: dict,
    artifacts: dict,
    seed_parameters: dict,
    elapsed_time_target: float,
    perform_first_step_replay_audit: bool,
    parent_restart_label: str = "wp10c5m",
    output_restart_label: str = "wp10c5n",
    work_package: str = "WP10c5n",
    parent_work_package: str = "WP10c5m",
    initial_dt_next_override: float | None = None,
    maximum_dt_override: float | None = None,
    step_residual_tolerance: float = 1.0e-10,
) -> tuple[dict, bool]:
    context = artifacts["context"]
    initial_vector = np.asarray(artifacts["old_vector"], dtype=float)
    base_dt = float(artifacts["timestep_seconds"])
    restart_path = (
        DEFAULT_RESTART_DIRECTORY
        / (
            f"causal_{parent_restart_label}_N"
            f"{n_cells:03d}_final.npz"
        )
    )
    if not restart_path.exists():
        return {
            "n_cells": n_cells,
            "restart_path": str(restart_path.relative_to(ROOT)),
            "passed": False,
            "decision": "required_parent_restart_missing",
        }, False
    checkpoint = load_causal_five_field_adaptive_restart(
        restart_path,
        context,
    )
    checkpoint_audit, checkpoint_state_passed = (
        _source_compatible_state_audit(
            context,
            checkpoint.state_vector,
        )
    )
    checkpoint_rank = _source_compatible_consistency_rank_audit(
        context,
        checkpoint.state_vector,
    )
    checkpoint_provenance_passed = bool(
        checkpoint.provenance.get("n_cells") == n_cells
        and checkpoint.provenance.get("work_package")
        == parent_work_package
        and "exact circularized regression stream"
        in str(checkpoint.provenance.get("source", ""))
    )
    checkpoint_passed = bool(
        initialization["resolution_passed"]
        and checkpoint_state_passed
        and checkpoint_rank["passed"]
        and checkpoint_provenance_passed
        and checkpoint.elapsed_time < elapsed_time_target
    )
    if not checkpoint_passed:
        return {
            "n_cells": n_cells,
            "restart_path": str(restart_path.relative_to(ROOT)),
            "checkpoint": {
                "elapsed_time_seconds": checkpoint.elapsed_time,
                "provenance": checkpoint.provenance,
                "provenance_passed": checkpoint_provenance_passed,
                "state_audit": checkpoint_audit,
                "rank_audit": checkpoint_rank,
            },
            "target_elapsed_time_seconds": elapsed_time_target,
            "passed": False,
            "decision": "parent_checkpoint_gate_failed",
        }, False

    maximum_dt = (
        16.0 * base_dt
        if maximum_dt_override is None
        else float(maximum_dt_override)
    )
    config = CausalFiveFieldAdaptiveStepConfig(
        minimum_dt=base_dt / 128.0,
        maximum_dt=maximum_dt,
        maximum_scaled_primitive_change=5.0e-4,
        maximum_scaled_total_change=1.0e-3,
        shrink_factor=0.5,
        growth_factor=1.5,
        maximum_retries=6,
        easy_iterations=3,
        residual_tolerance=step_residual_tolerance,
        algebraic_residual_tolerance=1.0e-11,
        conservation_tolerance=1.0e-10,
        finite_difference_step=FINITE_DIFFERENCE_STEP,
        maximum_newton_iterations=12,
    ).validated()
    if context.stream_sources is None:
        raise RuntimeError("duration extension requires a stream")
    source_rate = float(np.sum(context.stream_sources.rest_mass))
    loading_time = causal_five_field_loading_time(
        context,
        initial_vector,
    )
    state_vector = np.asarray(
        checkpoint.state_vector,
        dtype=float,
    )
    previous_increment = np.asarray(
        checkpoint.previous_physical_increment,
        dtype=float,
    )
    elapsed_time = float(checkpoint.elapsed_time)
    checkpoint_elapsed_time = elapsed_time
    dt_next = (
        float(checkpoint.dt_next)
        if initial_dt_next_override is None
        else float(initial_dt_next_override)
    )
    previous_dt = float(checkpoint.previous_dt)
    accepted_steps = int(checkpoint.accepted_steps)
    rejected_attempts = int(checkpoint.rejected_attempts)
    extension_accepted_steps = 0
    extension_rejected_attempts = 0
    first_step_replay_bitwise = not perform_first_step_replay_audit
    first_step_replay_completed = False
    step_rows: list[dict] = []
    physical_ledgers: list[CausalFiveFieldPhysicalStepLedger] = []
    actual_mass_increments: list[float] = []
    expected_mass_increments: list[float] = []
    all_step_gates_passed = True
    terminal_message = "target reached"
    target_tolerance = max(
        1.0e-20,
        5.0e-14 * elapsed_time_target,
    )

    while True:
        remaining = elapsed_time_target - elapsed_time
        if abs(remaining) <= target_tolerance:
            break
        if remaining <= 0.0:
            terminal_message = "elapsed-time target overshot"
            all_step_gates_passed = False
            break
        requested_dt = min(dt_next, remaining)
        local_config = config
        if requested_dt < config.minimum_dt:
            local_config = replace(
                config,
                minimum_dt=requested_dt,
            ).validated()
        if (
            perform_first_step_replay_audit
            and not first_step_replay_completed
        ):
            first = advance_causal_five_field_adaptive_backward_euler(
                context,
                state_vector,
                requested_dt,
                previous_increment,
                previous_dt,
                local_config,
            )
            replay = advance_causal_five_field_adaptive_backward_euler(
                context,
                state_vector,
                requested_dt,
                previous_increment,
                previous_dt,
                local_config,
            )
            first_step_replay_bitwise = _adaptive_results_are_bitwise(
                first,
                replay,
            )
            first_step_replay_completed = True
            result = first
            if not first_step_replay_bitwise:
                terminal_message = "first continuation step is not bitwise"
                all_step_gates_passed = False
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

        attempt_rejections = max(0, len(result.attempts) - 1)
        extension_rejected_attempts += attempt_rejections
        if not result.accepted:
            terminal_message = result.message
            all_step_gates_passed = False
            step_rows.append(
                {
                    "accepted_step": None,
                    "elapsed_time_seconds": elapsed_time,
                    "requested_dt_seconds": requested_dt,
                    "attempts": [
                        asdict(attempt) for attempt in result.attempts
                    ],
                    "accepted": False,
                    "message": result.message,
                }
            )
            break

        candidate_vector = np.asarray(
            result.state_vector,
            dtype=float,
        )
        candidate_audit, candidate_state_passed = (
            _source_compatible_state_audit(
                context,
                candidate_vector,
            )
        )
        step_gate_passed = bool(
            result.step.maximum_scaled_residual
            <= config.residual_tolerance
            and result.step.maximum_scaled_algebraic_residual
            <= config.algebraic_residual_tolerance
            and result.step.maximum_scaled_primitive_change
            <= config.maximum_scaled_primitive_change
            and result.step.maximum_scaled_total_change
            <= config.maximum_scaled_total_change
            and result.step.conservation_telescoping_relative_defect
            <= config.conservation_tolerance
            and candidate_state_passed
        )
        row = _adaptive_step_row(
            accepted_steps + 1,
            elapsed_time + result.dt_used,
            result,
        )
        row["state_audit"] = candidate_audit
        row["step_gate_passed"] = step_gate_passed
        step_rows.append(row)
        if not step_gate_passed:
            terminal_message = "accepted nonlinear state failed physical gate"
            all_step_gates_passed = False
            break

        ledger = causal_five_field_physical_step_ledger(
            context,
            state_vector,
            result.physical_increment,
            result.dt_used,
        )
        candidate_summary = candidate_audit["state"]
        physical_ledgers.append(ledger)
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
                + candidate_summary["inner_face_rates"][0]
                - candidate_summary["outer_face_rates"][0]
            )
        )
        state_vector = candidate_vector
        previous_increment = np.asarray(
            result.physical_increment,
            dtype=float,
        )
        previous_dt = float(result.dt_used)
        dt_next = float(result.dt_next)
        elapsed_time += result.dt_used
        accepted_steps += 1
        extension_accepted_steps += 1

    target_reached = bool(
        abs(elapsed_time - elapsed_time_target)
        <= target_tolerance
    )
    final_audit, final_state_passed = _source_compatible_state_audit(
        context,
        state_vector,
    )
    final_rank = (
        _source_compatible_consistency_rank_audit(
            context,
            state_vector,
        )
        if target_reached and all_step_gates_passed
        else None
    )
    extension_elapsed_time = (
        elapsed_time - checkpoint_elapsed_time
    )
    actual_mass_change = math.fsum(actual_mass_increments)
    expected_mass_change = math.fsum(expected_mass_increments)
    mass_budget_relative_defect = float(
        abs(actual_mass_change - expected_mass_change)
        / max(
            abs(actual_mass_change),
            abs(expected_mass_change),
            source_rate * extension_elapsed_time,
            1.0,
        )
    )
    physical_ledger = _aggregate_physical_step_ledgers(
        physical_ledgers
    )
    final_restart = CausalFiveFieldAdaptiveRestart(
        state_vector=state_vector,
        previous_physical_increment=previous_increment,
        elapsed_time=elapsed_time,
        dt_next=dt_next,
        previous_dt=previous_dt,
        accepted_steps=accepted_steps,
        rejected_attempts=(
            rejected_attempts + extension_rejected_attempts
        ),
        provenance={
            "work_package": work_package,
            "parent_work_package": parent_work_package,
            "n_cells": n_cells,
            "role": "bounded_billionth_loading_time_duration",
            "source": (
                "exact circularized regression stream; not ballistic "
                "Layer-1 calibration"
            ),
        },
    )
    final_path = (
        DEFAULT_RESTART_DIRECTORY
        / (
            f"causal_{output_restart_label}_N"
            f"{n_cells:03d}_final.npz"
        )
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
    h_over_r_response = _h_over_r_response_summary(
        context,
        initial_vector,
        state_vector,
    )
    extension_h_over_r_response = _h_over_r_response_summary(
        context,
        checkpoint.state_vector,
        state_vector,
    )
    passed = bool(
        target_reached
        and all_step_gates_passed
        and first_step_replay_bitwise
        and final_state_passed
        and final_rank is not None
        and final_rank["passed"]
        and mass_budget_relative_defect <= 1.0e-10
        and physical_ledger["maximum_relative_balance_defect"]
        <= 1.0e-10
        and final_restart_roundtrip_bitwise
    )
    return {
        "n_cells": n_cells,
        "seed_parameters": seed_parameters,
        "loading_time_seconds": loading_time,
        "target_elapsed_time_seconds": elapsed_time_target,
        "target_loading_time_fraction": (
            elapsed_time_target / loading_time
        ),
        "checkpoint": {
            "path": str(restart_path.relative_to(ROOT)),
            "elapsed_time_seconds": checkpoint_elapsed_time,
            "accepted_steps": checkpoint.accepted_steps,
            "rejected_attempts": checkpoint.rejected_attempts,
            "provenance": checkpoint.provenance,
            "provenance_passed": checkpoint_provenance_passed,
            "state_audit": checkpoint_audit,
            "rank_audit": checkpoint_rank,
            "passed": checkpoint_passed,
        },
        "extension": {
            "elapsed_time_seconds": extension_elapsed_time,
            "accepted_steps": extension_accepted_steps,
            "rejected_attempts": extension_rejected_attempts,
            "first_step_replay_bitwise": (
                first_step_replay_bitwise
            ),
            "steps": step_rows,
        },
        "elapsed_time_seconds": elapsed_time,
        "elapsed_loading_time_fraction": elapsed_time / loading_time,
        "accepted_steps_total": accepted_steps,
        "rejected_attempts_total": (
            rejected_attempts + extension_rejected_attempts
        ),
        "source_rate_g_s": source_rate,
        "final_state": final_audit["state"],
        "final_state_audit": final_audit,
        "final_rank_audit": final_rank,
        "h_over_r_response": h_over_r_response,
        "extension_h_over_r_response": (
            extension_h_over_r_response
        ),
        "mass_budget": {
            "scope": (
                f"{parent_work_package} checkpoint to "
                f"{work_package} endpoint"
            ),
            "cancellation_safe_actual_change_g": (
                actual_mass_change
            ),
            "expected_change_g": expected_mass_change,
            "injected_mass_g": (
                source_rate * extension_elapsed_time
            ),
            "relative_defect": mass_budget_relative_defect,
        },
        "physical_five_field_ledger": physical_ledger,
        "restart": {
            "final_path": str(final_path.relative_to(ROOT)),
            "final_roundtrip_bitwise": (
                final_restart_roundtrip_bitwise
            ),
        },
        "acceptance_tolerances": {
            "maximum_timestep_seconds": config.maximum_dt,
            "scaled_residual": config.residual_tolerance,
            "scaled_algebraic_residual": (
                config.algebraic_residual_tolerance
            ),
            "scaled_primitive_change": (
                config.maximum_scaled_primitive_change
            ),
            "scaled_total_change": (
                config.maximum_scaled_total_change
            ),
            "conservation_relative_defect": (
                config.conservation_tolerance
            ),
            "aggregate_mass_relative_defect": 1.0e-10,
            "aggregate_five_field_relative_defect": 1.0e-10,
        },
        "target_reached": target_reached,
        "all_step_gates_passed": all_step_gates_passed,
        "passed": passed,
        "terminal_message": terminal_message,
        "decision": (
            "source_compatible_duration_resolution_passed"
            if passed
            else "source_compatible_duration_resolution_failed"
        ),
    }, passed


def _source_compatible_duration_mesh_comparison(
    left_run: dict,
    right_run: dict,
    *,
    left_label: str = "n16",
    right_label: str = "n32",
) -> dict:
    def metrics(run: dict) -> dict:
        source_rate = run["source_rate_g_s"]
        extension = run["extension"]["elapsed_time_seconds"]
        return {
            "extension_mass_response_per_injected_mass": (
                run["mass_budget"][
                    "cancellation_safe_actual_change_g"
                ]
                / (source_rate * extension)
            ),
            "inner_mass_flux_over_supply": (
                run["final_state"]["inner_face_rates"][0]
                / source_rate
            ),
            "outer_mass_flux_over_supply": (
                run["final_state"]["outer_face_rates"][0]
                / source_rate
            ),
            "maximum_h_over_r": (
                run["final_state"]["maximum_h_over_r"]
            ),
            "minimum_scattering_optical_depth": (
                run["final_state_audit"][
                    "minimum_scattering_optical_depth"
                ]
            ),
        }

    left = metrics(left_run)
    right = metrics(right_run)
    left_radius = np.asarray(
        left_run["h_over_r_response"]["sample_radius_rg"],
        dtype=float,
    )
    right_radius = np.asarray(
        right_run["h_over_r_response"]["sample_radius_rg"],
        dtype=float,
    )
    if not np.array_equal(left_radius, right_radius):
        raise RuntimeError("duration responses do not share radii")
    left_response = np.asarray(
        left_run["h_over_r_response"]["delta_log_h_over_r"],
        dtype=float,
    )
    right_response = np.asarray(
        right_run["h_over_r_response"]["delta_log_h_over_r"],
        dtype=float,
    )
    response_difference = left_response - right_response
    exact_time_defect = abs(
        left_run["elapsed_time_seconds"]
        - right_run["elapsed_time_seconds"]
    )
    differences = {
        "extension_mass_response_per_injected_mass": abs(
            left["extension_mass_response_per_injected_mass"]
            - right["extension_mass_response_per_injected_mass"]
        ),
        "inner_mass_flux_over_supply": abs(
            left["inner_mass_flux_over_supply"]
            - right["inner_mass_flux_over_supply"]
        ),
        "outer_mass_flux_over_supply": abs(
            left["outer_mass_flux_over_supply"]
            - right["outer_mass_flux_over_supply"]
        ),
        "maximum_h_over_r_relative": abs(
            left["maximum_h_over_r"] - right["maximum_h_over_r"]
        )
        / max(
            abs(left["maximum_h_over_r"]),
            abs(right["maximum_h_over_r"]),
            np.finfo(float).tiny,
        ),
        "minimum_scattering_optical_depth_relative": abs(
            left["minimum_scattering_optical_depth"]
            - right["minimum_scattering_optical_depth"]
        )
        / max(
            abs(left["minimum_scattering_optical_depth"]),
            abs(right["minimum_scattering_optical_depth"]),
            np.finfo(float).tiny,
        ),
        "maximum_delta_log_h_over_r_response_difference": float(
            np.max(np.abs(response_difference))
        ),
        "rms_delta_log_h_over_r_response_difference": float(
            np.sqrt(np.mean(response_difference**2))
        ),
        "exact_elapsed_time_defect_seconds": exact_time_defect,
    }
    gates = {
        "extension_mass_response_per_injected_mass": 0.05,
        "inner_mass_flux_over_supply": 0.05,
        "outer_mass_flux_over_supply": 0.05,
        "maximum_delta_log_h_over_r_response_difference": 5.0e-3,
        "exact_elapsed_time_defect_seconds": max(
            1.0e-20,
            5.0e-14 * left_run["elapsed_time_seconds"],
        ),
    }
    passed = all(
        differences[name] <= limit
        for name, limit in gates.items()
    )
    return {
        left_label: left,
        right_label: right,
        "absolute_or_relative_differences": differences,
        "gates": gates,
        "passed": passed,
    }


def _run_source_compatible_duration_audit(
    args: argparse.Namespace,
) -> None:
    n16_initial, n16_bundle, n16_parameters = (
        _source_compatible_initialization(16)
    )
    n16_loading_time = causal_five_field_loading_time(
        n16_bundle[1]["context"],
        np.asarray(n16_bundle[1]["old_vector"], dtype=float),
    )
    n16_target = 1.0e-9 * n16_loading_time
    n16, n16_passed = (
        _continue_source_compatible_duration_resolution(
            16,
            initialization=n16_bundle[0],
            artifacts=n16_bundle[1],
            seed_parameters=n16_parameters,
            elapsed_time_target=n16_target,
            perform_first_step_replay_audit=True,
        )
        if n16_initial["passed"]
        else (
            {
                "n_cells": 16,
                "passed": False,
                "decision": "source_compatible_n16_setup_failed",
            },
            False,
        )
    )

    n32_initial = None
    n32 = None
    n32_passed = False
    mesh = None
    if n16_passed:
        n32_initial, n32_bundle, n32_parameters = (
            _source_compatible_initialization(32)
        )
        if n32_initial["passed"]:
            n32, n32_passed = (
                _continue_source_compatible_duration_resolution(
                    32,
                    initialization=n32_bundle[0],
                    artifacts=n32_bundle[1],
                    seed_parameters=n32_parameters,
                    elapsed_time_target=n16["elapsed_time_seconds"],
                    perform_first_step_replay_audit=False,
                )
            )
    if n32_passed:
        mesh = _source_compatible_duration_mesh_comparison(
            n16,
            n32,
        )
    passed = bool(
        n16_passed
        and n32_initial is not None
        and n32_initial["passed"]
        and n32_passed
        and mesh is not None
        and mesh["passed"]
    )
    output = {
        "work_package": "WP10c5n",
        "scope": (
            "bounded source-compatible no-tide duration extension from "
            "the certified WP10c5m restarts"
        ),
        "n16_initial_datum": n16_initial,
        "n16_duration": n16,
        "n32_initial_datum": n32_initial,
        "n32_duration": n32,
        "mesh_comparison": mesh,
        "gates": {
            "n16_billionth_loading_time_passed": n16_passed,
            "n32_attempted": n32 is not None,
            "n32_exact_time_passed": n32_passed,
            "mesh_gate_passed": (
                mesh["passed"] if mesh is not None else False
            ),
            "bounded_duration_extension_certified": passed,
            "further_no_tide_duration_extension_authorized": passed,
            "long_evolution_certified": False,
            "stability_certified": False,
            "hot_state_certified": False,
            "limit_cycle_certified": False,
            "tide_authorized": False,
            "wind_authorized": False,
        },
        "decision": (
            "source_compatible_billionth_loading_time_mesh_gate_passed"
            if passed
            else "stop_at_bounded_duration_gate"
        ),
    }
    output_path = _absolute(
        DEFAULT_SOURCE_COMPATIBLE_DURATION_OUTPUT
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


def _run_mesh_common_startup_duration_audit(
    args: argparse.Namespace,
) -> None:
    common_parameters, construction = (
        _mesh_common_source_compatible_seed_parameters()
    )
    n16_initial, n16_bundle, _n16_parameters = (
        _source_compatible_initialization(
            16,
            seed_parameters_override=common_parameters,
            construction_override=construction,
        )
    )
    n32_initial = None
    n32_bundle = None
    initial_profile = None
    if n16_initial["passed"]:
        n32_initial, n32_bundle, _n32_parameters = (
            _source_compatible_initialization(
                32,
                seed_parameters_override=common_parameters,
                construction_override=construction,
            )
        )
        if n32_initial["passed"]:
            initial_profile = _mesh_common_initial_profile_audit(
                n16_bundle[1],
                n32_bundle[1],
                construction,
            )

    short_n16 = None
    short_n32 = None
    short_mesh = None
    short_n16_passed = False
    short_n32_passed = False
    initial_gate_passed = bool(
        n16_initial["passed"]
        and n32_initial is not None
        and n32_initial["passed"]
        and initial_profile is not None
        and initial_profile["passed"]
    )
    if initial_gate_passed:
        short_n16, short_n16_passed = (
            _run_repeated_source_on_resolution(
                16,
                accepted_step_target=8,
                elapsed_time_target=None,
                perform_restart_resume_audit=True,
                seed_kwargs=common_parameters,
                restart_label="wp10c5o",
                restart_work_package="WP10c5o",
                initialization_bundle=n16_bundle,
                step_residual_tolerance=1.0e-10,
                step_algebraic_tolerance=1.0e-11,
                mass_budget_tolerance=1.0e-10,
            )
        )
    if short_n16_passed:
        assert n32_bundle is not None
        short_n32, short_n32_passed = (
            _run_repeated_source_on_resolution(
                32,
                accepted_step_target=None,
                elapsed_time_target=(
                    short_n16["elapsed_time_seconds"]
                ),
                perform_restart_resume_audit=False,
                seed_kwargs=common_parameters,
                restart_label="wp10c5o",
                restart_work_package="WP10c5o",
                initialization_bundle=n32_bundle,
                step_residual_tolerance=1.0e-10,
                step_algebraic_tolerance=1.0e-11,
                mass_budget_tolerance=1.0e-10,
            )
        )
    if short_n32_passed:
        short_mesh = _repeated_mesh_comparison(
            short_n16,
            short_n32,
        )
    short_passed = bool(
        initial_gate_passed
        and short_n16_passed
        and short_n32_passed
        and short_mesh is not None
        and short_mesh["passed"]
    )

    duration_n16 = None
    duration_n32 = None
    duration_mesh = None
    duration_n16_passed = False
    duration_n32_passed = False
    if short_passed:
        n16_loading_time = causal_five_field_loading_time(
            n16_bundle[1]["context"],
            np.asarray(n16_bundle[1]["old_vector"], dtype=float),
        )
        duration_n16, duration_n16_passed = (
            _continue_source_compatible_duration_resolution(
                16,
                initialization=n16_bundle[0],
                artifacts=n16_bundle[1],
                seed_parameters=common_parameters,
                elapsed_time_target=1.0e-9 * n16_loading_time,
                perform_first_step_replay_audit=True,
                parent_restart_label="wp10c5o",
                output_restart_label="wp10c5p",
                work_package="WP10c5p",
                parent_work_package="WP10c5o",
            )
        )
    if duration_n16_passed:
        assert n32_bundle is not None
        duration_n32, duration_n32_passed = (
            _continue_source_compatible_duration_resolution(
                32,
                initialization=n32_bundle[0],
                artifacts=n32_bundle[1],
                seed_parameters=common_parameters,
                elapsed_time_target=(
                    duration_n16["elapsed_time_seconds"]
                ),
                perform_first_step_replay_audit=False,
                parent_restart_label="wp10c5o",
                output_restart_label="wp10c5p",
                work_package="WP10c5p",
                parent_work_package="WP10c5o",
            )
        )
    if duration_n32_passed:
        duration_mesh = (
            _source_compatible_duration_mesh_comparison(
                duration_n16,
                duration_n32,
            )
        )
    duration_passed = bool(
        short_passed
        and duration_n16_passed
        and duration_n32_passed
        and duration_mesh is not None
        and duration_mesh["passed"]
    )
    output = {
        "work_package": "WP10c5o-p",
        "scope": (
            "fixed-anchor mesh-common source-compatible startup and "
            "conditionally authorized bounded duration rerun"
        ),
        "construction": construction,
        "common_seed_parameters": common_parameters,
        "n16_initial_datum": n16_initial,
        "n32_initial_datum": n32_initial,
        "initial_profile_mesh_audit": initial_profile,
        "short_startup": {
            "n16": short_n16,
            "n32": short_n32,
            "mesh_comparison": short_mesh,
            "passed": short_passed,
        },
        "bounded_duration": {
            "n16": duration_n16,
            "n32": duration_n32,
            "mesh_comparison": duration_mesh,
            "passed": duration_passed,
        },
        "gates": {
            "mesh_common_initial_data_passed": initial_gate_passed,
            "short_n16_passed": short_n16_passed,
            "short_n32_attempted": short_n32 is not None,
            "short_n32_passed": short_n32_passed,
            "short_mesh_gate_passed": (
                short_mesh["passed"]
                if short_mesh is not None
                else False
            ),
            "short_common_data_startup_certified": short_passed,
            "duration_n16_attempted": duration_n16 is not None,
            "duration_n16_passed": duration_n16_passed,
            "duration_n32_attempted": duration_n32 is not None,
            "duration_n32_passed": duration_n32_passed,
            "duration_mesh_gate_passed": (
                duration_mesh["passed"]
                if duration_mesh is not None
                else False
            ),
            "bounded_common_data_duration_certified": (
                duration_passed
            ),
            "long_evolution_certified": False,
            "stability_certified": False,
            "hot_state_certified": False,
            "limit_cycle_certified": False,
            "tide_authorized": False,
            "wind_authorized": False,
        },
        "decision": (
            "mesh_common_bounded_duration_gate_passed"
            if duration_passed
            else (
                "stop_after_short_common_data_gate"
                if short_passed
                else "stop_before_bounded_duration"
            )
        ),
    }
    output_path = _absolute(
        DEFAULT_MESH_COMMON_STARTUP_DURATION_OUTPUT
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


def _duration_response_difference(
    reference: dict,
    candidate: dict,
) -> dict:
    reference_radius = np.asarray(
        reference["h_over_r_response"]["sample_radius_rg"],
        dtype=float,
    )
    candidate_radius = np.asarray(
        candidate["h_over_r_response"]["sample_radius_rg"],
        dtype=float,
    )
    if not np.array_equal(reference_radius, candidate_radius):
        raise RuntimeError("duration controls do not share response radii")
    reference_response = np.asarray(
        reference["h_over_r_response"]["delta_log_h_over_r"],
        dtype=float,
    )
    candidate_response = np.asarray(
        candidate["h_over_r_response"]["delta_log_h_over_r"],
        dtype=float,
    )
    difference = candidate_response - reference_response
    maximum_index = int(np.argmax(np.abs(difference)))
    return {
        "maximum_absolute_delta_log_h_over_r_change": float(
            np.max(np.abs(difference))
        ),
        "rms_delta_log_h_over_r_change": float(
            np.sqrt(np.mean(difference**2))
        ),
        "maximum_change_radius_rg": float(
            reference_radius[maximum_index]
        ),
        "accepted_step_change": int(
            candidate["accepted_steps_total"]
            - reference["accepted_steps_total"]
        ),
        "inner_mass_flux_over_supply_change": float(
            (
                candidate["final_state"]["inner_face_rates"][0]
                / candidate["source_rate_g_s"]
            )
            - (
                reference["final_state"]["inner_face_rates"][0]
                / reference["source_rate_g_s"]
            )
        ),
        "maximum_h_over_r_change": float(
            candidate["final_state"]["maximum_h_over_r"]
            - reference["final_state"]["maximum_h_over_r"]
        ),
    }


def _run_mesh_common_temporal_parity_audit(
    args: argparse.Namespace,
) -> None:
    parent_path = DEFAULT_MESH_COMMON_STARTUP_DURATION_OUTPUT
    if not parent_path.exists():
        raise FileNotFoundError(
            "WP10c5o-p parent result is required for temporal parity"
        )
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    prerequisites_passed = bool(
        parent["gates"]["mesh_common_initial_data_passed"]
        and parent["gates"]["short_common_data_startup_certified"]
        and parent["gates"]["duration_n16_passed"]
        and parent["gates"]["duration_n32_passed"]
    )
    short = parent["short_startup"]
    regular_step_candidates = {
        label: float(
            max(
                row["dt_used_seconds"]
                for row in short[label]["steps"]
            )
        )
        for label in ("n16", "n32")
    }
    shared_timestep = min(regular_step_candidates.values())
    elapsed_time_target = float(
        parent["bounded_duration"]["n16"][
            "target_elapsed_time_seconds"
        ]
    )
    common_parameters, construction = (
        _mesh_common_source_compatible_seed_parameters()
    )
    controls: dict[str, dict | None] = {
        "n16": None,
        "n32": None,
    }
    control_passes = {"n16": False, "n32": False}
    if prerequisites_passed:
        for label, n_cells in (("n16", 16), ("n32", 32)):
            context = _context(n_cells, include_stream=True)
            initial_state = make_causal_five_field_seed(
                context,
                **common_parameters,
            )
            artifacts = {
                "context": context,
                "old_vector": pack_causal_five_field_state(
                    initial_state
                ),
                "timestep_seconds": float(
                    short[label]["steps"][0]["dt_used_seconds"]
                ),
            }
            control, control_passed = (
                _continue_source_compatible_duration_resolution(
                    n_cells,
                    initialization={"resolution_passed": True},
                    artifacts=artifacts,
                    seed_parameters=common_parameters,
                    elapsed_time_target=elapsed_time_target,
                    perform_first_step_replay_audit=(n_cells == 16),
                    parent_restart_label="wp10c5o",
                    output_restart_label="wp10c5q",
                    work_package="WP10c5q",
                    parent_work_package="WP10c5o",
                    initial_dt_next_override=shared_timestep,
                    maximum_dt_override=shared_timestep,
                )
            )
            controls[label] = control
            control_passes[label] = control_passed
            if not control_passed:
                break
    mesh_comparison = None
    if control_passes["n16"] and control_passes["n32"]:
        mesh_comparison = (
            _source_compatible_duration_mesh_comparison(
                controls["n16"],
                controls["n32"],
            )
        )
    passed = bool(
        prerequisites_passed
        and control_passes["n16"]
        and control_passes["n32"]
        and mesh_comparison is not None
        and mesh_comparison["passed"]
    )
    original = parent["bounded_duration"]
    output = {
        "work_package": "WP10c5q",
        "scope": (
            "shared-timestep causal duration control from the certified "
            "mesh-common WP10c5o checkpoints"
        ),
        "parent_result": str(parent_path.relative_to(ROOT)),
        "construction": construction,
        "common_seed_parameters": common_parameters,
        "temporal_parity": {
            "selection_rule": (
                "minimum across meshes of the maximum accepted "
                "short-startup timestep; changes no physical or nonlinear "
                "tolerance"
            ),
            "regular_step_candidates_seconds": (
                regular_step_candidates
            ),
            "shared_maximum_timestep_seconds": shared_timestep,
            "target_elapsed_time_seconds": elapsed_time_target,
        },
        "n16": controls["n16"],
        "n32": controls["n32"],
        "mesh_comparison": mesh_comparison,
        "change_from_uncontrolled_duration": {
            "n16": (
                _duration_response_difference(
                    original["n16"],
                    controls["n16"],
                )
                if control_passes["n16"]
                else None
            ),
            "n32": (
                _duration_response_difference(
                    original["n32"],
                    controls["n32"],
                )
                if control_passes["n32"]
                else None
            ),
        },
        "gates": {
            "parent_prerequisites_passed": prerequisites_passed,
            "n16_shared_timestep_duration_passed": (
                control_passes["n16"]
            ),
            "n32_shared_timestep_duration_passed": (
                control_passes["n32"]
            ),
            "shared_timestep_mesh_gate_passed": (
                mesh_comparison["passed"]
                if mesh_comparison is not None
                else False
            ),
            "bounded_common_data_duration_certified": passed,
            "long_evolution_certified": False,
            "stability_certified": False,
            "hot_state_certified": False,
            "limit_cycle_certified": False,
            "n64_authorized": False,
            "tide_authorized": False,
            "wind_authorized": False,
        },
        "decision": (
            "shared_timestep_bounded_duration_gate_passed"
            if passed
            else "stop_at_shared_timestep_duration_gate"
        ),
    }
    output_path = _absolute(
        DEFAULT_MESH_COMMON_TEMPORAL_PARITY_OUTPUT
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


def _mesh_contraction_summary(
    coarse_comparison: dict,
    fine_comparison: dict,
    *,
    coarse_pair: tuple[str, str] = ("n16", "n32"),
    fine_pair: tuple[str, str] = ("n32", "n64"),
) -> dict:
    coarse = coarse_comparison["absolute_or_relative_differences"]
    fine = fine_comparison["absolute_or_relative_differences"]
    coarse_key = "_".join(coarse_pair)
    fine_key = "_".join(fine_pair)
    metric_names = (
        "maximum_delta_log_h_over_r_response_difference",
        "rms_delta_log_h_over_r_response_difference",
    )
    metrics = {}
    for name in metric_names:
        coarse_error = float(coarse[name])
        fine_error = float(fine[name])
        if coarse_error > 0.0 and fine_error > 0.0:
            ratio = coarse_error / fine_error
            observed_order = float(np.log2(ratio))
        elif coarse_error > 0.0 and fine_error == 0.0:
            ratio = np.inf
            observed_order = np.inf
        else:
            ratio = np.nan
            observed_order = np.nan
        metrics[name] = {
            f"{coarse_key}_error": coarse_error,
            f"{fine_key}_error": fine_error,
            "coarse_to_fine_error_ratio": ratio,
            "observed_doubling_order": observed_order,
            "contracted": bool(fine_error < coarse_error),
        }
    maximum = metrics[
        "maximum_delta_log_h_over_r_response_difference"
    ]
    return {
        "method": (
            f"log2 of {coarse_pair[0].upper()}/"
            f"{coarse_pair[1].upper()} error divided by "
            f"{fine_pair[0].upper()}/{fine_pair[1].upper()} error "
            "at one exact common physical time and one shared "
            "maximum timestep"
        ),
        "coarse_pair": list(coarse_pair),
        "fine_pair": list(fine_pair),
        "metrics": metrics,
        "minimum_order_for_one_n128_confirmation": 0.75,
        "maximum_response_contracts": maximum["contracted"],
        "maximum_response_order_passes": bool(
            maximum["observed_doubling_order"] >= 0.75
        ),
    }


def _run_mesh_common_n64_confirmation_audit(
    args: argparse.Namespace,
) -> None:
    parent_paths = {
        "startup_duration": DEFAULT_MESH_COMMON_STARTUP_DURATION_OUTPUT,
        "temporal_parity": DEFAULT_MESH_COMMON_TEMPORAL_PARITY_OUTPUT,
        "spatial_response": DEFAULT_MESH_COMMON_SPATIAL_RESPONSE_OUTPUT,
    }
    missing = [
        name for name, path in parent_paths.items() if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "WP10c5s requires parent results: " + ", ".join(missing)
        )
    parents = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in parent_paths.items()
    }
    startup_parent = parents["startup_duration"]
    temporal_parent = parents["temporal_parity"]
    spatial_parent = parents["spatial_response"]
    common_parameters, construction = (
        _mesh_common_source_compatible_seed_parameters()
    )
    parent_contract = {
        "startup_short_passed": bool(
            startup_parent["gates"][
                "short_common_data_startup_certified"
            ]
        ),
        "temporal_n16_passed": bool(
            temporal_parent["gates"][
                "n16_shared_timestep_duration_passed"
            ]
        ),
        "temporal_n32_passed": bool(
            temporal_parent["gates"][
                "n32_shared_timestep_duration_passed"
            ]
        ),
        "temporal_mesh_failed_as_expected": bool(
            not temporal_parent["gates"][
                "shared_timestep_mesh_gate_passed"
            ]
        ),
        "spatial_classification_passed": bool(
            spatial_parent["passed"]
        ),
        "n64_authorized_by_spatial_audit": bool(
            spatial_parent["gates"][
                "n64_physical_evolution_authorized_for_next_wp"
            ]
        ),
        "startup_parameters_unchanged": bool(
            startup_parent["common_seed_parameters"]
            == common_parameters
            and startup_parent["construction"] == construction
        ),
        "temporal_parameters_unchanged": bool(
            temporal_parent["common_seed_parameters"]
            == common_parameters
            and temporal_parent["construction"] == construction
        ),
        "spatial_parameters_unchanged": bool(
            spatial_parent["common_seed_parameters"]
            == common_parameters
            and spatial_parent["construction"] == construction
        ),
    }
    prerequisites_passed = all(parent_contract.values())

    n64_initial = None
    n64_bundle = None
    initial_profile = None
    if prerequisites_passed:
        n64_initial, n64_bundle, _n64_parameters = (
            _source_compatible_initialization(
                64,
                seed_parameters_override=common_parameters,
                construction_override=construction,
                use_sparse_colored_initialization=True,
            )
        )
        if n64_initial["passed"]:
            n32_context = _context(32, include_stream=True)
            n32_state = make_causal_five_field_seed(
                n32_context,
                **common_parameters,
            )
            n32_artifacts = {
                "context": n32_context,
                "old_vector": pack_causal_five_field_state(
                    n32_state
                ),
            }
            initial_profile = _mesh_common_initial_profile_audit(
                n32_artifacts,
                n64_bundle[1],
                construction,
                left_label="n32",
                right_label="n64",
            )

    parent_short_n32 = startup_parent["short_startup"]["n32"]
    short_n64 = None
    short_mesh = None
    short_n64_passed = False
    initial_gate_passed = bool(
        prerequisites_passed
        and n64_initial is not None
        and n64_initial["passed"]
        and initial_profile is not None
        and initial_profile["passed"]
    )
    if initial_gate_passed:
        assert n64_bundle is not None
        short_n64, short_n64_passed = (
            _run_repeated_source_on_resolution(
                64,
                accepted_step_target=None,
                elapsed_time_target=float(
                    parent_short_n32["elapsed_time_seconds"]
                ),
                perform_restart_resume_audit=True,
                seed_kwargs=common_parameters,
                restart_label="wp10c5s",
                restart_work_package="WP10c5s",
                initialization_bundle=n64_bundle,
                step_residual_tolerance=1.0e-10,
                step_algebraic_tolerance=1.0e-11,
                mass_budget_tolerance=1.0e-10,
            )
        )
    if short_n64_passed:
        short_mesh = _repeated_mesh_comparison(
            parent_short_n32,
            short_n64,
            left_label="n32",
            right_label="n64",
        )
    short_gate_passed = bool(
        initial_gate_passed
        and short_n64_passed
        and short_mesh is not None
        and short_mesh["passed"]
    )

    shared_maximum_timestep = None
    duration_n32 = None
    duration_n64 = None
    duration_mesh = None
    duration_n32_passed = False
    duration_n64_passed = False
    if short_gate_passed:
        assert short_n64 is not None
        assert n64_bundle is not None
        n32_short_maximum = float(
            max(
                row["dt_used_seconds"]
                for row in parent_short_n32["steps"]
            )
        )
        n64_short_maximum = float(
            max(
                row["dt_used_seconds"]
                for row in short_n64["steps"]
            )
        )
        shared_maximum_timestep = min(
            n32_short_maximum,
            n64_short_maximum,
        )
        n32_context = _context(32, include_stream=True)
        n32_state = make_causal_five_field_seed(
            n32_context,
            **common_parameters,
        )
        n32_artifacts = {
            "context": n32_context,
            "old_vector": pack_causal_five_field_state(n32_state),
            "timestep_seconds": float(
                parent_short_n32["steps"][0]["dt_used_seconds"]
            ),
        }
        duration_target = float(
            temporal_parent["temporal_parity"][
                "target_elapsed_time_seconds"
            ]
        )
        duration_n32, duration_n32_passed = (
            _continue_source_compatible_duration_resolution(
                32,
                initialization={"resolution_passed": True},
                artifacts=n32_artifacts,
                seed_parameters=common_parameters,
                elapsed_time_target=duration_target,
                perform_first_step_replay_audit=False,
                parent_restart_label="wp10c5o",
                output_restart_label="wp10c5s_duration",
                work_package="WP10c5s",
                parent_work_package="WP10c5o",
                initial_dt_next_override=shared_maximum_timestep,
                maximum_dt_override=shared_maximum_timestep,
            )
        )
        if duration_n32_passed:
            duration_n64, duration_n64_passed = (
                _continue_source_compatible_duration_resolution(
                    64,
                    initialization=n64_bundle[0],
                    artifacts=n64_bundle[1],
                    seed_parameters=common_parameters,
                    elapsed_time_target=duration_target,
                    perform_first_step_replay_audit=True,
                    parent_restart_label="wp10c5s",
                    output_restart_label="wp10c5s_duration",
                    work_package="WP10c5s",
                    parent_work_package="WP10c5s",
                    initial_dt_next_override=(
                        shared_maximum_timestep
                    ),
                    maximum_dt_override=shared_maximum_timestep,
                )
            )
    if duration_n32_passed and duration_n64_passed:
        duration_mesh = (
            _source_compatible_duration_mesh_comparison(
                duration_n32,
                duration_n64,
                left_label="n32",
                right_label="n64",
            )
        )

    contraction = None
    if duration_mesh is not None:
        contraction = _mesh_contraction_summary(
            temporal_parent["mesh_comparison"],
            duration_mesh,
        )
    duration_execution_passed = bool(
        short_gate_passed
        and duration_n32_passed
        and duration_n64_passed
        and duration_mesh is not None
    )
    mesh_gate_certified = bool(
        duration_execution_passed and duration_mesh["passed"]
    )
    fine_maximum_error = (
        float(
            duration_mesh["absolute_or_relative_differences"][
                "maximum_delta_log_h_over_r_response_difference"
            ]
        )
        if duration_mesh is not None
        else np.inf
    )
    response_gate = (
        float(
            duration_mesh["gates"][
                "maximum_delta_log_h_over_r_response_difference"
            ]
        )
        if duration_mesh is not None
        else 5.0e-3
    )
    n128_authorized = bool(
        duration_execution_passed
        and not mesh_gate_certified
        and fine_maximum_error > response_gate
        and contraction is not None
        and contraction["maximum_response_contracts"]
        and contraction["maximum_response_order_passes"]
    )
    classification_passed = bool(
        mesh_gate_certified or n128_authorized
    )
    output = {
        "work_package": "WP10c5s",
        "scope": (
            "one bounded N64 confirmation from the certified fixed "
            "analytic causal datum, with an unchanged short gate and "
            "conditional exact-time duration contraction audit"
        ),
        "parent_results": {
            name: str(path.relative_to(ROOT))
            for name, path in parent_paths.items()
        },
        "parent_contract": parent_contract,
        "construction": construction,
        "common_seed_parameters": common_parameters,
        "n64_initial_datum": n64_initial,
        "n32_n64_initial_profile_audit": initial_profile,
        "short_startup": {
            "reference_n32": parent_short_n32,
            "n64": short_n64,
            "mesh_comparison": short_mesh,
            "passed": short_gate_passed,
        },
        "temporal_parity": {
            "selection_rule": (
                "minimum of the N32 and N64 maximum accepted "
                "short-startup timesteps; physical and nonlinear "
                "tolerances are unchanged"
            ),
            "shared_maximum_timestep_seconds": (
                shared_maximum_timestep
            ),
            "target_elapsed_time_seconds": float(
                temporal_parent["temporal_parity"][
                    "target_elapsed_time_seconds"
                ]
            ),
        },
        "bounded_duration": {
            "n32": duration_n32,
            "n64": duration_n64,
            "mesh_comparison": duration_mesh,
            "passed_individually": duration_execution_passed,
            "mesh_gate_certified": mesh_gate_certified,
        },
        "coarse_duration_reference": {
            "pair": "n16_n32",
            "mesh_comparison": temporal_parent["mesh_comparison"],
        },
        "duration_contraction": contraction,
        "gates": {
            "parent_prerequisites_passed": prerequisites_passed,
            "n64_initial_datum_passed": bool(
                n64_initial is not None and n64_initial["passed"]
            ),
            "n32_n64_initial_profile_passed": bool(
                initial_profile is not None
                and initial_profile["passed"]
            ),
            "n64_short_startup_passed": short_n64_passed,
            "n32_n64_short_mesh_gate_passed": bool(
                short_mesh is not None and short_mesh["passed"]
            ),
            "short_common_data_startup_certified": (
                short_gate_passed
            ),
            "n32_duration_passed": duration_n32_passed,
            "n64_duration_passed": duration_n64_passed,
            "n32_n64_duration_mesh_gate_passed": (
                mesh_gate_certified
            ),
            "bounded_duration_mesh_certified": mesh_gate_certified,
            "one_n128_confirmation_authorized": n128_authorized,
            "n64_physical_evolution_executed": (
                short_n64 is not None
            ),
            "n128_physical_evolution_executed": False,
            "n96_physical_evolution_executed": False,
            "operator_correction_authorized": False,
            "long_evolution_authorized": False,
            "tide_authorized": False,
            "wind_authorized": False,
            "stability_authorized": False,
            "hot_state_search_authorized": False,
            "limit_cycle_search_authorized": False,
        },
        "passed": classification_passed,
        "decision": (
            "mesh_gate_certified_at_n64"
            if mesh_gate_certified
            else (
                "one_bounded_n128_confirmation_authorized"
                if n128_authorized
                else (
                    "stop_after_n64_confirmation"
                    if duration_execution_passed
                    else (
                        "stop_after_n64_short_gate"
                        if short_n64 is not None
                        else "stop_before_n64_evolution"
                    )
                )
            )
        ),
    }
    output_path = _absolute(
        DEFAULT_MESH_COMMON_N64_CONFIRMATION_OUTPUT
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


def _run_mesh_common_n64_ledger_replay_audit(
    args: argparse.Namespace,
) -> None:
    parent_paths = {
        "n64_confirmation": (
            DEFAULT_MESH_COMMON_N64_CONFIRMATION_OUTPUT
        ),
        "temporal_parity": DEFAULT_MESH_COMMON_TEMPORAL_PARITY_OUTPUT,
    }
    missing = [
        name for name, path in parent_paths.items() if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "WP10c5t requires parent results: " + ", ".join(missing)
        )
    parent = json.loads(
        parent_paths["n64_confirmation"].read_text(encoding="utf-8")
    )
    temporal_parent = json.loads(
        parent_paths["temporal_parity"].read_text(encoding="utf-8")
    )
    baseline_n64 = parent["bounded_duration"]["n64"]
    reference_n32 = parent["bounded_duration"]["n32"]
    if baseline_n64 is None or reference_n32 is None:
        raise RuntimeError(
            "WP10c5t requires completed N32 and N64 duration attempts"
        )
    baseline_ledger_tolerance = float(
        baseline_n64["acceptance_tolerances"][
            "aggregate_five_field_relative_defect"
        ]
    )
    strict_residual_tolerance = 1.0e-11
    n32_step_residuals = [
        float(row["maximum_scaled_residual"])
        for row in reference_n32["extension"]["steps"]
        if row.get("step_gate_passed", False)
    ]
    baseline_n64_step_residuals = [
        float(row["maximum_scaled_residual"])
        for row in baseline_n64["extension"]["steps"]
        if row.get("step_gate_passed", False)
    ]
    ledger_only_failure = bool(
        not baseline_n64["passed"]
        and baseline_n64["terminal_message"] == "target reached"
        and baseline_n64["target_reached"]
        and baseline_n64["all_step_gates_passed"]
        and baseline_n64["extension"]["first_step_replay_bitwise"]
        and baseline_n64["final_state_audit"]["passed"]
        and baseline_n64["final_rank_audit"] is not None
        and baseline_n64["final_rank_audit"]["passed"]
        and baseline_n64["mass_budget"]["relative_defect"]
        <= baseline_n64["acceptance_tolerances"][
            "aggregate_mass_relative_defect"
        ]
        and baseline_n64["restart"]["final_roundtrip_bitwise"]
        and baseline_n64["physical_five_field_ledger"][
            "maximum_relative_balance_defect"
        ]
        > baseline_ledger_tolerance
    )
    n32_already_strict = bool(
        reference_n32["passed"]
        and n32_step_residuals
        and max(n32_step_residuals) <= strict_residual_tolerance
    )
    common_parameters, construction = (
        _mesh_common_source_compatible_seed_parameters()
    )
    parent_contract = {
        "work_package_is_wp10c5s": (
            parent["work_package"] == "WP10c5s"
        ),
        "short_gate_passed": parent["short_startup"]["passed"],
        "n32_duration_passed": reference_n32["passed"],
        "n64_failure_is_ledger_only": ledger_only_failure,
        "n32_already_satisfies_strict_residual": n32_already_strict,
        "common_parameters_unchanged": (
            parent["common_seed_parameters"] == common_parameters
        ),
        "construction_unchanged": (
            parent["construction"] == construction
        ),
        "baseline_ledger_gate_unchanged": (
            baseline_ledger_tolerance == 1.0e-10
        ),
    }
    prerequisites_passed = all(parent_contract.values())

    strict_n64 = None
    strict_n64_passed = False
    if prerequisites_passed:
        context = _context(64, include_stream=True)
        initial_state = make_causal_five_field_seed(
            context,
            **common_parameters,
        )
        artifacts = {
            "context": context,
            "old_vector": pack_causal_five_field_state(
                initial_state
            ),
            "timestep_seconds": float(
                parent["short_startup"]["n64"]["steps"][0][
                    "dt_used_seconds"
                ]
            ),
        }
        strict_n64, strict_n64_passed = (
            _continue_source_compatible_duration_resolution(
                64,
                initialization={"resolution_passed": True},
                artifacts=artifacts,
                seed_parameters=common_parameters,
                elapsed_time_target=float(
                    parent["temporal_parity"][
                        "target_elapsed_time_seconds"
                    ]
                ),
                perform_first_step_replay_audit=True,
                parent_restart_label="wp10c5s",
                output_restart_label="wp10c5t_duration",
                work_package="WP10c5t",
                parent_work_package="WP10c5s",
                initial_dt_next_override=float(
                    parent["temporal_parity"][
                        "shared_maximum_timestep_seconds"
                    ]
                ),
                maximum_dt_override=float(
                    parent["temporal_parity"][
                        "shared_maximum_timestep_seconds"
                    ]
                ),
                step_residual_tolerance=(
                    strict_residual_tolerance
                ),
            )
        )

    strict_step_residuals = (
        [
            float(row["maximum_scaled_residual"])
            for row in strict_n64["extension"]["steps"]
            if row.get("step_gate_passed", False)
        ]
        if strict_n64 is not None
        else []
    )
    strict_residual_contract_passed = bool(
        strict_n64_passed
        and strict_step_residuals
        and max(strict_step_residuals)
        <= strict_residual_tolerance
    )
    fine_comparison = None
    contraction = None
    if strict_residual_contract_passed:
        fine_comparison = (
            _source_compatible_duration_mesh_comparison(
                reference_n32,
                strict_n64,
                left_label="n32",
                right_label="n64",
            )
        )
        contraction = _mesh_contraction_summary(
            temporal_parent["mesh_comparison"],
            fine_comparison,
        )
    mesh_gate_certified = bool(
        fine_comparison is not None and fine_comparison["passed"]
    )
    fine_maximum_error = (
        float(
            fine_comparison["absolute_or_relative_differences"][
                "maximum_delta_log_h_over_r_response_difference"
            ]
        )
        if fine_comparison is not None
        else np.inf
    )
    response_gate = (
        float(
            fine_comparison["gates"][
                "maximum_delta_log_h_over_r_response_difference"
            ]
        )
        if fine_comparison is not None
        else 5.0e-3
    )
    n128_authorized = bool(
        strict_residual_contract_passed
        and not mesh_gate_certified
        and fine_maximum_error > response_gate
        and contraction is not None
        and contraction["maximum_response_contracts"]
        and contraction["maximum_response_order_passes"]
    )
    classification_passed = bool(
        mesh_gate_certified or n128_authorized
    )
    output = {
        "work_package": "WP10c5t",
        "scope": (
            "one bounded ledger-tight N64 replay from the certified "
            "WP10c5s short checkpoint after an otherwise passing "
            "duration accumulated nonlinear closure above the unchanged "
            "aggregate ledger gate"
        ),
        "parent_results": {
            name: str(path.relative_to(ROOT))
            for name, path in parent_paths.items()
        },
        "parent_contract": parent_contract,
        "construction": construction,
        "common_seed_parameters": common_parameters,
        "numerical_change": {
            "baseline_step_residual_tolerance": float(
                baseline_n64["acceptance_tolerances"][
                    "scaled_residual"
                ]
            ),
            "strict_step_residual_tolerance": (
                strict_residual_tolerance
            ),
            "aggregate_five_field_ledger_tolerance": (
                baseline_ledger_tolerance
            ),
            "aggregate_mass_ledger_tolerance": float(
                baseline_n64["acceptance_tolerances"][
                    "aggregate_mass_relative_defect"
                ]
            ),
            "physical_equations_changed": False,
            "physical_gates_changed": False,
            "acceptance_gate_relaxed": False,
        },
        "baseline_n64": {
            "maximum_step_residual": max(
                baseline_n64_step_residuals
            ),
            "aggregate_five_field_ledger_defect": (
                baseline_n64["physical_five_field_ledger"][
                    "maximum_relative_balance_defect"
                ]
            ),
            "aggregate_mass_ledger_defect": (
                baseline_n64["mass_budget"]["relative_defect"]
            ),
            "passed": baseline_n64["passed"],
        },
        "reference_n32": {
            "maximum_step_residual": max(n32_step_residuals),
            "already_satisfies_strict_residual": n32_already_strict,
            "duration": reference_n32,
        },
        "strict_n64": strict_n64,
        "change_from_baseline_n64": (
            _duration_response_difference(
                baseline_n64,
                strict_n64,
            )
            if strict_n64_passed
            else None
        ),
        "fine_mesh_comparison": fine_comparison,
        "duration_contraction": contraction,
        "gates": {
            "parent_prerequisites_passed": prerequisites_passed,
            "strict_n64_duration_passed": strict_n64_passed,
            "strict_n64_residual_contract_passed": (
                strict_residual_contract_passed
            ),
            "strict_n64_aggregate_ledger_gate_passed": bool(
                strict_n64 is not None
                and strict_n64[
                    "physical_five_field_ledger"
                ]["maximum_relative_balance_defect"]
                <= baseline_ledger_tolerance
            ),
            "n32_n64_duration_mesh_gate_passed": (
                mesh_gate_certified
            ),
            "bounded_duration_mesh_certified": mesh_gate_certified,
            "one_n128_confirmation_authorized": n128_authorized,
            "n128_physical_evolution_executed": False,
            "n96_physical_evolution_executed": False,
            "operator_correction_authorized": False,
            "long_evolution_authorized": False,
            "tide_authorized": False,
            "wind_authorized": False,
            "stability_authorized": False,
            "hot_state_search_authorized": False,
            "limit_cycle_search_authorized": False,
        },
        "passed": classification_passed,
        "decision": (
            "mesh_gate_certified_at_ledger_tight_n64"
            if mesh_gate_certified
            else (
                "one_bounded_n128_confirmation_authorized"
                if n128_authorized
                else (
                    "stop_after_ledger_tight_n64"
                    if strict_n64 is not None
                    else "stop_before_ledger_tight_n64"
                )
            )
        ),
    }
    output_path = _absolute(
        DEFAULT_MESH_COMMON_N64_LEDGER_REPLAY_OUTPUT
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


def _run_mesh_common_n128_confirmation_audit(
    args: argparse.Namespace,
) -> None:
    parent_paths = {
        "n64_ledger_replay": (
            DEFAULT_MESH_COMMON_N64_LEDGER_REPLAY_OUTPUT
        ),
        "n64_confirmation": (
            DEFAULT_MESH_COMMON_N64_CONFIRMATION_OUTPUT
        ),
    }
    missing = [
        name for name, path in parent_paths.items() if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "WP10c5u requires parent results: " + ", ".join(missing)
        )
    ledger_parent = json.loads(
        parent_paths["n64_ledger_replay"].read_text(
            encoding="utf-8"
        )
    )
    confirmation_parent = json.loads(
        parent_paths["n64_confirmation"].read_text(
            encoding="utf-8"
        )
    )
    reference_n64 = ledger_parent["strict_n64"]
    reference_short_n64 = confirmation_parent["short_startup"]["n64"]
    if reference_n64 is None or reference_short_n64 is None:
        raise RuntimeError(
            "WP10c5u requires accepted strict-duration and short N64 "
            "references"
        )

    common_parameters, construction = (
        _mesh_common_source_compatible_seed_parameters()
    )
    short_residual_tolerance = 1.0e-10
    strict_residual_tolerance = 1.0e-11
    target_elapsed_time = float(
        reference_n64["target_elapsed_time_seconds"]
    )
    maximum_timestep = float(
        reference_n64["acceptance_tolerances"][
            "maximum_timestep_seconds"
        ]
    )
    parent_contract = {
        "ledger_parent_is_wp10c5t": (
            ledger_parent["work_package"] == "WP10c5t"
        ),
        "n128_authorized_exactly_once": bool(
            ledger_parent["gates"][
                "one_n128_confirmation_authorized"
            ]
        ),
        "strict_n64_duration_passed": bool(
            ledger_parent["gates"][
                "strict_n64_duration_passed"
            ]
        ),
        "strict_n64_residual_contract_passed": bool(
            ledger_parent["gates"][
                "strict_n64_residual_contract_passed"
            ]
        ),
        "n32_n64_contraction_order_passed": bool(
            ledger_parent["duration_contraction"][
                "maximum_response_order_passes"
            ]
        ),
        "n64_short_startup_passed": bool(
            confirmation_parent["gates"][
                "n64_short_startup_passed"
            ]
        ),
        "short_residual_tolerance_unchanged": bool(
            reference_short_n64["acceptance_tolerances"][
                "scaled_residual"
            ]
            == short_residual_tolerance
        ),
        "common_parameters_unchanged": bool(
            ledger_parent["common_seed_parameters"]
            == common_parameters
            and confirmation_parent["common_seed_parameters"]
            == common_parameters
        ),
        "construction_unchanged": bool(
            ledger_parent["construction"] == construction
            and confirmation_parent["construction"] == construction
        ),
        "strict_residual_tolerance_unchanged": bool(
            reference_n64["acceptance_tolerances"][
                "scaled_residual"
            ]
            == strict_residual_tolerance
        ),
        "duration_target_unchanged": bool(
            target_elapsed_time
            == confirmation_parent["temporal_parity"][
                "target_elapsed_time_seconds"
            ]
        ),
        "maximum_timestep_unchanged": bool(
            maximum_timestep
            == confirmation_parent["temporal_parity"][
                "shared_maximum_timestep_seconds"
            ]
        ),
    }
    prerequisites_passed = all(parent_contract.values())

    n128_initial = None
    n128_bundle = None
    initial_profile = None
    if prerequisites_passed:
        n128_initial, n128_bundle, _n128_parameters = (
            _source_compatible_initialization(
                128,
                seed_parameters_override=common_parameters,
                construction_override=construction,
                use_sparse_colored_initialization=True,
            )
        )
        if n128_initial["passed"]:
            n64_context = _context(64, include_stream=True)
            n64_state = make_causal_five_field_seed(
                n64_context,
                **common_parameters,
            )
            n64_artifacts = {
                "context": n64_context,
                "old_vector": pack_causal_five_field_state(
                    n64_state
                ),
            }
            initial_profile = _mesh_common_initial_profile_audit(
                n64_artifacts,
                n128_bundle[1],
                construction,
                left_label="n64",
                right_label="n128",
            )

    initial_gate_passed = bool(
        prerequisites_passed
        and n128_initial is not None
        and n128_initial["passed"]
        and initial_profile is not None
        and initial_profile["passed"]
    )
    short_n128 = None
    short_n128_passed = False
    short_mesh = None
    if initial_gate_passed:
        assert n128_bundle is not None
        short_n128, short_n128_passed = (
            _run_repeated_source_on_resolution(
                128,
                accepted_step_target=None,
                elapsed_time_target=float(
                    reference_short_n64["elapsed_time_seconds"]
                ),
                perform_restart_resume_audit=True,
                seed_kwargs=common_parameters,
                restart_label="wp10c5u",
                restart_work_package="WP10c5u",
                initialization_bundle=n128_bundle,
                step_residual_tolerance=short_residual_tolerance,
                step_algebraic_tolerance=1.0e-11,
                mass_budget_tolerance=1.0e-10,
            )
        )
    if short_n128_passed:
        short_mesh = _repeated_mesh_comparison(
            reference_short_n64,
            short_n128,
            left_label="n64",
            right_label="n128",
        )
    short_gate_passed = bool(
        initial_gate_passed
        and short_n128_passed
        and short_mesh is not None
        and short_mesh["passed"]
    )

    duration_n128 = None
    duration_n128_passed = False
    duration_mesh = None
    contraction = None
    if short_gate_passed:
        assert n128_bundle is not None
        duration_n128, duration_n128_passed = (
            _continue_source_compatible_duration_resolution(
                128,
                initialization=n128_bundle[0],
                artifacts=n128_bundle[1],
                seed_parameters=common_parameters,
                elapsed_time_target=target_elapsed_time,
                perform_first_step_replay_audit=True,
                parent_restart_label="wp10c5u",
                output_restart_label="wp10c5u_duration",
                work_package="WP10c5u",
                parent_work_package="WP10c5u",
                initial_dt_next_override=maximum_timestep,
                maximum_dt_override=maximum_timestep,
                step_residual_tolerance=(
                    strict_residual_tolerance
                ),
            )
        )
    if duration_n128_passed:
        duration_mesh = (
            _source_compatible_duration_mesh_comparison(
                reference_n64,
                duration_n128,
                left_label="n64",
                right_label="n128",
            )
        )
        contraction = _mesh_contraction_summary(
            ledger_parent["fine_mesh_comparison"],
            duration_mesh,
            coarse_pair=("n32", "n64"),
            fine_pair=("n64", "n128"),
        )

    strict_step_residuals = (
        [
            float(row["maximum_scaled_residual"])
            for row in duration_n128["extension"]["steps"]
            if row.get("step_gate_passed", False)
        ]
        if duration_n128 is not None
        else []
    )
    strict_residual_contract_passed = bool(
        duration_n128_passed
        and strict_step_residuals
        and max(strict_step_residuals)
        <= strict_residual_tolerance
    )
    mesh_gate_certified = bool(
        strict_residual_contract_passed
        and duration_mesh is not None
        and duration_mesh["passed"]
    )
    bounded_classification_completed = bool(
        prerequisites_passed
        and (
            not initial_gate_passed
            or (
                short_n128 is not None
                and not short_gate_passed
            )
            or duration_n128 is not None
        )
    )
    output = {
        "work_package": "WP10c5u",
        "scope": (
            "exactly one bounded N128 confirmation from the fixed "
            "analytic datum, with an unchanged N64/N128 short gate "
            "and conditional strict-residual duration comparison"
        ),
        "parent_results": {
            name: str(path.relative_to(ROOT))
            for name, path in parent_paths.items()
        },
        "parent_contract": parent_contract,
        "construction": construction,
        "common_seed_parameters": common_parameters,
        "numerical_contract": {
            "short_step_residual_tolerance": (
                short_residual_tolerance
            ),
            "duration_step_residual_tolerance": (
                strict_residual_tolerance
            ),
            "maximum_timestep_seconds": maximum_timestep,
            "target_elapsed_time_seconds": target_elapsed_time,
            "physical_equations_changed": False,
            "physical_gates_changed": False,
            "operator_changed": False,
        },
        "n128_initial_datum": n128_initial,
        "n64_n128_initial_profile_audit": initial_profile,
        "short_startup": {
            "reference_n64": reference_short_n64,
            "n128": short_n128,
            "mesh_comparison": short_mesh,
            "passed": short_gate_passed,
        },
        "bounded_duration": {
            "reference_strict_n64": reference_n64,
            "n128": duration_n128,
            "mesh_comparison": duration_mesh,
            "duration_contraction": contraction,
            "mesh_gate_certified": mesh_gate_certified,
        },
        "gates": {
            "parent_prerequisites_passed": prerequisites_passed,
            "n128_initial_datum_passed": bool(
                n128_initial is not None and n128_initial["passed"]
            ),
            "n64_n128_initial_profile_passed": bool(
                initial_profile is not None
                and initial_profile["passed"]
            ),
            "n128_short_startup_passed": short_n128_passed,
            "n64_n128_short_mesh_gate_passed": bool(
                short_mesh is not None and short_mesh["passed"]
            ),
            "short_common_data_startup_certified": (
                short_gate_passed
            ),
            "n128_duration_attempted": duration_n128 is not None,
            "n128_duration_passed": duration_n128_passed,
            "n128_strict_residual_contract_passed": (
                strict_residual_contract_passed
            ),
            "n64_n128_duration_mesh_gate_passed": (
                mesh_gate_certified
            ),
            "bounded_duration_mesh_certified": mesh_gate_certified,
            "bounded_classification_completed": (
                bounded_classification_completed
            ),
            "further_mesh_confirmation_authorized": False,
            "n96_physical_evolution_executed": False,
            "n256_physical_evolution_executed": False,
            "operator_correction_authorized": False,
            "long_evolution_authorized": False,
            "tide_authorized": False,
            "wind_authorized": False,
            "stability_authorized": False,
            "hot_state_search_authorized": False,
            "limit_cycle_search_authorized": False,
        },
        "passed": mesh_gate_certified,
        "decision": (
            "bounded_first_order_mesh_gate_certified_at_n128"
            if mesh_gate_certified
            else (
                "stop_and_reassess_first_order_method"
                if duration_n128_passed
                else (
                    "stop_after_n128_duration_failure"
                    if duration_n128 is not None
                    else (
                        "stop_after_n128_short_gate"
                        if short_n128 is not None
                        else "stop_before_n128_evolution"
                    )
                )
            )
        ),
    }
    output_path = _absolute(
        DEFAULT_MESH_COMMON_N128_CONFIRMATION_OUTPUT
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
    if args.increment_primary_mesh_common_n128_confirmation_audit:
        _run_mesh_common_n128_confirmation_audit(args)
        return
    if args.increment_primary_mesh_common_n64_ledger_replay_audit:
        _run_mesh_common_n64_ledger_replay_audit(args)
        return
    if args.increment_primary_mesh_common_n64_confirmation_audit:
        _run_mesh_common_n64_confirmation_audit(args)
        return
    if args.increment_primary_mesh_common_spatial_response_audit:
        _run_mesh_common_spatial_response_audit(args)
        return
    if args.increment_primary_mesh_common_temporal_parity_audit:
        _run_mesh_common_temporal_parity_audit(args)
        return
    if args.increment_primary_mesh_common_startup_duration_audit:
        _run_mesh_common_startup_duration_audit(args)
        return
    if args.increment_primary_source_compatible_duration_audit:
        _run_source_compatible_duration_audit(args)
        return
    if args.increment_primary_source_compatible_startup_audit:
        _run_source_compatible_startup_audit(args)
        return
    if args.increment_primary_matched_source_control_audit:
        _run_matched_source_control_audit(args)
        return
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
