"""Versioned observables and local clocks for causal five-field evolution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imri_qpe.constants import C

from .causal_inner_dae import audit_causal_five_field_principal
from .causal_inner_dae_system import (
    CausalFiveFieldDAEContext,
    causal_five_field_cell_states,
    causal_five_field_dae_scaling,
    evaluate_causal_five_field_dae,
    unpack_causal_five_field_state,
)
from .causal_inner_evolution import (
    causal_five_field_h_over_r_profile,
    causal_five_field_loading_time,
)
from .causal_inner_thermal import (
    causal_diffusion_cooling_rate,
    kerr_schild_column_four_velocity,
)


CAUSAL_FIVE_FIELD_OBSERVABLE_SCHEMA_VERSION = (
    "causal-five-field-observables-v1"
)

CAUSAL_FIVE_FIELD_TEMPORAL_ACCURACY_GATES_V1 = {
    "cooling_power_proxy_relative": 1.0e-3,
    "cooling_power_proxy_outside_cutoff_relative": 1.0e-3,
    "inner_accretion_rate_relative": 1.0e-3,
    "maximum_log_h_over_r_profile": 2.0e-3,
    "maximum_integrated_conserved_relative": 1.0e-3,
    "maximum_baseline_scaled_state_difference": 2.0e-3,
}


@dataclass(frozen=True)
class CausalFiveFieldObservableSnapshot:
    """Declared temporal-accuracy observables for one accepted state."""

    schema_version: str
    cooling_power_proxy_erg_s: float
    cooling_power_proxy_outside_cutoff_erg_s: float
    cooling_inner_cutoff_cm: float
    inner_accretion_rate_g_s: float
    maximum_h_over_r: float
    h_over_r: np.ndarray
    integrated_conserved: np.ndarray


@dataclass(frozen=True)
class CausalFiveFieldLocalTimescaleAudit:
    """Cell-local coordinate-time clocks for the current DAE state."""

    characteristic_crossing_seconds: np.ndarray
    stress_relaxation_seconds: np.ndarray
    thermal_response_seconds: np.ndarray
    luminosity_response_seconds: np.ndarray
    cooling_log_temperature_derivative: np.ndarray
    radial_advection_seconds: np.ndarray
    local_loading_seconds: np.ndarray
    global_loading_seconds: float


def _positive_relative_difference(left: float, right: float) -> float:
    scale = max(abs(float(left)), abs(float(right)), np.finfo(float).tiny)
    return float(abs(float(left) - float(right)) / scale)


def causal_five_field_observable_snapshot(
    context: CausalFiveFieldDAEContext,
    vector: np.ndarray,
    *,
    cooling_inner_cutoff: float,
) -> CausalFiveFieldObservableSnapshot:
    """Evaluate the immutable v1 observables used by temporal control."""

    context = context.validated()
    cutoff = float(cooling_inner_cutoff)
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("cooling inner cutoff must be positive and finite")
    n_cells = int(context.grid.centers.size)
    state = unpack_causal_five_field_state(vector, n_cells)
    cells = causal_five_field_cell_states(context, vector)
    cooling = np.asarray(
        [
            causal_diffusion_cooling_rate(cell.thermodynamics)[0]
            for cell in cells
        ],
        dtype=float,
    )
    weighted_cooling = context.grid.cell_measures * cooling
    exterior = context.grid.centers >= cutoff
    if not np.any(exterior):
        raise ValueError("cooling cutoff excludes every grid cell")
    h_over_r = causal_five_field_h_over_r_profile(context, vector)
    integrated = np.sum(
        context.grid.cell_measures[:, None] * state.conserved,
        axis=0,
    )
    face_rates = C * state.weighted_face_fluxes_over_c
    return CausalFiveFieldObservableSnapshot(
        schema_version=CAUSAL_FIVE_FIELD_OBSERVABLE_SCHEMA_VERSION,
        cooling_power_proxy_erg_s=float(np.sum(weighted_cooling)),
        cooling_power_proxy_outside_cutoff_erg_s=float(
            np.sum(weighted_cooling[exterior])
        ),
        cooling_inner_cutoff_cm=cutoff,
        inner_accretion_rate_g_s=float(-face_rates[0, 0]),
        maximum_h_over_r=float(np.max(h_over_r)),
        h_over_r=np.asarray(h_over_r, dtype=float),
        integrated_conserved=np.asarray(integrated, dtype=float),
    )


def compare_causal_five_field_observables(
    full_step: CausalFiveFieldObservableSnapshot,
    two_half_steps: CausalFiveFieldObservableSnapshot,
) -> dict[str, float | list[float]]:
    """Return one-full-versus-two-half backward-Euler differences."""

    if (
        full_step.schema_version
        != CAUSAL_FIVE_FIELD_OBSERVABLE_SCHEMA_VERSION
        or two_half_steps.schema_version
        != CAUSAL_FIVE_FIELD_OBSERVABLE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported causal observable schema")
    if not np.isclose(
        full_step.cooling_inner_cutoff_cm,
        two_half_steps.cooling_inner_cutoff_cm,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("observable snapshots use different cutoffs")
    left_h = np.asarray(full_step.h_over_r, dtype=float)
    right_h = np.asarray(two_half_steps.h_over_r, dtype=float)
    left_integrated = np.asarray(
        full_step.integrated_conserved,
        dtype=float,
    )
    right_integrated = np.asarray(
        two_half_steps.integrated_conserved,
        dtype=float,
    )
    if left_h.shape != right_h.shape:
        raise ValueError("observable H/R profiles use different meshes")
    if left_integrated.shape != right_integrated.shape:
        raise ValueError("integrated observable vectors do not match")
    component_relative = np.asarray(
        [
            _positive_relative_difference(left, right)
            for left, right in zip(
                left_integrated,
                right_integrated,
                strict=True,
            )
        ],
        dtype=float,
    )
    return {
        "cooling_power_proxy_relative": (
            _positive_relative_difference(
                full_step.cooling_power_proxy_erg_s,
                two_half_steps.cooling_power_proxy_erg_s,
            )
        ),
        "cooling_power_proxy_outside_cutoff_relative": (
            _positive_relative_difference(
                full_step.cooling_power_proxy_outside_cutoff_erg_s,
                two_half_steps.cooling_power_proxy_outside_cutoff_erg_s,
            )
        ),
        "inner_accretion_rate_relative": (
            _positive_relative_difference(
                full_step.inner_accretion_rate_g_s,
                two_half_steps.inner_accretion_rate_g_s,
            )
        ),
        "maximum_h_over_r_absolute": float(
            abs(full_step.maximum_h_over_r - two_half_steps.maximum_h_over_r)
        ),
        "maximum_log_h_over_r_profile": float(
            np.max(np.abs(np.log(left_h / right_h)))
        ),
        "integrated_conserved_component_relative": [
            float(value) for value in component_relative
        ],
        "maximum_integrated_conserved_relative": float(
            np.max(component_relative)
        ),
    }


def compare_causal_five_field_endpoint_vectors(
    context: CausalFiveFieldDAEContext,
    baseline_vector: np.ndarray,
    left_vector: np.ndarray,
    right_vector: np.ndarray,
    *,
    cooling_inner_cutoff: float,
) -> dict[str, float | list[float]]:
    """Compare two endpoint states using the immutable v1 observables."""

    context = context.validated()
    n_cells = int(context.grid.centers.size)
    baseline = np.asarray(baseline_vector, dtype=float)
    left = np.asarray(left_vector, dtype=float)
    right = np.asarray(right_vector, dtype=float)
    if (
        baseline.shape != left.shape
        or left.shape != right.shape
        or np.any(~np.isfinite(baseline))
        or np.any(~np.isfinite(left))
        or np.any(~np.isfinite(right))
    ):
        raise ValueError("endpoint vectors must be finite and conformable")
    left_snapshot = causal_five_field_observable_snapshot(
        context,
        left,
        cooling_inner_cutoff=cooling_inner_cutoff,
    )
    right_snapshot = causal_five_field_observable_snapshot(
        context,
        right,
        cooling_inner_cutoff=cooling_inner_cutoff,
    )
    errors = compare_causal_five_field_observables(
        left_snapshot,
        right_snapshot,
    )
    baseline_state = unpack_causal_five_field_state(
        baseline,
        n_cells,
    )
    baseline_evaluation = evaluate_causal_five_field_dae(
        baseline,
        context,
    )
    scales = causal_five_field_dae_scaling(
        baseline_state,
        baseline_evaluation,
    ).column_scales
    errors["maximum_baseline_scaled_state_difference"] = float(
        np.max(np.abs((left - right) / scales))
    )
    return errors


def causal_five_field_temporal_error_ratio(
    errors: dict[str, float | list[float]],
    gates: dict[str, float],
) -> dict[str, object]:
    """Normalize declared temporal errors and identify controlling gates."""

    normalized: dict[str, float] = {}
    for name, limit in gates.items():
        tolerance = float(limit)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("temporal error gates must be positive")
        if name not in errors:
            raise ValueError(f"temporal error is missing {name}")
        value = float(errors[name])
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("temporal errors must be finite and non-negative")
        normalized[name] = value / tolerance
    maximum = max(normalized.values(), default=0.0)
    controlling = sorted(
        name
        for name, ratio in normalized.items()
        if np.isclose(ratio, maximum, rtol=1.0e-12, atol=0.0)
    )
    violated = sorted(
        name for name, ratio in normalized.items() if ratio > 1.0
    )
    return {
        "normalized_errors": normalized,
        "maximum_normalized_error": float(maximum),
        "controlling_observables": controlling,
        "violated_observables": violated,
        "passed": bool(maximum <= 1.0),
    }


def audit_causal_five_field_reference_convergence(
    coarse_errors: dict[str, float | list[float]],
    fine_errors: dict[str, float | list[float]],
    gates: dict[str, float],
    *,
    maximum_reference_uncertainty_fraction: float,
    minimum_observed_order: float,
    order_floor_fraction: float,
) -> dict[str, object]:
    """Audit halved-step reference convergence against declared gates."""

    uncertainty_limit = float(maximum_reference_uncertainty_fraction)
    minimum_order = float(minimum_observed_order)
    floor_fraction = float(order_floor_fraction)
    if (
        not np.isfinite(uncertainty_limit)
        or not 0.0 < uncertainty_limit < 1.0
        or not np.isfinite(minimum_order)
        or minimum_order <= 0.0
        or not np.isfinite(floor_fraction)
        or not 0.0 < floor_fraction < uncertainty_limit
    ):
        raise ValueError("reference-convergence contract is invalid")
    rows: dict[str, dict[str, object]] = {}
    for name, gate in gates.items():
        tolerance = float(gate)
        coarse = float(coarse_errors[name])
        fine = float(fine_errors[name])
        if (
            not np.isfinite(tolerance)
            or tolerance <= 0.0
            or not np.isfinite(coarse)
            or coarse < 0.0
            or not np.isfinite(fine)
            or fine < 0.0
        ):
            raise ValueError("reference errors and gates must be valid")
        coarse_ratio = coarse / tolerance
        fine_ratio = fine / tolerance
        below_floor = fine_ratio <= floor_fraction
        observed_order = None
        if fine > 0.0 and coarse > 0.0:
            observed_order = float(np.log2(coarse / fine))
        order_passed = bool(
            below_floor
            or (
                observed_order is not None
                and observed_order >= minimum_order
            )
        )
        uncertainty_passed = fine_ratio <= uncertainty_limit
        rows[name] = {
            "coarse_error": coarse,
            "fine_error": fine,
            "coarse_normalized_error": coarse_ratio,
            "fine_normalized_error": fine_ratio,
            "observed_order": observed_order,
            "below_order_floor": below_floor,
            "order_passed": order_passed,
            "uncertainty_passed": uncertainty_passed,
            "passed": bool(order_passed and uncertainty_passed),
        }
    failed = sorted(
        name for name, row in rows.items() if not bool(row["passed"])
    )
    return {
        "maximum_reference_uncertainty_fraction": uncertainty_limit,
        "minimum_observed_order": minimum_order,
        "order_floor_fraction": floor_fraction,
        "observables": rows,
        "failed_observables": failed,
        "passed": not failed,
    }


def audit_causal_five_field_endpoint_with_reference_uncertainty(
    endpoint_errors: dict[str, float | list[float]],
    reference_errors: dict[str, float | list[float]],
    gates: dict[str, float],
) -> dict[str, object]:
    """Conservatively add endpoint and selected-reference uncertainty."""

    endpoint = causal_five_field_temporal_error_ratio(
        endpoint_errors,
        gates,
    )
    reference = causal_five_field_temporal_error_ratio(
        reference_errors,
        gates,
    )
    combined = {
        name: (
            float(endpoint["normalized_errors"][name])
            + float(reference["normalized_errors"][name])
        )
        for name in gates
    }
    maximum = max(combined.values(), default=0.0)
    controlling = sorted(
        name
        for name, ratio in combined.items()
        if np.isclose(ratio, maximum, rtol=1.0e-12, atol=0.0)
    )
    violated = sorted(
        name for name, ratio in combined.items() if ratio > 1.0
    )
    return {
        "endpoint_gate_audit": endpoint,
        "reference_uncertainty_gate_audit": reference,
        "combined_normalized_errors": combined,
        "maximum_combined_normalized_error": float(maximum),
        "controlling_observables": controlling,
        "violated_observables": violated,
        "passed": not violated,
    }


def causal_backward_euler_step_doubling_factor(
    normalized_error: float,
    *,
    safety_factor: float = 0.8,
    minimum_factor: float = 0.25,
    maximum_factor: float = 2.0,
) -> float:
    """Return the first-order step-doubling timestep multiplier."""

    error = float(normalized_error)
    safety = float(safety_factor)
    minimum = float(minimum_factor)
    maximum = float(maximum_factor)
    values = (error, safety, minimum, maximum)
    if any(not np.isfinite(value) for value in values):
        raise ValueError("timestep-controller inputs must be finite")
    if error < 0.0 or safety <= 0.0:
        raise ValueError("error must be non-negative and safety positive")
    if not 0.0 < minimum <= maximum:
        raise ValueError("timestep-factor bounds are invalid")
    if error == 0.0:
        return maximum
    proposed = safety / np.sqrt(error)
    return float(np.clip(proposed, minimum, maximum))


def causal_backward_euler_horizon_budget_factor(
    normalized_error: float,
    *,
    safety_factor: float = 0.8,
    minimum_factor: float = 0.25,
    maximum_factor: float = 2.0,
) -> float:
    """Return the BE multiplier for a timestep-weighted global budget.

    Backward-Euler step-doubling differences are second order in the
    timestep. Dividing their gate by ``dt / T_output`` makes the normalized
    budget error first order in the timestep, so its controller exponent is
    one rather than one half.
    """

    error = float(normalized_error)
    safety = float(safety_factor)
    minimum = float(minimum_factor)
    maximum = float(maximum_factor)
    values = (error, safety, minimum, maximum)
    if any(not np.isfinite(value) for value in values):
        raise ValueError("horizon-controller inputs must be finite")
    if error < 0.0 or safety <= 0.0:
        raise ValueError("error must be non-negative and safety positive")
    if not 0.0 < minimum <= maximum:
        raise ValueError("timestep-factor bounds are invalid")
    if error == 0.0:
        return maximum
    proposed = safety / error
    return float(np.clip(proposed, minimum, maximum))


def causal_five_field_local_timescale_audit(
    context: CausalFiveFieldDAEContext,
    vector: np.ndarray,
) -> CausalFiveFieldLocalTimescaleAudit:
    """Measure causal, relaxation, thermal, advection, and loading clocks."""

    context = context.validated()
    n_cells = int(context.grid.centers.size)
    state = unpack_causal_five_field_state(vector, n_cells)
    cells = causal_five_field_cell_states(context, vector)
    crossing = np.empty(n_cells)
    relaxation = np.empty(n_cells)
    thermal = np.empty(n_cells)
    luminosity = np.empty(n_cells)
    cooling_derivative = np.full(n_cells, 4.0)
    advection = np.empty(n_cells)
    for index, (radius, cell) in enumerate(
        zip(context.grid.centers, cells, strict=True)
    ):
        principal = audit_causal_five_field_principal(
            cell.geometry,
            context.vertical_frequency.eos(float(radius)),
            cell.closure,
            surface_density=cell.primitive.surface_density,
            radial_velocity_over_c=(
                cell.primitive.radial_velocity_over_c
            ),
            azimuthal_velocity_over_c=(
                cell.primitive.azimuthal_velocity_over_c
            ),
            temperature=cell.thermodynamics.temperature,
        )
        maximum_speed = float(
            np.max(np.abs(principal.coordinate_speeds_over_c))
        )
        crossing[index] = (
            context.grid.edges[index + 1] - context.grid.edges[index]
        ) / (C * maximum_speed)
        relaxation[index] = (
            cell.stress.lorentz_factor
            * cell.closure.relaxation_time
            / cell.geometry.base.lapse
        )
        eos = context.vertical_frequency.eos(float(radius))
        derivatives = eos.derivatives(
            cell.primitive.surface_density,
            cell.thermodynamics.temperature,
        )
        cooling_rate = causal_diffusion_cooling_rate(
            cell.thermodynamics
        )[0]
        thermal_capacity = (
            cell.primitive.surface_density
            * derivatives.internal_energy_log_temperature
            + cell.primitive.integrated_pressure
            * derivatives.height_log_temperature
        )
        thermal[index] = thermal_capacity / cooling_rate
        luminosity[index] = (
            thermal[index] / cooling_derivative[index]
        )
        four_velocity = kerr_schild_column_four_velocity(
            cell.geometry,
            cell.primitive,
        )
        coordinate_radial_velocity = (
            C * four_velocity[1] / four_velocity[0]
        )
        advection[index] = (
            abs(float(radius) / coordinate_radial_velocity)
            if coordinate_radial_velocity != 0.0
            else np.inf
        )
    local_loading = np.full(n_cells, np.inf)
    if context.stream_sources is not None:
        active = context.stream_sources.rest_mass > 0.0
        local_loading[active] = (
            context.grid.cell_measures[active]
            * state.conserved[active, 0]
            / context.stream_sources.rest_mass[active]
        )
        global_loading = causal_five_field_loading_time(context, vector)
    else:
        global_loading = np.inf
    positive_arrays = (
        crossing,
        relaxation,
        thermal,
        luminosity,
        advection,
        local_loading,
    )
    if any(
        np.any((values <= 0.0) | np.isnan(values))
        for values in positive_arrays
    ):
        raise ValueError("causal local timescale audit is non-positive")
    return CausalFiveFieldLocalTimescaleAudit(
        characteristic_crossing_seconds=crossing,
        stress_relaxation_seconds=relaxation,
        thermal_response_seconds=thermal,
        luminosity_response_seconds=luminosity,
        cooling_log_temperature_derivative=cooling_derivative,
        radial_advection_seconds=advection,
        local_loading_seconds=local_loading,
        global_loading_seconds=float(global_loading),
    )


def audit_causal_five_field_state_gates(
    context: CausalFiveFieldDAEContext,
    vector: np.ndarray,
    *,
    maximum_h_over_r: float = 0.25,
    minimum_scattering_optical_depth: float = 1.0,
    maximum_inner_light_cone_excess: float = 1.0e-10,
    maximum_scaled_algebraic_residual: float = 1.0e-11,
) -> dict[str, object]:
    """Apply the established source-compatible causal state gates."""

    context = context.validated()
    n_cells = int(context.grid.centers.size)
    state = unpack_causal_five_field_state(vector, n_cells)
    evaluation = evaluate_causal_five_field_dae(vector, context)
    scaling = causal_five_field_dae_scaling(state, evaluation)
    cells = causal_five_field_cell_states(context, vector)
    inner = cells[0]
    principal = audit_causal_five_field_principal(
        inner.geometry,
        context.vertical_frequency.eos(float(context.grid.centers[0])),
        inner.closure,
        surface_density=inner.primitive.surface_density,
        radial_velocity_over_c=inner.primitive.radial_velocity_over_c,
        azimuthal_velocity_over_c=inner.primitive.azimuthal_velocity_over_c,
        temperature=inner.thermodynamics.temperature,
    )
    n_differential = 5 * n_cells
    algebraic = float(
        np.max(
            np.abs(
                evaluation.residual[n_differential:]
                / scaling.row_scales[n_differential:]
            )
        )
    )
    h_over_r = causal_five_field_h_over_r_profile(context, vector)
    measured = {
        "maximum_h_over_r": float(np.max(h_over_r)),
        "minimum_scattering_optical_depth": float(
            np.min(evaluation.scattering_optical_depths)
        ),
        "inner_incoming_characteristics": int(
            principal.incoming_inner_characteristics
        ),
        "maximum_inner_light_cone_excess": float(
            principal.maximum_light_cone_excess
        ),
        "outer_boundary_choked": bool(evaluation.outer_boundary_choked),
        "outer_incoming_characteristics": int(
            evaluation.outer_incoming_characteristics
        ),
        "maximum_scaled_algebraic_residual": algebraic,
    }
    gates = {
        "maximum_h_over_r": float(maximum_h_over_r),
        "minimum_scattering_optical_depth": float(
            minimum_scattering_optical_depth
        ),
        "inner_incoming_characteristics": 0,
        "maximum_inner_light_cone_excess": float(
            maximum_inner_light_cone_excess
        ),
        "outer_boundary_choked": False,
        "outer_incoming_characteristics": 2,
        "maximum_scaled_algebraic_residual": float(
            maximum_scaled_algebraic_residual
        ),
    }
    passed = bool(
        measured["maximum_h_over_r"] <= gates["maximum_h_over_r"]
        and measured["minimum_scattering_optical_depth"]
        > gates["minimum_scattering_optical_depth"]
        and measured["inner_incoming_characteristics"]
        == gates["inner_incoming_characteristics"]
        and measured["maximum_inner_light_cone_excess"]
        <= gates["maximum_inner_light_cone_excess"]
        and measured["outer_boundary_choked"]
        == gates["outer_boundary_choked"]
        and measured["outer_incoming_characteristics"]
        == gates["outer_incoming_characteristics"]
        and measured["maximum_scaled_algebraic_residual"]
        <= gates["maximum_scaled_algebraic_residual"]
    )
    return {
        "schema_version": "causal-five-field-state-gates-v1",
        "measured": measured,
        "gates": gates,
        "passed": passed,
    }
