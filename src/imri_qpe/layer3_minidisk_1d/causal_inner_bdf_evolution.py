"""Fixed-step increment-primary BDF evolution for the causal five-field DAE."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from imri_qpe.constants import C

from .causal_inner_bdf import (
    CausalFiveFieldBDFHistory,
    causal_bdf_coefficients,
    causal_bdf_weighted_increment,
)
from .causal_inner_dae import causal_five_field_dae_count
from .causal_inner_dae_system import (
    CausalFiveFieldDAEContext,
    causal_five_field_bdf_history,
    causal_five_field_colored_central_jacobian,
    causal_five_field_dae_jacobian_color_groups,
    causal_five_field_dae_jacobian_sparsity,
    causal_five_field_dae_scaling,
    causal_five_field_equilibrated_sparse_solve,
    causal_five_field_path_temporal_storage_increment,
    evaluate_causal_five_field_dae,
    evaluate_causal_five_field_increment_bdf,
    unpack_causal_five_field_state,
)
from .causal_inner_diagnostics import audit_causal_five_field_state_gates
from .causal_inner_evolution import CausalFiveFieldAdaptiveStepConfig


@dataclass(frozen=True)
class CausalFiveFieldBDFDiscreteLedger:
    """Component-separated ledger for the solved BDF formula."""

    weighted_conserved_storage: np.ndarray
    weighted_vertical_storage: np.ndarray
    endpoint_boundary_transport: np.ndarray
    endpoint_endogenous_source: np.ndarray
    endpoint_prescribed_stream_source: np.ndarray
    closure_defect: np.ndarray


@dataclass(frozen=True)
class CausalFiveFieldBDFPhysicalIntervalLedger:
    """Actual interval storage plus trapezoidal physical transport."""

    actual_conserved_storage: np.ndarray
    actual_vertical_storage: np.ndarray
    trapezoidal_boundary_transport: np.ndarray
    trapezoidal_endogenous_source: np.ndarray
    exact_prescribed_stream_source: np.ndarray
    closure_defect: np.ndarray


@dataclass(frozen=True)
class CausalFiveFieldBDFStepResult:
    """One accepted or rejected sparse BDF1/BDF2 step."""

    state_vector: np.ndarray
    physical_increment: np.ndarray
    history: CausalFiveFieldBDFHistory | None
    order: int
    accepted: bool
    timestep_seconds: float
    maximum_scaled_residual: float
    maximum_scaled_algebraic_residual: float
    maximum_scaled_primitive_change: float
    maximum_scaled_total_change: float
    maximum_discrete_ledger_relative_defect: float
    discrete_component_relative_defects: tuple[float, ...]
    physical_interval_ledger_relative_defect: float
    physical_interval_component_relative_defects: tuple[float, ...]
    discrete_ledger: CausalFiveFieldBDFDiscreteLedger
    physical_interval_ledger: CausalFiveFieldBDFPhysicalIntervalLedger
    minimum_scattering_optical_depth: float
    outer_boundary_choked_before: bool
    outer_boundary_choked_after: bool
    iterations: int
    function_evaluations: int
    jacobian_evaluations: int
    maximum_linear_residual: float
    jacobian_nonzeros: int
    jacobian_color_count: int
    message: str


@dataclass(frozen=True)
class CausalFiveFieldFixedBDF2Result:
    """One exact-duration fixed-step BDF1-started BDF2 trajectory."""

    state_vector: np.ndarray
    history: CausalFiveFieldBDFHistory | None
    subdivisions: int
    timestep_seconds: float
    completed_steps: int
    bdf1_steps: int
    bdf2_steps: int
    state_gates: dict[str, object]
    maximum_scaled_residual: float
    maximum_scaled_algebraic_residual: float
    maximum_scaled_primitive_change: float
    maximum_scaled_total_change: float
    maximum_discrete_ledger_relative_defect: float
    cumulative_physical_ledger: CausalFiveFieldBDFPhysicalIntervalLedger
    cumulative_physical_ledger_relative_defect: float
    cumulative_physical_component_relative_defects: tuple[float, ...]
    maximum_linear_residual: float
    maximum_newton_iterations: int
    function_evaluations: int
    jacobian_evaluations: int
    newton_iterations: int
    passed: bool
    message: str


def _prescribed_stream_increment(
    context: CausalFiveFieldDAEContext,
    timestep_seconds: float,
) -> np.ndarray:
    result = np.zeros(5, dtype=float)
    if context.stream_sources is not None:
        result[:4] = (
            float(timestep_seconds)
            * np.sum(context.stream_sources.matrix, axis=0)
        )
    return result


def _relative_ledger_defects(
    closure_defect: np.ndarray,
    components: tuple[np.ndarray, ...],
) -> tuple[float, tuple[float, ...]]:
    scale = np.sum(
        np.asarray([np.abs(values) for values in components]),
        axis=0,
    )
    relative = np.abs(closure_defect) / np.maximum(
        scale,
        np.finfo(float).tiny,
    )
    return (
        float(np.max(relative)),
        tuple(float(value) for value in relative),
    )


def causal_five_field_bdf_step_ledgers(
    context: CausalFiveFieldDAEContext,
    old_vector: np.ndarray,
    physical_increment: np.ndarray,
    timestep_seconds: float,
    *,
    order: int,
    history: CausalFiveFieldBDFHistory | None,
) -> tuple[
    CausalFiveFieldBDFDiscreteLedger,
    CausalFiveFieldBDFPhysicalIntervalLedger,
]:
    """Return the discrete BDF and actual physical interval ledgers."""

    context = context.validated()
    n_cells = int(context.grid.centers.size)
    count = causal_five_field_dae_count(n_cells)
    old_values = np.asarray(old_vector, dtype=float)
    increment = np.asarray(physical_increment, dtype=float)
    timestep = float(timestep_seconds)
    if (
        old_values.shape != (count.total_unknowns,)
        or increment.shape != old_values.shape
        or np.any(~np.isfinite(old_values))
        or np.any(~np.isfinite(increment))
        or not np.isfinite(timestep)
        or timestep <= 0.0
    ):
        raise ValueError("causal BDF ledger inputs are invalid")
    if order == 1:
        if history is not None:
            raise ValueError("BDF1 ledger does not consume history")
        validated_history = None
        coefficients = causal_bdf_coefficients(1, timestep)
    elif order == 2:
        if history is None:
            raise ValueError("BDF2 ledger requires history")
        validated_history = history.validated(
            total_unknowns=count.total_unknowns,
            n_cells=n_cells,
        )
        coefficients = causal_bdf_coefficients(
            2,
            timestep,
            validated_history.previous_timestep_seconds,
        )
    else:
        raise ValueError("causal BDF ledger order must be one or two")

    new_values = old_values + increment
    old_state = unpack_causal_five_field_state(old_values, n_cells)
    new_state = unpack_causal_five_field_state(new_values, n_cells)
    old_evaluation = evaluate_causal_five_field_dae(
        old_values,
        context,
    )
    new_evaluation = evaluate_causal_five_field_dae(
        new_values,
        context,
    )
    current_conserved = np.sum(
        context.grid.cell_measures[:, None]
        * increment[: 5 * n_cells].reshape(n_cells, 5),
        axis=0,
    )
    previous_conserved = None
    if validated_history is not None:
        previous_conserved = np.sum(
            context.grid.cell_measures[:, None]
            * validated_history.previous_physical_increment[
                : 5 * n_cells
            ].reshape(n_cells, 5),
            axis=0,
        )
    temporal = causal_five_field_path_temporal_storage_increment(
        context,
        old_state.primitives,
        new_state.primitives,
    )
    current_vertical = np.zeros(5, dtype=float)
    current_vertical[:4] = np.sum(
        context.grid.cell_measures[:, None]
        * temporal.vertical_killing_increment,
        axis=0,
    )
    previous_vertical = None
    if validated_history is not None:
        previous_vertical = np.zeros(5, dtype=float)
        previous_vertical[:4] = np.sum(
            context.grid.cell_measures[:, None]
            * validated_history.previous_vertical_killing_increment,
            axis=0,
        )

    weighted_conserved = causal_bdf_weighted_increment(
        current_conserved,
        previous_conserved,
        coefficients,
    )
    weighted_vertical = causal_bdf_weighted_increment(
        current_vertical,
        previous_vertical,
        coefficients,
    )
    endpoint_boundary = (
        C
        * timestep
        * (
            new_state.weighted_face_fluxes_over_c[-1]
            - new_state.weighted_face_fluxes_over_c[0]
        )
    )
    endpoint_total_source = (
        C
        * timestep
        * np.sum(new_evaluation.integrated_sources_per_ct, axis=0)
    )
    exact_stream = _prescribed_stream_increment(context, timestep)
    endpoint_endogenous = endpoint_total_source - exact_stream
    discrete_defect = (
        weighted_conserved
        + weighted_vertical
        + endpoint_boundary
        - endpoint_endogenous
        - exact_stream
    )
    discrete = CausalFiveFieldBDFDiscreteLedger(
        weighted_conserved_storage=weighted_conserved,
        weighted_vertical_storage=weighted_vertical,
        endpoint_boundary_transport=endpoint_boundary,
        endpoint_endogenous_source=endpoint_endogenous,
        endpoint_prescribed_stream_source=exact_stream,
        closure_defect=discrete_defect,
    )

    old_boundary_rate = C * (
        old_state.weighted_face_fluxes_over_c[-1]
        - old_state.weighted_face_fluxes_over_c[0]
    )
    new_boundary_rate = C * (
        new_state.weighted_face_fluxes_over_c[-1]
        - new_state.weighted_face_fluxes_over_c[0]
    )
    trapezoidal_boundary = (
        0.5 * timestep * (old_boundary_rate + new_boundary_rate)
    )
    old_total_source_rate = C * np.sum(
        old_evaluation.integrated_sources_per_ct,
        axis=0,
    )
    new_total_source_rate = C * np.sum(
        new_evaluation.integrated_sources_per_ct,
        axis=0,
    )
    trapezoidal_total_source = (
        0.5
        * timestep
        * (old_total_source_rate + new_total_source_rate)
    )
    trapezoidal_endogenous = trapezoidal_total_source - exact_stream
    physical_defect = (
        current_conserved
        + current_vertical
        + trapezoidal_boundary
        - trapezoidal_endogenous
        - exact_stream
    )
    physical = CausalFiveFieldBDFPhysicalIntervalLedger(
        actual_conserved_storage=current_conserved,
        actual_vertical_storage=current_vertical,
        trapezoidal_boundary_transport=trapezoidal_boundary,
        trapezoidal_endogenous_source=trapezoidal_endogenous,
        exact_prescribed_stream_source=exact_stream,
        closure_defect=physical_defect,
    )
    return discrete, physical


def advance_causal_five_field_increment_bdf(
    context: CausalFiveFieldDAEContext,
    old_vector: np.ndarray,
    timestep_seconds: float,
    initial_physical_increment: np.ndarray,
    config: CausalFiveFieldAdaptiveStepConfig,
    *,
    order: int,
    history: CausalFiveFieldBDFHistory | None = None,
) -> CausalFiveFieldBDFStepResult:
    """Advance one sparse increment-primary BDF1 or BDF2 step."""

    context = context.validated()
    config = config.validated()
    n_cells = int(context.grid.centers.size)
    count = causal_five_field_dae_count(n_cells)
    old_values = np.asarray(old_vector, dtype=float)
    predictor = np.asarray(initial_physical_increment, dtype=float)
    timestep = float(timestep_seconds)
    if (
        old_values.shape != (count.total_unknowns,)
        or predictor.shape != old_values.shape
        or np.any(~np.isfinite(old_values))
        or np.any(~np.isfinite(predictor))
        or not np.isfinite(timestep)
        or timestep <= 0.0
    ):
        raise ValueError("causal BDF step inputs are invalid")
    if order == 1:
        if history is not None:
            raise ValueError("BDF1 step does not consume history")
        validated_history = None
    elif order == 2:
        if history is None:
            raise ValueError("BDF2 step requires history")
        validated_history = history.validated(
            total_unknowns=count.total_unknowns,
            n_cells=n_cells,
        )
        causal_bdf_coefficients(
            2,
            timestep,
            validated_history.previous_timestep_seconds,
        )
    else:
        raise ValueError("causal BDF step order must be one or two")

    old_state = unpack_causal_five_field_state(old_values, n_cells)
    stationary = evaluate_causal_five_field_dae(old_values, context)
    scaling = causal_five_field_dae_scaling(old_state, stationary)
    state = predictor / scaling.column_scales
    pattern = causal_five_field_dae_jacobian_sparsity(
        n_cells,
        spatial_reconstruction=context.spatial_reconstruction,
        boundary_trace_reconstruction=(
            context.boundary_trace_reconstruction
        ),
        cell_rate_scheme=context.cell_rate_scheme,
        cell_source_quadrature=context.cell_source_quadrature,
        cell_storage_quadrature=context.cell_storage_quadrature,
    )
    color_count = len(
        causal_five_field_dae_jacobian_color_groups(pattern)
    )
    function_evaluations = 0
    jacobian_evaluations = 0

    def residual(scaled_increment: np.ndarray) -> np.ndarray:
        nonlocal function_evaluations
        function_evaluations += 1
        physical_increment = (
            scaling.column_scales
            * np.asarray(scaled_increment, dtype=float)
        )
        return (
            evaluate_causal_five_field_increment_bdf(
                physical_increment,
                context,
                old_vector=old_values,
                timestep_seconds=timestep,
                order=order,
                history=validated_history,
                temporal_height_scheme="path_integrated",
            ).residual
            / scaling.row_scales
        )

    bound = 1.25 * config.maximum_scaled_total_change
    values = residual(state)
    success = False
    message = "maximum Newton iterations reached"
    linear_residuals: list[float] = []
    iterations = 0
    matrix = None
    matrix_age = config.jacobian_reuse_iterations
    for iteration in range(config.maximum_newton_iterations + 1):
        iterations = iteration
        maximum_residual = float(np.max(np.abs(values)))
        if maximum_residual <= config.residual_tolerance:
            success = True
            message = "residual gate passed"
            break
        if iteration == config.maximum_newton_iterations:
            break
        rebuilt_matrix = bool(
            matrix is None
            or matrix_age >= config.jacobian_reuse_iterations
        )
        if rebuilt_matrix:
            matrix = causal_five_field_colored_central_jacobian(
                residual,
                state,
                pattern,
                finite_difference_step=config.finite_difference_step,
            )
            jacobian_evaluations += 1
            matrix_age = 0
        try:
            correction, linear_audit = (
                causal_five_field_equilibrated_sparse_solve(
                    matrix,
                    -values,
                )
            )
        except (np.linalg.LinAlgError, RuntimeError):
            message = "equilibrated sparse Newton matrix is singular"
            break
        linear_residuals.append(linear_audit.relative_linear_residual)
        alpha = 1.0
        positive = correction > 0.0
        negative = correction < 0.0
        if np.any(positive):
            alpha = min(
                alpha,
                float(
                    np.min(
                        (bound - state[positive])
                        / correction[positive]
                    )
                ),
            )
        if np.any(negative):
            alpha = min(
                alpha,
                float(
                    np.min(
                        (-bound - state[negative])
                        / correction[negative]
                    )
                ),
            )
        alpha = min(1.0, max(0.0, 0.99 * alpha))
        accepted_correction = False
        for _line_search in range(14):
            candidate = state + alpha * correction
            candidate_values = residual(candidate)
            if np.max(np.abs(candidate_values)) < maximum_residual:
                state = candidate
                values = candidate_values
                accepted_correction = True
                break
            alpha *= 0.5
        if not accepted_correction:
            if not rebuilt_matrix:
                matrix = None
                matrix_age = config.jacobian_reuse_iterations
                continue
            message = "bound-aware line search failed"
            break
        matrix_age += 1

    physical_increment = scaling.column_scales * state
    new_vector = old_values + physical_increment
    evaluation = evaluate_causal_five_field_increment_bdf(
        physical_increment,
        context,
        old_vector=old_values,
        timestep_seconds=timestep,
        order=order,
        history=validated_history,
        temporal_height_scheme="path_integrated",
    )
    scaled_residual = evaluation.residual / scaling.row_scales
    n_differential = 5 * n_cells
    maximum_scaled_residual = float(np.max(np.abs(scaled_residual)))
    maximum_algebraic_residual = float(
        np.max(np.abs(scaled_residual[n_differential:]))
    )
    maximum_primitive_change = float(
        np.max(np.abs(state[n_differential : 2 * n_differential]))
    )
    maximum_total_change = float(np.max(np.abs(state)))
    discrete, physical = causal_five_field_bdf_step_ledgers(
        context,
        old_values,
        physical_increment,
        timestep,
        order=order,
        history=validated_history,
    )
    discrete_maximum, discrete_components = _relative_ledger_defects(
        discrete.closure_defect,
        (
            discrete.weighted_conserved_storage,
            discrete.weighted_vertical_storage,
            discrete.endpoint_boundary_transport,
            discrete.endpoint_endogenous_source,
            discrete.endpoint_prescribed_stream_source,
        ),
    )
    physical_maximum, physical_components = _relative_ledger_defects(
        physical.closure_defect,
        (
            physical.actual_conserved_storage,
            physical.actual_vertical_storage,
            physical.trapezoidal_boundary_transport,
            physical.trapezoidal_endogenous_source,
            physical.exact_prescribed_stream_source,
        ),
    )
    minimum_optical_depth = float(
        np.min(evaluation.scattering_optical_depths)
    )
    accepted = bool(
        success
        and maximum_scaled_residual <= config.residual_tolerance
        and maximum_algebraic_residual
        <= config.algebraic_residual_tolerance
        and maximum_primitive_change
        <= config.maximum_scaled_primitive_change
        and maximum_total_change <= config.maximum_scaled_total_change
        and discrete_maximum <= config.conservation_tolerance
        and minimum_optical_depth > 1.0
        and evaluation.outer_boundary_choked
        == stationary.outer_boundary_choked
    )
    if success and not accepted:
        message = "root exceeds one or more BDF-step acceptance gates"
    accepted_history = (
        causal_five_field_bdf_history(
            context,
            new_vector,
            physical_increment,
            timestep,
            temporal_height_scheme="path_integrated",
        )
        if accepted
        else None
    )
    return CausalFiveFieldBDFStepResult(
        state_vector=new_vector if accepted else old_values,
        physical_increment=(
            physical_increment
            if accepted
            else np.zeros_like(physical_increment)
        ),
        history=accepted_history,
        order=order,
        accepted=accepted,
        timestep_seconds=timestep,
        maximum_scaled_residual=maximum_scaled_residual,
        maximum_scaled_algebraic_residual=maximum_algebraic_residual,
        maximum_scaled_primitive_change=maximum_primitive_change,
        maximum_scaled_total_change=maximum_total_change,
        maximum_discrete_ledger_relative_defect=discrete_maximum,
        discrete_component_relative_defects=discrete_components,
        physical_interval_ledger_relative_defect=physical_maximum,
        physical_interval_component_relative_defects=physical_components,
        discrete_ledger=discrete,
        physical_interval_ledger=physical,
        minimum_scattering_optical_depth=minimum_optical_depth,
        outer_boundary_choked_before=stationary.outer_boundary_choked,
        outer_boundary_choked_after=evaluation.outer_boundary_choked,
        iterations=iterations,
        function_evaluations=function_evaluations,
        jacobian_evaluations=jacobian_evaluations,
        maximum_linear_residual=(
            float(max(linear_residuals)) if linear_residuals else 0.0
        ),
        jacobian_nonzeros=int(pattern.nnz),
        jacobian_color_count=color_count,
        message=message,
    )


def _zero_physical_ledger() -> CausalFiveFieldBDFPhysicalIntervalLedger:
    zero = np.zeros(5, dtype=float)
    return CausalFiveFieldBDFPhysicalIntervalLedger(
        actual_conserved_storage=np.array(zero, copy=True),
        actual_vertical_storage=np.array(zero, copy=True),
        trapezoidal_boundary_transport=np.array(zero, copy=True),
        trapezoidal_endogenous_source=np.array(zero, copy=True),
        exact_prescribed_stream_source=np.array(zero, copy=True),
        closure_defect=np.array(zero, copy=True),
    )


def _add_physical_ledgers(
    left: CausalFiveFieldBDFPhysicalIntervalLedger,
    right: CausalFiveFieldBDFPhysicalIntervalLedger,
) -> CausalFiveFieldBDFPhysicalIntervalLedger:
    return CausalFiveFieldBDFPhysicalIntervalLedger(
        actual_conserved_storage=(
            left.actual_conserved_storage
            + right.actual_conserved_storage
        ),
        actual_vertical_storage=(
            left.actual_vertical_storage + right.actual_vertical_storage
        ),
        trapezoidal_boundary_transport=(
            left.trapezoidal_boundary_transport
            + right.trapezoidal_boundary_transport
        ),
        trapezoidal_endogenous_source=(
            left.trapezoidal_endogenous_source
            + right.trapezoidal_endogenous_source
        ),
        exact_prescribed_stream_source=(
            left.exact_prescribed_stream_source
            + right.exact_prescribed_stream_source
        ),
        closure_defect=left.closure_defect + right.closure_defect,
    )


def evolve_causal_five_field_fixed_bdf2(
    context: CausalFiveFieldDAEContext,
    initial_vector: np.ndarray,
    previous_physical_increment: np.ndarray,
    previous_dt: float,
    duration_seconds: float,
    subdivisions: int,
    step_config: CausalFiveFieldAdaptiveStepConfig,
    *,
    startup_with_bdf1: bool = True,
    initial_history: CausalFiveFieldBDFHistory | None = None,
    progress: Callable[
        [int, int, np.ndarray, CausalFiveFieldBDFHistory],
        None,
    ]
    | None = None,
) -> CausalFiveFieldFixedBDF2Result:
    """Evolve one exact-duration fixed-step BDF2 trajectory."""

    context = context.validated()
    step_config = step_config.validated()
    n_cells = int(context.grid.centers.size)
    count = causal_five_field_dae_count(n_cells)
    state = np.asarray(initial_vector, dtype=float)
    previous_increment = np.asarray(
        previous_physical_increment,
        dtype=float,
    )
    prior_dt = float(previous_dt)
    duration = float(duration_seconds)
    if (
        state.shape != (count.total_unknowns,)
        or previous_increment.shape != state.shape
        or np.any(~np.isfinite(state))
        or np.any(~np.isfinite(previous_increment))
        or not np.isfinite(prior_dt)
        or prior_dt <= 0.0
        or not np.isfinite(duration)
        or duration <= 0.0
    ):
        raise ValueError("fixed BDF2 trajectory inputs are invalid")
    if int(subdivisions) != subdivisions or subdivisions < 1:
        raise ValueError("fixed BDF2 subdivisions must be positive")
    n_steps = int(subdivisions)
    timestep = duration / n_steps
    if (
        timestep < step_config.minimum_dt
        or timestep > step_config.maximum_dt
    ):
        raise ValueError("fixed BDF2 timestep lies outside step bounds")
    if startup_with_bdf1:
        if initial_history is not None:
            raise ValueError("BDF1 startup does not accept initial history")
        history = None
    else:
        if initial_history is None:
            raise ValueError("continued BDF2 trajectory requires history")
        history = initial_history.validated(
            total_unknowns=count.total_unknowns,
            n_cells=n_cells,
        )
        if not np.isclose(
            history.previous_timestep_seconds,
            timestep,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError("continued BDF2 history has a different timestep")
        previous_increment = np.asarray(
            history.previous_physical_increment,
            dtype=float,
        )
        prior_dt = history.previous_timestep_seconds

    maximum_scaled_residual = 0.0
    maximum_scaled_algebraic_residual = 0.0
    maximum_scaled_primitive_change = 0.0
    maximum_scaled_total_change = 0.0
    maximum_discrete_ledger_defect = 0.0
    maximum_linear_residual = 0.0
    maximum_newton_iterations = 0
    function_evaluations = 0
    jacobian_evaluations = 0
    newton_iterations = 0
    completed = 0
    bdf1_steps = 0
    bdf2_steps = 0
    passed = True
    message = "fixed BDF2 trajectory completed"
    cumulative_physical = _zero_physical_ledger()

    for index in range(n_steps):
        order = 1 if startup_with_bdf1 and index == 0 else 2
        predictor = previous_increment * (timestep / prior_dt)
        step = advance_causal_five_field_increment_bdf(
            context,
            state,
            timestep,
            predictor,
            step_config,
            order=order,
            history=history if order == 2 else None,
        )
        maximum_scaled_residual = max(
            maximum_scaled_residual,
            step.maximum_scaled_residual,
        )
        maximum_scaled_algebraic_residual = max(
            maximum_scaled_algebraic_residual,
            step.maximum_scaled_algebraic_residual,
        )
        maximum_scaled_primitive_change = max(
            maximum_scaled_primitive_change,
            step.maximum_scaled_primitive_change,
        )
        maximum_scaled_total_change = max(
            maximum_scaled_total_change,
            step.maximum_scaled_total_change,
        )
        maximum_discrete_ledger_defect = max(
            maximum_discrete_ledger_defect,
            step.maximum_discrete_ledger_relative_defect,
        )
        maximum_linear_residual = max(
            maximum_linear_residual,
            step.maximum_linear_residual,
        )
        maximum_newton_iterations = max(
            maximum_newton_iterations,
            step.iterations,
        )
        function_evaluations += step.function_evaluations
        jacobian_evaluations += step.jacobian_evaluations
        newton_iterations += step.iterations
        if not step.accepted or step.history is None:
            passed = False
            message = "fixed BDF2 trajectory failed a step contract"
            break
        state_gates = audit_causal_five_field_state_gates(
            context,
            step.state_vector,
        )
        if not state_gates["passed"]:
            passed = False
            message = "fixed BDF2 trajectory failed a state gate"
            break
        state = np.asarray(step.state_vector, dtype=float)
        previous_increment = np.asarray(
            step.physical_increment,
            dtype=float,
        )
        history = step.history
        prior_dt = timestep
        cumulative_physical = _add_physical_ledgers(
            cumulative_physical,
            step.physical_interval_ledger,
        )
        completed = index + 1
        if order == 1:
            bdf1_steps += 1
        else:
            bdf2_steps += 1
        if progress is not None:
            progress(completed, n_steps, state, history)

    final_state_gates = audit_causal_five_field_state_gates(
        context,
        state,
    )
    cumulative_maximum, cumulative_components = _relative_ledger_defects(
        cumulative_physical.closure_defect,
        (
            cumulative_physical.actual_conserved_storage,
            cumulative_physical.actual_vertical_storage,
            cumulative_physical.trapezoidal_boundary_transport,
            cumulative_physical.trapezoidal_endogenous_source,
            cumulative_physical.exact_prescribed_stream_source,
        ),
    )
    passed = bool(
        passed
        and completed == n_steps
        and final_state_gates["passed"]
    )
    if (
        not final_state_gates["passed"]
        and message == "fixed BDF2 trajectory completed"
    ):
        message = "fixed BDF2 trajectory failed final state gates"
    return CausalFiveFieldFixedBDF2Result(
        state_vector=state,
        history=history,
        subdivisions=n_steps,
        timestep_seconds=timestep,
        completed_steps=completed,
        bdf1_steps=bdf1_steps,
        bdf2_steps=bdf2_steps,
        state_gates=final_state_gates,
        maximum_scaled_residual=maximum_scaled_residual,
        maximum_scaled_algebraic_residual=(
            maximum_scaled_algebraic_residual
        ),
        maximum_scaled_primitive_change=(
            maximum_scaled_primitive_change
        ),
        maximum_scaled_total_change=maximum_scaled_total_change,
        maximum_discrete_ledger_relative_defect=(
            maximum_discrete_ledger_defect
        ),
        cumulative_physical_ledger=cumulative_physical,
        cumulative_physical_ledger_relative_defect=(
            cumulative_maximum
        ),
        cumulative_physical_component_relative_defects=(
            cumulative_components
        ),
        maximum_linear_residual=maximum_linear_residual,
        maximum_newton_iterations=maximum_newton_iterations,
        function_evaluations=function_evaluations,
        jacobian_evaluations=jacobian_evaluations,
        newton_iterations=newton_iterations,
        passed=passed,
        message=message,
    )
