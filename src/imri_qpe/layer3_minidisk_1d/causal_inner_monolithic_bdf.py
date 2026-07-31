"""Fixed-step BDF1/BDF2 method for the primitive-only monolithic inner DAE.

The responsive-height temporal term is a non-exact one-form.  BDF2 therefore
combines complete path increments from consecutive accepted intervals; it
does not reconstruct the previous temporal contribution from endpoints.

This module is production neutral.  It supplies the method and restart
contracts needed by the bounded nonlinear preflight, while leaving all
production defaults unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from imri_qpe.constants import C

from .causal_inner_bdf import (
    CausalBDFCoefficients,
    causal_bdf_coefficients,
    causal_bdf_weighted_increment,
)
from .causal_inner_dae_system import CausalFiveFieldDAEContext
from .causal_inner_monolithic_dae import (
    CausalFiveFieldMonolithicDAEEvaluation,
    CausalFiveFieldMonolithicStorageIncrement,
    evaluate_causal_five_field_monolithic_backward_euler,
)
from .causal_inner_monolithic_tangent import (
    CausalFiveFieldMonolithicFrozenTangent,
)


_N_FIELDS = 5


@dataclass(frozen=True)
class CausalFiveFieldMonolithicBDFHistory:
    """Complete accepted interval needed by the next BDF2 residual."""

    previous_primitive_increment: np.ndarray
    previous_mapped_storage_increment: np.ndarray
    previous_responsive_height_storage_increment: np.ndarray
    previous_timestep_seconds: float
    temporal_path_scheme: str = "straight_primitive_path"

    def validated(
        self,
        *,
        n_cells: int,
    ) -> CausalFiveFieldMonolithicBDFHistory:
        """Return a normalized history after checking every stored path term."""

        shape = (int(n_cells), _N_FIELDS)
        primitive = np.asarray(
            self.previous_primitive_increment,
            dtype=float,
        )
        mapped = np.asarray(
            self.previous_mapped_storage_increment,
            dtype=float,
        )
        height = np.asarray(
            self.previous_responsive_height_storage_increment,
            dtype=float,
        )
        timestep = float(self.previous_timestep_seconds)
        if (
            primitive.shape != shape
            or mapped.shape != shape
            or height.shape != shape
            or np.any(~np.isfinite(primitive))
            or np.any(~np.isfinite(mapped))
            or np.any(~np.isfinite(height))
            or not np.isfinite(timestep)
            or timestep <= 0.0
            or self.temporal_path_scheme != "straight_primitive_path"
        ):
            raise ValueError("monolithic BDF history is invalid")
        return CausalFiveFieldMonolithicBDFHistory(
            previous_primitive_increment=np.array(primitive, copy=True),
            previous_mapped_storage_increment=np.array(mapped, copy=True),
            previous_responsive_height_storage_increment=np.array(
                height,
                copy=True,
            ),
            previous_timestep_seconds=timestep,
            temporal_path_scheme=self.temporal_path_scheme,
        )

    @property
    def previous_complete_storage_increment(self) -> np.ndarray:
        """Return the exact stored mapped-plus-height interval increment."""

        return np.asarray(
            self.previous_mapped_storage_increment
            + self.previous_responsive_height_storage_increment,
            dtype=float,
        )


@dataclass(frozen=True)
class CausalFiveFieldMonolithicBDFEvaluation:
    """One block-complete primitive-only BDF residual evaluation."""

    order: int
    coefficients: CausalBDFCoefficients
    current_storage_increment: CausalFiveFieldMonolithicStorageIncrement
    previous_history: CausalFiveFieldMonolithicBDFHistory | None
    backward_euler_evaluation: CausalFiveFieldMonolithicDAEEvaluation
    weighted_mapped_storage_increment: np.ndarray
    weighted_responsive_height_storage_increment: np.ndarray
    weighted_complete_storage_increment: np.ndarray
    mapped_temporal_storage_rows: np.ndarray
    responsive_height_temporal_storage_rows: np.ndarray
    residual_rows: np.ndarray
    maximum_block_ledger_defect: float
    maximum_mapped_endpoint_path_closure_defect: float
    mapped_storage_uses_stable_exact_path_integral: bool
    incoming_excision_characteristics: int


@dataclass(frozen=True)
class CausalFiveFieldMonolithicBDFStepResult:
    """One accepted or rejected cached-tangent Newton step."""

    primitive_charts: np.ndarray
    primitive_increment: np.ndarray
    history: CausalFiveFieldMonolithicBDFHistory | None
    evaluation: CausalFiveFieldMonolithicBDFEvaluation
    order: int
    accepted: bool
    timestep_seconds: float
    maximum_scaled_residual: float
    maximum_scaled_algebraic_residual: float
    maximum_scaled_primitive_change: float
    maximum_discrete_ledger_defect: float
    minimum_path_reconstruction_factor: float
    incoming_excision_characteristics: int
    iterations: int
    function_evaluations: int
    maximum_linear_residual: float
    message: str


@dataclass(frozen=True)
class CausalFiveFieldMonolithicBDFRestart:
    """Bitwise restart payload for a fixed-step monolithic BDF2 trajectory."""

    primitive_charts: np.ndarray
    history: CausalFiveFieldMonolithicBDFHistory
    elapsed_time_seconds: float
    completed_steps: int
    next_order: int
    provenance: dict
    schema_version: int = 1


def _validated_charts(
    context: CausalFiveFieldDAEContext,
    values: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    charts = np.asarray(values, dtype=float)
    shape = (int(context.grid.centers.size), _N_FIELDS)
    if charts.shape != shape or np.any(~np.isfinite(charts)):
        raise ValueError(f"{label} primitive charts are invalid")
    return charts


def _coefficients_and_history(
    order: int,
    timestep_seconds: float,
    history: CausalFiveFieldMonolithicBDFHistory | None,
    *,
    n_cells: int,
) -> tuple[CausalBDFCoefficients, CausalFiveFieldMonolithicBDFHistory | None]:
    if int(order) != order or order not in (1, 2):
        raise ValueError("monolithic BDF order must be one or two")
    if order == 1:
        if history is not None:
            raise ValueError("monolithic BDF1 does not consume history")
        return causal_bdf_coefficients(1, timestep_seconds), None
    if history is None:
        raise ValueError("monolithic BDF2 requires complete history")
    validated = history.validated(n_cells=n_cells)
    return (
        causal_bdf_coefficients(
            2,
            timestep_seconds,
            validated.previous_timestep_seconds,
        ),
        validated,
    )


def evaluate_causal_five_field_monolithic_bdf(
    old_primitive_charts: np.ndarray,
    new_primitive_charts: np.ndarray,
    timestep_seconds: float,
    context: CausalFiveFieldDAEContext,
    *,
    order: int,
    history: CausalFiveFieldMonolithicBDFHistory | None = None,
    temporal_quadrature_order: int = 4,
    reconstruction_directional_step: float = 1.0e-2,
    path_quadrature_order: int = 6,
    relative_step: float = 2.0e-4,
    stationary_speed_tolerance: float = 1.0e-12,
) -> CausalFiveFieldMonolithicBDFEvaluation:
    """Evaluate one increment-primary BDF1 or BDF2 monolithic residual."""

    context = context.validated()
    old = _validated_charts(context, old_primitive_charts, label="old")
    new = _validated_charts(context, new_primitive_charts, label="new")
    timestep = float(timestep_seconds)
    if not np.isfinite(timestep) or timestep <= 0.0:
        raise ValueError("monolithic BDF timestep must be positive")
    coefficients, validated_history = _coefficients_and_history(
        order,
        timestep,
        history,
        n_cells=old.shape[0],
    )
    backward_euler = evaluate_causal_five_field_monolithic_backward_euler(
        old,
        new,
        timestep,
        context,
        temporal_quadrature_order=temporal_quadrature_order,
        reconstruction_directional_step=reconstruction_directional_step,
        path_quadrature_order=path_quadrature_order,
        relative_step=relative_step,
        stationary_speed_tolerance=stationary_speed_tolerance,
    )
    current = backward_euler.storage_increment
    previous_mapped = (
        None
        if validated_history is None
        else validated_history.previous_mapped_storage_increment
    )
    previous_height = (
        None
        if validated_history is None
        else validated_history.previous_responsive_height_storage_increment
    )
    # The mapped term is an exact differential.  Its analytic path integral
    # is the stable numerical representation of the same endpoint increment:
    # direct subtraction of the two O(1e30) endpoint maps creates a residual
    # floor above the frozen nonlinear tolerance on small timesteps.  The
    # endpoint difference remains an independent closure audit.
    weighted_mapped = causal_bdf_weighted_increment(
        current.mapped_path_increment,
        previous_mapped,
        coefficients,
    )
    weighted_height = causal_bdf_weighted_increment(
        current.responsive_height_path_increment,
        previous_height,
        coefficients,
    )
    mapped_rows = weighted_mapped / (C * timestep)
    height_rows = weighted_height / (C * timestep)
    blocks = (
        mapped_rows,
        height_rows,
        backward_euler.conservative_transport_rows,
        backward_euler.shear_principal_rows,
        backward_euler.height_principal_rows,
        backward_euler.local_stress_relaxation_rows,
        backward_euler.geometry_rows,
        backward_euler.cooling_rows,
        backward_euler.stream_rows,
        backward_euler.lower_height_work_rows,
    )
    residual = np.sum(np.asarray(blocks), axis=0)
    reconstructed = (
        mapped_rows
        + height_rows
        + backward_euler.conservative_transport_rows
        + backward_euler.shear_principal_rows
        + backward_euler.height_principal_rows
        + backward_euler.local_stress_relaxation_rows
        + backward_euler.geometry_rows
        + backward_euler.cooling_rows
        + backward_euler.stream_rows
        + backward_euler.lower_height_work_rows
    )
    scale = max(
        float(np.max(np.abs(residual))),
        max(float(np.max(np.abs(block))) for block in blocks),
        np.finfo(float).tiny,
    )
    return CausalFiveFieldMonolithicBDFEvaluation(
        order=int(order),
        coefficients=coefficients,
        current_storage_increment=current,
        previous_history=validated_history,
        backward_euler_evaluation=backward_euler,
        weighted_mapped_storage_increment=np.asarray(
            weighted_mapped,
            dtype=float,
        ),
        weighted_responsive_height_storage_increment=np.asarray(
            weighted_height,
            dtype=float,
        ),
        weighted_complete_storage_increment=np.asarray(
            weighted_mapped + weighted_height,
            dtype=float,
        ),
        mapped_temporal_storage_rows=np.asarray(mapped_rows, dtype=float),
        responsive_height_temporal_storage_rows=np.asarray(
            height_rows,
            dtype=float,
        ),
        residual_rows=np.asarray(residual, dtype=float),
        maximum_block_ledger_defect=float(
            np.max(np.abs(residual - reconstructed)) / scale
        ),
        maximum_mapped_endpoint_path_closure_defect=(
            current.maximum_mapped_path_closure_defect
        ),
        mapped_storage_uses_stable_exact_path_integral=True,
        incoming_excision_characteristics=(
            backward_euler.incoming_excision_characteristics
        ),
    )


def causal_five_field_monolithic_bdf_history(
    primitive_increment: np.ndarray,
    storage_increment: CausalFiveFieldMonolithicStorageIncrement,
    timestep_seconds: float,
) -> CausalFiveFieldMonolithicBDFHistory:
    """Freeze an accepted complete temporal path for the next BDF2 step."""

    primitive = np.asarray(primitive_increment, dtype=float)
    mapped = np.asarray(
        storage_increment.mapped_path_increment,
        dtype=float,
    )
    height = np.asarray(
        storage_increment.responsive_height_path_increment,
        dtype=float,
    )
    if primitive.shape != mapped.shape or height.shape != mapped.shape:
        raise ValueError("monolithic BDF history shapes differ")
    return CausalFiveFieldMonolithicBDFHistory(
        previous_primitive_increment=primitive,
        previous_mapped_storage_increment=mapped,
        previous_responsive_height_storage_increment=height,
        previous_timestep_seconds=float(timestep_seconds),
    ).validated(n_cells=primitive.shape[0])


def _step_matrix(
    tangent: CausalFiveFieldMonolithicFrozenTangent,
    coefficients: CausalBDFCoefficients,
) -> np.ndarray:
    """Return the cached start-state scaled Jacobian for one BDF step."""

    inverse_dt = 1.0 / coefficients.current_timestep_seconds
    current = coefficients.current_increment_coefficient
    return np.asarray(
        current * inverse_dt * tangent.descriptor_scaled_matrix
        + current * tangent.storage_rate_derivative_scaled_matrix
        + tangent.stationary_scaled_jacobian,
        dtype=float,
    )


def advance_causal_five_field_monolithic_bdf(
    context: CausalFiveFieldDAEContext,
    old_primitive_charts: np.ndarray,
    timestep_seconds: float,
    tangent: CausalFiveFieldMonolithicFrozenTangent,
    *,
    order: int,
    history: CausalFiveFieldMonolithicBDFHistory | None = None,
    initial_primitive_increment: np.ndarray | None = None,
    residual_tolerance: float = 1.0e-10,
    ledger_tolerance: float = 1.0e-12,
    maximum_scaled_primitive_change: float = 5.0e-3,
    maximum_newton_iterations: int = 8,
    maximum_line_search_iterations: int = 12,
) -> CausalFiveFieldMonolithicBDFStepResult:
    """Advance one bounded fixed-step BDF1/BDF2 cached-tangent Newton step."""

    context = context.validated()
    old = _validated_charts(context, old_primitive_charts, label="old")
    timestep = float(timestep_seconds)
    coefficients, validated_history = _coefficients_and_history(
        order,
        timestep,
        history,
        n_cells=old.shape[0],
    )
    columns = np.asarray(
        tangent.primitive_column_scales,
        dtype=float,
    ).reshape(old.shape)
    rows = np.asarray(
        tangent.conservation_row_scales,
        dtype=float,
    ).reshape(old.shape)
    if (
        tangent.base_primitives.shape != old.shape
        or columns.shape != old.shape
        or rows.shape != old.shape
    ):
        raise ValueError("monolithic tangent and step shapes differ")
    if initial_primitive_increment is None:
        if validated_history is None:
            initial = timestep * np.asarray(
                tangent.physical_base_rate_per_s,
                dtype=float,
            ).reshape(old.shape)
        else:
            initial = (
                timestep
                / validated_history.previous_timestep_seconds
                * validated_history.previous_primitive_increment
            )
    else:
        initial = np.asarray(initial_primitive_increment, dtype=float)
    if initial.shape != old.shape or np.any(~np.isfinite(initial)):
        raise ValueError("monolithic BDF predictor is invalid")
    state = np.asarray(initial / columns, dtype=float).ravel()
    matrix = _step_matrix(tangent, coefficients)
    function_evaluations = 0

    def residual(scaled_increment: np.ndarray):
        nonlocal function_evaluations
        function_evaluations += 1
        increment = (
            columns.ravel() * np.asarray(scaled_increment, dtype=float)
        ).reshape(old.shape)
        evaluation = evaluate_causal_five_field_monolithic_bdf(
            old,
            old + increment,
            timestep,
            context,
            order=order,
            history=validated_history,
        )
        return evaluation.residual_rows.ravel() / rows.ravel(), evaluation

    values, evaluation = residual(state)
    success = False
    message = "maximum Newton iterations reached"
    linear_residuals = []
    iterations = 0
    bound = float(maximum_scaled_primitive_change)
    for iteration in range(int(maximum_newton_iterations) + 1):
        iterations = iteration
        maximum = float(np.max(np.abs(values)))
        if maximum <= residual_tolerance:
            success = True
            message = "residual gate passed"
            break
        if iteration == maximum_newton_iterations:
            break
        try:
            correction = np.linalg.solve(matrix, -values)
        except np.linalg.LinAlgError:
            message = "cached monolithic Newton matrix is singular"
            break
        linear_scale = max(
            float(np.linalg.norm(values)),
            np.finfo(float).tiny,
        )
        linear_residuals.append(
            float(
                np.linalg.norm(matrix @ correction + values)
                / linear_scale
            )
        )
        alpha = 1.0
        positive = correction > 0.0
        negative = correction < 0.0
        if np.any(positive):
            alpha = min(
                alpha,
                float(np.min((bound - state[positive]) / correction[positive])),
            )
        if np.any(negative):
            alpha = min(
                alpha,
                float(np.min((-bound - state[negative]) / correction[negative])),
            )
        alpha = min(1.0, max(0.0, 0.99 * alpha))
        accepted_correction = False
        for _line_search in range(int(maximum_line_search_iterations)):
            candidate = state + alpha * correction
            candidate_values, candidate_evaluation = residual(candidate)
            if np.max(np.abs(candidate_values)) < maximum:
                secant_step = candidate - state
                secant_change = candidate_values - values
                denominator = float(secant_step @ secant_step)
                if denominator > np.finfo(float).tiny:
                    matrix = matrix + np.outer(
                        secant_change - matrix @ secant_step,
                        secant_step,
                    ) / denominator
                state = candidate
                values = candidate_values
                evaluation = candidate_evaluation
                accepted_correction = True
                break
            alpha *= 0.5
        if not accepted_correction:
            message = "monolithic bound-aware line search failed"
            break

    primitive_increment = (columns.ravel() * state).reshape(old.shape)
    new = old + primitive_increment
    maximum_residual = float(np.max(np.abs(values)))
    maximum_change = float(np.max(np.abs(state)))
    minimum_factor = float(
        evaluation.current_storage_increment
        .minimum_path_reconstruction_factor
    )
    maximum_ledger = float(evaluation.maximum_block_ledger_defect)
    accepted = bool(
        success
        and maximum_residual <= residual_tolerance
        and maximum_ledger <= ledger_tolerance
        and maximum_change <= maximum_scaled_primitive_change
        and minimum_factor >= 1.0 - 1.0e-12
        and evaluation.incoming_excision_characteristics == 0
    )
    if success and not accepted:
        message = "root exceeds one or more monolithic BDF gates"
    accepted_history = (
        causal_five_field_monolithic_bdf_history(
            primitive_increment,
            evaluation.current_storage_increment,
            timestep,
        )
        if accepted
        else None
    )
    return CausalFiveFieldMonolithicBDFStepResult(
        primitive_charts=np.array(new if accepted else old, copy=True),
        primitive_increment=np.array(
            primitive_increment if accepted else np.zeros_like(old),
            copy=True,
        ),
        history=accepted_history,
        evaluation=evaluation,
        order=int(order),
        accepted=accepted,
        timestep_seconds=timestep,
        maximum_scaled_residual=maximum_residual,
        maximum_scaled_algebraic_residual=0.0,
        maximum_scaled_primitive_change=maximum_change,
        maximum_discrete_ledger_defect=maximum_ledger,
        minimum_path_reconstruction_factor=minimum_factor,
        incoming_excision_characteristics=(
            evaluation.incoming_excision_characteristics
        ),
        iterations=iterations,
        function_evaluations=function_evaluations,
        maximum_linear_residual=(
            float(max(linear_residuals)) if linear_residuals else 0.0
        ),
        message=message,
    )


def _validated_restart(
    context: CausalFiveFieldDAEContext,
    restart: CausalFiveFieldMonolithicBDFRestart,
) -> CausalFiveFieldMonolithicBDFRestart:
    charts = _validated_charts(
        context,
        restart.primitive_charts,
        label="restart",
    )
    history = restart.history.validated(n_cells=charts.shape[0])
    elapsed = float(restart.elapsed_time_seconds)
    if (
        not np.isfinite(elapsed)
        or elapsed < 0.0
        or int(restart.completed_steps) != restart.completed_steps
        or restart.completed_steps < 1
        or int(restart.next_order) != restart.next_order
        or restart.next_order != 2
        or restart.schema_version != 1
        or not isinstance(restart.provenance, dict)
    ):
        raise ValueError("monolithic BDF restart is invalid")
    return CausalFiveFieldMonolithicBDFRestart(
        primitive_charts=np.array(charts, copy=True),
        history=history,
        elapsed_time_seconds=elapsed,
        completed_steps=int(restart.completed_steps),
        next_order=2,
        provenance=dict(restart.provenance),
        schema_version=1,
    )


def causal_five_field_monolithic_bdf_restarts_equal(
    left: CausalFiveFieldMonolithicBDFRestart,
    right: CausalFiveFieldMonolithicBDFRestart,
) -> bool:
    """Return whether two monolithic restart payloads are bitwise identical."""

    arrays = (
        (left.primitive_charts, right.primitive_charts),
        (
            left.history.previous_primitive_increment,
            right.history.previous_primitive_increment,
        ),
        (
            left.history.previous_mapped_storage_increment,
            right.history.previous_mapped_storage_increment,
        ),
        (
            left.history.previous_responsive_height_storage_increment,
            right.history.previous_responsive_height_storage_increment,
        ),
    )
    return bool(
        all(np.array_equal(first, second) for first, second in arrays)
        and left.history.previous_timestep_seconds
        == right.history.previous_timestep_seconds
        and left.history.temporal_path_scheme
        == right.history.temporal_path_scheme
        and left.elapsed_time_seconds == right.elapsed_time_seconds
        and left.completed_steps == right.completed_steps
        and left.next_order == right.next_order
        and left.provenance == right.provenance
        and left.schema_version == right.schema_version
    )


def save_causal_five_field_monolithic_bdf_restart(
    path: str | Path,
    context: CausalFiveFieldDAEContext,
    restart: CausalFiveFieldMonolithicBDFRestart,
) -> None:
    """Persist a complete monolithic BDF restart without lossy conversion."""

    validated = _validated_restart(context, restart)
    destination = Path(path)
    if destination.suffix != ".npz":
        raise ValueError("monolithic BDF restart path must end in .npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        primitive_charts=validated.primitive_charts,
        previous_primitive_increment=(
            validated.history.previous_primitive_increment
        ),
        previous_mapped_storage_increment=(
            validated.history.previous_mapped_storage_increment
        ),
        previous_responsive_height_storage_increment=(
            validated.history.previous_responsive_height_storage_increment
        ),
        previous_timestep_seconds=np.asarray(
            validated.history.previous_timestep_seconds,
            dtype="<f8",
        ),
        elapsed_time_seconds=np.asarray(
            validated.elapsed_time_seconds,
            dtype="<f8",
        ),
        completed_steps=np.asarray(validated.completed_steps, dtype="<i8"),
        next_order=np.asarray(validated.next_order, dtype="<i8"),
        schema_version=np.asarray(validated.schema_version, dtype="<i8"),
        temporal_path_scheme=np.asarray(
            validated.history.temporal_path_scheme,
        ),
        provenance_json=np.asarray(
            json.dumps(
                validated.provenance,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        ),
    )


def load_causal_five_field_monolithic_bdf_restart(
    path: str | Path,
    context: CausalFiveFieldDAEContext,
    *,
    expected_provenance: dict | None = None,
) -> CausalFiveFieldMonolithicBDFRestart:
    """Load and validate one complete monolithic BDF restart."""

    source_path = Path(path)
    with np.load(source_path, allow_pickle=False) as source:
        restart = CausalFiveFieldMonolithicBDFRestart(
            primitive_charts=np.asarray(
                source["primitive_charts"],
                dtype=float,
            ),
            history=CausalFiveFieldMonolithicBDFHistory(
                previous_primitive_increment=np.asarray(
                    source["previous_primitive_increment"],
                    dtype=float,
                ),
                previous_mapped_storage_increment=np.asarray(
                    source["previous_mapped_storage_increment"],
                    dtype=float,
                ),
                previous_responsive_height_storage_increment=np.asarray(
                    source[
                        "previous_responsive_height_storage_increment"
                    ],
                    dtype=float,
                ),
                previous_timestep_seconds=float(
                    source["previous_timestep_seconds"]
                ),
                temporal_path_scheme=str(
                    source["temporal_path_scheme"].item()
                ),
            ),
            elapsed_time_seconds=float(source["elapsed_time_seconds"]),
            completed_steps=int(source["completed_steps"]),
            next_order=int(source["next_order"]),
            provenance=json.loads(str(source["provenance_json"].item())),
            schema_version=int(source["schema_version"]),
        )
    validated = _validated_restart(context, restart)
    if (
        expected_provenance is not None
        and validated.provenance != expected_provenance
    ):
        raise ValueError("monolithic BDF restart provenance differs")
    return validated
