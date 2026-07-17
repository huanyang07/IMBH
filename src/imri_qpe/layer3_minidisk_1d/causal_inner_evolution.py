"""Adaptive repeated stepping for the causal five-field increment DAE."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from imri_qpe.constants import C

from .causal_inner_dae import causal_five_field_dae_count
from .causal_inner_dae_system import (
    CausalFiveFieldDAEContext,
    causal_five_field_colored_central_jacobian,
    causal_five_field_dae_jacobian_color_groups,
    causal_five_field_dae_jacobian_sparsity,
    causal_five_field_dae_scaling,
    causal_five_field_equilibrated_sparse_solve,
    evaluate_causal_five_field_dae,
    evaluate_causal_five_field_increment_backward_euler,
    unpack_causal_five_field_state,
)


@dataclass(frozen=True)
class CausalFiveFieldAdaptiveStepConfig:
    """Deterministic reject/halve/grow policy for short causal startup."""

    minimum_dt: float
    maximum_dt: float
    maximum_scaled_primitive_change: float = 5.0e-4
    maximum_scaled_total_change: float = 1.0e-3
    shrink_factor: float = 0.5
    growth_factor: float = 1.5
    maximum_retries: int = 6
    easy_iterations: int = 3
    residual_tolerance: float = 1.0e-8
    algebraic_residual_tolerance: float = 1.0e-10
    conservation_tolerance: float = 1.0e-10
    finite_difference_step: float = 2.0e-6
    maximum_newton_iterations: int = 12

    def validated(self) -> CausalFiveFieldAdaptiveStepConfig:
        positive = (
            self.minimum_dt,
            self.maximum_dt,
            self.maximum_scaled_primitive_change,
            self.maximum_scaled_total_change,
            self.shrink_factor,
            self.growth_factor,
            self.residual_tolerance,
            self.algebraic_residual_tolerance,
            self.conservation_tolerance,
            self.finite_difference_step,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("causal adaptive values must be positive and finite")
        if self.minimum_dt > self.maximum_dt:
            raise ValueError("minimum_dt must not exceed maximum_dt")
        if not self.shrink_factor < 1.0:
            raise ValueError("shrink_factor must be below one")
        if not self.growth_factor > 1.0:
            raise ValueError("growth_factor must exceed one")
        if (
            int(self.maximum_retries) != self.maximum_retries
            or self.maximum_retries < 0
        ):
            raise ValueError("maximum_retries must be a non-negative integer")
        if int(self.easy_iterations) != self.easy_iterations:
            raise ValueError("easy_iterations must be an integer")
        if self.easy_iterations < 1:
            raise ValueError("easy_iterations must be positive")
        if (
            int(self.maximum_newton_iterations)
            != self.maximum_newton_iterations
            or self.maximum_newton_iterations < 1
        ):
            raise ValueError("maximum_newton_iterations must be positive")
        return self


@dataclass(frozen=True)
class CausalFiveFieldStepResult:
    """One accepted or rejected sparse backward-Euler step."""

    state_vector: np.ndarray
    physical_increment: np.ndarray
    accepted: bool
    timestep_seconds: float
    maximum_scaled_residual: float
    maximum_scaled_algebraic_residual: float
    maximum_scaled_primitive_change: float
    maximum_scaled_total_change: float
    conservation_telescoping_relative_defect: float
    component_conservation_defects: tuple[float, ...]
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
class CausalFiveFieldAdaptiveAttempt:
    """One adaptive trial and its physical-change controller."""

    timestep_seconds: float
    accepted: bool
    maximum_scaled_residual: float
    maximum_scaled_primitive_change: float
    maximum_scaled_total_change: float
    iterations: int
    message: str


@dataclass(frozen=True)
class CausalFiveFieldAdaptiveStepResult:
    """One adaptive accepted state, including rejected retries."""

    state_vector: np.ndarray
    physical_increment: np.ndarray
    accepted: bool
    dt_used: float
    dt_next: float
    step: CausalFiveFieldStepResult
    attempts: tuple[CausalFiveFieldAdaptiveAttempt, ...]
    message: str


@dataclass(frozen=True)
class CausalFiveFieldAdaptiveRestart:
    """Complete deterministic continuation payload for the causal DAE."""

    state_vector: np.ndarray
    previous_physical_increment: np.ndarray
    elapsed_time: float
    dt_next: float
    previous_dt: float
    accepted_steps: int
    rejected_attempts: int
    provenance: dict
    schema_version: int = 1


@dataclass(frozen=True)
class CausalFiveFieldPhysicalStepLedger:
    """Physical five-field balance for one backward-Euler step."""

    conserved_storage_change: np.ndarray
    vertical_storage_change: np.ndarray
    boundary_transport: np.ndarray
    endogenous_source: np.ndarray
    prescribed_stream_source: np.ndarray
    closure_defect: np.ndarray


def _ledger_defect(
    new_state,
    evaluation,
) -> tuple[float, tuple[float, ...]]:
    telescoped = np.asarray(
        [
            np.sum(evaluation.conservation_rows[:, field])
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
    expected = boundary + np.sum(cell_terms, axis=0)
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
    components = (telescoped - expected) / scale
    return (
        float(np.max(np.abs(components))),
        tuple(float(value) for value in components),
    )


def causal_five_field_physical_step_ledger(
    context: CausalFiveFieldDAEContext,
    old_vector: np.ndarray,
    physical_increment: np.ndarray,
    timestep_seconds: float,
) -> CausalFiveFieldPhysicalStepLedger:
    """Return a cancellation-aware physical ledger for one solved step.

    All entries use the physical mass-equivalent units of the conserved
    Killing chart. The balance is

    ``storage + vertical + boundary - endogenous - stream = 0``.
    """

    context = context.validated()
    n_cells = int(context.grid.centers.size)
    count = causal_five_field_dae_count(n_cells)
    old_values = np.asarray(old_vector, dtype=float)
    increment = np.asarray(physical_increment, dtype=float)
    if (
        old_values.shape != (count.total_unknowns,)
        or increment.shape != old_values.shape
        or np.any(~np.isfinite(old_values))
        or np.any(~np.isfinite(increment))
    ):
        raise ValueError("causal physical-ledger vectors are invalid")
    timestep = float(timestep_seconds)
    if not np.isfinite(timestep) or timestep <= 0.0:
        raise ValueError("causal physical-ledger timestep must be positive")

    evaluation = evaluate_causal_five_field_increment_backward_euler(
        increment,
        context,
        old_vector=old_values,
        timestep_seconds=timestep,
        temporal_height_scheme="path_integrated",
    )
    new_state = unpack_causal_five_field_state(
        old_values + increment,
        n_cells,
    )
    conserved_increment = increment[: 5 * n_cells].reshape(
        n_cells,
        5,
    )
    conserved_storage = np.sum(
        context.grid.cell_measures[:, None] * conserved_increment,
        axis=0,
    )
    vertical_storage = np.zeros(5, dtype=float)
    vertical_storage[:4] = (
        C
        * timestep
        * np.sum(evaluation.temporal_vertical_storage, axis=0)
    )
    boundary_transport = (
        C
        * timestep
        * (
            new_state.weighted_face_fluxes_over_c[-1]
            - new_state.weighted_face_fluxes_over_c[0]
        )
    )
    total_source = (
        C
        * timestep
        * np.sum(evaluation.integrated_sources_per_ct, axis=0)
    )
    prescribed_stream = np.zeros(5, dtype=float)
    if context.stream_sources is not None:
        prescribed_stream[:4] = (
            timestep
            * np.sum(context.stream_sources.matrix, axis=0)
        )
    endogenous_source = total_source - prescribed_stream
    closure_defect = (
        conserved_storage
        + vertical_storage
        + boundary_transport
        - endogenous_source
        - prescribed_stream
    )
    return CausalFiveFieldPhysicalStepLedger(
        conserved_storage_change=conserved_storage,
        vertical_storage_change=vertical_storage,
        boundary_transport=boundary_transport,
        endogenous_source=endogenous_source,
        prescribed_stream_source=prescribed_stream,
        closure_defect=closure_defect,
    )


def advance_causal_five_field_increment_backward_euler(
    context: CausalFiveFieldDAEContext,
    old_vector: np.ndarray,
    timestep_seconds: float,
    initial_physical_increment: np.ndarray,
    config: CausalFiveFieldAdaptiveStepConfig,
) -> CausalFiveFieldStepResult:
    """Advance one sparse increment-primary backward-Euler step."""

    context = context.validated()
    config = config.validated()
    n_cells = int(context.grid.centers.size)
    count = causal_five_field_dae_count(n_cells)
    old_values = np.asarray(old_vector, dtype=float)
    predictor = np.asarray(initial_physical_increment, dtype=float)
    if (
        old_values.shape != (count.total_unknowns,)
        or predictor.shape != old_values.shape
        or np.any(~np.isfinite(old_values))
        or np.any(~np.isfinite(predictor))
    ):
        raise ValueError("causal repeated-step vectors are invalid")
    timestep = float(timestep_seconds)
    if not np.isfinite(timestep) or timestep <= 0.0:
        raise ValueError("causal repeated-step timestep must be positive")
    old_state = unpack_causal_five_field_state(old_values, n_cells)
    stationary = evaluate_causal_five_field_dae(old_values, context)
    scaling = causal_five_field_dae_scaling(old_state, stationary)
    initial = predictor / scaling.column_scales
    pattern = causal_five_field_dae_jacobian_sparsity(n_cells)
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
            evaluate_causal_five_field_increment_backward_euler(
                physical_increment,
                context,
                old_vector=old_values,
                timestep_seconds=timestep,
                temporal_height_scheme="path_integrated",
            ).residual
            / scaling.row_scales
        )

    bound = 1.25 * config.maximum_scaled_total_change
    state = np.asarray(initial, dtype=float)
    values = residual(state)
    success = False
    message = "maximum Newton iterations reached"
    linear_residuals: list[float] = []
    iterations = 0
    for iteration in range(config.maximum_newton_iterations + 1):
        iterations = iteration
        maximum_residual = float(np.max(np.abs(values)))
        if maximum_residual <= config.residual_tolerance:
            success = True
            message = "residual gate passed"
            break
        if iteration == config.maximum_newton_iterations:
            break
        matrix = causal_five_field_colored_central_jacobian(
            residual,
            state,
            pattern,
            finite_difference_step=config.finite_difference_step,
        )
        jacobian_evaluations += 1
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
            message = "bound-aware line search failed"
            break

    physical_increment = scaling.column_scales * state
    new_vector = old_values + physical_increment
    new_state = unpack_causal_five_field_state(new_vector, n_cells)
    evaluation = evaluate_causal_five_field_increment_backward_euler(
        physical_increment,
        context,
        old_vector=old_values,
        timestep_seconds=timestep,
        temporal_height_scheme="path_integrated",
    )
    scaled_residual = evaluation.residual / scaling.row_scales
    n_differential = 5 * n_cells
    maximum_scaled_residual = float(
        np.max(np.abs(scaled_residual))
    )
    maximum_algebraic_residual = float(
        np.max(np.abs(scaled_residual[n_differential:]))
    )
    maximum_primitive_change = float(
        np.max(
            np.abs(
                state[n_differential : 2 * n_differential]
            )
        )
    )
    maximum_total_change = float(np.max(np.abs(state)))
    ledger_defect, component_defects = _ledger_defect(
        new_state,
        evaluation,
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
        and ledger_defect <= config.conservation_tolerance
        and minimum_optical_depth > 1.0
        and evaluation.outer_boundary_choked
        == stationary.outer_boundary_choked
    )
    if success and not accepted:
        message = "root exceeds one or more repeated-step acceptance gates"
    return CausalFiveFieldStepResult(
        state_vector=new_vector if accepted else old_values,
        physical_increment=(
            physical_increment
            if accepted
            else np.zeros_like(physical_increment)
        ),
        accepted=accepted,
        timestep_seconds=timestep,
        maximum_scaled_residual=maximum_scaled_residual,
        maximum_scaled_algebraic_residual=maximum_algebraic_residual,
        maximum_scaled_primitive_change=maximum_primitive_change,
        maximum_scaled_total_change=maximum_total_change,
        conservation_telescoping_relative_defect=ledger_defect,
        component_conservation_defects=component_defects,
        minimum_scattering_optical_depth=minimum_optical_depth,
        outer_boundary_choked_before=stationary.outer_boundary_choked,
        outer_boundary_choked_after=evaluation.outer_boundary_choked,
        iterations=iterations,
        function_evaluations=function_evaluations,
        jacobian_evaluations=jacobian_evaluations,
        maximum_linear_residual=(
            float(max(linear_residuals))
            if linear_residuals
            else 0.0
        ),
        jacobian_nonzeros=int(pattern.nnz),
        jacobian_color_count=color_count,
        message=message,
    )


def advance_causal_five_field_adaptive_backward_euler(
    context: CausalFiveFieldDAEContext,
    old_vector: np.ndarray,
    dt: float,
    previous_physical_increment: np.ndarray,
    previous_dt: float,
    config: CausalFiveFieldAdaptiveStepConfig,
) -> CausalFiveFieldAdaptiveStepResult:
    """Retry one source-on step with deterministic timestep control."""

    config = config.validated()
    trial_dt = float(np.clip(dt, config.minimum_dt, config.maximum_dt))
    previous_dt = float(previous_dt)
    if not np.isfinite(previous_dt) or previous_dt <= 0.0:
        raise ValueError("previous_dt must be positive and finite")
    attempts: list[CausalFiveFieldAdaptiveAttempt] = []
    last_step: CausalFiveFieldStepResult | None = None
    for _retry in range(config.maximum_retries + 1):
        predictor = (
            np.asarray(previous_physical_increment, dtype=float)
            * (trial_dt / previous_dt)
        )
        step = advance_causal_five_field_increment_backward_euler(
            context,
            old_vector,
            trial_dt,
            predictor,
            config,
        )
        last_step = step
        attempts.append(
            CausalFiveFieldAdaptiveAttempt(
                timestep_seconds=trial_dt,
                accepted=step.accepted,
                maximum_scaled_residual=(
                    step.maximum_scaled_residual
                ),
                maximum_scaled_primitive_change=(
                    step.maximum_scaled_primitive_change
                ),
                maximum_scaled_total_change=(
                    step.maximum_scaled_total_change
                ),
                iterations=step.iterations,
                message=step.message,
            )
        )
        if step.accepted:
            easy = bool(
                step.iterations <= config.easy_iterations
                and step.maximum_scaled_primitive_change
                <= 0.5 * config.maximum_scaled_primitive_change
                and step.maximum_scaled_total_change
                <= 0.5 * config.maximum_scaled_total_change
            )
            dt_next = (
                trial_dt * config.growth_factor
                if easy
                else trial_dt
            )
            return CausalFiveFieldAdaptiveStepResult(
                state_vector=step.state_vector,
                physical_increment=step.physical_increment,
                accepted=True,
                dt_used=trial_dt,
                dt_next=float(min(dt_next, config.maximum_dt)),
                step=step,
                attempts=tuple(attempts),
                message="accepted",
            )
        next_dt = trial_dt * config.shrink_factor
        if next_dt < config.minimum_dt:
            break
        trial_dt = next_dt
    assert last_step is not None
    return CausalFiveFieldAdaptiveStepResult(
        state_vector=np.asarray(old_vector, dtype=float),
        physical_increment=np.zeros_like(old_vector, dtype=float),
        accepted=False,
        dt_used=0.0,
        dt_next=float(max(config.minimum_dt, trial_dt)),
        step=last_step,
        attempts=tuple(attempts),
        message="adaptive retries exhausted without an accepted state",
    )


def causal_five_field_h_over_r_profile(
    context: CausalFiveFieldDAEContext,
    vector: np.ndarray,
) -> np.ndarray:
    """Return the proper half-thickness ratio at every cell center."""

    context = context.validated()
    n_cells = int(context.grid.centers.size)
    state = unpack_causal_five_field_state(vector, n_cells)
    sigma = np.exp(state.primitives[:, 0])
    temperature = np.exp(state.primitives[:, 3])
    half_thickness = np.asarray(
        [
            context.vertical_frequency.eos(float(radius))
            .from_surface_density_temperature(
                float(surface_density),
                float(cell_temperature),
            )
            .proper_half_thickness
            for radius, surface_density, cell_temperature in zip(
                context.grid.centers,
                sigma,
                temperature,
                strict=True,
            )
        ],
        dtype=float,
    )
    return half_thickness / context.grid.centers


def causal_five_field_state_summary(
    context: CausalFiveFieldDAEContext,
    vector: np.ndarray,
) -> dict:
    """Return compact physical diagnostics for one accepted causal state."""

    context = context.validated()
    n_cells = int(context.grid.centers.size)
    state = unpack_causal_five_field_state(vector, n_cells)
    sigma = np.exp(state.primitives[:, 0])
    temperature = np.exp(state.primitives[:, 3])
    h_over_r = causal_five_field_h_over_r_profile(context, vector)
    integrated = np.sum(
        context.grid.cell_measures[:, None] * state.conserved,
        axis=0,
    )
    face_rates = C * state.weighted_face_fluxes_over_c
    return {
        "integrated_conserved": [
            float(value) for value in integrated
        ],
        "inner_face_rates": [
            float(value) for value in face_rates[0]
        ],
        "outer_face_rates": [
            float(value) for value in face_rates[-1]
        ],
        "minimum_surface_density_g_cm2": float(np.min(sigma)),
        "maximum_surface_density_g_cm2": float(np.max(sigma)),
        "minimum_temperature_k": float(np.min(temperature)),
        "maximum_temperature_k": float(np.max(temperature)),
        "maximum_h_over_r": float(np.max(h_over_r)),
    }


def causal_five_field_loading_time(
    context: CausalFiveFieldDAEContext,
    vector: np.ndarray,
) -> float:
    """Return initial rest mass divided by the exact stream supply."""

    if context.stream_sources is None:
        raise ValueError("loading time requires an enabled stream")
    summary = causal_five_field_state_summary(context, vector)
    disk_mass = float(summary["integrated_conserved"][0])
    supply = float(np.sum(context.stream_sources.rest_mass))
    if disk_mass <= 0.0 or supply <= 0.0:
        raise ValueError("loading time requires positive mass and supply")
    return disk_mass / supply


def _restart_hash(
    context: CausalFiveFieldDAEContext,
    state_vector: np.ndarray,
    previous_increment: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for values in (
        context.grid.edges,
        state_vector,
        previous_increment,
    ):
        array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def save_causal_five_field_adaptive_restart(
    path: str | Path,
    context: CausalFiveFieldDAEContext,
    restart: CausalFiveFieldAdaptiveRestart,
) -> None:
    """Store all state and controller data needed for exact continuation."""

    context = context.validated()
    count = causal_five_field_dae_count(context.grid.centers.size)
    state = np.asarray(restart.state_vector, dtype=float)
    increment = np.asarray(
        restart.previous_physical_increment,
        dtype=float,
    )
    if (
        state.shape != (count.total_unknowns,)
        or increment.shape != state.shape
        or np.any(~np.isfinite(state))
        or np.any(~np.isfinite(increment))
    ):
        raise ValueError("causal restart vectors are invalid")
    if (
        not np.isfinite(restart.elapsed_time)
        or restart.elapsed_time < 0.0
        or not np.isfinite(restart.dt_next)
        or restart.dt_next <= 0.0
        or not np.isfinite(restart.previous_dt)
        or restart.previous_dt <= 0.0
    ):
        raise ValueError("causal restart times are invalid")
    if restart.accepted_steps < 0 or restart.rejected_attempts < 0:
        raise ValueError("causal restart counters cannot be negative")
    destination = Path(path)
    if destination.suffix != ".npz":
        raise ValueError("causal restart path must end in .npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    provenance = json.dumps(
        restart.provenance,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    checksum = _restart_hash(context, state, increment)
    np.savez_compressed(
        destination,
        schema_version=np.asarray(restart.schema_version, dtype=np.int64),
        grid_edges=context.grid.edges,
        state_vector=state,
        previous_physical_increment=increment,
        state_controller_sha256=np.asarray(checksum),
        elapsed_time=np.asarray(restart.elapsed_time),
        dt_next=np.asarray(restart.dt_next),
        previous_dt=np.asarray(restart.previous_dt),
        accepted_steps=np.asarray(restart.accepted_steps, dtype=np.int64),
        rejected_attempts=np.asarray(
            restart.rejected_attempts,
            dtype=np.int64,
        ),
        provenance_json=np.asarray(provenance),
    )


def load_causal_five_field_adaptive_restart(
    path: str | Path,
    context: CausalFiveFieldDAEContext,
) -> CausalFiveFieldAdaptiveRestart:
    """Load and verify one complete causal adaptive restart."""

    context = context.validated()
    with np.load(Path(path), allow_pickle=False) as data:
        edges = np.asarray(data["grid_edges"], dtype=float)
        if not np.array_equal(edges, context.grid.edges):
            raise ValueError("causal restart grid does not match")
        state = np.asarray(data["state_vector"], dtype=float)
        increment = np.asarray(
            data["previous_physical_increment"],
            dtype=float,
        )
        checksum = str(data["state_controller_sha256"].item())
        if checksum != _restart_hash(context, state, increment):
            raise ValueError("causal restart checksum mismatch")
        restart = CausalFiveFieldAdaptiveRestart(
            state_vector=state,
            previous_physical_increment=increment,
            elapsed_time=float(data["elapsed_time"]),
            dt_next=float(data["dt_next"]),
            previous_dt=float(data["previous_dt"]),
            accepted_steps=int(data["accepted_steps"]),
            rejected_attempts=int(data["rejected_attempts"]),
            provenance=json.loads(str(data["provenance_json"].item())),
            schema_version=int(data["schema_version"]),
        )
    if restart.schema_version != 1:
        raise ValueError("unsupported causal restart schema version")
    causal_five_field_dae_count(context.grid.centers.size)
    return restart
