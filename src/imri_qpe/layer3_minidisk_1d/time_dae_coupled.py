"""Fully coupled inner-transonic/outer-flux-primary backward-Euler DAE."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .coupled_inner_outer import CoupledInnerOuterContext
from .interface_flux import ConservedInterfaceFlux, transonic_profile_interface_flux
from .signed_flux_common_stress import positive_edge_reconstruction
from .time_dae_outer import (
    FluxPrimaryOuterDAEProfile,
    OuterRadialBoundaryState,
    OuterDAELedgerAudit,
    audit_outer_dae_backward_euler_ledgers,
    evaluate_flux_primary_outer_dae_profile,
    pack_outer_primitives,
    unpack_outer_primitives,
)
from .transonic_collocation import (
    profile_from_state_vector,
    state_bounds,
    transonic_core_residual_without_outer_boundary,
)


@dataclass(frozen=True)
class CoupledTimeDAEContext:
    """Fixed physics and variable scales for one no-tide open DAE solve."""

    base: CoupledInnerOuterContext
    mass_flux_scale: float
    angular_flux_scale: float
    energy_flux_scale: float

    def __post_init__(self) -> None:
        if self.base.coupling_fraction != 1.0:
            raise ValueError("time-dependent coupling requires full primitive coupling")
        if self.base.wall_pattern_power_fraction != 0.0:
            raise ValueError("the no-tide prototype cannot include wall power")
        for name in (
            "mass_flux_scale",
            "angular_flux_scale",
            "energy_flux_scale",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True)
class CoupledTimeDAEEvaluation:
    """Named residual blocks for one fully coupled backward-Euler state."""

    residual: np.ndarray
    inner_core: np.ndarray
    outer_mass: np.ndarray
    outer_angular_momentum: np.ndarray
    outer_energy: np.ndarray
    outer_stress: np.ndarray
    outer_radial: np.ndarray
    interface_continuity: np.ndarray
    interface_flux_extraction: np.ndarray
    open_edge: float
    inner_profile: object
    extracted_inner_flux: ConservedInterfaceFlux
    outer: FluxPrimaryOuterDAEProfile
    interface_energy_flux: float


@dataclass(frozen=True)
class CoupledTimeDAEStepResult:
    """One implicit step of the coupled no-tide open system."""

    state: np.ndarray
    evaluation: CoupledTimeDAEEvaluation
    ledger: OuterDAELedgerAudit
    accepted: bool
    maximum_residual: float
    nfev: int
    message: str
    iterations: int = 0
    linear_solver: str = "dense_least_squares"


@dataclass(frozen=True)
class CoupledTimeDAERestart:
    """Validated full DAE restart payload."""

    state: np.ndarray
    elapsed_time: float
    step_number: int


def coupled_time_dae_state_size(context: CoupledTimeDAEContext) -> int:
    """Return the exact selected count ``2 Ni + 5 No + 5``."""

    ni = context.base.inner_params.n_nodes
    no = context.base.outer_grid.centers.size
    return 2 * ni + 5 * no + 5


def coupled_time_dae_row_slices(context: CoupledTimeDAEContext) -> dict[str, slice]:
    """Return the declared residual partition for rank and defect audits."""

    ni = context.base.inner_params.n_nodes
    no = context.base.outer_grid.centers.size
    start = 0
    rows: dict[str, slice] = {}
    for name, width in (
        ("inner_core", 2 * ni),
        ("outer_mass", no),
        ("outer_angular_momentum", no),
        ("outer_energy", no),
        ("outer_stress", no),
        ("outer_radial", no),
        ("interface_continuity", 2),
        ("interface_flux_extraction", 2),
        ("open_edge", 1),
    ):
        rows[name] = slice(start, start + width)
        start += width
    if start != coupled_time_dae_state_size(context):
        raise RuntimeError("coupled time-DAE residual is not square")
    return rows


def coupled_time_dae_jacobian_sparsity(context: CoupledTimeDAEContext):
    """Return a conservative block-local sparsity pattern for backward Euler."""

    ni = context.base.inner_params.n_nodes
    no = context.base.outer_grid.centers.size
    size = coupled_time_dae_state_size(context)
    rows = coupled_time_dae_row_slices(context)
    pattern = lil_matrix((size, size), dtype=int)
    inner_end = 2 * ni + 2
    sigma_start = inner_end
    temperature_start = sigma_start + no
    omega_start = temperature_start + no
    mdot_start = omega_start + no
    angular_start = mdot_start + no + 1
    energy_col = size - 1

    row = rows["inner_core"].start
    for cell in range(ni - 1):
        columns = (
            cell,
            cell + 1,
            ni + cell,
            ni + cell + 1,
            2 * ni,
            2 * ni + 1,
            mdot_start,
        )
        for column in columns:
            pattern[row : row + 2, column] = 1
        row += 2
    for column in (0, ni, 2 * ni, 2 * ni + 1, mdot_start):
        pattern[row : row + 2, column] = 1

    for cell in range(no):
        q_local = (
            sigma_start + cell,
            temperature_start + cell,
            omega_start + cell,
        )
        faces = (cell, cell + 1)
        mass_row = rows["outer_mass"].start + cell
        pattern[mass_row, sigma_start + cell] = 1
        for face in faces:
            pattern[mass_row, mdot_start + face] = 1

        angular_row = rows["outer_angular_momentum"].start + cell
        pattern[angular_row, sigma_start + cell] = 1
        pattern[angular_row, omega_start + cell] = 1
        for face in faces:
            pattern[angular_row, angular_start + face] = 1

        neighbor_cells = range(max(0, cell - 2), min(no, cell + 3))
        neighbor_faces = range(max(0, cell - 2), min(no + 1, cell + 4))
        for block in ("outer_energy", "outer_radial"):
            target_row = rows[block].start + cell
            for neighbor in neighbor_cells:
                pattern[target_row, sigma_start + neighbor] = 1
                pattern[target_row, temperature_start + neighbor] = 1
                pattern[target_row, omega_start + neighbor] = 1
            for face in neighbor_faces:
                pattern[target_row, mdot_start + face] = 1
                if block == "outer_energy":
                    pattern[target_row, angular_start + face] = 1
        if cell == 0:
            pattern[rows["outer_energy"].start, energy_col] = 1
            if context.base.interface_stencil_fraction > 0.0:
                for column in (ni - 1, 2 * ni - 1, 2 * ni + 1, mdot_start):
                    pattern[rows["outer_energy"].start, column] = 1
                    pattern[rows["outer_radial"].start, column] = 1

        stress_row = rows["outer_stress"].start + cell
        for column in q_local:
            pattern[stress_row, column] = 1
        for face in faces:
            pattern[stress_row, mdot_start + face] = 1
            pattern[stress_row, angular_start + face] = 1

    continuity = rows["interface_continuity"]
    pattern[continuity, :inner_end] = 1
    pattern[continuity, mdot_start] = 1
    for cell in range(min(2, no)):
        pattern[continuity.start, sigma_start + cell] = 1
        pattern[continuity.start + 1, temperature_start + cell] = 1

    extraction = rows["interface_flux_extraction"]
    pattern[extraction, :inner_end] = 1
    pattern[extraction, mdot_start] = 1
    pattern[extraction.start, angular_start] = 1
    pattern[extraction.start + 1, energy_col] = 1

    edge = rows["open_edge"].start
    pattern[edge, mdot_start + no] = 1
    pattern[edge, angular_start + no] = 1
    for cell in range(max(0, no - 2), no):
        pattern[edge, omega_start + cell] = 1
    return pattern.tocsr()


def colored_coupled_time_dae_jacobian(
    state,
    old_state,
    dt: float,
    context: CoupledTimeDAEContext,
    *,
    relative_step: float = 1.0e-6,
):
    """Evaluate the declared Jacobian with deterministic graph coloring."""

    if relative_step <= 0.0:
        raise ValueError("relative_step must be positive")
    state = np.asarray(state, dtype=float)
    pattern = coupled_time_dae_jacobian_sparsity(context).tocsc()
    lower, upper = _step_bounds(old_state, context)
    color_rows: list[set[int]] = []
    colors = np.empty(state.size, dtype=int)
    column_rows = []
    for column in range(state.size):
        support = pattern.indices[
            pattern.indptr[column] : pattern.indptr[column + 1]
        ]
        row_set = set(int(value) for value in support)
        column_rows.append(support)
        for color, occupied in enumerate(color_rows):
            if row_set.isdisjoint(occupied):
                colors[column] = color
                occupied.update(row_set)
                break
        else:
            colors[column] = len(color_rows)
            color_rows.append(set(row_set))
    jacobian = lil_matrix(pattern.shape, dtype=float)
    evaluations = 0
    for color in range(len(color_rows)):
        columns = np.flatnonzero(colors == color)
        plus = np.array(state, copy=True)
        minus = np.array(state, copy=True)
        denominators: dict[int, float] = {}
        for column in columns:
            step = relative_step * max(1.0, abs(float(state[column])))
            forward_room = upper[column] - state[column]
            backward_room = state[column] - lower[column]
            if forward_room >= step and backward_room >= step:
                plus[column] += step
                minus[column] -= step
                denominator = 2.0 * step
            elif forward_room >= step:
                plus[column] += step
                denominator = step
            elif backward_room >= step:
                minus[column] -= step
                denominator = step
            else:
                available = max(forward_room, backward_room)
                if available <= 0.0:
                    continue
                step = 0.5 * available
                if forward_room >= backward_room:
                    plus[column] += step
                else:
                    minus[column] -= step
                denominator = step
            if denominator == 0.0:
                continue
            denominators[int(column)] = float(denominator)
        changed_plus = evaluate_coupled_time_dae_backward_euler_residual(
            plus, old_state, dt, context
        ).residual
        changed_minus = evaluate_coupled_time_dae_backward_euler_residual(
            minus, old_state, dt, context
        ).residual
        evaluations += 2
        difference = changed_plus - changed_minus
        for column, denominator in denominators.items():
            support = column_rows[column]
            jacobian[support, column] = (
                difference[support] / denominator
            )[:, None]
    return jacobian.tocsr(), evaluations, len(color_rows)


def pack_coupled_time_dae_state(
    inner_state,
    outer_surface_density,
    outer_temperature,
    outer_omega,
    mdot_faces,
    angular_flux_faces,
    interface_energy_flux: float,
    context: CoupledTimeDAEContext,
) -> np.ndarray:
    """Pack algebraic and differential variables with linear flux scaling."""

    inner = np.asarray(inner_state, dtype=float)
    ni = context.base.inner_params.n_nodes
    no = context.base.outer_grid.centers.size
    mdot = np.asarray(mdot_faces, dtype=float)
    angular = np.asarray(angular_flux_faces, dtype=float)
    if inner.shape != (2 * ni + 2,):
        raise ValueError("inner state has the wrong size")
    if mdot.shape != (no + 1,) or angular.shape != (no + 1,):
        raise ValueError("face flux arrays have the wrong size")
    if not np.isfinite(interface_energy_flux):
        raise ValueError("interface energy flux must be finite")
    outer = pack_outer_primitives(
        outer_surface_density, outer_temperature, outer_omega
    )
    return np.concatenate(
        (
            inner,
            outer,
            mdot / context.mass_flux_scale,
            angular / context.angular_flux_scale,
            [float(interface_energy_flux) / context.energy_flux_scale],
        )
    )


def unpack_coupled_time_dae_state(state, context: CoupledTimeDAEContext):
    """Unpack the selected flux-primary state without duplicating interface Mdot."""

    state = np.asarray(state, dtype=float)
    if state.shape != (coupled_time_dae_state_size(context),):
        raise ValueError("coupled time-DAE state has the wrong size")
    ni = context.base.inner_params.n_nodes
    no = context.base.outer_grid.centers.size
    inner_end = 2 * ni + 2
    outer_end = inner_end + 3 * no
    mdot_end = outer_end + no + 1
    angular_end = mdot_end + no + 1
    return (
        np.asarray(state[:inner_end], dtype=float),
        np.asarray(state[inner_end:outer_end], dtype=float),
        context.mass_flux_scale * np.asarray(state[outer_end:mdot_end], dtype=float),
        context.angular_flux_scale
        * np.asarray(state[mdot_end:angular_end], dtype=float),
        float(context.energy_flux_scale * state[-1]),
    )


def save_coupled_time_dae_restart(
    path,
    state,
    context: CoupledTimeDAEContext,
    *,
    elapsed_time: float,
    step_number: int,
) -> None:
    """Write the complete state plus mesh and scale compatibility metadata."""

    state = np.asarray(state, dtype=float)
    if state.shape != (coupled_time_dae_state_size(context),):
        raise ValueError("restart state has the wrong size")
    if not np.isfinite(elapsed_time) or elapsed_time < 0.0:
        raise ValueError("elapsed_time must be finite and non-negative")
    if int(step_number) != step_number or step_number < 0:
        raise ValueError("step_number must be a non-negative integer")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        state=state,
        elapsed_time=float(elapsed_time),
        step_number=int(step_number),
        inner_nodes=context.base.inner_params.n_nodes,
        outer_cells=context.base.outer_grid.centers.size,
        interface_radius=context.base.outer_grid.edges[0],
        interface_stencil_fraction=context.base.interface_stencil_fraction,
        mass_flux_scale=context.mass_flux_scale,
        angular_flux_scale=context.angular_flux_scale,
        energy_flux_scale=context.energy_flux_scale,
    )


def load_coupled_time_dae_restart(
    path,
    context: CoupledTimeDAEContext,
) -> CoupledTimeDAERestart:
    """Load a restart only when its mesh, interface, and scales still match."""

    with np.load(Path(path), allow_pickle=False) as data:
        state = np.asarray(data["state"], dtype=float)
        inner_nodes = int(data["inner_nodes"])
        outer_cells = int(data["outer_cells"])
        interface_radius = float(data["interface_radius"])
        interface_stencil_fraction = float(data["interface_stencil_fraction"])
        mass_scale = float(data["mass_flux_scale"])
        angular_scale = float(data["angular_flux_scale"])
        energy_scale = float(data["energy_flux_scale"])
        elapsed_time = float(data["elapsed_time"])
        step_number = int(data["step_number"])
    expected = (
        context.base.inner_params.n_nodes,
        context.base.outer_grid.centers.size,
    )
    if (inner_nodes, outer_cells) != expected:
        raise ValueError("restart mesh does not match the current context")
    comparisons = (
        (interface_radius, context.base.outer_grid.edges[0], "interface"),
        (
            interface_stencil_fraction,
            context.base.interface_stencil_fraction,
            "interface stencil fraction",
        ),
        (mass_scale, context.mass_flux_scale, "mass flux scale"),
        (angular_scale, context.angular_flux_scale, "angular flux scale"),
        (energy_scale, context.energy_flux_scale, "energy flux scale"),
    )
    for actual, wanted, name in comparisons:
        if not np.isclose(actual, wanted, rtol=1.0e-13, atol=0.0):
            raise ValueError(f"restart {name} does not match the current context")
    if state.shape != (coupled_time_dae_state_size(context),):
        raise ValueError("restart state has the wrong size")
    return CoupledTimeDAERestart(
        state=state,
        elapsed_time=elapsed_time,
        step_number=step_number,
    )


def _outer_profile(
    outer_state,
    mdot_faces,
    angular_flux_faces,
    interface_energy_flux: float,
    context: CoupledTimeDAEContext,
    inner_profile=None,
) -> FluxPrimaryOuterDAEProfile:
    base = context.base
    sigma, temperature, omega = unpack_outer_primitives(
        outer_state, base.outer_grid
    )
    template = base.outer_template
    inner_boundary = None
    if inner_profile is not None:
        inner_boundary = OuterRadialBoundaryState(
            radius=float(inner_profile.R[-1]),
            integrated_pressure=float(inner_profile.Pi[-1]),
            radial_velocity=-float(inner_profile.u[-1]),
            blend_fraction=context.base.interface_stencil_fraction,
        )
    return evaluate_flux_primary_outer_dae_profile(
        base.outer_grid,
        sigma,
        temperature,
        omega,
        mdot_faces,
        angular_flux_faces,
        base.inner_params.M2_g,
        alpha=base.alpha,
        closure=base.outer_closure,
        stress_factor=base.stress_factor,
        source_mass_rate_cells=template.source_mass_rate_cells,
        source_angular_rate_cells=template.source_angular_rate_cells,
        source_energy_rate_cells=template.source_total_energy_rate_cells,
        include_radiative_cooling=True,
        inner_energy_flux=interface_energy_flux,
        inner_radial_boundary=inner_boundary,
    )


def _row_scales(
    old_profile,
    dt: float,
    context: CoupledTimeDAEContext,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    source_mass = np.abs(old_profile.source_mass_rate_cells)
    source_angular = np.abs(old_profile.source_angular_rate_cells)
    source_energy = np.abs(old_profile.source_energy_rate_cells)
    radiative = old_profile.radiative_loss_rate_cells
    mass = np.maximum(
        dt
        * (
            np.abs(old_profile.mdot_faces[:-1])
            + np.abs(old_profile.mdot_faces[1:])
            + source_mass
        ),
        1.0e-12 * np.abs(old_profile.mass_cells),
    )
    angular = np.maximum(
        dt
        * (
            np.abs(old_profile.angular_flux_faces[:-1])
            + np.abs(old_profile.angular_flux_faces[1:])
            + source_angular
        ),
        1.0e-12 * np.abs(old_profile.angular_momentum_cells),
    )
    flux_divergence = (
        old_profile.energy_flux_faces[1:] - old_profile.energy_flux_faces[:-1]
    )
    radial_work = old_profile.energy_rhs - (
        flux_divergence
        + old_profile.source_energy_rate_cells
        - old_profile.radiative_loss_rate_cells
    )
    energy = np.maximum(
        dt
        * (
            np.abs(old_profile.energy_flux_faces[:-1])
            + np.abs(old_profile.energy_flux_faces[1:])
            + np.abs(radial_work)
            + source_energy
            + radiative
        ),
        1.0e-12
        * (
            np.abs(old_profile.energy_cells)
            + np.abs(old_profile.mass_cells * old_profile.enthalpy)
        ),
    )
    torque = max(
        float(np.max(np.abs(old_profile.torque_faces))),
        context.angular_flux_scale,
        1.0,
    )
    return mass, angular, energy, torque


def evaluate_coupled_time_dae_backward_euler_residual(
    new_state,
    old_state,
    dt: float,
    context: CoupledTimeDAEContext,
) -> CoupledTimeDAEEvaluation:
    """Evaluate the square fully coupled backward-Euler residual."""

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    old_inner, old_outer, old_mdot, old_angular, old_energy = (
        unpack_coupled_time_dae_state(old_state, context)
    )
    inner, outer_state, mdot, angular, interface_energy = (
        unpack_coupled_time_dae_state(new_state, context)
    )
    if mdot[0] <= 0.0:
        raise ValueError("the shared inner/interface mass flux must stay positive")
    base = context.base
    old_params = replace(base.inner_params, Mdot_g_s=float(old_mdot[0]))
    old_inner_profile = profile_from_state_vector(old_inner, old_params)
    trial_params = replace(base.inner_params, Mdot_g_s=float(mdot[0]))
    inner_core = transonic_core_residual_without_outer_boundary(
        inner, trial_params, pivot=base.sonic_pivot
    )
    inner_profile = profile_from_state_vector(inner, trial_params)
    extracted = transonic_profile_interface_flux(
        inner_profile, trial_params.M2_g, mdot[0], -1
    )
    old_outer_evaluation = _outer_profile(
        old_outer,
        old_mdot,
        old_angular,
        old_energy,
        context,
        inner_profile=old_inner_profile,
    )
    outer = _outer_profile(
        outer_state,
        mdot,
        angular,
        interface_energy,
        context,
        inner_profile=inner_profile,
    )
    old_profile = old_outer_evaluation.profile
    profile = outer.profile
    mass_scale, angular_scale, energy_scale, torque_scale = _row_scales(
        old_profile, dt, context
    )
    outer_mass = (
        profile.mass_cells - old_profile.mass_cells - dt * profile.mass_rhs
    ) / mass_scale
    outer_angular = (
        profile.angular_momentum_cells
        - old_profile.angular_momentum_cells
        - dt * profile.angular_rhs
    ) / angular_scale
    temporal_vertical_work = 0.5 * (
        profile.mass_cells * profile.enthalpy
        + old_profile.mass_cells * old_profile.enthalpy
    ) * np.log(profile.H / old_profile.H)
    outer_energy = (
        profile.energy_cells
        - old_profile.energy_cells
        + temporal_vertical_work
        - dt * profile.energy_rhs
    ) / energy_scale

    sigma, temperature, _omega = unpack_outer_primitives(
        outer_state, base.outer_grid
    )
    sigma_edge = float(positive_edge_reconstruction(base.outer_grid, sigma)[0])
    temperature_edge = float(
        positive_edge_reconstruction(base.outer_grid, temperature)[0]
    )
    continuity = np.asarray(
        [
            np.log(sigma_edge / inner_profile.Sigma[-1]),
            np.log(temperature_edge / inner_profile.T[-1]),
        ],
        dtype=float,
    )
    extraction = np.asarray(
        [
            (angular[0] - extracted.angular_momentum)
            / context.angular_flux_scale,
            (interface_energy - extracted.total_energy)
            / context.energy_flux_scale,
        ],
        dtype=float,
    )
    edge = float(profile.torque_faces[-1] / torque_scale)
    residual = np.concatenate(
        (
            inner_core,
            outer_mass,
            outer_angular,
            outer_energy,
            outer.stress_residual,
            profile.radial_residual,
            continuity,
            extraction,
            [edge],
        )
    )
    if residual.shape != (coupled_time_dae_state_size(context),):
        raise RuntimeError("coupled time-DAE residual is not square")
    return CoupledTimeDAEEvaluation(
        residual=np.asarray(residual, dtype=float),
        inner_core=np.asarray(inner_core, dtype=float),
        outer_mass=np.asarray(outer_mass, dtype=float),
        outer_angular_momentum=np.asarray(outer_angular, dtype=float),
        outer_energy=np.asarray(outer_energy, dtype=float),
        outer_stress=np.asarray(outer.stress_residual, dtype=float),
        outer_radial=np.asarray(profile.radial_residual, dtype=float),
        interface_continuity=continuity,
        interface_flux_extraction=extraction,
        open_edge=edge,
        inner_profile=inner_profile,
        extracted_inner_flux=extracted,
        outer=outer,
        interface_energy_flux=float(interface_energy),
    )


def _step_bounds(state, context: CoupledTimeDAEContext):
    inner, outer, mdot, angular, interface_energy = unpack_coupled_time_dae_state(
        state, context
    )
    params = replace(context.base.inner_params, Mdot_g_s=float(mdot[0]))
    inner_lower, inner_upper = state_bounds(params)
    no = context.base.outer_grid.centers.size
    outer_lower = outer - np.concatenate(
        (np.full(no, 4.0), np.full(no, 2.0), np.full(no, 1.0))
    )
    outer_upper = outer + np.concatenate(
        (np.full(no, 4.0), np.full(no, 2.0), np.full(no, 1.0))
    )
    mdot_scaled = mdot / context.mass_flux_scale
    mdot_extent = max(float(np.max(np.abs(mdot_scaled))), 1.0)
    mdot_lower = np.full(no + 1, -5.0 * mdot_extent)
    mdot_upper = np.full(no + 1, 5.0 * mdot_extent)
    mdot_lower[0] = max(1.0e-8, 1.0e-4 * mdot_scaled[0])
    angular_extent = max(
        float(np.max(np.abs(angular / context.angular_flux_scale))), 1.0
    )
    energy_scaled = interface_energy / context.energy_flux_scale
    lower = np.concatenate(
        (
            inner_lower,
            outer_lower,
            mdot_lower,
            np.full(no + 1, -10.0 * angular_extent),
            [energy_scaled - 10.0 * max(abs(energy_scaled), 1.0)],
        )
    )
    upper = np.concatenate(
        (
            inner_upper,
            outer_upper,
            mdot_upper,
            np.full(no + 1, 10.0 * angular_extent),
            [energy_scaled + 10.0 * max(abs(energy_scaled), 1.0)],
        )
    )
    return lower, upper


def advance_coupled_time_dae_backward_euler(
    state,
    dt: float,
    context: CoupledTimeDAEContext,
    *,
    jacobian_mode: str = "colored_central",
    tolerance: float = 1.0e-7,
    ledger_tolerance: float = 1.0e-7,
    max_nfev: int = 500,
) -> CoupledTimeDAEStepResult:
    """Advance one directly coupled no-tide open step."""

    state = np.asarray(state, dtype=float)
    if jacobian_mode not in {"colored_central", "scipy_forward"}:
        raise ValueError("jacobian_mode must be colored_central or scipy_forward")
    lower, upper = _step_bounds(state, context)

    def residual(trial):
        return evaluate_coupled_time_dae_backward_euler_residual(
            trial, state, dt, context
        ).residual

    if jacobian_mode == "colored_central":
        jacobian = lambda trial: colored_coupled_time_dae_jacobian(
            trial, state, dt, context
        )[0]
        jacobian_options = {"jac": jacobian}
    else:
        jacobian_options = {
            "jac_sparsity": coupled_time_dae_jacobian_sparsity(context)
        }

    def solve(initial):
        return least_squares(
            residual,
            np.clip(initial, lower + 1.0e-12, upper - 1.0e-12),
            bounds=(lower, upper),
            tr_solver="lsmr",
            tr_options={
                "atol": 1.0e-12,
                "btol": 1.0e-12,
                "maxiter": 1000,
                "regularize": True,
            },
            xtol=1.0e-13,
            ftol=1.0e-13,
            gtol=1.0e-13,
            max_nfev=int(max_nfev),
            **jacobian_options,
        )

    result = solve(state)
    evaluation = evaluate_coupled_time_dae_backward_euler_residual(
        result.x, state, dt, context
    )
    old_inner, old_outer, old_mdot, old_angular, old_energy = (
        unpack_coupled_time_dae_state(state, context)
    )
    old_params = replace(
        context.base.inner_params, Mdot_g_s=float(old_mdot[0])
    )
    old_inner_profile = profile_from_state_vector(old_inner, old_params)
    old_profile = _outer_profile(
        old_outer,
        old_mdot,
        old_angular,
        old_energy,
        context,
        inner_profile=old_inner_profile,
    ).profile
    ledger = audit_outer_dae_backward_euler_ledgers(
        old_profile, evaluation.outer.profile, dt
    )
    maximum = float(np.max(np.abs(evaluation.residual)))
    _inner, _outer, mdot, _angular, _energy = unpack_coupled_time_dae_state(
        result.x, context
    )
    accepted = bool(
        result.success
        and maximum <= tolerance
        and mdot[0] > 0.0
        and mdot[-1] <= 0.0
        and ledger.relative_mass_defect <= ledger_tolerance
        and ledger.relative_angular_momentum_defect <= ledger_tolerance
        and ledger.relative_energy_defect <= ledger_tolerance
    )
    return CoupledTimeDAEStepResult(
        state=np.asarray(result.x, dtype=float),
        evaluation=evaluation,
        ledger=ledger,
        accepted=accepted,
        maximum_residual=maximum,
        nfev=int(result.nfev),
        message=str(result.message),
        iterations=int(result.njev or 0),
        linear_solver=f"{jacobian_mode}_sparse_trust_region_lsmr",
    )
