"""Fully coupled transonic-inner and signed-reservoir steady residual."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import lil_matrix, vstack
from scipy.sparse.linalg import spsolve

from .grid import RadialGrid
from .interface_flux import (
    ConservedInterfaceFlux,
    conserved_interface_flux,
)
from .signed_flux_common_stress import (
    NonKeplerianResidualScales,
    evaluate_nonkeplerian_common_stress_residual,
    positive_edge_reconstruction,
)
from .signed_flux_disk import SignedFluxTransport
from .signed_flux_thermal import SignedThermalClosure
from .time_dae_boundary import remap_zero_torque_thermodynamics
from .transonic_collocation import (
    TransonicSlimParams,
    computational_grid,
    pack_state,
    profile_from_state_vector,
    sonic_residual_jacobian,
    state_bounds,
    transonic_core_jacobian_without_outer_boundary,
    transonic_core_residual_without_outer_boundary,
    unpack_state,
)
from .transonic_local import algebraic_state
from .transonic_potential import PaczynskiWiitaPotential


@dataclass(frozen=True)
class CoupledInnerOuterContext:
    """Fixed physics, scaling, and anchor data for one coupling stage."""

    inner_params: TransonicSlimParams
    outer_grid: RadialGrid
    outer_template: SignedFluxTransport
    outer_closure: SignedThermalClosure
    outer_scales: NonKeplerianResidualScales
    anchor_log_surface_density: float
    anchor_log_temperature: float
    reference_log_surface_density_jump: float
    reference_log_temperature_jump: float
    angular_flux_scale: float
    energy_flux_scale: float
    coupling_fraction: float = 0.0
    sonic_pivot: str = "C1"
    alpha: float = 0.01
    mu_stress: float = 0.0
    stress_factor: float = 1.0
    wall_pattern_omega: float | None = None
    wall_pattern_power_fraction: float = 0.0
    wall_power_weights: np.ndarray | None = None
    interface_stencil_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.coupling_fraction <= 1.0:
            raise ValueError("coupling_fraction must lie in [0,1]")
        if self.sonic_pivot not in {"C1", "C2", "K", "svd"}:
            raise ValueError("sonic_pivot must be frozen to C1, C2, or K")
        if self.angular_flux_scale <= 0.0 or self.energy_flux_scale <= 0.0:
            raise ValueError("interface flux scales must be positive")
        if not 0.0 <= self.wall_pattern_power_fraction <= 1.0:
            raise ValueError("wall_pattern_power_fraction must lie in [0,1]")
        if not 0.0 <= self.interface_stencil_fraction <= 1.0:
            raise ValueError("interface_stencil_fraction must lie in [0,1]")
        if self.wall_pattern_omega is not None and (
            not np.isfinite(self.wall_pattern_omega)
            or self.wall_pattern_omega <= 0.0
        ):
            raise ValueError("wall_pattern_omega must be positive and finite")
        if (
            self.wall_pattern_power_fraction > 0.0
            and self.wall_pattern_omega is None
        ):
            raise ValueError("positive wall pattern power requires pattern omega")
        if self.wall_power_weights is not None:
            weights = np.array(self.wall_power_weights, dtype=float, copy=True)
            if (
                weights.shape != self.outer_grid.centers.shape
                or np.any(~np.isfinite(weights))
                or np.any(weights < 0.0)
                or not np.isclose(
                    np.sum(weights),
                    1.0,
                    rtol=1.0e-12,
                    atol=1.0e-15,
                )
            ):
                raise ValueError("wall_power_weights must be normalized on the grid")
            weights.setflags(write=False)
            object.__setattr__(self, "wall_power_weights", weights)
        expected_radius = self.inner_params.R_out_rg * self.inner_params.r_g
        if not np.isclose(
            self.outer_grid.edges[0],
            expected_radius,
            rtol=2.0e-13,
            atol=0.0,
        ):
            raise ValueError("inner outer node and outer inner edge must coincide")


@dataclass(frozen=True)
class CoupledResidualEvaluation:
    """One evaluated coupled state and its named residual blocks."""

    residual: np.ndarray
    inner_core: np.ndarray
    outer_stress: np.ndarray
    outer_radial: np.ndarray
    outer_energy: np.ndarray
    flux_extraction: np.ndarray
    interface_boundary: np.ndarray
    anchor_boundary: np.ndarray
    continuity_boundary: np.ndarray
    inner_profile: object
    outer_transport: SignedFluxTransport
    outer_energy_profile: object
    interface_flux: ConservedInterfaceFlux
    extracted_inner_flux: ConservedInterfaceFlux
    outer_edge_surface_density: float
    outer_edge_temperature: float
    outer_edge_omega: float


@dataclass(frozen=True)
class CoupledInnerOuterResult:
    """Nonlinear result for one fixed coupling fraction."""

    state: np.ndarray
    evaluation: CoupledResidualEvaluation
    accepted: bool
    nfev: int
    maximum_residual: float
    message: str


@dataclass(frozen=True)
class CoupledRankAudit:
    """Scaled full-system, interface-response, and sonic rank diagnostics."""

    jacobian_shape: tuple[int, int]
    singular_values: np.ndarray
    ranks_by_relative_threshold: dict[str, int]
    condition_estimate: float
    weakest_right_block_norms: dict[str, float]
    weakest_left_block_norms: dict[str, float]
    preboundary_nullity: int
    interface_response: np.ndarray
    interface_response_singular_values: np.ndarray
    interface_response_rank: int
    sonic_singular_values: np.ndarray
    sonic_rank: int
    endpoint_null_vectors: dict[str, dict[str, float]]


def coupled_state_size(context: CoupledInnerOuterContext) -> int:
    ni = context.inner_params.n_nodes
    no = context.outer_grid.centers.size
    return 2 * ni + 3 * no + 4


def _inner_endpoint_state(inner_state, params: TransonicSlimParams):
    """Evaluate only the algebraic state needed at the coupling radius."""

    state = np.asarray(inner_state, dtype=float)
    ni = params.n_nodes
    return algebraic_state(
        np.log(params.R_out_rg * params.r_g),
        float(state[ni - 1]),
        float(state[2 * ni - 1]),
        float(state[-1]),
        params,
    )


def _inner_endpoint_flux(endpoint, params: TransonicSlimParams):
    potential = PaczynskiWiitaPotential(params.M2_g)
    torque = 2.0 * np.pi * endpoint.R**2 * endpoint.W
    bernoulli = (
        0.5 * endpoint.u**2
        + 0.5 * (endpoint.R * endpoint.Omega) ** 2
        + potential.phi(endpoint.R)
        + endpoint.e
        + endpoint.Pi / endpoint.Sigma
    )
    return conserved_interface_flux(
        params.Mdot_g_s,
        endpoint.l,
        torque,
        endpoint.Omega,
        bernoulli,
    )


def _cross_interface_radial_residual(
    endpoint,
    sigma,
    omega,
    outer_grid: RadialGrid,
    energy_profile,
    potential: PaczynskiWiitaPotential,
) -> float:
    """Evaluate radial force in cell zero with the inner endpoint as a ghost."""

    log_radius = np.log(outer_grid.centers)
    extended_radius = np.concatenate(([np.log(endpoint.R)], log_radius))
    radial_velocity = np.asarray(energy_profile.radial_velocity, dtype=float)
    pressure = np.asarray(
        energy_profile.vertically_integrated_pressure, dtype=float
    )
    inertia = 0.5 * np.gradient(
        np.concatenate(([(-endpoint.u) ** 2], radial_velocity**2)),
        extended_radius,
        edge_order=2,
    )[1]
    pressure_gradient = np.gradient(
        np.concatenate(([endpoint.Pi], pressure)),
        extended_radius,
        edge_order=2,
    )[1]
    omega_k = float(potential.omega_k(outer_grid.centers[0]))
    radius = float(outer_grid.centers[0])
    return float(
        (
            inertia
            - radius**2 * (omega[0] ** 2 - omega_k**2)
            + pressure_gradient / sigma[0]
        )
        / (radius**2 * omega_k**2)
    )
def canonical_anchor_inner_residual(
    inner_state,
    params: TransonicSlimParams,
    anchor_log_surface_density: float,
    anchor_log_temperature: float,
    *,
    sonic_pivot: str,
) -> np.ndarray:
    """Close the truncated inner core on the actual canonical interface state."""

    core = transonic_core_residual_without_outer_boundary(
        inner_state,
        params,
        pivot=sonic_pivot,
    )
    endpoint = _inner_endpoint_state(inner_state, params)
    anchor = np.asarray(
        [
            np.log(endpoint.Sigma) - float(anchor_log_surface_density),
            np.log(endpoint.T) - float(anchor_log_temperature),
        ],
        dtype=float,
    )
    return np.concatenate((core, anchor))


def _canonical_anchor_jacobian(
    state,
    params: TransonicSlimParams,
    *,
    sonic_pivot: str,
):
    """Combine the block-local core Jacobian with exact anchor rows."""

    ni = params.n_nodes
    anchor = lil_matrix((2, 2 * ni + 2), dtype=float)
    anchor[0, ni - 1] = -1.0
    anchor[1, 2 * ni - 1] = 1.0
    return vstack(
        (
            transonic_core_jacobian_without_outer_boundary(
                state,
                params,
                pivot=sonic_pivot,
            ),
            anchor.tocsr(),
        ),
        format="csr",
    )


def solve_canonical_anchored_inner(
    initial_state,
    params: TransonicSlimParams,
    anchor_log_surface_density: float,
    anchor_log_temperature: float,
    *,
    sonic_pivot: str,
    tolerance: float = 1.0e-7,
    max_nfev: int = 1000,
):
    """Polish a truncated transonic state with bounded sparse Newton steps."""

    state = np.asarray(initial_state, dtype=float)
    lower, upper = state_bounds(params)
    state = np.clip(state, lower + 1.0e-12, upper - 1.0e-12)
    maximum = np.inf
    for _iteration in range(int(max_nfev)):
        residual = canonical_anchor_inner_residual(
            state,
            params,
            anchor_log_surface_density,
            anchor_log_temperature,
            sonic_pivot=sonic_pivot,
        )
        maximum = float(np.max(np.abs(residual)))
        if maximum <= tolerance:
            break
        jacobian = _canonical_anchor_jacobian(
            state,
            params,
            sonic_pivot=sonic_pivot,
        )
        step_direction = np.asarray(spsolve(jacobian, -residual), dtype=float)
        if np.any(~np.isfinite(step_direction)):
            break
        norm = float(np.linalg.norm(residual))
        accepted = False
        for backtrack in range(20):
            step = 0.5**backtrack
            trial = np.clip(
                state + step * step_direction,
                lower + 1.0e-12,
                upper - 1.0e-12,
            )
            trial_residual = canonical_anchor_inner_residual(
                trial,
                params,
                anchor_log_surface_density,
                anchor_log_temperature,
                sonic_pivot=sonic_pivot,
            )
            if np.linalg.norm(trial_residual) < norm * (1.0 - 1.0e-4 * step):
                state = trial
                accepted = True
                break
        if not accepted:
            break
    residual = canonical_anchor_inner_residual(
        state,
        params,
        anchor_log_surface_density,
        anchor_log_temperature,
        sonic_pivot=sonic_pivot,
    )
    maximum = float(np.max(np.abs(residual)))
    return np.asarray(state, dtype=float), bool(maximum <= tolerance), maximum


def coupled_row_slices(context: CoupledInnerOuterContext) -> dict[str, slice]:
    ni = context.inner_params.n_nodes
    no = context.outer_grid.centers.size
    start = 0
    rows = {}
    for name, width in (
        ("inner_core", 2 * ni),
        ("outer_stress", no),
        ("outer_radial", no),
        ("outer_energy", no),
        ("flux_extraction", 2),
        ("interface_boundary", 2),
    ):
        rows[name] = slice(start, start + width)
        start += width
    if start != coupled_state_size(context):
        raise RuntimeError("coupled row count is not square")
    return rows


def pack_coupled_state(
    inner_state,
    outer_surface_density,
    outer_temperature,
    outer_omega,
    interface_angular_flux: float,
    interface_energy_flux: float,
    context: CoupledInnerOuterContext,
) -> np.ndarray:
    ni = context.inner_params.n_nodes
    no = context.outer_grid.centers.size
    inner = np.asarray(inner_state, dtype=float)
    sigma = np.asarray(outer_surface_density, dtype=float)
    temperature = np.asarray(outer_temperature, dtype=float)
    omega = np.asarray(outer_omega, dtype=float)
    if inner.shape != (2 * ni + 2,):
        raise ValueError("inner state has the wrong size")
    if any(value.shape != (no,) for value in (sigma, temperature, omega)):
        raise ValueError("outer state arrays have the wrong size")
    return np.concatenate(
        (
            inner,
            np.log(sigma),
            np.log(temperature),
            np.log(omega),
            np.asarray(
                [
                    interface_angular_flux / context.angular_flux_scale,
                    interface_energy_flux / context.energy_flux_scale,
                ],
                dtype=float,
            ),
        )
    )


def unpack_coupled_state(state, context: CoupledInnerOuterContext):
    ni = context.inner_params.n_nodes
    no = context.outer_grid.centers.size
    state = np.asarray(state, dtype=float)
    if state.shape != (coupled_state_size(context),):
        raise ValueError("coupled state has the wrong size")
    inner_end = 2 * ni + 2
    sigma_end = inner_end + no
    temperature_end = sigma_end + no
    omega_end = temperature_end + no
    return (
        state[:inner_end],
        np.exp(state[inner_end:sigma_end]),
        np.exp(state[sigma_end:temperature_end]),
        np.exp(state[temperature_end:omega_end]),
        float(state[-2] * context.angular_flux_scale),
        float(state[-1] * context.energy_flux_scale),
    )


def interpolate_coupled_state_components(
    state,
    source_context: CoupledInnerOuterContext,
    target_inner_params: TransonicSlimParams,
    target_outer_grid: RadialGrid,
    *,
    outer_remap: str = "log_primitives",
):
    """Interpolate a complete coupled root onto new inner and outer meshes."""

    if outer_remap not in {"log_primitives", "zero_torque"}:
        raise ValueError("outer_remap must be log_primitives or zero_torque")

    (
        inner_state,
        sigma,
        temperature,
        omega,
        interface_angular,
        interface_energy,
    ) = unpack_coupled_state(state, source_context)
    logu, logT, logR_son, lambda0, source_logR = unpack_state(
        inner_state,
        source_context.inner_params,
    )
    target_logR = computational_grid(target_inner_params, logR_son)
    target_inner_state = pack_state(
        np.interp(target_logR, source_logR, logu),
        np.interp(target_logR, source_logR, logT),
        logR_son,
        lambda0,
    )

    source_log_outer_radius = np.log(source_context.outer_grid.centers)
    target_log_outer_radius = np.log(target_outer_grid.centers)

    def positive_interpolate(values):
        return np.exp(
            np.interp(
                target_log_outer_radius,
                source_log_outer_radius,
                np.log(np.asarray(values, dtype=float)),
            )
        )

    if outer_remap == "zero_torque":
        if source_context.mu_stress != 0.0:
            raise ValueError(
                "zero_torque remap currently requires total-pressure stress"
            )
        remap = remap_zero_torque_thermodynamics(
            source_context.outer_grid,
            target_outer_grid,
            sigma,
            temperature,
            source_context.inner_params.M2_g,
            alpha=source_context.alpha,
            closure=source_context.outer_closure,
            stress_factor=source_context.stress_factor,
        )
        target_sigma = remap.surface_density
        target_temperature = remap.temperature
    else:
        target_sigma = positive_interpolate(sigma)
        target_temperature = positive_interpolate(temperature)

    return (
        target_inner_state,
        target_sigma,
        target_temperature,
        positive_interpolate(omega),
        interface_angular,
        interface_energy,
    )


def interpolate_coupled_state_across_interface(
    state,
    source_context: CoupledInnerOuterContext,
    target_inner_params: TransonicSlimParams,
    target_outer_grid: RadialGrid,
):
    """Remap a coupled composite when its numerical interface moves."""

    (
        inner_state,
        _sigma,
        _temperature,
        _omega,
        interface_angular,
        interface_energy,
    ) = unpack_coupled_state(state, source_context)
    evaluation = evaluate_coupled_inner_outer_residual(
        state,
        source_context,
    )
    inner = evaluation.inner_profile
    outer = evaluation.outer_energy_profile
    outer_transport = evaluation.outer_transport
    source_radius = np.concatenate(
        (inner.R, source_context.outer_grid.centers)
    )

    def interpolate_composite(inner_values, outer_values, target_radius):
        values = np.concatenate(
            (
                np.asarray(inner_values, dtype=float),
                np.asarray(outer_values, dtype=float),
            )
        )
        return np.exp(
            np.interp(
                np.log(target_radius),
                np.log(source_radius),
                np.log(values),
            )
        )

    _logu, _logT, logR_son, lambda0, _source_logR = unpack_state(
        inner_state,
        source_context.inner_params,
    )
    target_logR = computational_grid(target_inner_params, logR_son)
    target_inner_radius = np.exp(target_logR)
    outer_inflow_speed = np.maximum(
        -outer.radial_velocity,
        1.0e-300,
    )
    target_inner_state = pack_state(
        np.log(
            interpolate_composite(
                inner.u,
                outer_inflow_speed,
                target_inner_radius,
            )
        ),
        np.log(
            interpolate_composite(
                inner.T,
                outer.temperature,
                target_inner_radius,
            )
        ),
        logR_son,
        lambda0,
    )
    target_outer_radius = target_outer_grid.centers
    return (
        target_inner_state,
        interpolate_composite(
            inner.Sigma,
            outer_transport.surface_density,
            target_outer_radius,
        ),
        interpolate_composite(
            inner.T,
            outer.temperature,
            target_outer_radius,
        ),
        interpolate_composite(
            inner.Omega,
            outer_transport.omega,
            target_outer_radius,
        ),
        interface_angular,
        interface_energy,
    )


def evaluate_coupled_inner_outer_residual(
    state,
    context: CoupledInnerOuterContext,
    *,
    include_inner_profile: bool = True,
) -> CoupledResidualEvaluation:
    (
        inner_state,
        sigma,
        temperature,
        omega,
        interface_angular,
        interface_energy,
    ) = unpack_coupled_state(state, context)
    params = context.inner_params
    inner_core = transonic_core_residual_without_outer_boundary(
        inner_state,
        params,
        pivot=context.sonic_pivot,
    )
    endpoint = _inner_endpoint_state(inner_state, params)
    extracted = _inner_endpoint_flux(endpoint, params)
    inner_profile = (
        profile_from_state_vector(inner_state, params)
        if include_inner_profile
        else None
    )
    interface_flux = ConservedInterfaceFlux(
        mdot=params.Mdot_g_s,
        angular_momentum=interface_angular,
        total_energy=interface_energy,
    )
    outer_stress, outer_radial, outer_energy, transport, energy_profile = (
        evaluate_nonkeplerian_common_stress_residual(
            context.outer_grid,
            context.outer_template,
            sigma,
            temperature,
            omega,
            params.M2_g,
            alpha=context.alpha,
            closure=context.outer_closure,
            scales=context.outer_scales,
            prescribed_inner_flux=interface_flux,
            radial_support_fraction=1.0,
            mu_stress=context.mu_stress,
            stress_factor=context.stress_factor,
            wall_pattern_omega=context.wall_pattern_omega,
            wall_pattern_power_fraction=(
                context.wall_pattern_power_fraction
            ),
            wall_power_weights=context.wall_power_weights,
        )
    )
    if context.interface_stencil_fraction > 0.0:
        potential = PaczynskiWiitaPotential(params.M2_g)
        cross_radial = _cross_interface_radial_residual(
            endpoint,
            sigma,
            omega,
            context.outer_grid,
            energy_profile,
            potential,
        )
        outer_radial = np.array(outer_radial, copy=True)
        fraction = context.interface_stencil_fraction
        outer_radial[0] = (
            (1.0 - fraction) * outer_radial[0] + fraction * cross_radial
        )
    flux_rows = np.asarray(
        [
            (interface_angular - extracted.angular_momentum)
            / context.angular_flux_scale,
            (interface_energy - extracted.total_energy)
            / context.energy_flux_scale,
        ],
        dtype=float,
    )
    sigma_edge = float(positive_edge_reconstruction(context.outer_grid, sigma)[0])
    temperature_edge = float(
        positive_edge_reconstruction(context.outer_grid, temperature)[0]
    )
    omega_edge = float(positive_edge_reconstruction(context.outer_grid, omega)[0])
    inner_log_sigma = float(np.log(endpoint.Sigma))
    inner_log_temperature = float(np.log(endpoint.T))
    anchor = np.asarray(
        [
            inner_log_sigma - context.anchor_log_surface_density,
            inner_log_temperature - context.anchor_log_temperature,
        ],
        dtype=float,
    )
    continuity = np.asarray(
        [
            np.log(sigma_edge) - inner_log_sigma,
            np.log(temperature_edge) - inner_log_temperature,
        ],
        dtype=float,
    )
    interface_rows = (
        continuity
        - (1.0 - context.coupling_fraction)
        * np.asarray(
            [
                context.reference_log_surface_density_jump,
                context.reference_log_temperature_jump,
            ],
            dtype=float,
        )
    )
    residual = np.concatenate(
        (
            inner_core,
            outer_stress,
            outer_radial,
            outer_energy,
            flux_rows,
            interface_rows,
        )
    )
    if residual.shape != (coupled_state_size(context),):
        raise RuntimeError("coupled residual is not square")
    return CoupledResidualEvaluation(
        residual=residual,
        inner_core=inner_core,
        outer_stress=outer_stress,
        outer_radial=outer_radial,
        outer_energy=outer_energy,
        flux_extraction=flux_rows,
        interface_boundary=interface_rows,
        anchor_boundary=anchor,
        continuity_boundary=continuity,
        inner_profile=inner_profile,
        outer_transport=transport,
        outer_energy_profile=energy_profile,
        interface_flux=interface_flux,
        extracted_inner_flux=extracted,
        outer_edge_surface_density=sigma_edge,
        outer_edge_temperature=temperature_edge,
        outer_edge_omega=omega_edge,
    )


def coupled_jacobian_sparsity(context: CoupledInnerOuterContext):
    ni = context.inner_params.n_nodes
    no = context.outer_grid.centers.size
    size = coupled_state_size(context)
    rows = coupled_row_slices(context)
    pattern = lil_matrix((size, size), dtype=int)
    inner_size = 2 * ni + 2
    sigma_start = inner_size
    temperature_start = sigma_start + no
    omega_start = temperature_start + no
    j_col, e_col = size - 2, size - 1

    row = rows["inner_core"].start
    for idx in range(ni - 1):
        for col in (
            idx,
            idx + 1,
            ni + idx,
            ni + idx + 1,
            2 * ni,
            2 * ni + 1,
        ):
            pattern[row : row + 2, col] = 1
        row += 2
    for col in (0, ni, 2 * ni, 2 * ni + 1):
        pattern[row : row + 2, col] = 1

    for block_name in ("outer_stress", "outer_radial", "outer_energy"):
        radius = 0 if block_name == "outer_stress" else 2
        for cell in range(no):
            residual_row = rows[block_name].start + cell
            for neighbor in range(max(0, cell - radius), min(no, cell + radius + 1)):
                pattern[residual_row, sigma_start + neighbor] = 1
                pattern[residual_row, temperature_start + neighbor] = 1
                pattern[residual_row, omega_start + neighbor] = 1
            pattern[residual_row, j_col] = 1
            pattern[residual_row, e_col] = 1

    if context.interface_stencil_fraction > 0.0:
        first_radial = rows["outer_radial"].start
        for column in (ni - 1, 2 * ni - 1, 2 * ni + 1):
            pattern[first_radial, column] = 1

    for residual_row in range(rows["flux_extraction"].start, size):
        for col in (ni - 1, 2 * ni - 1, 2 * ni, 2 * ni + 1, j_col, e_col):
            pattern[residual_row, col] = 1
        for cell in range(min(3, no)):
            pattern[residual_row, sigma_start + cell] = 1
            pattern[residual_row, temperature_start + cell] = 1
            pattern[residual_row, omega_start + cell] = 1
    return pattern.tocsr()


def coupled_state_bounds(
    context: CoupledInnerOuterContext,
    seed_state,
) -> tuple[np.ndarray, np.ndarray]:
    ni = context.inner_params.n_nodes
    no = context.outer_grid.centers.size
    seed_state = np.asarray(seed_state, dtype=float)
    inner_lower, inner_upper = state_bounds(context.inner_params)
    _, sigma, _temperature, _omega, _j, _e = unpack_coupled_state(
        seed_state,
        context,
    )
    potential = PaczynskiWiitaPotential(context.inner_params.M2_g)
    omega_k = potential.omega_k(context.outer_grid.centers)
    lower_temperature, upper_temperature = context.outer_closure.temperature_bounds
    lower = np.concatenate(
        (
            inner_lower,
            np.log(sigma) - np.log(1.0e4),
            np.full(no, np.log(lower_temperature)),
            np.log(0.5 * omega_k),
            np.asarray([-10.0, -10.0]),
        )
    )
    upper = np.concatenate(
        (
            inner_upper,
            np.log(sigma) + np.log(1.0e4),
            np.full(no, np.log(upper_temperature)),
            np.log(1.2 * omega_k),
            np.asarray([10.0, 10.0]),
        )
    )
    if lower.shape != (2 * ni + 3 * no + 4,) or upper.shape != lower.shape:
        raise RuntimeError("coupled bounds have the wrong size")
    return lower, upper


def colored_coupled_jacobian(
    state,
    context: CoupledInnerOuterContext,
    *,
    relative_step: float = 1.0e-6,
):
    """Evaluate the declared sparse Jacobian with deterministic graph coloring."""

    if relative_step <= 0.0:
        raise ValueError("relative_step must be positive")
    state = np.asarray(state, dtype=float)
    pattern = coupled_jacobian_sparsity(context).tocsc()
    lower, upper = coupled_state_bounds(context, state)
    base = evaluate_coupled_inner_outer_residual(
        state,
        context,
        include_inner_profile=False,
    ).residual

    color_rows: list[set[int]] = []
    colors = np.empty(state.size, dtype=int)
    column_rows = []
    for column in range(state.size):
        rows = pattern.indices[pattern.indptr[column] : pattern.indptr[column + 1]]
        row_set = set(int(value) for value in rows)
        column_rows.append(rows)
        for color, occupied in enumerate(color_rows):
            if row_set.isdisjoint(occupied):
                colors[column] = color
                occupied.update(row_set)
                break
        else:
            colors[column] = len(color_rows)
            color_rows.append(set(row_set))

    jacobian = lil_matrix(pattern.shape, dtype=float)
    for color in range(len(color_rows)):
        columns = np.flatnonzero(colors == color)
        trial = np.array(state, copy=True)
        signed_steps = {}
        for column in columns:
            step = relative_step * max(1.0, abs(float(state[column])))
            forward_room = upper[column] - state[column]
            backward_room = state[column] - lower[column]
            if forward_room >= step:
                signed_step = step
            elif backward_room >= step:
                signed_step = -step
            else:
                signed_step = 0.5 * (forward_room if forward_room >= backward_room else -backward_room)
            if signed_step == 0.0:
                continue
            trial[column] += signed_step
            signed_steps[int(column)] = float(signed_step)
        changed = evaluate_coupled_inner_outer_residual(
            trial,
            context,
            include_inner_profile=False,
        ).residual
        difference = changed - base
        for column, signed_step in signed_steps.items():
            rows = column_rows[column]
            jacobian[rows, column] = (difference[rows] / signed_step)[:, None]
    return jacobian.tocsr()


def solve_coupled_inner_outer_steady(
    initial_state,
    context: CoupledInnerOuterContext,
    *,
    tolerance: float = 1.0e-7,
    max_nfev: int = 1000,
) -> CoupledInnerOuterResult:
    """Solve one fixed coupling fraction with bounded sparse Newton steps."""

    state = np.asarray(initial_state, dtype=float)
    lower, upper = coupled_state_bounds(context, state)
    state = np.clip(state, lower + 1.0e-12, upper - 1.0e-12)
    nfev = 0
    message = "maximum Newton iterations reached"
    for _iteration in range(int(max_nfev)):
        trial_evaluation = evaluate_coupled_inner_outer_residual(
            state,
            context,
            include_inner_profile=False,
        )
        nfev += 1
        residual = trial_evaluation.residual
        maximum = float(np.max(np.abs(residual)))
        if maximum <= tolerance:
            message = "residual tolerance reached"
            break
        jacobian = colored_coupled_jacobian(state, context)
        step_direction = np.asarray(spsolve(jacobian, -residual), dtype=float)
        if np.any(~np.isfinite(step_direction)):
            message = "non-finite Newton direction"
            break
        norm = float(np.linalg.norm(residual))
        accepted_step = False
        for backtrack in range(20):
            step = 0.5**backtrack
            candidate = np.clip(
                state + step * step_direction,
                lower + 1.0e-12,
                upper - 1.0e-12,
            )
            candidate_residual = evaluate_coupled_inner_outer_residual(
                candidate,
                context,
                include_inner_profile=False,
            ).residual
            nfev += 1
            if np.linalg.norm(candidate_residual) < norm * (1.0 - 1.0e-4 * step):
                state = candidate
                accepted_step = True
                break
        if not accepted_step:
            message = "Newton line search failed"
            break
    evaluation = evaluate_coupled_inner_outer_residual(state, context)
    maximum = float(np.max(np.abs(evaluation.residual)))
    return CoupledInnerOuterResult(
        state=np.asarray(state, dtype=float),
        evaluation=evaluation,
        accepted=bool(maximum <= tolerance),
        nfev=int(nfev),
        maximum_residual=maximum,
        message=message,
    )


def dense_coupled_jacobian(
    state,
    context: CoupledInnerOuterContext,
    *,
    relative_step: float = 3.0e-6,
) -> np.ndarray:
    """Return a bounded central finite-difference Jacobian for rank audits."""

    if relative_step <= 0.0:
        raise ValueError("relative_step must be positive")
    state = np.asarray(state, dtype=float)
    lower, upper = coupled_state_bounds(context, state)
    base = evaluate_coupled_inner_outer_residual(
        state,
        context,
        include_inner_profile=False,
    ).residual
    jacobian = np.empty((base.size, state.size), dtype=float)
    for column, value in enumerate(state):
        step = relative_step * max(1.0, abs(float(value)))
        step = min(step, 0.2 * max(upper[column] - lower[column], 1.0e-300))
        if value - step >= lower[column] and value + step <= upper[column]:
            plus = np.array(state, copy=True)
            minus = np.array(state, copy=True)
            plus[column] += step
            minus[column] -= step
            jacobian[:, column] = (
                evaluate_coupled_inner_outer_residual(
                    plus,
                    context,
                    include_inner_profile=False,
                ).residual
                - evaluate_coupled_inner_outer_residual(
                    minus,
                    context,
                    include_inner_profile=False,
                ).residual
            ) / (2.0 * step)
        elif value + step <= upper[column]:
            plus = np.array(state, copy=True)
            plus[column] += step
            jacobian[:, column] = (
                evaluate_coupled_inner_outer_residual(
                    plus,
                    context,
                    include_inner_profile=False,
                ).residual
                - base
            ) / step
        elif value - step >= lower[column]:
            minus = np.array(state, copy=True)
            minus[column] -= step
            jacobian[:, column] = (
                base
                - evaluate_coupled_inner_outer_residual(
                    minus,
                    context,
                    include_inner_profile=False,
                ).residual
            ) / step
        else:
            jacobian[:, column] = 0.0
    return jacobian


def _state_block_slices(context: CoupledInnerOuterContext) -> dict[str, slice]:
    ni = context.inner_params.n_nodes
    no = context.outer_grid.centers.size
    inner_end = 2 * ni + 2
    return {
        "inner": slice(0, inner_end),
        "outer_sigma": slice(inner_end, inner_end + no),
        "outer_temperature": slice(inner_end + no, inner_end + 2 * no),
        "outer_omega": slice(inner_end + 2 * no, inner_end + 3 * no),
        "interface_fluxes": slice(inner_end + 3 * no, inner_end + 3 * no + 2),
    }


def _row_block_slices(context: CoupledInnerOuterContext) -> dict[str, slice]:
    return coupled_row_slices(context)


def _normalized_block_norms(vector, blocks: dict[str, slice]) -> dict[str, float]:
    vector = np.asarray(vector, dtype=float)
    total = max(float(np.linalg.norm(vector)), 1.0e-300)
    return {
        name: float(np.linalg.norm(vector[block]) / total)
        for name, block in blocks.items()
    }


def audit_coupled_rank(
    state,
    context: CoupledInnerOuterContext,
    *,
    relative_step: float = 3.0e-6,
) -> CoupledRankAudit:
    """Audit full rank and the two-dimensional physical interface response."""

    state = np.asarray(state, dtype=float)
    jacobian = dense_coupled_jacobian(
        state,
        context,
        relative_step=relative_step,
    )
    left, singular, right_h = np.linalg.svd(jacobian, full_matrices=False)
    largest = max(float(singular[0]), 1.0e-300)
    thresholds = (1.0e-8, 1.0e-10, 1.0e-12)
    ranks = {
        f"{threshold:.0e}": int(np.sum(singular / largest > threshold))
        for threshold in thresholds
    }
    row_slices = coupled_row_slices(context)
    state_slices = _state_block_slices(context)

    boundary_rows = row_slices["interface_boundary"]
    preboundary = np.delete(
        jacobian,
        np.arange(boundary_rows.start, boundary_rows.stop),
        axis=0,
    )
    _pre_left, pre_singular, pre_right_h = np.linalg.svd(
        preboundary,
        full_matrices=True,
    )
    pre_largest = max(float(pre_singular[0]), 1.0e-300)
    pre_rank = int(np.sum(pre_singular / pre_largest > 1.0e-10))
    nullity = int(jacobian.shape[1] - pre_rank)
    null_basis = pre_right_h[-2:, :].T
    interface_response = jacobian[boundary_rows, :] @ null_basis
    response_singular = np.linalg.svd(
        interface_response,
        compute_uv=False,
    )
    response_rank = int(
        np.sum(
            response_singular
            / max(float(response_singular[0]), 1.0e-300)
            > 1.0e-10
        )
    )

    inner_state, *_rest = unpack_coupled_state(state, context)
    sonic_components = (
        "D",
        "K" if context.sonic_pivot in {"K", "svd"} else context.sonic_pivot,
    )
    sonic_jacobian = sonic_residual_jacobian(
        inner_state,
        context.inner_params,
        components=sonic_components,
        rel_step=min(relative_step, 1.0e-6),
    )
    sonic_singular = np.linalg.svd(sonic_jacobian, compute_uv=False)
    sonic_rank = int(
        np.sum(
            sonic_singular
            / max(float(sonic_singular[0]), 1.0e-300)
            > 1.0e-10
        )
    )

    endpoint_vectors = {}
    outer_radial = row_slices["outer_radial"]
    no = context.outer_grid.centers.size
    for name, removed_row in (
        ("inner_radial", outer_radial.start),
        ("outer_radial", outer_radial.stop - 1),
    ):
        reduced = np.delete(jacobian, removed_row, axis=0)
        _u, _s, vh = np.linalg.svd(reduced, full_matrices=True)
        null_vector = vh[-1]
        norms = _normalized_block_norms(null_vector, state_slices)
        inner_end = 2 * context.inner_params.n_nodes + 2
        endpoint_columns = np.r_[
            np.arange(inner_end, inner_end + min(3, no)),
            np.arange(inner_end + no, inner_end + no + min(3, no)),
            np.arange(inner_end + 2 * no, inner_end + 2 * no + min(3, no)),
        ]
        outer_columns = np.r_[
            np.arange(inner_end + max(0, no - 3), inner_end + no),
            np.arange(inner_end + 2 * no - min(3, no), inner_end + 2 * no),
            np.arange(inner_end + 3 * no - min(3, no), inner_end + 3 * no),
        ]
        norms["first_three_outer_cells"] = float(
            np.linalg.norm(null_vector[endpoint_columns])
            / max(np.linalg.norm(null_vector), 1.0e-300)
        )
        norms["last_three_outer_cells"] = float(
            np.linalg.norm(null_vector[outer_columns])
            / max(np.linalg.norm(null_vector), 1.0e-300)
        )
        endpoint_vectors[name] = norms

    return CoupledRankAudit(
        jacobian_shape=jacobian.shape,
        singular_values=np.asarray(singular, dtype=float),
        ranks_by_relative_threshold=ranks,
        condition_estimate=float(largest / max(float(singular[-1]), 1.0e-300)),
        weakest_right_block_norms=_normalized_block_norms(
            right_h[-1],
            state_slices,
        ),
        weakest_left_block_norms=_normalized_block_norms(
            left[:, -1],
            _row_block_slices(context),
        ),
        preboundary_nullity=nullity,
        interface_response=np.asarray(interface_response, dtype=float),
        interface_response_singular_values=np.asarray(
            response_singular,
            dtype=float,
        ),
        interface_response_rank=response_rank,
        sonic_singular_values=np.asarray(sonic_singular, dtype=float),
        sonic_rank=sonic_rank,
        endpoint_null_vectors=endpoint_vectors,
    )
