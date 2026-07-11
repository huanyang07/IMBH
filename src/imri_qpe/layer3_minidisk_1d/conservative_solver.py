"""Ordinary-grid solver for unified conservative slim-disk transport.

The unknown fields are ``(logu, logT, F, j, epsilon)`` plus the sonic radius,
where ``F`` is inward mass flux, ``j`` is inward angular-momentum flux, and
``epsilon`` is inward total-energy flux in normalized units.  Each interval
contains radial momentum, mass, angular momentum, energy conservation, and a
vertical-work compatibility row.

This module is intentionally independent of the legacy algebraic stream
angular offset.  Legacy checkpoints are accepted only as initial guesses.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from imri_qpe.constants import C

from .conservative_transport import (
    ConservativeIntervalTransport,
    ConservativeNodeState,
    ConservativeScales,
    PhysicalTransportClosure,
    carried_transport,
    conservative_source_terms,
    default_conservative_scales,
    integrate_interval_transport,
    integrate_sampled_interval_transport,
    reconstruct_conservative_state,
    wind_escape_diagnostics,
    wind_launch_energy,
)
from .transonic_collocation import (
    TransonicSlimParams,
    computational_grid,
    pack_state,
    sonic_residual_pair,
    unpack_state,
)
from .transonic_local import stream_source_interval_integral, stream_source_prime
from .winds import energy_limited_wind, q_edd_vertical


@dataclass(frozen=True)
class ConservativeBoundary:
    """Outer reservoir values used by the first conservative implementation."""

    outer_log_temperature: float
    outer_log_omega_ratio: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.outer_log_temperature):
            raise ValueError("outer_log_temperature must be finite")
        if not np.isfinite(self.outer_log_omega_ratio):
            raise ValueError("outer_log_omega_ratio must be finite")


@dataclass(frozen=True)
class ConservativeSolverParams:
    """Configuration for the ordinary unified conservative solve."""

    disk: TransonicSlimParams
    closure: PhysicalTransportClosure
    boundary: ConservativeBoundary
    scales: ConservativeScales | None = None
    sonic_pivot: str = "C2"
    residual_tolerance: float = 1.0e-6
    max_nfev: int = 300
    mass_weight: float = 1.0
    angular_momentum_weight: float = 1.0
    energy_flux_weight: float = 1.0
    energy_balance_weight: float = 1.0
    inner_mass_weight: float = 1.0
    sonic_mode: str = "legacy"
    sonic_weight: float = 1.0
    jacobian_rel_step: float | None = None
    wind_energy_transport_mode: str = "power"

    def __post_init__(self) -> None:
        if self.sonic_pivot not in {"C1", "C2", "K"}:
            raise ValueError("sonic_pivot must be C1, C2, or K")
        if not np.isfinite(self.residual_tolerance) or self.residual_tolerance <= 0.0:
            raise ValueError("residual_tolerance must be positive and finite")
        if self.max_nfev <= 0:
            raise ValueError("max_nfev must be positive")
        for name, value in {
            "mass_weight": self.mass_weight,
            "angular_momentum_weight": self.angular_momentum_weight,
            "energy_flux_weight": self.energy_flux_weight,
            "inner_mass_weight": self.inner_mass_weight,
        }.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not np.isfinite(self.energy_balance_weight) or self.energy_balance_weight <= 0.0:
            raise ValueError("energy_balance_weight must be positive and finite")
        if self.sonic_mode not in {"legacy", "conservative"}:
            raise ValueError("sonic_mode must be 'legacy' or 'conservative'")
        if not np.isfinite(self.sonic_weight) or self.sonic_weight <= 0.0:
            raise ValueError("sonic_weight must be positive and finite")
        if self.jacobian_rel_step is not None and (
            not np.isfinite(self.jacobian_rel_step) or self.jacobian_rel_step <= 0.0
        ):
            raise ValueError("jacobian_rel_step must be positive when supplied")
        if self.wind_energy_transport_mode not in {"power", "carried"}:
            raise ValueError("wind_energy_transport_mode must be 'power' or 'carried'")

    @property
    def flux_scales(self) -> ConservativeScales:
        return default_conservative_scales(self.disk) if self.scales is None else self.scales


@dataclass(frozen=True)
class ConservativeResidualAudit:
    """Maximum residual by production equation family."""

    radial: float
    mass: float
    angular_momentum: float
    energy: float
    energy_compatibility: float
    outer_temperature: float
    outer_omega: float
    sonic: float
    inner_mass: float
    inner_energy_anchor: float

    @property
    def maximum(self) -> float:
        return float(max(abs(value) for value in self.__dict__.values()))


@dataclass(frozen=True)
class ConservativeSolveResult:
    """One ordinary conservative least-squares solve."""

    x: np.ndarray
    success: bool
    accepted: bool
    nfev: int
    cost: float
    optimality: float
    message: str
    initial_audit: ConservativeResidualAudit
    final_audit: ConservativeResidualAudit


@dataclass(frozen=True)
class ConservativeJacobianDirectionalAudit:
    """Directional agreement between the block Jacobian and production residual."""

    steps: np.ndarray
    relative_errors: np.ndarray
    best_relative_error: float


@dataclass(frozen=True)
class ConservativePseudoArclengthResult:
    """One bordered continuation step in inverse launch energy."""

    x: np.ndarray
    eta_E: float
    success: bool
    accepted: bool
    nfev: int
    cost: float
    optimality: float
    arc_residual: float
    tangent_mu: float
    message: str
    final_audit: ConservativeResidualAudit


@dataclass(frozen=True)
class ConservativeSonicDiagnostics:
    """Criticality data for the five-field conservative local DAE."""

    determinant: float
    compatibility: float
    singular_values: np.ndarray
    smin_over_smax: float
    left_null: np.ndarray
    right_null: np.ndarray
    matrix: np.ndarray
    affine_rhs: np.ndarray
    reference_gradient: np.ndarray


def pack_conservative_state(
    logu,
    logT,
    F,
    j,
    epsilon,
    logR_son: float,
) -> np.ndarray:
    """Pack five node fields and the free sonic radius."""

    arrays = [np.asarray(value, dtype=float) for value in (logu, logT, F, j, epsilon)]
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays[1:]) or len(shape) != 1:
        raise ValueError("all conservative node fields must be one-dimensional with equal shape")
    return np.concatenate([*arrays, np.asarray([logR_son], dtype=float)])


def unpack_conservative_state(
    x,
    params: ConservativeSolverParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Unpack a conservative state and construct its free-boundary grid."""

    vector = np.asarray(x, dtype=float)
    n = int(params.disk.n_nodes)
    expected = 5 * n + 1
    if vector.shape != (expected,):
        raise ValueError(f"conservative state must have shape ({expected},)")
    logu = vector[:n]
    logT = vector[n : 2 * n]
    F = vector[2 * n : 3 * n]
    j = vector[3 * n : 4 * n]
    epsilon = vector[4 * n : 5 * n]
    logR_son = float(vector[-1])
    logR = computational_grid(params.disk, logR_son)
    return logu, logT, F, j, epsilon, logR_son, logR


def conservative_state_bounds(params: ConservativeSolverParams) -> tuple[np.ndarray, np.ndarray]:
    """Return broad physical bounds for the conservative state."""

    disk = params.disk
    n = int(disk.n_nodes)
    lower = np.concatenate(
        [
            np.full(n, disk.logu_bounds[0]),
            np.full(n, disk.logT_bounds[0]),
            np.full(n, 1.0e-6),
            np.full(n, -20.0),
            np.full(n, -20.0),
            np.asarray([np.log(disk.R_son_bounds_rg[0] * disk.r_g)]),
        ]
    )
    upper = np.concatenate(
        [
            np.full(n, disk.logu_bounds[1]),
            np.full(n, disk.logT_bounds[1]),
            np.full(n, 20.0),
            np.full(n, 20.0),
            np.full(n, 20.0),
            np.asarray([np.log(disk.R_son_bounds_rg[1] * disk.r_g)]),
        ]
    )
    return lower, upper


def _node_states(
    logR: np.ndarray,
    logu: np.ndarray,
    logT: np.ndarray,
    F: np.ndarray,
    j: np.ndarray,
    params: ConservativeSolverParams,
) -> list[ConservativeNodeState]:
    scales = params.flux_scales
    return [
        reconstruct_conservative_state(
            float(logR[idx]),
            float(logu[idx]),
            float(logT[idx]),
            float(F[idx]),
            float(j[idx]),
            params.disk,
            scales,
        )
        for idx in range(params.disk.n_nodes)
    ]


def _midpoint_state(
    idx: int,
    logR: np.ndarray,
    logu: np.ndarray,
    logT: np.ndarray,
    F: np.ndarray,
    j: np.ndarray,
    params: ConservativeSolverParams,
) -> ConservativeNodeState:
    return reconstruct_conservative_state(
        0.5 * float(logR[idx] + logR[idx + 1]),
        0.5 * float(logu[idx] + logu[idx + 1]),
        0.5 * float(logT[idx] + logT[idx + 1]),
        0.5 * float(F[idx] + F[idx + 1]),
        0.5 * float(j[idx] + j[idx + 1]),
        params.disk,
        params.flux_scales,
    )


def _wind_mass_power_details(
    state: ConservativeNodeState,
    *,
    dOmega_dx: float,
    de_dx: float,
    drho_dx: float,
    params: ConservativeSolverParams,
) -> tuple[float, float, float, float, bool]:
    """Return effective/raw wind mass and power per ``dlnR``."""

    disk = params.disk
    epsilon_w = float(disk.wind_energy_limited_epsilon)
    if epsilon_w == 0.0:
        return 0.0, 0.0, 0.0, 0.0, False
    Tdsdx = float(de_dx - state.P * drho_dx / state.rho**2)
    q_visc = float(-state.W * dOmega_dx)
    q_adv = float(-(state.Sigma * state.u / state.R) * Tdsdx)
    q_available = float(q_visc - q_adv)
    q_edd = float(q_edd_vertical(state.Omega_K, state.H, kappa=disk.kappa))
    launch = wind_launch_energy(state, params.closure, disk)
    width = float(disk.wind_activation_width_fraction) * q_edd
    q_wind, _q_rad_limited, _dot_sigma = energy_limited_wind(
        q_available,
        q_edd,
        launch,
        epsilon_w,
        chi_edd=float(disk.wind_eddington_chi),
        activation_width=width,
    )
    allocated_power_prime = float(2.0 * np.pi * state.R**2 * q_wind)
    raw_mass_prime = float(allocated_power_prime / launch)
    cap_ratio = params.closure.wind_mass_loading_cap_per_log_radius
    if cap_ratio is None:
        effective_mass_prime = raw_mass_prime
        cap_active = False
    else:
        cap = float(cap_ratio * state.mdot)
        effective_mass_prime = float(min(raw_mass_prime, cap))
        cap_active = bool(raw_mass_prime > cap)
    effective_power_prime = float(effective_mass_prime * launch)
    return (
        effective_mass_prime,
        effective_power_prime,
        raw_mass_prime,
        allocated_power_prime,
        cap_active,
    )


def _wind_mass_and_power_prime(
    state: ConservativeNodeState,
    *,
    dOmega_dx: float,
    de_dx: float,
    drho_dx: float,
    params: ConservativeSolverParams,
) -> tuple[float, float]:
    """Return effective wind mass loss and launch power per ``dlnR``."""

    details = _wind_mass_power_details(
        state,
        dOmega_dx=dOmega_dx,
        de_dx=de_dx,
        drho_dx=drho_dx,
        params=params,
    )
    return details[0], details[1]


def _wind_prime(
    state: ConservativeNodeState,
    *,
    dOmega_dx: float,
    de_dx: float,
    drho_dx: float,
    params: ConservativeSolverParams,
) -> float:
    """Compatibility wrapper returning only wind mass loss."""

    return _wind_mass_and_power_prime(
        state,
        dOmega_dx=dOmega_dx,
        de_dx=de_dx,
        drho_dx=drho_dx,
        params=params,
    )[0]


def _directional_node_derivatives(
    logR: float,
    state_vector: np.ndarray,
    gradient: np.ndarray,
    params: ConservativeSolverParams,
    *,
    step: float = 2.0e-6,
) -> tuple[ConservativeNodeState, dict[str, float]]:
    """Return local state and derivatives along ``d/dlnR``."""

    state_vector = np.asarray(state_vector, dtype=float)
    gradient = np.asarray(gradient, dtype=float)
    if state_vector.shape != (5,) or gradient.shape != (5,):
        raise ValueError("conservative local state and gradient must have shape (5,)")
    center = reconstruct_conservative_state(
        logR,
        state_vector[0],
        state_vector[1],
        state_vector[2],
        state_vector[3],
        params.disk,
        params.flux_scales,
    )
    plus_q = state_vector + step * gradient
    minus_q = state_vector - step * gradient
    if plus_q[2] <= 0.0 or minus_q[2] <= 0.0:
        raise ValueError("directional mass flux crossed zero")
    plus = reconstruct_conservative_state(
        logR + step,
        plus_q[0],
        plus_q[1],
        plus_q[2],
        plus_q[3],
        params.disk,
        params.flux_scales,
    )
    minus = reconstruct_conservative_state(
        logR - step,
        minus_q[0],
        minus_q[1],
        minus_q[2],
        minus_q[3],
        params.disk,
        params.flux_scales,
    )
    derivatives = {
        name: float((getattr(plus, name) - getattr(minus, name)) / (2.0 * step))
        for name in ("Pi", "rho", "e", "Omega", "mechanical_energy_flux")
    }
    return center, derivatives


def conservative_local_dae_residual(
    logR: float,
    state_vector,
    gradient,
    params: ConservativeSolverParams,
) -> np.ndarray:
    """Return the five local differential equations in conservative variables."""

    state_vector = np.asarray(state_vector, dtype=float)
    gradient = np.asarray(gradient, dtype=float)
    state, derivatives = _directional_node_derivatives(
        float(logR), state_vector, gradient, params
    )
    stream_prime = float(stream_source_prime(float(logR), params.disk))
    wind_prime, wind_power_prime = _wind_mass_and_power_prime(
        state,
        dOmega_dx=derivatives["Omega"],
        de_dx=derivatives["e"],
        drho_dx=derivatives["rho"],
        params=params,
    )
    source = conservative_source_terms(
        state,
        stream_prime=stream_prime,
        wind_prime=wind_prime,
        closure=params.closure,
        params=params.disk,
        wind_launch_power_prime=(
            wind_power_prime if params.wind_energy_transport_mode == "power" else None
        ),
    )
    radial_raw = float(
        state.u**2 * gradient[0]
        - state.R**2 * (state.Omega**2 - state.Omega_K**2)
        + derivatives["Pi"] / state.Sigma
    )
    radial_scale = max(
        state.u**2,
        state.R**2 * state.Omega_K**2,
        abs(derivatives["Pi"] / state.Sigma),
        1.0,
    )
    vertical_work = float(
        state.mdot
        * (
            derivatives["Pi"] / state.Sigma
            - state.P * derivatives["rho"] / state.rho**2
        )
    )
    scales = params.flux_scales
    return np.asarray(
        [
            radial_raw / radial_scale,
            gradient[2] - source.mass_rhs / scales.mdot,
            gradient[3] - source.angular_rhs / scales.angular_flux,
            gradient[4] - source.energy_rhs / scales.energy_flux,
            (derivatives["mechanical_energy_flux"] + vertical_work - source.energy_rhs)
            / scales.energy_flux,
        ],
        dtype=float,
    )


def conservative_local_dae_matrix(
    logR: float,
    state_vector,
    params: ConservativeSolverParams,
    *,
    reference_gradient=None,
    rel_step: float = 2.0e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearize the local conservative DAE as ``A g + c``."""

    state_vector = np.asarray(state_vector, dtype=float)
    reference = (
        np.zeros(5, dtype=float)
        if reference_gradient is None
        else np.asarray(reference_gradient, dtype=float)
    )
    if reference.shape != (5,):
        raise ValueError("reference_gradient must have shape (5,)")
    base = conservative_local_dae_residual(logR, state_vector, reference, params)
    matrix = np.empty((5, 5), dtype=float)
    for column in range(5):
        width = rel_step * max(1.0, abs(float(reference[column])))
        direction = np.zeros(5, dtype=float)
        direction[column] = width
        plus = conservative_local_dae_residual(
            logR, state_vector, reference + direction, params
        )
        minus = conservative_local_dae_residual(
            logR, state_vector, reference - direction, params
        )
        matrix[:, column] = (plus - minus) / (2.0 * width)
    affine_rhs = np.asarray(base - matrix @ reference, dtype=float)
    return matrix, affine_rhs


def conservative_sonic_diagnostics(
    logR: float,
    state_vector,
    params: ConservativeSolverParams,
    *,
    reference_gradient=None,
) -> ConservativeSonicDiagnostics:
    """Return singularity and compatibility diagnostics for the conservative DAE."""

    reference = (
        np.zeros(5, dtype=float)
        if reference_gradient is None
        else np.asarray(reference_gradient, dtype=float)
    )
    matrix, affine_rhs = conservative_local_dae_matrix(
        logR,
        state_vector,
        params,
        reference_gradient=reference,
    )
    left, singular_values, right_t = np.linalg.svd(matrix, full_matrices=True)
    left_null = np.asarray(left[:, -1], dtype=float)
    right_null = np.asarray(right_t[-1, :], dtype=float)
    pivot = int(np.argmax(np.abs(left_null)))
    if left_null[pivot] < 0.0:
        left_null = -left_null
        right_null = -right_null
    row_norms = np.maximum(np.linalg.norm(matrix, axis=1), 1.0e-300)
    determinant = float(np.linalg.det(matrix) / np.prod(row_norms))
    compatibility = float(np.dot(left_null, affine_rhs))
    ratio = float(singular_values[-1] / max(singular_values[0], 1.0e-300))
    return ConservativeSonicDiagnostics(
        determinant=determinant,
        compatibility=compatibility,
        singular_values=np.asarray(singular_values, dtype=float),
        smin_over_smax=ratio,
        left_null=left_null,
        right_null=right_null,
        matrix=matrix,
        affine_rhs=affine_rhs,
        reference_gradient=reference,
    )


def conservative_sonic_residual_pair(x, params: ConservativeSolverParams) -> np.ndarray:
    """Return ``[det(A), u_min^T c]`` at the free sonic boundary."""

    logu, logT, F, j, epsilon, _logR_son, logR = unpack_conservative_state(x, params)
    dx = float(logR[1] - logR[0])
    state0 = np.asarray([logu[0], logT[0], F[0], j[0], epsilon[0]], dtype=float)
    state1 = np.asarray([logu[1], logT[1], F[1], j[1], epsilon[1]], dtype=float)
    diagnostics = conservative_sonic_diagnostics(
        float(logR[0]),
        state0,
        params,
        reference_gradient=(state1 - state0) / dx,
    )
    return np.asarray([diagnostics.determinant, diagnostics.compatibility], dtype=float)


def _interval_rows(
    idx: int,
    logR: np.ndarray,
    logu: np.ndarray,
    logT: np.ndarray,
    F: np.ndarray,
    j: np.ndarray,
    epsilon: np.ndarray,
    nodes: list[ConservativeNodeState],
    params: ConservativeSolverParams,
    *,
    return_transport: bool = False,
) -> np.ndarray | tuple[np.ndarray, ConservativeIntervalTransport]:
    left = nodes[idx]
    right = nodes[idx + 1]
    middle = _midpoint_state(idx, logR, logu, logT, F, j, params)
    dx = float(logR[idx + 1] - logR[idx])
    dlogu_dx = float((logu[idx + 1] - logu[idx]) / dx)
    dPi_dx = float((right.Pi - left.Pi) / dx)
    drho_dx = float((right.rho - left.rho) / dx)
    de_dx = float((right.e - left.e) / dx)
    dOmega_dx = float((right.Omega - left.Omega) / dx)

    radial_raw = float(
        middle.u**2 * dlogu_dx
        - middle.R**2 * (middle.Omega**2 - middle.Omega_K**2)
        + dPi_dx / middle.Sigma
    )
    radial_scale = max(
        middle.u**2,
        middle.R**2 * middle.Omega_K**2,
        abs(dPi_dx / middle.Sigma),
        1.0,
    )

    quadrature = (left, middle, right)
    weights = (1.0, 4.0, 1.0)
    source_rows = []
    vertical_rows = []
    for state in quadrature:
        stream_prime = float(stream_source_prime(state.logR, params.disk))
        wind_prime, wind_power_prime = _wind_mass_and_power_prime(
            state,
            dOmega_dx=dOmega_dx,
            de_dx=de_dx,
            drho_dx=drho_dx,
            params=params,
        )
        source_rows.append(
            conservative_source_terms(
                state,
                stream_prime=stream_prime,
                wind_prime=wind_prime,
                closure=params.closure,
                params=params.disk,
                wind_launch_power_prime=(
                    wind_power_prime
                    if params.wind_energy_transport_mode == "power"
                    else None
                ),
            )
        )
        vertical_rows.append(
            state.mdot
            * (dPi_dx / state.Sigma - state.P * drho_dx / state.rho**2)
        )

    transport = integrate_interval_transport(
        dx,
        source_rows[0],
        source_rows[1],
        source_rows[2],
        exact_stream_mass=stream_source_interval_integral(
            float(logR[idx]), float(logR[idx + 1]), params.disk
        ),
    )
    vertical_integral = (dx / 6.0) * sum(
        weight * value for weight, value in zip(weights, vertical_rows)
    )
    scales = params.flux_scales
    mass_row = float((right.mdot - left.mdot - transport.mass_rhs) / scales.mdot)
    angular_row = float((right.J - left.J - transport.angular_rhs) / scales.angular_flux)
    energy_row = float(
        epsilon[idx + 1]
        - epsilon[idx]
        - transport.energy_rhs / scales.energy_flux
    )
    # This is the compatibility row after eliminating the repeated epsilon
    # increment against the energy-conservation row.  The transformed system
    # has the same roots but avoids an almost collinear pair of residuals.
    compatibility_row = float(
        (
            right.mechanical_energy_flux
            - left.mechanical_energy_flux
            + vertical_integral
            - transport.energy_rhs
        )
        / scales.energy_flux
    )
    rows = np.asarray(
        [
            radial_raw / radial_scale,
            params.mass_weight * mass_row,
            params.angular_momentum_weight * angular_row,
            params.energy_flux_weight * energy_row,
            params.energy_balance_weight * compatibility_row,
        ],
        dtype=float,
    )
    if return_transport:
        return rows, transport
    return rows


def conservative_residual(x, params: ConservativeSolverParams) -> np.ndarray:
    """Return the square ordinary conservative production residual."""

    n = int(params.disk.n_nodes)
    residual = np.zeros(5 * n + 1, dtype=float)
    try:
        logu, logT, F, j, epsilon, logR_son, logR = unpack_conservative_state(x, params)
        if np.any(np.diff(logR) <= 0.0) or np.any(F <= 0.0):
            raise ValueError("invalid conservative state")
        nodes = _node_states(logR, logu, logT, F, j, params)
        row = 0
        for idx in range(n - 1):
            residual[row : row + 5] = _interval_rows(
                idx, logR, logu, logT, F, j, epsilon, nodes, params
            )
            row += 5

        outer = nodes[-1]
        residual[row] = float(logT[-1] - params.boundary.outer_log_temperature)
        residual[row + 1] = float(
            np.log(outer.Omega / outer.Omega_K) - params.boundary.outer_log_omega_ratio
        )
        row += 2

        if params.sonic_mode == "conservative":
            residual[row : row + 2] = params.sonic_weight * conservative_sonic_residual_pair(x, params)
        else:
            lambda0 = float(j[0] * params.flux_scales.specific_angular_momentum / (params.disk.r_g * C))
            legacy_z = pack_state(logu, logT, logR_son, lambda0)
            residual[row : row + 2] = params.sonic_weight * sonic_residual_pair(
                legacy_z,
                params.disk,
                pivot=params.sonic_pivot,
            )
        row += 2
        residual[row] = float(params.inner_mass_weight * (F[0] - 1.0))
        residual[row + 1] = float(
            epsilon[0] - nodes[0].mechanical_energy_flux / params.flux_scales.energy_flux
        )
    except Exception:
        residual.fill(1.0e6)
    return residual


def conservative_residual_audit(x, params: ConservativeSolverParams) -> ConservativeResidualAudit:
    """Partition the square residual by equation family."""

    values = conservative_residual(x, params)
    n = int(params.disk.n_nodes)
    interval = values[: 5 * (n - 1)].reshape(n - 1, 5)
    tail = values[5 * (n - 1) :]
    maxima = np.max(np.abs(interval), axis=0)
    return ConservativeResidualAudit(
        radial=float(maxima[0]),
        mass=float(maxima[1] / params.mass_weight),
        angular_momentum=float(maxima[2] / params.angular_momentum_weight),
        energy=float(maxima[3] / params.energy_flux_weight),
        energy_compatibility=float(maxima[4] / params.energy_balance_weight),
        outer_temperature=float(abs(tail[0])),
        outer_omega=float(abs(tail[1])),
        sonic=float(np.max(np.abs(tail[2:4])) / params.sonic_weight),
        inner_mass=float(abs(tail[4]) / params.inner_mass_weight),
        inner_energy_anchor=float(abs(tail[5])),
    )


def conservative_residual_profile(x, params: ConservativeSolverParams) -> dict[str, np.ndarray]:
    """Return unweighted interval residuals and their midpoint radii."""

    values = conservative_residual(x, params)
    n = int(params.disk.n_nodes)
    interval = values[: 5 * (n - 1)].reshape(n - 1, 5).copy()
    interval[:, 1] /= params.mass_weight
    interval[:, 2] /= params.angular_momentum_weight
    interval[:, 3] /= params.energy_flux_weight
    interval[:, 4] /= params.energy_balance_weight
    *_fields, logR = unpack_conservative_state(x, params)
    midpoint_logR = 0.5 * (logR[:-1] + logR[1:])
    return {
        "R_mid_rg": np.exp(midpoint_logR) / params.disk.r_g,
        "radial": interval[:, 0],
        "mass": interval[:, 1],
        "angular_momentum": interval[:, 2],
        "energy": interval[:, 3],
        "energy_compatibility": interval[:, 4],
    }


def conservative_transport_profile(x, params: ConservativeSolverParams) -> dict[str, np.ndarray]:
    """Return decomposed interval source integrals used by production rows."""

    logu, logT, F, j, epsilon, _logR_son, logR = unpack_conservative_state(x, params)
    nodes = _node_states(logR, logu, logT, F, j, params)
    transports: list[ConservativeIntervalTransport] = []
    for idx in range(int(params.disk.n_nodes) - 1):
        _rows, transport = _interval_rows(
            idx,
            logR,
            logu,
            logT,
            F,
            j,
            epsilon,
            nodes,
            params,
            return_transport=True,
        )
        transports.append(transport)
    fields = tuple(ConservativeIntervalTransport.__dataclass_fields__)
    result = {
        name: np.asarray([getattr(item, name) for item in transports], dtype=float)
        for name in fields
    }
    result["stream_mass_quadrature_error"] = np.asarray(
        [item.stream_mass_quadrature_error for item in transports], dtype=float
    )
    result["mass_rhs"] = np.asarray([item.mass_rhs for item in transports], dtype=float)
    result["angular_rhs"] = np.asarray([item.angular_rhs for item in transports], dtype=float)
    result["energy_rhs"] = np.asarray([item.energy_rhs for item in transports], dtype=float)
    result["R_mid_rg"] = np.exp(0.5 * (logR[:-1] + logR[1:])) / params.disk.r_g
    return result


def conservative_wind_escape_profile(
    x,
    params: ConservativeSolverParams,
    *,
    target_terminal_bernoulli: float = 0.0,
) -> dict[str, np.ndarray]:
    """Return midpoint wind-launch and terminal-Bernoulli diagnostics."""

    logu, logT, F, j, _epsilon, _logR_son, logR = unpack_conservative_state(x, params)
    nodes = _node_states(logR, logu, logT, F, j, params)
    transport = conservative_transport_profile(x, params)
    diagnostics = []
    wind_prime = []
    wind_power_prime = []
    wind_raw_prime = []
    wind_allocated_power_prime = []
    wind_cap_active = []
    for idx in range(int(params.disk.n_nodes) - 1):
        dx = float(logR[idx + 1] - logR[idx])
        state = _midpoint_state(idx, logR, logu, logT, F, j, params)
        (
            mass_prime,
            power_prime,
            raw_mass_prime,
            allocated_power_prime,
            cap_active,
        ) = _wind_mass_power_details(
            state,
            dOmega_dx=float((nodes[idx + 1].Omega - nodes[idx].Omega) / dx),
            de_dx=float((nodes[idx + 1].e - nodes[idx].e) / dx),
            drho_dx=float((nodes[idx + 1].rho - nodes[idx].rho) / dx),
            params=params,
        )
        wind_prime.append(mass_prime)
        wind_power_prime.append(power_prime)
        wind_raw_prime.append(raw_mass_prime)
        wind_allocated_power_prime.append(allocated_power_prime)
        wind_cap_active.append(cap_active)
        diagnostics.append(
            wind_escape_diagnostics(
                state,
                params.closure,
                params.disk,
                target_terminal_bernoulli=target_terminal_bernoulli,
            )
        )

    fields = (
        "disk_bernoulli",
        "target_terminal_bernoulli",
        "required_launch_energy",
        "prescribed_launch_energy",
        "wind_bernoulli",
        "terminal_margin",
        "terminal_speed",
        "escaping",
    )
    result = {
        name: np.asarray([getattr(item, name) for item in diagnostics])
        for name in fields
    }
    result["R_mid_rg"] = np.asarray(transport["R_mid_rg"], dtype=float)
    result["wind_mass"] = np.asarray(transport["wind_mass"], dtype=float)
    result["wind_prime"] = np.asarray(wind_prime, dtype=float)
    result["wind_launch_power_prime"] = np.asarray(wind_power_prime, dtype=float)
    result["wind_raw_prime"] = np.asarray(wind_raw_prime, dtype=float)
    result["wind_allocated_power_prime"] = np.asarray(
        wind_allocated_power_prime, dtype=float
    )
    result["wind_cap_active"] = np.asarray(wind_cap_active, dtype=bool)
    return result


def conservative_transport_quadrature_profile(
    x,
    params: ConservativeSolverParams,
    *,
    order: int,
) -> dict[str, np.ndarray]:
    """Reintegrate transport with Gauss-Legendre samples for audit purposes."""

    if order < 2:
        raise ValueError("quadrature order must be at least two")
    logu, logT, F, j, _epsilon, _logR_son, logR = unpack_conservative_state(x, params)
    nodes = _node_states(logR, logu, logT, F, j, params)
    abscissa, gauss_weights = np.polynomial.legendre.leggauss(int(order))
    fractions = 0.5 * (abscissa + 1.0)
    normalized_weights = 0.5 * gauss_weights
    transports: list[ConservativeIntervalTransport] = []
    for idx in range(int(params.disk.n_nodes) - 1):
        left = nodes[idx]
        right = nodes[idx + 1]
        dx = float(logR[idx + 1] - logR[idx])
        d_omega = float((right.Omega - left.Omega) / dx)
        d_e = float((right.e - left.e) / dx)
        d_rho = float((right.rho - left.rho) / dx)
        samples = []
        for fraction in fractions:
            state = reconstruct_conservative_state(
                float((1.0 - fraction) * logR[idx] + fraction * logR[idx + 1]),
                float((1.0 - fraction) * logu[idx] + fraction * logu[idx + 1]),
                float((1.0 - fraction) * logT[idx] + fraction * logT[idx + 1]),
                float((1.0 - fraction) * F[idx] + fraction * F[idx + 1]),
                float((1.0 - fraction) * j[idx] + fraction * j[idx + 1]),
                params.disk,
                params.flux_scales,
            )
            wind_prime, wind_power_prime = _wind_mass_and_power_prime(
                state,
                dOmega_dx=d_omega,
                de_dx=d_e,
                drho_dx=d_rho,
                params=params,
            )
            samples.append(
                conservative_source_terms(
                    state,
                    stream_prime=float(stream_source_prime(state.logR, params.disk)),
                    wind_prime=wind_prime,
                    closure=params.closure,
                    params=params.disk,
                    wind_launch_power_prime=(
                        wind_power_prime
                        if params.wind_energy_transport_mode == "power"
                        else None
                    ),
                )
            )
        transports.append(
            integrate_sampled_interval_transport(
                dx,
                samples,
                normalized_weights,
                exact_stream_mass=stream_source_interval_integral(
                    float(logR[idx]), float(logR[idx + 1]), params.disk
                ),
            )
        )
    fields = tuple(ConservativeIntervalTransport.__dataclass_fields__)
    result = {
        name: np.asarray([getattr(item, name) for item in transports], dtype=float)
        for name in fields
    }
    result["mass_rhs"] = np.asarray([item.mass_rhs for item in transports], dtype=float)
    result["angular_rhs"] = np.asarray([item.angular_rhs for item in transports], dtype=float)
    result["energy_rhs"] = np.asarray([item.energy_rhs for item in transports], dtype=float)
    result["R_mid_rg"] = np.exp(0.5 * (logR[:-1] + logR[1:])) / params.disk.r_g
    return result


def residual_adapted_conservative_grid(
    x,
    params: ConservativeSolverParams,
    *,
    target_n: int | None = None,
    gain: float = 4.0,
    residual_floor: float = 1.0e-5,
    blend: float = 0.35,
    max_radius_rg: float | None = None,
) -> tuple[float, ...]:
    """Equidistribute nodes using the largest raw interval residual as monitor."""

    if target_n is None:
        target_n = int(params.disk.n_nodes)
    if target_n < 3:
        raise ValueError("target_n must be at least three")
    if not np.isfinite(gain) or gain < 0.0:
        raise ValueError("gain must be finite and non-negative")
    if not np.isfinite(residual_floor) or residual_floor <= 0.0:
        raise ValueError("residual_floor must be positive and finite")
    if not np.isfinite(blend) or not 0.0 <= blend <= 1.0:
        raise ValueError("blend must lie between zero and one")
    if max_radius_rg is not None and (
        not np.isfinite(max_radius_rg) or max_radius_rg <= 0.0
    ):
        raise ValueError("max_radius_rg must be positive and finite when supplied")

    profile = conservative_residual_profile(x, params)
    fields = ("radial", "mass", "angular_momentum", "energy", "energy_compatibility")
    score = np.max(np.vstack([np.abs(profile[name]) for name in fields]), axis=0)
    *_state, logR = unpack_conservative_state(x, params)
    xi = (logR - logR[0]) / (logR[-1] - logR[0])
    if max_radius_rg is not None:
        if target_n != xi.size:
            raise ValueError("radius-limited adaptation currently preserves node count")
        radius_rg = np.exp(logR) / params.disk.r_g
        cutoff = int(np.searchsorted(radius_rg, max_radius_rg, side="right") - 1)
        cutoff = min(max(cutoff, 2), xi.size - 2)
        active_intervals = cutoff
    else:
        cutoff = xi.size - 1
        active_intervals = score.size

    monitor = 1.0 + gain * np.minimum(
        score[:active_intervals] / residual_floor, 100.0
    )
    cumulative = np.concatenate(
        [[0.0], np.cumsum(monitor * np.diff(xi[: active_intervals + 1]))]
    )
    cumulative /= cumulative[-1]
    active_nodes = active_intervals + 1
    output_active_nodes = target_n if max_radius_rg is None else active_nodes
    target_coordinate = np.linspace(0.0, 1.0, output_active_nodes)
    adapted_xi = np.interp(target_coordinate, cumulative, xi[:active_nodes])
    baseline_xi = np.interp(
        target_coordinate,
        np.linspace(0.0, 1.0, active_nodes),
        xi[:active_nodes],
    )
    blended_active = (1.0 - blend) * baseline_xi + blend * adapted_xi
    if max_radius_rg is None:
        new_xi = blended_active
    else:
        new_xi = xi.copy()
        new_xi[:active_nodes] = blended_active
    new_xi[0] = 0.0
    new_xi[-1] = 1.0
    if np.any(np.diff(new_xi) <= 0.0):
        raise ValueError("adapted conservative grid is not strictly increasing")
    return tuple(float(value) for value in new_xi)


def source_block_refined_conservative_grid(
    x,
    params: ConservativeSolverParams,
    *,
    source_nodes: int,
) -> tuple[float, ...]:
    """Replace only the compact source band while preserving outside nodes."""

    if source_nodes < 5:
        raise ValueError("source_nodes must be at least five")
    *_fields, logR_son, logR = unpack_conservative_state(x, params)
    disk = params.disk
    center_fraction = float(disk.stream_source_center_fraction)
    width = float(disk.stream_source_log_width)
    if center_fraction <= 0.0 or width <= 0.0:
        raise ValueError("source center and width must be positive")
    center = float(np.log(center_fraction * disk.R_out))
    left = center - width
    right = center + width
    if left <= logR[0] or right >= logR[-1]:
        raise ValueError("source support must lie strictly inside the radial grid")

    source = np.linspace(left, right, int(source_nodes), dtype=float)
    source[int(np.argmin(np.abs(source - center)))] = center
    source = np.sort(source)
    inherited_inner = logR[logR < left]
    inherited_outer = logR[logR > right]
    combined = np.concatenate([inherited_inner, source, inherited_outer])
    if np.any(np.diff(combined) <= 0.0):
        raise ValueError("source-block refinement produced duplicate nodes")
    xi = (combined - logR_son) / (logR[-1] - logR_son)
    xi[0] = 0.0
    xi[-1] = 1.0
    return tuple(float(value) for value in xi)


def multidomain_conservative_grid(
    x,
    params: ConservativeSolverParams,
    *,
    target_n: int,
    source_nodes: int,
    frozen_inner_nodes: int = 12,
) -> tuple[float, ...]:
    """Refine broad disk domains while retaining a dedicated source block."""

    if target_n <= source_nodes + 4:
        raise ValueError("target_n must leave at least four nodes outside the source block")
    if frozen_inner_nodes < 2:
        raise ValueError("frozen_inner_nodes must be at least two")
    *_fields, logR_son, old_logR = unpack_conservative_state(x, params)
    disk = params.disk
    center = float(np.log(disk.stream_source_center_fraction * disk.R_out))
    width = float(disk.stream_source_log_width)
    left = center - width
    right = center + width
    old_inner = old_logR[old_logR < left]
    old_outer = old_logR[old_logR > right]
    if old_inner.size < frozen_inner_nodes or old_outer.size < 2:
        raise ValueError("existing grid does not resolve all multidomain regions")

    outside_n = int(target_n) - int(source_nodes)
    ratio = old_inner.size / (old_inner.size + old_outer.size)
    inner_n = int(round(outside_n * ratio))
    inner_n = min(max(inner_n, frozen_inner_nodes), outside_n - 2)
    outer_n = outside_n - inner_n

    frozen = old_inner[:frozen_inner_nodes]
    tail_n = inner_n - frozen_inner_nodes + 1
    tail = np.interp(
        np.linspace(0.0, 1.0, tail_n),
        np.linspace(0.0, 1.0, old_inner.size - frozen_inner_nodes + 1),
        old_inner[frozen_inner_nodes - 1 :],
    )
    inner = np.concatenate([frozen, tail[1:]])
    outer = np.interp(
        np.linspace(0.0, 1.0, outer_n),
        np.linspace(0.0, 1.0, old_outer.size),
        old_outer,
    )
    source = np.linspace(left, right, int(source_nodes), dtype=float)
    source[int(np.argmin(np.abs(source - center)))] = center
    source = np.sort(source)
    combined = np.concatenate([inner, source, outer])
    if combined.size != target_n or np.any(np.diff(combined) <= 0.0):
        raise ValueError("multidomain grid is not strictly increasing")
    xi = (combined - logR_son) / (old_logR[-1] - logR_son)
    xi[0] = 0.0
    xi[-1] = 1.0
    return tuple(float(value) for value in xi)


def nested_refined_conservative_grid(
    x,
    params: ConservativeSolverParams,
    *,
    target_n: int,
) -> tuple[float, ...]:
    """Insert midpoints into the longest intervals while preserving every node."""

    *_fields, logR_son, logR = unpack_conservative_state(x, params)
    current_n = int(logR.size)
    if target_n < current_n or target_n > 2 * current_n - 1:
        raise ValueError("one nested refinement pass requires N <= target_n <= 2*N-1")
    additions = int(target_n - current_n)
    if additions == 0:
        return tuple(float(value) for value in (logR - logR_son) / (logR[-1] - logR_son))
    intervals = np.argsort(np.diff(logR))[-additions:]
    refined = np.sort(
        np.concatenate([logR, 0.5 * (logR[intervals] + logR[intervals + 1])])
    )
    xi = (refined - logR_son) / (logR[-1] - logR_son)
    xi[0] = 0.0
    xi[-1] = 1.0
    return tuple(float(value) for value in xi)


def conservative_jacobian_sparsity(params: ConservativeSolverParams):
    """Return block-local sparsity for finite-difference Jacobian assembly."""

    n = int(params.disk.n_nodes)
    size = 5 * n + 1
    pattern = lil_matrix((size, size), dtype=int)
    sonic_radius_col = size - 1
    for idx in range(n - 1):
        rows = range(5 * idx, 5 * idx + 5)
        columns = []
        for field in range(5):
            columns.extend((field * n + idx, field * n + idx + 1))
        for row in rows:
            pattern[row, columns] = 1
            pattern[row, sonic_radius_col] = 1
    tail = 5 * (n - 1)
    outer_columns = [n - 1, 2 * n - 1, 3 * n - 1, 4 * n - 1]
    pattern[tail, n + n - 1] = 1
    pattern[tail + 1, outer_columns] = 1
    if params.sonic_mode == "conservative":
        sonic_columns = [sonic_radius_col]
        for field in range(5):
            sonic_columns.extend((field * n, field * n + 1))
    else:
        sonic_columns = [0, n, 3 * n, sonic_radius_col]
    pattern[tail + 2 : tail + 4, sonic_columns] = 1
    pattern[tail + 4, 2 * n] = 1
    pattern[tail + 5, [0, n, 2 * n, 3 * n, 4 * n, sonic_radius_col]] = 1
    return pattern.tocsr()


def _interval_block_from_state_vector(
    x,
    params: ConservativeSolverParams,
    idx: int,
) -> np.ndarray:
    logu, logT, F, j, epsilon, _logR_son, logR = unpack_conservative_state(x, params)
    nodes: list[ConservativeNodeState | None] = [None] * int(params.disk.n_nodes)
    for node_idx in (idx, idx + 1):
        nodes[node_idx] = reconstruct_conservative_state(
            float(logR[node_idx]),
            float(logu[node_idx]),
            float(logT[node_idx]),
            float(F[node_idx]),
            float(j[node_idx]),
            params.disk,
            params.flux_scales,
        )
    return _interval_rows(
        idx,
        logR,
        logu,
        logT,
        F,
        j,
        epsilon,
        nodes,  # type: ignore[arg-type]
        params,
    )


def _analytic_stream_sonic_column(
    x,
    params: ConservativeSolverParams,
    idx: int,
) -> np.ndarray:
    """Return the exact moving-grid stream contribution to ``dR/dlogRson``."""

    logu, logT, F, j, _epsilon, logR_son, logR = unpack_conservative_state(x, params)
    span = float(logR[-1] - logR_son)
    xi_left = float((logR[idx] - logR_son) / span)
    xi_right = float((logR[idx + 1] - logR_son) / span)
    stream_derivative = float(
        stream_source_prime(float(logR[idx + 1]), params.disk) * (1.0 - xi_right)
        - stream_source_prime(float(logR[idx]), params.disk) * (1.0 - xi_left)
    )
    midpoint = _midpoint_state(idx, logR, logu, logT, F, j, params)
    carried = carried_transport(midpoint, params.closure, params.disk)
    scales = params.flux_scales
    mass = stream_derivative / scales.mdot
    angular = stream_derivative * carried.l_stream / scales.angular_flux
    energy = stream_derivative * carried.B_stream / scales.energy_flux
    return np.asarray(
        [
            0.0,
            params.mass_weight * mass,
            params.angular_momentum_weight * angular,
            params.energy_flux_weight * energy,
            params.energy_balance_weight * energy,
        ],
        dtype=float,
    )


def _bounded_difference_column(
    function,
    x: np.ndarray,
    column: int,
    lower: np.ndarray,
    upper: np.ndarray,
    rel_step: float,
    *,
    base: np.ndarray,
    absolute_step: float | None = None,
) -> np.ndarray:
    value = float(x[column])
    step = float(
        rel_step * max(abs(value), 1.0)
        if absolute_step is None
        else absolute_step
    )
    plus_ok = value + step < upper[column]
    minus_ok = value - step > lower[column]
    if plus_ok and minus_ok:
        plus = np.array(x, copy=True)
        minus = np.array(x, copy=True)
        plus[column] += step
        minus[column] -= step
        return (function(plus) - function(minus)) / (2.0 * step)
    if plus_ok:
        plus = np.array(x, copy=True)
        plus[column] += step
        return (function(plus) - base) / step
    if minus_ok:
        minus = np.array(x, copy=True)
        minus[column] -= step
        return (base - function(minus)) / step
    return np.zeros_like(base)


def conservative_block_jacobian(
    x,
    params: ConservativeSolverParams,
    *,
    rel_step: float = 3.0e-6,
):
    """Return a sparse block-local Jacobian for the production residual."""

    if not np.isfinite(rel_step) or rel_step <= 0.0:
        raise ValueError("rel_step must be positive and finite")
    state = np.asarray(x, dtype=float)
    n = int(params.disk.n_nodes)
    size = 5 * n + 1
    if state.shape != (size,):
        raise ValueError(f"conservative state must have shape ({size},)")
    lower, upper = conservative_state_bounds(params)
    jacobian = lil_matrix((size, size), dtype=float)
    sonic_column = size - 1
    no_stream_params = replace(
        params,
        disk=replace(
            params.disk,
            stream_source_fraction=0.0,
            stream_mass_fraction=0.0,
        ),
    )

    for idx in range(n - 1):
        row = 5 * idx
        block = lambda trial, interval_idx=idx: _interval_block_from_state_vector(
            trial, params, interval_idx
        )
        base = block(state)
        for field in range(4):
            for node_idx in (idx, idx + 1):
                column = field * n + node_idx
                derivative = _bounded_difference_column(
                    block,
                    state,
                    column,
                    lower,
                    upper,
                    rel_step,
                    base=base,
                    absolute_step=1.0e-5 if field < 2 else 1.0e-6,
                )
                jacobian[row : row + 5, column] = derivative[:, None]
        jacobian[row + 3, 4 * n + idx] = -params.energy_flux_weight
        jacobian[row + 3, 4 * n + idx + 1] = params.energy_flux_weight
        no_stream_block = lambda trial, interval_idx=idx: _interval_block_from_state_vector(
            trial, no_stream_params, interval_idx
        )
        no_stream_base = no_stream_block(state)
        derivative = _bounded_difference_column(
            no_stream_block,
            state,
            sonic_column,
            lower,
            upper,
            rel_step,
            base=no_stream_base,
        )
        derivative += _analytic_stream_sonic_column(state, params, idx)
        jacobian[row : row + 5, sonic_column] = derivative[:, None]

    tail = 5 * (n - 1)
    tail_function = lambda trial: conservative_residual(trial, params)[tail:]
    tail_base = tail_function(state)
    tail_columns = {sonic_column}
    sonic_columns = {sonic_column}
    for field in range(5):
        tail_columns.update(
            {
                field * n,
                field * n + 1,
                field * n + n - 1,
            }
        )
        sonic_columns.update({field * n, field * n + 1})
    sonic_function = lambda trial: params.sonic_weight * conservative_sonic_residual_pair(
        trial, params
    )
    sonic_base = sonic_function(state)
    for column in sorted(tail_columns):
        derivative = _bounded_difference_column(
            tail_function,
            state,
            column,
            lower,
            upper,
            min(rel_step, 1.0e-6),
            base=tail_base,
        )
        if column in sonic_columns:
            derivative[2:4] = _bounded_difference_column(
                sonic_function,
                state,
                column,
                lower,
                upper,
                rel_step,
                base=sonic_base,
                absolute_step=1.0e-4,
            )
        jacobian[tail:, column] = derivative[:, None]
    return jacobian.tocsr()


def conservative_jacobian_directional_audit(
    x,
    params: ConservativeSolverParams,
    *,
    steps: tuple[float, ...] = (1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5, 1.0e-5),
    seed: int = 1234,
    jacobian_rel_step: float = 3.0e-6,
) -> ConservativeJacobianDirectionalAudit:
    """Compare the block Jacobian with centered directional differences."""

    if len(steps) == 0 or any(step <= 0.0 for step in steps):
        raise ValueError("directional audit steps must be positive")
    state = np.asarray(x, dtype=float)
    lower, upper = conservative_state_bounds(params)
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=state.size)
    direction /= max(float(np.linalg.norm(direction)), 1.0e-300)
    positive = direction > 0.0
    negative = direction < 0.0
    maximum = np.inf
    if np.any(positive):
        maximum = min(maximum, float(np.min((upper[positive] - state[positive]) / direction[positive])))
    if np.any(negative):
        maximum = min(maximum, float(np.min((lower[negative] - state[negative]) / direction[negative])))
    usable_steps = np.asarray([min(float(step), 0.25 * maximum) for step in steps], dtype=float)
    jacobian = conservative_block_jacobian(
        state, params, rel_step=jacobian_rel_step
    )
    expected = np.asarray(jacobian @ direction, dtype=float)
    expected_norm = max(float(np.linalg.norm(expected)), 1.0e-300)
    errors = []
    for step in usable_steps:
        plus = conservative_residual(state + step * direction, params)
        minus = conservative_residual(state - step * direction, params)
        measured = (plus - minus) / (2.0 * step)
        errors.append(float(np.linalg.norm(measured - expected) / expected_norm))
    error_array = np.asarray(errors, dtype=float)
    return ConservativeJacobianDirectionalAudit(
        steps=usable_steps,
        relative_errors=error_array,
        best_relative_error=float(np.min(error_array)),
    )


def _inverse_eta_params(
    params: ConservativeSolverParams,
    inverse_eta: float,
) -> ConservativeSolverParams:
    if not np.isfinite(inverse_eta) or inverse_eta <= 0.0:
        raise ValueError("inverse eta must be positive and finite")
    return replace(
        params,
        closure=replace(
            params.closure,
            wind_launch_energy_multiplier=float(1.0 / inverse_eta),
        ),
    )


def conservative_eta_bordered_jacobian(
    state,
    inverse_eta: float,
    params: ConservativeSolverParams,
    *,
    tangent: np.ndarray,
    scales: np.ndarray,
    jacobian_rel_step: float = 3.0e-6,
    inverse_eta_rel_step: float = 1.0e-5,
):
    """Return ``[dF/dx, dF/dmu; tangent/scales]`` for ``mu=1/eta``."""

    from scipy.sparse import csr_matrix, hstack, vstack

    state = np.asarray(state, dtype=float)
    tangent = np.asarray(tangent, dtype=float)
    scales = np.asarray(scales, dtype=float)
    if tangent.shape != (state.size + 1,) or scales.shape != tangent.shape:
        raise ValueError("border tangent and scales must match state plus inverse eta")
    local_params = _inverse_eta_params(params, inverse_eta)
    state_jacobian = conservative_block_jacobian(
        state, local_params, rel_step=jacobian_rel_step
    )
    step = float(inverse_eta_rel_step * max(abs(inverse_eta), 1.0e-2))
    plus = conservative_residual(
        state, _inverse_eta_params(params, inverse_eta + step)
    )
    minus = conservative_residual(
        state, _inverse_eta_params(params, inverse_eta - step)
    )
    mu_column = (plus - minus) / (2.0 * step)
    top = hstack([state_jacobian, csr_matrix(mu_column[:, None])], format="csr")
    border = csr_matrix((tangent / scales)[None, :])
    return vstack([top, border], format="csr")


def conservative_eta_pseudo_arclength_step(
    previous_state,
    previous_eta: float,
    current_state,
    current_eta: float,
    params: ConservativeSolverParams,
    *,
    step_factor: float = 0.5,
    jacobian_rel_step: float = 3.0e-6,
    max_nfev: int = 30,
    eta_bounds: tuple[float, float] = (1.0, 200.0),
) -> ConservativePseudoArclengthResult:
    """Take one bounded pseudo-arclength step using ``mu=1/eta_E``."""

    previous = np.asarray(previous_state, dtype=float)
    current = np.asarray(current_state, dtype=float)
    if previous.shape != current.shape:
        raise ValueError("continuation anchors must have matching shapes")
    if previous_eta <= 0.0 or current_eta <= 0.0:
        raise ValueError("continuation eta anchors must be positive")
    if step_factor <= 0.0:
        raise ValueError("step_factor must be positive")
    eta_min, eta_max = map(float, eta_bounds)
    if not 0.0 < eta_min < eta_max:
        raise ValueError("eta bounds must be positive and increasing")

    n = int(params.disk.n_nodes)
    state_scales = np.concatenate(
        [
            np.full(n, 0.1),
            np.full(n, 0.1),
            np.full(n, 0.05),
            np.full(n, 0.05),
            np.full(n, 0.01),
            np.asarray([0.01]),
        ]
    )
    scales = np.concatenate([state_scales, np.asarray([0.01])])
    previous_w = np.concatenate([previous, np.asarray([1.0 / previous_eta])])
    current_w = np.concatenate([current, np.asarray([1.0 / current_eta])])
    scaled_secant = (current_w - previous_w) / scales
    secant_norm = float(np.linalg.norm(scaled_secant))
    if not np.isfinite(secant_norm) or secant_norm <= 0.0:
        raise ValueError("continuation anchors do not define a finite secant")
    tangent = scaled_secant / secant_norm
    predicted = current_w + float(step_factor * secant_norm) * scales * tangent

    state_lower, state_upper = conservative_state_bounds(params)
    lower = np.concatenate([state_lower, np.asarray([1.0 / eta_max])])
    upper = np.concatenate([state_upper, np.asarray([1.0 / eta_min])])
    predicted = np.clip(predicted, lower + 1.0e-12, upper - 1.0e-12)

    def residual(w):
        inverse_eta = float(w[-1])
        equation = conservative_residual(
            w[:-1], _inverse_eta_params(params, inverse_eta)
        )
        arc = float(np.dot((w - predicted) / scales, tangent))
        return np.concatenate([equation, np.asarray([arc])])

    def jacobian(w):
        return conservative_eta_bordered_jacobian(
            w[:-1],
            float(w[-1]),
            params,
            tangent=tangent,
            scales=scales,
            jacobian_rel_step=jacobian_rel_step,
        )

    result = least_squares(
        residual,
        predicted,
        bounds=(lower, upper),
        jac=jacobian,
        x_scale=scales,
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-10,
        max_nfev=int(max_nfev),
        tr_solver="lsmr",
        tr_options={"regularize": False},
        verbose=0,
    )
    inverse_eta = float(result.x[-1])
    final_params = _inverse_eta_params(params, inverse_eta)
    final_audit = conservative_residual_audit(result.x[:-1], final_params)
    arc = float(np.dot((result.x - predicted) / scales, tangent))
    return ConservativePseudoArclengthResult(
        x=np.asarray(result.x[:-1], dtype=float),
        eta_E=float(1.0 / inverse_eta),
        success=bool(result.success),
        accepted=bool(final_audit.maximum <= params.residual_tolerance),
        nfev=int(result.nfev),
        cost=float(result.cost),
        optimality=float(result.optimality),
        arc_residual=arc,
        tangent_mu=float(tangent[-1]),
        message=str(result.message),
        final_audit=final_audit,
    )


def conservative_seed_from_legacy(
    z,
    disk: TransonicSlimParams,
    closure: PhysicalTransportClosure,
) -> tuple[np.ndarray, ConservativeSolverParams]:
    """Map a legacy transonic checkpoint into conservative variables."""

    logu, logT, logR_son, lambda0, logR = unpack_state(z, disk)
    scales = default_conservative_scales(disk)
    F = np.empty(disk.n_nodes, dtype=float)
    j = np.empty(disk.n_nodes, dtype=float)
    nodes: list[ConservativeNodeState] = []
    for idx in range(disk.n_nodes):
        from .transonic_local import algebraic_state, stream_mass_rate_and_derivative

        legacy = algebraic_state(logR[idx], logu[idx], logT[idx], lambda0, disk)
        mdot, _prime = stream_mass_rate_and_derivative(logR[idx], disk)
        torque = float(2.0 * np.pi * legacy.R**2 * legacy.W)
        F[idx] = float(mdot / scales.mdot)
        j[idx] = float((mdot * legacy.l - torque) / scales.angular_flux)
        nodes.append(
            reconstruct_conservative_state(
                logR[idx], logu[idx], logT[idx], F[idx], j[idx], disk, scales
            )
        )
    epsilon = np.empty(disk.n_nodes, dtype=float)
    epsilon[0] = nodes[0].mechanical_energy_flux / scales.energy_flux
    for idx in range(disk.n_nodes - 1):
        dx = float(logR[idx + 1] - logR[idx])
        dPi_dx = float((nodes[idx + 1].Pi - nodes[idx].Pi) / dx)
        drho_dx = float((nodes[idx + 1].rho - nodes[idx].rho) / dx)
        middle = reconstruct_conservative_state(
            0.5 * float(logR[idx] + logR[idx + 1]),
            0.5 * float(logu[idx] + logu[idx + 1]),
            0.5 * float(logT[idx] + logT[idx + 1]),
            0.5 * float(F[idx] + F[idx + 1]),
            0.5 * float(j[idx] + j[idx + 1]),
            disk,
            scales,
        )
        vertical = float(
            dx
            * middle.mdot
            * (dPi_dx / middle.Sigma - middle.P * drho_dx / middle.rho**2)
        )
        epsilon[idx + 1] = float(
            epsilon[idx]
            + (nodes[idx + 1].mechanical_energy_flux - nodes[idx].mechanical_energy_flux + vertical)
            / scales.energy_flux
        )
    boundary = ConservativeBoundary(
        outer_log_temperature=float(logT[-1]),
        outer_log_omega_ratio=float(np.log(nodes[-1].Omega / nodes[-1].Omega_K)),
    )
    solver_params = ConservativeSolverParams(
        disk=disk,
        closure=closure,
        boundary=boundary,
        scales=scales,
    )
    return pack_conservative_state(logu, logT, F, j, epsilon, logR_son), solver_params


def remap_conservative_state(
    x,
    old_params: ConservativeSolverParams,
    new_disk: TransonicSlimParams,
    *,
    method: str = "pchip",
) -> tuple[np.ndarray, ConservativeSolverParams]:
    """Remap a conservative solution to a new radial grid.

    Dimensional fluxes are interpolated before normalization, so the routine
    also remains well-defined when the reference accretion rate changes.
    """

    old = unpack_conservative_state(x, old_params)
    logu_old, logT_old, F_old, j_old, epsilon_old, logR_son, logR_old = old
    logR_new = computational_grid(new_disk, logR_son)
    old_scales = old_params.flux_scales
    new_scales = default_conservative_scales(new_disk)

    interpolation = str(method).strip().lower()
    if interpolation == "pchip":
        from scipy.interpolate import PchipInterpolator

        def interp(values):
            return np.asarray(PchipInterpolator(logR_old, values, extrapolate=True)(logR_new), dtype=float)

    elif interpolation == "linear":
        def interp(values):
            return np.asarray(np.interp(logR_new, logR_old, values), dtype=float)

    else:
        raise ValueError("conservative remap method must be 'linear' or 'pchip'")

    logu = interp(logu_old)
    logT = interp(logT_old)
    mdot = interp(F_old * old_scales.mdot)
    angular_flux = interp(j_old * old_scales.angular_flux)
    energy_flux = interp(epsilon_old * old_scales.energy_flux)
    F = np.maximum(mdot / new_scales.mdot, 1.0e-12)
    j = angular_flux / new_scales.angular_flux
    epsilon = energy_flux / new_scales.energy_flux
    remapped = pack_conservative_state(logu, logT, F, j, epsilon, logR_son)
    new_params = ConservativeSolverParams(
        disk=new_disk,
        closure=old_params.closure,
        boundary=old_params.boundary,
        scales=new_scales,
        sonic_pivot=old_params.sonic_pivot,
        residual_tolerance=old_params.residual_tolerance,
        max_nfev=old_params.max_nfev,
        mass_weight=old_params.mass_weight,
        angular_momentum_weight=old_params.angular_momentum_weight,
        energy_flux_weight=old_params.energy_flux_weight,
        energy_balance_weight=old_params.energy_balance_weight,
        inner_mass_weight=old_params.inner_mass_weight,
        sonic_mode=old_params.sonic_mode,
        sonic_weight=old_params.sonic_weight,
        jacobian_rel_step=old_params.jacobian_rel_step,
        wind_energy_transport_mode=old_params.wind_energy_transport_mode,
    )
    return remapped, new_params


def solve_conservative_disk(
    seed,
    params: ConservativeSolverParams,
) -> ConservativeSolveResult:
    """Solve the ordinary conservative system with a sparse local Jacobian."""

    x0 = np.asarray(seed, dtype=float)
    initial = conservative_residual_audit(x0, params)
    lower, upper = conservative_state_bounds(params)
    result = least_squares(
        lambda trial: conservative_residual(trial, params),
        np.clip(x0, lower + 1.0e-12, upper - 1.0e-12),
        bounds=(lower, upper),
        jac_sparsity=conservative_jacobian_sparsity(params),
        x_scale="jac",
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-10,
        max_nfev=int(params.max_nfev),
        diff_step=params.jacobian_rel_step,
        verbose=0,
    )
    final = conservative_residual_audit(result.x, params)
    return ConservativeSolveResult(
        x=np.asarray(result.x, dtype=float),
        success=bool(result.success),
        accepted=bool(final.maximum <= params.residual_tolerance),
        nfev=int(result.nfev),
        cost=float(result.cost),
        optimality=float(result.optimality),
        message=str(result.message),
        initial_audit=initial,
        final_audit=final,
    )


def solve_conservative_disk_block_jacobian(
    seed,
    params: ConservativeSolverParams,
    *,
    jacobian_rel_step: float = 3.0e-6,
    max_nfev: int | None = None,
) -> ConservativeSolveResult:
    """Solve using the interval-local production Jacobian."""

    x0 = np.asarray(seed, dtype=float)
    initial = conservative_residual_audit(x0, params)
    lower, upper = conservative_state_bounds(params)
    result = least_squares(
        lambda trial: conservative_residual(trial, params),
        np.clip(x0, lower + 1.0e-12, upper - 1.0e-12),
        bounds=(lower, upper),
        jac=lambda trial: conservative_block_jacobian(
            trial, params, rel_step=jacobian_rel_step
        ),
        x_scale="jac",
        ftol=1.0e-13,
        xtol=1.0e-13,
        gtol=1.0e-11,
        max_nfev=int(params.max_nfev if max_nfev is None else max_nfev),
        tr_solver="lsmr",
        tr_options={"regularize": False},
        verbose=0,
    )
    final = conservative_residual_audit(result.x, params)
    return ConservativeSolveResult(
        x=np.asarray(result.x, dtype=float),
        success=bool(result.success),
        accepted=bool(final.maximum <= params.residual_tolerance),
        nfev=int(result.nfev),
        cost=float(result.cost),
        optimality=float(result.optimality),
        message=str(result.message),
        initial_audit=initial,
        final_audit=final,
    )
