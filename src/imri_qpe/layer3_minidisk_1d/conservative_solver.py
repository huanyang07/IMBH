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

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from imri_qpe.constants import C, G

from .conservative_transport import (
    ConservativeNodeState,
    ConservativeScales,
    PhysicalTransportClosure,
    conservative_source_terms,
    default_conservative_scales,
    reconstruct_conservative_state,
)
from .transonic_collocation import (
    TransonicSlimParams,
    computational_grid,
    pack_state,
    sonic_residual_pair,
    unpack_state,
)
from .transonic_local import stream_source_prime
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


def _wind_prime(
    state: ConservativeNodeState,
    *,
    dOmega_dx: float,
    de_dx: float,
    drho_dx: float,
    params: ConservativeSolverParams,
) -> float:
    """Return energy-limited wind mass loss per ``dlnR`` at a quadrature point."""

    disk = params.disk
    epsilon_w = float(disk.wind_energy_limited_epsilon)
    if epsilon_w == 0.0:
        return 0.0
    Tdsdx = float(de_dx - state.P * drho_dx / state.rho**2)
    q_visc = float(-state.W * dOmega_dx)
    q_adv = float(-(state.Sigma * state.u / state.R) * Tdsdx)
    q_available = float(q_visc - q_adv)
    q_edd = float(q_edd_vertical(state.Omega_K, state.H, kappa=disk.kappa))
    launch = float(
        params.closure.wind_launch_energy_multiplier
        * (disk.M2_g * G)
        / (2.0 * state.R)
    )
    width = float(disk.wind_activation_width_fraction) * q_edd
    q_wind, _q_rad_limited, _dot_sigma = energy_limited_wind(
        q_available,
        q_edd,
        launch,
        epsilon_w,
        chi_edd=float(disk.wind_eddington_chi),
        activation_width=width,
    )
    return float(2.0 * np.pi * state.R**2 * q_wind / launch)


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
    wind_prime = _wind_prime(
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
) -> np.ndarray:
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
        wind_prime = _wind_prime(
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
            )
        )
        vertical_rows.append(
            state.mdot
            * (dPi_dx / state.Sigma - state.P * drho_dx / state.rho**2)
        )

    mass_integral = (dx / 6.0) * sum(
        weight * source.mass_rhs for weight, source in zip(weights, source_rows)
    )
    angular_integral = (dx / 6.0) * sum(
        weight * source.angular_rhs for weight, source in zip(weights, source_rows)
    )
    energy_integral = (dx / 6.0) * sum(
        weight * source.energy_rhs for weight, source in zip(weights, source_rows)
    )
    vertical_integral = (dx / 6.0) * sum(
        weight * value for weight, value in zip(weights, vertical_rows)
    )
    scales = params.flux_scales
    mass_row = float((right.mdot - left.mdot - mass_integral) / scales.mdot)
    angular_row = float((right.J - left.J - angular_integral) / scales.angular_flux)
    energy_row = float(
        epsilon[idx + 1]
        - epsilon[idx]
        - energy_integral / scales.energy_flux
    )
    # This is the compatibility row after eliminating the repeated epsilon
    # increment against the energy-conservation row.  The transformed system
    # has the same roots but avoids an almost collinear pair of residuals.
    compatibility_row = float(
        (
            right.mechanical_energy_flux
            - left.mechanical_energy_flux
            + vertical_integral
            - energy_integral
        )
        / scales.energy_flux
    )
    return np.asarray(
        [
            radial_raw / radial_scale,
            params.mass_weight * mass_row,
            params.angular_momentum_weight * angular_row,
            params.energy_flux_weight * energy_row,
            params.energy_balance_weight * compatibility_row,
        ],
        dtype=float,
    )


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
