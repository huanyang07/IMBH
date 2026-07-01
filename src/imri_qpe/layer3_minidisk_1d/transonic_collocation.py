"""Free-boundary collocation solver for an isolated transonic slim disk."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from imri_qpe.constants import C, DEFAULT_KAPPA_ES, DEFAULT_MU_MOL
from imri_qpe.scales import eddington_mdot

from .grid import make_log_grid
from .isolated_slim_solver import IsolatedSlimParams, IsolatedSlimProfile, solve_isolated_slim_disk
from .transonic_local import (
    algebraic_state,
    differential_residual_scales,
    differential_residual,
    entropy_gradient_log,
    scaled_differential_matrix,
    sonic_diagnostics,
    state_partials,
    xi_eff_from_gradient,
)
from .transonic_potential import PaczynskiWiitaPotential


@dataclass(frozen=True)
class TransonicSlimParams:
    """Parameters for the isolated no-wind transonic slim-disk solver."""

    M2_g: float
    Mdot_g_s: float
    alpha: float
    mu_stress: float = 0.0
    stress_factor: float = 1.5
    mu_mol: float = DEFAULT_MU_MOL
    kappa: float = DEFAULT_KAPPA_ES
    gamma_gas: float = 5.0 / 3.0
    R_out_rg: float = 1000.0
    n_nodes: int = 48
    grid_power: float = 1.0
    custom_grid_xi: tuple[float, ...] | None = None
    partial_eps: float = 1.0e-5
    logu_bounds: tuple[float, float] = (np.log(1.0e-2), np.log(1.5 * C))
    logT_bounds: tuple[float, float] = (np.log(1.0e3), np.log(1.0e10))
    R_son_bounds_rg: tuple[float, float] = (2.05, 60.0)
    lambda0_bounds: tuple[float, float] = (0.01, 12.0)
    outer_closure: str = "thin_value"
    outer_match_log_slopes: tuple[float, float] | None = None
    outer_temperature_logT: float | None = None
    outer_entropy_logK: float | None = None
    outer_omega_log_offset: float = 0.0
    interval_residual_form: str = "differential"
    integrated_residual_weighting: str = "none"
    max_nfev: int = 400
    residual_tol: float = 1.0e-5

    def __post_init__(self) -> None:
        if self.M2_g <= 0.0:
            raise ValueError("M2_g must be positive")
        if self.Mdot_g_s <= 0.0:
            raise ValueError("Mdot_g_s must be positive")
        if self.alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        if not 0.0 <= self.mu_stress <= 1.0:
            raise ValueError("mu_stress must be between zero and one")
        if self.stress_factor <= 0.0:
            raise ValueError("stress_factor must be positive")
        if self.mu_mol <= 0.0:
            raise ValueError("mu_mol must be positive")
        if self.kappa <= 0.0:
            raise ValueError("kappa must be positive")
        if self.gamma_gas <= 1.0:
            raise ValueError("gamma_gas must exceed one")
        if self.R_out_rg <= self.R_son_bounds_rg[1]:
            raise ValueError("R_out_rg must exceed the sonic-radius upper bound")
        if self.n_nodes < 4:
            raise ValueError("n_nodes must be at least four")
        if self.grid_power <= 0.0:
            raise ValueError("grid_power must be positive")
        if self.custom_grid_xi is not None:
            xi = np.asarray(self.custom_grid_xi, dtype=float)
            if xi.shape != (self.n_nodes,):
                raise ValueError("custom_grid_xi must have one entry per node")
            if not np.all(np.isfinite(xi)):
                raise ValueError("custom_grid_xi entries must be finite")
            if not np.isclose(xi[0], 0.0) or not np.isclose(xi[-1], 1.0):
                raise ValueError("custom_grid_xi must start at 0 and end at 1")
            if np.any(np.diff(xi) <= 0.0):
                raise ValueError("custom_grid_xi must be strictly increasing")
        if self.partial_eps <= 0.0:
            raise ValueError("partial_eps must be positive")
        if self.logu_bounds[1] <= self.logu_bounds[0]:
            raise ValueError("logu_bounds must be increasing")
        if self.logT_bounds[1] <= self.logT_bounds[0]:
            raise ValueError("logT_bounds must be increasing")
        if self.R_son_bounds_rg[0] <= 2.0 or self.R_son_bounds_rg[1] <= self.R_son_bounds_rg[0]:
            raise ValueError("R_son_bounds_rg must be outside the pseudo-horizon and increasing")
        if self.lambda0_bounds[1] <= self.lambda0_bounds[0]:
            raise ValueError("lambda0_bounds must be increasing")
        if self.outer_closure not in {
            "thin_value",
            "pressure_supported_thin_energy",
            "pressure_supported_temperature",
            "pressure_supported_entropy",
            "matched_outer_state",
            "full_slope_match",
        }:
            raise ValueError(
                "outer_closure must be 'thin_value', 'pressure_supported_thin_energy', "
                "'pressure_supported_temperature', 'pressure_supported_entropy', "
                "'matched_outer_state', or 'full_slope_match'"
            )
        if self.outer_closure == "pressure_supported_temperature":
            if self.outer_temperature_logT is None or not np.isfinite(float(self.outer_temperature_logT)):
                raise ValueError("pressure_supported_temperature requires finite outer_temperature_logT")
        if self.outer_closure == "pressure_supported_entropy":
            if self.outer_entropy_logK is None or not np.isfinite(float(self.outer_entropy_logK)):
                raise ValueError("pressure_supported_entropy requires finite outer_entropy_logK")
        if not np.isfinite(float(self.outer_omega_log_offset)):
            raise ValueError("outer_omega_log_offset must be finite")
        if self.interval_residual_form not in {"differential", "integrated"}:
            raise ValueError("interval_residual_form must be 'differential' or 'integrated'")
        if self.integrated_residual_weighting not in {"none", "inverse_sqrt_dx", "inverse_dx"}:
            raise ValueError("integrated_residual_weighting must be 'none', 'inverse_sqrt_dx', or 'inverse_dx'")
        if self.outer_match_log_slopes is not None:
            if len(self.outer_match_log_slopes) != 2:
                raise ValueError("outer_match_log_slopes must be a pair (dlnu/dlnR, dlnT/dlnR)")
            if not np.all(np.isfinite(np.asarray(self.outer_match_log_slopes, dtype=float))):
                raise ValueError("outer_match_log_slopes must be finite")

    @property
    def potential(self) -> PaczynskiWiitaPotential:
        return PaczynskiWiitaPotential(self.M2_g)

    @property
    def r_g(self) -> float:
        return self.potential.r_g

    @property
    def R_out(self) -> float:
        return self.R_out_rg * self.r_g

    @property
    def mdot_edd_ratio(self) -> float:
        return self.Mdot_g_s / eddington_mdot(self.M2_g, kappa=self.kappa)


@dataclass(frozen=True)
class TransonicSlimProfile:
    """Diagnostic profile from the transonic collocation variables."""

    R: np.ndarray
    u: np.ndarray
    T: np.ndarray
    Sigma: np.ndarray
    H: np.ndarray
    rho: np.ndarray
    P: np.ndarray
    Pi: np.ndarray
    e: np.ndarray
    tau: np.ndarray
    Omega: np.ndarray
    Omega_K: np.ndarray
    l: np.ndarray
    l_K: np.ndarray
    W: np.ndarray
    Q_visc: np.ndarray
    Q_rad: np.ndarray
    Q_adv: np.ndarray
    xi_eff: np.ndarray
    radial_residual: np.ndarray
    energy_residual: np.ndarray
    normalized_energy_residual: np.ndarray
    sonic_D: np.ndarray
    sonic_C1: np.ndarray
    sonic_C2: np.ndarray
    sonic_K: np.ndarray
    sonic_N: np.ndarray
    sonic_smin_over_smax: np.ndarray
    sonic_null_radial_fraction: np.ndarray
    sonic_M_eff: np.ndarray
    H_over_R: np.ndarray
    sonic_radius: float
    l0: float
    lambda0: float
    integrated_advective_fraction: float
    energy_L1: float
    max_abs_residual: float
    sonic_crossings: int


@dataclass(frozen=True)
class TransonicResidualAudit:
    """Residual blocks and physical sanity diagnostics for one state vector."""

    interval_radial_max: float
    interval_radial_l2: float
    interval_energy_max: float
    interval_energy_l2: float
    outer_omega: float
    outer_energy: float
    sonic_D: float
    sonic_C1: float
    sonic_C2: float
    sonic_K: float
    sonic_N: float
    sonic_smin_over_smax: float
    sonic_null_radial_fraction: float
    sonic_M_eff: float
    outer_H_over_R: float
    outer_Qadv_over_Qvisc: float
    lambda0_over_lK_isco: float
    active_bounds: tuple[str, ...]


@dataclass(frozen=True)
class TransonicSolveStatus:
    """Scientific status flags for a transonic solve."""

    optimizer_converged: bool
    optimizer_acceptable: bool
    equations_converged: bool
    sonic_regular: bool
    physically_valid: bool
    active_bounds_clear: bool
    positive_state: bool
    one_sonic_crossing: bool
    thin_limit_ok: bool
    outer_thin: bool


@dataclass(frozen=True)
class TransonicSolveResult:
    """Result from the free-boundary transonic solve."""

    profile: TransonicSlimProfile | None
    converged: bool
    status: TransonicSolveStatus
    residual_audit: TransonicResidualAudit
    cost: float
    max_residual: float
    nfev: int
    njev: int
    optimality: float
    optimizer_status: int
    active_mask: np.ndarray
    message: str
    optimizer_success: bool


@dataclass(frozen=True)
class TransonicHomotopyStageResult:
    """One stage in a low-Mdot staged transonic solve."""

    name: str
    z: np.ndarray
    max_residual: float
    cost: float
    nfev: int
    optimizer_success: bool
    message: str


@dataclass(frozen=True)
class TransonicHomotopyResult:
    """Staged low-Mdot solve and final hardened audit result."""

    stages: tuple[TransonicHomotopyStageResult, ...]
    final_result: TransonicSolveResult
    fixed_R_son: float
    fixed_lambda0: float


@dataclass(frozen=True)
class TransonicJacobianDirectionalAudit:
    """Directional finite-difference check of a sparse residual Jacobian."""

    steps: np.ndarray
    median_relative_error: np.ndarray
    max_relative_error: np.ndarray
    best_step: float
    best_median_error: float
    n_directions: int
    pivot: str


@dataclass(frozen=True)
class TransonicSquarePolishResult:
    """Fixed-Mdot polish result using the square sonic residual system."""

    z: np.ndarray
    pivot: str
    method: str
    result: TransonicSolveResult
    initial_square_max_residual: float
    final_square_max_residual: float
    unused_compatibility: float
    iterations: int
    line_search_reductions: int
    final_step_norm: float
    final_linear_damping: float
    final_merit: float


def computational_grid(params: TransonicSlimParams, logR_son: float) -> np.ndarray:
    """Return collocation node positions in ``ln R``."""

    if params.custom_grid_xi is None:
        xi = np.linspace(0.0, 1.0, params.n_nodes)
        mapped = xi**params.grid_power
    else:
        mapped = np.asarray(params.custom_grid_xi, dtype=float)
    return logR_son + mapped * (np.log(params.R_out) - logR_son)


def pack_state(logu, logT, logR_son: float, lambda0: float) -> np.ndarray:
    """Pack node variables and two eigenparameters into one vector."""

    logu = np.asarray(logu, dtype=float)
    logT = np.asarray(logT, dtype=float)
    if logu.shape != logT.shape:
        raise ValueError("logu and logT must have the same shape")
    return np.concatenate([logu, logT, np.array([logR_son, lambda0], dtype=float)])


def unpack_state(z, params: TransonicSlimParams) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    """Unpack state vector into node arrays and eigenparameters."""

    z = np.asarray(z, dtype=float)
    expected = 2 * params.n_nodes + 2
    if z.shape != (expected,):
        raise ValueError(f"z must have shape ({expected},)")
    logu = z[: params.n_nodes]
    logT = z[params.n_nodes : 2 * params.n_nodes]
    logR_son = float(z[-2])
    lambda0 = float(z[-1])
    logR = computational_grid(params, logR_son)
    return logu, logT, logR_son, lambda0, logR


def _residual_scales(logR: float, y, params: TransonicSlimParams, lambda0: float) -> tuple[float, float]:
    return differential_residual_scales(logR, y, lambda0, params)


def _residual_size(params: TransonicSlimParams) -> int:
    return 2 * params.n_nodes + 3


def _square_residual_size(params: TransonicSlimParams) -> int:
    return 2 * params.n_nodes + 2


def _optimizer_tolerance(params: TransonicSlimParams) -> float:
    return float(min(2.0e-6, max(1.0e-10, 1.0e-2 * params.residual_tol)))


def _outer_thin_boundary_residual(logR: float, y, lambda0: float, params: TransonicSlimParams) -> np.ndarray:
    state = algebraic_state(logR, float(y[0]), float(y[1]), lambda0, params)
    potential = params.potential
    shear = float(potential.dln_omega_k_dlnR(state.R))
    Q_visc_thin = -state.W * state.Omega_K * shear
    B_omega = np.log(state.Omega / state.Omega_K)
    B_energy = (Q_visc_thin - state.Q_rad) / (abs(Q_visc_thin) + abs(state.Q_rad) + 1.0e-300)
    return np.asarray([B_omega, B_energy], dtype=float)


def _outer_log_slope(values: np.ndarray, radii: np.ndarray, n_fit: int = 8) -> float:
    """Return a smoothed outer ``d ln(value) / d ln(R)`` estimate."""

    values = np.asarray(values, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if values.shape != radii.shape:
        raise ValueError("values and radii must have the same shape")
    if len(values) < 3:
        raise ValueError("at least three points are required for an outer slope")
    count = min(int(n_fit), len(values))
    x = np.log(radii[-count:])
    y = np.log(values[-count:])
    degree = min(2, count - 1)
    coeff = np.polyfit(x - x[-1], y, degree)
    return float(np.polyder(np.poly1d(coeff))(0.0))


def reduced_outer_log_slopes(params: TransonicSlimParams, lambda0: float, n_grid: int | None = None) -> tuple[float, float]:
    """Return reduced-solver outer slopes ``(dlnu/dlnR, dlnT/dlnR)``.

    The reduced solve uses the same accretion rate, opacity, thermodynamics,
    and the transonic eigenvalue as its inner angular-momentum constant.  The
    slope is measured from a small polynomial fit over the outer annulus, not
    from a single noisy two-point derivative.
    """

    potential = params.potential
    n_grid = max(int(n_grid or 2 * params.n_nodes), 24)
    inner_candidates = (
        1.08 * potential.r_isco,
        params.R_out * np.exp(-3.0),
        params.R_out * np.exp(-2.0),
        params.R_out * np.exp(-1.0),
    )
    last_message = "not attempted"
    profile = None
    for R_in in inner_candidates:
        R_in = max(float(R_in), 1.08 * potential.r_isco)
        if R_in >= 0.95 * params.R_out:
            continue
        grid = make_log_grid(R_in, params.R_out, n_grid)
        reduced_params = IsolatedSlimParams(
            M2_g=params.M2_g,
            Mdot_g_s=params.Mdot_g_s,
            R_in=R_in,
            alpha=params.alpha,
            mu_mol=params.mu_mol,
            kappa=params.kappa,
            gamma_gas=params.gamma_gas,
            l_in=float(lambda0 * params.r_g * C),
            sigma_brackets=120,
            T_bounds=(np.exp(params.logT_bounds[0]), np.exp(params.logT_bounds[1])),
        )
        result = solve_isolated_slim_disk(grid, reduced_params, max_iter=100, tol=3.0e-3, damping=0.6)
        last_message = result.message
        if result.profile is not None:
            profile = result.profile
            break
    if profile is None:
        raise RuntimeError(f"reduced outer-slope solve failed: {last_message}")
    u = -np.asarray(profile.v_R, dtype=float)
    if np.any(u <= 0.0) or np.any(profile.T <= 0.0):
        raise RuntimeError("reduced outer-slope solve returned non-positive u or T")
    return (
        _outer_log_slope(u, profile.R),
        _outer_log_slope(profile.T, profile.R),
    )


def pressure_supported_omega_target(
    logR: float,
    y,
    g_match: tuple[float, float] | np.ndarray,
    lambda0: float,
    params: TransonicSlimParams,
) -> float:
    """Return the finite-radius target for ``ln(Omega/Omega_K)``.

    The sign follows the implemented radial residual convention
    ``u^2 g_u - R^2 (Omega^2 - Omega_K^2) + dPi/dx/Sigma = 0``.
    For a smooth outer disk with outwardly declining pressure, this target is
    typically negative.  A positive target requires either radial inertia or a
    pressure gradient large enough to make the right-hand side positive.
    """

    g_match = np.asarray(g_match, dtype=float)
    state = algebraic_state(logR, float(y[0]), float(y[1]), lambda0, params)
    partials = state_partials(logR, y, lambda0, params, eps_x=params.partial_eps, eps_y=params.partial_eps)
    dPi_dx = partials.x["Pi"] + float(np.dot(partials.y["Pi"], g_match))
    g_Pi = dPi_dx / (state.Pi + 1.0e-300)
    denom = state.R**2 * state.Omega_K**2 + 1.0e-300
    fractional_omega2 = float((state.u**2 / denom) * g_match[0] + (state.Pi / (state.Sigma * denom)) * g_Pi)
    if fractional_omega2 <= -1.0:
        return -1.0e6
    return float(0.5 * np.log1p(fractional_omega2))


def _outer_pressure_supported_boundary_residual(logR: float, y, lambda0: float, params: TransonicSlimParams) -> np.ndarray:
    thin = _outer_thin_boundary_residual(logR, y, lambda0, params)
    g_match = params.outer_match_log_slopes
    if g_match is None:
        g_match = reduced_outer_log_slopes(params, lambda0)
    target = pressure_supported_omega_target(logR, y, g_match, lambda0, params) + float(params.outer_omega_log_offset)
    return np.asarray([thin[0] - target, thin[1]], dtype=float)


def _pressure_supported_omega_residual(logR: float, y, lambda0: float, params: TransonicSlimParams) -> float:
    thin = _outer_thin_boundary_residual(logR, y, lambda0, params)
    g_match = params.outer_match_log_slopes
    if g_match is None:
        g_match = reduced_outer_log_slopes(params, lambda0)
    target = pressure_supported_omega_target(logR, y, g_match, lambda0, params) + float(params.outer_omega_log_offset)
    return float(thin[0] - target)


def _outer_entropy_proxy(logR: float, y, lambda0: float, params: TransonicSlimParams) -> float:
    state = algebraic_state(logR, float(y[0]), float(y[1]), lambda0, params)
    return float(np.log(state.P + 1.0e-300) - params.gamma_gas * np.log(state.rho + 1.0e-300))


def _outer_pressure_temperature_boundary_residual(logR: float, y, lambda0: float, params: TransonicSlimParams) -> np.ndarray:
    return np.asarray(
        [
            _pressure_supported_omega_residual(logR, y, lambda0, params),
            float(y[1] - float(params.outer_temperature_logT)),
        ],
        dtype=float,
    )


def _outer_pressure_entropy_boundary_residual(logR: float, y, lambda0: float, params: TransonicSlimParams) -> np.ndarray:
    return np.asarray(
        [
            _pressure_supported_omega_residual(logR, y, lambda0, params),
            _outer_entropy_proxy(logR, y, lambda0, params) - float(params.outer_entropy_logK),
        ],
        dtype=float,
    )


def _scaled_local_differential_residual(logR: float, y, g_match, lambda0: float, params: TransonicSlimParams) -> np.ndarray:
    raw = differential_residual(logR, y, g_match, lambda0, params)
    radial_scale, energy_scale = _residual_scales(logR, y, params, lambda0)
    return raw / np.array([radial_scale, energy_scale], dtype=float)


def matched_outer_state(
    logR: float,
    lambda0: float,
    params: TransonicSlimParams,
    *,
    g_match: tuple[float, float] | np.ndarray | None = None,
    initial_y=None,
) -> np.ndarray:
    """Return local full-equation outer match ``[logu, logT]``.

    The supplied slopes are treated as the asymptotic outer-annulus slopes.
    The returned state solves the local radial and energy equations at
    ``R_out`` using those slopes; the global BVP can then impose value matching
    without forcing exact Keplerian rotation at finite radius.
    """

    try:
        from scipy.optimize import least_squares
    except Exception as exc:
        raise RuntimeError("scipy is required for matched_outer_state") from exc

    if g_match is None:
        g_match = params.outer_match_log_slopes
    if g_match is None:
        g_match = reduced_outer_log_slopes(params, lambda0)
    g_match = np.asarray(g_match, dtype=float)
    if g_match.shape != (2,) or not np.all(np.isfinite(g_match)):
        raise ValueError("g_match must be a finite pair")
    lower = np.array([params.logu_bounds[0], params.logT_bounds[0]], dtype=float)
    upper = np.array([params.logu_bounds[1], params.logT_bounds[1]], dtype=float)
    if initial_y is None:
        initial_y = np.array([0.5 * (lower[0] + upper[0]), 0.5 * (lower[1] + upper[1])], dtype=float)
    y0 = np.clip(np.asarray(initial_y, dtype=float), lower + 1.0e-12, upper - 1.0e-12)

    result = least_squares(
        lambda trial: _scaled_local_differential_residual(logR, trial, g_match, lambda0, params),
        y0,
        bounds=(lower, upper),
        x_scale="jac",
        diff_step=3.0e-5,
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-10,
        max_nfev=80,
    )
    return np.asarray(result.x, dtype=float)


def _outer_matched_state_boundary_residual(logR: float, y, lambda0: float, params: TransonicSlimParams) -> np.ndarray:
    g_match = params.outer_match_log_slopes
    if g_match is None:
        g_match = reduced_outer_log_slopes(params, lambda0)
    y = np.asarray(y, dtype=float)
    y_match = matched_outer_state(logR, lambda0, params, g_match=g_match, initial_y=y)
    return y - y_match


def _outer_full_slope_match_boundary_residual(logR: float, y, lambda0: float, params: TransonicSlimParams) -> np.ndarray:
    g_match = params.outer_match_log_slopes
    if g_match is None:
        g_match = reduced_outer_log_slopes(params, lambda0)
    return _scaled_local_differential_residual(logR, y, g_match, lambda0, params)


def _outer_boundary_residual(logR: float, y, lambda0: float, params: TransonicSlimParams) -> np.ndarray:
    if params.outer_closure == "thin_value":
        return _outer_thin_boundary_residual(logR, y, lambda0, params)
    if params.outer_closure == "pressure_supported_thin_energy":
        return _outer_pressure_supported_boundary_residual(logR, y, lambda0, params)
    if params.outer_closure == "pressure_supported_temperature":
        return _outer_pressure_temperature_boundary_residual(logR, y, lambda0, params)
    if params.outer_closure == "pressure_supported_entropy":
        return _outer_pressure_entropy_boundary_residual(logR, y, lambda0, params)
    if params.outer_closure == "matched_outer_state":
        return _outer_matched_state_boundary_residual(logR, y, lambda0, params)
    if params.outer_closure == "full_slope_match":
        return _outer_full_slope_match_boundary_residual(logR, y, lambda0, params)
    raise ValueError(f"unknown outer_closure {params.outer_closure!r}")


def _heating_terms_from_gradient(logR: float, y, g, lambda0: float, params: TransonicSlimParams) -> tuple[float, float, float, float]:
    """Return ``Q_visc, Q_rad, Q_adv, energy_residual`` at one local state."""

    state = algebraic_state(logR, float(y[0]), float(y[1]), lambda0, params)
    partials = state_partials(logR, y, lambda0, params, eps_x=params.partial_eps, eps_y=params.partial_eps)
    dOmega_dx = partials.x["Omega"] + float(np.dot(partials.y["Omega"], g))
    Tdsdx = entropy_gradient_log(logR, y, g, lambda0, params)
    Q_visc = -state.W * dOmega_dx
    Q_adv = -(state.Sigma * state.u / state.R) * Tdsdx
    energy = Q_visc - state.Q_rad - Q_adv
    return Q_visc, state.Q_rad, Q_adv, energy


def _interval_geometry(logu, logT, logR, idx: int) -> tuple[float, np.ndarray, np.ndarray, float]:
    dx = logR[idx + 1] - logR[idx]
    y_left = np.array([logu[idx], logT[idx]], dtype=float)
    y_right = np.array([logu[idx + 1], logT[idx + 1]], dtype=float)
    ym = 0.5 * (y_left + y_right)
    xm = 0.5 * (logR[idx] + logR[idx + 1])
    return float(dx), y_left, y_right, float(xm)


def _differential_interval_residual_from_unpacked(logu, logT, logR, lambda0: float, params: TransonicSlimParams, idx: int) -> np.ndarray:
    """Return the scaled differential residual for one midpoint interval."""

    dx, y_left, y_right, xm = _interval_geometry(logu, logT, logR, idx)
    ym = 0.5 * (y_left + y_right)
    gm = (y_right - y_left) / dx
    raw = differential_residual(xm, ym, gm, lambda0, params)
    radial_scale, energy_scale = _residual_scales(xm, ym, params, lambda0)
    return raw / np.array([radial_scale, energy_scale])


def _integrated_interval_residual_from_unpacked(logu, logT, logR, lambda0: float, params: TransonicSlimParams, idx: int) -> np.ndarray:
    """Return the integrated collocation defect for one midpoint interval."""

    dx, y_left, y_right, xm = _interval_geometry(logu, logT, logR, idx)
    ym = 0.5 * (y_left + y_right)
    Abar, cbar, _radial_scale, _energy_scale = scaled_differential_matrix(xm, ym, lambda0, params)
    residual = Abar @ (y_right - y_left) + dx * cbar
    if params.integrated_residual_weighting == "none":
        return residual
    if params.integrated_residual_weighting == "inverse_sqrt_dx":
        return residual / np.sqrt(dx)
    if params.integrated_residual_weighting == "inverse_dx":
        return residual / dx
    raise ValueError(f"unknown integrated_residual_weighting {params.integrated_residual_weighting!r}")


def _interval_residual_from_unpacked(logu, logT, logR, lambda0: float, params: TransonicSlimParams, idx: int) -> np.ndarray:
    """Return the configured residual for one midpoint collocation interval."""

    if params.interval_residual_form == "differential":
        return _differential_interval_residual_from_unpacked(logu, logT, logR, lambda0, params, idx)
    if params.interval_residual_form == "integrated":
        return _integrated_interval_residual_from_unpacked(logu, logT, logR, lambda0, params, idx)
    raise ValueError(f"unknown interval_residual_form {params.interval_residual_form!r}")


def _differential_interval_residuals_from_unpacked(logu, logT, logR, lambda0: float, params: TransonicSlimParams) -> np.ndarray:
    """Return physical differential residuals independent of solver scaling."""

    return np.asarray(
        [
            _differential_interval_residual_from_unpacked(logu, logT, logR, lambda0, params, idx)
            for idx in range(len(logR) - 1)
        ],
        dtype=float,
    )


def _interval_residual_block(z, params: TransonicSlimParams, idx: int) -> np.ndarray:
    try:
        logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
        if np.any(np.diff(logR) <= 0.0):
            raise ValueError("mapped radius must increase")
        return _interval_residual_from_unpacked(logu, logT, logR, lambda0, params, idx)
    except Exception:
        return np.full(2, 1.0e6)


def _outer_residual_block(z, params: TransonicSlimParams) -> np.ndarray:
    try:
        logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
        return _outer_boundary_residual(
            logR[-1],
            np.array([logu[-1], logT[-1]]),
            lambda0,
            params,
        )
    except Exception:
        return np.full(2, 1.0e6)


def _sonic_residual_block(z, params: TransonicSlimParams) -> np.ndarray:
    try:
        logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
        sonic = sonic_diagnostics(logR[0], np.array([logu[0], logT[0]]), lambda0, params)
        return np.array([sonic.D, sonic.C1, sonic.C2], dtype=float)
    except Exception:
        return np.full(3, 1.0e6)


def select_sonic_compatibility_pivot(z, params: TransonicSlimParams) -> str:
    """Choose the better-conditioned sonic compatibility residual.

    The determinant ``D`` is always kept.  Only one compatibility equation is
    needed in the square fixed-Mdot system.  The pivot rule follows the column
    norm of the scaled local differential matrix at the sonic point and should
    be frozen for a Newton/corrector step.
    """

    try:
        logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
        A, _rhs, _radial_scale, _energy_scale = scaled_differential_matrix(
            logR[0],
            np.array([logu[0], logT[0]]),
            lambda0,
            params,
        )
        col0_norm2 = float(A[0, 0] ** 2 + A[1, 0] ** 2)
        col1_norm2 = float(A[0, 1] ** 2 + A[1, 1] ** 2)
        return "C2" if col0_norm2 >= col1_norm2 else "C1"
    except Exception:
        return "C1"


def _resolve_sonic_pivot(z, params: TransonicSlimParams, pivot: str) -> str:
    if pivot == "auto":
        return select_sonic_compatibility_pivot(z, params)
    if pivot == "svd":
        return "K"
    if pivot not in {"C1", "C2", "K"}:
        raise ValueError("sonic compatibility pivot must be 'auto', 'svd', 'K', 'C1', or 'C2'")
    return pivot


def sonic_residual_pair(z, params: TransonicSlimParams, pivot: str = "auto") -> np.ndarray:
    """Return the square sonic residual pair ``[D, C_selected]``."""

    resolved = _resolve_sonic_pivot(z, params, pivot)
    try:
        D, C1, C2 = _sonic_residual_block(z, params)
        if resolved == "K":
            return _sonic_component_values(z, params, ("D", "K"))
        return np.array([D, C1 if resolved == "C1" else C2], dtype=float)
    except Exception:
        return np.full(2, 1.0e6)


def unused_sonic_compatibility(z, params: TransonicSlimParams, pivot: str = "auto") -> float:
    """Return the compatibility residual not used by ``sonic_residual_pair``."""

    resolved = _resolve_sonic_pivot(z, params, pivot)
    try:
        _D, C1, C2 = _sonic_residual_block(z, params)
        if resolved == "K":
            return float(max(abs(C1), abs(C2)))
        return float(C2 if resolved == "C1" else C1)
    except Exception:
        return 1.0e6


def _sonic_component_values(z, params: TransonicSlimParams, components: tuple[str, ...]) -> np.ndarray:
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    sonic = sonic_diagnostics(logR[0], np.array([logu[0], logT[0]]), lambda0, params)
    values = {
        "D": sonic.D,
        "C1": sonic.C1,
        "C2": sonic.C2,
        "K": sonic.compatibility,
    }
    return np.asarray([values[name] for name in components], dtype=float)


def _sonic_component_values_local(local, params: TransonicSlimParams, components: tuple[str, ...]) -> np.ndarray:
    local = np.asarray(local, dtype=float)
    sonic = sonic_diagnostics(float(local[2]), np.array([local[0], local[1]]), float(local[3]), params)
    values = {
        "D": sonic.D,
        "C1": sonic.C1,
        "C2": sonic.C2,
        "K": sonic.compatibility,
    }
    return np.asarray([values[name] for name in components], dtype=float)


def _richardson_finite_difference_column_vector(
    block_func,
    x,
    column: int,
    lower,
    upper,
    rel_step: float = 1.0e-6,
    base=None,
) -> np.ndarray:
    if base is None:
        base = block_func(x)
    value = float(x[column])
    step = rel_step * max(1.0, abs(value))
    step = min(step, 0.25 * max(upper[column] - lower[column], 1.0e-300))
    if step <= 0.0:
        return np.zeros_like(base)

    def central(width: float):
        if value - width < lower[column] or value + width > upper[column]:
            return None
        plus = np.array(x, copy=True)
        minus = np.array(x, copy=True)
        plus[column] += width
        minus[column] -= width
        return (block_func(plus) - block_func(minus)) / (2.0 * width)

    full = central(step)
    half = central(0.5 * step)
    if full is not None and half is not None:
        return (4.0 * half - full) / 3.0
    if half is not None:
        return half
    if full is not None:
        return full
    if value + step <= upper[column]:
        plus = np.array(x, copy=True)
        plus[column] += step
        half_plus = np.array(x, copy=True)
        half_plus[column] += 0.5 * step
        first = (block_func(plus) - base) / step
        second = (block_func(half_plus) - base) / (0.5 * step)
        return 2.0 * second - first
    if value - step >= lower[column]:
        minus = np.array(x, copy=True)
        minus[column] -= step
        half_minus = np.array(x, copy=True)
        half_minus[column] -= 0.5 * step
        first = (base - block_func(minus)) / step
        second = (base - block_func(half_minus)) / (0.5 * step)
        return 2.0 * second - first
    return np.zeros_like(base)


def sonic_residual_jacobian(
    z,
    params: TransonicSlimParams,
    *,
    components: tuple[str, ...] = ("D", "C1", "C2"),
    rel_step: float = 1.0e-6,
) -> np.ndarray:
    """Return the local sonic-residual Jacobian in ``[logu0, logT0, logR_son, lambda0]``."""

    if rel_step <= 0.0:
        raise ValueError("rel_step must be positive")
    logu, logT, logR_son, lambda0, _logR = unpack_state(z, params)
    lower, upper = state_bounds(params)
    local = np.array([logu[0], logT[0], logR_son, lambda0], dtype=float)
    lower_local = np.array([lower[0], lower[params.n_nodes], lower[-2], lower[-1]], dtype=float)
    upper_local = np.array([upper[0], upper[params.n_nodes], upper[-2], upper[-1]], dtype=float)
    block_func = lambda trial: _sonic_component_values_local(trial, params, components)
    base = block_func(local)
    jac = np.zeros((len(components), 4), dtype=float)
    for col in range(4):
        jac[:, col] = _richardson_finite_difference_column_vector(
            block_func,
            local,
            col,
            lower_local,
            upper_local,
            rel_step=rel_step,
            base=base,
        )
    return jac


def collocation_residual(z, params: TransonicSlimParams) -> np.ndarray:
    """Return the scaled free-boundary collocation residual."""

    return _collocation_residual_weighted(z, params)


def square_collocation_residual(z, params: TransonicSlimParams, pivot: str = "auto") -> np.ndarray:
    """Return the square free-boundary residual using two sonic equations."""

    resolved = _resolve_sonic_pivot(z, params, pivot)
    residual = np.zeros(_square_residual_size(params), dtype=float)
    try:
        logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
        if np.any(np.diff(logR) <= 0.0):
            raise ValueError("mapped radius must increase")
        row = 0
        for idx in range(params.n_nodes - 1):
            residual[row : row + 2] = _interval_residual_from_unpacked(logu, logT, logR, lambda0, params, idx)
            row += 2

        residual[row : row + 2] = _outer_residual_block(z, params)
        row += 2
        residual[row : row + 2] = sonic_residual_pair(z, params, pivot=resolved)
    except Exception:
        residual.fill(1.0e6)
    return residual


def _collocation_residual_weighted(
    z,
    params: TransonicSlimParams,
    outer_weight: float = 1.0,
    sonic_weight: float = 1.0,
) -> np.ndarray:
    """Return collocation residual with optional outer/sonic row weights."""

    residual = np.zeros(_residual_size(params), dtype=float)
    try:
        if outer_weight <= 0.0 or sonic_weight <= 0.0:
            raise ValueError("residual weights must be positive")
        logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
        if np.any(np.diff(logR) <= 0.0):
            raise ValueError("mapped radius must increase")
        row = 0
        for idx in range(params.n_nodes - 1):
            residual[row : row + 2] = _interval_residual_from_unpacked(logu, logT, logR, lambda0, params, idx)
            row += 2

        residual[row : row + 2] = outer_weight * _outer_residual_block(z, params)
        row += 2
        residual[row : row + 3] = sonic_weight * _sonic_residual_block(z, params)
    except Exception:
        residual.fill(1.0e6)
    return residual


def _profile_residual_from_unknowns(profile_unknowns, params: TransonicSlimParams, logR_son: float, lambda0: float) -> np.ndarray:
    residual = np.zeros(2 * params.n_nodes, dtype=float)
    try:
        logu = np.asarray(profile_unknowns[: params.n_nodes], dtype=float)
        logT = np.asarray(profile_unknowns[params.n_nodes : 2 * params.n_nodes], dtype=float)
        logR = computational_grid(params, logR_son)
        if np.any(np.diff(logR) <= 0.0):
            raise ValueError("mapped radius must increase")
        row = 0
        for idx in range(params.n_nodes - 1):
            residual[row : row + 2] = _interval_residual_from_unpacked(logu, logT, logR, lambda0, params, idx)
            row += 2
        residual[row : row + 2] = _outer_boundary_residual(logR[-1], np.array([logu[-1], logT[-1]]), lambda0, params)
    except Exception:
        residual.fill(1.0e6)
    return residual


def _profile_interval_residual_block(profile_unknowns, params: TransonicSlimParams, logR_son: float, lambda0: float, idx: int) -> np.ndarray:
    try:
        logu = np.asarray(profile_unknowns[: params.n_nodes], dtype=float)
        logT = np.asarray(profile_unknowns[params.n_nodes : 2 * params.n_nodes], dtype=float)
        logR = computational_grid(params, logR_son)
        return _interval_residual_from_unpacked(logu, logT, logR, lambda0, params, idx)
    except Exception:
        return np.full(2, 1.0e6)


def _profile_outer_residual_block(profile_unknowns, params: TransonicSlimParams, logR_son: float, lambda0: float) -> np.ndarray:
    try:
        logu = np.asarray(profile_unknowns[: params.n_nodes], dtype=float)
        logT = np.asarray(profile_unknowns[params.n_nodes : 2 * params.n_nodes], dtype=float)
        logR = computational_grid(params, logR_son)
        return _outer_boundary_residual(logR[-1], np.array([logu[-1], logT[-1]]), lambda0, params)
    except Exception:
        return np.full(2, 1.0e6)


def _pack_profile_unknowns_to_state(profile_unknowns, params: TransonicSlimParams, logR_son: float, lambda0: float) -> np.ndarray:
    logu = np.asarray(profile_unknowns[: params.n_nodes], dtype=float)
    logT = np.asarray(profile_unknowns[params.n_nodes : 2 * params.n_nodes], dtype=float)
    return pack_state(logu, logT, logR_son, lambda0)


def _profile_unknown_bounds(params: TransonicSlimParams) -> tuple[np.ndarray, np.ndarray]:
    lower = np.concatenate([np.full(params.n_nodes, params.logu_bounds[0]), np.full(params.n_nodes, params.logT_bounds[0])])
    upper = np.concatenate([np.full(params.n_nodes, params.logu_bounds[1]), np.full(params.n_nodes, params.logT_bounds[1])])
    return lower, upper


def _profile_jac_sparsity_pattern(params: TransonicSlimParams):
    try:
        from scipy.sparse import lil_matrix
    except Exception:
        return None

    n_unknown = 2 * params.n_nodes
    pattern = lil_matrix((2 * params.n_nodes, n_unknown), dtype=int)
    row = 0
    for idx in range(params.n_nodes - 1):
        for col in (idx, idx + 1, params.n_nodes + idx, params.n_nodes + idx + 1):
            pattern[row : row + 2, col] = 1
        row += 2
    for col in (params.n_nodes - 1, 2 * params.n_nodes - 1):
        pattern[row : row + 2, col] = 1
    return pattern.tocsr()


def _finite_difference_column_vector(block_func, x, column: int, lower, upper, rel_step: float = 1.0e-6, base=None) -> np.ndarray:
    if base is None:
        base = block_func(x)
    value = float(x[column])
    step = rel_step * max(1.0, abs(value))
    step = min(step, 0.25 * max(upper[column] - lower[column], 1.0e-300))
    if step <= 0.0:
        return np.zeros_like(base)

    if value - step >= lower[column] and value + step <= upper[column]:
        plus = np.array(x, copy=True)
        minus = np.array(x, copy=True)
        plus[column] += step
        minus[column] -= step
        return (block_func(plus) - block_func(minus)) / (2.0 * step)
    if value + step <= upper[column]:
        plus = np.array(x, copy=True)
        plus[column] += step
        return (block_func(plus) - base) / step
    if value - step >= lower[column]:
        minus = np.array(x, copy=True)
        minus[column] -= step
        return (base - block_func(minus)) / step
    return np.zeros_like(base)


def _profile_jacobian_from_unknowns(profile_unknowns, params: TransonicSlimParams, logR_son: float, lambda0: float, rel_step: float = 1.0e-6):
    try:
        from scipy.sparse import lil_matrix
    except Exception as exc:
        raise RuntimeError("scipy is required for profile jacobian") from exc

    x = np.asarray(profile_unknowns, dtype=float)
    lower, upper = _profile_unknown_bounds(params)
    jac = lil_matrix((2 * params.n_nodes, 2 * params.n_nodes), dtype=float)
    row = 0
    for idx in range(params.n_nodes - 1):
        columns = (idx, idx + 1, params.n_nodes + idx, params.n_nodes + idx + 1)
        block_func = lambda trial, interval_idx=idx: _profile_interval_residual_block(trial, params, logR_son, lambda0, interval_idx)
        base = block_func(x)
        for col in columns:
            jac[row : row + 2, col] = _finite_difference_column_vector(block_func, x, col, lower, upper, rel_step, base=base)[:, None]
        row += 2
    block_func = lambda trial: _profile_outer_residual_block(trial, params, logR_son, lambda0)
    base = block_func(x)
    for col in (params.n_nodes - 1, 2 * params.n_nodes - 1):
        jac[row : row + 2, col] = _finite_difference_column_vector(block_func, x, col, lower, upper, rel_step, base=base)[:, None]
    return jac.tocsr()


def _free_rson_residual_from_unknowns(unknowns, params: TransonicSlimParams, lambda0: float, sonic_components: tuple[str, ...], sonic_weight: float = 1.0) -> np.ndarray:
    profile_unknowns = np.asarray(unknowns[: 2 * params.n_nodes], dtype=float)
    logR_son = float(unknowns[-1])
    profile_residual = _profile_residual_from_unknowns(profile_unknowns, params, logR_son, lambda0)
    z = _pack_profile_unknowns_to_state(profile_unknowns, params, logR_son, lambda0)
    try:
        sonic = sonic_weight * _sonic_component_values(z, params, sonic_components)
    except Exception:
        sonic = np.full(len(sonic_components), 1.0e6)
    return np.concatenate([profile_residual, sonic])


def _free_rson_interval_residual_block(unknowns, params: TransonicSlimParams, lambda0: float, idx: int) -> np.ndarray:
    try:
        profile_unknowns = np.asarray(unknowns[: 2 * params.n_nodes], dtype=float)
        logR_son = float(unknowns[-1])
        logu = np.asarray(profile_unknowns[: params.n_nodes], dtype=float)
        logT = np.asarray(profile_unknowns[params.n_nodes : 2 * params.n_nodes], dtype=float)
        logR = computational_grid(params, logR_son)
        return _interval_residual_from_unpacked(logu, logT, logR, lambda0, params, idx)
    except Exception:
        return np.full(2, 1.0e6)


def _free_rson_outer_residual_block(unknowns, params: TransonicSlimParams, lambda0: float) -> np.ndarray:
    try:
        profile_unknowns = np.asarray(unknowns[: 2 * params.n_nodes], dtype=float)
        logR_son = float(unknowns[-1])
        logu = np.asarray(profile_unknowns[: params.n_nodes], dtype=float)
        logT = np.asarray(profile_unknowns[params.n_nodes : 2 * params.n_nodes], dtype=float)
        logR = computational_grid(params, logR_son)
        return _outer_boundary_residual(logR[-1], np.array([logu[-1], logT[-1]]), lambda0, params)
    except Exception:
        return np.full(2, 1.0e6)


def _free_rson_sonic_residual_block(unknowns, params: TransonicSlimParams, lambda0: float, sonic_components: tuple[str, ...], sonic_weight: float = 1.0) -> np.ndarray:
    try:
        profile_unknowns = np.asarray(unknowns[: 2 * params.n_nodes], dtype=float)
        logR_son = float(unknowns[-1])
        z = _pack_profile_unknowns_to_state(profile_unknowns, params, logR_son, lambda0)
        return sonic_weight * _sonic_component_values(z, params, sonic_components)
    except Exception:
        return np.full(len(sonic_components), 1.0e6)


def _determinant_only_state_residual(z, params: TransonicSlimParams, sonic_weight: float = 1.0) -> np.ndarray:
    try:
        logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
        residual = np.zeros(2 * params.n_nodes + 1, dtype=float)
        row = 0
        for idx in range(params.n_nodes - 1):
            residual[row : row + 2] = _interval_residual_from_unpacked(logu, logT, logR, lambda0, params, idx)
            row += 2
        residual[row : row + 2] = _outer_boundary_residual(logR[-1], np.array([logu[-1], logT[-1]]), lambda0, params)
        row += 2
        residual[row] = sonic_weight * sonic_diagnostics(logR[0], np.array([logu[0], logT[0]]), lambda0, params).D
    except Exception:
        residual = np.full(2 * params.n_nodes + 1, 1.0e6, dtype=float)
    return residual


def _determinant_only_sparsity_pattern(params: TransonicSlimParams):
    try:
        from scipy.sparse import lil_matrix
    except Exception:
        return None

    unknown_size = 2 * params.n_nodes + 2
    pattern = lil_matrix((2 * params.n_nodes + 1, unknown_size), dtype=int)
    row = 0
    for idx in range(params.n_nodes - 1):
        columns = (
            idx,
            idx + 1,
            params.n_nodes + idx,
            params.n_nodes + idx + 1,
            unknown_size - 2,
            unknown_size - 1,
        )
        for col in columns:
            pattern[row : row + 2, col] = 1
        row += 2
    for col in (params.n_nodes - 1, 2 * params.n_nodes - 1, unknown_size - 2, unknown_size - 1):
        pattern[row : row + 2, col] = 1
    row += 2
    for col in (0, params.n_nodes, unknown_size - 2, unknown_size - 1):
        pattern[row, col] = 1
    return pattern.tocsr()


def _free_rson_jac_sparsity_pattern(params: TransonicSlimParams, n_sonic_components: int):
    try:
        from scipy.sparse import lil_matrix
    except Exception:
        return None

    n_unknown = 2 * params.n_nodes + 1
    n_residual = 2 * params.n_nodes + n_sonic_components
    pattern = lil_matrix((n_residual, n_unknown), dtype=int)
    row = 0
    logR_col = n_unknown - 1
    for idx in range(params.n_nodes - 1):
        for col in (idx, idx + 1, params.n_nodes + idx, params.n_nodes + idx + 1, logR_col):
            pattern[row : row + 2, col] = 1
        row += 2
    for col in (params.n_nodes - 1, 2 * params.n_nodes - 1, logR_col):
        pattern[row : row + 2, col] = 1
    row += 2
    for col in (0, params.n_nodes, logR_col):
        pattern[row : row + n_sonic_components, col] = 1
    return pattern.tocsr()


def _free_rson_jacobian_from_unknowns(
    unknowns,
    params: TransonicSlimParams,
    lambda0: float,
    sonic_components: tuple[str, ...],
    sonic_weight: float = 1.0,
    rel_step: float = 1.0e-6,
):
    try:
        from scipy.sparse import lil_matrix
    except Exception as exc:
        raise RuntimeError("scipy is required for free-rson jacobian") from exc

    x = np.asarray(unknowns, dtype=float)
    profile_lower, profile_upper = _profile_unknown_bounds(params)
    lower_state, upper_state = state_bounds(params)
    lower = np.concatenate([profile_lower, np.array([lower_state[-2]])])
    upper = np.concatenate([profile_upper, np.array([upper_state[-2]])])
    n_unknown = 2 * params.n_nodes + 1
    n_sonic = len(sonic_components)
    jac = lil_matrix((2 * params.n_nodes + n_sonic, n_unknown), dtype=float)
    logR_col = n_unknown - 1
    row = 0
    for idx in range(params.n_nodes - 1):
        columns = (idx, idx + 1, params.n_nodes + idx, params.n_nodes + idx + 1, logR_col)
        block_func = lambda trial, interval_idx=idx: _free_rson_interval_residual_block(trial, params, lambda0, interval_idx)
        base = block_func(x)
        for col in columns:
            jac[row : row + 2, col] = _finite_difference_column_vector(block_func, x, col, lower, upper, rel_step, base=base)[:, None]
        row += 2
    block_func = lambda trial: _free_rson_outer_residual_block(trial, params, lambda0)
    base = block_func(x)
    for col in (params.n_nodes - 1, 2 * params.n_nodes - 1, logR_col):
        jac[row : row + 2, col] = _finite_difference_column_vector(block_func, x, col, lower, upper, rel_step, base=base)[:, None]
    row += 2
    block_func = lambda trial: _free_rson_sonic_residual_block(trial, params, lambda0, sonic_components, sonic_weight)
    base = block_func(x)
    for col in (0, params.n_nodes, logR_col):
        jac[row : row + n_sonic, col] = _finite_difference_column_vector(block_func, x, col, lower, upper, rel_step, base=base)[:, None]
    return jac.tocsr()


def state_bounds(params: TransonicSlimParams) -> tuple[np.ndarray, np.ndarray]:
    """Return lower and upper bounds for the nonlinear unknowns."""

    lower = np.concatenate(
        [
            np.full(params.n_nodes, params.logu_bounds[0]),
            np.full(params.n_nodes, params.logT_bounds[0]),
            np.array([np.log(params.R_son_bounds_rg[0] * params.r_g), params.lambda0_bounds[0]]),
        ]
    )
    upper = np.concatenate(
        [
            np.full(params.n_nodes, params.logu_bounds[1]),
            np.full(params.n_nodes, params.logT_bounds[1]),
            np.array([np.log(params.R_son_bounds_rg[1] * params.r_g), params.lambda0_bounds[1]]),
        ]
    )
    return lower, upper


def jac_sparsity_pattern(params: TransonicSlimParams):
    """Return the block-banded sparsity pattern for ``least_squares``."""

    try:
        from scipy.sparse import lil_matrix
    except Exception:
        return None

    unknown_size = 2 * params.n_nodes + 2
    pattern = lil_matrix((_residual_size(params), unknown_size), dtype=int)
    row = 0
    for idx in range(params.n_nodes - 1):
        columns = [
            idx,
            idx + 1,
            params.n_nodes + idx,
            params.n_nodes + idx + 1,
            unknown_size - 2,
            unknown_size - 1,
        ]
        for col in columns:
            pattern[row : row + 2, col] = 1
        row += 2
    for col in (params.n_nodes - 1, 2 * params.n_nodes - 1, unknown_size - 2, unknown_size - 1):
        pattern[row : row + 2, col] = 1
    row += 2
    for col in (0, params.n_nodes, unknown_size - 2, unknown_size - 1):
        pattern[row : row + 3, col] = 1
    return pattern.tocsr()


def square_jac_sparsity_pattern(params: TransonicSlimParams):
    """Return the block-banded sparsity pattern for the square residual."""

    try:
        from scipy.sparse import lil_matrix
    except Exception:
        return None

    unknown_size = 2 * params.n_nodes + 2
    pattern = lil_matrix((_square_residual_size(params), unknown_size), dtype=int)
    row = 0
    for idx in range(params.n_nodes - 1):
        columns = [
            idx,
            idx + 1,
            params.n_nodes + idx,
            params.n_nodes + idx + 1,
            unknown_size - 2,
            unknown_size - 1,
        ]
        for col in columns:
            pattern[row : row + 2, col] = 1
        row += 2
    for col in (params.n_nodes - 1, 2 * params.n_nodes - 1, unknown_size - 2, unknown_size - 1):
        pattern[row : row + 2, col] = 1
    row += 2
    for col in (0, params.n_nodes, unknown_size - 2, unknown_size - 1):
        pattern[row : row + 2, col] = 1
    return pattern.tocsr()


def _finite_difference_column(block_func, z, params: TransonicSlimParams, column: int, lower, upper, rel_step: float, base=None) -> np.ndarray:
    """Return one block-Jacobian column with bound-aware finite differences."""

    if base is None:
        base = block_func(z, params)
    value = float(z[column])
    step = rel_step * max(1.0, abs(value))
    step = min(step, 0.25 * max(upper[column] - lower[column], 1.0e-300))
    if step <= 0.0:
        return np.zeros_like(base)

    if value - step >= lower[column] and value + step <= upper[column]:
        plus = np.array(z, copy=True)
        minus = np.array(z, copy=True)
        plus[column] += step
        minus[column] -= step
        return (block_func(plus, params) - block_func(minus, params)) / (2.0 * step)
    if value + step <= upper[column]:
        plus = np.array(z, copy=True)
        plus[column] += step
        return (block_func(plus, params) - base) / step
    if value - step >= lower[column]:
        minus = np.array(z, copy=True)
        minus[column] -= step
        return (base - block_func(minus, params)) / step
    return np.zeros_like(base)


def collocation_jacobian(z, params: TransonicSlimParams, rel_step: float = 1.0e-6):
    """Return a block-local sparse finite-difference Jacobian.

    SciPy's sparse finite-difference Jacobian still calls the full residual
    repeatedly. This routine perturbs the same unknowns but evaluates only the
    residual block that depends on each column.
    """

    try:
        from scipy.sparse import lil_matrix
    except Exception as exc:
        raise RuntimeError("scipy is required for collocation_jacobian") from exc

    if rel_step <= 0.0:
        raise ValueError("rel_step must be positive")
    z = np.asarray(z, dtype=float)
    lower, upper = state_bounds(params)
    unknown_size = 2 * params.n_nodes + 2
    jac = lil_matrix((_residual_size(params), unknown_size), dtype=float)

    row = 0
    for idx in range(params.n_nodes - 1):
        columns = [
            idx,
            idx + 1,
            params.n_nodes + idx,
            params.n_nodes + idx + 1,
            unknown_size - 2,
            unknown_size - 1,
        ]
        block_func = lambda trial, p, interval_idx=idx: _interval_residual_block(trial, p, interval_idx)
        base = block_func(z, params)
        for col in columns:
            jac[row : row + 2, col] = _finite_difference_column(block_func, z, params, col, lower, upper, rel_step, base=base)[:, None]
        row += 2

    outer_base = _outer_residual_block(z, params)
    for col in (params.n_nodes - 1, 2 * params.n_nodes - 1, unknown_size - 2, unknown_size - 1):
        jac[row : row + 2, col] = _finite_difference_column(_outer_residual_block, z, params, col, lower, upper, rel_step, base=outer_base)[:, None]
    row += 2
    sonic_jac = sonic_residual_jacobian(z, params, rel_step=min(rel_step, 1.0e-6))
    for local_col, col in enumerate((0, params.n_nodes, unknown_size - 2, unknown_size - 1)):
        jac[row : row + 3, col] = sonic_jac[:, local_col][:, None]

    return jac.tocsr()


def square_collocation_jacobian(z, params: TransonicSlimParams, pivot: str = "auto", rel_step: float = 3.0e-5):
    """Return a sparse finite-difference Jacobian for ``square_collocation_residual``."""

    try:
        from scipy.sparse import lil_matrix
    except Exception as exc:
        raise RuntimeError("scipy is required for square_collocation_jacobian") from exc

    if rel_step <= 0.0:
        raise ValueError("rel_step must be positive")
    z = np.asarray(z, dtype=float)
    resolved = _resolve_sonic_pivot(z, params, pivot)
    lower, upper = state_bounds(params)
    unknown_size = 2 * params.n_nodes + 2
    jac = lil_matrix((_square_residual_size(params), unknown_size), dtype=float)

    row = 0
    for idx in range(params.n_nodes - 1):
        columns = [
            idx,
            idx + 1,
            params.n_nodes + idx,
            params.n_nodes + idx + 1,
            unknown_size - 2,
            unknown_size - 1,
        ]
        block_func = lambda trial, p, interval_idx=idx: _interval_residual_block(trial, p, interval_idx)
        base = block_func(z, params)
        for col in columns:
            jac[row : row + 2, col] = _finite_difference_column(block_func, z, params, col, lower, upper, rel_step, base=base)[:, None]
        row += 2

    outer_base = _outer_residual_block(z, params)
    for col in (params.n_nodes - 1, 2 * params.n_nodes - 1, unknown_size - 2, unknown_size - 1):
        jac[row : row + 2, col] = _finite_difference_column(_outer_residual_block, z, params, col, lower, upper, rel_step, base=outer_base)[:, None]
    row += 2
    sonic_components = ("D", resolved)
    sonic_jac = sonic_residual_jacobian(z, params, components=sonic_components, rel_step=min(rel_step, 1.0e-6))
    for local_col, col in enumerate((0, params.n_nodes, unknown_size - 2, unknown_size - 1)):
        jac[row : row + 2, col] = sonic_jac[:, local_col][:, None]

    return jac.tocsr()


def _collocation_jacobian_weighted(
    z,
    params: TransonicSlimParams,
    outer_weight: float = 1.0,
    sonic_weight: float = 1.0,
    rel_step: float = 1.0e-6,
):
    """Return the block-local Jacobian with residual row weights applied."""

    if outer_weight <= 0.0 or sonic_weight <= 0.0:
        raise ValueError("residual weights must be positive")
    try:
        from scipy.sparse import diags
    except Exception as exc:
        raise RuntimeError("scipy is required for weighted collocation_jacobian") from exc

    weights = np.ones(_residual_size(params), dtype=float)
    outer_row = 2 * (params.n_nodes - 1)
    weights[outer_row : outer_row + 2] = outer_weight
    weights[outer_row + 2 : outer_row + 5] = sonic_weight
    return diags(weights) @ collocation_jacobian(z, params, rel_step=rel_step)


def jacobian_directional_error(
    z,
    params: TransonicSlimParams,
    *,
    pivot: str = "auto",
    steps: tuple[float, ...] = (1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5, 1.0e-5, 3.0e-6, 1.0e-6),
    n_directions: int = 4,
    seed: int = 1234,
    jacobian_rel_step: float = 3.0e-5,
) -> TransonicJacobianDirectionalAudit:
    """Compare the square sparse Jacobian against directional finite differences."""

    if n_directions <= 0:
        raise ValueError("n_directions must be positive")
    if len(steps) == 0 or any(step <= 0.0 for step in steps):
        raise ValueError("steps must be positive")
    z = np.asarray(z, dtype=float)
    resolved = _resolve_sonic_pivot(z, params, pivot)
    jac = square_collocation_jacobian(z, params, pivot=resolved, rel_step=jacobian_rel_step)
    rng = np.random.default_rng(seed)
    lower, upper = state_bounds(params)
    directions: list[np.ndarray] = []
    for _ in range(n_directions):
        v = rng.normal(size=z.size)
        norm = float(np.linalg.norm(v))
        if norm <= 0.0:
            continue
        v = v / norm
        max_step = np.inf
        positive = v > 0.0
        negative = v < 0.0
        if np.any(positive):
            max_step = min(max_step, float(np.min((upper[positive] - z[positive]) / v[positive])))
        if np.any(negative):
            max_step = min(max_step, float(np.min((lower[negative] - z[negative]) / v[negative])))
        if np.isfinite(max_step) and max_step > 0.0:
            v = v * min(1.0, 0.25 * max_step / max(steps))
        directions.append(v)
    if not directions:
        raise RuntimeError("failed to generate finite-difference directions")

    median_errors: list[float] = []
    max_errors: list[float] = []
    for step in steps:
        errors = []
        for v in directions:
            jv = np.asarray(jac @ v, dtype=float)
            plus = square_collocation_residual(z + step * v, params, pivot=resolved)
            minus = square_collocation_residual(z - step * v, params, pivot=resolved)
            finite = (plus - minus) / (2.0 * step)
            denom = float(np.linalg.norm(jv) + np.linalg.norm(finite) + 1.0e-300)
            errors.append(float(np.linalg.norm(jv - finite) / denom))
        median_errors.append(float(np.median(errors)))
        max_errors.append(float(np.max(errors)))
    median_array = np.asarray(median_errors, dtype=float)
    max_array = np.asarray(max_errors, dtype=float)
    best_index = int(np.argmin(median_array))
    return TransonicJacobianDirectionalAudit(
        steps=np.asarray(steps, dtype=float),
        median_relative_error=median_array,
        max_relative_error=max_array,
        best_step=float(steps[best_index]),
        best_median_error=float(median_array[best_index]),
        n_directions=len(directions),
        pivot=resolved,
    )


def _active_mask_from_bounds(z, lower, upper, tolerance: float = 1.0e-10) -> np.ndarray:
    mask = np.zeros(np.asarray(z).shape, dtype=int)
    mask[np.asarray(z) <= lower + tolerance] = -1
    mask[np.asarray(z) >= upper - tolerance] = 1
    return mask


def _max_alpha_inside_bounds(z, step, lower, upper, safety: float = 0.995) -> float:
    max_alpha = np.inf
    positive = step > 0.0
    negative = step < 0.0
    if np.any(positive):
        max_alpha = min(max_alpha, float(np.min((upper[positive] - z[positive]) / step[positive])))
    if np.any(negative):
        max_alpha = min(max_alpha, float(np.min((lower[negative] - z[negative]) / step[negative])))
    if not np.isfinite(max_alpha):
        return 1.0
    return max(0.0, min(1.0, safety * max_alpha))


def _equilibrated_sparse_newton_step(
    jac,
    residual,
    *,
    damping: float = 0.0,
    use_direct: bool = False,
    solver_tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Solve a sparse Newton correction after row/column equilibration."""

    try:
        from scipy.sparse import diags
        from scipy.sparse.linalg import lsmr, splu
    except Exception as exc:
        raise RuntimeError("scipy is required for sparse Newton polish") from exc

    jac_csr = jac.tocsr()
    row_norm = np.sqrt(np.asarray(jac_csr.multiply(jac_csr).sum(axis=1)).ravel())
    row_scale = 1.0 / np.maximum(row_norm, 1.0e-12)
    row_scaled = diags(row_scale) @ jac_csr
    col_norm = np.sqrt(np.asarray(row_scaled.multiply(row_scaled).sum(axis=0)).ravel())
    col_scale = 1.0 / np.maximum(col_norm, 1.0e-12)
    balanced = (row_scaled @ diags(col_scale)).tocsc()
    rhs = -row_scale * np.asarray(residual, dtype=float)
    if damping < 0.0:
        raise ValueError("linear damping must be non-negative")
    if use_direct and damping == 0.0:
        try:
            y = splu(balanced, permc_spec="COLAMD").solve(rhs)
            return col_scale * np.asarray(y, dtype=float)
        except Exception:
            pass
    y = lsmr(
        balanced,
        rhs,
        damp=float(damping),
        atol=solver_tolerance,
        btol=solver_tolerance,
        maxiter=max(20, 5 * balanced.shape[1]),
    )[0]
    return col_scale * np.asarray(y, dtype=float)


def _regularized_damping_sequence(values: tuple[float, ...]) -> tuple[float, ...]:
    if len(values) == 0:
        raise ValueError("linear_dampings must contain at least one value")
    cleaned = tuple(float(value) for value in values)
    if any(value < 0.0 for value in cleaned):
        raise ValueError("linear dampings must be non-negative")
    return cleaned


def _direct_damping_sequence() -> tuple[float, ...]:
    return (0.0,)


def _linear_solver_uses_direct(linear_solver: str) -> bool:
    if linear_solver == "direct":
        return True
    if linear_solver == "regularized_lsmr":
        return False
    raise ValueError("linear_solver must be 'direct' or 'regularized_lsmr'")


def _linear_damping_candidates(linear_solver: str, linear_dampings: tuple[float, ...]) -> tuple[float, ...]:
    return _direct_damping_sequence() if _linear_solver_uses_direct(linear_solver) else _regularized_damping_sequence(linear_dampings)


def _try_square_newton_step(
    z,
    step,
    residual,
    params: TransonicSlimParams,
    *,
    pivot: str,
    lower,
    upper,
    line_search_min_alpha: float,
    line_search_max_reductions: int,
) -> tuple[bool, np.ndarray, np.ndarray, int, int]:
    merit = _square_residual_merit(residual)
    alpha = _max_alpha_inside_bounds(z, step, lower, upper)
    if alpha <= line_search_min_alpha:
        return False, z, residual, 0, 0
    reductions = 0
    evaluations = 0
    for _ in range(line_search_max_reductions + 1):
        trial = np.clip(z + alpha * step, lower + 1.0e-12, upper - 1.0e-12)
        trial_residual = square_collocation_residual(trial, params, pivot=pivot)
        evaluations += 1
        trial_merit = _square_residual_merit(trial_residual)
        if trial_merit < merit:
            return True, trial, trial_residual, reductions, evaluations
        alpha *= 0.5
        reductions += 1
        if alpha < line_search_min_alpha:
            break
    return False, z, residual, reductions, evaluations


def _square_residual_merit(residual: np.ndarray) -> float:
    return 0.5 * float(np.dot(residual, residual))


def _interp_log_profile(logR_nodes: np.ndarray, R_source: np.ndarray, values: np.ndarray) -> np.ndarray:
    logR_source = np.log(np.asarray(R_source, dtype=float))
    log_values = np.log(np.asarray(values, dtype=float))
    return np.interp(logR_nodes, logR_source, log_values, left=log_values[0], right=log_values[-1])


def _active_bound_names(z, params: TransonicSlimParams, tolerance: float = 1.0e-4) -> tuple[str, ...]:
    lower, upper = state_bounds(params)
    span = np.maximum(upper - lower, 1.0e-300)
    z = np.asarray(z, dtype=float)
    distance = np.minimum((z - lower) / span, (upper - z) / span)
    names: list[str] = []
    for idx in np.flatnonzero(distance < tolerance):
        if idx < params.n_nodes:
            names.append(f"logu[{idx}]")
        elif idx < 2 * params.n_nodes:
            names.append(f"logT[{idx - params.n_nodes}]")
        elif idx == 2 * params.n_nodes:
            names.append("R_son")
        else:
            names.append("lambda0")
    return tuple(names)


def residual_audit_from_state_vector(z, params: TransonicSlimParams) -> TransonicResidualAudit:
    """Return separated residual blocks and physical sanity diagnostics."""

    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    interval = _differential_interval_residuals_from_unpacked(logu, logT, logR, lambda0, params)
    outer = _outer_boundary_residual(logR[-1], np.array([logu[-1], logT[-1]]), lambda0, params)
    sonic = sonic_diagnostics(logR[0], np.array([logu[0], logT[0]]), lambda0, params)

    outer_state = algebraic_state(logR[-1], logu[-1], logT[-1], lambda0, params)
    dx_outer = logR[-1] - logR[-2]
    g_outer = np.array([(logu[-1] - logu[-2]) / dx_outer, (logT[-1] - logT[-2]) / dx_outer])
    q_visc, _q_rad, q_adv, _energy = _heating_terms_from_gradient(logR[-1], np.array([logu[-1], logT[-1]]), g_outer, lambda0, params)
    lambda_k_isco = float(params.potential.l_k(params.potential.r_isco) / (params.r_g * C))

    return TransonicResidualAudit(
        interval_radial_max=float(np.max(np.abs(interval[:, 0]))),
        interval_radial_l2=float(np.sqrt(np.mean(interval[:, 0] ** 2))),
        interval_energy_max=float(np.max(np.abs(interval[:, 1]))),
        interval_energy_l2=float(np.sqrt(np.mean(interval[:, 1] ** 2))),
        outer_omega=float(outer[0]),
        outer_energy=float(outer[1]),
        sonic_D=float(sonic.D),
        sonic_C1=float(sonic.C1),
        sonic_C2=float(sonic.C2),
        sonic_K=float(sonic.compatibility),
        sonic_N=float(sonic.N),
        sonic_smin_over_smax=float(sonic.smin_over_smax),
        sonic_null_radial_fraction=float(sonic.null_radial_fraction),
        sonic_M_eff=float(sonic.M_eff),
        outer_H_over_R=float(outer_state.H_over_R),
        outer_Qadv_over_Qvisc=float(q_adv / (q_visc + 1.0e-300)),
        lambda0_over_lK_isco=float(lambda0 / lambda_k_isco),
        active_bounds=_active_bound_names(z, params),
    )


def _status_from_profile(
    profile: TransonicSlimProfile,
    audit: TransonicResidualAudit,
    params: TransonicSlimParams,
    optimizer_converged: bool,
    max_residual: float,
) -> TransonicSolveStatus:
    tol = float(params.residual_tol)
    equations_converged = bool(
        audit.interval_radial_max <= tol
        and audit.interval_energy_max <= tol
        and abs(audit.outer_omega) <= tol
        and abs(audit.outer_energy) <= tol
    )
    boundary_sonic_point = bool(abs(audit.sonic_D) <= tol and audit.sonic_null_radial_fraction > 0.3)
    one_sonic_crossing = bool(profile.sonic_crossings == 1 or boundary_sonic_point)
    sonic_regular = bool(
        abs(audit.sonic_D) <= tol
        and abs(audit.sonic_K) <= tol
        and audit.sonic_smin_over_smax <= max(tol, 1.0e-8)
        and audit.sonic_null_radial_fraction > 0.3
        and one_sonic_crossing
    )
    active_bounds_clear = len(audit.active_bounds) == 0
    positive_state = bool(
        np.all(profile.Sigma > 0.0)
        and np.all(profile.T > 0.0)
        and np.all(profile.u > 0.0)
        and np.all(np.isfinite(profile.tau))
        and np.all(profile.Q_visc > 0.0)
    )
    thin_branch_check = params.mdot_edd_ratio <= 3.0e-2
    thin_limit_ok = bool((not thin_branch_check) or (0.8 <= audit.lambda0_over_lK_isco <= 1.1))
    outer_thin = bool(audit.outer_H_over_R < 0.05 and abs(audit.outer_omega) < 0.01)
    residual_acceptable = bool(np.isfinite(max_residual) and max_residual <= tol)
    optimizer_acceptable = bool(
        optimizer_converged
        or (
            residual_acceptable
            and equations_converged
            and sonic_regular
            and active_bounds_clear
            and positive_state
            and thin_limit_ok
            and outer_thin
        )
    )
    physically_valid = bool(
        optimizer_acceptable
        and equations_converged
        and sonic_regular
        and active_bounds_clear
        and positive_state
        and thin_limit_ok
        and outer_thin
    )
    return TransonicSolveStatus(
        optimizer_converged=bool(optimizer_converged),
        optimizer_acceptable=optimizer_acceptable,
        equations_converged=equations_converged,
        sonic_regular=sonic_regular,
        physically_valid=physically_valid,
        active_bounds_clear=active_bounds_clear,
        positive_state=positive_state,
        one_sonic_crossing=one_sonic_crossing,
        thin_limit_ok=thin_limit_ok,
        outer_thin=outer_thin,
    )


def initial_guess_from_reduced_solver(
    reduced_profile: IsolatedSlimProfile,
    params: TransonicSlimParams,
    R_son: float | None = None,
    lambda0: float | None = None,
) -> np.ndarray:
    """Construct a transonic initial guess from a repaired reduced solution."""

    potential = params.potential
    R_son = potential.r_isco if R_son is None else float(R_son)
    lambda0 = float(potential.l_k(potential.r_isco) / (potential.r_g * C)) if lambda0 is None else float(lambda0)
    logR_son = float(np.log(R_son))
    logR_nodes = computational_grid(params, logR_son)
    R_nodes = np.exp(logR_nodes)

    logT = _interp_log_profile(logR_nodes, reduced_profile.R, reduced_profile.T)
    logSigma_thin = _interp_log_profile(logR_nodes, reduced_profile.R, reduced_profile.Sigma)
    logu_thin = np.log(params.Mdot_g_s) - np.log(2.0 * np.pi) - logR_nodes - logSigma_thin

    sonic_state = algebraic_state(logR_nodes[0], logu_thin[0], logT[0], lambda0, params)
    u_sonic = float(np.clip(sonic_state.H * sonic_state.Omega_K, 1.0e4, 0.5 * C))
    xi = (logR_nodes - logR_nodes[0]) / (logR_nodes[-1] - logR_nodes[0])
    weight = (1.0 - xi) ** 2
    logu = (1.0 - weight) * logu_thin + weight * np.log(u_sonic)
    logu = np.clip(logu, params.logu_bounds[0], params.logu_bounds[1])
    logT = np.clip(logT, params.logT_bounds[0], params.logT_bounds[1])
    _ = R_nodes
    return pack_state(logu, logT, logR_son, lambda0)


def initial_guess_from_repaired_reduced_solver(params: TransonicSlimParams) -> tuple[np.ndarray, str]:
    """Run the repaired reduced solver and convert the result to a transonic guess."""

    potential = params.potential
    R_in = 1.08 * potential.r_isco
    grid = make_log_grid(R_in, params.R_out, max(params.n_nodes, 12))
    reduced_params = IsolatedSlimParams(
        M2_g=params.M2_g,
        Mdot_g_s=params.Mdot_g_s,
        R_in=R_in,
        alpha=params.alpha,
        mu_mol=params.mu_mol,
        kappa=params.kappa,
        gamma_gas=params.gamma_gas,
        sigma_brackets=120,
        T_bounds=(np.exp(params.logT_bounds[0]), np.exp(params.logT_bounds[1])),
    )
    result = solve_isolated_slim_disk(grid, reduced_params, max_iter=80, tol=3.0e-3, damping=0.6)
    if result.profile is None:
        raise RuntimeError(f"reduced initial guess failed: {result.message}")
    return initial_guess_from_reduced_solver(result.profile, params), result.message


def _profile_unknowns_from_state_on_grid(z, params: TransonicSlimParams, logR_son: float) -> np.ndarray:
    logu, logT, _old_logR_son, _lambda0, old_logR = unpack_state(z, params)
    new_logR = computational_grid(params, logR_son)
    new_logu = np.interp(new_logR, old_logR, logu, left=logu[0], right=logu[-1])
    new_logT = np.interp(new_logR, old_logR, logT, left=logT[0], right=logT[-1])
    return np.concatenate([new_logu, new_logT])


def _homotopy_stage_result(name: str, z: np.ndarray, residual: np.ndarray, result) -> TransonicHomotopyStageResult:
    return TransonicHomotopyStageResult(
        name=name,
        z=np.asarray(z, dtype=float),
        max_residual=float(np.max(np.abs(residual))),
        cost=float(result.cost),
        nfev=int(result.nfev),
        optimizer_success=bool(result.success),
        message=str(result.message),
    )


def solve_low_mdot_transonic_homotopy(
    params: TransonicSlimParams,
    initial_guess=None,
    fixed_R_son: float | None = None,
    fixed_lambda0: float | None = None,
    max_nfev_per_stage: int | None = None,
    final_max_nfev: int | None = None,
    stage_b_sonic_weight: float = 1.0,
    sonic_weight_sequence: tuple[float, ...] = (1.0,),
    outer_weight_sequence: tuple[float, ...] = (),
    use_stage_block_jacobian: bool = False,
    verbose: int = 0,
) -> TransonicHomotopyResult:
    """Solve a low-Mdot transonic branch with staged eigenparameter release.

    Stage A solves only the nodal profile with ``R_son`` and ``lambda0`` fixed.
    Stage B frees ``R_son`` and enforces the scaled sonic determinant.
    Stage C frees both eigenparameters and solves the full sonic compatibility
    system used by :func:`solve_transonic_outer_branch`.
    """

    try:
        from scipy.optimize import least_squares
    except Exception as exc:
        raise RuntimeError("scipy is required for solve_low_mdot_transonic_homotopy") from exc

    potential = params.potential
    fixed_R_son = float(potential.r_isco if fixed_R_son is None else fixed_R_son)
    fixed_lambda0 = float(potential.l_k(potential.r_isco) / (potential.r_g * C) if fixed_lambda0 is None else fixed_lambda0)
    fixed_logR_son = float(np.log(fixed_R_son))
    stage_max_nfev = int(max_nfev_per_stage if max_nfev_per_stage is not None else max(params.max_nfev, 800))

    if initial_guess is None:
        initial_guess, _ = initial_guess_from_repaired_reduced_solver(params)
    profile_lower, profile_upper = _profile_unknown_bounds(params)
    optimizer_tol = _optimizer_tolerance(params)
    x0_profile = np.clip(
        _profile_unknowns_from_state_on_grid(np.asarray(initial_guess, dtype=float), params, fixed_logR_son),
        profile_lower + 1.0e-12,
        profile_upper - 1.0e-12,
    )

    stage_results: list[TransonicHomotopyStageResult] = []
    stage_a_kwargs = {"jac": (lambda x: _profile_jacobian_from_unknowns(x, params, fixed_logR_son, fixed_lambda0))} if use_stage_block_jacobian else {"jac_sparsity": _profile_jac_sparsity_pattern(params)}
    stage_a = least_squares(
        lambda x: _profile_residual_from_unknowns(x, params, fixed_logR_son, fixed_lambda0),
        x0_profile,
        bounds=(profile_lower, profile_upper),
        x_scale="jac",
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=optimizer_tol,
        max_nfev=stage_max_nfev,
        verbose=verbose,
        **stage_a_kwargs,
    )
    z_a = _pack_profile_unknowns_to_state(stage_a.x, params, fixed_logR_son, fixed_lambda0)
    residual_a = _profile_residual_from_unknowns(stage_a.x, params, fixed_logR_son, fixed_lambda0)
    stage_results.append(_homotopy_stage_result("A_fixed_eigen_profile", z_a, residual_a, stage_a))

    lower_state, upper_state = state_bounds(params)
    lower_b = np.concatenate([profile_lower, np.array([lower_state[-2]])])
    upper_b = np.concatenate([profile_upper, np.array([upper_state[-2]])])
    x0_b = np.clip(np.concatenate([stage_a.x, np.array([fixed_logR_son])]), lower_b + 1.0e-12, upper_b - 1.0e-12)
    sonic_components_b = ("D",)
    stage_b_kwargs = {"jac": (lambda x: _free_rson_jacobian_from_unknowns(x, params, fixed_lambda0, sonic_components_b, sonic_weight=stage_b_sonic_weight))} if use_stage_block_jacobian else {"jac_sparsity": _free_rson_jac_sparsity_pattern(params, len(sonic_components_b))}
    stage_b = least_squares(
        lambda x: _free_rson_residual_from_unknowns(x, params, fixed_lambda0, sonic_components_b, sonic_weight=stage_b_sonic_weight),
        x0_b,
        bounds=(lower_b, upper_b),
        x_scale="jac",
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=optimizer_tol,
        max_nfev=stage_max_nfev,
        verbose=verbose,
        **stage_b_kwargs,
    )
    z_b = _pack_profile_unknowns_to_state(stage_b.x[: 2 * params.n_nodes], params, float(stage_b.x[-1]), fixed_lambda0)
    residual_b = _free_rson_residual_from_unknowns(stage_b.x, params, fixed_lambda0, sonic_components_b, sonic_weight=stage_b_sonic_weight)
    stage_results.append(_homotopy_stage_result("B_free_Rson_fixed_lambda", z_b, residual_b, stage_b))

    z_seed = z_b
    lower_full, upper_full = state_bounds(params)
    final_nfev = int(params.max_nfev if final_max_nfev is None else final_max_nfev)
    full_params = replace(params, max_nfev=final_nfev)
    for weight in sonic_weight_sequence:
        if weight <= 0.0:
            raise ValueError("sonic weights must be positive")
        z0 = np.clip(np.asarray(z_seed, dtype=float), lower_full + 1.0e-12, upper_full - 1.0e-12)
        stage_b2 = least_squares(
            lambda z, sonic_weight=weight: _determinant_only_state_residual(z, full_params, sonic_weight=sonic_weight),
            z0,
            bounds=(lower_full, upper_full),
            jac_sparsity=_determinant_only_sparsity_pattern(full_params),
            x_scale="jac",
            ftol=1.0e-10,
            xtol=1.0e-10,
            gtol=optimizer_tol,
            max_nfev=stage_max_nfev,
            verbose=verbose,
        )
        residual_b2 = _determinant_only_state_residual(stage_b2.x, full_params, sonic_weight=weight)
        z_seed = stage_b2.x
        stage_results.append(_homotopy_stage_result(f"B2_free_Rson_lambda_D_weight_{weight:g}", z_seed, residual_b2, stage_b2))

    ramp_params = replace(params, max_nfev=stage_max_nfev)
    for weight in outer_weight_sequence:
        if weight <= 0.0:
            raise ValueError("outer weights must be positive")
        stage_b3 = solve_transonic_outer_branch(
            ramp_params,
            initial_guess=z_seed,
            outer_residual_weight=weight,
            verbose=verbose,
        )
        z_seed = pack_state(
            np.log(stage_b3.profile.u),
            np.log(stage_b3.profile.T),
            np.log(stage_b3.profile.sonic_radius),
            stage_b3.profile.lambda0,
        )
        stage_results.append(
            TransonicHomotopyStageResult(
                name=f"B3_outer_weight_{weight:g}",
                z=np.asarray(z_seed, dtype=float),
                max_residual=float(stage_b3.max_residual),
                cost=float(stage_b3.cost),
                nfev=int(stage_b3.nfev),
                optimizer_success=bool(stage_b3.optimizer_success),
                message=str(stage_b3.message),
            )
        )

    final = solve_transonic_outer_branch(full_params, initial_guess=z_seed, verbose=verbose)
    stage_results.append(
        TransonicHomotopyStageResult(
            name="C_free_Rson_free_lambda_full",
            z=np.asarray(pack_state(np.log(final.profile.u), np.log(final.profile.T), np.log(final.profile.sonic_radius), final.profile.lambda0), dtype=float),
            max_residual=float(final.max_residual),
            cost=float(final.cost),
            nfev=int(final.nfev),
            optimizer_success=bool(final.optimizer_success),
            message=str(final.message),
        )
    )
    return TransonicHomotopyResult(
        stages=tuple(stage_results),
        final_result=final,
        fixed_R_son=fixed_R_son,
        fixed_lambda0=fixed_lambda0,
    )


def profile_from_state_vector(z, params: TransonicSlimParams) -> TransonicSlimProfile:
    """Convert collocation unknowns into a diagnostic profile."""

    logu, logT, logR_son, lambda0, logR = unpack_state(z, params)
    R = np.exp(logR)
    u = np.exp(logu)
    T = np.exp(logT)
    gu = np.gradient(logu, logR, edge_order=1)
    gT = np.gradient(logT, logR, edge_order=1)

    states = [algebraic_state(x, lu, lt, lambda0, params) for x, lu, lt in zip(logR, logu, logT)]
    Sigma = np.asarray([state.Sigma for state in states])
    H = np.asarray([state.H for state in states])
    rho = np.asarray([state.rho for state in states])
    P = np.asarray([state.P for state in states])
    Pi = np.asarray([state.Pi for state in states])
    e = np.asarray([state.e for state in states])
    tau = np.asarray([state.tau for state in states])
    Omega = np.asarray([state.Omega for state in states])
    Omega_K = np.asarray([state.Omega_K for state in states])
    l = np.asarray([state.l for state in states])
    l_K = np.asarray([state.l_K for state in states])
    W = np.asarray([state.W for state in states])
    Q_rad = np.asarray([state.Q_rad for state in states])
    H_over_R = np.asarray([state.H_over_R for state in states])

    Q_visc = np.empty_like(R)
    Q_adv = np.empty_like(R)
    xi = np.empty_like(R)
    radial = np.empty_like(R)
    energy = np.empty_like(R)
    D = np.empty_like(R)
    C1 = np.empty_like(R)
    C2 = np.empty_like(R)
    K = np.empty_like(R)
    N = np.empty_like(R)
    smin_over_smax = np.empty_like(R)
    null_radial_fraction = np.empty_like(R)
    M_eff = np.empty_like(R)
    for idx, x in enumerate(logR):
        y = np.array([logu[idx], logT[idx]])
        g = np.array([gu[idx], gT[idx]])
        partials = state_partials(x, y, lambda0, params, eps_x=params.partial_eps, eps_y=params.partial_eps)
        dOmega_dx = partials.x["Omega"] + float(np.dot(partials.y["Omega"], g))
        Tdsdx = entropy_gradient_log(x, y, g, lambda0, params)
        Q_visc[idx] = -W[idx] * dOmega_dx
        Q_adv[idx] = -(Sigma[idx] * u[idx] / R[idx]) * Tdsdx
        xi[idx] = xi_eff_from_gradient(x, y, g, lambda0, params)
        raw = differential_residual(x, y, g, lambda0, params)
        radial[idx] = raw[0]
        energy[idx] = raw[1]
        sonic = sonic_diagnostics(x, y, lambda0, params)
        D[idx] = sonic.D
        C1[idx] = sonic.C1
        C2[idx] = sonic.C2
        K[idx] = sonic.compatibility
        N[idx] = sonic.N
        smin_over_smax[idx] = sonic.smin_over_smax
        null_radial_fraction[idx] = sonic.null_radial_fraction
        M_eff[idx] = sonic.M_eff

    scale = np.abs(Q_visc) + np.abs(Q_rad) + np.abs(Q_adv) + 1.0e-300
    normalized_energy = energy / scale
    interval_weights = []
    interval_Qvisc = []
    interval_Qadv = []
    interval_energy = []
    for idx in range(len(R) - 1):
        xm = 0.5 * (logR[idx] + logR[idx + 1])
        ym = np.array([0.5 * (logu[idx] + logu[idx + 1]), 0.5 * (logT[idx] + logT[idx + 1])])
        gm = np.array([(logu[idx + 1] - logu[idx]) / (logR[idx + 1] - logR[idx]), (logT[idx + 1] - logT[idx]) / (logR[idx + 1] - logR[idx])])
        Rm = float(np.exp(xm))
        dR = float(R[idx + 1] - R[idx])
        qv, _qr, qa, qe = _heating_terms_from_gradient(xm, ym, gm, lambda0, params)
        interval_weights.append(2.0 * np.pi * Rm * dR)
        interval_Qvisc.append(qv)
        interval_Qadv.append(qa)
        interval_energy.append(qe)
    weights = np.asarray(interval_weights)
    interval_Qvisc = np.asarray(interval_Qvisc)
    interval_Qadv = np.asarray(interval_Qadv)
    interval_energy = np.asarray(interval_energy)
    norm = float(np.sum(weights * np.abs(interval_Qvisc)) + 1.0e-300)
    integrated_adv = float(np.sum(weights * interval_Qadv) / norm)
    energy_L1 = float(np.sum(weights * np.abs(interval_energy)) / norm)
    sonic_crossings = int(np.count_nonzero(np.diff(np.signbit(D)))) + int(abs(D[0]) < 1.0e-4)
    residual = collocation_residual(z, params)
    return TransonicSlimProfile(
        R=R,
        u=u,
        T=T,
        Sigma=Sigma,
        H=H,
        rho=rho,
        P=P,
        Pi=Pi,
        e=e,
        tau=tau,
        Omega=Omega,
        Omega_K=Omega_K,
        l=l,
        l_K=l_K,
        W=W,
        Q_visc=Q_visc,
        Q_rad=Q_rad,
        Q_adv=Q_adv,
        xi_eff=xi,
        radial_residual=radial,
        energy_residual=energy,
        normalized_energy_residual=normalized_energy,
        sonic_D=D,
        sonic_C1=C1,
        sonic_C2=C2,
        sonic_K=K,
        sonic_N=N,
        sonic_smin_over_smax=smin_over_smax,
        sonic_null_radial_fraction=null_radial_fraction,
        sonic_M_eff=M_eff,
        H_over_R=H_over_R,
        sonic_radius=float(np.exp(logR_son)),
        l0=float(lambda0 * params.r_g * C),
        lambda0=float(lambda0),
        integrated_advective_fraction=integrated_adv,
        energy_L1=energy_L1,
        max_abs_residual=float(np.max(np.abs(residual))),
        sonic_crossings=sonic_crossings,
    )


def solve_square_transonic_polish(
    params: TransonicSlimParams,
    initial_guess,
    *,
    pivot: str = "auto",
    method: str = "newton",
    max_iter: int | None = None,
    max_nfev: int | None = None,
    residual_tol: float | None = None,
    jacobian_rel_step: float = 3.0e-5,
    use_block_jacobian: bool = False,
    line_search_min_alpha: float = 1.0e-6,
    line_search_max_reductions: int = 12,
    linear_solver: str = "regularized_lsmr",
    linear_dampings: tuple[float, ...] = (0.0, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0),
    max_step_norm: float = 2.0,
    verbose: int = 0,
) -> TransonicSquarePolishResult:
    """Polish a fixed-Mdot branch point with the square collocation system."""

    try:
        from scipy.optimize import least_squares
    except Exception as exc:
        raise RuntimeError("scipy is required for solve_square_transonic_polish") from exc

    if method not in {"newton", "least_squares"}:
        raise ValueError("method must be 'newton' or 'least_squares'")
    polish_params = replace(
        params,
        max_nfev=int(params.max_nfev if max_nfev is None else max_nfev),
        residual_tol=float(params.residual_tol if residual_tol is None else residual_tol),
    )
    if jacobian_rel_step <= 0.0:
        raise ValueError("jacobian_rel_step must be positive")
    if line_search_min_alpha <= 0.0:
        raise ValueError("line_search_min_alpha must be positive")
    if line_search_max_reductions < 0:
        raise ValueError("line_search_max_reductions must be non-negative")
    if max_step_norm <= 0.0:
        raise ValueError("max_step_norm must be positive")
    use_direct_linear_solver = _linear_solver_uses_direct(linear_solver)
    damping_candidates = _linear_damping_candidates(linear_solver, linear_dampings)
    lower, upper = state_bounds(polish_params)
    z0 = np.clip(np.asarray(initial_guess, dtype=float), lower + 1.0e-12, upper - 1.0e-12)
    resolved = _resolve_sonic_pivot(z0, polish_params, pivot)
    initial_square = square_collocation_residual(z0, polish_params, pivot=resolved)
    optimizer_tol = _optimizer_tolerance(polish_params)
    iterations = 0
    line_search_reductions = 0
    final_step_norm = 0.0
    final_linear_damping = 0.0

    if method == "least_squares":
        jacobian_kwargs = (
            {
                "jac": lambda z: square_collocation_jacobian(
                    z,
                    polish_params,
                    pivot=resolved,
                    rel_step=jacobian_rel_step,
                )
            }
            if use_block_jacobian
            else {"jac_sparsity": square_jac_sparsity_pattern(polish_params)}
        )
        lsq = least_squares(
            lambda z: square_collocation_residual(z, polish_params, pivot=resolved),
            z0,
            bounds=(lower, upper),
            x_scale="jac",
            ftol=1.0e-12,
            xtol=1.0e-12,
            gtol=optimizer_tol,
            max_nfev=polish_params.max_nfev,
            verbose=verbose,
            **jacobian_kwargs,
        )
        z = np.asarray(lsq.x, dtype=float)
        optimizer_success = bool(lsq.success)
        optimizer_status = int(lsq.status)
        nfev = int(lsq.nfev)
        njev = -1 if lsq.njev is None else int(lsq.njev)
        cost = float(lsq.cost)
        optimality = float(lsq.optimality)
        active_mask = np.asarray(lsq.active_mask, dtype=int)
        message = str(lsq.message)
    else:
        z = np.array(z0, copy=True)
        residual = np.array(initial_square, copy=True)
        max_iterations = int(min(polish_params.max_nfev, 12) if max_iter is None else max_iter)
        nfev = 1
        njev = 0
        optimizer_success = False
        optimizer_status = 0
        message = "maximum Newton iterations reached"
        if max_iterations < 0:
            raise ValueError("max_iter must be non-negative")
        for iteration in range(max_iterations + 1):
            square_max = float(np.max(np.abs(residual)))
            if square_max <= polish_params.residual_tol:
                optimizer_success = True
                optimizer_status = 1
                message = "square Newton polish converged"
                iterations = iteration
                break
            if iteration == max_iterations:
                iterations = iteration
                break

            jac = square_collocation_jacobian(z, polish_params, pivot=resolved, rel_step=jacobian_rel_step)
            njev += 1
            accepted = False
            for damping in damping_candidates:
                step = _equilibrated_sparse_newton_step(
                    jac,
                    residual,
                    damping=damping,
                    use_direct=use_direct_linear_solver,
                    solver_tolerance=optimizer_tol,
                )
                final_linear_damping = damping
                final_step_norm = float(np.linalg.norm(step, ord=np.inf))
                if not np.isfinite(final_step_norm):
                    continue
                if final_step_norm > max_step_norm:
                    step = step * (max_step_norm / final_step_norm)
                    final_step_norm = max_step_norm
                accepted, trial_z, trial_residual, reductions, evaluations = _try_square_newton_step(
                    z,
                    step,
                    residual,
                    polish_params,
                    pivot=resolved,
                    lower=lower,
                    upper=upper,
                    line_search_min_alpha=line_search_min_alpha,
                    line_search_max_reductions=line_search_max_reductions,
                )
                line_search_reductions += reductions
                nfev += evaluations
                if accepted:
                    z = trial_z
                    residual = trial_residual
                    break
            iterations = iteration + 1
            if not accepted:
                optimizer_status = -3
                message = "regularized Newton steps failed to reduce the square residual"
                break
        cost = _square_residual_merit(residual)
        optimality = float(np.max(np.abs(residual)))
        active_mask = _active_mask_from_bounds(z, lower, upper)

    profile = profile_from_state_vector(z, polish_params)
    audit = residual_audit_from_state_vector(z, polish_params)
    full_residual = collocation_residual(z, polish_params)
    full_max = float(np.max(np.abs(full_residual)))
    square_residual = square_collocation_residual(z, polish_params, pivot=resolved)
    square_max = float(np.max(np.abs(square_residual)))
    final_merit = _square_residual_merit(square_residual)
    status = _status_from_profile(profile, audit, polish_params, optimizer_success, square_max)
    result = TransonicSolveResult(
        profile=profile,
        converged=status.physically_valid,
        status=status,
        residual_audit=audit,
        cost=cost,
        max_residual=full_max,
        nfev=nfev,
        njev=njev,
        optimality=optimality,
        optimizer_status=optimizer_status,
        active_mask=active_mask,
        message=message,
        optimizer_success=optimizer_success,
    )
    return TransonicSquarePolishResult(
        z=z,
        pivot=resolved,
        method=method,
        result=result,
        initial_square_max_residual=float(np.max(np.abs(initial_square))),
        final_square_max_residual=square_max,
        unused_compatibility=unused_sonic_compatibility(z, polish_params, pivot=resolved),
        iterations=iterations,
        line_search_reductions=line_search_reductions,
        final_step_norm=final_step_norm,
        final_linear_damping=final_linear_damping,
        final_merit=final_merit,
    )


def solve_transonic_outer_branch(
    params: TransonicSlimParams,
    initial_guess=None,
    outer_residual_weight: float = 1.0,
    sonic_residual_weight: float = 1.0,
    use_collocation_jacobian: bool = False,
    verbose: int = 0,
) -> TransonicSolveResult:
    """Solve the outer free-boundary transonic branch with least squares."""

    try:
        from scipy.optimize import least_squares
    except Exception as exc:
        raise RuntimeError("scipy is required for solve_transonic_outer_branch") from exc

    if initial_guess is None:
        initial_guess, _ = initial_guess_from_repaired_reduced_solver(params)
    if outer_residual_weight <= 0.0 or sonic_residual_weight <= 0.0:
        raise ValueError("residual weights must be positive")
    lower, upper = state_bounds(params)
    z0 = np.clip(np.asarray(initial_guess, dtype=float), lower + 1.0e-12, upper - 1.0e-12)
    optimizer_tol = _optimizer_tolerance(params)
    jacobian_kwargs = (
        {
            "jac": lambda z: _collocation_jacobian_weighted(
                z,
                params,
                outer_weight=outer_residual_weight,
                sonic_weight=sonic_residual_weight,
            )
        }
        if use_collocation_jacobian
        else {"jac_sparsity": jac_sparsity_pattern(params)}
    )
    result = least_squares(
        lambda z: _collocation_residual_weighted(
            z,
            params,
            outer_weight=outer_residual_weight,
            sonic_weight=sonic_residual_weight,
        ),
        z0,
        bounds=(lower, upper),
        x_scale="jac",
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=optimizer_tol,
        max_nfev=params.max_nfev,
        verbose=verbose,
        **jacobian_kwargs,
    )
    profile = profile_from_state_vector(result.x, params)
    residual = collocation_residual(result.x, params)
    max_residual = float(np.max(np.abs(residual)))
    audit = residual_audit_from_state_vector(result.x, params)
    status = _status_from_profile(profile, audit, params, bool(result.success), max_residual)
    return TransonicSolveResult(
        profile=profile,
        converged=status.physically_valid,
        status=status,
        residual_audit=audit,
        cost=float(result.cost),
        max_residual=max_residual,
        nfev=int(result.nfev),
        njev=-1 if result.njev is None else int(result.njev),
        optimality=float(result.optimality),
        optimizer_status=int(result.status),
        active_mask=np.asarray(result.active_mask, dtype=int),
        message=str(result.message),
        optimizer_success=bool(result.success),
    )


def replace_mdot(params: TransonicSlimParams, Mdot_g_s: float) -> TransonicSlimParams:
    """Return a copy with updated accretion rate."""

    return replace(params, Mdot_g_s=float(Mdot_g_s))
