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
    differential_residual,
    entropy_gradient_log,
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
    partial_eps: float = 1.0e-5
    logu_bounds: tuple[float, float] = (np.log(1.0e-2), np.log(1.5 * C))
    logT_bounds: tuple[float, float] = (np.log(1.0e3), np.log(1.0e10))
    R_son_bounds_rg: tuple[float, float] = (2.05, 60.0)
    lambda0_bounds: tuple[float, float] = (0.01, 12.0)
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
    sonic_N: np.ndarray
    H_over_R: np.ndarray
    sonic_radius: float
    l0: float
    lambda0: float
    integrated_advective_fraction: float
    energy_L1: float
    max_abs_residual: float
    sonic_crossings: int


@dataclass(frozen=True)
class TransonicSolveResult:
    """Result from the free-boundary transonic solve."""

    profile: TransonicSlimProfile | None
    converged: bool
    cost: float
    max_residual: float
    nfev: int
    message: str
    optimizer_success: bool


def computational_grid(params: TransonicSlimParams, logR_son: float) -> np.ndarray:
    """Return collocation node positions in ``ln R``."""

    xi = np.linspace(0.0, 1.0, params.n_nodes)
    return logR_son + xi * (np.log(params.R_out) - logR_son)


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
    state = algebraic_state(logR, float(y[0]), float(y[1]), lambda0, params)
    radial_scale = max(
        state.u**2,
        state.R**2 * state.Omega_K**2,
        abs(state.Pi / state.Sigma),
        1.0e-300,
    )
    energy_scale = max(
        abs(state.W * state.Omega),
        abs(state.Q_rad),
        abs(state.Sigma * state.u * state.e / state.R),
        1.0e-300,
    )
    return radial_scale, energy_scale


def _outer_boundary_residual(logR: float, y, lambda0: float, params: TransonicSlimParams) -> np.ndarray:
    state = algebraic_state(logR, float(y[0]), float(y[1]), lambda0, params)
    potential = params.potential
    shear = float(potential.dln_omega_k_dlnR(state.R))
    Q_visc_thin = -state.W * state.Omega_K * shear
    B_omega = np.log(state.Omega / state.Omega_K)
    B_energy = (Q_visc_thin - state.Q_rad) / (abs(Q_visc_thin) + abs(state.Q_rad) + 1.0e-300)
    return np.asarray([B_omega, B_energy], dtype=float)


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


def collocation_residual(z, params: TransonicSlimParams) -> np.ndarray:
    """Return the scaled free-boundary collocation residual."""

    residual = np.zeros(2 * params.n_nodes + 2, dtype=float)
    try:
        logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
        if np.any(np.diff(logR) <= 0.0):
            raise ValueError("mapped radius must increase")
        row = 0
        for idx in range(params.n_nodes - 1):
            dx = logR[idx + 1] - logR[idx]
            ym = np.array(
                [
                    0.5 * (logu[idx] + logu[idx + 1]),
                    0.5 * (logT[idx] + logT[idx + 1]),
                ]
            )
            gm = np.array(
                [
                    (logu[idx + 1] - logu[idx]) / dx,
                    (logT[idx + 1] - logT[idx]) / dx,
                ]
            )
            xm = 0.5 * (logR[idx] + logR[idx + 1])
            raw = differential_residual(xm, ym, gm, lambda0, params)
            radial_scale, energy_scale = _residual_scales(xm, ym, params, lambda0)
            residual[row : row + 2] = raw / np.array([radial_scale, energy_scale])
            row += 2

        residual[row : row + 2] = _outer_boundary_residual(
            logR[-1],
            np.array([logu[-1], logT[-1]]),
            lambda0,
            params,
        )
        row += 2
        sonic = sonic_diagnostics(logR[0], np.array([logu[0], logT[0]]), lambda0, params)
        residual[row : row + 2] = np.array([sonic.D, sonic.N])
    except Exception:
        residual.fill(1.0e6)
    return residual


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

    size = 2 * params.n_nodes + 2
    pattern = lil_matrix((size, size), dtype=int)
    row = 0
    for idx in range(params.n_nodes - 1):
        columns = [
            idx,
            idx + 1,
            params.n_nodes + idx,
            params.n_nodes + idx + 1,
            size - 2,
            size - 1,
        ]
        for col in columns:
            pattern[row : row + 2, col] = 1
        row += 2
    for col in (params.n_nodes - 1, 2 * params.n_nodes - 1, size - 2, size - 1):
        pattern[row : row + 2, col] = 1
    row += 2
    for col in (0, params.n_nodes, size - 2, size - 1):
        pattern[row : row + 2, col] = 1
    return pattern.tocsr()


def _interp_log_profile(logR_nodes: np.ndarray, R_source: np.ndarray, values: np.ndarray) -> np.ndarray:
    logR_source = np.log(np.asarray(R_source, dtype=float))
    log_values = np.log(np.asarray(values, dtype=float))
    return np.interp(logR_nodes, logR_source, log_values, left=log_values[0], right=log_values[-1])


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
    N = np.empty_like(R)
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
        N[idx] = sonic.N

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
        sonic_N=N,
        H_over_R=H_over_R,
        sonic_radius=float(np.exp(logR_son)),
        l0=float(lambda0 * params.r_g * C),
        lambda0=float(lambda0),
        integrated_advective_fraction=integrated_adv,
        energy_L1=energy_L1,
        max_abs_residual=float(np.max(np.abs(residual))),
        sonic_crossings=sonic_crossings,
    )


def solve_transonic_outer_branch(
    params: TransonicSlimParams,
    initial_guess=None,
    verbose: int = 0,
) -> TransonicSolveResult:
    """Solve the outer free-boundary transonic branch with least squares."""

    try:
        from scipy.optimize import least_squares
    except Exception as exc:
        raise RuntimeError("scipy is required for solve_transonic_outer_branch") from exc

    if initial_guess is None:
        initial_guess, _ = initial_guess_from_repaired_reduced_solver(params)
    lower, upper = state_bounds(params)
    z0 = np.clip(np.asarray(initial_guess, dtype=float), lower + 1.0e-12, upper - 1.0e-12)
    result = least_squares(
        lambda z: collocation_residual(z, params),
        z0,
        bounds=(lower, upper),
        jac_sparsity=jac_sparsity_pattern(params),
        x_scale="jac",
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
        max_nfev=params.max_nfev,
        verbose=verbose,
    )
    profile = profile_from_state_vector(result.x, params)
    max_residual = float(np.max(np.abs(collocation_residual(result.x, params))))
    converged = bool(result.success and max_residual <= params.residual_tol)
    return TransonicSolveResult(
        profile=profile,
        converged=converged,
        cost=float(result.cost),
        max_residual=max_residual,
        nfev=int(result.nfev),
        message=str(result.message),
        optimizer_success=bool(result.success),
    )


def replace_mdot(params: TransonicSlimParams, Mdot_g_s: float) -> TransonicSlimParams:
    """Return a copy with updated accretion rate."""

    return replace(params, Mdot_g_s=float(Mdot_g_s))
