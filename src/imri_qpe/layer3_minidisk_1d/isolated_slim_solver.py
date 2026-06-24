"""Isolated no-wind global slim-disk benchmark solver.

This module implements Sprint B from ``Note/CODEX_GLOBAL_SLIM_NEXT_STEPS.md``.
It is intentionally narrower than the IMRI minidisk problem: no stream source,
no tidal torque, and no wind. The solver assumes nearly Keplerian rotation,
imposes a constant inward accretion rate, solves the steady angular-momentum
closure for ``Sigma(R)``, and relaxes ``T(R)`` against
``Q_visc = Q_rad + Q_adv``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from imri_qpe.constants import C, DEFAULT_KAPPA_ES, DEFAULT_MU_MOL, G, SIGMA_SB
from imri_qpe.scales import eddington_mdot

from .audit_metrics import energy_residual_metrics
from .entropy_advection import entropy_temperature_gradient, mdot_from_vr, q_advective, xi_eff
from .global_slim import (
    alpha_viscosity,
    keplerian_specific_angular_momentum,
    keplerian_stress,
    keplerian_viscous_heating,
    vertical_structure_arrays,
)
from .grid import RadialGrid


@dataclass(frozen=True)
class IsolatedSlimParams:
    """Parameters for an isolated nearly Keplerian slim-disk benchmark."""

    M2_g: float
    Mdot_g_s: float
    R_in: float
    alpha: float = 0.01
    mu_mol: float = DEFAULT_MU_MOL
    kappa: float = DEFAULT_KAPPA_ES
    gamma_gas: float = 5.0 / 3.0
    l_in: float | None = None
    sigma_bounds: tuple[float, float] = (1.0e-6, 1.0e12)
    T_bounds: tuple[float, float] = (1.0e3, 1.0e10)
    sigma_brackets: int = 260
    branch: str = "nearest"

    def __post_init__(self) -> None:
        if self.M2_g <= 0.0:
            raise ValueError("M2_g must be positive")
        if self.Mdot_g_s <= 0.0:
            raise ValueError("Mdot_g_s must be positive")
        if self.R_in <= 0.0:
            raise ValueError("R_in must be positive")
        if self.alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        if self.mu_mol <= 0.0:
            raise ValueError("mu_mol must be positive")
        if self.kappa <= 0.0:
            raise ValueError("kappa must be positive")
        if self.gamma_gas <= 1.0:
            raise ValueError("gamma_gas must exceed 1")
        if self.sigma_bounds[0] <= 0.0 or self.sigma_bounds[1] <= self.sigma_bounds[0]:
            raise ValueError("sigma_bounds must be positive and increasing")
        if self.T_bounds[0] <= 0.0 or self.T_bounds[1] <= self.T_bounds[0]:
            raise ValueError("T_bounds must be positive and increasing")
        if self.sigma_brackets < 8:
            raise ValueError("sigma_brackets must be at least 8")
        if self.branch not in {"nearest", "largest", "smallest"}:
            raise ValueError("branch must be 'nearest', 'largest', or 'smallest'")

    @property
    def inner_angular_momentum(self) -> float:
        """Return the inner angular-momentum integration constant."""

        if self.l_in is not None:
            return self.l_in
        R_isco = 6.0 * G * self.M2_g / C**2
        return float(keplerian_specific_angular_momentum(self.M2_g, R_isco))

    @property
    def mdot_edd_ratio(self) -> float:
        """Return ``Mdot/Mdot_Edd`` using the project default Eddington rate."""

        return self.Mdot_g_s / eddington_mdot(self.M2_g)


@dataclass(frozen=True)
class IsolatedSlimProfile:
    """Steady isolated slim-disk diagnostic profile."""

    R: np.ndarray
    area: np.ndarray
    Sigma: np.ndarray
    T: np.ndarray
    Omega_K: np.ndarray
    H: np.ndarray
    rho: np.ndarray
    P: np.ndarray
    e: np.ndarray
    tau: np.ndarray
    nu: np.ndarray
    v_R: np.ndarray
    Mdot: np.ndarray
    W_required: np.ndarray
    W_model: np.ndarray
    angular_momentum_residual: np.ndarray
    TdsdR: np.ndarray
    xi_eff: np.ndarray
    Q_visc: np.ndarray
    Q_rad: np.ndarray
    Q_adv: np.ndarray
    energy_residual: np.ndarray
    normalized_energy_residual: np.ndarray
    advective_fraction: np.ndarray
    H_over_R: np.ndarray


@dataclass(frozen=True)
class IsolatedSlimSolveResult:
    """Result from one isolated constant-Mdot solve."""

    profile: IsolatedSlimProfile | None
    converged: bool
    failed: bool
    iterations: int
    max_abs_residual: float
    L1_residual: float
    message: str
    history: np.ndarray


@dataclass(frozen=True)
class IsolatedSlimContinuationResult:
    """A sequence of isolated slim-disk solves along Mdot continuation."""

    results: tuple[IsolatedSlimSolveResult, ...]
    mdot_values: np.ndarray


def required_keplerian_stress(Mdot_g_s: float, M2_g: float, R, l_in: float):
    """Return ``W`` required by steady Keplerian angular momentum."""

    R = np.asarray(R, dtype=float)
    l = keplerian_specific_angular_momentum(M2_g, R)
    W = Mdot_g_s * (l - l_in) / (2.0 * np.pi * R**2)
    return np.asarray(W, dtype=float)


def _stress_model(Sigma: float, T: float, R: float, params: IsolatedSlimParams) -> float:
    Omega, H, _, _, _, _ = vertical_structure_arrays(
        np.array([Sigma]),
        np.array([T]),
        params.M2_g,
        np.array([R]),
        params.mu_mol,
        params.kappa,
        params.gamma_gas,
    )
    nu = alpha_viscosity(params.alpha, H, Omega)
    return float(keplerian_stress(np.array([Sigma]), nu, Omega)[0])


def _stress_log_residual(log_sigma: float, T: float, R: float, params: IsolatedSlimParams) -> float:
    Sigma = float(np.exp(log_sigma))
    W_req = float(required_keplerian_stress(params.Mdot_g_s, params.M2_g, R, params.inner_angular_momentum))
    return float(np.log(_stress_model(Sigma, T, R, params) / W_req))


def sigma_root_near_guess(T: float, R: float, params: IsolatedSlimParams, guess: float) -> float | None:
    """Find an angular-momentum root near a supplied ``Sigma`` guess."""

    if guess <= 0.0 or not np.isfinite(guess):
        return None
    log_min = np.log(params.sigma_bounds[0])
    log_max = np.log(params.sigma_bounds[1])
    center = float(np.clip(np.log(guess), log_min, log_max))
    try:
        f_center = _stress_log_residual(center, T, R, params)
    except Exception:
        return None
    if abs(f_center) < 1.0e-10:
        return float(np.exp(center))

    for width in (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8):
        lo = max(log_min, center - width)
        hi = min(log_max, center + width)
        try:
            flo = _stress_log_residual(lo, T, R, params)
            fhi = _stress_log_residual(hi, T, R, params)
        except Exception:
            continue
        if not np.isfinite(flo) or not np.isfinite(fhi):
            continue
        if flo == 0.0:
            return float(np.exp(lo))
        if fhi == 0.0:
            return float(np.exp(hi))
        if flo * fhi > 0.0:
            continue
        for _ in range(70):
            mid = 0.5 * (lo + hi)
            fmid = _stress_log_residual(mid, T, R, params)
            if flo * fmid <= 0.0:
                hi = mid
            else:
                lo = mid
                flo = fmid
        return float(np.exp(0.5 * (lo + hi)))
    return None


def sigma_roots_for_temperature(T: float, R: float, params: IsolatedSlimParams) -> np.ndarray:
    """Return all positive ``Sigma`` roots of the angular-momentum closure."""

    if T <= 0.0 or R <= 0.0:
        raise ValueError("T and R must be positive")
    W_req = float(required_keplerian_stress(params.Mdot_g_s, params.M2_g, R, params.inner_angular_momentum))
    if W_req <= 0.0:
        return np.array([], dtype=float)

    log_sigma = np.linspace(np.log(params.sigma_bounds[0]), np.log(params.sigma_bounds[1]), params.sigma_brackets)
    values = np.empty_like(log_sigma)
    for idx, log_value in enumerate(log_sigma):
        values[idx] = _stress_log_residual(float(log_value), T, R, params)

    roots: list[float] = []
    for idx in range(len(log_sigma) - 1):
        left = values[idx]
        right = values[idx + 1]
        if not np.isfinite(left) or not np.isfinite(right):
            continue
        if left == 0.0:
            roots.append(float(log_sigma[idx]))
            continue
        if left * right > 0.0:
            continue
        lo = float(log_sigma[idx])
        hi = float(log_sigma[idx + 1])
        flo = left
        for _ in range(70):
            mid = 0.5 * (lo + hi)
            fmid = _stress_log_residual(mid, T, R, params)
            if flo * fmid <= 0.0:
                hi = mid
            else:
                lo = mid
                flo = fmid
        root = 0.5 * (lo + hi)
        if not roots or abs(root - roots[-1]) > 1.0e-8:
            roots.append(root)

    return np.exp(np.asarray(roots, dtype=float))


def choose_sigma_root(roots, guess: float | None = None, branch: str = "nearest") -> float | None:
    """Choose one root according to branch preference and optional guess."""

    roots = np.asarray(roots, dtype=float)
    if len(roots) == 0:
        return None
    if branch == "largest":
        return float(np.max(roots))
    if branch == "smallest":
        return float(np.min(roots))
    if guess is None or not np.isfinite(guess) or guess <= 0.0:
        return float(np.max(roots))
    return float(roots[int(np.argmin(np.abs(np.log(roots / guess))))])


def solve_sigma_profile_from_temperature(
    grid: RadialGrid,
    T,
    params: IsolatedSlimParams,
    Sigma_guess=None,
) -> tuple[np.ndarray | None, str]:
    """Solve the angular-momentum closure for ``Sigma(R)``."""

    T = np.asarray(T, dtype=float)
    if T.shape != grid.centers.shape:
        raise ValueError("T must match grid centers")
    if np.any(T <= 0.0):
        raise ValueError("T must be positive")

    guesses = np.full_like(T, np.nan) if Sigma_guess is None else np.asarray(Sigma_guess, dtype=float)
    if guesses.shape != T.shape:
        raise ValueError("Sigma_guess must match grid centers")

    Sigma = np.empty_like(T)
    for idx, (R, temp) in enumerate(zip(grid.centers, T)):
        guess = None if not np.isfinite(guesses[idx]) else float(guesses[idx])
        root = sigma_root_near_guess(float(temp), float(R), params, guess) if guess is not None else None
        if root is None:
            root = choose_sigma_root(sigma_roots_for_temperature(float(temp), float(R), params), guess, params.branch)
        if root is None:
            return None, f"no angular-momentum Sigma root at cell {idx}"
        Sigma[idx] = root
    return Sigma, "ok"


def evaluate_isolated_slim_profile(grid: RadialGrid, Sigma, T, params: IsolatedSlimParams) -> IsolatedSlimProfile:
    """Evaluate the isolated no-wind constant-Mdot slim-disk equations."""

    R = grid.centers
    Sigma = np.asarray(Sigma, dtype=float)
    T = np.asarray(T, dtype=float)
    if Sigma.shape != R.shape or T.shape != R.shape:
        raise ValueError("Sigma and T must match grid centers")
    if np.any(Sigma <= 0.0) or np.any(T <= 0.0):
        raise ValueError("Sigma and T must be positive")

    Omega, H, rho, P, e, tau = vertical_structure_arrays(
        Sigma,
        T,
        params.M2_g,
        R,
        params.mu_mol,
        params.kappa,
        params.gamma_gas,
    )
    nu = alpha_viscosity(params.alpha, H, Omega)
    W_model = keplerian_stress(Sigma, nu, Omega)
    W_required = required_keplerian_stress(params.Mdot_g_s, params.M2_g, R, params.inner_angular_momentum)
    angular_residual = np.divide(
        W_model - W_required,
        np.abs(W_required) + 1.0e-300,
        out=np.full_like(W_model, np.nan),
        where=W_required != 0.0,
    )
    v_R = -params.Mdot_g_s / (2.0 * np.pi * R * Sigma)
    Mdot = np.asarray(mdot_from_vr(R, Sigma, v_R), dtype=float)
    TdsdR = entropy_temperature_gradient(R, rho, T, P=P, e=e)
    xi = np.asarray(xi_eff(R, rho, P, TdsdR), dtype=float)
    Q_adv = np.asarray(q_advective(Sigma, v_R, TdsdR), dtype=float)
    Q_visc = keplerian_viscous_heating(nu, Sigma, Omega)
    Q_rad = 16.0 * SIGMA_SB * T**4 / (3.0 * params.kappa * Sigma)
    energy_residual = Q_visc - Q_rad - Q_adv
    scale = np.abs(Q_visc) + np.abs(Q_rad) + np.abs(Q_adv) + 1.0e-300
    normalized_energy_residual = energy_residual / scale
    advective_fraction = np.divide(Q_adv, Q_visc, out=np.full_like(Q_adv, np.nan), where=Q_visc != 0.0)

    return IsolatedSlimProfile(
        R=R,
        area=grid.area,
        Sigma=Sigma,
        T=T,
        Omega_K=Omega,
        H=H,
        rho=rho,
        P=P,
        e=e,
        tau=tau,
        nu=nu,
        v_R=v_R,
        Mdot=Mdot,
        W_required=W_required,
        W_model=W_model,
        angular_momentum_residual=angular_residual,
        TdsdR=TdsdR,
        xi_eff=xi,
        Q_visc=Q_visc,
        Q_rad=Q_rad,
        Q_adv=Q_adv,
        energy_residual=energy_residual,
        normalized_energy_residual=normalized_energy_residual,
        advective_fraction=advective_fraction,
        H_over_R=H / R,
    )


def best_local_temperature_profile(
    grid: RadialGrid,
    params: IsolatedSlimParams,
    previous_T=None,
    previous_Sigma=None,
    n_T: int = 120,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """Build a smooth initial profile from local no-advection balance."""

    if n_T < 8:
        raise ValueError("n_T must be at least 8")
    previous_T = None if previous_T is None else np.asarray(previous_T, dtype=float)
    previous_Sigma = None if previous_Sigma is None else np.asarray(previous_Sigma, dtype=float)
    if previous_T is not None and previous_T.shape != grid.centers.shape:
        raise ValueError("previous_T must match grid centers")
    if previous_Sigma is not None and previous_Sigma.shape != grid.centers.shape:
        raise ValueError("previous_Sigma must match grid centers")

    logT_grid = np.linspace(np.log(params.T_bounds[0]), np.log(params.T_bounds[1]), n_T)
    T_profile = np.empty_like(grid.centers)
    Sigma_profile = np.empty_like(grid.centers)

    for cell, R in enumerate(grid.centers):
        candidates: list[tuple[float, float, float, float]] = []
        sigma_guess = None if previous_Sigma is None else float(previous_Sigma[cell])
        for logT in logT_grid:
            T = float(np.exp(logT))
            roots = sigma_roots_for_temperature(T, float(R), params)
            for Sigma in roots:
                Omega, H, _, _, _, _ = vertical_structure_arrays(
                    np.array([Sigma]),
                    np.array([T]),
                    params.M2_g,
                    np.array([R]),
                    params.mu_mol,
                    params.kappa,
                    params.gamma_gas,
                )
                nu = alpha_viscosity(params.alpha, H, Omega)
                Q_visc = float(keplerian_viscous_heating(nu, np.array([Sigma]), Omega)[0])
                Q_rad = float(16.0 * SIGMA_SB * T**4 / (3.0 * params.kappa * Sigma))
                energy_mismatch = abs(np.log(Q_visc / Q_rad))
                branch_penalty = 0.0
                if previous_T is not None:
                    branch_penalty += 0.05 * abs(np.log(T / previous_T[cell]))
                if sigma_guess is not None and sigma_guess > 0.0:
                    branch_penalty += 0.02 * abs(np.log(Sigma / sigma_guess))
                candidates.append((energy_mismatch + branch_penalty, energy_mismatch, T, float(Sigma)))
        if not candidates:
            return None, None, f"no local initial candidate at cell {cell}"
        candidates.sort(key=lambda item: item[0])
        _, _, T_profile[cell], Sigma_profile[cell] = candidates[0]

    return T_profile, Sigma_profile, "ok"


def state_vector_from_profile(Sigma, T) -> np.ndarray:
    """Return ``z = [ln Sigma, ln T]`` for positive profile arrays."""

    Sigma = np.asarray(Sigma, dtype=float)
    T = np.asarray(T, dtype=float)
    if Sigma.shape != T.shape:
        raise ValueError("Sigma and T must have the same shape")
    if np.any(Sigma <= 0.0) or np.any(T <= 0.0):
        raise ValueError("Sigma and T must be positive")
    return np.concatenate([np.log(Sigma), np.log(T)])


def profile_from_state_vector(grid: RadialGrid, z, params: IsolatedSlimParams) -> IsolatedSlimProfile:
    """Evaluate an isolated profile from ``z = [ln Sigma, ln T]``."""

    z = np.asarray(z, dtype=float)
    n = len(grid.centers)
    if z.shape != (2 * n,):
        raise ValueError("state vector must have length 2 * number of grid cells")
    log_sigma = np.clip(z[:n], np.log(params.sigma_bounds[0]), np.log(params.sigma_bounds[1]))
    log_T = np.clip(z[n:], np.log(params.T_bounds[0]), np.log(params.T_bounds[1]))
    return evaluate_isolated_slim_profile(grid, np.exp(log_sigma), np.exp(log_T), params)


def isolated_residual_vector(grid: RadialGrid, z, params: IsolatedSlimParams) -> np.ndarray:
    """Return simultaneous angular-momentum and energy residuals."""

    profile = profile_from_state_vector(grid, z, params)
    angular = np.log((2.0 * np.pi * profile.R**2 * profile.W_model) / (params.Mdot_g_s * (keplerian_specific_angular_momentum(params.M2_g, profile.R) - params.inner_angular_momentum)))
    energy = profile.normalized_energy_residual
    residual = np.concatenate([angular, energy])
    return np.where(np.isfinite(residual), residual, 1.0e30)


def _finite_difference_jacobian(grid: RadialGrid, z: np.ndarray, params: IsolatedSlimParams, residual: np.ndarray) -> np.ndarray:
    """Return a dense finite-difference Jacobian for the small development solver."""

    jac = np.empty((len(residual), len(z)), dtype=float)
    for column in range(len(z)):
        step = 1.0e-5 * max(1.0, abs(float(z[column])))
        trial = z.copy()
        trial[column] += step
        jac[:, column] = (isolated_residual_vector(grid, trial, params) - residual) / step
    return jac


def _solve_damped_least_squares_step(jac: np.ndarray, residual: np.ndarray, regularization: float) -> np.ndarray:
    """Return a damped Gauss-Newton step."""

    lhs = jac.T @ jac + regularization * np.eye(jac.shape[1])
    rhs = -(jac.T @ residual)
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def solve_isolated_slim_disk(
    grid: RadialGrid,
    params: IsolatedSlimParams,
    T_initial=None,
    Sigma_initial=None,
    max_iter: int = 160,
    tol: float = 1.0e-3,
    damping: float = 0.35,
    max_log_step: float = 0.2,
    boundary_exclude: int = 2,
) -> IsolatedSlimSolveResult:
    """Solve one isolated constant-Mdot no-wind benchmark by simultaneous residual solve."""

    if max_iter < 0:
        raise ValueError("max_iter must be non-negative")
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    if damping <= 0.0:
        raise ValueError("damping must be positive")
    if max_log_step <= 0.0:
        raise ValueError("max_log_step must be positive")

    if T_initial is None:
        T_initial, Sigma_initial, message = best_local_temperature_profile(grid, params, previous_Sigma=Sigma_initial)
        if T_initial is None:
            return IsolatedSlimSolveResult(None, False, True, 0, np.inf, np.inf, message, np.array([], dtype=float))
    else:
        T_initial = np.asarray(T_initial, dtype=float)
        if T_initial.shape != grid.centers.shape:
            raise ValueError("T_initial must match grid centers")
        if np.any(T_initial <= 0.0):
            raise ValueError("T_initial must be positive")

    if Sigma_initial is None:
        Sigma_initial, message = solve_sigma_profile_from_temperature(grid, T_initial, params)
        if Sigma_initial is None:
            return IsolatedSlimSolveResult(None, False, True, 0, np.inf, np.inf, message, np.array([], dtype=float))
    else:
        Sigma_initial = np.asarray(Sigma_initial, dtype=float)
        if Sigma_initial.shape != grid.centers.shape:
            raise ValueError("Sigma_initial must match grid centers")
        if np.any(Sigma_initial <= 0.0):
            raise ValueError("Sigma_initial must be positive")

    z = state_vector_from_profile(Sigma_initial, T_initial)
    z_min = np.concatenate(
        [
            np.full(len(grid.centers), np.log(params.sigma_bounds[0])),
            np.full(len(grid.centers), np.log(params.T_bounds[0])),
        ]
    )
    z_max = np.concatenate(
        [
            np.full(len(grid.centers), np.log(params.sigma_bounds[1])),
            np.full(len(grid.centers), np.log(params.T_bounds[1])),
        ]
    )
    history: list[float] = []
    best_profile: IsolatedSlimProfile | None = None
    best_max = np.inf
    best_L1 = np.inf
    message = "maximum iterations reached"
    regularization = 1.0e-6

    idx = slice(None) if boundary_exclude <= 0 or 2 * boundary_exclude >= len(grid.centers) else slice(boundary_exclude, -boundary_exclude)

    for iteration in range(max_iter + 1):
        profile = profile_from_state_vector(grid, z, params)
        residual = isolated_residual_vector(grid, z, params)
        no_wind = np.zeros_like(profile.Q_visc)
        metrics = energy_residual_metrics(
            profile.area,
            profile.Q_visc,
            profile.Q_rad,
            profile.Q_adv,
            no_wind,
            boundary_exclude=boundary_exclude,
        )
        angular_max = float(np.max(np.abs(profile.angular_momentum_residual[idx])))
        energy_max = float(np.max(np.abs(profile.normalized_energy_residual[idx])))
        max_abs = max(angular_max, energy_max)
        objective = 0.5 * float(np.mean(residual**2))
        history.append(max_abs)
        if metrics.L1 < best_L1:
            best_profile = profile
            best_L1 = metrics.L1
            best_max = max_abs
        if max_abs <= tol and metrics.L1 <= 10.0 * tol and angular_max <= tol:
            return IsolatedSlimSolveResult(
                profile,
                True,
                False,
                iteration,
                max_abs,
                metrics.L1,
                "converged",
                np.asarray(history, dtype=float),
            )

        if iteration == max_iter:
            break

        jac = _finite_difference_jacobian(grid, z, params, residual)
        step = _solve_damped_least_squares_step(jac, residual, regularization)
        max_component = float(np.max(np.abs(step)))
        if max_component > max_log_step:
            step *= max_log_step / max_component
        accepted = False
        for factor in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            trial_z = np.clip(z + damping * factor * step, z_min, z_max)
            trial_residual = isolated_residual_vector(grid, trial_z, params)
            trial_objective = 0.5 * float(np.mean(trial_residual**2))
            if trial_objective < objective:
                z = trial_z
                regularization = max(1.0e-10, 0.3 * regularization)
                accepted = True
                break
        if not accepted:
            regularization *= 10.0
            if regularization < 1.0e6:
                continue
            message = "line search failed"
            break

    return IsolatedSlimSolveResult(
        best_profile,
        False,
        best_profile is None,
        max_iter,
        best_max,
        best_L1,
        message,
        np.asarray(history, dtype=float),
    )


def continue_isolated_slim_branch(
    grid: RadialGrid,
    base_params: IsolatedSlimParams,
    mdot_values,
    max_iter: int = 160,
    tol: float = 1.0e-3,
    damping: float = 0.35,
    boundary_exclude: int = 2,
) -> IsolatedSlimContinuationResult:
    """Run continuation over increasing imposed accretion rates."""

    mdot_values = np.asarray(mdot_values, dtype=float)
    if mdot_values.ndim != 1 or len(mdot_values) == 0:
        raise ValueError("mdot_values must be a non-empty one-dimensional array")
    if np.any(mdot_values <= 0.0):
        raise ValueError("mdot_values must be positive")

    results: list[IsolatedSlimSolveResult] = []
    previous_T = None
    previous_Sigma = None
    previous_mdot = None
    for Mdot in mdot_values:
        params = replace(base_params, Mdot_g_s=float(Mdot))
        if previous_T is None:
            T_initial, Sigma_initial, _ = best_local_temperature_profile(grid, params)
        else:
            scale = (float(Mdot) / float(previous_mdot)) ** 0.25
            T_initial = previous_T * scale
            Sigma_initial = previous_Sigma
        result = solve_isolated_slim_disk(
            grid,
            params,
            T_initial=T_initial,
            Sigma_initial=Sigma_initial,
            max_iter=max_iter,
            tol=tol,
            damping=damping,
            boundary_exclude=boundary_exclude,
        )
        results.append(result)
        if result.converged and result.profile is not None:
            previous_T = result.profile.T
            previous_Sigma = result.profile.Sigma
            previous_mdot = result.profile.Mdot[0]
    return IsolatedSlimContinuationResult(results=tuple(results), mdot_values=mdot_values)
