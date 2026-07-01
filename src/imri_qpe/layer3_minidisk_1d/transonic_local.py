"""Local differential system for an isolated transonic slim disk."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from imri_qpe.constants import A_RAD, C
from imri_qpe.scales import gas_constant_per_gram

from .transonic_potential import PaczynskiWiitaPotential
from .transonic_thermo import integrated_stress, radiative_cooling, surface_density, vertical_state


@dataclass(frozen=True)
class AlgebraicTransonicState:
    """Algebraic state at one radius for the transonic equations."""

    R: float
    u: float
    T: float
    Sigma: float
    H: float
    rho: float
    P_gas: float
    P_rad: float
    P: float
    Pi: float
    e: float
    tau: float
    W: float
    l0: float
    l: float
    Omega: float
    Omega_K: float
    l_K: float
    Q_rad: float
    H_over_R: float


@dataclass(frozen=True)
class LocalPartials:
    """Explicit and state partials of local quantities in log variables."""

    x: dict[str, float]
    y: dict[str, np.ndarray]


@dataclass(frozen=True)
class SonicDiagnostics:
    """Matrix criticality and compatibility diagnostics."""

    D: float
    C1: float
    C2: float
    compatibility: float
    N: float
    smin_over_smax: float
    singular_values: np.ndarray
    left_null: np.ndarray
    right_null: np.ndarray
    null_radial_fraction: float
    M_eff: float
    radial_scale: float
    energy_scale: float


@dataclass(frozen=True)
class SonicNullVectors:
    """SVD null-vector data for the scaled local sonic matrix."""

    left_null: np.ndarray
    right_null: np.ndarray
    singular_values: np.ndarray
    smin_over_smax: float
    matrix: np.ndarray
    rhs: np.ndarray


@dataclass(frozen=True)
class SonicDerivativeBranch:
    """One regular sonic derivative candidate from an L'Hopital branch scan."""

    kind: str
    form: str
    a: float
    gradient: np.ndarray
    lhopital_raw: float
    lhopital_normalized: float


@dataclass(frozen=True)
class PhaseSpaceTangentDiagnostics:
    """Diagnostics for the desingularized phase-space tangent."""

    tangent: np.ndarray
    B: np.ndarray
    residual: np.ndarray
    singular_values_B: np.ndarray
    smin_over_smax_B: float
    singular_values_A: np.ndarray
    smin_over_smax_A: float
    px: float


def algebraic_state(logR: float, logu: float, logT: float, lambda0: float, params) -> AlgebraicTransonicState:
    """Return local algebraic state from ``x=ln R`` and ``y=[ln u, ln T]``."""

    potential = PaczynskiWiitaPotential(params.M2_g)
    R = float(np.exp(logR))
    u = float(np.exp(logu))
    T = float(np.exp(logT))
    Mdot_local, _dMdot_dx = stream_mass_rate_and_derivative(logR, params)
    Sigma = float(surface_density(Mdot_local, R, u))
    vertical = vertical_state(
        Sigma,
        T,
        R,
        potential,
        mu_mol=params.mu_mol,
        kappa=params.kappa,
        gamma_gas=params.gamma_gas,
    )
    W = float(integrated_stress(vertical, params.alpha, mu_stress=params.mu_stress, stress_factor=params.stress_factor))
    l0 = float(lambda0 * potential.r_g * C)
    stream_l, _stream_dl_dx = stream_torque_specific_l_and_derivative(logR, params)
    l = float(l0 + 2.0 * np.pi * R**2 * W / Mdot_local + stream_l)
    Omega = float(l / R**2)
    Omega_K = float(potential.omega_k(R))
    Q_rad = float(radiative_cooling(vertical, kappa=params.kappa))
    return AlgebraicTransonicState(
        R=R,
        u=u,
        T=T,
        Sigma=Sigma,
        H=float(vertical.H),
        rho=float(vertical.rho),
        P_gas=float(vertical.P_gas),
        P_rad=float(vertical.P_rad),
        P=float(vertical.P_tot),
        Pi=float(vertical.Pi),
        e=float(vertical.e),
        tau=float(vertical.tau),
        W=W,
        l0=l0,
        l=l,
        Omega=Omega,
        Omega_K=Omega_K,
        l_K=float(potential.l_k(R)),
        Q_rad=Q_rad,
        H_over_R=float(vertical.H) / R,
    )


def stream_annulus_shape_and_derivative(logR: float, center_fraction: float, log_width: float, R_out: float) -> tuple[float, float]:
    """Return a smooth cumulative annulus profile and d/dlnR."""

    if center_fraction <= 0.0 or log_width <= 0.0 or R_out <= 0.0:
        raise ValueError("stream annulus center, width, and R_out must be positive")
    logR_center = float(np.log(center_fraction * R_out))
    arg = (float(logR) - logR_center) / float(log_width)
    shape = 0.5 * (1.0 + np.tanh(arg))
    if abs(arg) > 40.0:
        dshape_dx = 0.0
    else:
        sech = 1.0 / np.cosh(arg)
        dshape_dx = 0.5 * sech * sech / float(log_width)
    return float(shape), float(dshape_dx)


def stream_mass_rate_and_derivative(logR: float, params) -> tuple[float, float]:
    """Return local inward accretion rate and dMdot/dlnR for a stream annulus."""

    fraction = float(getattr(params, "stream_mass_fraction", 0.0))
    if fraction == 0.0:
        return float(params.Mdot_g_s), 0.0
    if fraction <= -1.0:
        raise ValueError("stream_mass_fraction must exceed -1")
    center_fraction = float(getattr(params, "stream_mass_center_fraction", 0.8))
    log_width = float(getattr(params, "stream_mass_log_width", 0.08))
    shape, dshape_dx = stream_annulus_shape_and_derivative(logR, center_fraction, log_width, float(params.R_out))
    factor = 1.0 + fraction * shape
    if factor <= 0.0:
        raise ValueError("stream mass profile produced non-positive Mdot")
    return float(params.Mdot_g_s * factor), float(params.Mdot_g_s * fraction * dshape_dx)


def stream_torque_specific_l_and_derivative(logR: float, params) -> tuple[float, float]:
    """Return cumulative stream-torque specific angular momentum and d/dlnR."""

    amplitude = float(getattr(params, "stream_torque_delta_l_fraction", 0.0))
    if amplitude == 0.0:
        return 0.0, 0.0
    center_fraction = float(getattr(params, "stream_torque_center_fraction", 0.8))
    log_width = float(getattr(params, "stream_torque_log_width", 0.08))
    if center_fraction <= 0.0 or log_width <= 0.0:
        raise ValueError("stream torque center and width must be positive")

    potential = PaczynskiWiitaPotential(params.M2_g)
    R_center = float(center_fraction * params.R_out)
    shape, dshape_dx = stream_annulus_shape_and_derivative(logR, center_fraction, log_width, float(params.R_out))
    l_ref = float(potential.l_k(R_center))
    return float(amplitude * l_ref * shape), float(amplitude * l_ref * dshape_dx)


def stream_heating_rate(logR: float, params) -> float:
    """Return annular stream shock/heating source per disk face area."""

    efficiency = float(getattr(params, "stream_heating_efficiency", 0.0))
    if efficiency == 0.0:
        return 0.0
    if efficiency < 0.0:
        raise ValueError("stream_heating_efficiency must be non-negative")
    dMdot_dx = stream_mass_rate_and_derivative(logR, params)[1]
    if dMdot_dx <= 0.0:
        return 0.0
    potential = PaczynskiWiitaPotential(params.M2_g)
    R = float(np.exp(logR))
    orbital_specific_energy = 0.5 * (R * float(potential.omega_k(R))) ** 2
    return float(efficiency * dMdot_dx * orbital_specific_energy / (2.0 * np.pi * R**2))


def _quantity_vector(logR: float, y, lambda0: float, params) -> dict[str, float]:
    state = algebraic_state(logR, float(y[0]), float(y[1]), lambda0, params)
    return {
        "Pi": state.Pi,
        "rho": state.rho,
        "e": state.e,
        "Omega": state.Omega,
    }


def finite_difference_state_partials(
    logR: float,
    y,
    lambda0: float,
    params,
    eps_x: float = 1.0e-5,
    eps_y: float = 1.0e-5,
) -> LocalPartials:
    """Return finite-difference partial derivatives in ``x=ln R`` and ``y``."""

    y = np.asarray(y, dtype=float)
    keys = tuple(_quantity_vector(logR, y, lambda0, params).keys())
    plus_x = _quantity_vector(logR + eps_x, y, lambda0, params)
    minus_x = _quantity_vector(logR - eps_x, y, lambda0, params)
    x_partials = {key: (plus_x[key] - minus_x[key]) / (2.0 * eps_x) for key in keys}

    y_partials: dict[str, np.ndarray] = {}
    for key in keys:
        columns = []
        for column in range(2):
            delta = np.zeros(2)
            delta[column] = eps_y
            plus_y = _quantity_vector(logR, y + delta, lambda0, params)
            minus_y = _quantity_vector(logR, y - delta, lambda0, params)
            columns.append((plus_y[key] - minus_y[key]) / (2.0 * eps_y))
        y_partials[key] = np.asarray(columns, dtype=float)

    return LocalPartials(x=x_partials, y=y_partials)


def _directional_derivatives(state: AlgebraicTransonicState, params, dln_sigma: float, dln_T: float, dln_omega_k: float, dln_R: float) -> dict[str, float]:
    """Return analytic local derivatives for one log-variable direction."""

    R_gas = gas_constant_per_gram(params.mu_mol)
    omega2 = state.Omega_K**2
    radiation_term = 2.0 * A_RAD * state.T**4 / (3.0 * state.Sigma)
    dA = R_gas * state.T * dln_T
    dB = radiation_term * (4.0 * dln_T - dln_sigma)
    dOmega2 = 2.0 * omega2 * dln_omega_k
    denominator = 2.0 * omega2 * state.H - radiation_term
    dH = (state.H * dB + dA - state.H**2 * dOmega2) / denominator
    dln_H = dH / state.H
    dln_rho = dln_sigma - dln_H
    drho = state.rho * dln_rho

    dP_gas = state.P_gas * (dln_rho + dln_T)
    dP_rad = state.P_rad * 4.0 * dln_T
    dP = dP_gas + dP_rad
    dPi = state.Pi * (dln_H + dP / state.P)

    e_gas = R_gas * state.T / (params.gamma_gas - 1.0)
    e_rad = A_RAD * state.T**4 / state.rho
    de = e_gas * dln_T + e_rad * (4.0 * dln_T - dln_rho)

    dln_P_gas = dP_gas / state.P_gas
    dln_P = dP / state.P
    dW = state.W * (dln_H + params.mu_stress * dln_P_gas + (1.0 - params.mu_stress) * dln_P)
    Mdot_local, dMdot_dx = stream_mass_rate_and_derivative(np.log(state.R), params)
    dln_mdot = (dMdot_dx / Mdot_local) * dln_R
    stream_l, stream_dl_dx = stream_torque_specific_l_and_derivative(np.log(state.R), params)
    dOmega_stream = (stream_dl_dx - 2.0 * stream_l) * dln_R / state.R**2
    dOmega_visc = (2.0 * np.pi / Mdot_local) * (dW - state.W * dln_mdot)
    dOmega = -2.0 * (state.l0 / state.R**2) * dln_R + dOmega_visc + dOmega_stream

    return {
        "Pi": float(dPi),
        "rho": float(drho),
        "e": float(de),
        "Omega": float(dOmega),
    }


def analytic_state_partials(logR: float, y, lambda0: float, params) -> LocalPartials:
    """Return analytic partial derivatives in ``x=ln R`` and ``y``."""

    y = np.asarray(y, dtype=float)
    state = algebraic_state(logR, float(y[0]), float(y[1]), lambda0, params)
    potential = PaczynskiWiitaPotential(params.M2_g)
    dln_omega_k_dlnR = float(potential.dln_omega_k_dlnR(state.R))
    Mdot_local, dMdot_dx = stream_mass_rate_and_derivative(logR, params)
    dln_mdot_dx = dMdot_dx / Mdot_local

    x_partials = _directional_derivatives(
        state,
        params,
        dln_sigma=dln_mdot_dx - 1.0,
        dln_T=0.0,
        dln_omega_k=dln_omega_k_dlnR,
        dln_R=1.0,
    )
    logu_partials = _directional_derivatives(
        state,
        params,
        dln_sigma=-1.0,
        dln_T=0.0,
        dln_omega_k=0.0,
        dln_R=0.0,
    )
    logT_partials = _directional_derivatives(
        state,
        params,
        dln_sigma=0.0,
        dln_T=1.0,
        dln_omega_k=0.0,
        dln_R=0.0,
    )
    y_partials = {key: np.asarray([logu_partials[key], logT_partials[key]], dtype=float) for key in x_partials}
    return LocalPartials(x=x_partials, y=y_partials)


def state_partials(logR: float, y, lambda0: float, params, eps_x: float = 1.0e-5, eps_y: float = 1.0e-5) -> LocalPartials:
    """Return analytic partial derivatives in ``x=ln R`` and ``y``."""

    _ = eps_x, eps_y
    return analytic_state_partials(logR, y, lambda0, params)


def differential_residual(logR: float, y, g, lambda0: float, params) -> np.ndarray:
    """Return radial-momentum and energy residuals for a local gradient."""

    y = np.asarray(y, dtype=float)
    g = np.asarray(g, dtype=float)
    state = algebraic_state(logR, y[0], y[1], lambda0, params)
    partials = state_partials(logR, y, lambda0, params, eps_x=params.partial_eps, eps_y=params.partial_eps)

    dPi_dx = partials.x["Pi"] + float(np.dot(partials.y["Pi"], g))
    drho_dx = partials.x["rho"] + float(np.dot(partials.y["rho"], g))
    de_dx = partials.x["e"] + float(np.dot(partials.y["e"], g))
    dOmega_dx = partials.x["Omega"] + float(np.dot(partials.y["Omega"], g))

    radial = state.u**2 * g[0] - state.R**2 * (state.Omega**2 - state.Omega_K**2) + dPi_dx / state.Sigma
    Tdsdx = de_dx - state.P / state.rho**2 * drho_dx
    Q_visc = -state.W * dOmega_dx
    Q_adv = -(state.Sigma * state.u / state.R) * Tdsdx
    Q_stream = stream_heating_rate(logR, params)
    energy = Q_visc + Q_stream - state.Q_rad - Q_adv
    return np.asarray([radial, energy], dtype=float)


def differential_matrix(logR: float, y, lambda0: float, params) -> tuple[np.ndarray, np.ndarray]:
    """Return ``A, c`` such that ``F(g) = A @ g + c``."""

    y = np.asarray(y, dtype=float)
    c = differential_residual(logR, y, np.zeros(2), lambda0, params)
    col0 = differential_residual(logR, y, np.array([1.0, 0.0]), lambda0, params) - c
    col1 = differential_residual(logR, y, np.array([0.0, 1.0]), lambda0, params) - c
    return np.column_stack([col0, col1]), c


def differential_residual_scales(logR: float, y, lambda0: float, params, floor: float = 1.0e-300) -> tuple[float, float]:
    """Return smooth radial-momentum and energy scales for local residuals."""

    state = algebraic_state(logR, float(y[0]), float(y[1]), lambda0, params)
    radial_scale = float(
        np.sqrt(
            state.u**4
            + (state.R**2 * state.Omega_K**2) ** 2
            + (state.Pi / state.Sigma) ** 2
            + floor**2
        )
    )
    energy_scale = float(
        np.sqrt(
            (state.W * state.Omega) ** 2
            + stream_heating_rate(logR, params) ** 2
            + state.Q_rad**2
            + (state.Sigma * state.u * state.e / state.R) ** 2
            + floor**2
        )
    )
    return radial_scale, energy_scale


def scaled_differential_matrix(logR: float, y, lambda0: float, params) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return the local differential matrix after smooth equation scaling."""

    A, c = differential_matrix(logR, y, lambda0, params)
    radial_scale, energy_scale = differential_residual_scales(logR, y, lambda0, params)
    scales = np.array([radial_scale, energy_scale], dtype=float)
    return A / scales[:, None], c / scales, radial_scale, energy_scale


def extended_phase_space_matrix(logR: float, y, lambda0: float, params) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``B=[c A]`` for the phase-space curve equation ``B p = 0``.

    The phase-space tangent is ``p = [dx/ds, dlogu/ds, dlogT/ds]``. Away from
    critical points, a tangent with ``p_x != 0`` is equivalent to the usual
    radial ODE gradient ``dy/dx = p_y / p_x``.
    """

    A, c, _radial_scale, _energy_scale = scaled_differential_matrix(logR, y, lambda0, params)
    return np.column_stack([c, A]), A, c


def B_rank_minors(logR: float, y, lambda0: float, params) -> np.ndarray:
    """Return the three ``2x2`` minors of the phase-space matrix ``B=[c A]``."""

    B, _A, _c = extended_phase_space_matrix(logR, y, lambda0, params)
    return np.asarray(
        [
            B[0, 0] * B[1, 1] - B[0, 1] * B[1, 0],
            B[0, 0] * B[1, 2] - B[0, 2] * B[1, 0],
            B[0, 1] * B[1, 2] - B[0, 2] * B[1, 1],
        ],
        dtype=float,
    )


def _metric_array(metric, size: int) -> np.ndarray:
    if metric is None:
        return np.eye(size)
    metric_array = np.asarray(metric, dtype=float)
    if metric_array.shape == (size,):
        return np.diag(metric_array)
    if metric_array.shape == (size, size):
        return metric_array
    raise ValueError(f"metric must have shape ({size},) or ({size}, {size})")


def _metric_dot(a: np.ndarray, b: np.ndarray, metric: np.ndarray) -> float:
    return float(np.dot(a, metric @ b))


def phase_space_null_tangent(
    logR: float,
    y,
    lambda0: float,
    params,
    *,
    metric=None,
    previous=None,
    prefer_positive_x: bool = True,
    floor: float = 1.0e-300,
) -> PhaseSpaceTangentDiagnostics:
    """Return the normalized right-null tangent of ``B=[c A]``.

    The tangent is oriented continuously against ``previous`` when supplied.
    Otherwise it is oriented with positive ``p_x`` by default.
    """

    B, A, _c = extended_phase_space_matrix(logR, y, lambda0, params)
    U_B, singular_values_B, Vt_B = np.linalg.svd(B, full_matrices=True)
    tangent = np.asarray(Vt_B[-1, :], dtype=float)
    metric_array = _metric_array(metric, 3)
    norm = float(np.sqrt(max(_metric_dot(tangent, tangent, metric_array), floor)))
    tangent = tangent / norm
    if previous is not None:
        previous_array = np.asarray(previous, dtype=float)
        if previous_array.shape != (3,):
            raise ValueError("previous tangent must have shape (3,)")
        if _metric_dot(tangent, previous_array, metric_array) < 0.0:
            tangent = -tangent
    elif prefer_positive_x and tangent[0] < 0.0:
        tangent = -tangent

    singular_values_A = np.linalg.svd(A, compute_uv=False)
    smax_B = float(np.max(singular_values_B)) if singular_values_B.size else 0.0
    smin_B = float(np.min(singular_values_B)) if singular_values_B.size else 0.0
    smax_A = float(np.max(singular_values_A)) if singular_values_A.size else 0.0
    smin_A = float(np.min(singular_values_A)) if singular_values_A.size else 0.0
    residual = B @ tangent
    return PhaseSpaceTangentDiagnostics(
        tangent=tangent,
        B=B,
        residual=np.asarray(residual, dtype=float),
        singular_values_B=np.asarray(singular_values_B, dtype=float),
        smin_over_smax_B=smin_B / (smax_B + floor),
        singular_values_A=np.asarray(singular_values_A, dtype=float),
        smin_over_smax_A=smin_A / (smax_A + floor),
        px=float(tangent[0]),
    )


def phase_space_tangent_derivative(
    logR: float,
    y,
    lambda0: float,
    params,
    p,
    *,
    metric=None,
    eps: float = 1.0e-5,
) -> np.ndarray:
    """Return centered finite-difference derivative of the tangent field along ``p``."""

    z = np.asarray([logR, *np.asarray(y, dtype=float)], dtype=float)
    tangent = np.asarray(p, dtype=float)
    if tangent.shape != (3,):
        raise ValueError("p must have shape (3,)")
    z_plus = z + eps * tangent
    z_minus = z - eps * tangent
    p_plus = phase_space_null_tangent(
        float(z_plus[0]),
        z_plus[1:],
        lambda0,
        params,
        metric=metric,
        previous=tangent,
    ).tangent
    p_minus = phase_space_null_tangent(
        float(z_minus[0]),
        z_minus[1:],
        lambda0,
        params,
        metric=metric,
        previous=tangent,
    ).tangent
    return (p_plus - p_minus) / (2.0 * eps)


def local_unscaled_residual(logR: float, y, g, lambda0: float, params) -> np.ndarray:
    """Return the unscaled local residual ``A @ g + c``."""

    A, c = differential_matrix(logR, y, lambda0, params)
    return A @ np.asarray(g, dtype=float) + c


def local_scaled_residual(logR: float, y, g, lambda0: float, params) -> np.ndarray:
    """Return the scaled local residual ``A @ g + c``."""

    A, c, _radial_scale, _energy_scale = scaled_differential_matrix(logR, y, lambda0, params)
    return A @ np.asarray(g, dtype=float) + c


def _null_vectors_from_matrix(A: np.ndarray, c: np.ndarray, floor: float) -> SonicNullVectors:
    U, singular_values, Vt = np.linalg.svd(A)
    left_null = U[:, -1].copy()
    left_orient_idx = int(np.argmax(np.abs(left_null)))
    if left_null[left_orient_idx] < 0.0:
        left_null = -left_null
    right_null = Vt[-1, :].copy()
    right_orient_idx = int(np.argmax(np.abs(right_null)))
    if right_null[right_orient_idx] < 0.0:
        right_null = -right_null
    smax = float(np.max(singular_values))
    smin = float(np.min(singular_values))
    return SonicNullVectors(
        left_null=left_null,
        right_null=right_null,
        singular_values=singular_values,
        smin_over_smax=smin / (smax + floor),
        matrix=A,
        rhs=c,
    )


def sonic_unscaled_null_vectors(logR: float, y, lambda0: float, params, floor: float = 1.0e-300) -> SonicNullVectors:
    """Return consistently oriented null vectors of the unscaled sonic matrix."""

    A, c = differential_matrix(logR, y, lambda0, params)
    return _null_vectors_from_matrix(A, c, floor)


def sonic_null_vectors(logR: float, y, lambda0: float, params, floor: float = 1.0e-300) -> SonicNullVectors:
    """Return consistently oriented null vectors of the scaled sonic matrix."""

    A, c, _radial_scale, _energy_scale = scaled_differential_matrix(logR, y, lambda0, params)
    return _null_vectors_from_matrix(A, c, floor)


def sonic_directional_B(logR: float, y, g, lambda0: float, params, eps: float = 1.0e-5) -> np.ndarray:
    """Return directional derivative of ``A(x,y)g+c(x,y)`` along ``dy/dx=g``."""

    y = np.asarray(y, dtype=float)
    g = np.asarray(g, dtype=float)
    plus = local_scaled_residual(logR + eps, y + eps * g, g, lambda0, params)
    minus = local_scaled_residual(logR - eps, y - eps * g, g, lambda0, params)
    return (plus - minus) / (2.0 * eps)


def sonic_unscaled_directional_B(logR: float, y, g, lambda0: float, params, eps: float = 1.0e-5) -> np.ndarray:
    """Return directional derivative of the unscaled local residual."""

    y = np.asarray(y, dtype=float)
    g = np.asarray(g, dtype=float)
    plus = local_unscaled_residual(logR + eps, y + eps * g, g, lambda0, params)
    minus = local_unscaled_residual(logR - eps, y - eps * g, g, lambda0, params)
    return (plus - minus) / (2.0 * eps)


def sonic_frozen_scaled_directional_B(logR: float, y, g, lambda0: float, params, eps: float = 1.0e-5) -> np.ndarray:
    """Return unscaled directional derivative divided by sonic-point scales."""

    radial_scale, energy_scale = differential_residual_scales(logR, y, lambda0, params)
    scales = np.array([radial_scale, energy_scale], dtype=float)
    return sonic_unscaled_directional_B(logR, y, g, lambda0, params, eps=eps) / scales


def sonic_lhopital_residual(
    logR: float,
    y,
    g,
    lambda0: float,
    params,
    eps: float = 1.0e-5,
    floor: float = 1.0e-300,
) -> float:
    """Return normalized L'Hopital compatibility ``l.T @ B(g)``."""

    nulls = sonic_null_vectors(logR, y, lambda0, params, floor=floor)
    B = sonic_directional_B(logR, y, g, lambda0, params, eps=eps)
    return float(np.dot(nulls.left_null, B) / (np.linalg.norm(B) + floor))


def sonic_lhopital_residual_form(
    logR: float,
    y,
    g,
    lambda0: float,
    params,
    eps: float = 1.0e-5,
    form: str = "scaled",
    floor: float = 1.0e-300,
) -> float:
    """Return normalized L'Hopital residual for a selected equation scaling."""

    if form == "scaled":
        return sonic_lhopital_residual(logR, y, g, lambda0, params, eps=eps, floor=floor)
    if form == "frozen_scaled":
        nulls = sonic_null_vectors(logR, y, lambda0, params, floor=floor)
        B = sonic_frozen_scaled_directional_B(logR, y, g, lambda0, params, eps=eps)
    elif form == "raw":
        nulls = sonic_unscaled_null_vectors(logR, y, lambda0, params, floor=floor)
        B = sonic_unscaled_directional_B(logR, y, g, lambda0, params, eps=eps)
    else:
        raise ValueError(f"unknown L'Hopital form {form!r}")
    return float(np.dot(nulls.left_null, B) / (np.linalg.norm(B) + floor))


def local_gradient(logR: float, y, lambda0: float, params) -> np.ndarray:
    """Return the nonsingular local gradient solving ``A g + c = 0``."""

    A, c = differential_matrix(logR, y, lambda0, params)
    return np.linalg.solve(A, -c)


def local_ode_rhs(logR: float, y, lambda0: float, params) -> np.ndarray:
    """Return ``dy/dlnR`` for the nonsingular local ODE ``A g + c = 0``."""

    A, c, _radial_scale, _energy_scale = scaled_differential_matrix(logR, y, lambda0, params)
    return np.linalg.solve(A, -c)


def _sonic_form_null_vectors(logR: float, y, lambda0: float, params, form: str) -> SonicNullVectors:
    if form == "raw":
        return sonic_unscaled_null_vectors(logR, y, lambda0, params)
    if form in {"scaled", "frozen_scaled"}:
        return sonic_null_vectors(logR, y, lambda0, params)
    raise ValueError(f"unknown L'Hopital form {form!r}")


def _sonic_form_directional_B(logR: float, y, g, lambda0: float, params, eps: float, form: str) -> np.ndarray:
    if form == "scaled":
        return sonic_directional_B(logR, y, g, lambda0, params, eps=eps)
    if form == "frozen_scaled":
        return sonic_frozen_scaled_directional_B(logR, y, g, lambda0, params, eps=eps)
    if form == "raw":
        return sonic_unscaled_directional_B(logR, y, g, lambda0, params, eps=eps)
    raise ValueError(f"unknown L'Hopital form {form!r}")


def sonic_derivative_branches(
    logR: float,
    y,
    lambda0: float,
    params,
    *,
    eps: float = 1.0e-5,
    form: str = "scaled",
    a_center: float | None = None,
    half_width: float = 1000.0,
    scan_points: int = 2001,
) -> tuple[SonicDerivativeBranch, ...]:
    """Return L'Hopital sonic-derivative branches ``g = g_p + a r``.

    The roots are found by scanning the scalar regularity condition
    ``l.T @ d(A g + c)/dlnR = 0`` along the right-null direction of the sonic
    matrix.  Branches are sorted by the scalar coordinate ``a`` so callers can
    run fixed discrete branches reproducibly.
    """

    if scan_points < 3:
        raise ValueError("scan_points must be at least three")
    y = np.asarray(y, dtype=float)
    nulls = _sonic_form_null_vectors(logR, y, lambda0, params, form)
    g_p = np.linalg.lstsq(nulls.matrix, -nulls.rhs, rcond=None)[0]
    r = nulls.right_null / (np.linalg.norm(nulls.right_null) + 1.0e-300)
    if a_center is None:
        a_center = 0.0
    a_values = np.linspace(float(a_center) - half_width, float(a_center) + half_width, int(scan_points))

    def raw_from_a(a: float) -> float:
        g = g_p + float(a) * r
        B = _sonic_form_directional_B(logR, y, g, lambda0, params, eps=eps, form=form)
        return float(np.dot(nulls.left_null, B))

    raw_values = np.full_like(a_values, np.nan, dtype=float)
    for idx, a in enumerate(a_values):
        try:
            raw_values[idx] = raw_from_a(float(a))
        except Exception:
            raw_values[idx] = np.nan

    branches: list[SonicDerivativeBranch] = []

    def append_branch(kind: str, a: float) -> None:
        g = g_p + float(a) * r
        branches.append(
            SonicDerivativeBranch(
                kind=kind,
                form=form,
                a=float(a),
                gradient=np.asarray(g, dtype=float),
                lhopital_raw=raw_from_a(float(a)),
                lhopital_normalized=sonic_lhopital_residual_form(logR, y, g, lambda0, params, eps=eps, form=form),
            )
        )

    for idx in range(len(a_values) - 1):
        left_a = float(a_values[idx])
        right_a = float(a_values[idx + 1])
        left_f = float(raw_values[idx])
        right_f = float(raw_values[idx + 1])
        if not np.isfinite(left_f) or not np.isfinite(right_f):
            continue
        if left_f == 0.0:
            append_branch("sign", left_a)
        elif left_f * right_f <= 0.0:
            try:
                root_a = float(brentq(raw_from_a, left_a, right_a, xtol=1.0e-10, rtol=1.0e-10, maxiter=80))
            except ValueError:
                continue
            append_branch("sign", root_a)

    finite = np.isfinite(raw_values)
    if np.any(finite):
        finite_indices = np.flatnonzero(finite)
        best_idx = int(finite_indices[int(np.nanargmin(np.abs(raw_values[finite])))])
        left = float(a_values[max(0, best_idx - 4)])
        right = float(a_values[min(len(a_values) - 1, best_idx + 4)])
        if right > left:
            try:
                minimum = minimize_scalar(
                    lambda a: abs(raw_from_a(float(a))),
                    bounds=(left, right),
                    method="bounded",
                    options={"xatol": 1.0e-8},
                )
                if minimum.success and abs(raw_from_a(float(minimum.x))) < 1.0e-8:
                    append_branch("minimum", float(minimum.x))
            except ValueError:
                pass

    unique: list[SonicDerivativeBranch] = []
    for branch in sorted(branches, key=lambda item: item.a):
        if unique and abs(branch.a - unique[-1].a) < 1.0e-5:
            if abs(branch.lhopital_raw) < abs(unique[-1].lhopital_raw):
                unique[-1] = branch
            continue
        unique.append(branch)
    return tuple(unique)


def entropy_gradient_log(logR: float, y, g, lambda0: float, params) -> float:
    """Return ``T ds/dlnR`` along a local gradient."""

    y = np.asarray(y, dtype=float)
    g = np.asarray(g, dtype=float)
    state = algebraic_state(logR, y[0], y[1], lambda0, params)
    partials = state_partials(logR, y, lambda0, params, eps_x=params.partial_eps, eps_y=params.partial_eps)
    drho_dx = partials.x["rho"] + float(np.dot(partials.y["rho"], g))
    de_dx = partials.x["e"] + float(np.dot(partials.y["e"], g))
    return de_dx - state.P / state.rho**2 * drho_dx


def xi_eff_from_gradient(logR: float, y, g, lambda0: float, params) -> float:
    """Return local ``xi_eff`` using ``d/dlnR`` form."""

    state = algebraic_state(logR, float(y[0]), float(y[1]), lambda0, params)
    return -state.rho / state.P * entropy_gradient_log(logR, y, g, lambda0, params)


def sonic_diagnostics(logR: float, y, lambda0: float, params, floor: float = 1.0e-300) -> SonicDiagnostics:
    """Return criticality and compatibility diagnostics."""

    A, c, radial_scale, energy_scale = scaled_differential_matrix(logR, y, lambda0, params)
    U, singular_values, Vt = np.linalg.svd(A)
    a, b = float(A[0, 0]), float(A[0, 1])
    cmat, d = float(A[1, 0]), float(A[1, 1])
    e, f = float(c[0]), float(c[1])
    c_norm = float(np.sqrt(e**2 + f**2))
    col0_norm = float(np.sqrt(a**2 + cmat**2))
    col1_norm = float(np.sqrt(b**2 + d**2))
    det_angle = float((a * d - b * cmat) / (col0_norm * col1_norm + floor))
    C1 = float((d * e - b * f) / (np.sqrt(d**2 + b**2) * c_norm + floor))
    C2 = float((a * f - cmat * e) / (np.sqrt(a**2 + cmat**2) * c_norm + floor))
    left_null = U[:, -1]
    orient_idx = int(np.argmax(np.abs(left_null)))
    if left_null[orient_idx] < 0.0:
        left_null = -left_null
    right_null = Vt[-1, :]
    compatibility = float(np.dot(left_null, np.array([e, f], dtype=float)) / (c_norm + floor))
    N = float(max(abs(C1), abs(C2)))
    smax = float(np.max(singular_values))
    smin = float(np.min(singular_values))
    D = float(np.sign(det_angle) * smin / (smax + floor))
    null_norm = float(np.linalg.norm(right_null))
    null_radial_fraction = float(abs(right_null[0]) / (null_norm + floor))
    state = algebraic_state(logR, float(np.asarray(y, dtype=float)[0]), float(np.asarray(y, dtype=float)[1]), lambda0, params)
    M_eff = float(state.u / (state.H * state.Omega_K + floor))
    return SonicDiagnostics(
        D=D,
        C1=C1,
        C2=C2,
        compatibility=compatibility,
        N=N,
        smin_over_smax=smin / (smax + floor),
        singular_values=singular_values,
        left_null=left_null,
        right_null=right_null,
        null_radial_fraction=null_radial_fraction,
        M_eff=M_eff,
        radial_scale=radial_scale,
        energy_scale=energy_scale,
    )
