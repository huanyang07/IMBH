"""Energy-limited wind helper functions for the 1D minidisk model."""

from __future__ import annotations

import numpy as np

from imri_qpe.constants import C, DEFAULT_KAPPA_ES, G


def _as_float_or_array(value):
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    return array


def q_edd_vertical(Omega_K, H, kappa: float = DEFAULT_KAPPA_ES):
    """Return the two-sided local vertical Eddington flux."""

    if np.any(np.asarray(Omega_K) <= 0.0):
        raise ValueError("Omega_K must be positive")
    if np.any(np.asarray(H) <= 0.0):
        raise ValueError("H must be positive")
    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    return _as_float_or_array(2.0 * C * np.asarray(Omega_K, dtype=float) ** 2 * np.asarray(H, dtype=float) / kappa)


def q_available(Q_visc, Q_stream=0.0, Q_tide=0.0, Q_adv=0.0):
    """Return energy available for radiation/wind after advection."""

    return _as_float_or_array(
        np.asarray(Q_visc, dtype=float)
        + np.asarray(Q_stream, dtype=float)
        + np.asarray(Q_tide, dtype=float)
        - np.asarray(Q_adv, dtype=float)
    )


def wind_energy_per_mass(M2_g: float, R_cm, v_inf=0.0, h_w=0.0, torque_work=0.0):
    """Return specific energy needed/removed by wind material."""

    if M2_g <= 0.0:
        raise ValueError("M2_g must be positive")
    if np.any(np.asarray(R_cm) <= 0.0):
        raise ValueError("R_cm must be positive")
    return _as_float_or_array(
        G * M2_g / (2.0 * np.asarray(R_cm, dtype=float))
        + 0.5 * np.asarray(v_inf, dtype=float) ** 2
        + np.asarray(h_w, dtype=float)
        + np.asarray(torque_work, dtype=float)
    )


def wind_mass_loss_prime_from_energy(Q_wind, R_cm, E_w):
    """Return positive wind mass loss per ``dlnR`` from an energy sink.

    ``Q_wind`` is the two-sided vertically integrated loss rate per disk area.
    The annulus luminosity per logarithmic radius is ``2 pi R^2 Q_wind``.
    """

    if np.any(np.asarray(R_cm) <= 0.0):
        raise ValueError("R_cm must be positive")
    if np.any(np.asarray(E_w) <= 0.0):
        raise ValueError("E_w must be positive")
    Q_wind = np.asarray(Q_wind, dtype=float)
    R_cm = np.asarray(R_cm, dtype=float)
    E_w = np.asarray(E_w, dtype=float)
    return _as_float_or_array(2.0 * np.pi * R_cm**2 * Q_wind / E_w)


def effective_wind_powerlaw_slope(Q_wind, R_cm, Mdot, E_w, floor: float = 1.0e-300):
    """Return ``s_eff = dln(Mdot)/dlnR`` implied by an energy-coupled wind."""

    if np.any(np.asarray(Mdot) <= 0.0):
        raise ValueError("Mdot must be positive")
    if floor <= 0.0:
        raise ValueError("floor must be positive")
    wind_prime = wind_mass_loss_prime_from_energy(Q_wind, R_cm, E_w)
    return _as_float_or_array(np.asarray(wind_prime, dtype=float) / np.maximum(np.asarray(Mdot, dtype=float), floor))


def required_wind_energy_for_powerlaw_slope(Q_wind, R_cm, Mdot, s_target, floor: float = 1.0e-300):
    """Return the launch energy that would reproduce a target ``Mdot ~ R^s``."""

    if np.any(np.asarray(R_cm) <= 0.0):
        raise ValueError("R_cm must be positive")
    if np.any(np.asarray(Mdot) <= 0.0):
        raise ValueError("Mdot must be positive")
    if np.any(np.asarray(s_target) <= 0.0):
        raise ValueError("s_target must be positive")
    if floor <= 0.0:
        raise ValueError("floor must be positive")
    Q_wind = np.asarray(Q_wind, dtype=float)
    R_cm = np.asarray(R_cm, dtype=float)
    Mdot = np.asarray(Mdot, dtype=float)
    s_target = np.asarray(s_target, dtype=float)
    denominator = np.maximum(s_target * Mdot, floor)
    return _as_float_or_array(2.0 * np.pi * R_cm**2 * Q_wind / denominator)


def energy_limited_wind(
    Q_avail,
    Q_edd,
    E_w,
    epsilon_w: float,
    chi_edd: float = 1.0,
    activation_width: float = 0.0,
):
    """Return ``(Q_wind, Q_rad, dotSigma_w)`` with no energy double counting."""

    if not 0.0 <= epsilon_w <= 1.0:
        raise ValueError("epsilon_w must be between 0 and 1")
    if not 0.0 < chi_edd <= 1.0:
        raise ValueError("chi_edd must be in (0, 1]")
    width = float(activation_width)
    if not np.isfinite(width) or width < 0.0:
        raise ValueError("activation_width must be finite and non-negative")
    if np.any(np.asarray(E_w) <= 0.0):
        raise ValueError("E_w must be positive")

    Q_avail = np.asarray(Q_avail, dtype=float)
    Q_edd = np.asarray(Q_edd, dtype=float)
    positive_available = np.maximum(Q_avail, 0.0)
    threshold_excess = positive_available - float(chi_edd) * Q_edd
    if width == 0.0:
        excess = np.maximum(threshold_excess, 0.0)
    else:
        excess = width * np.logaddexp(0.0, threshold_excess / width)
        excess = np.minimum(excess, positive_available)
    Q_wind = epsilon_w * excess
    Q_rad = positive_available - Q_wind
    dotSigma_w = Q_wind / np.asarray(E_w, dtype=float)
    return _as_float_or_array(Q_wind), _as_float_or_array(Q_rad), _as_float_or_array(dotSigma_w)


def energy_limited_wind_derivatives(
    Q_avail,
    Q_edd,
    epsilon_w: float,
    chi_edd: float = 1.0,
    activation_width: float = 0.0,
    activation_width_dQedd: float = 0.0,
):
    """Return ``(dQ_wind/dQ_avail, dQ_wind/dQ_edd)``.

    ``activation_width_dQedd`` should be set when the smooth activation width
    is tied to the local Eddington flux, e.g. ``activation_width = f Q_edd``.
    """

    if not 0.0 <= epsilon_w <= 1.0:
        raise ValueError("epsilon_w must be between 0 and 1")
    if not 0.0 < chi_edd <= 1.0:
        raise ValueError("chi_edd must be in (0, 1]")
    width = float(activation_width)
    if not np.isfinite(width) or width < 0.0:
        raise ValueError("activation_width must be finite and non-negative")
    width_dQedd = float(activation_width_dQedd)
    if not np.isfinite(width_dQedd) or width_dQedd < 0.0:
        raise ValueError("activation_width_dQedd must be finite and non-negative")

    Q_avail = np.asarray(Q_avail, dtype=float)
    Q_edd = np.asarray(Q_edd, dtype=float)
    positive_available = np.maximum(Q_avail, 0.0)
    dpositive_davail = (Q_avail > 0.0).astype(float)
    threshold_excess = positive_available - float(chi_edd) * Q_edd

    if width == 0.0:
        raw_excess = np.maximum(threshold_excess, 0.0)
        active = threshold_excess > 0.0
        draw_davail = active.astype(float) * dpositive_davail
        draw_dedd = np.where(active, -float(chi_edd), 0.0)
    else:
        scaled = threshold_excess / width
        sigmoid = 1.0 / (1.0 + np.exp(-np.clip(scaled, -700.0, 700.0)))
        softplus = np.logaddexp(0.0, scaled)
        raw_excess = width * softplus
        draw_davail = sigmoid * dpositive_davail
        draw_dwidth = softplus - scaled * sigmoid
        draw_dedd = -float(chi_edd) * sigmoid + width_dQedd * draw_dwidth

    capped = raw_excess >= positive_available
    dexcess_davail = np.where(capped, dpositive_davail, draw_davail)
    dexcess_dedd = np.where(capped, 0.0, draw_dedd)
    return _as_float_or_array(epsilon_w * dexcess_davail), _as_float_or_array(epsilon_w * dexcess_dedd)
