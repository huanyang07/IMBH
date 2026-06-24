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


def energy_limited_wind(Q_avail, Q_edd, E_w, epsilon_w: float):
    """Return ``(Q_wind, Q_rad, dotSigma_w)`` with no energy double counting."""

    if not 0.0 <= epsilon_w <= 1.0:
        raise ValueError("epsilon_w must be between 0 and 1")
    if np.any(np.asarray(E_w) <= 0.0):
        raise ValueError("E_w must be positive")

    Q_avail = np.asarray(Q_avail, dtype=float)
    Q_edd = np.asarray(Q_edd, dtype=float)
    positive_available = np.maximum(Q_avail, 0.0)
    excess = np.maximum(positive_available - Q_edd, 0.0)
    Q_wind = epsilon_w * excess
    Q_rad = positive_available - Q_wind
    dotSigma_w = Q_wind / np.asarray(E_w, dtype=float)
    return _as_float_or_array(Q_wind), _as_float_or_array(Q_rad), _as_float_or_array(dotSigma_w)
