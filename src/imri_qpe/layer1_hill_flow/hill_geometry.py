"""Geometry helpers for gas captured by the secondary Hill sphere."""

from __future__ import annotations

import math

import numpy as np

from imri_qpe.constants import C, G


def hill_radius(a_cm: float, q: float) -> float:
    """Return R_H = a (q / 3)^(1/3)."""

    if a_cm <= 0.0:
        raise ValueError("a_cm must be positive")
    if q <= 0.0:
        raise ValueError("q must be positive")
    return a_cm * (q / 3.0) ** (1.0 / 3.0)


def binary_omega(M_smbh_g: float, a_cm: float) -> float:
    """Return orbital frequency around the SMBH."""

    if M_smbh_g <= 0.0:
        raise ValueError("M_smbh_g must be positive")
    if a_cm <= 0.0:
        raise ValueError("a_cm must be positive")
    return math.sqrt(G * M_smbh_g / a_cm**3)


def circularization_radius(R_H_cm: float, lambda_j: float) -> float:
    """Return R_c = lambda_j^2 R_H / 3."""

    if R_H_cm <= 0.0:
        raise ValueError("R_H_cm must be positive")
    return lambda_j**2 * R_H_cm / 3.0


def tidal_truncation_radius(R_H_cm: float, f_t: float) -> float:
    """Return R_out = f_t R_H."""

    if R_H_cm <= 0.0:
        raise ValueError("R_H_cm must be positive")
    if f_t <= 0.0:
        raise ValueError("f_t must be positive")
    return f_t * R_H_cm


def schwarzschild_isco_radius(M_g: float) -> float:
    """Return the Schwarzschild ISCO radius, 6 GM/c^2."""

    if M_g <= 0.0:
        raise ValueError("M_g must be positive")
    return 6.0 * G * M_g / C**2


def is_minidisk_allowed(R_c_cm: float, R_isco_cm: float, R_out_cm: float):
    """Check whether R_ISCO < R_c < R_out.

    Scalar inputs return a Python ``bool``. Array-like inputs return a NumPy
    boolean array, which is useful for parameter scans.
    """

    allowed = np.logical_and(np.asarray(R_isco_cm) < np.asarray(R_c_cm), np.asarray(R_c_cm) < np.asarray(R_out_cm))
    if np.ndim(allowed) == 0:
        return bool(allowed)
    return allowed

