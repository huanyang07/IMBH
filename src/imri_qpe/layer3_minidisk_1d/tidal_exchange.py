"""Binary/Hill scaling and conservative tidal-deposition profiles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imri_qpe.layer1_hill_flow.hill_geometry import binary_omega, hill_radius
from imri_qpe.parameters import FiducialParams

from .grid import RadialGrid


@dataclass(frozen=True)
class HillTidalGeometry:
    """Binary pattern speed and Hill scales in physical units."""

    hill_radius: float
    pattern_omega: float
    truncation_radius: float


def fiducial_hill_tidal_geometry(
    params: FiducialParams | None = None,
) -> HillTidalGeometry:
    """Return the checked-in binary/Hill geometry used by the minidisk."""

    params = FiducialParams() if params is None else params
    radius = hill_radius(params.a_cm, params.q)
    return HillTidalGeometry(
        hill_radius=radius,
        pattern_omega=binary_omega(params.M_smbh_g, params.a_cm),
        truncation_radius=params.tidal_truncation_fraction * radius,
    )


def hill_outer_torque_weights(
    grid: RadialGrid,
    hill_radius_cm: float,
    *,
    onset_hill_fraction: float = 0.35,
    radial_power: float = 2.0,
) -> np.ndarray:
    """Return normalized cell integrals of an outer Hill-torque kernel.

    The kernel is zero below ``onset_hill_fraction*R_H`` and rises as
    ``(R/R_H-onset)**radial_power``. Cell integrals are analytic, making the
    total deposited torque/power independent of grid resolution.
    """

    if not np.isfinite(hill_radius_cm) or hill_radius_cm <= 0.0:
        raise ValueError("hill_radius_cm must be positive and finite")
    if not 0.0 < onset_hill_fraction < 1.0:
        raise ValueError("onset_hill_fraction must lie in (0,1)")
    if not np.isfinite(radial_power) or radial_power <= -1.0:
        raise ValueError("radial_power must be finite and greater than -1")
    coordinate = np.maximum(
        grid.edges / float(hill_radius_cm) - float(onset_hill_fraction),
        0.0,
    )
    primitive = coordinate ** (float(radial_power) + 1.0) / (
        float(radial_power) + 1.0
    )
    weights = np.diff(primitive)
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("grid does not overlap the configured tidal region")
    return np.asarray(weights / total, dtype=float)
