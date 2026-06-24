"""Pseudo-Newtonian potential helpers for transonic slim-disk tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imri_qpe.constants import C, G


def _require_positive(name: str, value) -> None:
    if np.any(np.asarray(value) <= 0.0):
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class PaczynskiWiitaPotential:
    """Paczynski-Wiita potential around the secondary black hole."""

    M_g: float

    def __post_init__(self) -> None:
        _require_positive("M_g", self.M_g)

    @property
    def r_g(self) -> float:
        """Return ``GM/c^2``."""

        return G * self.M_g / C**2

    @property
    def r_pw(self) -> float:
        """Return the pseudo-horizon radius ``2 GM/c^2``."""

        return 2.0 * self.r_g

    @property
    def r_isco(self) -> float:
        """Return the Paczynski-Wiita marginally stable orbit."""

        return 3.0 * self.r_pw

    def _require_outside_horizon(self, R) -> np.ndarray:
        R = np.asarray(R, dtype=float)
        if np.any(R <= self.r_pw):
            raise ValueError("R must be larger than the Paczynski-Wiita radius")
        return R

    def phi(self, R):
        """Return ``Phi = -GM/(R - R_PW)``."""

        R = self._require_outside_horizon(R)
        return -G * self.M_g / (R - self.r_pw)

    def dphi_dR(self, R):
        """Return the outward radial derivative of the potential."""

        R = self._require_outside_horizon(R)
        return G * self.M_g / (R - self.r_pw) ** 2

    def omega_k(self, R):
        """Return the circular-orbit angular frequency."""

        R = self._require_outside_horizon(R)
        return np.sqrt(G * self.M_g / (R * (R - self.r_pw) ** 2))

    def dln_omega_k_dlnR(self, R):
        """Return ``d ln Omega_K / d ln R``."""

        R = self._require_outside_horizon(R)
        return -0.5 - R / (R - self.r_pw)

    def l_k(self, R):
        """Return the Keplerian specific angular momentum."""

        R = self._require_outside_horizon(R)
        return R**2 * self.omega_k(R)
