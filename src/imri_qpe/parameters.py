"""Parameter containers for the IMRI QPE model."""

from __future__ import annotations

from dataclasses import dataclass

from .constants import DEFAULT_KAPPA_ES, DEFAULT_MU_MOL
from .units import gravitational_radius, solar_masses_to_g


@dataclass(frozen=True)
class FiducialParams:
    """Fiducial system from the project note.

    The code keeps mean molecular weight as ``mu_mol`` and stress-law exponent
    as ``mu_stress`` elsewhere to avoid overloading the symbol ``mu``.
    """

    M_smbh_msun: float = 1.0e6
    M2_msun: float = 1.0e4
    separation_rg_smbh: float = 50.0
    mu_mol: float = DEFAULT_MU_MOL
    kappa_es: float = DEFAULT_KAPPA_ES
    lambda_j: float = 1.0
    tidal_truncation_fraction: float = 0.5
    unstable_radius_hill_fraction: float = 0.3
    alpha_cool: float = 0.01
    f_area: float = 1.0
    eta: float = 0.1

    @property
    def M_smbh_g(self) -> float:
        """Central SMBH mass in grams."""

        return solar_masses_to_g(self.M_smbh_msun)

    @property
    def M2_g(self) -> float:
        """Secondary mass in grams."""

        return solar_masses_to_g(self.M2_msun)

    @property
    def q(self) -> float:
        """Mass ratio M2 / M_smbh."""

        return self.M2_g / self.M_smbh_g

    @property
    def r_g_smbh_cm(self) -> float:
        """Gravitational radius of the SMBH."""

        return gravitational_radius(self.M_smbh_g)

    @property
    def a_cm(self) -> float:
        """Binary separation in cm."""

        return self.separation_rg_smbh * self.r_g_smbh_cm

