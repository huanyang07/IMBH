from __future__ import annotations

import unittest

import numpy as np

from imri_qpe.constants import C, G
from imri_qpe.layer3_minidisk_1d import PaczynskiWiitaPotential
from imri_qpe.units import solar_masses_to_g


class TransonicPotentialTests(unittest.TestCase):
    def setUp(self) -> None:
        self.M_g = solar_masses_to_g(1.0e4)
        self.potential = PaczynskiWiitaPotential(self.M_g)

    def test_paczynski_wiita_radii(self) -> None:
        r_g = G * self.M_g / C**2

        self.assertAlmostEqual(self.potential.r_g / r_g, 1.0)
        self.assertAlmostEqual(self.potential.r_pw / r_g, 2.0)
        self.assertAlmostEqual(self.potential.r_isco / r_g, 6.0)

    def test_omega_derivative_matches_finite_difference(self) -> None:
        R = 20.0 * self.potential.r_g
        eps = 1.0e-5
        omega_plus = self.potential.omega_k(R * np.exp(eps))
        omega_minus = self.potential.omega_k(R * np.exp(-eps))
        finite = (np.log(omega_plus) - np.log(omega_minus)) / (2.0 * eps)

        self.assertAlmostEqual(float(self.potential.dln_omega_k_dlnR(R)), float(finite), places=8)

    def test_keplerian_angular_momentum_extremum_at_isco(self) -> None:
        R = self.potential.r_isco
        eps = 1.0e-5
        l_plus = self.potential.l_k(R * np.exp(eps))
        l_minus = self.potential.l_k(R * np.exp(-eps))
        dlnl_dlnR = (np.log(l_plus) - np.log(l_minus)) / (2.0 * eps)

        self.assertAlmostEqual(float(dlnl_dlnR), 0.0, places=7)

    def test_radius_must_exceed_pseudo_horizon(self) -> None:
        with self.assertRaises(ValueError):
            self.potential.omega_k(self.potential.r_pw)


if __name__ == "__main__":
    unittest.main()
