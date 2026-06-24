from __future__ import annotations

import math
import unittest

from imri_qpe.constants import G, M_SUN
from imri_qpe.layer1_hill_flow.hill_geometry import (
    binary_omega,
    circularization_radius,
    hill_radius,
    is_minidisk_allowed,
    schwarzschild_isco_radius,
    tidal_truncation_radius,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import (
    critical_participating_mass,
    eddington_luminosity,
    eddington_mdot,
    omega_k,
    transition_surface_density,
    transition_temperature,
)
from imri_qpe.units import g_per_s_to_msun_per_year


class FiducialScaleTests(unittest.TestCase):
    def assert_rel_close(self, value: float, expected: float, rel: float) -> None:
        self.assertLessEqual(abs(value - expected), rel * abs(expected))

    def setUp(self) -> None:
        self.params = FiducialParams()
        self.R_H = hill_radius(self.params.a_cm, self.params.q)
        self.R_u = self.params.unstable_radius_hill_fraction * self.R_H

    def test_fiducial_hill_geometry_matches_note(self) -> None:
        self.assert_rel_close(self.R_H, 1.10e12, rel=0.01)
        self.assert_rel_close(self.R_u, 3.31e11, rel=0.01)

        Omega_b = binary_omega(self.params.M_smbh_g, self.params.a_cm)
        self.assert_rel_close(G * self.params.M2_g, 3.0 * Omega_b**2 * self.R_H**3, rel=1.0e-12)

    def test_fiducial_minidisk_is_allowed_for_lambda_one(self) -> None:
        R_c = circularization_radius(self.R_H, self.params.lambda_j)
        R_isco = schwarzschild_isco_radius(self.params.M2_g)
        R_out = tidal_truncation_radius(self.R_H, self.params.tidal_truncation_fraction)

        self.assertTrue(is_minidisk_allowed(R_c, R_isco, R_out))
        self.assertFalse(is_minidisk_allowed(1.5 * R_out, R_isco, R_out))
        self.assertFalse(is_minidisk_allowed(0.5 * R_isco, R_isco, R_out))

    def test_transition_estimates_match_note(self) -> None:
        Omega_K = omega_k(self.params.M2_g, self.R_u)
        T_tr = transition_temperature(Omega_K, self.params.alpha_cool, self.params.kappa_es)
        Sigma_tr = transition_surface_density(
            T_tr,
            Omega_K,
            self.params.alpha_cool,
            self.params.kappa_es,
            self.params.mu_mol,
        )
        Mcrit = critical_participating_mass(self.R_u, Sigma_tr, self.params.f_area)

        self.assert_rel_close(Omega_K, 6.05e-3, rel=0.01)
        self.assert_rel_close(T_tr, 2.08e6, rel=0.01)
        self.assert_rel_close(Sigma_tr, 9.41e5, rel=0.01)
        self.assert_rel_close(Mcrit / M_SUN, 1.6e-4, rel=0.03)

    def test_alpha_point_one_mass_estimate_matches_note(self) -> None:
        Omega_K = omega_k(self.params.M2_g, self.R_u)
        T_tr = transition_temperature(Omega_K, 0.1, self.params.kappa_es)
        Sigma_tr = transition_surface_density(T_tr, Omega_K, 0.1, self.params.kappa_es, self.params.mu_mol)
        Mcrit = critical_participating_mass(self.R_u, Sigma_tr, self.params.f_area)

        self.assert_rel_close(Mcrit / M_SUN, 2.2e-5, rel=0.03)

    def test_eddington_scales_for_secondary(self) -> None:
        L_edd = eddington_luminosity(self.params.M2_g, self.params.kappa_es)
        mdot_edd = eddington_mdot(self.params.M2_g, self.params.eta, self.params.kappa_es)

        self.assert_rel_close(L_edd, 1.4705e42, rel=1.0e-4)
        self.assert_rel_close(g_per_s_to_msun_per_year(mdot_edd), 2.5967e-4, rel=1.0e-4)
        self.assertTrue(math.isfinite(L_edd))
        self.assertTrue(math.isfinite(mdot_edd))


if __name__ == "__main__":
    unittest.main()

