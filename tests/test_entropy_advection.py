from __future__ import annotations

import unittest

import numpy as np

from imri_qpe.constants import A_RAD
from imri_qpe.layer3_minidisk_1d import (
    entropy_gradient_consistency_error,
    entropy_gradient_log_formula,
    entropy_temperature_gradient,
    mdot_from_vr,
    q_advective,
    specific_internal_energy,
    total_pressure,
    xi_eff,
)
from imri_qpe.scales import gas_constant_per_gram


class EntropyAdvectionTests(unittest.TestCase):
    def test_specific_internal_energy_gas_limit(self) -> None:
        rho = 1.0e-6
        T = 10.0
        gamma = 5.0 / 3.0
        expected = gas_constant_per_gram() * T / (gamma - 1.0)

        self.assertAlmostEqual(specific_internal_energy(rho, T, gamma_gas=gamma) / expected, 1.0, places=5)

    def test_entropy_gradient_nearly_zero_for_adiabatic_gas_profile(self) -> None:
        gamma = 5.0 / 3.0
        R = np.geomspace(1.0, 10.0, 128)
        rho = 1.0e-6 * R**-1.5
        T = 10.0 * (rho / rho[0]) ** (gamma - 1.0)

        TdsdR = entropy_temperature_gradient(R, rho, T)

        thermal_scale = gas_constant_per_gram() * np.max(np.abs(np.gradient(T, R)))
        self.assertLess(np.max(np.abs(TdsdR[4:-4])) / thermal_scale, 3.0e-2)

    def test_log_entropy_formula_matches_first_law_for_smooth_profile(self) -> None:
        R = np.geomspace(1.0, 20.0, 128)
        rho = 2.0e-7 * R**-0.8
        T = 2.0e4 * R**-0.35

        first_law = entropy_temperature_gradient(R, rho, T, gradient_method="centered")
        log_formula = entropy_gradient_log_formula(R, rho, T, gradient_method="centered")
        error = entropy_gradient_consistency_error(first_law, log_formula)

        self.assertLess(np.median(error[4:-4]), 1.0e-3)

    def test_log_entropy_formula_gas_isentropic_profile(self) -> None:
        gamma = 5.0 / 3.0
        R = np.geomspace(1.0, 20.0, 128)
        rho = 1.0e-6 * R**-1.1
        T = 10.0 * (rho / rho[0]) ** (gamma - 1.0)

        TdsdR = entropy_gradient_log_formula(R, rho, T, gamma_gas=gamma)

        thermal_scale = gas_constant_per_gram() * np.max(np.abs(np.gradient(T, R)))
        self.assertLess(np.max(np.abs(TdsdR[4:-4])) / thermal_scale, 1.0e-10)

    def test_log_entropy_formula_radiation_isentropic_profile(self) -> None:
        R = np.geomspace(1.0, 20.0, 128)
        rho = 1.0e-10 * R**-1.2
        T = 1.0e7 * (rho / rho[0]) ** (1.0 / 3.0)

        TdsdR = entropy_gradient_log_formula(R, rho, T)

        rad_scale = np.max(4.0 * A_RAD * T**4 / rho * np.abs(np.gradient(np.log(T), R)))
        self.assertLess(np.max(np.abs(TdsdR[4:-4])) / rad_scale, 2.0e-2)

    def test_q_advective_sign_convention(self) -> None:
        Sigma = np.array([1.0, 1.0])
        v_R = np.array([-2.0, -2.0])
        entropy_gradient = np.array([-3.0, 3.0])

        Q_adv = q_advective(Sigma, v_R, entropy_gradient)

        self.assertGreater(Q_adv[0], 0.0)
        self.assertLess(Q_adv[1], 0.0)

    def test_xi_eff_recovery_from_synthetic_profile(self) -> None:
        R = np.geomspace(1.0, 10.0, 8)
        rho = np.full_like(R, 2.0)
        P = np.full_like(R, 5.0)
        xi = 0.3
        TdsdR = -xi * P / (R * rho)

        np.testing.assert_allclose(xi_eff(R, rho, P, TdsdR), xi)

    def test_mdot_from_inward_velocity_is_positive(self) -> None:
        R = np.array([1.0, 2.0])
        Sigma = np.array([3.0, 4.0])
        v_R = np.array([-5.0, -6.0])

        mdot = mdot_from_vr(R, Sigma, v_R)

        self.assertTrue(np.all(mdot > 0.0))


if __name__ == "__main__":
    unittest.main()
