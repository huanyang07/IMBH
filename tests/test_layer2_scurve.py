from __future__ import annotations

import math
import unittest

import numpy as np

from imri_qpe.constants import SIGMA_SB
from imri_qpe.layer1_hill_flow.hill_geometry import hill_radius
from imri_qpe.layer2_scurve import (
    ThermalEquilibriumParams,
    compute_scurve,
    find_turning_points,
    gas_pressure,
    instability_mu_criterion,
    q_adv_minus,
    q_plus_alpha,
    q_rad_minus,
    q_wind_minus,
    radiation_pressure,
    solve_scale_height,
    stress_pressure,
    temperature_activated_cooling_fraction,
    vertical_state,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import gas_constant_per_gram, omega_k


class VerticalStructureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.params = FiducialParams()
        self.R_H = hill_radius(self.params.a_cm, self.params.q)
        self.R_u = 0.3 * self.R_H

    def test_scale_height_solves_hydrostatic_balance(self) -> None:
        Sigma = 9.41e5
        T_c = 2.08e6
        state = vertical_state(Sigma, T_c, self.params.M2_g, self.R_u)

        hydro_pressure = 0.5 * Sigma * state.Omega_K**2 * state.H
        self.assertAlmostEqual(hydro_pressure / state.P_tot, 1.0, places=12)
        self.assertAlmostEqual(state.rho_c, Sigma / (2.0 * state.H), places=12)
        self.assertGreater(state.tau, 1.0)

    def test_pressure_helpers(self) -> None:
        self.assertAlmostEqual(gas_pressure(2.0, 3.0, mu_mol=1.0), 2.0 * gas_constant_per_gram(1.0) * 3.0)
        self.assertAlmostEqual(radiation_pressure(10.0), radiation_pressure(np.array(10.0)))

    def test_invalid_scale_height_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            solve_scale_height(0.0, 1.0e5, self.params.M2_g, self.R_u)


class StressLawTests(unittest.TestCase):
    def test_stress_pressure_special_cases(self) -> None:
        Pgas = np.array([2.0, 4.0])
        Ptot = np.array([8.0, 16.0])

        np.testing.assert_allclose(stress_pressure(Pgas, Ptot, alpha=0.1, mu_stress=0.0), 0.1 * Ptot)
        np.testing.assert_allclose(stress_pressure(Pgas, Ptot, alpha=0.1, mu_stress=1.0), 0.1 * Pgas)

    def test_instability_mu_criterion_boundary(self) -> None:
        self.assertTrue(instability_mu_criterion(0.0))
        self.assertTrue(instability_mu_criterion(0.5))
        self.assertFalse(instability_mu_criterion(4.0 / 7.0))
        self.assertFalse(instability_mu_criterion(1.0))


class ThermalEquilibriumTests(unittest.TestCase):
    def setUp(self) -> None:
        self.params = FiducialParams()
        self.R_u = 0.3 * hill_radius(self.params.a_cm, self.params.q)
        self.Omega_K = omega_k(self.params.M2_g, self.R_u)

    def test_radiative_cooling_matches_note_formula(self) -> None:
        T = 1.0e6
        Sigma = 1.0e5
        expected = 16.0 * SIGMA_SB * T**4 / (3.0 * self.params.kappa_es * Sigma)

        self.assertAlmostEqual(q_rad_minus(T, Sigma, self.params.kappa_es), expected)

    def test_advective_cooling_is_positive_and_scales_with_xi(self) -> None:
        Sigma = 1.0e5
        T = 2.0e6
        Q_plus = q_plus_alpha(Sigma, T, self.R_u, self.params.M2_g, 0.01, 0.0, self.params.mu_mol)

        Q_adv_1 = q_adv_minus(Sigma, T, self.R_u, self.params.M2_g, Q_plus, xi=0.5)
        Q_adv_2 = q_adv_minus(Sigma, T, self.R_u, self.params.M2_g, Q_plus, xi=1.0)

        self.assertGreater(Q_adv_1, 0.0)
        self.assertAlmostEqual(Q_adv_2 / Q_adv_1, 2.0)

    def test_wind_cooling_activates_above_eddington(self) -> None:
        low = q_wind_minus(
            1.0e10,
            self.R_u,
            self.params.M2_g,
            strength=1.0,
            eta=self.params.eta,
            kappa=self.params.kappa_es,
        )
        high = q_wind_minus(
            1.0e26,
            self.R_u,
            self.params.M2_g,
            strength=1.0,
            eta=self.params.eta,
            kappa=self.params.kappa_es,
        )

        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0.0)

    def test_gas_pressure_branch_residual_matches_analytic_scaling(self) -> None:
        alpha = 0.01
        Sigma = 1.0e3
        R_gas = gas_constant_per_gram(self.params.mu_mol)
        T_gas = ((27.0 / 128.0) * alpha * self.params.kappa_es * R_gas * Sigma**2 * self.Omega_K / SIGMA_SB) ** (
            1.0 / 3.0
        )

        Q_plus = q_plus_alpha(Sigma, T_gas, self.R_u, self.params.M2_g, alpha, 0.0, self.params.mu_mol)
        Q_minus = q_rad_minus(T_gas, Sigma, self.params.kappa_es)

        self.assertLess(abs(Q_plus - Q_minus) / Q_minus, 2.0e-4)

    def test_total_pressure_stress_produces_unstable_radiation_roots(self) -> None:
        result = compute_scurve(
            self.R_u,
            np.geomspace(1.0e4, 1.0e6, 12),
            self.params.M2_g,
            ThermalEquilibriumParams(alpha=0.01, mu_stress=0.0),
            n_T_brackets=220,
        )

        self.assertGreater(len(result.Sigma), 0)
        self.assertIn(0, result.branch_index)
        self.assertIn(1, result.branch_index)
        self.assertGreater(np.count_nonzero(result.stable), 0)
        self.assertGreater(np.count_nonzero(~result.stable), 0)

    def test_gas_pressure_stress_scan_is_stable_in_simplified_model(self) -> None:
        result = compute_scurve(
            self.R_u,
            np.geomspace(1.0e4, 1.0e7, 12),
            self.params.M2_g,
            ThermalEquilibriumParams(alpha=0.01, mu_stress=1.0),
            n_T_brackets=220,
        )

        self.assertGreater(len(result.Sigma), 0)
        self.assertTrue(np.all(result.stable))
        np.testing.assert_array_equal(result.branch_index, np.zeros_like(result.branch_index))

    def test_temperature_activated_cooling_can_stabilize_hot_branch(self) -> None:
        params = ThermalEquilibriumParams(
            alpha=0.01,
            mu_stress=0.0,
            advective_temperature=3.0e6,
            advective_power=6.0,
        )
        result = compute_scurve(
            self.R_u,
            np.geomspace(1.0e4, 3.0e5, 12),
            self.params.M2_g,
            params,
            T_bounds=(1.0e4, 1.0e8),
            n_T_brackets=400,
        )

        self.assertEqual(set(result.branch_index.tolist()), {0, 1, 2})
        for branch, expected_stable in [(0, True), (1, False), (2, True)]:
            mask = result.branch_index == branch
            self.assertTrue(np.all(result.stable[mask] == expected_stable))

    def test_temperature_activated_cooling_fraction_limits(self) -> None:
        self.assertEqual(temperature_activated_cooling_fraction(1.0e6, None, 6.0), 0.0)
        self.assertLess(temperature_activated_cooling_fraction(1.0e5, 1.0e6, 6.0), 1.0e-5)
        self.assertGreater(temperature_activated_cooling_fraction(1.0e7, 1.0e6, 6.0), 0.999)


class TurningPointTests(unittest.TestCase):
    def test_find_turning_points_on_synthetic_curve(self) -> None:
        Sigma = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        Mdot = np.array([1.0, 3.0, 2.0, 1.0, 2.0])

        turns = find_turning_points(Sigma, Mdot)

        np.testing.assert_array_equal(turns["maxima"], np.array([1]))
        np.testing.assert_array_equal(turns["minima"], np.array([3]))
        self.assertEqual(turns["Sigma_max"], 2.0)
        self.assertEqual(turns["Sigma_min"], 4.0)

    def test_find_turning_points_rejects_non_positive_values(self) -> None:
        with self.assertRaises(ValueError):
            find_turning_points(np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
