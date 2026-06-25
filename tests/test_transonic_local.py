from __future__ import annotations

import types
import unittest

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    PaczynskiWiitaPotential,
    algebraic_state,
    analytic_state_partials,
    differential_matrix,
    differential_residual_scales,
    differential_residual,
    finite_difference_state_partials,
    integrated_stress,
    local_gradient,
    radiative_cooling,
    scaled_differential_matrix,
    sonic_diagnostics,
    surface_density,
    vertical_state,
    xi_eff_from_gradient,
)
from imri_qpe.scales import eddington_mdot
from imri_qpe.units import solar_masses_to_g


class TransonicLocalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.M2_g = solar_masses_to_g(1.0e4)
        self.potential = PaczynskiWiitaPotential(self.M2_g)
        self.params = types.SimpleNamespace(
            M2_g=self.M2_g,
            Mdot_g_s=1.0e-3 * eddington_mdot(self.M2_g),
            alpha=0.01,
            mu_stress=0.0,
            stress_factor=1.5,
            mu_mol=0.62,
            kappa=0.34,
            gamma_gas=5.0 / 3.0,
            partial_eps=1.0e-5,
        )
        self.logR = float(np.log(30.0 * self.potential.r_g))
        self.y = np.log(np.array([1.0e6, 2.0e6]))
        self.lambda0 = float(self.potential.l_k(self.potential.r_isco) / (self.potential.r_g * 2.99792458e10))

    def test_surface_density_continuity(self) -> None:
        R = 20.0 * self.potential.r_g
        u = 3.0e6
        Sigma = surface_density(self.params.Mdot_g_s, R, u)

        self.assertAlmostEqual(2.0 * np.pi * R * Sigma * u / self.params.Mdot_g_s, 1.0)

    def test_vertical_closure_is_positive_and_hydrostatic(self) -> None:
        R = 30.0 * self.potential.r_g
        Sigma = 1.0e4
        T = 2.0e6

        state = vertical_state(Sigma, T, R, self.potential)

        for value in (state.H, state.rho, state.P_gas, state.P_rad, state.P_tot, state.Pi, state.e, state.tau):
            self.assertGreater(value, 0.0)
        lhs = state.Omega_K**2 * state.H**2
        rhs = state.P_tot / state.rho
        self.assertAlmostEqual(lhs / rhs, 1.0, places=10)

    def test_alpha_stress_and_radiative_cooling_are_positive(self) -> None:
        R = 30.0 * self.potential.r_g
        state = vertical_state(1.0e4, 2.0e6, R, self.potential)

        self.assertGreater(integrated_stress(state, alpha=0.01), 0.0)
        self.assertGreater(radiative_cooling(state), 0.0)

    def test_algebraic_state_is_finite(self) -> None:
        state = algebraic_state(self.logR, self.y[0], self.y[1], self.lambda0, self.params)

        for value in (
            state.Sigma,
            state.H,
            state.rho,
            state.P,
            state.Pi,
            state.W,
            state.l,
            state.Omega,
            state.Q_rad,
        ):
            self.assertTrue(np.isfinite(value))
            self.assertGreater(value, 0.0)

    def test_differential_system_is_linear_in_gradient(self) -> None:
        A, c = differential_matrix(self.logR, self.y, self.lambda0, self.params)
        g = np.array([0.17, -0.44])

        direct = differential_residual(self.logR, self.y, g, self.lambda0, self.params)
        linear = A @ g + c
        scale = np.maximum(np.abs(direct), np.abs(linear)) + 1.0

        self.assertLess(float(np.max(np.abs(direct - linear) / scale)), 1.0e-12)

    def test_scaled_differential_matrix_matches_smooth_scales(self) -> None:
        A, c = differential_matrix(self.logR, self.y, self.lambda0, self.params)
        radial_scale, energy_scale = differential_residual_scales(self.logR, self.y, self.lambda0, self.params)
        A_scaled, c_scaled, radial_returned, energy_returned = scaled_differential_matrix(self.logR, self.y, self.lambda0, self.params)

        np.testing.assert_allclose(A_scaled, A / np.array([[radial_scale], [energy_scale]]))
        np.testing.assert_allclose(c_scaled, c / np.array([radial_scale, energy_scale]))
        self.assertAlmostEqual(radial_returned, radial_scale)
        self.assertAlmostEqual(energy_returned, energy_scale)

    def test_analytic_partials_match_finite_difference_partials(self) -> None:
        analytic = analytic_state_partials(self.logR, self.y, self.lambda0, self.params)
        finite = finite_difference_state_partials(self.logR, self.y, self.lambda0, self.params, eps_x=3.0e-6, eps_y=3.0e-6)

        for key in ("Pi", "rho", "e", "Omega"):
            scale_x = max(abs(analytic.x[key]), abs(finite.x[key]), 1.0)
            self.assertLess(abs(analytic.x[key] - finite.x[key]) / scale_x, 3.0e-5, key)
            scale_y = np.maximum(np.maximum(np.abs(analytic.y[key]), np.abs(finite.y[key])), 1.0)
            self.assertLess(float(np.max(np.abs(analytic.y[key] - finite.y[key]) / scale_y)), 3.0e-5, key)

    def test_analytic_partials_support_mixed_stress_exponent(self) -> None:
        params = types.SimpleNamespace(**{**self.params.__dict__, "mu_stress": 0.4})
        analytic = analytic_state_partials(self.logR, self.y, self.lambda0, params)
        finite = finite_difference_state_partials(self.logR, self.y, self.lambda0, params, eps_x=3.0e-6, eps_y=3.0e-6)

        scale = np.maximum(np.maximum(np.abs(analytic.y["Omega"]), np.abs(finite.y["Omega"])), 1.0)
        self.assertLess(float(np.max(np.abs(analytic.y["Omega"] - finite.y["Omega"]) / scale)), 3.0e-5)

    def test_local_gradient_solves_residual(self) -> None:
        g = local_gradient(self.logR, self.y, self.lambda0, self.params)
        residual = differential_residual(self.logR, self.y, g, self.lambda0, self.params)
        scale = np.maximum(np.abs(differential_residual(self.logR, self.y, np.zeros(2), self.lambda0, self.params)), 1.0)

        self.assertLess(float(np.max(np.abs(residual) / scale)), 1.0e-7)

    def test_sonic_diagnostics_are_finite(self) -> None:
        diagnostics = sonic_diagnostics(self.logR, self.y, self.lambda0, self.params)

        self.assertTrue(np.isfinite(diagnostics.D))
        self.assertTrue(np.isfinite(diagnostics.C1))
        self.assertTrue(np.isfinite(diagnostics.C2))
        self.assertTrue(np.isfinite(diagnostics.N))
        self.assertTrue(np.isfinite(diagnostics.smin_over_smax))
        self.assertTrue(np.isfinite(diagnostics.null_radial_fraction))
        self.assertTrue(np.isfinite(diagnostics.M_eff))
        self.assertGreater(diagnostics.radial_scale, 0.0)
        self.assertGreater(diagnostics.energy_scale, 0.0)
        self.assertEqual(diagnostics.singular_values.shape, (2,))
        self.assertEqual(diagnostics.left_null.shape, (2,))
        self.assertEqual(diagnostics.right_null.shape, (2,))

    def test_xi_eff_from_gradient_is_finite(self) -> None:
        g = np.array([-0.5, -0.75])
        xi = xi_eff_from_gradient(self.logR, self.y, g, self.lambda0, self.params)

        self.assertTrue(np.isfinite(xi))


if __name__ == "__main__":
    unittest.main()
