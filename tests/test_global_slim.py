from __future__ import annotations

import unittest

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    GlobalSlimParams,
    evaluate_global_slim_profile,
    integrated_energy_residual,
    make_log_grid,
    radial_velocity_from_angular_momentum,
    relax_temperature_energy_balance,
    vertical_structure_arrays,
)
from imri_qpe.units import solar_masses_to_g


class GlobalSlimDiagnosticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.M2_g = solar_masses_to_g(1.0e4)
        self.grid = make_log_grid(1.0e10, 1.0e12, 48)
        x = self.grid.centers / self.grid.centers[0]
        self.Sigma = 2.0e5 * x**-0.4
        self.T = 2.0e6 * x**-0.6

    def test_vertical_structure_arrays_are_finite_and_positive(self) -> None:
        Omega, H, rho, P, e, tau = vertical_structure_arrays(self.Sigma, self.T, self.M2_g, self.grid.centers)

        for array in (Omega, H, rho, P, e, tau):
            self.assertTrue(np.all(np.isfinite(array)))
            self.assertTrue(np.all(array > 0.0))

    def test_stress_gradient_drives_inward_flow(self) -> None:
        R = self.grid.centers
        Sigma = np.full_like(R, 1.0e4)
        W = np.full_like(R, 1.0e15)

        v_R = radial_velocity_from_angular_momentum(R, Sigma, W, self.M2_g, zero_inner_torque=False)

        self.assertTrue(np.all(v_R < 0.0))

    def test_evaluate_global_profile_energy_accounting(self) -> None:
        params = GlobalSlimParams(self.M2_g, alpha=0.02, epsilon_wind=0.4, zero_inner_torque=False)

        profile = evaluate_global_slim_profile(self.grid, self.Sigma, self.T, params)

        self.assertEqual(profile.R.shape, self.grid.centers.shape)
        self.assertEqual(profile.area.shape, self.grid.area.shape)
        for array in (
            profile.v_R,
            profile.Mdot,
            profile.TdsdR,
            profile.xi_eff,
            profile.Q_visc,
            profile.Q_adv,
            profile.Q_wind,
            profile.energy_residual,
        ):
            self.assertTrue(np.all(np.isfinite(array)))

        np.testing.assert_allclose(profile.Q_adv, profile.Sigma * profile.v_R * profile.TdsdR)
        available = profile.Q_visc - profile.Q_adv
        np.testing.assert_allclose(profile.Q_rad_limited + profile.Q_wind, np.maximum(available, 0.0))
        np.testing.assert_allclose(
            profile.energy_residual,
            np.minimum(available, 0.0),
            atol=1.0e-12 * float(np.max(np.abs(profile.Q_visc))),
        )
        self.assertTrue(np.isfinite(integrated_energy_residual(profile)))

    def test_integrated_residual_uses_grid_area(self) -> None:
        params = GlobalSlimParams(self.M2_g, alpha=0.02, epsilon_wind=0.0, zero_inner_torque=False)
        profile = evaluate_global_slim_profile(self.grid, self.Sigma, self.T, params)

        expected = np.sum(profile.energy_residual * self.grid.area) / np.sum(profile.Q_visc * self.grid.area)

        self.assertAlmostEqual(integrated_energy_residual(profile), expected)

    def test_no_wind_residual_uses_diffusion_radiation(self) -> None:
        params = GlobalSlimParams(self.M2_g, alpha=0.02, epsilon_wind=0.0, zero_inner_torque=False)

        profile = evaluate_global_slim_profile(self.grid, self.Sigma, self.T, params)

        np.testing.assert_allclose(profile.Q_wind, 0.0)
        np.testing.assert_allclose(profile.Q_rad_limited, profile.Q_rad_diffusion)
        np.testing.assert_allclose(profile.energy_residual, profile.Q_visc - profile.Q_rad_diffusion - profile.Q_adv)

    def test_temperature_relaxation_returns_finite_diagnostic(self) -> None:
        params = GlobalSlimParams(self.M2_g, alpha=0.02, epsilon_wind=0.2, zero_inner_torque=False)

        result = relax_temperature_energy_balance(
            self.grid,
            self.Sigma,
            self.T,
            params,
            max_iter=2,
            damping=0.1,
        )

        self.assertLessEqual(result.iterations, 2)
        self.assertGreaterEqual(len(result.history), 1)
        self.assertTrue(np.all(np.isfinite(result.history)))
        self.assertTrue(np.all(np.isfinite(result.profile.T)))
        self.assertTrue(np.isfinite(result.max_normalized_residual))


if __name__ == "__main__":
    unittest.main()
