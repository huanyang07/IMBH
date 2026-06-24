from __future__ import annotations

import unittest

import numpy as np

from imri_qpe.constants import C, G
from imri_qpe.layer1_hill_flow import hill_radius
from imri_qpe.layer1_hill_flow.hill_geometry import schwarzschild_isco_radius
from imri_qpe.layer3_minidisk_1d import (
    IsolatedSlimParams,
    choose_sigma_root,
    evaluate_isolated_slim_profile,
    isolated_residual_vector,
    make_log_grid,
    required_keplerian_stress,
    sigma_roots_for_temperature,
    solve_isolated_slim_disk,
    solve_sigma_profile_from_temperature,
    state_vector_from_profile,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot


class IsolatedSlimSolverTests(unittest.TestCase):
    def setUp(self) -> None:
        params = FiducialParams()
        self.params = params
        self.R_H = hill_radius(params.a_cm, params.q)
        self.R_in = 1.2 * schwarzschild_isco_radius(params.M2_g)
        self.grid = make_log_grid(self.R_in, 0.12 * self.R_H, 6)
        self.slim_params = IsolatedSlimParams(
            params.M2_g,
            0.01 * eddington_mdot(params.M2_g),
            self.R_in,
            sigma_brackets=120,
            T_bounds=(1.0e5, 5.0e6),
        )

    def test_required_stress_is_positive_outside_inner_boundary(self) -> None:
        stress = required_keplerian_stress(
            self.slim_params.Mdot_g_s,
            self.slim_params.M2_g,
            self.grid.centers,
            self.slim_params.inner_angular_momentum,
        )

        self.assertTrue(np.all(stress > 0.0))

    def test_default_inner_angular_momentum_uses_physical_isco(self) -> None:
        R_isco = 6.0 * G * self.params.M2_g / C**2
        expected = np.sqrt(G * self.params.M2_g * R_isco)

        self.assertAlmostEqual(self.slim_params.inner_angular_momentum / expected, 1.0)

    def test_sigma_roots_and_branch_choice(self) -> None:
        roots = sigma_roots_for_temperature(1.5e6, float(self.grid.centers[2]), self.slim_params)

        self.assertGreaterEqual(len(roots), 1)
        self.assertEqual(choose_sigma_root(roots, branch="largest"), float(np.max(roots)))
        self.assertEqual(choose_sigma_root(roots, branch="smallest"), float(np.min(roots)))

    def test_sigma_profile_satisfies_angular_momentum_closure(self) -> None:
        T = np.full_like(self.grid.centers, 1.5e6)
        Sigma, message = solve_sigma_profile_from_temperature(self.grid, T, self.slim_params)

        self.assertEqual(message, "ok")
        self.assertIsNotNone(Sigma)
        profile = evaluate_isolated_slim_profile(self.grid, Sigma, T, self.slim_params)

        self.assertTrue(np.allclose(profile.Mdot, self.slim_params.Mdot_g_s))
        self.assertLess(np.nanmax(np.abs(profile.angular_momentum_residual)), 1.0e-10)

    def test_two_face_viscous_flux_matches_thin_disk_formula(self) -> None:
        T = np.full_like(self.grid.centers, 1.5e6)
        Sigma, _ = solve_sigma_profile_from_temperature(self.grid, T, self.slim_params)
        profile = evaluate_isolated_slim_profile(self.grid, Sigma, T, self.slim_params)
        R0 = (self.slim_params.inner_angular_momentum**2) / (G * self.slim_params.M2_g)
        expected = (
            3.0
            * G
            * self.slim_params.M2_g
            * self.slim_params.Mdot_g_s
            / (4.0 * np.pi * profile.R**3)
            * (1.0 - np.sqrt(R0 / profile.R))
        )

        np.testing.assert_allclose(profile.Q_visc, expected, rtol=1.0e-10)

    def test_state_vector_residual_has_small_angular_part_for_closed_profile(self) -> None:
        T = np.full_like(self.grid.centers, 1.5e6)
        Sigma, _ = solve_sigma_profile_from_temperature(self.grid, T, self.slim_params)
        z = state_vector_from_profile(Sigma, T)

        residual = isolated_residual_vector(self.grid, z, self.slim_params)

        self.assertLess(np.max(np.abs(residual[: len(self.grid.centers)])), 1.0e-10)

    def test_relaxation_returns_finite_best_profile_when_not_converged(self) -> None:
        T = np.full_like(self.grid.centers, 1.5e6)
        Sigma, _ = solve_sigma_profile_from_temperature(self.grid, T, self.slim_params)

        result = solve_isolated_slim_disk(
            self.grid,
            self.slim_params,
            T_initial=T,
            Sigma_initial=Sigma,
            max_iter=2,
            boundary_exclude=1,
        )

        self.assertFalse(result.failed)
        self.assertIsNotNone(result.profile)
        self.assertGreaterEqual(len(result.history), 1)
        self.assertTrue(np.all(np.isfinite(result.history)))


if __name__ == "__main__":
    unittest.main()
