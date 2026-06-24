from __future__ import annotations

import math
import unittest

import numpy as np

from imri_qpe.constants import DAY, M_SUN
from imri_qpe.layer1_hill_flow import hill_radius
from imri_qpe.layer3_minidisk_1d import (
    explicit_diffusion_step,
    gaussian_source,
    gaussian_source_on_grid,
    make_log_grid,
    mass_from_surface_density,
    one_zone_cycle,
    prescribed_scurve_step,
    update_branch_state,
    viscous_diffusion_rhs,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import omega_k
from imri_qpe.units import g_per_s_to_msun_per_year, msun_per_year_to_g_per_s, solar_masses_to_g


class OneZoneCycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.params = FiducialParams()
        R_u = 0.3 * hill_radius(self.params.a_cm, self.params.q)
        self.Omega_K = omega_k(self.params.M2_g, R_u)

    def assert_rel_close(self, value: float, expected: float, rel: float) -> None:
        self.assertLessEqual(abs(value - expected), rel * abs(expected))

    def test_fiducial_one_zone_cycle_matches_note(self) -> None:
        result = one_zone_cycle(
            Mmax=solar_masses_to_g(1.6e-4),
            zeta=0.2,
            mdot_cap=msun_per_year_to_g_per_s(1.0e-2),
            mdot_low=0.0,
            alpha_hot=0.1,
            H_over_R_hot=0.1,
            Omega_K=self.Omega_K,
        )

        self.assert_rel_close(result.delta_M / M_SUN, 1.28e-4, rel=1.0e-12)
        self.assert_rel_close(result.t_load / DAY, 4.8, rel=0.04)
        self.assert_rel_close(result.t_high / DAY, 1.9, rel=0.02)
        self.assert_rel_close(result.P_QPE / DAY, 6.7, rel=0.03)
        self.assert_rel_close(result.duty_cycle, 0.29, rel=0.02)
        self.assert_rel_close(g_per_s_to_msun_per_year(result.mdot_burst), 2.5e-2, rel=0.03)

    def test_one_zone_cycle_rejects_non_loading_case(self) -> None:
        with self.assertRaises(ValueError):
            one_zone_cycle(1.0, 0.2, mdot_cap=1.0, mdot_low=1.0, alpha_hot=0.1, H_over_R_hot=0.1, Omega_K=1.0)


class GridAndSourceTests(unittest.TestCase):
    def test_log_grid_has_expected_geometry(self) -> None:
        grid = make_log_grid(1.0, 100.0, 4)

        self.assertEqual(len(grid.centers), 4)
        np.testing.assert_allclose(grid.centers, np.sqrt(grid.edges[:-1] * grid.edges[1:]))
        self.assertAlmostEqual(np.sum(grid.area), math.pi * (100.0**2 - 1.0**2))

    def test_gaussian_source_integrates_to_requested_rate(self) -> None:
        R = np.geomspace(1.0e10, 1.0e12, 500)
        source = gaussian_source(R, R_c=3.0e11, width=5.0e10, Mdot_cap=1.0e20)
        total = np.trapezoid(2.0 * math.pi * R * source, R)

        self.assertAlmostEqual(total / 1.0e20, 1.0, places=12)

    def test_grid_gaussian_source_uses_annular_areas(self) -> None:
        grid = make_log_grid(1.0e10, 1.0e12, 96)
        source = gaussian_source_on_grid(grid, R_c=3.0e11, width=5.0e10, Mdot_cap=2.5e20)

        self.assertAlmostEqual(np.sum(source * grid.area) / 2.5e20, 1.0, places=12)


class DiffusionSolverTests(unittest.TestCase):
    def test_zero_viscosity_and_no_source_is_static(self) -> None:
        grid = make_log_grid(1.0, 100.0, 16)
        Sigma = np.geomspace(1.0, 10.0, len(grid.centers))
        Sigma_new = explicit_diffusion_step(grid, Sigma, nu=0.0, dt=100.0)

        np.testing.assert_allclose(Sigma_new, Sigma)

    def test_source_only_step_increases_mass_by_mdot_dt(self) -> None:
        grid = make_log_grid(1.0e10, 1.0e12, 64)
        Sigma = np.full_like(grid.centers, 10.0)
        source = gaussian_source_on_grid(grid, R_c=3.0e11, width=5.0e10, Mdot_cap=1.0e20)
        dt = 123.0

        mass_before = mass_from_surface_density(grid, Sigma)
        Sigma_new = explicit_diffusion_step(grid, Sigma, nu=0.0, dt=dt, source=source)
        mass_after = mass_from_surface_density(grid, Sigma_new)

        self.assertAlmostEqual((mass_after - mass_before) / (1.0e20 * dt), 1.0, places=12)

    def test_viscous_rhs_conserves_mass_with_zero_flux_boundaries(self) -> None:
        grid = make_log_grid(1.0, 100.0, 64)
        Sigma = 1.0 + np.exp(-0.5 * ((np.log(grid.centers) - np.log(10.0)) / 0.5) ** 2)
        nu = np.full_like(Sigma, 1.0e-2)
        rhs = viscous_diffusion_rhs(grid, Sigma, nu)

        self.assertAlmostEqual(np.sum(rhs * grid.area), 0.0, places=12)

    def test_branch_state_hysteresis(self) -> None:
        Sigma = np.array([0.8, 1.5, 2.5, 0.8])
        previous_hot = np.array([True, True, False, False])
        updated = update_branch_state(Sigma, Sigma_min=1.0, Sigma_max=2.0, is_hot=previous_hot)

        np.testing.assert_array_equal(updated, np.array([False, True, True, False]))

    def test_prescribed_scurve_step_returns_hot_viscosity_for_hot_cells(self) -> None:
        grid = make_log_grid(1.0, 10.0, 4)
        Sigma = np.array([0.8, 1.5, 2.5, 0.8])
        previous_hot = np.array([False, True, False, False])
        result = prescribed_scurve_step(
            grid,
            Sigma,
            previous_hot,
            dt=0.0,
            nu_cold=1.0,
            nu_hot=10.0,
            Sigma_min=1.0,
            Sigma_max=2.0,
        )

        np.testing.assert_array_equal(result.is_hot, np.array([False, True, True, False]))
        np.testing.assert_allclose(result.nu, np.array([1.0, 10.0, 10.0, 1.0]))
        np.testing.assert_allclose(result.Sigma, Sigma)


if __name__ == "__main__":
    unittest.main()

