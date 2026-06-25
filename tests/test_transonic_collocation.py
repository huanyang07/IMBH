from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d.transonic_collocation import (
    TransonicSlimParams,
    collocation_jacobian,
    collocation_residual,
    computational_grid,
    jac_sparsity_pattern,
    pack_state,
    profile_from_state_vector,
    residual_audit_from_state_vector,
    solve_low_mdot_transonic_homotopy,
    state_bounds,
    unpack_state,
)
from imri_qpe.layer3_minidisk_1d.transonic_potential import PaczynskiWiitaPotential
from imri_qpe.scales import eddington_mdot
from imri_qpe.units import solar_masses_to_g


class TransonicCollocationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.M2_g = solar_masses_to_g(1.0e4)
        self.params = TransonicSlimParams(
            M2_g=self.M2_g,
            Mdot_g_s=1.0e-3 * eddington_mdot(self.M2_g),
            alpha=0.01,
            n_nodes=10,
            R_out_rg=300.0,
            max_nfev=10,
        )
        potential = PaczynskiWiitaPotential(self.M2_g)
        logR_son = np.log(potential.r_isco)
        logR = computational_grid(self.params, logR_son)
        xi = (logR - logR[0]) / (logR[-1] - logR[0])
        logu = np.log(2.0e7 * (1.0 - xi) + 2.0e5 * xi)
        logT = np.log(3.0e6 * np.exp(-0.6 * (logR - logR[0])))
        lambda0 = float(potential.l_k(potential.r_isco) / (potential.r_g * C))
        self.z = pack_state(logu, logT, logR_son, lambda0)

    def test_pack_unpack_round_trip(self) -> None:
        logu, logT, logR_son, lambda0, logR = unpack_state(self.z, self.params)
        repacked = pack_state(logu, logT, logR_son, lambda0)

        np.testing.assert_allclose(repacked, self.z)
        self.assertEqual(logR.shape, (self.params.n_nodes,))
        self.assertTrue(np.all(np.diff(logR) > 0.0))

    def test_grid_power_clusters_nodes_near_sonic_point(self) -> None:
        clustered = replace(self.params, grid_power=2.0)
        logR_son = np.log(self.params.potential.r_isco)
        uniform_logR = computational_grid(self.params, logR_son)
        clustered_logR = computational_grid(clustered, logR_son)

        self.assertEqual(clustered_logR[0], uniform_logR[0])
        self.assertEqual(clustered_logR[-1], uniform_logR[-1])
        self.assertLess(clustered_logR[1] - clustered_logR[0], uniform_logR[1] - uniform_logR[0])
        self.assertTrue(np.all(np.diff(clustered_logR) > 0.0))

    def test_state_bounds_match_unknown_vector(self) -> None:
        lower, upper = state_bounds(self.params)

        self.assertEqual(lower.shape, self.z.shape)
        self.assertEqual(upper.shape, self.z.shape)
        self.assertTrue(np.all(upper > lower))

    def test_collocation_residual_has_expected_shape(self) -> None:
        residual = collocation_residual(self.z, self.params)

        self.assertEqual(residual.shape, (self.z.size + 1,))
        self.assertTrue(np.all(np.isfinite(residual)))
        self.assertLess(float(np.max(np.abs(residual))), 1.0e6)

    def test_jac_sparsity_shape(self) -> None:
        pattern = jac_sparsity_pattern(self.params)

        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.shape, (self.z.size + 1, self.z.size))
        self.assertGreater(pattern.nnz, 0)

    def test_block_local_jacobian_matches_full_finite_difference_columns(self) -> None:
        jac = collocation_jacobian(self.z, self.params).toarray()
        columns = [0, 1, self.params.n_nodes, self.params.n_nodes + 1, self.z.size - 2, self.z.size - 1]
        for column in columns:
            step = 1.0e-6 * max(1.0, abs(float(self.z[column])))
            plus = np.array(self.z, copy=True)
            minus = np.array(self.z, copy=True)
            plus[column] += step
            minus[column] -= step
            finite = (collocation_residual(plus, self.params) - collocation_residual(minus, self.params)) / (2.0 * step)
            scale = np.maximum(np.maximum(np.abs(finite), np.abs(jac[:, column])), 1.0)
            self.assertLess(float(np.max(np.abs(jac[:, column] - finite) / scale)), 1.0e-4, column)

    def test_profile_from_state_vector_is_finite(self) -> None:
        profile = profile_from_state_vector(self.z, self.params)

        self.assertEqual(profile.R.shape, (self.params.n_nodes,))
        for array in (
            profile.u,
            profile.T,
            profile.Sigma,
            profile.H,
            profile.Omega,
            profile.Q_rad,
            profile.sonic_D,
            profile.sonic_C1,
            profile.sonic_C2,
            profile.sonic_smin_over_smax,
            profile.sonic_null_radial_fraction,
            profile.sonic_M_eff,
        ):
            self.assertTrue(np.all(np.isfinite(array)))
        self.assertGreater(profile.sonic_radius, self.params.potential.r_pw)

    def test_residual_audit_reports_blocks(self) -> None:
        audit = residual_audit_from_state_vector(self.z, self.params)

        self.assertTrue(np.isfinite(audit.interval_radial_max))
        self.assertTrue(np.isfinite(audit.interval_energy_max))
        self.assertTrue(np.isfinite(audit.outer_omega))
        self.assertTrue(np.isfinite(audit.sonic_C1))
        self.assertTrue(np.isfinite(audit.lambda0_over_lK_isco))
        self.assertIsInstance(audit.active_bounds, tuple)

    def test_low_mdot_homotopy_returns_stages(self) -> None:
        params = replace(self.params, n_nodes=6, max_nfev=2)
        logu, logT, logR_son, lambda0, _logR = unpack_state(self.z, self.params)
        z0 = pack_state(logu[: params.n_nodes], logT[: params.n_nodes], logR_son, lambda0)
        result = solve_low_mdot_transonic_homotopy(params, initial_guess=z0, max_nfev_per_stage=1)

        self.assertEqual([stage.name for stage in result.stages], [
            "A_fixed_eigen_profile",
            "B_free_Rson_fixed_lambda",
            "B2_free_Rson_lambda_D_weight_1",
            "C_free_Rson_free_lambda_full",
        ])
        self.assertIsNotNone(result.final_result.profile)

    def test_low_mdot_homotopy_can_ramp_outer_weight(self) -> None:
        params = replace(self.params, n_nodes=6, max_nfev=2)
        logu, logT, logR_son, lambda0, _logR = unpack_state(self.z, self.params)
        z0 = pack_state(logu[: params.n_nodes], logT[: params.n_nodes], logR_son, lambda0)
        result = solve_low_mdot_transonic_homotopy(
            params,
            initial_guess=z0,
            max_nfev_per_stage=1,
            outer_weight_sequence=(0.5,),
        )

        self.assertIn("B3_outer_weight_0.5", [stage.name for stage in result.stages])
        self.assertIsNotNone(result.final_result.profile)


if __name__ == "__main__":
    unittest.main()
