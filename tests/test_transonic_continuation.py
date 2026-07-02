from __future__ import annotations

import unittest

import numpy as np

from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d.transonic_collocation import (
    TransonicSlimParams,
    computational_grid,
    pack_state,
    profile_from_state_vector,
    replace_mdot,
    unpack_state,
)
from imri_qpe.layer3_minidisk_1d.transonic_continuation import (
    blockwise_continuation_metric,
    pseudo_arclength_step,
    remap_profile_to_new_sonic_grid,
    tangent_audit_from_scaled_tangent,
)
from imri_qpe.layer3_minidisk_1d.transonic_potential import PaczynskiWiitaPotential
from imri_qpe.scales import eddington_mdot
from imri_qpe.units import solar_masses_to_g


class TransonicContinuationTests(unittest.TestCase):
    def setUp(self) -> None:
        M2_g = solar_masses_to_g(1.0e4)
        self.params = TransonicSlimParams(
            M2_g=M2_g,
            Mdot_g_s=1.0e-3 * eddington_mdot(M2_g),
            alpha=0.01,
            n_nodes=8,
            R_out_rg=300.0,
        )
        potential = PaczynskiWiitaPotential(M2_g)
        logR_son = np.log(potential.r_isco)
        logR = computational_grid(self.params, logR_son)
        xi = (logR - logR[0]) / (logR[-1] - logR[0])
        logu = np.log(1.0e7 * (1.0 - xi) + 1.0e5 * xi)
        logT = np.log(2.0e6 * np.exp(-0.5 * (logR - logR[0])))
        lambda0 = float(potential.l_k(potential.r_isco) / (potential.r_g * C))
        self.profile = profile_from_state_vector(pack_state(logu, logT, logR_son, lambda0), self.params)

    def test_replace_mdot_updates_only_accretion_rate(self) -> None:
        updated = replace_mdot(self.params, 2.0 * self.params.Mdot_g_s)

        self.assertEqual(updated.Mdot_g_s, 2.0 * self.params.Mdot_g_s)
        self.assertEqual(updated.n_nodes, self.params.n_nodes)

    def test_remap_profile_to_new_mdot_preserves_shape(self) -> None:
        new_params = replace_mdot(self.params, 3.0 * self.params.Mdot_g_s)
        z = remap_profile_to_new_sonic_grid(self.profile, new_params)
        logu, logT, logR_son, lambda0, logR = unpack_state(z, new_params)

        self.assertEqual(logu.shape, (new_params.n_nodes,))
        self.assertEqual(logT.shape, (new_params.n_nodes,))
        self.assertAlmostEqual(np.exp(logR_son), self.profile.sonic_radius)
        self.assertAlmostEqual(lambda0, self.profile.lambda0)
        self.assertTrue(np.all(np.diff(logR) > 0.0))

    def test_pchip_remap_profile_to_new_grid_preserves_bounds(self) -> None:
        new_params = replace_mdot(self.params, 2.0 * self.params.Mdot_g_s)
        new_params = TransonicSlimParams(**{**new_params.__dict__, "n_nodes": 11})
        z = remap_profile_to_new_sonic_grid(self.profile, new_params, method="pchip")
        logu, logT, logR_son, lambda0, logR = unpack_state(z, new_params)

        self.assertEqual(logu.shape, (11,))
        self.assertEqual(logT.shape, (11,))
        self.assertAlmostEqual(np.exp(logR_son), self.profile.sonic_radius)
        self.assertAlmostEqual(lambda0, self.profile.lambda0)
        self.assertTrue(np.all(np.diff(logR) > 0.0))
        self.assertTrue(np.all(np.isfinite(logu)))
        self.assertTrue(np.all(np.isfinite(logT)))

    def test_blockwise_continuation_metric_reports_tangent_fractions(self) -> None:
        previous_params = replace_mdot(self.params, self.params.Mdot_g_s)
        current_params = replace_mdot(self.params, 1.2 * self.params.Mdot_g_s)
        z_previous = remap_profile_to_new_sonic_grid(self.profile, previous_params)
        z_current = remap_profile_to_new_sonic_grid(self.profile, current_params)
        metric = blockwise_continuation_metric(z_previous, 1.0, z_current, 1.2, current_params)
        scales = metric.scale_vector()
        tangent = (np.concatenate([z_current, [np.log(1.2)]]) - np.concatenate([z_previous, [0.0]])) / scales
        tangent = tangent / np.linalg.norm(tangent)
        audit = tangent_audit_from_scaled_tangent(tangent, metric, method="secant")

        self.assertEqual(scales.shape, (z_current.size + 1,))
        self.assertGreaterEqual(metric.logu_scale, 2.0e-2)
        self.assertGreaterEqual(metric.logT_scale, 1.0e-2)
        self.assertAlmostEqual(
            audit.logu_fraction
            + audit.logT_fraction
            + audit.logR_son_fraction
            + audit.lambda0_fraction
            + audit.mu_fraction,
            1.0,
        )
        self.assertEqual(audit.method, "secant")

    def test_pseudo_arclength_step_accepts_jacobian_tangent_mode(self) -> None:
        previous_params = replace_mdot(self.params, self.params.Mdot_g_s)
        current_params = replace_mdot(self.params, 1.05 * self.params.Mdot_g_s)
        previous_profile = profile_from_state_vector(remap_profile_to_new_sonic_grid(self.profile, previous_params), previous_params)
        current_profile = profile_from_state_vector(remap_profile_to_new_sonic_grid(self.profile, current_params), current_params)

        result = pseudo_arclength_step(
            self.params,
            self.params.Mdot_g_s,
            previous_profile,
            1.0,
            current_profile,
            1.05,
            step_multiplier=0.1,
            max_nfev=1,
            residual_tol=1.0e-3,
            residual_mode="square",
            sonic_pivot="K",
            metric_mode="blockwise",
            tangent_mode="jacobian",
        )

        self.assertEqual(result.tangent.shape, (2 * self.params.n_nodes + 3,))
        self.assertEqual(result.metric.n_nodes, self.params.n_nodes)
        self.assertIn(result.tangent_audit.method, {"jacobian", "jacobian_mu", "secant_fallback"})
        self.assertIn(result.corrector_method, {"bordered_newton", "hybrid_least_squares", "least_squares"})
        self.assertTrue(np.isfinite(result.initial_max_residual))
        self.assertTrue(np.isfinite(result.predictor_correction_norm))
        self.assertTrue(np.isfinite(result.arclength_residual))


if __name__ == "__main__":
    unittest.main()
