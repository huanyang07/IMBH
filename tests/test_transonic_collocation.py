from __future__ import annotations

import unittest

import numpy as np

from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d.transonic_collocation import (
    TransonicSlimParams,
    collocation_residual,
    computational_grid,
    jac_sparsity_pattern,
    pack_state,
    profile_from_state_vector,
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

    def test_state_bounds_match_unknown_vector(self) -> None:
        lower, upper = state_bounds(self.params)

        self.assertEqual(lower.shape, self.z.shape)
        self.assertEqual(upper.shape, self.z.shape)
        self.assertTrue(np.all(upper > lower))

    def test_collocation_residual_has_expected_shape(self) -> None:
        residual = collocation_residual(self.z, self.params)

        self.assertEqual(residual.shape, self.z.shape)
        self.assertTrue(np.all(np.isfinite(residual)))
        self.assertLess(float(np.max(np.abs(residual))), 1.0e6)

    def test_jac_sparsity_shape(self) -> None:
        pattern = jac_sparsity_pattern(self.params)

        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.shape, (self.z.size, self.z.size))
        self.assertGreater(pattern.nnz, 0)

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
        ):
            self.assertTrue(np.all(np.isfinite(array)))
        self.assertGreater(profile.sonic_radius, self.params.potential.r_pw)


if __name__ == "__main__":
    unittest.main()
