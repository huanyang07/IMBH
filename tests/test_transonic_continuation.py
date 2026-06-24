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
from imri_qpe.layer3_minidisk_1d.transonic_continuation import remap_profile_to_new_sonic_grid
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


if __name__ == "__main__":
    unittest.main()
