from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from imri_qpe.layer3_minidisk_1d.transonic_collocation import TransonicSlimParams
from imri_qpe.layer3_minidisk_1d.transonic_plunge import (
    TransonicPlungeProfile,
    _resolved_outer_gradient,
    continue_transonic_supersonic_plunge,
)
from imri_qpe.scales import eddington_mdot
from imri_qpe.units import solar_masses_to_g


class TransonicPlungeTests(unittest.TestCase):
    def setUp(self) -> None:
        mass = solar_masses_to_g(1.0e4)
        self.params = TransonicSlimParams(
            M2_g=mass,
            Mdot_g_s=eddington_mdot(mass),
            alpha=0.1,
            n_nodes=8,
            R_out_rg=100.0,
        )

    def test_resolved_outer_gradient_uses_first_interval(self) -> None:
        profile = SimpleNamespace(
            R=np.asarray([2.0, 4.0]),
            u=np.asarray([8.0, 2.0]),
            T=np.asarray([3.0, 6.0]),
        )

        gradient = _resolved_outer_gradient(profile)

        np.testing.assert_allclose(gradient, [-2.0, 1.0])

    def test_profile_reports_causally_outgoing_inner_edge(self) -> None:
        values = np.ones(3)
        profile = TransonicPlungeProfile(
            R=values,
            u=values,
            T=values,
            Sigma=values,
            H=values,
            rho=values,
            P=values,
            Pi=values,
            e=values,
            tau=values,
            Omega=values,
            Omega_K=values,
            l=values,
            W=values,
            effective_sound_speed=values,
            radial_mach_number=-values,
            incoming_characteristics=np.asarray([0, 0, 1]),
            selected_sonic_gradient=np.zeros(2),
            resolved_outer_gradient=np.zeros(2),
            sonic_gradient_mismatch=0.0,
            maximum_scaled_differential_residual=0.0,
            sonic_offset=1.0e-6,
        )

        self.assertTrue(profile.inner_is_causally_outgoing)

    def test_inner_radius_must_exceed_paczynski_wiita_radius(self) -> None:
        profile = SimpleNamespace(
            sonic_radius=5.0 * self.params.r_g,
        )

        with self.assertRaisesRegex(ValueError, "between r_pw and R_son"):
            continue_transonic_supersonic_plunge(
                profile,
                self.params,
                self.params.potential.r_pw,
            )


if __name__ == "__main__":
    unittest.main()
