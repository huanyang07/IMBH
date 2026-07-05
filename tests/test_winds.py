from __future__ import annotations

import math
import unittest

from imri_qpe.layer3_minidisk_1d import (
    effective_wind_powerlaw_slope,
    energy_limited_wind,
    energy_limited_wind_derivatives,
    q_available,
    q_edd_vertical,
    required_wind_energy_for_powerlaw_slope,
    wind_energy_per_mass,
    wind_mass_loss_prime_from_energy,
)


class WindClosureTests(unittest.TestCase):
    def test_vertical_eddington_flux_is_positive(self) -> None:
        self.assertGreater(q_edd_vertical(1.0e-3, 1.0e10), 0.0)

    def test_available_energy_subtracts_advection(self) -> None:
        self.assertEqual(q_available(10.0, Q_stream=2.0, Q_tide=1.0, Q_adv=4.0), 9.0)

    def test_wind_energy_per_mass_keplerian_unbinding(self) -> None:
        E_w = wind_energy_per_mass(2.0, 4.0)

        self.assertGreater(E_w, 0.0)

    def test_wind_mass_loss_prime_from_energy(self) -> None:
        mdot_prime = wind_mass_loss_prime_from_energy(Q_wind=3.0, R_cm=2.0, E_w=6.0)

        self.assertAlmostEqual(mdot_prime, 4.0 * math.pi)

    def test_effective_wind_powerlaw_slope(self) -> None:
        s_eff = effective_wind_powerlaw_slope(Q_wind=3.0, R_cm=2.0, Mdot=8.0 * math.pi, E_w=6.0)

        self.assertAlmostEqual(s_eff, 0.5)

    def test_required_wind_energy_for_powerlaw_slope(self) -> None:
        E_required = required_wind_energy_for_powerlaw_slope(
            Q_wind=3.0,
            R_cm=2.0,
            Mdot=8.0 * math.pi,
            s_target=0.25,
        )

        self.assertAlmostEqual(E_required, 12.0)

    def test_energy_limited_wind_partition(self) -> None:
        Q_wind, Q_rad, dotSigma_w = energy_limited_wind(Q_avail=10.0, Q_edd=6.0, E_w=2.0, epsilon_w=0.25)

        self.assertEqual(Q_wind, 1.0)
        self.assertEqual(Q_rad, 9.0)
        self.assertEqual(dotSigma_w, 0.5)
        self.assertEqual(Q_wind + Q_rad, 10.0)

    def test_energy_limited_wind_inactive_below_eddington(self) -> None:
        Q_wind, Q_rad, dotSigma_w = energy_limited_wind(Q_avail=5.0, Q_edd=6.0, E_w=2.0, epsilon_w=1.0)

        self.assertEqual(Q_wind, 0.0)
        self.assertEqual(Q_rad, 5.0)
        self.assertEqual(dotSigma_w, 0.0)

    def test_energy_limited_wind_soft_threshold(self) -> None:
        Q_wind, Q_rad, dotSigma_w = energy_limited_wind(
            Q_avail=9.95,
            Q_edd=10.0,
            E_w=2.0,
            epsilon_w=0.5,
            chi_edd=0.99,
        )

        self.assertAlmostEqual(Q_wind, 0.025)
        self.assertAlmostEqual(Q_rad, 9.925)
        self.assertAlmostEqual(dotSigma_w, 0.0125)
        self.assertAlmostEqual(Q_wind + Q_rad, 9.95)

    def test_energy_limited_wind_smooth_activation(self) -> None:
        Q_wind, Q_rad, dotSigma_w = energy_limited_wind(
            Q_avail=9.9,
            Q_edd=10.0,
            E_w=2.0,
            epsilon_w=0.5,
            chi_edd=0.99,
            activation_width=0.1,
        )

        expected_wind = 0.5 * 0.1 * math.log(2.0)
        self.assertAlmostEqual(Q_wind, expected_wind)
        self.assertAlmostEqual(Q_rad, 9.9 - expected_wind)
        self.assertAlmostEqual(dotSigma_w, expected_wind / 2.0)
        self.assertAlmostEqual(Q_wind + Q_rad, 9.9)

    def test_energy_limited_wind_smooth_activation_caps_excess(self) -> None:
        Q_wind, Q_rad, dotSigma_w = energy_limited_wind(
            Q_avail=0.0,
            Q_edd=10.0,
            E_w=2.0,
            epsilon_w=1.0,
            chi_edd=0.99,
            activation_width=10.0,
        )

        self.assertEqual(Q_wind, 0.0)
        self.assertEqual(Q_rad, 0.0)
        self.assertEqual(dotSigma_w, 0.0)

    def test_energy_limited_wind_disallows_negative_radiation(self) -> None:
        Q_wind, Q_rad, dotSigma_w = energy_limited_wind(Q_avail=-3.0, Q_edd=6.0, E_w=2.0, epsilon_w=1.0)

        self.assertEqual(Q_wind, 0.0)
        self.assertEqual(Q_rad, 0.0)
        self.assertEqual(dotSigma_w, 0.0)

    def test_energy_limited_wind_derivatives_match_centered_difference(self) -> None:
        q_avail = 10.2
        q_edd = 10.0
        epsilon = 0.3
        chi = 0.99
        width_fraction = 0.02
        width = width_fraction * q_edd
        dQdQa, dQdQe = energy_limited_wind_derivatives(
            q_avail,
            q_edd,
            epsilon,
            chi_edd=chi,
            activation_width=width,
            activation_width_dQedd=width_fraction,
        )
        h = 1.0e-5
        plus_a = energy_limited_wind(
            q_avail + h,
            q_edd,
            E_w=2.0,
            epsilon_w=epsilon,
            chi_edd=chi,
            activation_width=width,
        )[0]
        minus_a = energy_limited_wind(
            q_avail - h,
            q_edd,
            E_w=2.0,
            epsilon_w=epsilon,
            chi_edd=chi,
            activation_width=width,
        )[0]
        plus_e = energy_limited_wind(
            q_avail,
            q_edd + h,
            E_w=2.0,
            epsilon_w=epsilon,
            chi_edd=chi,
            activation_width=width_fraction * (q_edd + h),
        )[0]
        minus_e = energy_limited_wind(
            q_avail,
            q_edd - h,
            E_w=2.0,
            epsilon_w=epsilon,
            chi_edd=chi,
            activation_width=width_fraction * (q_edd - h),
        )[0]

        self.assertAlmostEqual(dQdQa, (plus_a - minus_a) / (2.0 * h), places=8)
        self.assertAlmostEqual(dQdQe, (plus_e - minus_e) / (2.0 * h), places=8)


if __name__ == "__main__":
    unittest.main()
