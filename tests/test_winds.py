from __future__ import annotations

import unittest

from imri_qpe.layer3_minidisk_1d import energy_limited_wind, q_available, q_edd_vertical, wind_energy_per_mass


class WindClosureTests(unittest.TestCase):
    def test_vertical_eddington_flux_is_positive(self) -> None:
        self.assertGreater(q_edd_vertical(1.0e-3, 1.0e10), 0.0)

    def test_available_energy_subtracts_advection(self) -> None:
        self.assertEqual(q_available(10.0, Q_stream=2.0, Q_tide=1.0, Q_adv=4.0), 9.0)

    def test_wind_energy_per_mass_keplerian_unbinding(self) -> None:
        E_w = wind_energy_per_mass(2.0, 4.0)

        self.assertGreater(E_w, 0.0)

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

    def test_energy_limited_wind_disallows_negative_radiation(self) -> None:
        Q_wind, Q_rad, dotSigma_w = energy_limited_wind(Q_avail=-3.0, Q_edd=6.0, E_w=2.0, epsilon_w=1.0)

        self.assertEqual(Q_wind, 0.0)
        self.assertEqual(Q_rad, 0.0)
        self.assertEqual(dotSigma_w, 0.0)


if __name__ == "__main__":
    unittest.main()
