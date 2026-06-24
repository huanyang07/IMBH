from __future__ import annotations

import math
import unittest

import numpy as np

from imri_qpe.layer1_hill_flow.capture_diagnostics import (
    capture_fraction,
    mean_specific_angular_momentum,
    recycling_time,
)
from imri_qpe.layer1_hill_flow.hill_geometry import (
    binary_omega,
    circularization_radius,
    hill_radius,
    is_minidisk_allowed,
    schwarzschild_isco_radius,
    tidal_truncation_radius,
)
from imri_qpe.layer1_hill_flow.stress_diagnostics import alpha_shock, reynolds_stress


class HillGeometryTests(unittest.TestCase):
    def test_invalid_geometry_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            hill_radius(1.0, 0.0)
        with self.assertRaises(ValueError):
            binary_omega(-1.0, 1.0)
        with self.assertRaises(ValueError):
            tidal_truncation_radius(1.0, 0.0)

    def test_minidisk_allowed_supports_arrays(self) -> None:
        R_isco = 1.0
        R_out = 10.0
        allowed = is_minidisk_allowed(np.array([0.5, 5.0, 12.0]), R_isco, R_out)

        np.testing.assert_array_equal(allowed, np.array([False, True, False]))

    def test_circularization_radius_uses_lambda_squared(self) -> None:
        self.assertEqual(circularization_radius(9.0, 2.0), 12.0)
        self.assertEqual(circularization_radius(9.0, -2.0), 12.0)

    def test_schwarzschild_isco_is_positive(self) -> None:
        self.assertGreater(schwarzschild_isco_radius(1.0), 0.0)


class CaptureDiagnosticsTests(unittest.TestCase):
    def test_capture_fraction_is_zero_safe(self) -> None:
        self.assertEqual(capture_fraction(2.0, 10.0), 0.2)
        self.assertEqual(capture_fraction(2.0, 0.0), 0.0)

        result = capture_fraction(np.array([1.0, 2.0, 3.0]), np.array([2.0, 0.0, 6.0]))
        np.testing.assert_allclose(result, np.array([0.5, 0.0, 0.5]))

    def test_mean_specific_angular_momentum_from_fluxes(self) -> None:
        self.assertEqual(mean_specific_angular_momentum([2.0, 3.0], [20.0, 45.0]), 13.0)
        self.assertEqual(mean_specific_angular_momentum([0.0, 0.0], [1.0, 2.0]), 0.0)

    def test_recycling_time_uses_uncaptured_inflow(self) -> None:
        self.assertEqual(recycling_time(10.0, 4.0, 2.0), 5.0)
        self.assertTrue(math.isinf(recycling_time(10.0, 2.0, 4.0)))

        result = recycling_time(np.array([10.0, 10.0]), np.array([5.0, 1.0]), np.array([3.0, 2.0]))
        np.testing.assert_allclose(result[0], 5.0)
        self.assertTrue(math.isinf(result[1]))


class StressDiagnosticsTests(unittest.TestCase):
    def test_reynolds_stress_integrates_last_axis(self) -> None:
        rho = np.array([1.0, 2.0, 3.0])
        dv_R = np.array([2.0, 2.0, 2.0])
        dv_phi = np.array([0.5, 1.0, 1.5])

        self.assertAlmostEqual(reynolds_stress(rho, dv_R, dv_phi, dz=0.1), 1.4)

    def test_alpha_shock_divides_by_pressure_column(self) -> None:
        W_Rphi = 1.4
        P = np.array([10.0, 20.0, 30.0])

        self.assertAlmostEqual(alpha_shock(W_Rphi, P, dz=0.1), 1.4 / 6.0)
        self.assertTrue(math.isnan(alpha_shock(W_Rphi, np.array([0.0, 0.0]), dz=1.0)))


if __name__ == "__main__":
    unittest.main()

