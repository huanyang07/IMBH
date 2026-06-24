from __future__ import annotations

import unittest

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    energy_residual_metrics,
    mdot_continuity_residual,
    normalized_mdot_continuity_residual,
    pointwise_energy_residual,
)


class AuditMetricsTests(unittest.TestCase):
    def test_pointwise_energy_residual(self) -> None:
        Qplus = np.array([10.0, 10.0])
        Qrad = np.array([5.0, 12.0])
        Qadv = np.array([3.0, 1.0])
        Qwind = np.array([0.0, 0.0])

        residual = pointwise_energy_residual(Qplus, Qrad, Qadv, Qwind)

        np.testing.assert_allclose(residual, np.array([2.0 / 18.0, -3.0 / 23.0]))

    def test_integrated_metrics_do_not_hide_cancellation(self) -> None:
        area = np.ones(2)
        Qplus = np.array([10.0, 10.0])
        Qrad = np.array([5.0, 15.0])
        Qadv = np.zeros(2)
        Qwind = np.zeros(2)

        metrics = energy_residual_metrics(area, Qplus, Qrad, Qadv, Qwind)

        self.assertAlmostEqual(metrics.signed, 0.0)
        self.assertAlmostEqual(metrics.L1, 0.5)
        self.assertGreater(metrics.L2, 0.0)

    def test_boundary_excluded_max_tracks_interior(self) -> None:
        area = np.ones(5)
        Qplus = np.ones(5)
        Qrad = np.array([100.0, 1.0, 0.2, 1.0, 1.0])
        Qadv = np.zeros(5)
        Qwind = np.zeros(5)

        metrics = energy_residual_metrics(area, Qplus, Qrad, Qadv, Qwind, R=np.arange(5.0), boundary_exclude=1)

        self.assertEqual(metrics.max_abs_index, 0)
        self.assertEqual(metrics.max_abs_interior_index, 2)
        self.assertTrue(metrics.max_abs_is_boundary)

    def test_mdot_continuity_residual_zero_for_matching_source(self) -> None:
        R = np.linspace(1.0, 10.0, 32)
        Mdot = 3.0 * R + 2.0
        source = np.zeros_like(R)
        wind = np.full_like(R, 3.0) / (2.0 * np.pi * R)

        residual = mdot_continuity_residual(R, Mdot, source, wind, gradient_method="centered")

        np.testing.assert_allclose(residual, 0.0, atol=1.0e-12)

    def test_normalized_mdot_continuity_residual_is_bounded(self) -> None:
        R = np.linspace(1.0, 10.0, 32)
        Mdot = R**2

        residual = normalized_mdot_continuity_residual(R, Mdot, gradient_method="centered")

        self.assertTrue(np.all(np.abs(residual) <= 1.0))


if __name__ == "__main__":
    unittest.main()
