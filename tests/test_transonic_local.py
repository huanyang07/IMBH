from __future__ import annotations

import types
import unittest

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    B_rank_minors,
    PaczynskiWiitaPotential,
    algebraic_state,
    analytic_state_partials,
    differential_matrix,
    differential_residual_scales,
    differential_residual,
    finite_difference_state_partials,
    extended_phase_space_matrix,
    integrated_stress,
    local_gradient,
    local_ode_rhs,
    local_scaled_residual,
    local_unscaled_residual,
    mdot_profile_from_source_sink,
    phase_space_null_tangent,
    phase_space_tangent_derivative,
    radiative_cooling,
    scaled_differential_matrix,
    sonic_directional_B,
    sonic_frozen_scaled_directional_B,
    sonic_diagnostics,
    sonic_lhopital_residual,
    sonic_lhopital_residual_form,
    sonic_null_vectors,
    sonic_unscaled_directional_B,
    sonic_unscaled_null_vectors,
    stream_heating_rate,
    stream_annulus_shape_and_derivative,
    stream_mass_rate_and_derivative,
    stream_source_prime,
    stream_torque_specific_l_and_derivative,
    surface_density,
    vertical_state,
    wind_sink_prime,
    xi_eff_from_gradient,
)
from imri_qpe.scales import eddington_mdot
from imri_qpe.units import solar_masses_to_g


class TransonicLocalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.M2_g = solar_masses_to_g(1.0e4)
        self.potential = PaczynskiWiitaPotential(self.M2_g)
        self.params = types.SimpleNamespace(
            M2_g=self.M2_g,
            Mdot_g_s=1.0e-3 * eddington_mdot(self.M2_g),
            alpha=0.01,
            mu_stress=0.0,
            stress_factor=1.5,
            mu_mol=0.62,
            kappa=0.34,
            gamma_gas=5.0 / 3.0,
            partial_eps=1.0e-5,
        )
        self.logR = float(np.log(30.0 * self.potential.r_g))
        self.y = np.log(np.array([1.0e6, 2.0e6]))
        self.lambda0 = float(self.potential.l_k(self.potential.r_isco) / (self.potential.r_g * 2.99792458e10))

    def test_surface_density_continuity(self) -> None:
        R = 20.0 * self.potential.r_g
        u = 3.0e6
        Sigma = surface_density(self.params.Mdot_g_s, R, u)

        self.assertAlmostEqual(2.0 * np.pi * R * Sigma * u / self.params.Mdot_g_s, 1.0)

    def test_vertical_closure_is_positive_and_hydrostatic(self) -> None:
        R = 30.0 * self.potential.r_g
        Sigma = 1.0e4
        T = 2.0e6

        state = vertical_state(Sigma, T, R, self.potential)

        for value in (state.H, state.rho, state.P_gas, state.P_rad, state.P_tot, state.Pi, state.e, state.tau):
            self.assertGreater(value, 0.0)
        lhs = state.Omega_K**2 * state.H**2
        rhs = state.P_tot / state.rho
        self.assertAlmostEqual(lhs / rhs, 1.0, places=10)

    def test_alpha_stress_and_radiative_cooling_are_positive(self) -> None:
        R = 30.0 * self.potential.r_g
        state = vertical_state(1.0e4, 2.0e6, R, self.potential)

        self.assertGreater(integrated_stress(state, alpha=0.01), 0.0)
        self.assertGreater(radiative_cooling(state), 0.0)

    def test_algebraic_state_is_finite(self) -> None:
        state = algebraic_state(self.logR, self.y[0], self.y[1], self.lambda0, self.params)

        for value in (
            state.Sigma,
            state.H,
            state.rho,
            state.P,
            state.Pi,
            state.W,
            state.l,
            state.Omega,
            state.Q_rad,
        ):
            self.assertTrue(np.isfinite(value))
            self.assertGreater(value, 0.0)

    def test_zero_stream_torque_preserves_algebraic_state(self) -> None:
        params_with_zero_torque = types.SimpleNamespace(
            **{
                **self.params.__dict__,
                "R_out": 300.0 * self.potential.r_g,
                "stream_torque_delta_l_fraction": 0.0,
                "stream_torque_center_fraction": 0.8,
                "stream_torque_log_width": 0.08,
            }
        )

        baseline = algebraic_state(self.logR, self.y[0], self.y[1], self.lambda0, self.params)
        with_zero_torque = algebraic_state(self.logR, self.y[0], self.y[1], self.lambda0, params_with_zero_torque)

        self.assertAlmostEqual(with_zero_torque.l / baseline.l, 1.0)
        self.assertAlmostEqual(with_zero_torque.Omega / baseline.Omega, 1.0)

    def test_stream_source_makes_outer_mdot_smaller_than_inner(self) -> None:
        params = types.SimpleNamespace(
            **{
                **self.params.__dict__,
                "R_out": 300.0 * self.potential.r_g,
                "stream_source_fraction": 0.03,
                "stream_source_center_fraction": 0.8,
                "stream_source_log_width": 0.08,
            }
        )
        logR_center = float(np.log(params.stream_source_center_fraction * params.R_out))
        logR_outer = float(np.log(params.R_out))
        eps = 2.0e-5

        mdot, derivative = stream_mass_rate_and_derivative(logR_center, params)
        mdot_plus, _ = stream_mass_rate_and_derivative(logR_center + eps, params)
        mdot_minus, _ = stream_mass_rate_and_derivative(logR_center - eps, params)
        finite_difference = (mdot_plus - mdot_minus) / (2.0 * eps)
        mdot_outer, derivative_outer = stream_mass_rate_and_derivative(logR_outer, params)
        source_prime = stream_source_prime(logR_center, params)
        outer_shape, _outer_dshape = stream_annulus_shape_and_derivative(
            logR_outer,
            params.stream_source_center_fraction,
            params.stream_source_log_width,
            params.R_out,
        )

        self.assertAlmostEqual(mdot / self.params.Mdot_g_s, 0.985)
        self.assertAlmostEqual(mdot_outer / self.params.Mdot_g_s, 1.0 - 0.03 * outer_shape)
        self.assertAlmostEqual(derivative / self.params.Mdot_g_s, -0.5 * 0.03 / params.stream_source_log_width)
        self.assertAlmostEqual(source_prime / self.params.Mdot_g_s, 0.5 * 0.03 / params.stream_source_log_width)
        self.assertLess(derivative_outer, 0.0)
        np.testing.assert_allclose(derivative, finite_difference, rtol=1.0e-7)

    def test_source_sink_budget_helpers_are_explicit(self) -> None:
        params = types.SimpleNamespace(
            **{
                **self.params.__dict__,
                "R_out": 300.0 * self.potential.r_g,
                "stream_source_fraction": 0.03,
                "stream_source_center_fraction": 0.8,
                "stream_source_log_width": 0.08,
                "wind_sink_fraction": 0.01,
                "wind_sink_center_fraction": 0.9,
                "wind_sink_log_width": 0.05,
            }
        )
        logR = float(np.log(0.85 * params.R_out))

        mdot, derivative = mdot_profile_from_source_sink(logR, params)
        source = stream_source_prime(logR, params)
        wind = wind_sink_prime(logR, params)

        self.assertGreater(mdot, 0.0)
        self.assertGreaterEqual(source, 0.0)
        self.assertGreaterEqual(wind, 0.0)
        self.assertAlmostEqual(derivative, wind - source)

    def test_stream_heating_rate_tracks_positive_mass_deposition(self) -> None:
        params = types.SimpleNamespace(
            **{
                **self.params.__dict__,
                "R_out": 300.0 * self.potential.r_g,
                "stream_source_fraction": 0.03,
                "stream_source_center_fraction": 0.8,
                "stream_source_log_width": 0.08,
                "stream_heating_efficiency": 0.01,
            }
        )
        logR_center = float(np.log(params.stream_source_center_fraction * params.R_out))
        cold_params = types.SimpleNamespace(**{**params.__dict__, "stream_heating_efficiency": 0.0})
        no_mass_params = types.SimpleNamespace(**{**params.__dict__, "stream_source_fraction": 0.0})

        self.assertLess(stream_mass_rate_and_derivative(logR_center, params)[1], 0.0)
        self.assertGreater(stream_heating_rate(logR_center, params), 0.0)
        self.assertEqual(stream_heating_rate(logR_center, cold_params), 0.0)
        self.assertEqual(stream_heating_rate(logR_center, no_mass_params), 0.0)

    def test_stream_torque_derivative_matches_centered_difference(self) -> None:
        params = types.SimpleNamespace(
            **{
                **self.params.__dict__,
                "R_out": 300.0 * self.potential.r_g,
                "stream_torque_delta_l_fraction": 1.0e-3,
                "stream_torque_center_fraction": 0.8,
                "stream_torque_log_width": 0.08,
            }
        )
        R_center = params.stream_torque_center_fraction * params.R_out
        logR_center = float(np.log(R_center))
        eps = 2.0e-5

        value, derivative = stream_torque_specific_l_and_derivative(logR_center, params)
        value_plus, _ = stream_torque_specific_l_and_derivative(logR_center + eps, params)
        value_minus, _ = stream_torque_specific_l_and_derivative(logR_center - eps, params)
        finite_difference = (value_plus - value_minus) / (2.0 * eps)
        l_ref = self.potential.l_k(R_center)

        self.assertAlmostEqual(value / l_ref, 0.5e-3)
        self.assertAlmostEqual(derivative / l_ref, 0.5e-3 / params.stream_torque_log_width)
        np.testing.assert_allclose(derivative, finite_difference, rtol=1.0e-7)

    def test_differential_system_is_linear_in_gradient(self) -> None:
        A, c = differential_matrix(self.logR, self.y, self.lambda0, self.params)
        g = np.array([0.17, -0.44])

        direct = differential_residual(self.logR, self.y, g, self.lambda0, self.params)
        linear = A @ g + c
        scale = np.maximum(np.abs(direct), np.abs(linear)) + 1.0

        self.assertLess(float(np.max(np.abs(direct - linear) / scale)), 1.0e-12)

    def test_scaled_differential_matrix_matches_smooth_scales(self) -> None:
        A, c = differential_matrix(self.logR, self.y, self.lambda0, self.params)
        radial_scale, energy_scale = differential_residual_scales(self.logR, self.y, self.lambda0, self.params)
        A_scaled, c_scaled, radial_returned, energy_returned = scaled_differential_matrix(self.logR, self.y, self.lambda0, self.params)

        np.testing.assert_allclose(A_scaled, A / np.array([[radial_scale], [energy_scale]]))
        np.testing.assert_allclose(c_scaled, c / np.array([radial_scale, energy_scale]))
        self.assertAlmostEqual(radial_returned, radial_scale)
        self.assertAlmostEqual(energy_returned, energy_scale)

    def test_local_scaled_residual_matches_scaled_matrix(self) -> None:
        A_scaled, c_scaled, _radial_scale, _energy_scale = scaled_differential_matrix(self.logR, self.y, self.lambda0, self.params)
        g = np.array([0.12, -0.31])

        np.testing.assert_allclose(local_scaled_residual(self.logR, self.y, g, self.lambda0, self.params), A_scaled @ g + c_scaled)

    def test_local_unscaled_residual_matches_raw_matrix(self) -> None:
        A, c = differential_matrix(self.logR, self.y, self.lambda0, self.params)
        g = np.array([0.12, -0.31])

        np.testing.assert_allclose(local_unscaled_residual(self.logR, self.y, g, self.lambda0, self.params), A @ g + c)

    def test_sonic_null_vectors_match_svd_diagnostics(self) -> None:
        nulls = sonic_null_vectors(self.logR, self.y, self.lambda0, self.params)
        diagnostics = sonic_diagnostics(self.logR, self.y, self.lambda0, self.params)

        self.assertEqual(nulls.matrix.shape, (2, 2))
        self.assertEqual(nulls.rhs.shape, (2,))
        np.testing.assert_allclose(np.abs(nulls.left_null), np.abs(diagnostics.left_null))
        np.testing.assert_allclose(np.abs(nulls.right_null), np.abs(diagnostics.right_null))
        np.testing.assert_allclose(nulls.singular_values, diagnostics.singular_values)
        self.assertAlmostEqual(nulls.smin_over_smax, diagnostics.smin_over_smax)

    def test_sonic_unscaled_null_vectors_are_consistent(self) -> None:
        nulls = sonic_unscaled_null_vectors(self.logR, self.y, self.lambda0, self.params)

        self.assertEqual(nulls.matrix.shape, (2, 2))
        self.assertLess(abs(float(nulls.left_null @ nulls.matrix[:, -1])), np.linalg.norm(nulls.matrix) + 1.0)
        self.assertLess(float(np.linalg.norm(nulls.matrix @ nulls.right_null)), np.linalg.norm(nulls.matrix) + 1.0)

    def test_sonic_directional_B_matches_centered_difference(self) -> None:
        g = np.array([0.17, -0.44])
        eps = 2.0e-5
        plus = local_scaled_residual(self.logR + eps, self.y + eps * g, g, self.lambda0, self.params)
        minus = local_scaled_residual(self.logR - eps, self.y - eps * g, g, self.lambda0, self.params)
        expected = (plus - minus) / (2.0 * eps)

        np.testing.assert_allclose(sonic_directional_B(self.logR, self.y, g, self.lambda0, self.params, eps=eps), expected)

    def test_frozen_scaled_directional_B_matches_unscaled_over_sonic_scales(self) -> None:
        g = np.array([0.17, -0.44])
        eps = 2.0e-5
        radial_scale, energy_scale = differential_residual_scales(self.logR, self.y, self.lambda0, self.params)
        expected = sonic_unscaled_directional_B(self.logR, self.y, g, self.lambda0, self.params, eps=eps) / np.array([radial_scale, energy_scale])

        np.testing.assert_allclose(sonic_frozen_scaled_directional_B(self.logR, self.y, g, self.lambda0, self.params, eps=eps), expected)

    def test_sonic_lhopital_residual_is_normalized_and_finite(self) -> None:
        g = np.array([0.17, -0.44])
        residual = sonic_lhopital_residual(self.logR, self.y, g, self.lambda0, self.params, eps=2.0e-5)

        self.assertTrue(np.isfinite(residual))
        self.assertLessEqual(abs(residual), 1.0 + 1.0e-12)

    def test_sonic_lhopital_residual_forms_are_normalized_and_finite(self) -> None:
        g = np.array([0.17, -0.44])
        for form in ("scaled", "frozen_scaled", "raw"):
            residual = sonic_lhopital_residual_form(self.logR, self.y, g, self.lambda0, self.params, eps=2.0e-5, form=form)
            self.assertTrue(np.isfinite(residual), form)
            self.assertLessEqual(abs(residual), 1.0 + 1.0e-12, form)

    def test_analytic_partials_match_finite_difference_partials(self) -> None:
        analytic = analytic_state_partials(self.logR, self.y, self.lambda0, self.params)
        finite = finite_difference_state_partials(self.logR, self.y, self.lambda0, self.params, eps_x=3.0e-6, eps_y=3.0e-6)

        for key in ("Pi", "rho", "e", "Omega"):
            scale_x = max(abs(analytic.x[key]), abs(finite.x[key]), 1.0)
            self.assertLess(abs(analytic.x[key] - finite.x[key]) / scale_x, 3.0e-5, key)
            scale_y = np.maximum(np.maximum(np.abs(analytic.y[key]), np.abs(finite.y[key])), 1.0)
            self.assertLess(float(np.max(np.abs(analytic.y[key] - finite.y[key]) / scale_y)), 3.0e-5, key)

    def test_analytic_partials_match_finite_difference_with_stream_mass(self) -> None:
        params = types.SimpleNamespace(
            **{
                **self.params.__dict__,
                "R_out": 300.0 * self.potential.r_g,
                "stream_source_fraction": 0.03,
                "stream_source_center_fraction": 0.8,
                "stream_source_log_width": 0.08,
            }
        )
        logR = float(np.log(params.stream_source_center_fraction * params.R_out))
        analytic = analytic_state_partials(logR, self.y, self.lambda0, params)
        finite = finite_difference_state_partials(logR, self.y, self.lambda0, params, eps_x=3.0e-6, eps_y=3.0e-6)

        for key in ("Pi", "rho", "e", "Omega"):
            scale_x = max(abs(analytic.x[key]), abs(finite.x[key]), 1.0)
            self.assertLess(abs(analytic.x[key] - finite.x[key]) / scale_x, 5.0e-5, key)
            scale_y = np.maximum(np.maximum(np.abs(analytic.y[key]), np.abs(finite.y[key])), 1.0)
            self.assertLess(float(np.max(np.abs(analytic.y[key] - finite.y[key]) / scale_y)), 5.0e-5, key)

    def test_analytic_partials_support_mixed_stress_exponent(self) -> None:
        params = types.SimpleNamespace(**{**self.params.__dict__, "mu_stress": 0.4})
        analytic = analytic_state_partials(self.logR, self.y, self.lambda0, params)
        finite = finite_difference_state_partials(self.logR, self.y, self.lambda0, params, eps_x=3.0e-6, eps_y=3.0e-6)

        scale = np.maximum(np.maximum(np.abs(analytic.y["Omega"]), np.abs(finite.y["Omega"])), 1.0)
        self.assertLess(float(np.max(np.abs(analytic.y["Omega"] - finite.y["Omega"]) / scale)), 3.0e-5)

    def test_local_gradient_solves_residual(self) -> None:
        g = local_gradient(self.logR, self.y, self.lambda0, self.params)
        residual = differential_residual(self.logR, self.y, g, self.lambda0, self.params)
        scale = np.maximum(np.abs(differential_residual(self.logR, self.y, np.zeros(2), self.lambda0, self.params)), 1.0)

        self.assertLess(float(np.max(np.abs(residual) / scale)), 1.0e-7)

    def test_local_ode_rhs_solves_scaled_residual(self) -> None:
        g = local_ode_rhs(self.logR, self.y, self.lambda0, self.params)
        residual = local_scaled_residual(self.logR, self.y, g, self.lambda0, self.params)

        self.assertLess(float(np.max(np.abs(residual))), 1.0e-7)

    def test_phase_space_matrix_tangent_solves_null_equation(self) -> None:
        diagnostics = phase_space_null_tangent(self.logR, self.y, self.lambda0, self.params)
        B, _A, _c = extended_phase_space_matrix(self.logR, self.y, self.lambda0, self.params)

        self.assertEqual(B.shape, (2, 3))
        np.testing.assert_allclose(B @ diagnostics.tangent, diagnostics.residual)
        self.assertLess(float(np.max(np.abs(diagnostics.residual))), 1.0e-10)
        self.assertAlmostEqual(float(np.linalg.norm(diagnostics.tangent)), 1.0)
        self.assertGreater(diagnostics.px, 0.0)

    def test_phase_space_tangent_matches_radial_ode_away_from_criticality(self) -> None:
        diagnostics = phase_space_null_tangent(self.logR, self.y, self.lambda0, self.params)
        g = local_ode_rhs(self.logR, self.y, self.lambda0, self.params)
        implied = diagnostics.tangent[1:] / diagnostics.tangent[0]

        np.testing.assert_allclose(implied, g, rtol=1.0e-8, atol=1.0e-10)

    def test_phase_space_tangent_orients_with_previous(self) -> None:
        first = phase_space_null_tangent(self.logR, self.y, self.lambda0, self.params)
        second = phase_space_null_tangent(self.logR, self.y, self.lambda0, self.params, previous=-first.tangent)

        np.testing.assert_allclose(second.tangent, -first.tangent)

    def test_B_rank_minors_match_cross_product(self) -> None:
        B, _A, _c = extended_phase_space_matrix(self.logR, self.y, self.lambda0, self.params)
        minors = B_rank_minors(self.logR, self.y, self.lambda0, self.params)
        expected = np.array(
            [
                B[0, 0] * B[1, 1] - B[0, 1] * B[1, 0],
                B[0, 0] * B[1, 2] - B[0, 2] * B[1, 0],
                B[0, 1] * B[1, 2] - B[0, 2] * B[1, 1],
            ],
            dtype=float,
        )

        np.testing.assert_allclose(minors, expected)
        self.assertEqual(minors.shape, (3,))

    def test_phase_space_tangent_derivative_matches_centered_difference(self) -> None:
        eps = 2.0e-5
        diagnostics = phase_space_null_tangent(self.logR, self.y, self.lambda0, self.params)
        tangent = diagnostics.tangent
        z = np.array([self.logR, *self.y], dtype=float)
        plus = z + eps * tangent
        minus = z - eps * tangent
        p_plus = phase_space_null_tangent(plus[0], plus[1:], self.lambda0, self.params, previous=tangent).tangent
        p_minus = phase_space_null_tangent(minus[0], minus[1:], self.lambda0, self.params, previous=tangent).tangent
        expected = (p_plus - p_minus) / (2.0 * eps)

        np.testing.assert_allclose(
            phase_space_tangent_derivative(self.logR, self.y, self.lambda0, self.params, tangent, eps=eps),
            expected,
        )

    def test_sonic_diagnostics_are_finite(self) -> None:
        diagnostics = sonic_diagnostics(self.logR, self.y, self.lambda0, self.params)

        self.assertTrue(np.isfinite(diagnostics.D))
        self.assertTrue(np.isfinite(diagnostics.C1))
        self.assertTrue(np.isfinite(diagnostics.C2))
        self.assertTrue(np.isfinite(diagnostics.compatibility))
        self.assertTrue(np.isfinite(diagnostics.N))
        self.assertTrue(np.isfinite(diagnostics.smin_over_smax))
        self.assertTrue(np.isfinite(diagnostics.null_radial_fraction))
        self.assertTrue(np.isfinite(diagnostics.M_eff))
        self.assertGreater(diagnostics.radial_scale, 0.0)
        self.assertGreater(diagnostics.energy_scale, 0.0)
        self.assertEqual(diagnostics.singular_values.shape, (2,))
        self.assertEqual(diagnostics.left_null.shape, (2,))
        self.assertEqual(diagnostics.right_null.shape, (2,))

    def test_xi_eff_from_gradient_is_finite(self) -> None:
        g = np.array([-0.5, -0.75])
        xi = xi_eff_from_gradient(self.logR, self.y, g, self.lambda0, self.params)

        self.assertTrue(np.isfinite(xi))


if __name__ == "__main__":
    unittest.main()
