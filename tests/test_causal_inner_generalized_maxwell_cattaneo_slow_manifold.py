from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_semidiscrete import (
    generalized_maxwell_cattaneo_hydrostatic_embedding,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_slow_manifold import (
    FAST_CHART_INDICES,
    directional_jacobian_relative_defect,
    generalized_maxwell_cattaneo_fast_charts,
    generalized_maxwell_cattaneo_fast_rate_scales,
    generalized_maxwell_cattaneo_projected_fast_evaluation,
    generalized_maxwell_cattaneo_reconstruct_fixed_slow,
    generalized_maxwell_cattaneo_slow_targets,
    radius_one_colored_jacobian,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_geometry import (
    make_kerr_schild_column_grid,
)


class _Frequency:
    def frequency(self, _radius):
        return 2.7491520839259703


def _fixture():
    grid = make_kerr_schild_column_grid(5.0e9, 6.0e9, 3, 1.48e9)
    chart5 = np.asarray([4.74082887, -0.330628060, 0.662598339, 14.9471713, 2.13041458e-4])
    chart7 = generalized_maxwell_cattaneo_hydrostatic_embedding(
        chart5, proper_vertical_frequency=2.7491520839259703
    )
    context = SimpleNamespace(
        grid=grid,
        vertical_frequency=_Frequency(),
        alpha=0.1,
        stress_factor=1.0,
        kappa=0.34,
        stream_sources=None,
        outer_boundary_flux_mode="frozen_exterior_rusanov",
        outer_boundary_frozen_exterior_chart=np.array(chart5, copy=True),
    )
    return context, np.repeat(chart7[None, :], 3, axis=0)


def test_fixed_slow_reconstruction_closes_after_fast_perturbation() -> None:
    context, charts = _fixture()
    targets = generalized_maxwell_cattaneo_slow_targets(context, charts)
    fast = generalized_maxwell_cattaneo_fast_charts(charts)
    fast[:, 0] += np.asarray((1.0e-5, -2.0e-5, 1.0e-5))
    fast[:, 2] += np.asarray((2.0e-5, 0.0, -1.0e-5))
    result = generalized_maxwell_cattaneo_reconstruct_fixed_slow(
        context, targets, fast, template_charts=charts
    )
    reconstructed = generalized_maxwell_cattaneo_slow_targets(
        context, result.primitive_charts
    )
    np.testing.assert_allclose(reconstructed, targets, rtol=1.0e-11, atol=0.0)
    np.testing.assert_allclose(
        result.primitive_charts[:, FAST_CHART_INDICES], fast, rtol=0.0, atol=0.0
    )


def test_projected_fast_field_is_tangent_to_fixed_slow_constraints() -> None:
    context, charts = _fixture()
    targets = generalized_maxwell_cattaneo_slow_targets(context, charts)
    fast = generalized_maxwell_cattaneo_fast_charts(charts)
    first = generalized_maxwell_cattaneo_projected_fast_evaluation(
        context, targets, fast, template_charts=charts, quadrature_order=4
    )
    scales = generalized_maxwell_cattaneo_fast_rate_scales(
        first.projected_fast_rates_per_second
    )
    second = generalized_maxwell_cattaneo_projected_fast_evaluation(
        context,
        targets,
        fast,
        template_charts=charts,
        fast_rate_scales_per_second=scales,
        quadrature_order=4,
    )
    assert second.projected_fast_rates_per_second.shape == (3, 4)
    assert second.slow_integrated_drift_per_second.shape == (3, 3)
    assert np.all(np.isfinite(second.normalized_fast_rates))
    assert second.maximum_temporal_projection_solve_relative_defect <= 1.0e-10


def test_radius_one_coloring_recovers_nearest_neighbor_jacobian() -> None:
    rng = np.random.default_rng(20260826)
    matrix = rng.normal(size=(4, 4))
    left = rng.normal(size=(4, 4))
    right = rng.normal(size=(4, 4))
    point = rng.normal(size=(8, 4))

    def function(values):
        result = values @ matrix.T
        result[1:] += values[:-1] @ left.T
        result[:-1] += values[1:] @ right.T
        return result

    _base, jacobian = radius_one_colored_jacobian(function, point)
    direction = rng.normal(size=point.shape)
    defect = directional_jacobian_relative_defect(
        function, point, jacobian, direction
    )
    assert defect <= 1.0e-8
    for output_cell in range(point.shape[0]):
        for input_cell in range(point.shape[0]):
            if abs(output_cell - input_cell) > 1:
                block = jacobian[
                    4 * output_cell : 4 * output_cell + 4,
                    4 * input_cell : 4 * input_cell + 4,
                ]
                assert np.array_equal(block, np.zeros((4, 4)))


def test_fast_rate_scales_are_positive() -> None:
    rates = np.zeros((3, 4))
    rates[0, 0] = 2.0
    scales = generalized_maxwell_cattaneo_fast_rate_scales(rates)
    assert scales.shape == (4,)
    assert np.all(scales > 0.0)
    assert scales[0] == 2.0
