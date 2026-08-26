from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_radial import (
    generalized_maxwell_cattaneo_radial_operator,
    generalized_maxwell_cattaneo_ssprk2_step,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_semidiscrete import (
    generalized_maxwell_cattaneo_hydrostatic_embedding,
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


def test_radial_operator_is_finite_and_exact_rows_use_shared_fluxes() -> None:
    context, charts = _fixture()
    operator = generalized_maxwell_cattaneo_radial_operator(
        context, charts, timestep_seconds=1.0e-6, quadrature_order=4
    )
    expected = (
        operator.weighted_shared_exact_fluxes_over_c[1:]
        - operator.weighted_shared_exact_fluxes_over_c[:-1]
    )
    np.testing.assert_allclose(
        operator.weighted_spatial_equation_residuals_over_c[:, [0, 1, 2, 3, 5, 6]],
        expected,
        rtol=0.0,
        atol=0.0,
    )
    assert np.all(np.isfinite(operator.primitive_rates_per_ct))
    assert np.max(operator.temporal_solve_relative_residuals) <= 1.0e-10


def test_ssprk2_step_is_deterministic() -> None:
    context, charts = _fixture()
    first = generalized_maxwell_cattaneo_ssprk2_step(
        context, charts, timestep_seconds=1.0e-8, quadrature_order=4
    )
    second = generalized_maxwell_cattaneo_ssprk2_step(
        context, charts, timestep_seconds=1.0e-8, quadrature_order=4
    )
    assert np.array_equal(first.euler_stage_charts, second.euler_stage_charts)
    assert np.array_equal(first.accepted_charts, second.accepted_charts)
    assert first.maximum_scaled_chart_change >= 0.0
    assert first.exact_flux_balance_relative_defect >= 0.0
