from __future__ import annotations

import numpy as np

from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_semidiscrete import (
    generalized_maxwell_cattaneo_hydrostatic_embedding,
    generalized_maxwell_cattaneo_lower_source,
    generalized_maxwell_cattaneo_periodic_operator,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_geometry import (
    kerr_schild_column_geometry,
)


RADIUS = 5.599841633135499e9
GRAVITATIONAL_RADIUS = 1.48e9
OMEGA = 2.7491520839259703
ALPHA = 0.1


def _fixture():
    geometry = kerr_schild_column_geometry(RADIUS, GRAVITATIONAL_RADIUS)
    chart5 = np.asarray(
        [4.74082887, -0.330628060, 0.662598339, 14.9471713, 2.13041458e-4],
        dtype=float,
    )
    chart7 = generalized_maxwell_cattaneo_hydrostatic_embedding(
        chart5, proper_vertical_frequency=OMEGA
    )
    return geometry, chart5, chart7


def test_hydrostatic_embedding_zeroes_vertical_material_sources() -> None:
    geometry, _chart5, chart7 = _fixture()
    source = generalized_maxwell_cattaneo_lower_source(
        geometry,
        chart7,
        proper_vertical_frequency=OMEGA,
        alpha=ALPHA,
    )
    scale = max(
        abs(source.vertical_acceleration_cm_per_s2),
        OMEGA**2 * np.exp(chart7[5]),
        1.0,
    )
    assert abs(source.hydrostatic_force_acceleration_cm_per_s2) / scale <= 1.0e-12
    assert source.height_material_source_per_cm[5] == 0.0
    assert abs(source.vertical_momentum_source_per_cm[6]) <= 1.0e-12 * scale


def test_lower_source_keeps_vertical_energy_ledger_closed() -> None:
    geometry, _chart5, chart7 = _fixture()
    perturbed = np.array(chart7, copy=True)
    perturbed[5] += 0.05
    perturbed[6] = 0.02
    source = generalized_maxwell_cattaneo_lower_source(
        geometry,
        perturbed,
        proper_vertical_frequency=OMEGA,
        alpha=ALPHA,
    )
    assert source.source_ledger.vertical_total_energy_relative_defect <= 1.0e-12
    assert source.source_ledger.vertical_reversible_exchange_relative_defect <= 1.0e-12
    assert source.scattering_optical_depth > 1.0


def test_periodic_constant_spatial_operator_is_zero_without_sources() -> None:
    geometry, _chart5, chart7 = _fixture()
    charts = np.repeat(chart7[None, :], 4, axis=0)
    operator = generalized_maxwell_cattaneo_periodic_operator(
        geometry,
        charts,
        cell_spacing_cm=1.0e7,
        proper_vertical_frequency=OMEGA,
        alpha=ALPHA,
        include_lower_sources=False,
    )
    np.testing.assert_allclose(operator.spatial_equation_residuals_per_cm, 0.0)
    np.testing.assert_allclose(operator.primitive_rates_per_ct, 0.0)
    assert operator.global_exact_flux_ledger_relative_defect == 0.0


def test_periodic_exact_rows_telescope_and_temporal_solves_close() -> None:
    geometry, _chart5, chart7 = _fixture()
    perturbations = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1e-3, 1e-4, -1e-4, 8e-4, 1e-8, 5e-4, 1e-4],
            [-5e-4, -8e-5, 5e-5, -4e-4, -1e-8, -2e-4, -5e-5],
            [3e-4, 4e-5, 3e-5, 2e-4, 5e-9, 1e-4, 2e-5],
        ],
        dtype=float,
    )
    operator = generalized_maxwell_cattaneo_periodic_operator(
        geometry,
        chart7 + perturbations,
        cell_spacing_cm=1.0e7,
        proper_vertical_frequency=OMEGA,
        alpha=ALPHA,
        include_lower_sources=True,
    )
    assert operator.global_exact_flux_ledger_relative_defect <= 1.0e-10
    assert operator.maximum_interface_split_relative_defect <= 1.0e-12
    assert np.max(operator.temporal_solve_relative_residuals) <= 1.0e-10
    assert np.all(np.isfinite(operator.primitive_rates_per_ct))
