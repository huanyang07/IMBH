from __future__ import annotations

import numpy as np

from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo import (
    generalized_maxwell_cattaneo_principal,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_fixed_slow_root import (
    equation_rate_parity_relative_defect,
    fixed_slow_equation_row_scales_per_cm,
    physical_coordinate_rate_jacobian_at_root,
    projected_fast_temporal_blocks,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_semidiscrete import (
    generalized_maxwell_cattaneo_hydrostatic_embedding,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_geometry import (
    kerr_schild_column_geometry,
    make_kerr_schild_column_grid,
)


class _Frequency:
    def frequency(self, _radius):
        return 2.7491520839259703


def _context_and_charts():
    grid = make_kerr_schild_column_grid(5.0e9, 6.0e9, 3, 1.48e9)
    chart5 = np.asarray(
        [4.74082887, -0.330628060, 0.662598339, 14.9471713, 2.13041458e-4]
    )
    chart7 = generalized_maxwell_cattaneo_hydrostatic_embedding(
        chart5, proper_vertical_frequency=2.7491520839259703
    )
    context = type(
        "Context",
        (),
        {
            "grid": grid,
            "vertical_frequency": _Frequency(),
            "alpha": 0.1,
            "stress_factor": 1.0,
        },
    )()
    return context, np.repeat(chart7[None, :], 3, axis=0)


def test_equation_row_scales_are_one_second_chart_scales() -> None:
    blocks = np.repeat(np.eye(4)[None, :, :], 3, axis=0)
    chart_scales = np.asarray((0.1, 1.0e-4, 1.0, 0.03))
    result = fixed_slow_equation_row_scales_per_cm(
        blocks, fast_chart_scales=chart_scales
    )
    np.testing.assert_allclose(
        result, np.repeat((chart_scales / C)[None, :], 3, axis=0)
    )


def test_projected_temporal_tangent_annihilates_slow_rows() -> None:
    context, charts = _context_and_charts()
    blocks, tangents = projected_fast_temporal_blocks(context, charts)
    assert blocks.shape == (3, 4, 4)
    for cell, radius_value in enumerate(context.grid.centers):
        radius = float(radius_value)
        principal = generalized_maxwell_cattaneo_principal(
            kerr_schild_column_geometry(
                radius, context.grid.gravitational_radius
            ),
            charts[cell],
            proper_vertical_frequency=context.vertical_frequency.frequency(radius),
            alpha=context.alpha,
            stress_factor=context.stress_factor,
        )
        slow_temporal = principal.temporal_matrix[[0, 2, 3]]
        defect = slow_temporal @ tangents[cell]
        scale = np.max(np.abs(slow_temporal) @ np.abs(tangents[cell]))
        assert np.max(np.abs(defect)) / scale <= 1.0e-12
        np.testing.assert_allclose(
            tangents[cell, [1, 4, 5, 6]], np.eye(4), rtol=0.0, atol=0.0
        )


def test_equation_rate_parity_is_exact_for_identity_blocks() -> None:
    blocks = np.repeat(np.eye(4)[None, :, :], 2, axis=0)
    rates = np.asarray(((1.0, 2.0, 3.0, 4.0), (-1.0, 0.0, 2.0, 1.0)))
    assert equation_rate_parity_relative_defect(blocks, rates, rates / C) == 0.0


def test_root_tangent_transform_has_physical_eigenvalues() -> None:
    blocks = np.repeat(np.eye(4)[None, :, :], 2, axis=0)
    chart_scales = np.asarray((0.1, 1.0e-4, 1.0, 0.03))
    row_scales = np.repeat((chart_scales / C)[None, :], 2, axis=0)
    physical = np.diag(np.asarray((-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0)))
    normalized_equation = physical
    recovered = physical_coordinate_rate_jacobian_at_root(
        blocks,
        row_scales,
        normalized_equation,
        fast_chart_scales=chart_scales,
    )
    np.testing.assert_allclose(recovered, physical)
