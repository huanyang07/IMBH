"""Conditioned equation form for the cellwise fixed-slow fast root."""

from __future__ import annotations

import numpy as np

from imri_qpe.constants import C

from .causal_inner_generalized_maxwell_cattaneo import (
    generalized_maxwell_cattaneo_principal,
)
from .causal_inner_generalized_maxwell_cattaneo_slow_manifold import (
    FAST_CHART_INDICES,
    FAST_CHART_SCALES,
    FAST_EQUATION_ROWS,
    SLOW_CHART_INDICES,
    SLOW_EXACT_ROWS,
)
from .causal_inner_geometry import kerr_schild_column_geometry


def projected_fast_temporal_blocks(
    context,
    primitive_charts,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``A=T_fast P`` and the fixed-slow chart tangent ``P``.

    ``P`` maps the four independent fast-chart rates to all seven chart
    rates while imposing zero rates in the three exact slow states.
    """

    charts = np.asarray(primitive_charts, dtype=float)
    n_cells = int(context.grid.centers.size)
    if charts.shape != (n_cells, 7) or np.any(~np.isfinite(charts)):
        raise ValueError("projected temporal-block charts are invalid")
    blocks = np.empty((n_cells, 4, 4), dtype=float)
    tangents = np.zeros((n_cells, 7, 4), dtype=float)
    tangents[:, FAST_CHART_INDICES, :] = np.eye(4)[None, :, :]
    for cell, radius_value in enumerate(np.asarray(context.grid.centers, dtype=float)):
        radius = float(radius_value)
        principal = generalized_maxwell_cattaneo_principal(
            kerr_schild_column_geometry(
                radius, context.grid.gravitational_radius
            ),
            charts[cell],
            proper_vertical_frequency=float(
                context.vertical_frequency.frequency(radius)
            ),
            alpha=float(context.alpha),
            stress_factor=float(context.stress_factor),
        )
        temporal = np.asarray(principal.temporal_matrix, dtype=float)
        slow_slow = temporal[np.ix_(SLOW_EXACT_ROWS, SLOW_CHART_INDICES)]
        slow_fast = temporal[np.ix_(SLOW_EXACT_ROWS, FAST_CHART_INDICES)]
        tangents[cell, SLOW_CHART_INDICES, :] = -np.linalg.solve(
            slow_slow, slow_fast
        )
        blocks[cell] = temporal[FAST_EQUATION_ROWS] @ tangents[cell]
    return blocks, tangents


def fixed_slow_equation_row_scales_per_cm(
    projected_temporal_blocks,
    *,
    fast_chart_scales=FAST_CHART_SCALES,
    reference_time_seconds: float = 1.0,
    relative_floor: float = 1.0e-12,
) -> np.ndarray:
    """Scale each fast equation by one chart-scale change per reference time."""

    blocks = np.asarray(projected_temporal_blocks, dtype=float)
    scales = np.asarray(fast_chart_scales, dtype=float)
    reference_time = float(reference_time_seconds)
    floor = float(relative_floor)
    if (
        blocks.ndim != 3
        or blocks.shape[1:] != (4, 4)
        or np.any(~np.isfinite(blocks))
        or scales.shape != (4,)
        or np.any(~np.isfinite(scales))
        or np.any(scales <= 0.0)
        or not np.isfinite(reference_time)
        or reference_time <= 0.0
        or not np.isfinite(floor)
        or floor <= 0.0
    ):
        raise ValueError("fixed-slow equation row-scale inputs are invalid")
    result = np.max(np.abs(blocks * scales[None, None, :]), axis=2)
    result /= C * reference_time
    global_floor = max(float(np.max(result)) * floor, np.finfo(float).tiny)
    return np.maximum(result, global_floor)


def equation_rate_parity_relative_defect(
    projected_temporal_blocks,
    fast_rates_per_second,
    fast_equation_right_hand_sides_per_cm,
) -> float:
    """Audit ``A dot(z_fast)/c = RHS_fast``."""

    blocks = np.asarray(projected_temporal_blocks, dtype=float)
    rates = np.asarray(fast_rates_per_second, dtype=float)
    right = np.asarray(fast_equation_right_hand_sides_per_cm, dtype=float)
    if (
        blocks.shape != (rates.shape[0], 4, 4)
        or rates.shape != right.shape
        or rates.ndim != 2
        or rates.shape[1] != 4
        or np.any(~np.isfinite(blocks))
        or np.any(~np.isfinite(rates))
        or np.any(~np.isfinite(right))
    ):
        raise ValueError("equation/rate parity inputs are invalid")
    predicted = np.einsum("nij,nj->ni", blocks, rates / C)
    scale = max(
        float(np.max(np.abs(predicted))),
        float(np.max(np.abs(right))),
        np.finfo(float).tiny,
    )
    return float(np.max(np.abs(predicted - right)) / scale)


def physical_coordinate_rate_jacobian_at_root(
    projected_temporal_blocks,
    equation_row_scales_per_cm,
    normalized_equation_jacobian,
    *,
    fast_chart_scales=FAST_CHART_SCALES,
) -> np.ndarray:
    """Convert a root equation Jacobian into the physical rate tangent.

    At an exact root, differentiation of ``A(z)^{-1} RHS(z)`` contains no
    derivative-of-``A`` term because ``RHS=0``.  The returned matrix is the
    derivative of ``dot(z_fast)/D`` with respect to ``z_fast/D`` and hence
    has physical eigenvalues in inverse seconds.
    """

    blocks = np.asarray(projected_temporal_blocks, dtype=float)
    row_scales = np.asarray(equation_row_scales_per_cm, dtype=float)
    jacobian = np.asarray(normalized_equation_jacobian, dtype=float)
    chart_scales = np.asarray(fast_chart_scales, dtype=float)
    n_cells = blocks.shape[0]
    if (
        blocks.shape != (n_cells, 4, 4)
        or row_scales.shape != (n_cells, 4)
        or jacobian.shape != (4 * n_cells, 4 * n_cells)
        or chart_scales.shape != (4,)
        or np.any(~np.isfinite(blocks))
        or np.any(~np.isfinite(row_scales))
        or np.any(row_scales <= 0.0)
        or np.any(~np.isfinite(jacobian))
        or np.any(~np.isfinite(chart_scales))
        or np.any(chart_scales <= 0.0)
    ):
        raise ValueError("physical root-tangent inputs are invalid")
    left = np.zeros_like(jacobian)
    for cell in range(n_cells):
        transform = np.diag(1.0 / chart_scales) @ (
            C
            * np.linalg.solve(
                blocks[cell], np.diag(row_scales[cell])
            )
        )
        rows = slice(4 * cell, 4 * cell + 4)
        left[rows, rows] = transform
    return left @ jacobian


__all__ = (
    "equation_rate_parity_relative_defect",
    "fixed_slow_equation_row_scales_per_cm",
    "physical_coordinate_rate_jacobian_at_root",
    "projected_fast_temporal_blocks",
)
