import numpy as np

from imri_qpe.layer3_minidisk_1d.causal_inner_conservative_entropy_projection_microstep import (
    EquilibriumPrimitiveSeed,
    conservative_entropy_projected_midpoint_microstep,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_geometry import (
    kerr_schild_column_geometry,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_nonlinear_port_atlas import (
    equilibrium_entropy_point_from_primitive,
)


def _three_cell_patch():
    geometry = kerr_schild_column_geometry(3.2e9, 8.0e8)
    height = 2.4e8
    base = (2.1e-7, 4.3e6, -0.21, 0.31)
    perturbations = (
        (-0.002, 0.001, -0.0003, 0.0002),
        (0.001, -0.002, 0.0002, -0.0003),
        (0.001, 0.001, 0.0001, 0.0001),
    )
    seeds = []
    points = []
    for delta in perturbations:
        seed = EquilibriumPrimitiveSeed(
            base[0] * np.exp(delta[0]),
            base[1] * np.exp(delta[1]),
            base[2] + delta[2],
            base[3] + delta[3],
        )
        seeds.append(seed)
        points.append(
            equilibrium_entropy_point_from_primitive(
                geometry,
                density=seed.density,
                temperature=seed.temperature,
                proper_half_thickness=height,
                radial_velocity_over_c=seed.radial_velocity_over_c,
                azimuthal_velocity_over_c=seed.azimuthal_velocity_over_c,
            )
        )
    return geometry, height, tuple(points), tuple(seeds)


def test_three_cell_projection_is_conservative_and_entropy_closed():
    geometry, height, points, seeds = _three_cell_patch()
    result = conservative_entropy_projected_midpoint_microstep(
        geometry=geometry,
        proper_half_thickness=height,
        points=points,
        seeds=seeds,
        courant_factor=0.02,
    )
    assert result.passed
    assert result.projection_entropy_slope < 0.0
    assert abs(result.projection_theta) <= 1.0
    assert result.correction_relative_norm <= 0.05


def test_projection_rejects_invalid_courant_factor():
    geometry, height, points, seeds = _three_cell_patch()
    for courant in (0.0, 0.051):
        try:
            conservative_entropy_projected_midpoint_microstep(
                geometry=geometry,
                proper_half_thickness=height,
                points=points,
                seeds=seeds,
                courant_factor=courant,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid Courant factor was accepted")
