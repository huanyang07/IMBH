from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_quasisteady import (
    hydrostatic_invariant_local_scaled_jacobian,
    reconstruct_hydrostatic_fixed_invariants,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_semidiscrete import (
    generalized_maxwell_cattaneo_hydrostatic_embedding,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo_slow_manifold import (
    generalized_maxwell_cattaneo_slow_targets,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_geometry import (
    make_kerr_schild_column_grid,
)


class _Frequency:
    def frequency(self, _radius):
        return 2.7491520839259703


def _fixture():
    grid = make_kerr_schild_column_grid(5.0e9, 6.0e9, 3, 1.48e9)
    chart5 = np.asarray(
        [4.74082887, -0.330628060, 0.662598339, 14.9471713, 2.13041458e-4]
    )
    chart7 = generalized_maxwell_cattaneo_hydrostatic_embedding(
        chart5, proper_vertical_frequency=2.7491520839259703
    )
    context = SimpleNamespace(
        grid=grid,
        vertical_frequency=_Frequency(),
        alpha=0.1,
        stress_factor=1.0,
    )
    return context, np.repeat(chart7[None, :], 3, axis=0)


def test_exact_anchor_roundtrips_bitwise() -> None:
    context, charts = _fixture()
    targets = generalized_maxwell_cattaneo_slow_targets(context, charts)
    result = reconstruct_hydrostatic_fixed_invariants(
        context,
        targets,
        charts[:, 1],
        charts[:, 4],
        template_charts=charts,
    )
    assert np.array_equal(result.primitive_charts, charts)
    assert result.maximum_constraint_relative_defect == 0.0
    assert result.maximum_newton_corrections == 0


def test_invariants_close_after_target_and_auxiliary_perturbations() -> None:
    context, charts = _fixture()
    targets = generalized_maxwell_cattaneo_slow_targets(context, charts)
    perturbed_targets = np.array(targets, copy=True)
    perturbed_targets[1] *= np.asarray((1.0 + 1.0e-6, 1.0 - 1.0e-6, 1.0 + 5.0e-7))
    radial = np.array(charts[:, 1], copy=True)
    radial[1] += 1.0e-6
    stress = np.array(charts[:, 4], copy=True)
    stress[1] *= 1.0 + 1.0e-4
    result = reconstruct_hydrostatic_fixed_invariants(
        context,
        perturbed_targets,
        radial,
        stress,
        template_charts=charts,
    )
    reconstructed = generalized_maxwell_cattaneo_slow_targets(
        context, result.primitive_charts
    )
    np.testing.assert_allclose(
        reconstructed, perturbed_targets, rtol=1.0e-10, atol=0.0
    )
    np.testing.assert_allclose(result.primitive_charts[:, 1], radial, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.primitive_charts[:, 4], stress, rtol=0.0, atol=0.0)
    assert np.array_equal(result.primitive_charts[:, 6], np.zeros(3))
    assert result.maximum_newton_corrections <= 8
    assert np.isfinite(result.maximum_scaled_local_inverse_condition_number)


def test_invalid_auxiliary_velocity_fails_closed() -> None:
    context, charts = _fixture()
    targets = generalized_maxwell_cattaneo_slow_targets(context, charts)
    radial = np.array(charts[:, 1], copy=True)
    radial[0] = 1.0
    try:
        reconstruct_hydrostatic_fixed_invariants(
            context,
            targets,
            radial,
            charts[:, 4],
            template_charts=charts,
        )
    except ValueError:
        return
    raise AssertionError("inadmissible radial velocity was not rejected")


def test_local_scaled_inverse_tangent_matches_central_reconstruction() -> None:
    context, charts = _fixture()
    targets = generalized_maxwell_cattaneo_slow_targets(context, charts)
    matrix = hydrostatic_invariant_local_scaled_jacobian(context, 1, charts[1])
    direction = np.asarray((0.3, -0.8, 0.5), dtype=float)
    direction /= np.linalg.norm(direction)
    step = 2.0e-6
    plus_targets = np.array(targets, copy=True)
    minus_targets = np.array(targets, copy=True)
    scale = np.abs(targets[1])
    plus_targets[1] += step * scale * direction
    minus_targets[1] -= step * scale * direction
    plus = reconstruct_hydrostatic_fixed_invariants(
        context,
        plus_targets,
        charts[:, 1],
        charts[:, 4],
        template_charts=charts,
    )
    minus = reconstruct_hydrostatic_fixed_invariants(
        context,
        minus_targets,
        charts[:, 1],
        charts[:, 4],
        template_charts=charts,
    )
    finite = (
        plus.primitive_charts[1, [0, 2, 3]]
        - minus.primitive_charts[1, [0, 2, 3]]
    ) / (2.0 * step * np.asarray((1.0, 0.1, 1.0)))
    analytic = np.linalg.solve(matrix, direction)
    defect = np.max(np.abs(finite - analytic)) / max(
        np.max(np.abs(finite)), np.max(np.abs(analytic))
    )
    assert defect <= 2.0e-5
