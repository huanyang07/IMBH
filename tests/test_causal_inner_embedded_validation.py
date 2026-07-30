from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from imri_qpe.layer3_minidisk_1d.causal_inner_embedded_validation import (
    causal_characteristic_energy_history,
    causal_dimensionless_characteristic_inverse,
    causal_embedded_active_direct_observables,
    causal_embedded_active_observable_audit,
)


def _fake_tangent():
    cells = 3
    dimensions = 15
    rng = np.random.default_rng(20260729)
    face = rng.normal(size=(cells + 1, 5, dimensions))
    transport = np.empty((cells, 5, dimensions))
    for cell in range(cells):
        transport[cell] = face[cell + 1] - face[cell]
    cooling = rng.normal(size=(cells, 5, dimensions))
    height = rng.normal(size=(cells, 5, dimensions))
    other = rng.normal(size=(cells, 5, dimensions))
    stationary = transport + cooling + height + other
    blocks = {
        "candidate_conservative_transport": transport.reshape(
            dimensions, dimensions
        ),
        "candidate_cooling": cooling.reshape(dimensions, dimensions),
        "candidate_lower_height_work": height.reshape(
            dimensions, dimensions
        ),
    }
    spatial = SimpleNamespace(
        shared_face_flux_scaled_jacobians=face,
        block_scaled_jacobians=blocks,
    )
    return SimpleNamespace(
        base_primitives=np.ones((cells, 5)),
        conservation_row_scales=np.ones(dimensions),
        stationary_scaled_jacobian=stationary.reshape(
            dimensions, dimensions
        ),
        spatial_tangent=spatial,
    )


def test_active_observable_map_closes_shared_flux_and_prefix_ledgers() -> None:
    audit = causal_embedded_active_observable_audit(
        _fake_tangent(),
        coupling_face_index=2,
    )
    assert audit.observable_map.shape == (13, 15)
    assert audit.lower_height_cell_map.shape == (3, 5, 15)
    assert audit.conservative_transport_telescoping_defect <= 2.0e-15
    assert audit.active_prefix_ledger_defect <= 2.0e-15
    assert audit.active_cell_count == 2
    assert audit.coupling_face_index == 2


def test_active_direct_observables_use_only_cells_inside_coupling() -> None:
    residual = np.arange(15, dtype=float).reshape(3, 5)
    cooling = 0.1 * residual
    height = -0.2 * residual
    flux = np.arange(20, dtype=float).reshape(4, 5)
    evaluation = SimpleNamespace(
        residual_rows=residual,
        cooling_rows=cooling,
        lower_height_work_rows=height,
        stationary_ledger=SimpleNamespace(
            interfaces=SimpleNamespace(
                candidate_shared_face_fluxes_over_c=flux,
            )
        ),
    )
    result = causal_embedded_active_direct_observables(
        evaluation,
        coupling_face_index=2,
    )
    expected = np.concatenate(
        (
            flux[0, (0, 2, 4)],
            flux[2, (0, 2, 4)],
            -np.sum(residual[:2, (0, 2, 4)], axis=0),
            -np.sum(cooling[:2, (2, 4)], axis=0),
            -np.sum(height[:2, (2, 4)], axis=0),
        )
    )
    np.testing.assert_array_equal(result, expected)


def test_characteristic_energy_uses_fixed_scales_and_cell_measures() -> None:
    right = np.repeat(np.eye(5)[None, :, :], 2, axis=0)
    scales = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    inverse = causal_dimensionless_characteristic_inverse(right, scales)
    values = np.zeros((3, 2, 5))
    values[:, 0, 1] = 2.0
    values[:, 1, 3] = 4.0
    energy = causal_characteristic_energy_history(
        values,
        inverse,
        scales,
        np.asarray([2.0, 3.0]),
    )
    assert energy.shape == (3, 2, 5)
    np.testing.assert_allclose(energy[:, 0, 1], 2.0)
    np.testing.assert_allclose(energy[:, 1, 3], 3.0)
    assert np.count_nonzero(energy) == 6
