import numpy as np

from imri_qpe.layer3_minidisk_1d.causal_inner_family_energy_transfer import (
    causal_physical_family_transfer_ledger,
    causal_polynomial_spectral_projectors,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_scattering_energy import (
    causal_normalization_invariant_scattering_energy,
)


def test_polynomial_projectors_match_eigenvector_independent_projectors():
    rng = np.random.default_rng(20260730)
    transform = rng.normal(size=(5, 5))
    while np.linalg.cond(transform) > 20.0:
        transform = rng.normal(size=(5, 5))
    speeds = np.asarray((-1.7, -0.6, 0.1, 0.8, 1.9))
    evolution = transform @ np.diag(speeds) @ np.linalg.inv(transform)
    temporal = np.diag((1.2, 0.9, 1.4, 0.8, 1.1))
    spatial = temporal @ evolution
    scales = np.asarray((2.0, 0.5, 1.5, 3.0, 0.8))

    polynomial = causal_polynomial_spectral_projectors(
        temporal,
        spatial,
        scales,
    )
    eigenvector = causal_normalization_invariant_scattering_energy(
        temporal,
        spatial,
        scales,
    )
    np.testing.assert_allclose(
        polynomial.characteristic_speeds,
        eigenvector.characteristic_speeds,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        polynomial.primitive_projectors,
        eigenvector.primitive_projectors,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        polynomial.primitive_energy_metric,
        eigenvector.primitive_energy_metric,
        rtol=3.0e-12,
        atol=3.0e-12,
    )
    assert polynomial.maximum_identity_defect < 1.0e-12
    assert polynomial.maximum_idempotence_defect < 1.0e-12
    assert polynomial.maximum_cross_projector_defect < 1.0e-12
    assert polynomial.maximum_energy_orthogonality_defect < 1.0e-12
    assert polynomial.maximum_symmetrizer_defect < 1.0e-12


def test_positive_family_transfer_closes_by_block_source_and_receiver():
    cells = 2
    fields = 5
    dimensions = cells * fields
    times = np.linspace(0.0, 0.4, 21)
    projectors = np.zeros((cells, fields, fields, fields))
    for family in range(fields):
        projectors[:, family, family, family] = 1.0
    energy = np.repeat(
        np.diag((1.0, 2.0, 3.0, 4.0, 5.0))[None],
        cells,
        axis=0,
    )
    first = np.diag(np.linspace(-0.5, 0.4, dimensions))
    second = np.zeros((dimensions, dimensions))
    second[1, 0] = 0.3
    second[7, 6] = -0.2
    blocks = {"diagonal": first, "conversion": second}
    generator = first + second
    initial = np.linspace(0.2, 1.1, dimensions)
    history = np.asarray(
        [
            initial + time * (generator @ initial)
            for time in times
        ]
    ).reshape(times.size, cells, fields)

    ledger = causal_physical_family_transfer_ledger(
        history,
        times,
        log_edges=np.asarray((0.0, 0.7, 1.5)),
        primitive_energy_metrics=energy,
        primitive_projectors=projectors,
        scaled_generator_per_s=generator,
        scaled_generator_blocks_per_s=blocks,
        primitive_column_scales=np.ones(dimensions),
        lower_face=0,
        upper_face=2,
    )
    assert ledger.maximum_family_partition_defect < 2.0e-15
    assert ledger.maximum_power_closure_defect < 2.0e-15
    assert ledger.maximum_block_matrix_closure_defect < 2.0e-15
    np.testing.assert_allclose(
        np.sum(
            ledger.block_source_receiver_power_per_s,
            axis=(0, 2, 3),
        ),
        ledger.total_power_per_s,
        rtol=2.0e-15,
        atol=2.0e-14,
    )
    assert (
        ledger.integrated_block_source_receiver_cell_work.shape
        == (2, 5, 5, 2)
    )

    residual_ledger = causal_physical_family_transfer_ledger(
        history,
        times,
        log_edges=np.asarray((0.0, 0.7, 1.5)),
        primitive_energy_metrics=energy,
        primitive_projectors=projectors,
        scaled_generator_per_s=generator,
        descriptor_scaled_matrix=np.eye(dimensions),
        scaled_residual_blocks={
            name: -matrix for name, matrix in blocks.items()
        },
        primitive_column_scales=np.ones(dimensions),
        lower_face=0,
        upper_face=2,
    )
    np.testing.assert_allclose(
        residual_ledger.block_source_receiver_power_per_s,
        ledger.block_source_receiver_power_per_s,
        rtol=2.0e-15,
        atol=2.0e-14,
    )
    assert residual_ledger.maximum_block_matrix_closure_defect < 2.0e-15
