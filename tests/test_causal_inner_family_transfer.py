import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES,
    CausalCharacteristicFamilyProjectors,
    causal_block_family_receiver_action,
    causal_block_family_transfer_ledger,
    causal_five_field_characteristic_family_decomposition,
    causal_local_quadratic_energy_work_ledger,
    causal_pairwise_family_cross_work,
    causal_pairwise_weighted_gram_ledger,
)


def _coordinate_projectors(n_cells: int):
    matrices = np.zeros((5, n_cells, 5, 5))
    for family in range(5):
        matrices[family, :, family, family] = 1.0
    return CausalCharacteristicFamilyProjectors(
        family_labels=CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES,
        primitive_projectors=matrices,
        maximum_identity_closure_defect=0.0,
        maximum_idempotence_defect=0.0,
        maximum_cross_projector_defect=0.0,
        maximum_basis_condition_number=1.0,
        maximum_eigenpair_defect=0.0,
    )


def test_characteristic_family_decomposition_is_exact():
    rng = np.random.default_rng(20260726)
    values = rng.normal(size=(7, 4, 5))
    components = causal_five_field_characteristic_family_decomposition(
        values,
        _coordinate_projectors(4),
    )
    assert components.shape == (5, 7, 4, 5)
    np.testing.assert_array_equal(np.sum(components, axis=0), values)


def test_pairwise_gram_and_cross_work_close():
    rng = np.random.default_rng(20260727)
    components = rng.normal(size=(5, 6, 3, 5))
    weights = np.array([1.0, 2.0, 4.0])
    ledger = causal_pairwise_weighted_gram_ledger(
        components,
        weights,
    )
    assert ledger.maximum_closure_defect < 2.0e-15

    generator = rng.normal(size=(15, 15))
    work = causal_pairwise_family_cross_work(
        components,
        generator,
        weights,
    )
    total = np.sum(components, axis=0).reshape(6, -1)
    rate = total @ generator.T
    normalized = weights / np.sum(weights)
    expected = np.einsum(
        "tci,tci,c->t",
        total.reshape(6, 3, 5),
        rate.reshape(6, 3, 5),
        normalized,
    )
    np.testing.assert_allclose(
        np.sum(work, axis=(1, 2)),
        expected,
        rtol=2.0e-15,
        atol=2.0e-14,
    )


def test_local_energy_work_ledger_closes_by_cell_and_block():
    times = np.linspace(0.0, 0.2, 41)
    first = np.diag([-2.0, -1.0, -0.5, -0.25, -0.125])
    second = np.diag([-1.5, -0.75, -0.4, -0.2, -0.1])
    generator = np.zeros((10, 10))
    generator[:5, :5] = first
    generator[5:, 5:] = second
    blocks = {
        "fast": 0.4 * generator,
        "slow": 0.6 * generator,
    }
    initial = np.arange(1.0, 11.0)
    history = np.asarray(
        [
            np.exp(np.diag(generator) * time) * initial
            for time in times
        ]
    ).reshape(times.size, 2, 5)
    grams = np.repeat(np.eye(5)[None], 2, axis=0)
    ledger = causal_local_quadratic_energy_work_ledger(
        history,
        times,
        grams,
        blocks,
    )
    assert ledger.maximum_instantaneous_block_closure_defect < 1.0e-15
    # The trapezoidal integral has finite second-order error.
    assert ledger.maximum_integrated_energy_closure_defect < 2.0e-4
    np.testing.assert_allclose(
        ledger.rate_by_block_and_cell_per_s["fast"],
        (2.0 / 3.0)
        * ledger.rate_by_block_and_cell_per_s["slow"],
        rtol=2.0e-15,
        atol=2.0e-14,
    )


def test_block_family_rate_and_cross_work_ledgers_close():
    rng = np.random.default_rng(20260728)
    states = rng.normal(size=(5, 8, 3, 5))
    first = rng.normal(size=(15, 15))
    second = rng.normal(size=(15, 15))
    projectors = _coordinate_projectors(3)
    receiver = causal_block_family_receiver_action(
        states[2],
        first,
        projectors,
    )
    expected = (
        states[2].reshape(8, 15) @ first.T
    ).reshape(8, 3, 5)
    np.testing.assert_allclose(
        np.sum(receiver, axis=0),
        expected,
        rtol=2.0e-15,
        atol=2.0e-14,
    )

    ledger = causal_block_family_transfer_ledger(
        states,
        {"first": first, "second": second},
        projectors,
        np.array([1.0, 2.0, 3.0]),
    )
    assert ledger.block_names == ("first", "second")
    assert ledger.maximum_rate_action_closure_defect < 2.0e-15
    assert ledger.maximum_cross_work_closure_defect < 2.0e-15
