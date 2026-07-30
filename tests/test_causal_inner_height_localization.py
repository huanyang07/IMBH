from __future__ import annotations

import numpy as np

from imri_qpe.layer3_minidisk_1d.causal_inner_height_localization import (
    causal_partition_cell_integrals,
    causal_prefix_suffix_histories,
    causal_restrict_cell_integrals,
    causal_signed_band_gram_matrix,
)


def test_cell_integral_restriction_and_partition_preserve_sum() -> None:
    values = np.arange(1.0, 17.0).reshape(2, 8)
    restricted = causal_restrict_cell_integrals(
        values,
        refinement_factor=2,
    )
    np.testing.assert_allclose(
        restricted,
        values.reshape(2, 4, 2).sum(axis=-1),
    )
    bands = causal_partition_cell_integrals(
        restricted,
        np.asarray((0, 1, 3, 4)),
    )
    np.testing.assert_allclose(
        np.sum(bands, axis=-1),
        np.sum(values, axis=-1),
    )


def test_prefix_suffix_and_signed_gram_retain_cancellation() -> None:
    cells = np.asarray(((1.0, -0.75, 0.25), (2.0, -1.5, 0.5)))
    prefix, suffix = causal_prefix_suffix_histories(cells)
    np.testing.assert_allclose(prefix[:, -1], np.sum(cells, axis=1))
    np.testing.assert_allclose(suffix[:, 0], np.sum(cells, axis=1))
    gram = causal_signed_band_gram_matrix(
        cells,
        physical_scale=2.0,
        time_weights=np.asarray((0.25, 0.75)),
    )
    assert gram.shape == (3, 3)
    np.testing.assert_allclose(gram, gram.T)
    total_squared = float(np.sum(gram))
    expected = float(
        np.sum(
            np.asarray((0.25, 0.75))
            * (np.sum(cells, axis=1) / 2.0) ** 2
        )
    )
    assert np.isclose(total_squared, expected)
