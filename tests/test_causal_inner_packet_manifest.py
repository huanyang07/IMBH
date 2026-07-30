from __future__ import annotations

import numpy as np

from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (
    causal_array_sha256,
    causal_canonical_json_sha256,
    causal_characteristic_purity,
    causal_scaled_variant_defect,
)


def test_canonical_json_hash_is_order_independent() -> None:
    first = {"b": [2, 3], "a": 1}
    second = {"a": 1, "b": [2, 3]}
    assert causal_canonical_json_sha256(first) == (
        causal_canonical_json_sha256(second)
    )


def test_array_hash_includes_dtype_and_shape() -> None:
    values = np.arange(6, dtype=np.float64)
    assert causal_array_sha256(values) != causal_array_sha256(
        values.astype(np.float32)
    )
    assert causal_array_sha256(values) != causal_array_sha256(
        values.reshape(2, 3)
    )


def test_characteristic_purity_recovers_selected_family() -> None:
    scales = np.asarray((1.0, 2.0, 3.0, 4.0, 5.0))
    basis = np.broadcast_to(
        np.diag(scales),
        (7, 5, 5),
    ).copy()
    values = np.zeros((7, 5))
    values[:, 3] = scales[3] * np.linspace(0.1, 1.0, 7)
    report = causal_characteristic_purity(
        values,
        basis,
        scales,
        np.linspace(1.0, 2.0, 7),
        selected_family=3,
    )
    np.testing.assert_allclose(
        report.family_energy_fractions,
        np.asarray((0.0, 0.0, 0.0, 1.0, 0.0)),
        atol=1.0e-15,
    )
    assert report.maximum_reconstruction_defect == 0.0
    assert report.minimum_active_cell_selected_fraction == 1.0


def test_scaled_variant_defect_detects_exact_sign_and_amplitude() -> None:
    base = np.arange(15, dtype=float).reshape(3, 5)
    assert causal_scaled_variant_defect(
        base,
        -0.5 * base,
        expected_factor=-0.5,
    ) == 0.0
    assert causal_scaled_variant_defect(
        base,
        -0.51 * base,
        expected_factor=-0.5,
    ) > 0.0
