from __future__ import annotations

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES,
    causal_characteristic_packet_moments,
    causal_five_field_characteristic_basis,
    causal_five_field_characteristic_coefficients,
    causal_five_field_characteristic_packet,
    make_causal_five_field_regression_context,
    make_causal_five_field_seed,
)


def _packet_inputs(n_cells: int = 32):
    context = make_causal_five_field_regression_context(n_cells)
    primitives = make_causal_five_field_seed(context).primitives
    amplitudes = np.broadcast_to(
        np.asarray([1.0e-2, 2.0e-3, 2.0e-3, 1.0e-2, 1.0e-5]),
        primitives.shape,
    ).copy()
    return context, primitives, amplitudes


def test_characteristic_basis_is_ordered_and_closes_principal_pairs() -> None:
    context, primitives, amplitudes = _packet_inputs()
    cell = 4
    basis = causal_five_field_characteristic_basis(
        context,
        float(context.grid.centers[cell]),
        primitives[cell],
        amplitudes[cell],
    )

    assert basis.family_labels == (
        CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES
    )
    assert basis.local_rest_speeds_over_c.shape == (5,)
    assert basis.coordinate_speeds_over_c.shape == (5,)
    assert basis.physical_right_eigenvectors.shape == (5, 5)
    assert basis.dimensionless_right_eigenvectors.shape == (5, 5)
    assert np.all(np.diff(basis.local_rest_speeds_over_c) >= 0.0)
    assert np.all(np.diff(basis.coordinate_speeds_over_c) >= 0.0)
    assert basis.maximum_eigenpair_defect <= 2.0e-12
    np.testing.assert_allclose(
        np.linalg.norm(basis.dimensionless_right_eigenvectors, axis=0),
        np.ones(5),
        rtol=3.0e-14,
        atol=3.0e-14,
    )


def test_compact_characteristic_packet_is_family_pure() -> None:
    context, primitives, amplitudes = _packet_inputs()
    inner = 2.1 * context.grid.gravitational_radius
    outer = 25.0 * context.grid.gravitational_radius
    packet, bases = causal_five_field_characteristic_packet(
        context,
        primitives,
        amplitudes,
        family="inward_shear",
        support_inner_radius=inner,
        support_outer_radius=outer,
    )
    coefficients = causal_five_field_characteristic_coefficients(
        packet,
        bases,
    )
    selected = CAUSAL_FIVE_FIELD_CHARACTERISTIC_FAMILIES.index(
        "inward_shear"
    )
    other = np.delete(coefficients, selected, axis=1)

    assert np.max(np.abs(other)) <= 2.0e-15
    assert np.max(np.abs(coefficients[:, selected])) > 0.0
    outside = (
        (context.grid.centers <= inner)
        | (context.grid.centers >= outer)
    )
    assert np.array_equal(packet[outside], np.zeros_like(packet[outside]))


def test_material_packet_has_no_opposite_family_by_definition() -> None:
    context, primitives, amplitudes = _packet_inputs()
    packet, bases = causal_five_field_characteristic_packet(
        context,
        primitives,
        amplitudes,
        family="material",
        support_inner_radius=2.1 * context.grid.gravitational_radius,
        support_outer_radius=25.0 * context.grid.gravitational_radius,
    )
    history = np.stack((packet, 0.8 * packet), axis=0)
    moments = causal_characteristic_packet_moments(
        history,
        bases,
        context.grid.centers,
        context.grid.cell_measures,
        family="material",
    )

    np.testing.assert_allclose(
        moments.selected_family_fraction,
        np.ones(2),
        rtol=3.0e-14,
        atol=3.0e-14,
    )
    assert np.array_equal(
        moments.opposite_family_fraction,
        np.zeros(2),
    )
