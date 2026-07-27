from __future__ import annotations

from dataclasses import replace

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    causal_five_field_radial_candidate_face_flux,
    causal_five_field_radial_candidate_ledger,
    causal_five_field_radial_candidate_lower_source_totals,
    causal_five_field_radial_frozen_candidate,
    causal_five_field_radial_reduced_jacobian_pattern,
    make_causal_five_field_regression_context,
    make_causal_five_field_seed,
)


def test_face_local_candidate_flux_matches_complete_ledger() -> None:
    context = make_causal_five_field_regression_context(8)
    primitives = make_causal_five_field_seed(context).primitives
    ledger = causal_five_field_radial_candidate_ledger(context, primitives)
    for face in (0, 3, 8):
        production, candidate = (
            causal_five_field_radial_candidate_face_flux(
                context,
                primitives,
                face,
            )
        )
        np.testing.assert_allclose(
            production,
            ledger.interfaces.production_shared_face_fluxes_over_c[face],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            candidate,
            ledger.interfaces.candidate_shared_face_fluxes_over_c[face],
            rtol=0.0,
            atol=0.0,
        )
    lower = causal_five_field_radial_candidate_lower_source_totals(
        context,
        primitives,
    )
    for name, values in lower.items():
        np.testing.assert_allclose(
            values,
            np.sum(
                ledger.integrated_lower_source_components_per_ct[name],
                axis=0,
            ),
            rtol=0.0,
            atol=0.0,
        )


def test_reduced_pattern_contains_declared_radial_band() -> None:
    pattern = causal_five_field_radial_reduced_jacobian_pattern(9)
    assert pattern.shape == (45, 45)
    dense = pattern.toarray()
    assert np.all(dense[20:25, 5:40] == 1)
    assert np.all(dense[20:25, :5] == 0)
    assert np.all(dense[20:25, 40:] == 0)


def test_frozen_candidate_changes_only_stationary_spatial_block() -> None:
    context = make_causal_five_field_regression_context(3)
    primitives = make_causal_five_field_seed(context).primitives
    context = replace(
        context,
        outer_boundary_flux_mode="frozen_exterior_rusanov",
        outer_boundary_frozen_exterior_chart=np.array(
            primitives[-1],
            copy=True,
        ),
    ).validated()
    size = primitives.size
    production = np.zeros((size, size), dtype=float)
    result = causal_five_field_radial_frozen_candidate(
        context,
        primitives,
        production,
        finite_difference_step=4.0e-5,
    )
    assert result.same_temporal_descriptor is True
    assert result.same_base_rate_storage_derivative is True
    np.testing.assert_allclose(
        result.candidate_scaled_generator_per_s,
        -result.descriptor_solve_scaled_correction,
        rtol=0.0,
        atol=0.0,
    )
    assert result.maximum_descriptor_solve_relative_defect <= 1.0e-10
    assert result.maximum_mass_off_pattern_relative_entry <= 2.0e-2
