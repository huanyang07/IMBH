from __future__ import annotations

import numpy as np

from imri_qpe.layer3_minidisk_1d.causal_inner_integral_conditioning import (
    causal_absolute_band_error_envelope,
    causal_cancellation_ratio,
    causal_integral_conditioning_decision,
)


def _decision(**changes):
    arguments = {
        "global_rms_order": -0.1,
        "global_maximum_order": 0.2,
        "global_fine_maximum": 1.0e-5,
        "cell_rms_orders": np.array([1.8, 2.0, 1.9]),
        "active_cells": np.array([True, True, True]),
        "band_rms_orders": np.array([1.8, 2.1]),
        "band_maximum_orders": np.array([1.7, 2.0]),
        "band_error_cosines": np.array([0.99, 0.98]),
        "active_bands": np.array([True, True]),
        "absolute_band_error_envelope": 2.0e-4,
        "coarse_medium_cancellation_ratio": 0.05,
        "medium_fine_cancellation_ratio": 0.08,
        "direct_sum_defect": 1.0e-15,
        "gram_closure_defect": 1.0e-15,
        "continuum_uncertainty_to_fine": 1.0e-3,
        "minimum_order": 0.75,
        "minimum_error_cosine": 0.90,
        "maximum_fine_difference": 0.05,
        "maximum_cancellation_ratio": 0.25,
        "maximum_ledger_defect": 1.0e-12,
        "maximum_continuum_ratio": 0.10,
    }
    arguments.update(changes)
    return causal_integral_conditioning_decision(**arguments)


def test_causal_integral_conditioning_selects_alternate_route() -> None:
    result = _decision()
    assert result.passed
    assert result.route == "cancellation_conditioned_band_envelope"
    assert result.active_band_count == 2


def test_causal_integral_conditioning_preserves_direct_route() -> None:
    result = _decision(
        global_rms_order=1.8,
        global_maximum_order=1.7,
        coarse_medium_cancellation_ratio=0.8,
        medium_fine_cancellation_ratio=0.9,
    )
    assert result.passed
    assert result.route == "direct_component_order"


def test_causal_integral_conditioning_rejects_bad_local_or_reference_gate() -> None:
    assert not _decision(
        band_error_cosines=np.array([0.89, 0.99])
    ).passed
    assert not _decision(
        cell_rms_orders=np.array([1.8, 0.70, 1.9])
    ).passed
    assert _decision(
        cell_rms_orders=np.array([1.8, 0.10, 1.9]),
        active_cells=np.array([True, False, True]),
    ).passed
    assert not _decision(
        continuum_uncertainty_to_fine=0.11
    ).passed
    assert not _decision(
        absolute_band_error_envelope=0.051
    ).passed


def test_causal_integral_conditioning_error_measures_include_signed_sum() -> None:
    errors = np.array(
        [
            [1.0, -0.9],
            [2.0, -1.8],
            [1.0, -0.9],
        ]
    )
    weights = np.array([0.25, 0.5, 0.25])
    ratio = causal_cancellation_ratio(errors, time_weights=weights)
    assert 0.04 < ratio < 0.06
    envelope = causal_absolute_band_error_envelope(
        errors,
        physical_scale=10.0,
    )
    assert abs(envelope - 0.38) <= 1.0e-15
