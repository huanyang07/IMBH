from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_causal_interface_state_sufficiency_wp10c8r import (  # noqa: E402
    _gate_normalized_interface_half_difference,
    _infer_loading_time_seconds,
    _significance_filtered_transport_audit,
)


def test_interface_half_difference_applies_only_relative_gate() -> None:
    minus = np.zeros(12)
    plus = 2.0e-3 * np.arange(1.0, 13.0)

    result = _gate_normalized_interface_half_difference(plus, minus)

    np.testing.assert_array_equal(
        result,
        np.arange(1.0, 13.0).reshape(4, 3),
    )


def test_significance_filter_rejects_unit_normalized_noise() -> None:
    values = np.asarray(
        [
            [0.3, 0.2, 0.3],
            1.0e-8 * np.asarray([1.0, -3.0, 1.0]),
            2.0e-9 * np.asarray([-1.0, 2.0, -1.0]),
        ]
    )

    audit = _significance_filtered_transport_audit(
        values,
        ("physical", "noise_a", "noise_b"),
    )

    assert audit["significant_sample_count"] == 1
    assert audit["significant_families"] == ("physical",)
    assert audit["supported_dimension_at_ratio_0p1"] == 1
    assert not audit["rank_two_authorized"]


def test_significance_filter_requires_independent_families_for_rank_two() -> None:
    values = np.asarray(
        [
            [0.4, 0.0, 0.0],
            [0.0, 0.4, 0.0],
        ]
    )

    repeated = _significance_filtered_transport_audit(
        values,
        ("same", "same"),
    )
    independent = _significance_filtered_transport_audit(
        values,
        ("first", "second"),
    )

    assert repeated["supported_dimension_at_ratio_0p1"] == 2
    assert not repeated["rank_two_authorized"]
    assert independent["rank_two_authorized"]


def test_loading_time_is_recovered_from_saved_rate_conversion() -> None:
    loading = 8.5e5
    minus = np.asarray([1.0, -2.0, 0.5])
    plus = np.asarray([1.2, -1.8, 0.4])
    raw_half = 0.5 * (plus - minus)
    slow = raw_half * loading / 0.025

    inferred, defect = _infer_loading_time_seconds(minus, plus, slow)

    assert inferred == pytest.approx(loading)
    assert defect <= 2.0e-15
