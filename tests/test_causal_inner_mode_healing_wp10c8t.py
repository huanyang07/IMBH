from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for path in (SCRIPTS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_mode_healing_wp10c8t as wp10c8t


def _pair(
    spreads: np.ndarray,
    slips: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    values = np.asarray(spreads, dtype=float)
    n_times = values.shape[0]
    if slips is None:
        slips = np.zeros((n_times, 2), dtype=float)
    return {
        "times": np.arange(n_times, dtype=float),
        "full_names": np.asarray(
            tuple(f"rate_{index}" for index in range(values.shape[1])),
            dtype="U",
        ),
        "full_spreads": values,
        "signed_coordinate_slip": np.asarray(slips, dtype=float),
    }


def test_wp10c8t_schedule_is_nested_and_exact() -> None:
    wp10c8t._validate_schedule()
    assert (
        wp10c8t.TIMESTEP_SECONDS["coarse"]
        == 2.0 * wp10c8t.TIMESTEP_SECONDS["fine"]
    )
    assert wp10c8t.TOTAL_SUBDIVISIONS == {"coarse": 100, "fine": 200}
    assert wp10c8t.PARENT_SUBDIVISIONS == {"coarse": 20, "fine": 40}


def test_pair_arrays_integrates_slow_rate_in_slow_time() -> None:
    coordinate_scales = np.asarray((2.0, 4.0))
    minus = {
        "output_times": np.asarray((0.0, 1.0, 2.0)),
        "static_output_gates": np.asarray((1.0,)),
        "static_outputs": np.zeros((3, 1)),
        "static_output_names": np.asarray(("static",), dtype="U"),
        "coordinate_names": np.asarray(("a", "b"), dtype="U"),
        "coordinates": np.zeros((3, 2)),
        "normalized_coordinate_rates": np.zeros((3, 2)),
    }
    plus = {
        **minus,
        "coordinates": np.asarray(
            ((0.0, 0.0), (2.0, 4.0), (4.0, 8.0))
        ),
        "normalized_coordinate_rates": np.ones((3, 2)),
    }
    result = wp10c8t._pair_arrays(
        minus=minus,
        plus=plus,
        coordinate_scales=coordinate_scales,
        loading_time_seconds=10.0,
    )
    np.testing.assert_allclose(
        result["signed_coordinate_slip"][-1],
        (1.0, 1.0),
    )
    assert result["signed_slow_rate_half_difference"].shape == (3, 2)
    assert result["rate_integrated_slip"].shape == (3, 2)


def test_decision_accepts_resolved_healing_with_small_slip() -> None:
    coarse = _pair(
        np.asarray(((10.0,), (1.0,), (0.05,))),
        np.asarray(((0.0, 0.0), (0.01, 0.01), (0.02, 0.02))),
    )
    fine = _pair(
        np.asarray(((10.0,), (0.99,), (0.04,))),
        np.asarray(((0.0, 0.0), (0.009, 0.009), (0.018, 0.018))),
    )
    decision, _arrays = wp10c8t._decision(
        coarse=coarse,
        fine=fine,
        all_contracts_passed=True,
    )
    assert decision["temporal_curve_passed"]
    assert decision["natural_healing_with_small_slip_passed"]
    assert (
        decision["classification"]
        == "n64_fast_initial_layer_with_small_slip_supported"
    )
    assert not decision["relaxation_fit_authorized"]


def test_decision_resolves_persistent_mode_with_lower_bound() -> None:
    coarse = _pair(np.asarray(((100.0,), (40.0,), (20.0,))))
    fine = _pair(np.asarray(((100.0,), (35.0,), (15.0,))))
    decision, arrays = wp10c8t._decision(
        coarse=coarse,
        fine=fine,
        all_contracts_passed=True,
    )
    assert not decision["temporal_curve_passed"]
    assert decision["persistence_separated_from_healing_gate"]
    assert (
        decision["classification"]
        == "n64_persistent_localized_inner_mode_through_0p125s"
    )
    assert arrays["uncertainty_exclusive_lower_spreads"][-1, 0] == 10.0
    assert decision["n128_architecture_confirmation_required"]


def test_decision_remains_inconclusive_without_contracts() -> None:
    values = np.asarray(((2.0,), (1.0,), (0.5,)))
    decision, _arrays = wp10c8t._decision(
        coarse=_pair(values),
        fine=_pair(values),
        all_contracts_passed=False,
    )
    assert (
        decision["classification"]
        == "n64_inner_mode_healing_numerically_inconclusive"
    )
    assert not decision["n128_architecture_confirmation_required"]


def test_decision_handles_development_run_without_significant_rates() -> None:
    values = np.zeros((3, 2), dtype=float)
    decision, arrays = wp10c8t._decision(
        coarse=_pair(values),
        fine=_pair(values),
        all_contracts_passed=False,
    )
    assert (
        decision["classification"]
        == "n64_inner_mode_healing_numerically_inconclusive"
    )
    assert decision["measured_minimum_controlling_e_folds"] == 0.0
    assert not np.any(arrays["significant_initial_output_mask"])


def test_case_array_filter_excludes_embedded_healing_evidence() -> None:
    values = {
        f"{wp10c8t.CASE_ID}_minus_state_vector": np.zeros(3),
        f"{wp10c8t.CASE_ID}_plus_state_vector": np.ones(3),
        f"{wp10c8t.CASE_ID}_coordinate_names": np.asarray(("a",), dtype="U"),
        f"{wp10c8t.CASE_ID}_coordinate_scales": np.ones(1),
        f"{wp10c8t.CASE_ID}_interface_flux_scales": np.ones(3),
        f"{wp10c8t.CASE_ID}_healing_fine_times": np.ones(2),
        f"{wp10c8t.CASE_ID}_localization_radius_rg": np.ones(2),
    }
    result = wp10c8t._case_arrays(values)
    assert "minus_state_vector" in result
    assert "healing_fine_times" not in result
    assert "localization_radius_rg" not in result
