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

import run_causal_inner_mode_phase_average_audit_wp10c8u as wp10c8u


def _dense(
    *,
    times: np.ndarray,
    coordinate_scales: np.ndarray,
    physical_rates: np.ndarray,
    coordinates: np.ndarray,
) -> dict[str, np.ndarray]:
    n_times, n_coordinates = physical_rates.shape
    return {
        "times": np.asarray(times, dtype=float),
        "coordinate_names": np.asarray(
            tuple(f"q{index}" for index in range(n_coordinates)),
            dtype="U",
        ),
        "coordinate_scales": np.asarray(coordinate_scales, dtype=float),
        "coordinates": np.asarray(coordinates, dtype=float),
        "normalized_coordinate_rates": (
            wp10c8u.wp10c8o.COORDINATE_RATE_WINDOW_SECONDS
            * np.asarray(physical_rates, dtype=float)
            / np.asarray(coordinate_scales, dtype=float)[None, :]
        ),
        "scaled_primitive_rates_per_s": np.zeros((n_times, 5)),
        "primitive_column_scales": np.ones(5),
        "primitives": np.zeros((n_times, 1, 5)),
    }


def test_dense_schedule_contains_every_nested_state() -> None:
    coarse = wp10c8u._all_output_times("coarse")
    fine = wp10c8u._all_output_times("fine")
    assert len(coarse) == 101
    assert len(fine) == 201
    np.testing.assert_array_equal(coarse, fine[::2])


def test_direction_gate_is_signed() -> None:
    aligned = wp10c8u._direction_metrics(
        np.asarray(((1.0, 0.0),)),
        np.asarray(((1.1, 0.0),)),
    )
    reversed_direction = wp10c8u._direction_metrics(
        np.asarray(((1.0, 0.0),)),
        np.asarray(((-1.0, 0.0),)),
    )
    assert aligned["same_time_gate_passed"][0]
    assert aligned["signed_cosine"][0] == 1.0
    assert reversed_direction["absolute_cosine"][0] == 1.0
    assert reversed_direction["signed_cosine"][0] == -1.0
    assert not reversed_direction["same_time_gate_passed"][0]


def test_pair_history_uses_common_physical_scale_and_slow_time() -> None:
    times = np.asarray((0.0, 1.0, 2.0))
    mesh_scales = np.asarray((2.0, 8.0))
    common_scales = np.asarray((4.0, 4.0))
    minus = _dense(
        times=times,
        coordinate_scales=mesh_scales,
        physical_rates=np.zeros((3, 2)),
        coordinates=np.zeros((3, 2)),
    )
    plus = _dense(
        times=times,
        coordinate_scales=mesh_scales,
        physical_rates=np.asarray(((2.0, 4.0),) * 3),
        coordinates=np.asarray(
            ((0.0, 0.0), (2.0, 4.0), (4.0, 8.0))
        ),
    )
    result = wp10c8u._pair_history(
        minus=minus,
        plus=plus,
        common_coordinate_scales=common_scales,
        loading_time_seconds=10.0,
    )
    np.testing.assert_allclose(
        result["signed_slow_rate_half_difference"],
        np.asarray(((2.5, 5.0),) * 3),
    )
    np.testing.assert_allclose(
        result["rate_integrated_slip"][-1],
        (0.5, 1.0),
    )
    np.testing.assert_allclose(
        result["signed_coordinate_slip"][-1],
        (0.5, 1.0),
    )
    assert result[
        "physical_primitive_rate_half_difference_per_s"
    ].shape == (3, 1, 5)
    assert result["absolute_impulse"][-1] > 0.0


def test_window_statistics_reproduce_constant_mean_and_rms() -> None:
    times = np.linspace(0.0, 1.0, 11)
    values = np.tile(np.asarray((3.0, 4.0)), (times.size, 1))
    result = wp10c8u._window_statistics(times, values, 0.5)
    assert result["start_times"].shape == (6,)
    np.testing.assert_allclose(
        result["mean_vectors"],
        np.tile(np.asarray((3.0, 4.0)), (6, 1)),
    )
    np.testing.assert_allclose(result["mean_norms"], 5.0)
    np.testing.assert_allclose(result["rms_amplitudes"], 5.0)


def test_component_zero_crossings_are_linearly_interpolated() -> None:
    result = wp10c8u._component_zero_crossings(
        np.asarray((0.0, 1.0, 2.0)),
        np.asarray((1.0, -1.0, 3.0)),
    )
    np.testing.assert_allclose(result, (0.5, 1.25))


def test_weighted_pod_applies_absolute_significance_filter() -> None:
    snapshots = np.zeros((3, 2, 5))
    snapshots[0, 0, 0] = 1.0
    snapshots[1, 1, 1] = 1.0
    snapshots[2, :, :] = 100.0
    result = wp10c8u._weighted_pod(
        snapshots,
        cell_measures=np.asarray((1.0, 3.0)),
        shell_stop=2,
        significant_mask=np.asarray((True, True, False)),
    )
    assert result["singular_values"].shape == (2,)
    np.testing.assert_array_equal(result["selected_indices"], (0, 1))


def test_label_round_trip() -> None:
    label = wp10c8u._trajectory_label(128, "fine", "plus")
    assert label == "N128_fine_plus"
    assert wp10c8u._parse_trajectory_label(label) == (128, "fine", "plus")
